"""PyTorch -> ONNX export and numerical equivalence checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

INPUT_NAME = "input"
OUTPUT_NAMES = ("pixel_map", "binary")


def export_onnx(model, path: Path, opset: int = 17, size: int = 224) -> dict[str, Any]:
    """Export DeepPixBiS (eval mode, no grad) with a dynamic batch axis."""
    import torch

    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, size, size, dtype=torch.float32, device=next(model.parameters()).device)
    with torch.no_grad():
        torch.onnx.export(model, (dummy,), str(path), opset_version=opset, input_names=[INPUT_NAME],
                          output_names=list(OUTPUT_NAMES), do_constant_folding=True, dynamo=False,
                          dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAMES[0]: {0: "batch"},
                                        OUTPUT_NAMES[1]: {0: "batch"}})
    return validate_onnx(path)


def validate_onnx(path: Path) -> dict[str, Any]:
    """Run the ONNX checker and record input/output names, shapes and operator set."""
    import onnx

    m = onnx.load(str(path))
    onnx.checker.check_model(m)

    def dims(vi) -> list[Any]:
        return [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim]

    return {
        "path": str(path), "opset": [{"domain": o.domain, "version": o.version} for o in m.opset_import],
        "inputs": {i.name: dims(i) for i in m.graph.input},
        "outputs": {o.name: dims(o) for o in m.graph.output},
        "op_types": sorted({n.op_type for n in m.graph.node}), "n_nodes": len(m.graph.node),
    }


def onnxruntime_backend(path: Path, providers: Sequence[str] = ("CPUExecutionProvider",)):
    """ONNX Runtime session as a map backend (returns the pixel map)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=list(providers))

    def run(x: np.ndarray) -> np.ndarray:
        return sess.run([OUTPUT_NAMES[0]], {INPUT_NAME: x.astype(np.float32)})[0]

    return run


def compare_scores(a, b, tolerance: float, key_cols: Sequence[str] = ("video_id", "frame_idx"),
                   score_col: str = "frame_score", label_a: str = "a", label_b: str = "b") -> dict[str, Any]:
    """Max/mean absolute difference between two score tables on identical keys."""
    m = a[list(key_cols) + [score_col]].merge(b[list(key_cols) + [score_col]], on=list(key_cols),
                                              suffixes=(f"_{label_a}", f"_{label_b}"), validate="one_to_one")
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError(f"score tables do not share identical keys ({len(a)}, {len(b)}, joined {len(m)})")
    diff = np.abs(m[f"{score_col}_{label_a}"] - m[f"{score_col}_{label_b}"]).to_numpy()
    failing = m.loc[diff > tolerance, list(key_cols)].assign(abs_diff=diff[diff > tolerance])
    worst = int(np.argmax(diff)) if len(diff) else 0
    return {
        "n": int(len(m)), "tolerance": tolerance, "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
        "mean_abs_diff": float(diff.mean()) if len(diff) else 0.0,
        "worst_key": {k: str(m.iloc[worst][k]) for k in key_cols} if len(diff) else None,
        "n_failing": int(len(failing)), "passed": bool(len(failing) == 0),
        "failing_examples": failing.head(20).to_dict(orient="records"),
    }
