"""Pilot diagnostics: why INT8 engines with different calibration mixes behave alike.

Notebook 07b reads the saved pilot outputs (no GPU); notebook 07c builds diagnostic
engines (per-layer precision, rebuild noise, positive controls). Everything here
reads result files written by notebooks 05 and 07 and never changes them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import decision_flips

CLASSES = ("bona_fide", "print", "replay")
# Children of the DeepPixBiS encoder (first 8 children of DenseNet-161 ``features``).
ENCODER_BLOCKS = {0: "conv0", 1: "norm0", 2: "relu0", 3: "pool0", 4: "denseblock1", 5: "transition1",
                  6: "denseblock2", 7: "transition2"}
BLOCK_ORDER = ("input", *ENCODER_BLOCKS.values(), "pixel-map head", "binary head", "other")


# ---------------------------------------------------------------------------------------
# Saved results
# ---------------------------------------------------------------------------------------
def engine_metadata_table(engine_dirs: Mapping[str, Path], expected_batches: int,
                          expected_tensors: int | None = None) -> pd.DataFrame:
    """One row per engine from ``engine_meta.json``; ``ok`` means it saw its full calibration set.

    ``expected_tensors`` defaults to the most common scaled-tensor count across the engines.
    """
    rows = []
    for tag, d in engine_dirs.items():
        m = json.loads((Path(d) / "engine_meta.json").read_text())
        hw = m.get("hardware", {})
        rows.append({"engine": tag, "method": m.get("method"), "calibration_images": m.get("n_calibration_images"),
                     "batches_served": m.get("calibration_batches_served"), "scaled_tensors": m.get("n_scaled_tensors"),
                     "tensorrt": hw.get("tensorrt"), "gpu": hw.get("gpu_name"),
                     "build_seconds": round(float(m.get("build_seconds", np.nan)), 1), "sha256": m["sha256"][:12]})
    t = pd.DataFrame(rows)
    if expected_tensors is None:
        expected_tensors = int(t["scaled_tensors"].mode().iloc[0])
    t["ok"] = (t["batches_served"] == expected_batches) & (t["scaled_tensors"] == expected_tensors)
    return t


def read_test_scores(result_dir: Path, policy: str = "B") -> tuple[pd.DataFrame, float]:
    """Test video scores and the threshold used by ``policy`` (an INT8 result dir or the FP32 one)."""
    result_dir = Path(result_dir)
    m = json.loads((result_dir / "metrics.json").read_text())
    if "threshold_B_engine" in m:
        thr = m["threshold_A_fp32"] if policy == "A" else m["threshold_B_engine"]["threshold"]
    else:
        thr = json.loads((result_dir / "threshold.json").read_text())["threshold"]
    return pd.read_csv(result_dir / "test_video_scores.csv", dtype={"video_id": str}), float(thr)


def margin_table(videos: pd.DataFrame, threshold: float, widths: Sequence[float] = (0.02, 0.05, 0.10)
                 ) -> pd.DataFrame:
    """Per class: how many videos score within ``w`` of the threshold (the only ones a small shift can flip)."""
    dist = (videos["score"] - threshold).abs()
    rows = []
    for c in CLASSES:
        sel = (videos["class"] == c).to_numpy()
        row = {"class": c, "videos": int(sel.sum()), "median_distance": float(np.median(dist[sel]))}
        row.update({f"within_{w:g}": int((dist[sel] <= w).sum()) for w in widths})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_score_shift(ref: pd.DataFrame, other: pd.DataFrame, moved: float = 0.01) -> pd.DataFrame:
    """Per class: how far ``other``'s video scores moved from ``ref``'s on the same videos."""
    m = ref[["video_id", "class", "score"]].merge(other[["video_id", "score"]], on="video_id",
                                                  suffixes=("_ref", "_other"), validate="one_to_one")
    if len(m) != len(ref) or len(m) != len(other):
        raise ValueError("models were not scored on the same videos")
    m["delta"] = m["score_other"] - m["score_ref"]
    rows = []
    for c in CLASSES:
        d = m.loc[m["class"] == c, "delta"]
        rows.append({"class": c, "mean_delta": float(d.mean()), "mean_abs_delta": float(d.abs().mean()),
                     "median_abs_delta": float(d.abs().median()), "max_abs_delta": float(d.abs().max()),
                     f"moved_over_{moved:g}": int((d.abs() > moved).sum())})
    return pd.DataFrame(rows)


def compare_pairs(pairs: Iterable[tuple[str, Path, Path]], policy: str = "B") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score shifts and decision flips for each ``(label, reference_dir, other_dir)``."""
    shifts, flips = [], []
    for label, ref_dir, other_dir in pairs:
        (a, ta), (b, tb) = read_test_scores(ref_dir, policy), read_test_scores(other_dir, policy)
        shifts.append(paired_score_shift(a, b).assign(comparison=label))
        flips.append({"comparison": label, "policy": policy, **decision_flips(a, b, ta, tb)})
    shift = pd.concat(shifts, ignore_index=True)
    return shift[["comparison", *[c for c in shift.columns if c != "comparison"]]], pd.DataFrame(flips)


# ---------------------------------------------------------------------------------------
# Where the calibration scales changed
# ---------------------------------------------------------------------------------------
def block_of(node_name: str) -> str:
    """Model block of an exported ONNX node (``/enc/enc.4/...`` -> ``denseblock1``)."""
    m = re.search(r"/enc/enc\.(\d+)", node_name)
    if m:
        return ENCODER_BLOCKS.get(int(m.group(1)), f"enc.{m.group(1)}")
    if node_name.startswith("/dec"):
        return "pixel-map head"
    if "linear" in node_name:
        return "binary head"
    return "other"


def onnx_tensor_order(onnx_path: Path) -> pd.DataFrame:
    """Every tensor of the exported graph with its position (0 = input) and model block."""
    import onnx

    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    rows = [{"tensor": i.name, "position": 0, "op": "Input", "node": "", "block": "input"} for i in graph.input]
    for pos, node in enumerate(graph.node, start=1):
        rows += [{"tensor": out, "position": pos, "op": node.op_type, "node": node.name, "block": block_of(node.name)}
                 for out in node.output]
    t = pd.DataFrame(rows)
    t["depth"] = t["position"] / max(len(graph.node), 1)
    return t


def scale_changes_by_layer(scales: Mapping[str, Mapping[str, float]], reference: str, order: pd.DataFrame
                           ) -> pd.DataFrame:
    """Per tensor and engine: |relative scale change| vs ``reference``, located in the graph."""
    from .calibration import compare_scales

    comp = compare_scales(scales, reference)
    comp = comp[comp["engine"] != reference].assign(abs_rel=lambda d: d["rel_diff"].abs())
    out = comp.merge(order[["tensor", "position", "depth", "op", "block"]], on="tensor", how="left")
    out["block"] = out["block"].fillna("not in ONNX graph")
    return out.sort_values(["engine", "position"]).reset_index(drop=True)


def scale_change_by_block(changes: pd.DataFrame) -> pd.DataFrame:
    """Median and maximum |relative scale change| per engine and model block, in graph order."""
    g = changes.groupby(["engine", "block"])["abs_rel"]
    t = pd.DataFrame({"tensors": g.size(), "median_change_pct": 100 * g.median(),
                      "max_change_pct": 100 * g.max()}).reset_index()
    rank = {b: i for i, b in enumerate(BLOCK_ORDER)}
    return t.sort_values(["engine", "block"], key=lambda s: s.map(rank).fillna(len(rank)) if s.name == "block" else s
                         ).reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Per-layer precision (engine inspector output)
# ---------------------------------------------------------------------------------------
def datatype_of(fmt: str) -> str:
    """Data type in a TensorRT ``Format/Datatype`` string, e.g. ``... Int8 format`` -> ``INT8``."""
    s = fmt.lower()
    for key, name in (("int8", "INT8"), ("fp16", "FP16"), ("half", "FP16"), ("fp32", "FP32"), ("float", "FP32"),
                      ("int32", "INT32"), ("int64", "INT64"), ("bool", "BOOL")):
        if key in s:
            return name
    return "unknown"


def _tensor_types(entries: Sequence[Any]) -> list[str]:
    types = []
    for e in entries or []:
        fmt = next((v for k, v in e.items() if "datatype" in k.lower() or "format" in k.lower()), "") \
            if isinstance(e, dict) else str(e)
        types.append(datatype_of(str(fmt)))
    return types


def layer_precision_table(layers: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """One row per engine layer; a layer counts as INT8 when its kernel reads INT8 input."""
    rows = []
    for i, layer in enumerate(layers):
        ins, outs = _tensor_types(layer.get("Inputs", [])), _tensor_types(layer.get("Outputs", []))
        if "INT8" in ins:
            precision = "INT8"
        elif "FP16" in ins:
            precision = "FP16"
        elif "FP32" in ins or "FP32" in outs:
            precision = "FP32"
        else:
            precision = outs[0] if outs else "unknown"
        rows.append({"index": i, "name": layer.get("Name", ""), "type": layer.get("LayerType", ""),
                     "inputs": ",".join(ins), "outputs": ",".join(outs), "precision": precision,
                     "tactic": layer.get("TacticName", "")})
    return pd.DataFrame(rows)


def precision_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Layer counts per precision, compute layers only (format-conversion ``Reformat`` layers excluded)."""
    compute = table[~table["type"].str.contains("Reformat", case=False, na=False)]
    s = compute.groupby("precision").size().rename("layers").reset_index()
    s["share_pct"] = 100 * s["layers"] / max(len(compute), 1)
    return s.sort_values("layers", ascending=False).reset_index(drop=True)
