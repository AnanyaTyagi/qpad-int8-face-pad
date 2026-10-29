"""Nested calibration-set sampling, TensorRT calibrators and calibration-range audits.

Calibration NEVER updates weights: it only chooses per-tensor INT8 scales from
activation statistics of the 600 calibration images.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

CLASS_ORDER = ("bona_fide", "print", "replay")


def validate_conditions(conditions: Mapping[str, Mapping[str, int]], budget_videos: int) -> None:
    for name, comp in conditions.items():
        total = sum(int(comp[c]) for c in CLASS_ORDER)
        if total != budget_videos:
            raise ValueError(f"condition {name} uses {total} videos, budget is {budget_videos}")


def nested_calibration_sets(train_videos: pd.DataFrame, conditions: Mapping[str, Mapping[str, int]],
                            seeds: Sequence[int], budget_videos: int) -> pd.DataFrame:
    """Select calibration videos with *nested* sampling.

    For each seed, each class list (sorted by ``video_id`` for a deterministic
    starting order) is shuffled ONCE with ``numpy.random.default_rng(seed)``;
    every condition takes the required prefix from those same shuffled lists.
    So, e.g., the first 40 print videos of *natural* are also the first 40 of
    *print_heavy* for the same seed.

    Returns one row per (seed, condition, video).
    """
    validate_conditions(conditions, budget_videos)
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        shuffled = {}
        for c in CLASS_ORDER:
            ids = sorted(train_videos.loc[train_videos["class"] == c, "video_id"].astype(str))
            shuffled[c] = [ids[i] for i in rng.permutation(len(ids))]
        for cond, comp in conditions.items():
            for c in CLASS_ORDER:
                k = int(comp[c])
                if k > len(shuffled[c]):
                    raise ValueError(f"{cond} needs {k} {c} videos; only {len(shuffled[c])} available")
                for rank, vid in enumerate(shuffled[c][:k]):
                    rows.append({"seed": int(seed), "condition": cond, "class": c, "rank_in_class": rank,
                                 "video_id": vid})
    return pd.DataFrame(rows)


def attach_calibration_frames(selection: pd.DataFrame, calib_frames: pd.DataFrame, frames_per_video: int
                              ) -> pd.DataFrame:
    """Join each selected video with its saved calibration frames (role CALIB) and verify counts."""
    f = calib_frames[["video_id", "frame_idx"]].astype({"video_id": str})
    out = selection.merge(f, on="video_id", how="left", validate="many_to_many")
    counts = out.groupby(["seed", "condition", "video_id"])["frame_idx"].count()
    bad = counts[counts != frames_per_video]
    if len(bad):
        raise ValueError(f"{len(bad)} selected videos lack exactly {frames_per_video} calibration frames")
    return out.sort_values(["seed", "condition", "class", "rank_in_class", "frame_idx"]).reset_index(drop=True)


def make_calibrator(images_nchw: np.ndarray, batch_size: int, cache_path: Path, method: str = "entropy"):
    """Create a TensorRT INT8 calibrator over a fixed float32 NCHW array.

    ``method``: ``entropy`` (IInt8EntropyCalibrator2, pre-registered primary) or
    ``minmax`` (IInt8MinMaxCalibrator, planned sensitivity). If the installed
    TensorRT no longer provides the class, this raises instead of silently
    switching methods.
    """
    import torch

    from .tensorrt_utils import require_tensorrt

    trt = require_tensorrt()
    base_name = {"entropy": "IInt8EntropyCalibrator2", "minmax": "IInt8MinMaxCalibrator"}[method]
    base = getattr(trt, base_name, None)
    if base is None:
        raise RuntimeError(
            f"TensorRT {trt.__version__} does not provide {base_name}. The pre-registered method cannot be used "
            "with this version. STOP: choose a TensorRT version that supports it, or document and justify a "
            "switch to explicit Q/DQ quantization (ModelOpt) before building any engine.")
    if len(images_nchw) % batch_size:
        raise ValueError(f"{len(images_nchw)} calibration images not divisible by batch {batch_size}")
    if cache_path.exists():
        raise FileExistsError(f"{cache_path} exists; every engine must be calibrated from scratch")

    class _Calibrator(base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            base.__init__(self)
            self.data = np.ascontiguousarray(images_nchw, dtype=np.float32)
            self.batch_size = batch_size
            self.pos = 0
            self.device_buf = torch.empty((batch_size, *self.data.shape[1:]), dtype=torch.float32, device="cuda")
            self.batches_served = 0

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names):  # noqa: ANN001 - TensorRT signature
            if self.pos >= len(self.data):
                return None
            self.device_buf.copy_(torch.from_numpy(self.data[self.pos:self.pos + self.batch_size]))
            self.pos += self.batch_size
            self.batches_served += 1
            return [int(self.device_buf.data_ptr())]

        def read_calibration_cache(self):
            return None  # never reuse a cache: each engine is calibrated on its own data

        def write_calibration_cache(self, cache) -> None:  # noqa: ANN001
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(bytes(cache))

    return _Calibrator()


def parse_calibration_cache(data: bytes | str) -> dict[str, float]:
    """Parse a TensorRT calibration cache into ``{tensor_name: scale}``.

    Format: a header line (e.g. ``TRT-100x-EntropyCalibration2``) then
    ``name: <8 hex digits>`` lines, the hex being the big-endian float32 scale.
    """
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    scales = {}
    for line in text.splitlines()[1:]:
        if ":" not in line:
            continue
        name, hexval = line.rsplit(":", 1)
        hexval = hexval.strip()
        if len(hexval) != 8:
            continue
        scales[name.strip()] = struct.unpack("!f", bytes.fromhex(hexval))[0]
    return scales


def compare_scales(scales: Mapping[str, Mapping[str, float]], reference: str) -> pd.DataFrame:
    """Per-tensor scale and relative difference from the reference engine's scale."""
    ref = scales[reference]
    rows = []
    for engine, sc in scales.items():
        for tensor, s in sc.items():
            r = ref.get(tensor)
            rows.append({"engine": engine, "tensor": tensor, "scale": s, "reference_scale": r,
                         "dynamic_range": s * 127.0,
                         "rel_diff": (s - r) / r if r not in (None, 0) else np.nan})
    return pd.DataFrame(rows)


def scale_summary(comp: pd.DataFrame) -> pd.DataFrame:
    """Per engine: median / 95th-percentile / max absolute relative scale difference."""
    g = comp.assign(abs_rel=comp["rel_diff"].abs()).groupby("engine")["abs_rel"]
    return pd.DataFrame({"median_abs_rel_diff": g.median(), "p95_abs_rel_diff": g.quantile(0.95),
                         "max_abs_rel_diff": g.max(), "n_tensors": g.size()}).reset_index()
