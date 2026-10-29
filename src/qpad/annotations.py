"""Parsing and validating OULU-NPU per-frame eye-coordinate annotations.

Each line is expected to be ``frame, x1, y1, x2, y2`` (comma and/or whitespace
separated). The two points are the eye centres. Which point is which eye is
not assumed: alignment orders the eyes by their x-coordinate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ANNOTATION_COLUMNS = ("frame", "x1", "y1", "x2", "y2")
_SPLIT = re.compile(r"[,\s;]+")
MAX_MALFORMED_EXAMPLES = 5


@dataclass
class AnnotationParse:
    table: pd.DataFrame                  # columns ANNOTATION_COLUMNS (frame as int)
    n_lines: int
    n_malformed: int
    malformed_examples: list[str] = field(default_factory=list)


def parse_eye_annotation(text: str) -> AnnotationParse:
    """Parse annotation text; malformed lines are counted and reported, never silently dropped."""
    rows: list[tuple[float, ...]] = []
    malformed: list[str] = []
    n_lines = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        n_lines += 1
        parts = [p for p in _SPLIT.split(line) if p]
        try:
            if len(parts) != len(ANNOTATION_COLUMNS):
                raise ValueError(f"{len(parts)} fields")
            vals = tuple(float(p) for p in parts)
            if not float(vals[0]).is_integer():
                raise ValueError("non-integer frame number")
        except ValueError:
            malformed.append(line)
            continue
        rows.append(vals)
    table = pd.DataFrame(rows, columns=list(ANNOTATION_COLUMNS))
    if len(table):
        table["frame"] = table["frame"].astype(np.int64)
    else:
        table["frame"] = table["frame"].astype(np.int64)
    return AnnotationParse(table, n_lines, len(malformed), malformed[:MAX_MALFORMED_EXAMPLES])


def valid_eye_mask(table: pd.DataFrame, min_eye_distance_px: float) -> np.ndarray:
    """Rows whose coordinates are finite, non-zero and at least ``min_eye_distance_px`` apart."""
    xy = table[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    finite = np.isfinite(xy).all(axis=1)
    nonzero = ~(xy == 0).all(axis=1)
    dist = np.hypot(xy[:, 0] - xy[:, 2], xy[:, 1] - xy[:, 3])
    return finite & nonzero & (dist >= min_eye_distance_px)


def infer_frame_base(frames: np.ndarray, n_video_frames: int) -> str:
    """Infer whether annotation frame numbers are 0- or 1-based.

    Returns ``"0"``, ``"1"`` or ``"ambiguous"``. Rules:
      * minimum frame 0                      -> 0-based
      * maximum frame equals the frame count -> 1-based
      * otherwise                            -> ambiguous (needs manual inspection)
    """
    if len(frames) == 0 or n_video_frames <= 0:
        return "ambiguous"
    lo, hi = int(np.min(frames)), int(np.max(frames))
    if lo == 0:
        return "0" if hi <= n_video_frames - 1 else "ambiguous"
    if hi == n_video_frames:
        return "1"
    return "ambiguous"


def resolve_dataset_frame_base(configured: str | int, per_video_bases: pd.Series) -> int:
    """Return the frame base for the whole dataset or raise if it cannot be decided safely."""
    if str(configured) in {"0", "1"}:
        return int(configured)
    values = per_video_bases.astype(str)
    decided = values[values.isin(["0", "1"])].unique()
    if len(decided) == 1 and (values.isin(["0", "1"]) | (values == "ambiguous")).all():
        return int(decided[0])
    raise ValueError(
        f"Cannot infer annotation frame base automatically (observed: {values.value_counts().to_dict()}). "
        "Inspect the audit report and set annotations.frame_index_base to 0 or 1 in dataset.yaml.")


def to_zero_based(table: pd.DataFrame, base: int) -> pd.DataFrame:
    """Return a copy with ``frame_idx`` = 0-based decoded frame index."""
    out = table.copy()
    out["frame_idx"] = out["frame"].astype(np.int64) - int(base)
    return out


def annotation_summary(parsed: AnnotationParse, n_video_frames: int, min_eye_distance_px: float) -> dict:
    """Per-video annotation statistics stored in the audit manifest."""
    t = parsed.table
    valid = valid_eye_mask(t, min_eye_distance_px) if len(t) else np.zeros(0, bool)
    base = infer_frame_base(t["frame"].to_numpy(), n_video_frames)
    shift = int(base) if base in {"0", "1"} else 0
    idx = t["frame"].to_numpy() - shift
    in_range = (idx >= 0) & (idx < max(n_video_frames, 0))
    return {
        "ann_rows": int(len(t)),
        "ann_lines": int(parsed.n_lines),
        "ann_malformed_rows": int(parsed.n_malformed),
        "ann_malformed_examples": " || ".join(parsed.malformed_examples),
        "ann_valid_rows": int(valid.sum()),
        "ann_min_frame": int(t["frame"].min()) if len(t) else None,
        "ann_max_frame": int(t["frame"].max()) if len(t) else None,
        "ann_duplicate_frames": int(t["frame"].duplicated().sum()) if len(t) else 0,
        "ann_frame_base": base,
        "ann_valid_in_range": int((valid & in_range).sum()),
    }
