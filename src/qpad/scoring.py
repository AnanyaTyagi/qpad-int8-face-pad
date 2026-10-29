"""Frame and video scoring shared by every backend."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .datasets import FrameStore

# A backend maps a float32 NCHW batch to a (N, 1, 14, 14) or (N, 14, 14) pixel map.
MapBackend = Callable[[np.ndarray], np.ndarray]


def grid_to_frame_scores(maps: np.ndarray) -> np.ndarray:
    """Mean of each pixel map -> one score per frame (float64 accumulation)."""
    maps = np.asarray(maps)
    return maps.reshape(maps.shape[0], -1).astype(np.float64).mean(axis=1)


def score_store(store: FrameStore, backend: MapBackend, norm_cfg, batch_size: int = 64,
                progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    """Score every frame in ``store`` with ``backend``; returns one row per frame."""
    from .preprocessing import to_model_input

    scores = np.empty(len(store), dtype=np.float64)
    for start in range(0, len(store), batch_size):
        stop = min(start + batch_size, len(store))
        x = to_model_input(np.asarray(store.images[start:stop]), norm_cfg)
        scores[start:stop] = grid_to_frame_scores(backend(x))
        if progress:
            progress(stop, len(store))
    keep = [c for c in ("video_id", "frame_idx", "frame_number", "split", "class", "label", "subject",
                        "phone", "session", "pai_code") if c in store.index]
    out = store.index[keep].copy()
    out["frame_score"] = scores
    return out


def aggregate_video_scores(frame_scores: pd.DataFrame, expected_frames: "int | Mapping[str, int] | None" = None
                           ) -> pd.DataFrame:
    """Video score = mean of its frame scores. Raises if any video has the wrong frame count.

    ``expected_frames`` is one count for every video, or a per-video mapping (e.g.
    :class:`qpad.data_exceptions.FrameCounts`, which covers documented reduced-frame videos).
    """
    meta = [c for c in ("split", "class", "label", "subject", "phone", "session", "pai_code") if c in frame_scores]
    g = frame_scores.groupby("video_id", sort=True)
    out = g[meta].first()
    out["n_frames"] = g.size()
    out["score"] = g["frame_score"].mean()
    out = out.reset_index()
    if expected_frames is not None:
        want = (out["video_id"].map(lambda v: expected_frames[v]) if isinstance(expected_frames, Mapping)
                else expected_frames)
        bad = out[out["n_frames"] != want]
        if len(bad):
            raise ValueError(f"{len(bad)} videos do not have {expected_frames} frames, e.g. {bad['video_id'].tolist()[:5]}")
    return out


def torch_backend(model, device: str = "cpu") -> MapBackend:
    """Wrap a PyTorch DeepPixBiS (eval mode, no gradients) as a :data:`MapBackend`."""
    import torch

    model.eval()

    def run(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            out_map, _ = model(torch.from_numpy(x).to(device))
        return out_map.float().cpu().numpy()

    return run
