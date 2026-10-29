"""Pre-registered, documented data exceptions (see ``data_exceptions`` in configs/dataset.yaml).

Two kinds:
* ``damaged_frames``: decoded frame indices that must never be selected;
* ``reduced_frame_videos``: videos scored on all of their (fewer than 20) valid annotated frames.

Every consumer (audit, preprocessing, output verification, score aggregation) reads the same object,
so no video can receive fewer frames unless it is listed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class FrameCounts(dict):
    """``{video_id: expected evaluation frames}`` that falls back to the default for unlisted videos."""

    def __init__(self, default: int, overrides: Mapping[str, int]) -> None:
        super().__init__(overrides)
        self.default = int(default)

    def __missing__(self, key: str) -> int:
        return self.default


@dataclass(frozen=True)
class DataExceptions:
    damaged_frames: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    reduced_frame_videos: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, dataset_cfg: Mapping[str, Any]) -> "DataExceptions":
        ex = dataset_cfg.get("data_exceptions") or {}
        damaged = {str(k): tuple(int(i) for i in v) for k, v in (ex.get("damaged_frames") or {}).items()}
        reduced = {str(k): int(v) for k, v in (ex.get("reduced_frame_videos") or {}).items()}
        if any(n < 1 for n in reduced.values()):
            raise ValueError("reduced_frame_videos counts must be >= 1")
        return cls(damaged, reduced)

    def excluded_frames(self, video_id: str) -> set[int]:
        return set(self.damaged_frames.get(video_id, ()))

    def eval_frames(self, video_id: str, default: int) -> int:
        return self.reduced_frame_videos.get(video_id, default)

    def expected_eval_counts(self, default: int) -> FrameCounts:
        return FrameCounts(default, self.reduced_frame_videos)

    def documented(self, video_id: str) -> bool:
        return video_id in self.damaged_frames or video_id in self.reduced_frame_videos


def expected_eval_counts(cfgs: Mapping[str, Any]) -> FrameCounts:
    """Expected evaluation frames per video for the loaded configs."""
    return DataExceptions.from_cfg(cfgs["dataset"]).expected_eval_counts(
        int(cfgs["preprocessing"]["frames"]["evaluation_per_video"]))
