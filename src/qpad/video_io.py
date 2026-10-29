"""Video probing (ffprobe), full-decode checks (ffmpeg) and frame reading (OpenCV)."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np

FFPROBE_TIMEOUT_S = 300
FFMPEG_TIMEOUT_S = 900


def ffprobe_video(path: Path | str) -> dict[str, Any]:
    """Frame count (by counting decoded frames), resolution, FPS, duration and codec."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
           "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_read_frames,duration",
           "-show_entries", "format=duration", "-of", "json", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT_S)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr.strip()[:500]}")
    data = json.loads(res.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe found no video stream")
    s = streams[0]
    rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(rate)) if rate not in ("0/0", "") else float("nan")
    except (ValueError, ZeroDivisionError):
        fps = float("nan")
    duration = s.get("duration") or (data.get("format") or {}).get("duration")
    n = s.get("nb_read_frames")
    return {
        "codec": s.get("codec_name"),
        "width": int(s["width"]) if s.get("width") else None,
        "height": int(s["height"]) if s.get("height") else None,
        "fps": fps,
        "frame_count": int(n) if n not in (None, "N/A") else None,
        "duration_s": float(duration) if duration not in (None, "N/A") else None,
    }


def ffmpeg_full_decode(path: Path | str) -> tuple[str, str]:
    """Decode every frame; returns ``("ok", "")`` or ``("failed", <stderr excerpt>)``."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S)
    err = res.stderr.strip()
    if res.returncode != 0 or err:
        return "failed", err[:500] or f"exit code {res.returncode}"
    return "ok", ""


def read_frames(path: Path | str, wanted_indices: Iterable[int]) -> dict[int, np.ndarray]:
    """Decode sequentially and return the requested 0-based frames as **RGB** uint8 arrays.

    Sequential decoding (no seeking) is used because AVI seeking in OpenCV can be
    inexact; it guarantees frame ``i`` is the i-th decoded frame.
    """
    import cv2

    wanted = sorted(set(int(i) for i in wanted_indices))
    if not wanted:
        return {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    out: dict[int, np.ndarray] = {}
    target = set(wanted)
    last = wanted[-1]
    idx = 0
    try:
        while idx <= last:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx in target:
                out[idx] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()
    missing = target - out.keys()
    if missing:
        raise RuntimeError(f"Decoded only {idx} frames; missing requested frames {sorted(missing)[:5]}")
    return out
