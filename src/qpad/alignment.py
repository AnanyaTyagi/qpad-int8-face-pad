"""Deterministic two-eye similarity alignment.

Two points determine a similarity transform (rotation + uniform scale +
translation) exactly. We map the image-left eye (smaller x) and image-right eye
to fixed target positions in a ``size x size`` crop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

INTERPOLATION = {"linear": 1, "cubic": 2, "area": 3, "nearest": 0}  # cv2.INTER_* codes


@dataclass(frozen=True)
class AlignmentResult:
    crop: np.ndarray            # (size, size, 3) uint8, RGB
    matrix: np.ndarray          # (2, 3) affine mapping source (x, y) -> crop (x, y)
    crop_bounds: tuple[float, float, float, float]  # source-image bbox (x0, y0, x1, y1) of the crop
    outside_fraction: float     # fraction of crop pixels sampled outside the source frame
    eye_left: tuple[float, float]
    eye_right: tuple[float, float]


def order_eyes(p1: Sequence[float], p2: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (image-left eye, image-right eye) ordered by x; ties broken by y."""
    a, b = np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64)
    return (a, b) if (a[0], a[1]) <= (b[0], b[1]) else (b, a)


def similarity_from_eyes(src_left: Sequence[float], src_right: Sequence[float],
                         dst_left: Sequence[float], dst_right: Sequence[float]) -> np.ndarray:
    """Exact 2x3 similarity matrix with ``M @ [src, 1] = dst`` for both eyes.

    Using complex numbers: ``z_dst = a * z_src + b`` with
    ``a = (dr - dl) / (sr - sl)`` and ``b = dl - a * sl``.
    """
    sl, sr = complex(*src_left), complex(*src_right)
    dl, dr = complex(*dst_left), complex(*dst_right)
    if abs(sr - sl) < 1e-9:
        raise ValueError("eye coordinates coincide; cannot align")
    a = (dr - dl) / (sr - sl)
    b = dl - a * sl
    return np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag]], dtype=np.float64)


def invert_affine(m: np.ndarray) -> np.ndarray:
    full = np.vstack([m, [0.0, 0.0, 1.0]])
    return np.linalg.inv(full)[:2]


def apply_affine(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return pts @ m[:, :2].T + m[:, 2]


def align_face(image_rgb: np.ndarray, eye1: Sequence[float], eye2: Sequence[float], size: int,
               target_left: Sequence[float], target_right: Sequence[float],
               interpolation: str = "linear", border_value: int = 0) -> AlignmentResult:
    """Warp ``image_rgb`` so the eyes land on the target positions in a ``size x size`` crop."""
    import cv2

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError(f"expected HxWx3 uint8 RGB image, got {image_rgb.shape} {image_rgb.dtype}")
    left, right = order_eyes(eye1, eye2)
    m = similarity_from_eyes(left, right, target_left, target_right)
    crop = cv2.warpAffine(image_rgb, m, (size, size), flags=INTERPOLATION[interpolation],
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(border_value,) * 3)
    inv = invert_affine(m)
    corners = apply_affine(inv, np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float64))
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    # Fraction of output pixel centres that map outside the source image.
    ys, xs = np.mgrid[0:size, 0:size]
    src = apply_affine(inv, np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1))
    h, w = image_rgb.shape[:2]
    outside = ((src[:, 0] < 0) | (src[:, 0] >= w) | (src[:, 1] < 0) | (src[:, 1] >= h)).mean()
    return AlignmentResult(crop=crop, matrix=m, crop_bounds=(float(x0), float(y0), float(x1), float(y1)),
                           outside_fraction=float(outside), eye_left=tuple(map(float, left)),
                           eye_right=tuple(map(float, right)))
