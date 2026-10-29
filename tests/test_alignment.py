import numpy as np
import pytest

from qpad.alignment import align_face, apply_affine, order_eyes, similarity_from_eyes

TL, TR = (55.0, 44.0), (168.0, 44.0)


def test_eyes_map_exactly_to_targets():
    m = similarity_from_eyes((300, 400), (420, 380), TL, TR)
    out = apply_affine(m, np.array([[300, 400], [420, 380]]))
    np.testing.assert_allclose(out, [TL, TR], atol=1e-9)
    # similarity: equal scale on both axes, orthogonal rotation
    a, b = m[0, 0], m[1, 0]
    np.testing.assert_allclose(m[:, :2], [[a, -b], [b, a]])


def test_eye_order_is_by_x():
    l, r = order_eyes((400, 10), (100, 20))
    assert tuple(l) == (100, 20) and tuple(r) == (400, 10)


def test_align_face_geometry_and_channel_order():
    img = np.zeros((480, 640, 3), np.uint8)
    img[:, :, 0] = 255  # pure red in RGB
    ex, ey = (250.0, 200.0), (370.0, 200.0)
    res = align_face(img, ey, ex, 224, TL, TR)  # deliberately pass eyes in reversed order
    assert res.crop.shape == (224, 224, 3) and res.crop.dtype == np.uint8
    centre = res.crop[100:120, 100:120]
    assert (centre[..., 0] == 255).all() and (centre[..., 1:] == 0).all()   # RGB preserved
    assert res.eye_left == ex
    x0, y0, x1, y1 = res.crop_bounds
    scale = (TR[0] - TL[0]) / (ey[0] - ex[0])
    assert x1 - x0 == pytest.approx(224 / scale)
    assert res.outside_fraction == 0.0


def test_outside_fraction_reported():
    img = np.full((100, 100, 3), 128, np.uint8)
    res = align_face(img, (2, 5), (12, 5), 224, TL, TR)
    # scale = 113/10; source x < 0 for output x < 55 - 2 * 11.3 = 32.4 -> 32 of 224 columns; no rows outside
    assert res.outside_fraction == 32 / 224


def test_rejects_bgr_or_wrong_dtype():
    with pytest.raises(ValueError):
        align_face(np.zeros((10, 10), np.uint8), (1, 1), (5, 1), 224, TL, TR)
