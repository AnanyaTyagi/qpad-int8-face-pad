import numpy as np
import pytest

from qpad.preprocessing import (ROLE_CALIB, ROLE_EVAL, ROLE_TRAIN, InsufficientFramesError, VideoSample,
                                decode_sample, encode_sample, iter_shard, plan_frames, plan_shards,
                                select_evenly_spaced, to_model_input, verify_shard, write_shard_tar)

FR = {"evaluation_per_video": 20, "training_per_video": 20, "training_sampling": "evenly_spaced",
      "calibration_per_video": 5}


def test_evenly_spaced_includes_ends_and_is_deterministic():
    c = list(range(150))
    s = select_evenly_spaced(c, 20)
    assert len(s) == 20 and len(set(s)) == 20 and s[0] == 0 and s[-1] == 149
    assert s == select_evenly_spaced(list(reversed(c)), 20)
    gaps = np.diff(s)
    assert gaps.max() - gaps.min() <= 1


def test_evenly_spaced_skips_invalid_frames():
    c = [i for i in range(100) if i % 7]   # some frames invalid
    assert set(select_evenly_spaced(c, 20)) <= set(c)


def test_insufficient_frames_raises():
    with pytest.raises(InsufficientFramesError):
        select_evenly_spaced(range(19), 20)


def test_plan_roles_train_vs_eval():
    plan_tr = plan_frames(range(150), "train", FR)
    calib = [f for f, r in plan_tr.items() if r & ROLE_CALIB]
    train = [f for f, r in plan_tr.items() if r & ROLE_TRAIN]
    assert len(calib) == 5 and set(calib) <= set(train) and len(train) == 20
    assert sum(1 for r in plan_tr.values() if r & ROLE_EVAL) == 20
    plan_dev = plan_frames(range(150), "dev", FR)
    assert all(r == ROLE_EVAL for r in plan_dev.values()) and len(plan_dev) == 20


def test_to_model_input_normalization():
    norm = {"scale": 255.0, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    x = np.zeros((2, 4, 4, 3), np.uint8)
    x[..., 0] = 255
    y = to_model_input(x, norm)
    assert y.shape == (2, 3, 4, 4) and y.dtype == np.float32 and y.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(y[0, 0, 0, 0], (1 - 0.485) / 0.229, rtol=1e-6)
    np.testing.assert_allclose(y[0, 1, 0, 0], (0 - 0.456) / 0.224, rtol=1e-6)


def test_plan_shards_deterministic():
    ids = [f"v{i:03d}" for i in range(7)]
    a = plan_shards(list(reversed(ids)), "train", 3)
    assert a == plan_shards(ids, "train", 3)
    assert [n for n, _ in a] == ["train_shard_0000", "train_shard_0001", "train_shard_0002"]
    assert a[-1][1] == ["v006"]


def _sample(vid, n=3):
    rng = np.random.default_rng(0)
    return VideoSample(vid, rng.integers(0, 255, (n, 8, 8, 3), dtype=np.uint8), np.arange(n), np.arange(n) + 1,
                       np.full(n, 1, np.int8), np.zeros((n, 4)), np.zeros((n, 2, 3)), np.zeros((n, 4)),
                       np.zeros(n), [{}] * n)


def test_shard_roundtrip_and_verification(tmp_path):
    s1, s2 = _sample("a"), _sample("b")
    members = {"a.npz": encode_sample(s1), "b.npz": encode_sample(s2)}
    p = tmp_path / "shard.tar"
    write_shard_tar(p, members)
    verify_shard(p, ["a.npz", "b.npz"])
    got = dict(iter_shard(p))
    np.testing.assert_array_equal(got["a.npz"]["images"], s1.images)
    assert str(got["b.npz"]["video_id"]) == "b"
    with pytest.raises(IOError):
        verify_shard(p, ["a.npz"])
    assert decode_sample(members["a.npz"])["frame_number"].tolist() == [1, 2, 3]


def test_same_size_corrupted_shard_is_not_complete(tmp_path):
    import json

    from qpad.preprocessing import shard_is_complete
    from qpad.reproducibility import file_sha256

    tar = tmp_path / "train_shard_000.tar"
    write_shard_tar(tar, {"a.npz": encode_sample(_sample("a"))})
    meta = {"shard": tar.stem, "sha256": file_sha256(tar), "size_bytes": tar.stat().st_size, "failed": {}}
    (tmp_path / f"{tar.stem}.json").write_text(json.dumps(meta))
    assert shard_is_complete(tmp_path, tar.stem, retry_failed=False)
    data = bytearray(tar.read_bytes())
    data[len(data) // 2] ^= 0xFF                      # same size, different content
    tar.write_bytes(bytes(data))
    assert not shard_is_complete(tmp_path, tar.stem, retry_failed=False)
