import struct

import pandas as pd
import pytest

from qpad.calibration import (attach_calibration_frames, compare_scales, nested_calibration_sets,
                              parse_calibration_cache, validate_conditions)


@pytest.fixture
def train_videos():
    rows = [{"video_id": f"{c[0]}{i:03d}", "class": c} for c in ("bona_fide", "print", "replay") for i in range(360)]
    return pd.DataFrame(rows)


def test_all_conditions_use_budget(int8_cfg):
    validate_conditions(int8_cfg["conditions"], int8_cfg["calibration_budget"]["videos"])
    with pytest.raises(ValueError):
        validate_conditions({"bad": {"bona_fide": 1, "print": 1, "replay": 1}}, 120)


def test_nested_prefix_property(train_videos, int8_cfg):
    sel = nested_calibration_sets(train_videos, int8_cfg["conditions"], [1, 2, 3, 4, 5], 120)
    assert sel.groupby(["seed", "condition"]).size().eq(120).all()
    for seed, g in sel.groupby("seed"):
        for c in ("bona_fide", "print", "replay"):
            lists = {cond: d.sort_values("rank_in_class")["video_id"].tolist()
                     for cond, d in g[g["class"] == c].groupby("condition")}
            longest = max(lists.values(), key=len)
            for lst in lists.values():
                assert lst == longest[:len(lst)], f"seed {seed} {c} not nested"
    comp = sel.groupby(["condition", "class"]).size().unstack(fill_value=0) / 5
    for cond, spec in int8_cfg["conditions"].items():
        for c in ("bona_fide", "print", "replay"):
            assert comp.loc[cond].get(c, 0) == spec[c]


def test_seeds_deterministic_and_different(train_videos, int8_cfg):
    a = nested_calibration_sets(train_videos, int8_cfg["conditions"], [1, 2], 120)
    b = nested_calibration_sets(train_videos.sample(frac=1, random_state=0), int8_cfg["conditions"], [1, 2], 120)
    pd.testing.assert_frame_equal(a, b)   # input order does not matter
    s1 = set(a[(a.seed == 1) & (a.condition == "natural")]["video_id"])
    s2 = set(a[(a.seed == 2) & (a.condition == "natural")]["video_id"])
    assert s1 != s2


def test_attach_frames_requires_five(train_videos, int8_cfg):
    sel = nested_calibration_sets(train_videos, {"natural": int8_cfg["conditions"]["natural"]}, [1], 120)
    frames = pd.DataFrame([{"video_id": v, "frame_idx": k} for v in train_videos["video_id"] for k in range(5)])
    out = attach_calibration_frames(sel, frames, 5)
    assert len(out) == 600
    with pytest.raises(ValueError):
        attach_calibration_frames(sel, frames[frames.frame_idx < 4], 5)


def test_parse_calibration_cache():
    h = struct.pack("!f", 0.0125).hex()
    txt = f"TRT-100x-EntropyCalibration2\ninput: {h}\n/enc/conv0/Conv_output_0: {struct.pack('!f', 2.5).hex()}\n"
    sc = parse_calibration_cache(txt.encode())
    assert sc["input"] == pytest.approx(0.0125) and sc["/enc/conv0/Conv_output_0"] == pytest.approx(2.5)
    comp = compare_scales({"nat": sc, "other": {"input": 0.025}}, "nat")
    assert comp[(comp.engine == "other") & (comp.tensor == "input")]["rel_diff"].iloc[0] == pytest.approx(1.0)
