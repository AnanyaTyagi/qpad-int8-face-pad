import pandas as pd
import pytest

from qpad.data_exceptions import DataExceptions, expected_eval_counts
from qpad.manifest import validate_audit
from qpad.preprocessing import InsufficientFramesError, plan_frames, ROLE_EVAL
from qpad.scoring import aggregate_video_scores

FR = {"evaluation_per_video": 20, "training_per_video": 20, "training_sampling": "evenly_spaced",
      "calibration_per_video": 5}


def test_real_config_exceptions(real_cfg):
    ex = DataExceptions.from_cfg(real_cfg)
    assert ex.excluded_frames("6_1_12_1") == {90}
    assert ex.reduced_frame_videos == {"3_1_54_5": 2, "5_1_54_3": 19}
    counts = ex.expected_eval_counts(20)
    assert counts["3_1_54_5"] == 2 and counts["1_1_01_1"] == 20


def test_reduced_plan_uses_all_valid_frames():
    plan = plan_frames([3, 40], "test", FR, n_eval=2)
    assert plan == {3: ROLE_EVAL, 40: ROLE_EVAL}
    with pytest.raises(InsufficientFramesError):
        plan_frames([3, 40, 77], "test", FR, n_eval=2)       # documented count must match exactly
    with pytest.raises(ValueError):
        plan_frames(list(range(5)), "train", FR, n_eval=5)    # never in train
    with pytest.raises(InsufficientFramesError):
        plan_frames(list(range(19)), "test", FR)              # undocumented short video still fails


def test_aggregation_with_per_video_counts(real_cfg, pre_cfg):
    f = pd.DataFrame({"video_id": ["3_1_54_5"] * 2 + ["1_1_36_1"] * 20, "class": ["replay"] * 2 + ["bona_fide"] * 20,
                      "frame_score": [0.2, 0.4] + [0.9] * 20})
    v = aggregate_video_scores(f, expected_eval_counts({"dataset": real_cfg, "preprocessing": pre_cfg}))
    assert v.set_index("video_id").loc["3_1_54_5", "score"] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        aggregate_video_scores(f.iloc[1:], expected_eval_counts({"dataset": real_cfg, "preprocessing": pre_cfg}))


def _audited(rows):
    base = {"status": "ok", "error": "", "frame_count": 151, "decode_status": "ok", "ann_malformed_rows": 0,
            "ann_valid_in_range": 151, "ann_frame_base": "0", "split": "test"}
    return pd.DataFrame([{**base, **r} for r in rows])


def _checks(df, cfg):
    return {c.name: c for c in validate_audit(df, cfg, True, len(df))}


def test_documented_exceptions_accepted(real_cfg):
    df = _audited([{"video_id": "6_1_12_1", "split": "train", "status": "failed", "decode_status": "failed",
                    "error": "decode: [mjpeg] error dc", "ann_valid_in_range": 121},
                   {"video_id": "3_1_54_5", "ann_valid_in_range": 2},
                   {"video_id": "5_1_54_3", "ann_valid_in_range": 19},
                   {"video_id": "1_1_36_1"}])
    c = _checks(df, real_cfg)
    for name in ("audit_errors", "full_decode", "annotation_coverage", "documented_exceptions_match_data"):
        assert c[name].status == "pass", (name, c[name].examples)


def test_undocumented_or_mismatched_cases_fail(real_cfg):
    df = _audited([{"video_id": "6_1_12_1", "split": "train", "status": "failed", "decode_status": "failed",
                    "error": "decode: x | annotation: FileNotFoundError", "ann_valid_in_range": 121},  # extra error
                   {"video_id": "3_1_54_5", "ann_valid_in_range": 3},                              # count changed
                   {"video_id": "5_1_54_3", "ann_valid_in_range": 19},
                   {"video_id": "1_1_36_1", "ann_valid_in_range": 10, "decode_status": "failed", "status": "failed",
                    "error": "decode: y"}])                                                         # undocumented
    c = _checks(df, real_cfg)
    assert set(c["audit_errors"].examples) == {"6_1_12_1", "1_1_36_1"}
    assert c["full_decode"].examples == ["1_1_36_1"]
    assert set(c["annotation_coverage"].examples) == {"3_1_54_5", "1_1_36_1"}
    assert c["documented_exceptions_match_data"].status == "fail"


def test_exception_that_no_longer_applies_fails(real_cfg):
    df = _audited([{"video_id": "6_1_12_1", "split": "train", "ann_valid_in_range": 121},   # decodes cleanly
                   {"video_id": "3_1_54_5", "ann_valid_in_range": 2}, {"video_id": "5_1_54_3", "ann_valid_in_range": 19}])
    assert _checks(df, real_cfg)["documented_exceptions_match_data"].status == "fail"
