import numpy as np
import pandas as pd
import pytest

from qpad.metrics import auroc, decision_flips, error_breakdown, pad_metrics
from qpad.scoring import aggregate_video_scores, grid_to_frame_scores
from qpad.thresholds import predict_bona_fide, select_eer_threshold, threshold_table


def test_grid_to_frame_score_is_mean():
    maps = np.zeros((2, 1, 14, 14), np.float32)
    maps[0] = 1.0
    maps[1, 0, :7] = 1.0
    np.testing.assert_allclose(grid_to_frame_scores(maps), [1.0, 0.5])


def test_frame_to_video_aggregation_and_count_check():
    f = pd.DataFrame({"video_id": ["a"] * 3 + ["b"] * 3, "class": ["bona_fide"] * 3 + ["print"] * 3,
                      "frame_score": [0.9, 0.6, 0.6, 0.1, 0.2, 0.3]})
    v = aggregate_video_scores(f, expected_frames=3)
    np.testing.assert_allclose(v.set_index("video_id")["score"], [0.7, 0.2])
    with pytest.raises(ValueError):
        aggregate_video_scores(f.iloc[:5], expected_frames=3)


def test_decision_rule_is_greater_equal():
    assert predict_bona_fide(np.array([0.5, 0.49999]), 0.5).tolist() == [True, False]


def _videos(bona, prnt, repl):
    return pd.DataFrame({"video_id": [f"v{i}" for i in range(len(bona) + len(prnt) + len(repl))],
                         "class": ["bona_fide"] * len(bona) + ["print"] * len(prnt) + ["replay"] * len(repl),
                         "score": list(bona) + list(prnt) + list(repl)})


def test_apcer_bpcer_acer_counts():
    v = _videos([0.9, 0.8, 0.4, 0.7], [0.1, 0.6, 0.2, 0.3], [0.65, 0.55, 0.1, 0.0])
    m = pad_metrics(v, 0.5)
    assert m["bpcer"]["errors"] == 1 and m["bpcer"]["total"] == 4
    assert m["print_apcer"]["errors"] == 1 and m["print_apcer"]["total"] == 4
    assert m["replay_apcer"]["errors"] == 2
    assert m["apcer_pooled"]["errors"] == 3 and m["apcer_pooled"]["total"] == 8
    assert m["acer"] == pytest.approx((3 / 8 + 1 / 4) / 2)
    assert m["acer_max_pai"] == pytest.approx((2 / 4 + 1 / 4) / 2)
    assert m["print_apcer"]["text"] == "1/4 = 25.00%"
    assert sum(m["confusion"].values()) == 12


def test_auroc_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    s = np.round(rng.random(200), 2)   # with ties
    y = rng.random(200) > 0.4
    assert auroc(s, y) == pytest.approx(sk.roc_auc_score(y, s))


def test_threshold_minimises_abs_difference():
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    bona = np.array([1, 1, 1, 0, 0, 0], bool)
    res, table = select_eer_threshold(scores, bona)
    assert res.apcer_errors == 0 and res.bpcer_errors == 0
    assert res.threshold == 0.7   # ties at APCER=BPCER=0 for t in (0.3, 0.7] -> highest candidate
    assert table["selected"].sum() == 1


def test_threshold_tie_break_acer_then_higher():
    # Candidates where |APCER-BPCER| ties; lowest ACER must win.
    scores = np.array([0.6, 0.4, 0.5, 0.45])
    bona = np.array([1, 1, 0, 0], bool)
    res, t = select_eer_threshold(scores, bona)
    tied = t[t["abs_diff_key"] == t["abs_diff_key"].min()]
    best_acer = tied["acer_key"].min()
    assert res.threshold == tied.loc[tied["acer_key"] == best_acer, "threshold"].max()


def test_threshold_tie_break_exact_integer_arithmetic():
    # 1/3 vs 2/6 must be treated as an exact tie (float subtraction would not be exact).
    scores = np.array([0.95, 0.9, 0.85, 0.15, 0.1, 0.05, 0.92, 0.12, 0.5])
    bona = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0], bool)
    t = threshold_table(scores, bona)
    assert (t["abs_diff_key"] >= 0).all()
    res, _ = select_eer_threshold(scores, bona)
    row = t[t["threshold"] == res.threshold].iloc[0]
    assert row["abs_diff_key"] == t["abs_diff_key"].min()


def test_threshold_rejects_single_class():
    with pytest.raises(ValueError):
        select_eer_threshold(np.array([0.1, 0.2]), np.array([True, True]))


def test_error_breakdown_and_flips():
    v = _videos([0.9, 0.4], [0.6, 0.1], [0.7, 0.2])
    v["phone"] = [1, 2, 1, 2, 1, 2]
    b = error_breakdown(v, 0.5, by=["phone"]).set_index("phone")
    assert b.loc[1, "print_apcer_errors"] == 1 and b.loc[2, "bpcer_errors"] == 1
    other = v.copy()
    other["score"] = [0.9, 0.6, 0.4, 0.1, 0.7, 0.6]
    f = decision_flips(v, other, 0.5, 0.5)
    assert f["attack_accepted_to_rejected"] == 1 and f["attack_rejected_to_accepted"] == 1
    assert f["bona_fide_rejected_to_accepted"] == 1
