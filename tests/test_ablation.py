"""One-factor ablation analysis on synthetic scores (no TensorRT, no biometric data)."""

import numpy as np
import pandas as pd
import pytest

from qpad import ablation as AB
from qpad.calibration import nested_calibration_sets
from qpad.config import load_config
from qpad.reproducibility import write_json
from qpad.thresholds import select_bpcer_threshold

from conftest import REPO


def videos(bona, prnt, replay):
    scores = [*bona, *prnt, *replay]
    cls = ["bona_fide"] * len(bona) + ["print"] * len(prnt) + ["replay"] * len(replay)
    return pd.DataFrame({"video_id": [f"v{i}" for i in range(len(scores))], "class": cls, "score": scores})


def test_bpcer_threshold_rejects_at_most_the_target_share():
    bona = np.linspace(0.5, 0.99, 20)            # sorted, distinct
    s = np.r_[bona, np.full(10, 0.1)]
    is_bona = np.r_[np.ones(20, bool), np.zeros(10, bool)]
    res, table = select_bpcer_threshold(s, is_bona, 0.10)
    assert res.bpcer_errors == 2 and res.threshold == pytest.approx(bona[2])
    assert table["selected"].sum() == 1
    assert select_bpcer_threshold(s, is_bona, 0.15)[0].bpcer_errors == 3   # floor(0.15 * 20)
    assert select_bpcer_threshold(s, is_bona, 0.0)[0].bpcer_errors == 0
    with pytest.raises(ValueError):
        select_bpcer_threshold(s, is_bona, 1.0)


def test_settings_table_matches_the_config():
    cfg = load_config(REPO / "configs")["int8"]
    t = AB.settings_table(cfg["ablation"], 5, 20).set_index("setting")
    assert t.loc["natural", "images"] == 600 and t.loc["natural", "attack_pct"] == pytest.approx(200 / 3)
    assert t.loc["attack0", "attack_pct"] == 0 and t.loc["attack10", "attack_pct"] == pytest.approx(10)
    assert list(t.loc[["images200", "images60", "images20"], "images"]) == [200, 60, 20]
    assert (t.loc[t.index.str.startswith("attack"), "images"] == 600).all()
    assert t.loc["natural", "lines"] == "attack_share,images"
    assert dict(t.loc["natural", ["bona_fide", "print", "replay"]]) == cfg["conditions"]["natural"]


def test_settings_table_rejects_a_budget_off_the_batch():
    acfg = {"baseline": "a", "settings": {"a": {"bona_fide": 1, "print": 1, "replay": 1}}, "lines": {"x": ["a"]}}
    with pytest.raises(ValueError):
        AB.settings_table(acfg, 5, 20)


def test_ablation_sets_are_nested_within_a_seed():
    train = pd.DataFrame({"video_id": [f"{c}{i:03d}" for c in "bpr" for i in range(150)],
                          "class": [c for c in ("bona_fide", "print", "replay") for _ in range(150)]})
    big = nested_calibration_sets(train, {"natural": {"bona_fide": 40, "print": 40, "replay": 40}}, [1], 120)
    small = nested_calibration_sets(train, {"images60": {"bona_fide": 4, "print": 4, "replay": 4}}, [1], 12)
    for c in ("bona_fide", "print", "replay"):
        first = big[(big["class"] == c) & (big["rank_in_class"] < 4)]["video_id"].tolist()
        assert small[small["class"] == c]["video_id"].tolist() == first


def write_result(d, dev, test, thr_a, thr_b):
    d.mkdir(parents=True)
    dev.to_csv(d / "dev_video_scores.csv", index=False)
    test.to_csv(d / "test_video_scores.csv", index=False)
    write_json(d / "metrics.json", {"threshold_A_fp32": thr_a, "threshold_B_engine": {"threshold": thr_b}})


def test_rules_ranges_and_shifts(tmp_path):
    dev = videos(np.linspace(0.6, 0.99, 20), np.full(10, 0.1), np.full(10, 0.2))
    dirs = {}
    for setting, shift in (("natural", 0.0), ("attack0", 0.5)):
        for seed in (1, 2, 3):
            test = videos([0.9] * 10, [0.3 + shift] * 10, [0.2] * 10)
            d = tmp_path / f"{setting}_s{seed}"
            write_result(d, dev, test, 0.5, 0.55 + 0.01 * seed)
            dirs[(setting, seed)] = d
    long = AB.collect(dirs, [0.10, 0.15])
    assert set(long["rule"]) == {"A_fp32", "B_dev_eer", "C_bpcer10", "C_bpcer15"} and len(long) == 24
    c10 = long[long["rule"] == "C_bpcer10"].iloc[0]
    assert c10["dev_bpcer_errors"] == 2 and c10["dev_bona_fide"] == 20
    ranges = AB.setting_ranges(long, "natural").set_index(["rule", "setting", "metric"])
    hit = ranges.loc[("A_fp32", "attack0", "print_apcer_pct")]
    assert hit["separated"] and hit["direction"] == "higher" and hit["min"] == 100
    assert not ranges.loc[("A_fp32", "attack0", "replay_apcer_pct"), "separated"]
    assert ranges.loc[("B_dev_eer", "natural", "threshold"), "max"] == pytest.approx(0.58)
    text = AB.range_text(ranges.reset_index(), ["A_fp32"], ["print_apcer_pct"])
    assert text.loc["attack0"].iloc[0].endswith("*")
    shifts = AB.score_shifts_vs_baseline(dirs, "natural").set_index(["setting", "seed", "class"])
    assert shifts.loc[("attack0", 1, "print"), "mean_delta"] == pytest.approx(0.5)
    assert set(shifts.index.get_level_values("setting")) == {"attack0"}


def test_ablation_figure(tmp_path):
    pytest.importorskip("matplotlib")
    long = pd.DataFrame([{"setting": s, "seed": seed, "rule": r, "print_apcer_pct": seed, "replay_apcer_pct": 2.0,
                          "bpcer_pct": 1.0} for s in ("natural", "attack0") for seed in (1, 2) for r in ("A_fp32",)])
    table = pd.DataFrame({"setting": ["natural", "attack0"], "attack_pct": [66.7, 0.0], "images": [600, 600]})
    out = AB.fig_ablation_line(long, table, "attack_share", ["natural", "attack0"], ["A_fp32"], tmp_path, "f")
    assert (tmp_path / "f.png").is_file() and out["caption"]
