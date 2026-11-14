"""One-factor calibration ablation (notebooks 08 and 08b).

Starting from the natural calibration set (600 images, 67% attack images), one factor changes at a
time: the attack share, or the number of images. Every setting is built with several seeds, and each
engine is evaluated under several threshold rules, because the pilot showed that the engine's own
dev-EER threshold moves more between engines than the scores do.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .calibration import CLASS_ORDER
from .metrics import pad_metrics
from .thresholds import select_bpcer_threshold

RULE_LABELS = {"A_fp32": "A: fixed FP32 threshold", "B_dev_eer": "B: engine dev EER",
               "C_bpcer10": "C: dev BPCER 10%", "C_bpcer15": "C: dev BPCER 15%"}
RULE_COLORS = {"A_fp32": "#2a78d6", "C_bpcer10": "#eb6834", "C_bpcer15": "#1baf7a", "B_dev_eer": "#52514e"}
RANGE_METRICS = ("print_apcer_pct", "replay_apcer_pct", "bpcer_pct", "acer_pct", "threshold", "auroc")


def bpcer_rule(target: float) -> str:
    return f"C_bpcer{round(100 * target)}"


def settings_table(acfg: Mapping[str, Any], frames_per_video: int, batch: int) -> pd.DataFrame:
    """One row per ablation setting: videos per class, images, attack share and the lines it belongs to."""
    settings, lines = acfg["settings"], acfg["lines"]
    if acfg["baseline"] not in settings:
        raise ValueError(f"baseline {acfg['baseline']} is not a setting")
    for line, members in lines.items():
        if members[0] != acfg["baseline"] or not set(members) <= set(settings):
            raise ValueError(f"line {line} must start at the baseline and use known settings")
    rows = []
    for name, mix in settings.items():
        videos = sum(int(mix[c]) for c in CLASS_ORDER)
        images = videos * frames_per_video
        if images % batch:
            raise ValueError(f"{name}: {images} images is not a multiple of the calibration batch {batch}")
        rows.append({"setting": name, **{c: int(mix[c]) for c in CLASS_ORDER}, "videos": videos, "images": images,
                     "attack_pct": 100 * (int(mix["print"]) + int(mix["replay"])) / videos,
                     "lines": ",".join(line for line, members in lines.items() if name in members)})
    return pd.DataFrame(rows)


def rule_metrics(dev: pd.DataFrame, test: pd.DataFrame, fixed: Mapping[str, float],
                 targets: Sequence[float]) -> pd.DataFrame:
    """Test metrics under each threshold rule; ``fixed`` gives the rules set elsewhere (A and B)."""
    bona = (dev["class"] == "bona_fide").to_numpy()
    thresholds = dict(fixed)
    for t in targets:
        thresholds[bpcer_rule(t)] = select_bpcer_threshold(dev["score"].to_numpy(), bona, t)[0].threshold
    rows = []
    for rule, thr in thresholds.items():
        m = pad_metrics(test, thr)
        rows.append({"rule": rule, "threshold": thr,
                     "dev_bpcer_errors": int((dev.loc[bona, "score"] < thr).sum()), "dev_bona_fide": int(bona.sum()),
                     "print_apcer_errors": m["print_apcer"]["errors"], "print_apcer_pct": m["print_apcer"]["percent"],
                     "replay_apcer_errors": m["replay_apcer"]["errors"], "replay_apcer_pct": m["replay_apcer"]["percent"],
                     "bpcer_errors": m["bpcer"]["errors"], "bpcer_pct": m["bpcer"]["percent"],
                     "acer_pct": 100 * m["acer"], "auroc": m["auroc"]})
    return pd.DataFrame(rows)


def read_videos(result_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(Path(result_dir) / f"{split}_video_scores.csv", dtype={"video_id": str})


def engine_rule_metrics(result_dir: Path, targets: Sequence[float]) -> pd.DataFrame:
    """All threshold rules for one INT8 result directory written by ``evaluate_backend``."""
    m = json.loads((Path(result_dir) / "metrics.json").read_text())
    fixed = {"A_fp32": m["threshold_A_fp32"], "B_dev_eer": m["threshold_B_engine"]["threshold"]}
    return rule_metrics(read_videos(result_dir, "dev"), read_videos(result_dir, "test"), fixed, targets)


def collect(result_dirs: Mapping[tuple[str, int], Path], targets: Sequence[float]) -> pd.DataFrame:
    """Long table: one row per (setting, seed, rule)."""
    parts = [engine_rule_metrics(d, targets).assign(setting=s, seed=seed) for (s, seed), d in result_dirs.items()]
    out = pd.concat(parts, ignore_index=True)
    return out[["setting", "seed", *[c for c in out.columns if c not in ("setting", "seed")]]]


def setting_ranges(long: pd.DataFrame, baseline: str, metrics: Sequence[str] = RANGE_METRICS) -> pd.DataFrame:
    """Min–max over seeds per (rule, setting, metric), and whether the range is clear of the baseline's.

    ``separated`` means the two seed ranges do not overlap, the criterion suggested for a likely real
    difference; with three seeds it is a screening rule, not a significance test.
    """
    g = long.melt(id_vars=["setting", "seed", "rule"], value_vars=list(metrics), var_name="metric")
    r = g.groupby(["rule", "setting", "metric"])["value"].agg(["min", "max", "mean", "count"]).reset_index()
    r = r.rename(columns={"count": "seeds"})
    base = r[r["setting"] == baseline][["rule", "metric", "min", "max"]].rename(
        columns={"min": "baseline_min", "max": "baseline_max"})
    r = r.merge(base, on=["rule", "metric"], how="left")
    above, below = r["min"] > r["baseline_max"], r["max"] < r["baseline_min"]
    r["separated"] = above | below
    r["direction"] = np.select([above, below], ["higher", "lower"], "overlaps")
    return r


def range_text(ranges: pd.DataFrame, rules: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Readable pivot: ``min–max`` per setting, with ``*`` when clear of the baseline range."""
    r = ranges[ranges["rule"].isin(rules) & ranges["metric"].isin(metrics)].copy()
    r["text"] = [f"{a:.2f}–{b:.2f}" + ("*" if s else "") for a, b, s in zip(r["min"], r["max"], r["separated"])]
    return r.pivot_table(index="setting", columns=["rule", "metric"], values="text", aggfunc="first")


def score_shifts_vs_baseline(result_dirs: Mapping[tuple[str, int], Path], baseline: str) -> pd.DataFrame:
    """Threshold-free check: test score change of each setting against the baseline of the same seed."""
    from .diagnostics import paired_score_shift

    rows = []
    for (s, seed), d in result_dirs.items():
        ref = result_dirs.get((baseline, seed))
        if s == baseline or ref is None:
            continue
        rows.append(paired_score_shift(read_videos(ref, "test"), read_videos(d, "test")).assign(setting=s, seed=seed))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[["setting", "seed", *[c for c in out.columns if c not in ("setting", "seed")]]]


def fig_ablation_line(long: pd.DataFrame, table: pd.DataFrame, line: str, levels: Sequence[str], rules: Sequence[str],
                      out_dir: Path, name: str) -> dict[str, str]:
    """Per-seed values and min–max range per setting along one ablation line, one panel per error rate."""
    from .plotting import finalize, save_figure, _plt

    plt = _plt()
    info = table.set_index("setting")
    labels = [f"{info.loc[s, 'attack_pct']:.0f}%" if line == "attack_share" else f"{info.loc[s, 'images']}"
              for s in levels]
    xlabel = "Attack images in the calibration set" if line == "attack_share" else "Calibration images"
    panels = (("print_apcer_pct", "Print APCER (%)"), ("replay_apcer_pct", "Replay APCER (%)"), ("bpcer_pct", "BPCER (%)"))
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    offsets = np.linspace(-0.24, 0.24, len(rules)) if len(rules) > 1 else [0.0]
    for ax, (metric, ylabel) in zip(axes, panels):
        for off, rule in zip(offsets, rules):
            for i, s in enumerate(levels):
                v = long[(long["setting"] == s) & (long["rule"] == rule)][metric].to_numpy()
                if not len(v):
                    continue
                ax.vlines(i + off, v.min(), v.max(), color=RULE_COLORS.get(rule, "#52514e"), linewidth=2)
                ax.scatter(np.full(len(v), i + off), v, s=14, color=RULE_COLORS.get(rule, "#52514e"),
                           label=RULE_LABELS.get(rule, rule) if i == 0 else None, zorder=3)
        ax.set_xticks(range(len(levels)), labels)
        finalize(ax, ylabel.replace(" (%)", ""), xlabel, ylabel, legend=metric == panels[0][0])
    return save_figure(fig, out_dir, name,
                       f"Test error rates of the INT8 engines along the {line.replace('_', ' ')} line; each dot is one "
                       "seed, the bar its min–max range, colors are threshold rules set on the development set.")
