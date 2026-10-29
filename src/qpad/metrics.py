"""PAD metrics (ISO/IEC 30107-3 terminology) reported with counts.

* APCER: proportion of attack presentations classified as bona fide.
* BPCER: proportion of bona fide presentations classified as attacks.
* ACER (project definition): (APCER_pooled + BPCER) / 2.
* ACER_max_pai: (max over PAI species of APCER + BPCER) / 2 — the OULU-NPU
  benchmark convention, reported for comparability with published numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .thresholds import predict_bona_fide, select_eer_threshold

ATTACK_TYPES = ("print", "replay")


@dataclass(frozen=True)
class Rate:
    errors: int
    total: int

    @property
    def value(self) -> float:
        return self.errors / self.total if self.total else float("nan")

    def __str__(self) -> str:
        return f"{self.errors}/{self.total} = {100 * self.value:.2f}%"

    def as_dict(self) -> dict[str, Any]:
        return {"errors": self.errors, "total": self.total, "rate": self.value, "percent": 100 * self.value,
                "text": str(self)}


def auroc(scores: np.ndarray, is_positive: np.ndarray) -> float:
    """Area under the ROC curve (bona fide = positive) via the Mann–Whitney U statistic, ties = 0.5."""
    s = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(is_positive, dtype=bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def pad_metrics(videos: pd.DataFrame, threshold: float, score_col: str = "score") -> dict[str, Any]:
    """All headline metrics at ``threshold`` plus threshold-free EER and AUROC.

    ``videos`` needs columns ``class`` (bona_fide/print/replay) and ``score_col``.
    """
    s = videos[score_col].to_numpy(dtype=np.float64)
    cls = videos["class"].to_numpy()
    accepted = predict_bona_fide(s, threshold)
    bona = cls == "bona_fide"
    out: dict[str, Any] = {"threshold": float(threshold), "n_videos": int(len(videos))}
    per_pai = {}
    for a in ATTACK_TYPES:
        m = cls == a
        per_pai[a] = Rate(int(accepted[m].sum()), int(m.sum()))
        out[f"{a}_apcer"] = per_pai[a].as_dict()
    attacks = ~bona
    apcer = Rate(int(accepted[attacks].sum()), int(attacks.sum()))
    bpcer = Rate(int((~accepted[bona]).sum()), int(bona.sum()))
    worst = max(per_pai.values(), key=lambda r: (r.value, r.errors))
    out["apcer_pooled"] = apcer.as_dict()
    out["apcer_max_pai"] = worst.as_dict()
    out["bpcer"] = bpcer.as_dict()
    out["acer"] = (apcer.value + bpcer.value) / 2
    out["acer_max_pai"] = (worst.value + bpcer.value) / 2
    out["confusion"] = {
        "bona_fide_accepted": int(accepted[bona].sum()), "bona_fide_rejected": int((~accepted[bona]).sum()),
        "print_accepted": int(accepted[cls == "print"].sum()), "print_rejected": int((~accepted[cls == "print"]).sum()),
        "replay_accepted": int(accepted[cls == "replay"].sum()),
        "replay_rejected": int((~accepted[cls == "replay"]).sum()),
    }
    out["auroc"] = auroc(s, bona)
    eer_res, _ = select_eer_threshold(s, bona)
    out["eer"] = {"value": eer_res.acer, "apcer_at_eer": eer_res.apcer, "bpcer_at_eer": eer_res.bpcer,
                  "threshold_on_this_set": eer_res.threshold,
                  "note": "EER computed on this split's own scores (descriptive; not used for any decision)"}
    return out


def error_breakdown(videos: pd.DataFrame, threshold: float, by: Sequence[str] = ("phone",),
                    score_col: str = "score") -> pd.DataFrame:
    """APCER per attack type and BPCER, with counts, for each group of ``by``."""
    v = videos.copy()
    v["accepted"] = predict_bona_fide(v[score_col].to_numpy(), threshold)
    rows = []
    for key, g in v.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        for c in ("bona_fide",) + ATTACK_TYPES:
            m = g["class"] == c
            n = int(m.sum())
            err = int((~g.loc[m, "accepted"]).sum()) if c == "bona_fide" else int(g.loc[m, "accepted"].sum())
            name = "bpcer" if c == "bona_fide" else f"{c}_apcer"
            row[f"{name}_errors"], row[f"{name}_total"] = err, n
            row[name] = err / n if n else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def decision_flips(ref: pd.DataFrame, other: pd.DataFrame, ref_threshold: float, other_threshold: float,
                   score_col: str = "score") -> dict[str, int]:
    """Paired decision changes between two models on the same videos."""
    m = ref[["video_id", "class", score_col]].merge(other[["video_id", score_col]], on="video_id",
                                                    suffixes=("_ref", "_other"), validate="one_to_one")
    if len(m) != len(ref) or len(m) != len(other):
        raise ValueError("models were not scored on the same videos")
    a = predict_bona_fide(m[f"{score_col}_ref"].to_numpy(), ref_threshold)
    b = predict_bona_fide(m[f"{score_col}_other"].to_numpy(), other_threshold)
    attack = (m["class"] != "bona_fide").to_numpy()
    return {
        "attack_rejected_to_accepted": int((attack & ~a & b).sum()),
        "attack_accepted_to_rejected": int((attack & a & ~b).sum()),
        "bona_fide_accepted_to_rejected": int((~attack & a & ~b).sum()),
        "bona_fide_rejected_to_accepted": int((~attack & ~a & b).sum()),
        "n_videos": int(len(m)),
    }


def metrics_rows(name: str, m: dict[str, Any]) -> dict[str, Any]:
    """Flatten a :func:`pad_metrics` dict into one table row (for CSV / LaTeX)."""
    return {
        "model": name, "threshold": m["threshold"],
        "print_apcer": m["print_apcer"]["text"], "replay_apcer": m["replay_apcer"]["text"],
        "apcer_pooled": m["apcer_pooled"]["text"], "bpcer": m["bpcer"]["text"],
        "acer_pct": 100 * m["acer"], "acer_max_pai_pct": 100 * m["acer_max_pai"],
        "eer_pct": 100 * m["eer"]["value"], "auroc": m["auroc"],
    }
