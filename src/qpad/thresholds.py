"""Development-set EER threshold selection with a pre-registered tie-breaking rule.

Decision rule everywhere: ``score >= threshold`` -> bona fide, else attack.

Rule:
1. candidates = every distinct development score, plus one value above the maximum
   (rejects everything);
2. minimise ``|APCER - BPCER|``;
3. ties -> lowest ACER;
4. still tied -> highest threshold (more secure).

Comparisons are done with exact integer arithmetic on error counts, so ties are
never decided by floating-point noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    apcer_errors: int
    n_attacks: int
    bpcer_errors: int
    n_bona_fide: int

    @property
    def apcer(self) -> float:
        return self.apcer_errors / self.n_attacks

    @property
    def bpcer(self) -> float:
        return self.bpcer_errors / self.n_bona_fide

    @property
    def acer(self) -> float:
        return (self.apcer + self.bpcer) / 2

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(apcer=self.apcer, bpcer=self.bpcer, acer=self.acer)
        return d


def predict_bona_fide(scores: np.ndarray, threshold: float) -> np.ndarray:
    """The one decision rule: ``score >= threshold`` means bona fide."""
    return np.asarray(scores, dtype=np.float64) >= float(threshold)


def threshold_table(scores: np.ndarray, is_bona_fide: np.ndarray) -> pd.DataFrame:
    """APCER/BPCER counts at every candidate threshold (ascending)."""
    s = np.asarray(scores, dtype=np.float64)
    bona = np.asarray(is_bona_fide, dtype=bool)
    if bona.all() or (~bona).all():
        raise ValueError("threshold selection needs both bona fide and attack scores")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN/inf")
    uniq = np.unique(s)
    cand = np.append(uniq, np.nextafter(uniq[-1], np.inf))
    att_sorted = np.sort(s[~bona])
    bon_sorted = np.sort(s[bona])
    n_a, n_b = len(att_sorted), len(bon_sorted)
    # attacks accepted: score >= t ; bona fide rejected: score < t
    att_accepted = n_a - np.searchsorted(att_sorted, cand, side="left")
    bon_rejected = np.searchsorted(bon_sorted, cand, side="left")
    t = pd.DataFrame({"threshold": cand, "apcer_errors": att_accepted, "n_attacks": n_a,
                      "bpcer_errors": bon_rejected, "n_bona_fide": n_b})
    t["apcer"] = t["apcer_errors"] / n_a
    t["bpcer"] = t["bpcer_errors"] / n_b
    t["acer"] = (t["apcer"] + t["bpcer"]) / 2
    # Exact integer keys: |a/A - b/B| * A*B and (a/A + b/B) * A*B
    t["abs_diff_key"] = np.abs(t["apcer_errors"].astype(np.int64) * n_b - t["bpcer_errors"].astype(np.int64) * n_a)
    t["acer_key"] = t["apcer_errors"].astype(np.int64) * n_b + t["bpcer_errors"].astype(np.int64) * n_a
    return t


def select_eer_threshold(scores: np.ndarray, is_bona_fide: np.ndarray) -> tuple[ThresholdResult, pd.DataFrame]:
    """Apply the pre-registered rule; returns the chosen operating point and the full table."""
    t = threshold_table(scores, is_bona_fide)
    order = t.sort_values(["abs_diff_key", "acer_key", "threshold"], ascending=[True, True, False], kind="mergesort")
    best = order.iloc[0]
    t["selected"] = t.index == best.name
    res = ThresholdResult(float(best["threshold"]), int(best["apcer_errors"]), int(best["n_attacks"]),
                          int(best["bpcer_errors"]), int(best["n_bona_fide"]))
    return res, t
