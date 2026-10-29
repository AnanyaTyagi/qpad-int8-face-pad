"""Report figures. Every figure gets a title, labeled axes, a legend when needed,
a one-sentence caption, and is saved as 300-dpi PNG + vector PDF.

Colors: categorical slots 1-3 of the validated reference palette (blue, orange,
aqua), which pass the all-pairs color-vision-deficiency checks for three series.
Identity is never color-only: legends are always present for >= 2 series.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

CLASS_COLORS = {"bona_fide": "#2a78d6", "print": "#eb6834", "replay": "#1baf7a"}
CLASS_LABELS = {"bona_fide": "Bona fide", "print": "Print attack", "replay": "Replay attack"}
NEUTRAL = "#52514e"
TEXT = "#0b0b0b"
GRID = "#e4e3df"
SPLIT_ORDER = ("train", "dev", "test")
SPLIT_LABELS = {"train": "Train", "dev": "Development", "test": "Test"}


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": NEUTRAL, "axes.labelcolor": TEXT, "xtick.color": NEUTRAL, "ytick.color": NEUTRAL,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 2.0, "pdf.fonttype": 42,
    })
    return plt


def finalize(ax, title: str, xlabel: str, ylabel: str, legend: bool) -> None:
    """Enforce the figure contract (title + axis labels + optional legend)."""
    if not (title and xlabel and ylabel):
        raise ValueError("every figure needs a title and both axis labels")
    ax.set_title(title, loc="left", color=TEXT)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()


def save_figure(fig, out_dir: Path, name: str, caption: str) -> dict[str, str]:
    """Save PNG + PDF and record the caption in ``captions.json`` next to them."""
    if not caption.strip():
        raise ValueError("every figure needs a caption sentence")
    out_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = out_dir / f"{name}.png", out_dir / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    cap_path = out_dir / "captions.json"
    caps = json.loads(cap_path.read_text()) if cap_path.is_file() else {}
    caps[name] = caption
    cap_path.write_text(json.dumps(caps, indent=2, sort_keys=True))
    import matplotlib.pyplot as plt

    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf), "caption": caption}


def _grouped_bars(ax, table: pd.DataFrame, groups: Sequence[str], group_labels: Mapping[str, str] | None = None,
                  annotate: bool = True) -> None:
    """``table``: index = x categories, columns = classes (counts)."""
    classes = [c for c in CLASS_COLORS if c in table.columns]
    x = np.arange(len(groups))
    width = 0.8 / len(classes)
    for i, c in enumerate(classes):
        vals = table.reindex(groups)[c].fillna(0).to_numpy()
        bars = ax.bar(x + (i - (len(classes) - 1) / 2) * width, vals, width * 0.92, color=CLASS_COLORS[c],
                      label=CLASS_LABELS[c])
        if annotate:
            ax.bar_label(bars, labels=[f"{int(v)}" for v in vals], padding=2, fontsize=7, color=TEXT)
    ax.set_xticks(x, [group_labels.get(g, str(g)) if group_labels else str(g) for g in groups])


# ---------------------------------------------------------------------------------------
# EDA figures
# ---------------------------------------------------------------------------------------
def fig_class_balance(manifest: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    t = manifest.groupby(["split", "class"]).size().unstack(fill_value=0)
    _grouped_bars(ax, t, list(SPLIT_ORDER), SPLIT_LABELS)
    finalize(ax, "Protocol II videos per split and class", "Split", "Number of videos", legend=True)
    balanced = bool((t.nunique(axis=1) == 1).all())
    n_att = int(t.drop(columns="bona_fide", errors="ignore").to_numpy().sum())
    n_bf = int(t.get("bona_fide", pd.Series(dtype=int)).sum())
    caption = (("Every split has equal bona fide, print and replay counts" if balanced
                else "Class counts differ within at least one split")
               + f"; attacks outnumber bona fide videos {n_att}:{n_bf}, so APCER and BPCER must be reported "
               "separately rather than as a single accuracy.")
    return save_figure(fig, out_dir, "class_balance_by_split", caption)


def fig_subjects_per_split(manifest: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    n = manifest.groupby("split")["subject"].nunique().reindex(list(SPLIT_ORDER)).fillna(0)
    bars = ax.bar([SPLIT_LABELS[s] for s in SPLIT_ORDER], n.to_numpy(), 0.6, color=CLASS_COLORS["bona_fide"])
    ax.bar_label(bars, labels=[f"{int(v)}" for v in n], padding=2, fontsize=8)
    finalize(ax, "Distinct subjects per split", "Split", "Number of subjects", legend=False)
    disjoint = bool((manifest.groupby("subject")["split"].nunique() <= 1).all())
    caption = ("No subject appears in more than one split, so test performance measures generalization to unseen "
               "people." if disjoint else "WARNING: at least one subject appears in several splits (leakage).")
    return save_figure(fig, out_dir, "subjects_per_split", caption)


def fig_by_factor(manifest: pd.DataFrame, factor: str, label: str, out_dir: Path) -> dict[str, str]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    t = manifest.groupby([factor, "class"]).size().unstack(fill_value=0)
    _grouped_bars(ax, t, list(t.index), None, annotate=False)
    finalize(ax, f"Protocol II videos per {label.lower()} and class", label, "Number of videos", legend=True)
    even = bool(t.to_numpy().min() == t.to_numpy().max())
    caption = (f"Every {label.lower()} contributes the same number of videos per class, so no single "
               f"{label.lower()} dominates the data." if even else
               f"Video counts vary across {label.lower()}s (min {int(t.to_numpy().min())}, max "
               f"{int(t.to_numpy().max())} per {label.lower()} and class).")
    return save_figure(fig, out_dir, f"{factor}_distribution", caption)


def fig_hist_by_class(df: pd.DataFrame, col: str, xlabel: str, title: str, name: str, caption: str,
                      out_dir: Path, bins: int = 30, log_x: bool = False) -> dict[str, str]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    vals = df[col].dropna().astype(float)
    if log_x:
        vals = vals[vals > 0]
        edges = np.logspace(np.log10(vals.min()), np.log10(vals.max()), bins + 1) if len(vals) else bins
        ax.set_xscale("log")
    else:
        edges = np.histogram_bin_edges(vals, bins=bins) if len(vals) else bins
    for c in CLASS_COLORS:
        v = df.loc[df["class"] == c, col].dropna().astype(float)
        if log_x:
            v = v[v > 0]
        if len(v):
            ax.hist(v, bins=edges, histtype="step", linewidth=2, color=CLASS_COLORS[c], label=CLASS_LABELS[c])
    finalize(ax, title, xlabel, "Count", legend=True)
    return save_figure(fig, out_dir, name, caption)


def fig_box_by_class(df: pd.DataFrame, col: str, ylabel: str, title: str, name: str, caption: str,
                     out_dir: Path, log_y: bool = False) -> dict[str, str]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    classes = [c for c in CLASS_COLORS if (df["class"] == c).any()]
    data = [df.loc[df["class"] == c, col].dropna().astype(float).to_numpy() for c in classes]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False)
    ax.set_xticks(range(1, len(classes) + 1), [CLASS_LABELS[c] for c in classes])
    for patch, c in zip(bp["boxes"], classes):
        patch.set_facecolor(CLASS_COLORS[c])
        patch.set_alpha(0.85)
        patch.set_edgecolor(NEUTRAL)
    for med in bp["medians"]:
        med.set_color(TEXT)
    if log_y:
        ax.set_yscale("log")
    finalize(ax, title, "Class", ylabel, legend=False)  # classes are named on the x-axis
    return save_figure(fig, out_dir, name, caption)


def fig_scatter_by_class(df: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, title: str, name: str,
                         caption: str, out_dir: Path, log_y: bool = False) -> dict[str, str]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for c in CLASS_COLORS:
        d = df[df["class"] == c]
        if len(d):
            ax.scatter(d[x], d[y], s=14, alpha=0.6, color=CLASS_COLORS[c], edgecolors="white", linewidths=0.4,
                       label=CLASS_LABELS[c])
    if log_y:
        ax.set_yscale("log")
    finalize(ax, title, xlabel, ylabel, legend=True)
    return save_figure(fig, out_dir, name, caption)


def fig_failures(fail_counts: Mapping[str, int], out_dir: Path) -> dict[str, str]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    names = list(fail_counts)
    vals = [int(fail_counts[k]) for k in names]
    bars = ax.barh(names, vals, 0.6, color=NEUTRAL)
    ax.bar_label(bars, labels=[str(v) for v in vals], padding=2, fontsize=8)
    from matplotlib.ticker import MaxNLocator

    ax.set_xlim(0, max([1, *vals]) * 1.15)            # counts are never negative, even when all are zero
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    finalize(ax, "Missing or failed items by check", "Number of items", "Check", legend=False)
    caption = ("Counts of items that failed each data-quality check; zero bars confirm the check passed for every "
               "Protocol II video." if not any(vals) else
               "Counts of items that failed each data-quality check; non-zero bars are the documented data exceptions "
               "listed in the audit report.")
    return save_figure(fig, out_dir, "missing_failed_items", caption)


# ---------------------------------------------------------------------------------------
# Training / evaluation figures
# ---------------------------------------------------------------------------------------
def fig_training_curves(history: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(history["epoch"], history["train_loss"], color=CLASS_COLORS["bona_fide"], label="Train loss", marker="o",
            markersize=4)
    ax.plot(history["epoch"], history["dev_loss"], color=CLASS_COLORS["print"], label="Development loss", marker="o",
            markersize=4)
    best = history.loc[history["dev_loss"].idxmin()]
    ax.axvline(best["epoch"], color=NEUTRAL, linewidth=1, linestyle="--")
    ax.annotate(f"selected epoch {int(best['epoch'])}", (best["epoch"], best["dev_loss"]), xytext=(6, 10),
                textcoords="offset points", fontsize=8, color=TEXT)
    finalize(ax, "DeepPixBiS training and development loss", "Epoch", "Loss (0.5 pixel BCE + 0.5 binary BCE)",
             legend=True)
    return save_figure(fig, out_dir, "training_curves",
                       "The checkpoint with the lowest development loss is selected; the test split is never used "
                       "during training or model selection.")


def fig_threshold_curve(table: pd.DataFrame, chosen: float, out_dir: Path, name: str = "dev_threshold_curve"
                        ) -> dict[str, str]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(table["threshold"], 100 * table["apcer"], color=CLASS_COLORS["print"], label="APCER (attacks accepted)")
    ax.plot(table["threshold"], 100 * table["bpcer"], color=CLASS_COLORS["bona_fide"],
            label="BPCER (bona fide rejected)")
    ax.axvline(chosen, color=NEUTRAL, linewidth=1, linestyle="--")
    ax.annotate(f"threshold = {chosen:.4f}", (chosen, 50), xytext=(6, 0), textcoords="offset points", fontsize=8)
    finalize(ax, "Development-set error rates versus threshold", "Threshold on video score", "Error rate (%)",
             legend=True)
    return save_figure(fig, out_dir, name,
                       "The operating threshold is the development-set point where APCER and BPCER are closest "
                       "(ties: lower ACER, then higher threshold) and is frozen before any test evaluation.")


def fig_score_hist(videos: pd.DataFrame, threshold: float, split_label: str, out_dir: Path, name: str
                   ) -> dict[str, str]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    edges = np.linspace(0, 1, 41)
    for c in CLASS_COLORS:
        v = videos.loc[videos["class"] == c, "score"]
        if len(v):
            ax.hist(v, bins=edges, histtype="step", linewidth=2, color=CLASS_COLORS[c], label=CLASS_LABELS[c])
    ax.axvline(threshold, color=NEUTRAL, linewidth=1, linestyle="--", label="Frozen threshold")
    finalize(ax, f"{split_label} video scores by class", "Video score (mean pixel-map value)", "Number of videos",
             legend=True)
    return save_figure(fig, out_dir, name,
                       f"Distribution of {split_label.lower()} video scores; attacks to the right of the dashed "
                       "threshold are accepted as bona fide (APCER errors).")


def fig_roc(videos: pd.DataFrame, split_label: str, out_dir: Path, name: str) -> dict[str, str]:
    from .thresholds import threshold_table

    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    bona = (videos["class"] == "bona_fide").to_numpy()
    for c in ("print", "replay"):
        m = bona | (videos["class"] == c).to_numpy()
        t = threshold_table(videos["score"].to_numpy()[m], bona[m])
        ax.plot(100 * t["apcer"], 100 * (1 - t["bpcer"]), color=CLASS_COLORS[c], label=CLASS_LABELS[c])
    ax.plot([0, 100], [0, 100], color=GRID, linewidth=1)
    finalize(ax, f"{split_label} ROC per attack type", "APCER (%)", "100 - BPCER (%)", legend=True)
    return save_figure(fig, out_dir, name,
                       "ROC curves separate print and replay attacks from bona fide presentations; curves closer to "
                       "the top-left corner indicate better separation.")
