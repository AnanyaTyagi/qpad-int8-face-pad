"""Generate report-ready LaTeX tables and value macros from saved result files.

Numbers in the report are never typed by hand: ``build_report_assets`` reads
``metrics.json`` / ``threshold.json`` / EDA CSVs and writes
``tables/generated_values.tex`` (macros) and table snippets. Any value whose
source file does not exist yet becomes an explicit ``[INSERT ...]`` placeholder.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

def placeholder(src: str) -> str:
    """Explicit marker for a value that has not been measured yet."""
    return r"\textbf{[INSERT MEASURED RESULT FROM " + tex_escape(src) + "]}"


def tex_escape(s: Any) -> str:
    out = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        out = out.replace(a, b)
    return out


def rate_tex(r: Mapping[str, Any]) -> str:
    return f"{r['errors']}/{r['total']} = {r['percent']:.2f}\\%"


def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def fp32_macros(metrics: Mapping[str, Any] | None) -> dict[str, str]:
    keys = {"PrintAPCER": "print_apcer", "ReplayAPCER": "replay_apcer", "APCER": "apcer_pooled",
            "APCERmax": "apcer_max_pai", "BPCER": "bpcer"}
    out: dict[str, str] = {}
    for split in ("Dev", "Test"):
        m = metrics[split.lower()] if metrics else None
        for mac, key in keys.items():
            out[f"QPAD{split}{mac}"] = rate_tex(m[key]) if m else placeholder("metrics.json")
        out[f"QPAD{split}ACER"] = f"{100 * m['acer']:.2f}\\%" if m else placeholder("metrics.json")
        out[f"QPAD{split}ACERmax"] = f"{100 * m['acer_max_pai']:.2f}\\%" if m else placeholder("metrics.json")
        out[f"QPAD{split}EER"] = f"{100 * m['eer']['value']:.2f}\\%" if m else placeholder("metrics.json")
        out[f"QPAD{split}AUROC"] = f"{m['auroc']:.4f}" if m else placeholder("metrics.json")
    thr = metrics["threshold"] if metrics else None
    out["QPADThreshold"] = f"{thr['threshold']:.6f}" if thr else placeholder("threshold.json")
    return out


def fp32_table(metrics: Mapping[str, Any]) -> str:
    rows = [("Print APCER", "print_apcer"), ("Replay APCER", "replay_apcer"), ("APCER (pooled)", "apcer_pooled"),
            ("APCER (worst PAI)", "apcer_max_pai"), ("BPCER", "bpcer")]
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Metric & Development & Test \\", r"\midrule"]
    for label, key in rows:
        lines.append(f"{label} & {rate_tex(metrics['dev'][key])} & {rate_tex(metrics['test'][key])} \\\\")
    for label, f in (("ACER (pooled)", lambda m: f"{100 * m['acer']:.2f}\\%"),
                     ("ACER (worst PAI)", lambda m: f"{100 * m['acer_max_pai']:.2f}\\%"),
                     ("EER (own scores)", lambda m: f"{100 * m['eer']['value']:.2f}\\%"),
                     ("AUROC", lambda m: f"{m['auroc']:.4f}")):
        lines.append(f"{label} & {f(metrics['dev'])} & {f(metrics['test'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def counts_table(class_counts_csv: Path) -> str:
    t = pd.read_csv(class_counts_csv).set_index("split").reindex(["train", "dev", "test"])
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Split & Bona fide & Print & Replay & Total \\", r"\midrule"]
    for s, r in t.iterrows():
        lines.append(f"{ {'train': 'Train', 'dev': 'Development', 'test': 'Test'}[s]} & {int(r['bona_fide'])} & "
                     f"{int(r['print'])} & {int(r['replay'])} & {int(r['total'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def breakdown_table(csv: Path, factor: str) -> str:
    t = pd.read_csv(csv)
    lines = [r"\begin{tabular}[t]{lrrr}", r"\toprule", f"{tex_escape(factor.title())} & Print APCER & Replay APCER & BPCER \\\\",
             r"\midrule"]
    for r in t.to_dict("records"):   # keeps integer columns as ints (iterrows upcasts rows to float)
        cells = [f"{int(r[f'{k}_errors'])}/{int(r[f'{k}_total'])}" for k in ("print_apcer", "replay_apcer", "bpcer")]
        lines.append(f"{r[factor]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_report_assets(results_root: Path, reports_root: Path, report_dir: Path, git: Mapping[str, Any]) -> dict:
    """Write macros/tables into ``report_dir/tables`` and copy figures into ``report_dir/figures``.

    ``results_root`` = ``<PROJECT>/results``; ``reports_root`` = ``<PROJECT>/reports``.
    Figures of aligned faces are never copied (only aggregate plots).
    """
    tables, figures = report_dir / "tables", report_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {"tables": [], "figures": [], "placeholders": []}
    # Remove everything this builder generates, so outputs whose sources disappeared are not reused.
    for t in ("fp32_results.tex", "errors_by_phone.tex", "errors_by_session.tex", "class_counts.tex"):
        (tables / t).unlink(missing_ok=True)
    for old in [*figures.glob("*.pdf"), figures / "captions.json"]:
        old.unlink(missing_ok=True)
    mpath = results_root / "fp32" / "metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.is_file() else None
    macros = fp32_macros(metrics)
    audit_path = reports_root / "dataset_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else None
    macros["QPADAuditStatus"] = audit["overall_status"] if audit else placeholder("dataset_audit.json")
    macros["QPADNumVideos"] = str(audit["n_protocol_videos"]) if audit else placeholder("dataset_audit.json")
    res = audit.get("resolution_counts") if audit else None
    macros["QPADResolutions"] = tex_escape(", ".join(f"{k} ({v})" for k, v in res.items())) if res else \
        placeholder("dataset_audit.json")
    def _stat(key: str, fmt: str) -> str:
        v = (audit or {}).get(key)
        return (fmt.format(**v) if v else placeholder("dataset_audit.json"))
    macros["QPADFrameCount"] = _stat("frame_count", "{min:.0f}--{max:.0f} (median {median:.0f})")
    macros["QPADDuration"] = _stat("duration_s", "{min:.2f}--{max:.2f}\\,s (median {median:.2f}\\,s)")
    macros["QPADFps"] = _stat("fps", "{median:.2f}")
    macros["QPADNumVideoMembers"] = str(audit["n_video_members_all_archives"]) if audit else placeholder("dataset_audit.json")
    macros["QPADNumFailed"] = (str(len(audit.get("videos_failing_checks", audit["failed_videos"]))) + " ("
                               + tex_escape(", ".join(v["video_id"] for v in audit.get("videos_failing_checks", []))) + ")"
                               if audit else placeholder("dataset_audit.json"))
    hist_path = results_root / "training_history.csv"
    if hist_path.is_file():
        hist = pd.read_csv(hist_path)
        best = hist.loc[hist["dev_loss"].idxmin()]   # first minimum = the epoch training kept
        macros["QPADEpochsRun"] = str(len(hist))
        macros["QPADBestEpoch"] = str(int(best["epoch"]))
        macros["QPADBestDevLoss"] = f"{best['dev_loss']:.4f}"
    else:
        for k in ("QPADEpochsRun", "QPADBestEpoch", "QPADBestDevLoss"):
            macros[k] = placeholder("training_history.csv")
    commit = git.get("commit")
    macros["QPADGitCommit"] = (tex_escape(commit[:12]) + (r"\textsuperscript{dirty}" if git.get("dirty") else "")) \
        if commit else placeholder("git rev-parse HEAD")
    if metrics:
        thr = metrics["threshold"]
        macros["QPADDevAPCERatThr"] = f"{thr['apcer_errors']}/{thr['n_attacks']}"
        macros["QPADDevBPCERatThr"] = f"{thr['bpcer_errors']}/{thr['n_bona_fide']}"
    else:
        macros["QPADDevAPCERatThr"] = macros["QPADDevBPCERatThr"] = placeholder("threshold.json")
    written["placeholders"] = [k for k, v in macros.items() if "INSERT" in v]
    (tables / "generated_values.tex").write_text(
        "% AUTO-GENERATED by qpad.reporting.build_report_assets — do not edit by hand.\n"
        + "\n".join(_macro(k, v) for k, v in sorted(macros.items())) + "\n")
    written["tables"].append("generated_values.tex")
    if metrics:
        (tables / "fp32_results.tex").write_text(fp32_table(metrics) + "\n")
        written["tables"].append("fp32_results.tex")
        for factor in ("phone", "session"):
            csv = results_root / "fp32" / f"test_errors_by_{factor}.csv"
            if csv.is_file():
                (tables / f"errors_by_{factor}.tex").write_text(breakdown_table(csv, factor) + "\n")
                written["tables"].append(f"errors_by_{factor}.tex")
    cc = results_root / "eda" / "class_counts.csv"
    if cc.is_file():
        (tables / "class_counts.tex").write_text(counts_table(cc) + "\n")
        written["tables"].append("class_counts.tex")
    for sub in ("eda", "fp32", ""):
        d = results_root / sub if sub else results_root
        for pdf in sorted(d.glob("*.pdf")):
            shutil.copyfile(pdf, figures / pdf.name)
            written["figures"].append(pdf.name)
        caps = d / "captions.json"
        if caps.is_file():
            all_caps = json.loads((figures / "captions.json").read_text()) if (figures / "captions.json").is_file() else {}
            all_caps.update(json.loads(caps.read_text()))
            (figures / "captions.json").write_text(json.dumps(all_caps, indent=2, sort_keys=True))
    return written
