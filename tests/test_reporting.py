import json

from qpad.reporting import build_report_assets, rate_tex, tex_escape


def test_placeholders_when_nothing_measured(tmp_path):
    w = build_report_assets(tmp_path / "results", tmp_path / "reports", tmp_path / "report", {"commit": None})
    tex = (tmp_path / "report" / "tables" / "generated_values.tex").read_text()
    assert "INSERT MEASURED RESULT FROM metrics.json" in tex and "\\QPADTestPrintAPCER" in tex
    assert "QPADTestPrintAPCER" in w["placeholders"] and not w["figures"]


def test_values_from_metrics(tmp_path):
    rate = lambda e, t: {"errors": e, "total": t, "rate": e / t, "percent": 100 * e / t}  # noqa: E731
    m = {k: rate(36, 360) for k in ("print_apcer", "replay_apcer", "apcer_pooled", "apcer_max_pai")}
    m.update(bpcer=rate(2, 360), acer=0.055, acer_max_pai=0.0528, auroc=0.97, eer={"value": 0.04})
    metrics = {"dev": m, "test": m, "threshold": {"threshold": 0.5, "apcer_errors": 3, "n_attacks": 540,
                                                 "bpcer_errors": 4, "n_bona_fide": 270}}
    (tmp_path / "results" / "fp32").mkdir(parents=True)
    (tmp_path / "results" / "fp32" / "metrics.json").write_text(json.dumps(metrics))
    build_report_assets(tmp_path / "results", tmp_path / "reports", tmp_path / "report", {"commit": "abcdef1234567"})
    tex = (tmp_path / "report" / "tables" / "generated_values.tex").read_text()
    assert "\\newcommand{\\QPADTestReplayAPCER}{36/360 = 10.00\\%}" in tex
    assert "abcdef123456" in tex
    assert (tmp_path / "report" / "tables" / "fp32_results.tex").is_file()


def test_tex_helpers():
    assert tex_escape("a_b & 5%") == "a\\_b \\& 5\\%"
    assert rate_tex({"errors": 1, "total": 4, "percent": 25.0}) == "1/4 = 25.00\\%"


def test_stale_tables_and_figures_are_removed(tmp_path):
    report = tmp_path / "report"
    for p in (report / "tables" / "fp32_results.tex", report / "figures" / "old.pdf"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("measured earlier")
    build_report_assets(tmp_path / "results", tmp_path / "reports", report, {"commit": None})
    assert not (report / "tables" / "fp32_results.tex").exists()
    assert not (report / "figures" / "old.pdf").exists()


def test_breakdown_table_keeps_integer_factor(tmp_path):
    import pandas as pd

    from qpad.reporting import breakdown_table

    csv = tmp_path / "by_phone.csv"
    pd.DataFrame({"phone": [2], **{f"{k}_{s}": [1] for k in ("print_apcer", "replay_apcer", "bpcer")
                                   for s in ("errors", "total")}}).to_csv(csv, index=False)
    assert "\n2 & 1/1" in breakdown_table(csv, "phone")
