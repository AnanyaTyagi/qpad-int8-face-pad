# Checkpoint 1 report — how to build and verify

## Build
1. Run notebooks 00–05 with `DRY_RUN = False` (Colab), then regenerate assets:
   `python scripts/build_report_assets.py --project-root "/content/drive/MyDrive/QPAD_Project"`
   (notebook 05 does this automatically after the test evaluation).
2. Upload `reports/checkpoint1/` to Overleaf (it uses `IEEEtran`, shipped with Overleaf) and compile with
   pdfLaTeX + BibTeX. Figures/tables that do not exist yet render as red `[INSERT …]` boxes.

## What is generated vs written
* `tables/generated_values.tex` — every number in the text (macros such as `\QPADTestPrintAPCER`).
  Values not yet measured expand to **[INSERT MEASURED RESULT FROM …]**. Never edit by hand.
* `tables/*.tex` — counts, FP32 results, error breakdowns (from CSV/JSON results).
* `figures/*.pdf` — aggregate plots only (no faces); captions in `figures/captions.json`.
* Prose, including red `\todo{…}` sentences, is written and approved by the students.

## Before submission (students must verify)
- [ ] Every `[INSERT …]` / red `\todo{…}` is resolved from a measured file (none left in the PDF).
- [x] Data access: provided by the course instructor on request (no team-signed EULA); report says so.
- [ ] Every bibliography entry checked against its DOI/publisher page.
- [x] Published DeepPixBiS Protocol II numbers verified in the paper (arXiv:1907.04047, Table 2: 11.39 / 0.56 / 5.97).
- [ ] Scope changes (blue `\scope{…}`) match what changed since the proposal.
- [ ] `\QPADGitCommit` is the commit that produced the results (commit before running; rebuild assets after).
- [ ] 3–4 pages excluding cover block and references.
- [ ] Deadline confirmed with the professor (PDF says "Thursday, October 27, 2026"; that date is a Tuesday).
