# INT8 calibration ablation (notebooks 08, 08b)

Colab L4, TensorRT 10.16.1.11, run of 2026-11-12. The pilot (07) ran on an A100; the 24 ablation engines are
compared with each other only. Natural s1/s2 were also checked against the A100 pilot engines (08b section 5).

- `08_int8_ablation_executed.ipynb`: builds and scores the 24 engines (8 settings x seeds 1-3); policy A per engine
  and the calibration check (every engine received its full set).
- `08b_int8_ablation_analysis_executed.ipynb`: threshold rules A (fixed FP32 0.7317), B (engine dev EER),
  C10/C15 (dev BPCER 10%/15%); seed ranges against natural, score shifts, pilot comparison.
- `results/`: copy of `QPAD_Project/results/int8/ablation/` from Drive, without the per-frame score files.
  - `int8_entropy_<setting>_s<seed>/`: `dev_video_scores.csv`, `test_video_scores.csv` (one row per video),
    `dev_threshold_candidates.csv`, `metrics.json`.
  - `calibration_manifest.csv` (video IDs and frame indices used for calibration), `engines.csv`.
  - `analysis/`: `ablation_metrics.csv`, `ablation_ranges.csv`, `ablation_score_shifts.csv`,
    `pilot_vs_ablation_natural.csv` and the two figures.

No images, videos or model files are included.
