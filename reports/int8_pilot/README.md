# INT8 pilot results (notebooks 07, 07b, 07c)

Colab A100, TensorRT 10.16.1.11, runs of 2026-10-28 (07) and 2026-10-30 (07b, 07c).

- `07b_pilot_diagnostics_saved_results_executed.ipynb`, `07c_pilot_diagnostics_gpu_executed.ipynb`: executed notebooks with all tables.
- `results/`: copy of `QPAD_Project/results/int8/` from Drive, without the per-frame score files.
  - `int8_entropy_<condition>_s<seed>/`: the six pilot engines. `test_video_scores.csv` and `dev_video_scores.csv`
    (one row per video: ID, class, subject, phone, session, attack code, score), `dev_threshold_candidates.csv`, `metrics.json`.
  - `pilot_summary.csv`, `pilot_scale_audit.csv`, `calibration_manifest.csv` (video IDs and frame indices used for calibration).
  - `diagnostics/`: 07b tables (margins, score shifts, decision flips, scale changes), 07c tables (layer precision,
    rebuild noise, control metrics) and the per-video results of the three rebuilds and two control engines.

No images, videos or model files are included.
