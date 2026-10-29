# QPAD — Calibration-Set Composition Bias in INT8 Face Presentation Attack Detection

> **Repository scope: Checkpoint 1.** This repository contains the Checkpoint 1 stage: notebooks 00–05
> (archive inventory, dataset audit, preprocessing, EDA, FP32 DeepPixBiS training, threshold selection and FP32
> evaluation), their code, tests and the Checkpoint 1 report. The INT8/TensorRT stages (notebooks 06–10) described
> below are developed separately and will be added for the final submission.

CS 228 Biometric Security with AI · San José State University · Prof. Amith Kamath Belman
Team: Ananya Tyagi, Vindhya

> **Research question.** Does the composition of the post-training INT8 calibration set cause
> attack-specific degradation (print or replay APCER) in face presentation attack detection, after every
> INT8 engine receives its own development-set threshold?

**In plain words:** we train one FP32 DeepPixBiS face anti-spoofing model, freeze it, and build many INT8
versions that differ only in *which training images* TensorRT looks at to choose its quantization ranges.
Then we check whether an unusual calibration mixture makes the INT8 model miss more print or replay attacks
than the INT8 model calibrated on the natural mixture.

---

## 1. Status: measured vs planned

| Stage | Status |
|---|---|
| Repository, notebooks 00–10, unit + synthetic end-to-end tests | **Done.** 115 tests pass locally (`pytest -q`). |
| Archive inventory on the real OULU-NPU archives | **Measured** on a local copy, 2026-10-16 (see `reports/` on the private Drive / local output folder). |
| Phase 1 audit on the real data | **Measured** 2026-10-16 on a local copy: status **PASS** with 3 documented exceptions (re-evaluated 2026-10-23), see below and `reports/dataset_audit.md` in the private output folder. |
| Preprocessing, EDA pixel statistics, FP32 training, threshold, test baseline | **Not yet run on real data.** Run notebooks 02–05 in Colab. |
| Export equivalence (06), INT8 pilot (07) | **Run 2026-10-28** on Colab A100, TensorRT 10.16.1.11; see `reports/lab_notes/`. |
| Full experiment (08), statistics (09), latency (10) | **Implemented, not run**; waiting for the Tier 2A/2B decision. |

**Real-data audit findings (2026-10-16, local copy, all 2,970 Protocol II videos inspected):**
all counts (360/360/360, 270/270/270, 360/360/360), subject ranges, leakage, duplicate and official
`Protocol_2` list checks **pass**; every video is 1080×1920 MJPEG AVI; annotations are 0-based with one row per
frame; frame counts range 41–151 (phone 2 and some phone 3 videos record at ~20 fps, 96 frames). Three videos
fail per-video checks and are handled by a documented rule (`data_exceptions` in `configs/dataset.yaml`,
decided 2026-10-23 before any training or test scoring); with it the audit status is **PASS**:

| Video | Split / class | Finding |
|---|---|---|
| `6_1_12_1` | train / bona fide | decoded frame 90 (of 121) has a damaged MJPEG block → kept; frame 90 never selected |
| `3_1_54_5` | test / replay | only 2 of 151 frames have valid eye annotations → kept; scored on those 2 frames |
| `5_1_54_3` | test / print | only 19 of 126 frames have valid eye annotations → kept; scored on those 19 frames |

Rationale: dropping hard-to-annotate test attacks could bias APCER downwards. The audit accepts only these exact
cases and fails if any documented exception stops matching the data.

No numbers in this repository or report are typed by hand: they are generated from result files, and
anything not yet measured appears as `[INSERT MEASURED RESULT FROM …]`.

---

## 2. Private-data rules (OULU-NPU licence)

* The dataset lives only in private Google Drive (`MyDrive/QPAD_Project/OULU_NPU/raw/`) and processed outputs only in
  `MyDrive/QPAD_Project/`. Nothing from either location is committed (see `.gitignore`).
* Never create share links, never upload videos/crops to GitHub, never store credentials in notebooks.
* Face thumbnails are disabled by default (`SHOW_FACES = False`); report figures are aggregate plots only.
* The repository contains code, configs, aggregate tables/figures and instructions — nothing else.

---

## 3. Quick start in Google Colab

1. **Put the code on Drive** (simplest): upload this whole `qpad/` folder to
   `MyDrive/QPAD_Project/code/qpad`. *Or* push it to a **private** GitHub repo and set `REPO_URL` in the
   second cell of each notebook (Colab will ask you to authenticate; never paste tokens).
2. **Open a notebook**: in Colab, *File → Open notebook → Google Drive* →
   `QPAD_Project/code/qpad/notebooks/00_colab_setup_and_archive_inventory.ipynb`
   (or *File → Upload notebook*). Choose *Runtime → Change runtime type → GPU* for notebooks 04–10.
3. **Run top to bottom.** Every notebook starts with `DRY_RUN = True` and processes only 2 videos per split and
   class. Dry-run outputs go to `QPAD_Project/_dry_run/`, so they can never overwrite real results.
4. **Inspect** the "Before you continue" checklist at the end, then set `DRY_RUN = False` and re-run.
5. Long cells print a **pre-flight** summary (items, sources, destinations, free disk, config, seed) first.
   Everything is resumable: re-running a cell after a disconnect skips completed work.

### Notebook execution order

| # | Notebook | Priority | GPU | Main outputs (under `MyDrive/QPAD_Project/`) |
|---|---|---|---|---|
| 00 | `00_colab_setup_and_archive_inventory` | 0 | no | `manifests/archive_inventory.json`, archive indexes |
| 01 | `01_dataset_audit` | 0 | no | `manifests/oulu_manifest.{csv,parquet}`, `reports/dataset_audit.{json,md}` |
| 02 | `02_preprocess_oulu` (run once per split: train → dev → test) | 0 | no | `processed/<split>/*_shard_*.tar`, `processed/processed_manifest_<split>.csv` |
| 03 | `03_eda` | 0 | no | `results/eda/*.{png,pdf}`, `captions.json`, tables |
| 04 | `04_train_deeppixbis_fp32` | 0 | **yes** | `models/deeppixbis_best_fp32.pt` (+ `.FROZEN.json`), `results/training_*` |
| 05 | `05_threshold_and_fp32_evaluation` | 0 | yes | `results/fp32/{threshold,metrics}.json`, score CSVs, breakdowns, report assets |
| 06 | `06_export_onnx_tensorrt_validation` | 1 | yes + TRT | `models/deeppixbis_fp32.onnx`, `results/export/equivalence.json` |
| 07 | `07_int8_calibration_pilot` | 1 | yes + TRT | 6 pilot engines, scale audit, `pilot_summary.csv` |
| 08 | `08_full_int8_experiment` | 2 | yes + TRT | 35 engines + 3 rebuilds, `master_results.csv`, `rebuild_noise.json` |
| 09 | `09_statistical_analysis` | 2 | no | `results/stats/comparisons.{csv,tex}`, `analysis.json` |
| 10 | `10_latency_and_final_tables` | 2 | yes + TRT (fixed GPU) | `results/latency/latency.{csv,tex}` |

Gates that require a deliberate manual change: `DRY_RUN = False` (all), `FREEZE_CHECKPOINT = True` (04),
`CONFIRM_THRESHOLD_FROZEN = True` (05), `CONTINUE_FULL_EXPERIMENT`/`RUN_FULL_EXPERIMENT = True` (07/08, only after
the pilot decision with the professor), `EXPECTED_GPU` (10).

---

## 4. Repository layout

```
qpad/
├── configs/            dataset.yaml, preprocessing.yaml, training.yaml, int8_experiments.yaml
├── notebooks/          00 … 10 (generated by scripts/build_notebooks.py; committed without outputs)
├── src/qpad/           all logic (notebooks only orchestrate)
│   ├── config.py           authoritative paths, YAML loading, logging, seeding, disk checks
│   ├── archive_io.py       TAR/ZIP header indexing, single-member extraction by byte offset
│   ├── manifest.py         filename parsing, PAI mapping, Protocol II membership, validation checks
│   ├── annotations.py      eye-coordinate parsing, validity, frame-numbering inference
│   ├── video_io.py         ffprobe, full ffmpeg decode, sequential OpenCV frame reading (RGB)
│   ├── audit.py            per-video audit worker, resumable audit, reports
│   ├── alignment.py        exact two-eye similarity alignment
│   ├── preprocessing.py    frame selection, crops, quality stats, normalization, shards
│   ├── datasets.py         shard → local memmap frame store, seeded PyTorch loaders
│   ├── deeppixbis.py       DeepPixBiS network and loss
│   ├── training.py         resumable training, dev-loss selection, checkpoint freezing
│   ├── scoring.py          grid → frame → video scores (shared by every backend)
│   ├── thresholds.py       pre-registered EER rule with exact tie-breaking
│   ├── metrics.py          APCER/BPCER/ACER/EER/AUROC with counts, breakdowns, decision flips
│   ├── export.py           ONNX export + equivalence comparison
│   ├── calibration.py      nested calibration sampling, TensorRT calibrators, scale audit
│   ├── tensorrt_utils.py   engine building (TF32 control), inference, latency benchmark
│   ├── bootstrap.py        subject-clustered paired bootstrap
│   ├── permutation.py      subject-level paired permutation test, Holm correction
│   ├── decision_rules.py   mechanical application of the pre-registered result rules
│   ├── experiments.py      Priority 1/2 stage functions (export, INT8, rebuild noise, statistics)
│   ├── pipeline.py         Priority 0 stage functions (inventory … FP32 evaluation), DRY_RUN isolation
│   ├── plotting.py         report figures (title/axes/legend/caption enforced; PNG + PDF)
│   ├── reporting.py        LaTeX tables/macros generated from result files
│   ├── reproducibility.py  hashes, versions, git commit, experiment_metadata.json
│   ├── checkpointing.py    resumable record store, verified copies
│   └── nbutils.py          progress bars, pre-flight summaries
├── tests/              unit tests + synthetic OULU-like dataset + notebook dry-run execution
├── scripts/            build_notebooks.py, build_report_assets.py, run_checks.sh
├── reports/checkpoint1/  IEEE report source, generated tables/figures
├── requirements-colab.txt, requirements-dev.txt, .gitignore, pytest.ini
```

---

## 5. Key design decisions (and why)

**Data access.** The TARs are never extracted. Notebook 00 scans their headers once (no payload is read) and
caches every member's byte offset; later stages copy *one* video at a time to `/content`, process it and delete
it. Real archive layout (discovered, not assumed): `Train_files/<id>.avi` + `<id>.txt` etc.; the metadata ZIP
nests `OULU-NPU/Protocols.tar` (official lists, extracted automatically), `Baseline.tar` (not used) and `Readme.pdf`.

**Protocol II membership.** Subject range × all phones × all sessions × PAI codes {1,2,4} (train/dev) or
{1,3,5} (test). The audit cross-checks this rule against the official `Protocol_2/{Train,Dev,Test}.txt`.
The split archives also contain videos outside Protocol II (e.g. Printer 2/Display 2 in train); they are
listed in the manifest with `in_protocol = False` and never used.

**Frames.** 20 evenly spaced *valid* annotated frames per video (positions `round(linspace(0, n−1, 20))` over
valid frames); training uses the same rule (`training_per_video: 20`); 5 calibration frames per training
video are an evenly spaced subset of those 20. Frame indices are saved and shared by every backend.

**Alignment.** Exact similarity transform placing the image-left eye at (55, 44) and image-right eye at
(168, 44) in a 224×224 crop (bob.bio.face default geometry), bilinear, black border. Colour order RGB.
Normalization (ImageNet mean/std) is applied at load time by one function, `qpad.preprocessing.to_model_input`.

**Storage.** One compressed `.npz` per video inside 50-video TAR shards with JSON sidecars and per-frame CSVs
(≈60 files per split on Drive instead of ~20,000 images). A shard counts as done only after it is verified,
copied with a SHA-256 check and its sidecar written.

**Model and training.** DeepPixBiS = first 8 children of ImageNet DenseNet-161 → 1×1 conv + sigmoid (14×14 map)
→ linear + sigmoid; loss 0.5·pixel BCE + 0.5·binary BCE; Adam lr 1e-4, weight decay 1e-5, batch 32; frame
score = mean of the map; video score = mean of 20 frame scores; higher = bona fide. Epoch selected by
**development loss**; test never used.

**Threshold.** `score >= threshold` → bona fide. Candidates = distinct dev scores + one above the max;
minimise |APCER − BPCER| → lowest ACER → highest threshold, compared with exact integer arithmetic.

**APCER/ACER conventions.** We report per-attack APCER, pooled APCER, worst-PAI APCER, BPCER and two ACERs:
`ACER = (pooled APCER + BPCER)/2` (project definition) and `ACER_max_pai = (max-PAI APCER + BPCER)/2`
(OULU benchmark convention, needed to compare with published numbers). The threshold rule uses pooled APCER.

### Documented deviations / uncertain assumptions (verify before the report)

1. Alignment uses the supplied eye coordinates and our crop geometry; the DeepPixBiS paper detects faces with MTCNN,
   localises landmarks with SDM, levels the eyes and resizes to 224 x 224.
2. Training frame sampling (20 evenly spaced frames) is our choice; it is **not** claimed to be the original rule.
3. Augmentation: horizontal flip only (colour jitter optional, off), no class balancing, 30 epochs with early
   stopping (patience 5). The paper undersamples the majority class, adds brightness/contrast/saturation jitter
   and trains for 50 epochs (optimizer, learning rate, weight decay and batch size match).
4. Published DeepPixBiS Protocol II result (George & Marcel, ICB 2019, Table 2; verified): APCER 11.39 %,
   BPCER 0.56 %, ACER 5.97 % (worst-PAI APCER, dev-EER threshold). Used as a sanity reference, not a target.
5. Primary INT8 method is TensorRT entropy calibration (`IInt8EntropyCalibrator2`, implicit quantization).
   It is deprecated in recent TensorRT 10 releases; if the installed version lacks it, the code **stops**
   instead of switching to ModelOpt Q/DQ silently. Pin one TensorRT version before building engines.
6. The Tier 2A/2B pivot threshold and the permutation test sidedness (default two-sided) should be confirmed
   with the professor before the pilot.

---

## 6. Reproducibility

* Seeds: `RANDOM_SEED = 42` (notebooks), training seed in `configs/training.yaml`, calibration seeds 1–5,
  bootstrap seed in `configs/int8_experiments.yaml`. Deterministic cuDNN by default.
* Every stage writes a `*_metadata.json` (git commit + dirty flag, configs, seeds, input file SHA-256s,
  package versions, GPU/driver). The frozen checkpoint's hash is checked by every later stage.
* The threshold is frozen before the test split is scored: `test_evaluation_lock.json` stores its hash and any
  later test evaluation with a different threshold is refused (re-running with the same threshold is allowed).
* Report numbers come from `scripts/build_report_assets.py --project-root <QPAD_Project>`.

**Commit the code before running full experiments**, so the recorded commit identifies the code that produced
each result.

---

## 7. Testing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # plus ffmpeg on PATH
bash scripts/run_checks.sh                   # build notebooks, compileall, pytest -q
```

Tests use a generated synthetic OULU-like dataset (no biometric data): filename/PAI parsing, Protocol II
membership, leakage, annotation parsing and frame-base inference, frame selection, alignment geometry, RGB
order, grid→frame→video scoring, APCER/BPCER/ACER with counts, threshold tie-breaking, nested calibration
sampling, bootstrap, permutation, Holm, decision rules, resume behaviour, shard verification, ONNX equivalence,
and dry-run execution of notebooks 00–06. TensorRT code paths cannot be tested without a CUDA GPU.

**Running stages outside Colab.** The pipeline functions accept environment overrides used only for testing
or local runs: `QPAD_DATASET_ROOT`, `QPAD_PROJECT_DRIVE_ROOT`, `QPAD_LOCAL_WORK_ROOT` (printed as warnings).

---

## 8. Scope changes since the proposal

Voice-PA dataset → face PAD; OULU-NPU Protocol II; print and replay attacks; DeepPixBiS; INT8
calibration-composition bias. (Highlighted in blue in the checkpoint report.) Replay-Attack cross-dataset
validation is optional and kept separate; conclusions are limited to OULU-NPU Protocol II.

## 9. Work division

Ananya: audit/manifest, preprocessing, EDA, statistics, report integration. Vindhya: training, FP32
evaluation, export, INT8 engines, latency. Both review thresholds, calibration manifests, final tables and
conclusions.

## 10. Open questions for the professor

* Checkpoint deadline: the PDF says "Thursday, October 27, 2026", but that date is a Tuesday.
* Tier 2A/2B decision. The pilot ran on 2026-10-28 (`reports/lab_notes/2026-10-28_export_and_int8_pilot.md`):
  unusual vs natural differences were ≤ 0.83 pp with signs flipping between seeds, so by the plan's rule
  (2A only for meaningful or repeatable differences) we propose **Tier 2B** (robustness). Awaiting confirmation.
* One- vs two-sided permutation test for the directional hypothesis.
