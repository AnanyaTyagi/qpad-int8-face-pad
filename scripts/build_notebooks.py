"""Generate the QPAD Colab notebooks (``notebooks/*.ipynb``) from one source of truth.

Run:  python scripts/build_notebooks.py
Every notebook shares the same setup cells (Drive mount, code import, paths,
versions, seeds, DRY_RUN switches) and calls functions from ``src/qpad``; no
major logic lives in notebook cells.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


# ---------------------------------------------------------------------------------------
# Shared cells
# ---------------------------------------------------------------------------------------
def header(title: str, purpose: str, simple: str, inputs: list[str], outputs: list[str], priority: str):
    return md(f"""
# {title}

**QPAD — Calibration-Set Composition Bias in INT8 Face Presentation Attack Detection**
CS 228 Biometric Security with AI · San José State University · Ananya Tyagi & Vindhya Channarayapatna Ravishankar

**Priority:** {priority}

**Purpose.** {purpose}

**In plain words.** {simple}

**Expected inputs**
{chr(10).join(f"- {i}" for i in inputs)}

**Expected outputs**
{chr(10).join(f"- {o}" for o in outputs)}

**Safety rules.** The OULU-NPU dataset is access-restricted. This notebook never uploads data, never creates
share links, never prints credentials, and never writes videos or face crops to the Git repository.
Long operations run only after you change `DRY_RUN = True` to `False` and read the pre-flight summary.
""")


MOUNT = code('''
# --- Step 1: mount Google Drive (private storage) -------------------------------------------
# Authentication happens in Colab's own pop-up; nothing about it is stored in this notebook.
import os, sys, json, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and not os.environ.get("QPAD_SKIP_DRIVE_MOUNT"):
    from google.colab import drive
    drive.mount("/content/drive")
else:
    print("Drive mount skipped (not in Colab, or test mode).")
''')

CODE_IMPORT = code('''
# --- Step 2: make the QPAD code importable ------------------------------------------------------
# Option A (default): upload the whole `qpad/` repository folder to
#   MyDrive/QPAD_Project/code/qpad
# Option B: set REPO_URL to your PRIVATE GitHub repository; Colab/GitHub will ask you to authenticate.
#   Never paste access tokens into this notebook.
REPO_URL = ""  # e.g. "https://github.com/<account>/qpad.git"
QPAD_CODE_DIR = Path(os.environ.get("QPAD_CODE_DIR", "/content/drive/MyDrive/QPAD_Project/code/qpad"))
if REPO_URL:
    QPAD_CODE_DIR = Path("/content/qpad")
    if not (QPAD_CODE_DIR / ".git").exists():
        subprocess.run(["git", "clone", REPO_URL, str(QPAD_CODE_DIR)], check=True)
if not (QPAD_CODE_DIR / "src" / "qpad").is_dir():
    raise FileNotFoundError(f"QPAD code not found at {QPAD_CODE_DIR}. Upload the repo folder or set REPO_URL.")
sys.path.insert(0, str(QPAD_CODE_DIR / "src"))

if IN_COLAB and not os.environ.get("QPAD_SKIP_PIP"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    str(QPAD_CODE_DIR / "requirements-colab.txt")], check=True)
import qpad
print("qpad", qpad.__version__, "from", QPAD_CODE_DIR)
''')

PATHS = code('''
# --- Step 3: authoritative paths -------------------------------------------------------------------
# These are the ONLY data/output locations. pathlib handles any spaces in paths.
from qpad.config import ProjectPaths, load_config, get_logger, set_global_seed, disk_free_gb

DATASET_ROOT = Path("/content/drive/MyDrive/QPAD_Project/OULU_NPU/raw")
TRAIN_TAR = DATASET_ROOT / "Train_files-002.tar"
DEV_TAR = DATASET_ROOT / "Dev_files-001.tar"
TEST_TAR = DATASET_ROOT / "Test_files-003.tar"
METADATA_ZIP = DATASET_ROOT / "OULU-NPU-20261010T032906Z-1-004.zip"
PROJECT_DRIVE_ROOT = Path("/content/drive/MyDrive/QPAD_Project")
LOCAL_WORK_ROOT = Path("/content/qpad_work")

cfgs = load_config(QPAD_CODE_DIR / "configs")
paths = ProjectPaths.from_config(cfgs["dataset"])
if not paths.overrides and not cfgs["_overrides"]:  # YAML must agree with the constants above
    assert paths.dataset_root == DATASET_ROOT and paths.project_drive_root == PROJECT_DRIVE_ROOT
    assert paths.archive("train") == TRAIN_TAR and paths.archive("dev") == DEV_TAR
    assert paths.archive("test") == TEST_TAR and paths.archive("metadata") == METADATA_ZIP
    assert paths.local_work_root == LOCAL_WORK_ROOT
for o in cfgs["_overrides"]:
    print("WARNING (test mode):", o)
paths.ensure_output_dirs()
print(paths.describe())
''')

VERSIONS = code('''
# --- Step 4: software, GPU and disk information ------------------------------------------------
from qpad.reproducibility import environment_info, git_commit
from qpad.nbutils import df_h, tqdm_progress, preflight

ENV_INFO = environment_info()
print(json.dumps(ENV_INFO, indent=1, default=str))
print("git:", git_commit(QPAD_CODE_DIR))
print(df_h(LOCAL_WORK_ROOT if LOCAL_WORK_ROOT.exists() else Path("/content") if Path("/content").exists() else Path.cwd()))
''')


def run_config(nb_id: str, extra: str = "") -> object:
    return code(f'''
# --- Step 5: run configuration and seeds ---------------------------------------------------------
DRY_RUN = True                 # change to False ONLY after the dry run looks correct
DRY_RUN_VIDEOS_PER_CLASS = 2   # per split and class, deterministic (first IDs)
FORCE_REBUILD = False          # True re-creates outputs that already exist
RETRY_FAILED = False           # True re-processes items recorded as failed
RANDOM_SEED = 42
{extra}
if os.environ.get("QPAD_FORCE_DRY_RUN"):
    DRY_RUN = True
SEED_INFO = set_global_seed(RANDOM_SEED)
logger = get_logger("qpad.{nb_id}", paths.logs / "{nb_id}.log")
logger.info("DRY_RUN=%s seed=%s", DRY_RUN, RANDOM_SEED)
from qpad.pipeline import output_paths
OUT = output_paths(paths, DRY_RUN)
OUT.ensure_output_dirs()
print("Outputs of this run go to:", OUT.project_drive_root)
''')


def footer(next_nb: str | None):
    return md(f"**Next notebook:** `{next_nb}`" if next_nb else "This is the last notebook.")


def notebook(name: str, cells: list) -> None:
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU"})
    for i, cell in enumerate(nb.cells):  # deterministic IDs -> clean git diffs on regeneration
        cell["id"] = f"{name[:2]}-{i:02d}"
    nbformat.validate(nb)
    NB_DIR.mkdir(exist_ok=True)
    nbformat.write(nb, NB_DIR / name)


def setup(nb_id: str, extra: str = "") -> list:
    return [MOUNT, CODE_IMPORT, PATHS, VERSIONS, run_config(nb_id, extra)]


# ---------------------------------------------------------------------------------------
# 00 — setup and archive inventory
# ---------------------------------------------------------------------------------------
def nb00():
    cells = [header(
        "00 · Colab setup and archive inventory",
        "Verify the four OULU-NPU archives exist, record their sizes, and discover the internal layout "
        "(videos, annotations, protocol files, naming) by reading archive headers only.",
        "Before touching any video we look *inside* the archives the way you would read a table of contents. "
        "A TAR file stores a small header before each file, so we can list every member and its size without "
        "extracting 105 GB.",
        ["The four archives in `MyDrive/QPAD_Project/OULU_NPU/raw/`."],
        ["`QPAD_Project/manifests/archive_inventory.json`", "`QPAD_Project/manifests/archive_index_<archive>.json.gz` "
         "(member names, sizes and byte offsets; reused by every later notebook)"],
        "0 (Checkpoint 1)")]
    cells += setup("nb00", "PEEK_MEMBERS = 200          # DRY_RUN: read only this many TAR headers per archive")
    cells += [
        md("## 1. Do the archives exist?\nIf a path is wrong, the error lists the nearby Drive folders "
           "(only the immediate parent folders — the Drive is never searched)."),
        code('''
from qpad.archive_io import check_archives_exist
ARCHIVES = {k: paths.archive(k) for k in ("train", "dev", "test", "metadata")}
check_archives_exist(ARCHIVES)
expected_gb = {"train": 37.75, "dev": 28.15, "test": 39.21, "metadata": 0.057}
for k, p in ARCHIVES.items():
    gb = p.stat().st_size / 1e9
    print(f"{k:9s} {gb:8.2f} GB   (expected ~{expected_gb[k]} GB)   {p}")
print(f"total     {sum(p.stat().st_size for p in ARCHIVES.values()) / 1e9:8.2f} GB")
print(f"local free space: {disk_free_gb(LOCAL_WORK_ROOT):.1f} GB  -> never extract all archives at once")
'''),
        md("## 2. Peek (dry run) or full header index\n"
           "`DRY_RUN = True` reads only the first few TAR headers (seconds). `DRY_RUN = False` indexes every header "
           "of every archive once and caches the result on Drive (minutes; no payload is read)."),
        code('''
from qpad.archive_io import peek_tar, index_zip, summarize_members
disc = cfgs["dataset"]["discovery"]
EXTS = (disc["video_extensions"], disc["annotation_extensions"], disc["protocol_path_tokens"])
if DRY_RUN:
    peek = {k: (index_zip(p) if k == "metadata" else peek_tar(p, PEEK_MEMBERS)) for k, p in ARCHIVES.items()}
    for k, recs in peek.items():
        s = summarize_members(recs, *EXTS)
        print(f"\\n== {k}: first {len(recs)} members ==")
        print("kinds:", s["kinds"]); print("extensions:", s["extensions"]); print("dirs:", s["directories"])
        for kind, names in s["sample_names"].items():
            print(f"  {kind}: {names[:4]}")
    INVENTORY = None
else:
    from qpad.pipeline import archive_inventory
    preflight("archive header index", 4, list(ARCHIVES.values()), [paths.manifests], LOCAL_WORK_ROOT,
              {"discovery": disc}, RANDOM_SEED, DRY_RUN, "reads headers only; minutes on Drive")
    INVENTORY = archive_inventory(paths, cfgs["dataset"], force=FORCE_REBUILD, logger=logger)
'''),
        md("## 3. What did we discover?\nCheck the extensions, the directory layout, how videos pair with annotation "
           "files, and where the Protocol_2 lists live."),
        code('''
import pandas as pd
if INVENTORY is not None:
    rows = []
    for k, a in INVENTORY["archives"].items():
        rows.append({"archive": k, "members": a["n_members"], "files": a["n_files"],
                     "videos": a["kinds"].get("video", 0), "annotations": a["kinds"].get("annotation", 0),
                     "protocol_files": a["kinds"].get("protocol", 0), "other": a["kinds"].get("other", 0),
                     "uncompressed_GB": round(a["total_uncompressed_bytes"] / 1e9, 2),
                     "videos_with_same_stem_annotation": a["stem_pairing"]["videos_with_same_stem_annotation"]})
    display(pd.DataFrame(rows))
    for k, a in INVENTORY["archives"].items():
        print(f"\\n{k}: extensions {a['extensions']}")
        print(f"  top directories {list(a['directories'])[:5]}")
        for kind, names in a["sample_names"].items():
            print(f"  {kind}: {names[:3]}")
    print("\\nProtocol files:", json.dumps(INVENTORY["protocol_files"], indent=1))
    print("Discovery:", json.dumps(INVENTORY["discovery"], indent=1))
else:
    print("Dry run: set DRY_RUN = False to build the full inventory.")
'''),
        code('''
# --- Final summary -----------------------------------------------------------------------------
print("Archives verified:", list(ARCHIVES))
if INVENTORY is not None:
    print("Inventory saved:", paths.manifests / "archive_inventory.json")
    print("Total uncompressed:", round(INVENTORY["discovery"]["total_uncompressed_bytes"] / 1e9, 2), "GB")
'''),
        footer("01_dataset_audit.ipynb"),
    ]
    notebook("00_colab_setup_and_archive_inventory.ipynb", cells)


# ---------------------------------------------------------------------------------------
# 01 — audit
# ---------------------------------------------------------------------------------------
def nb01():
    cells = [header(
        "01 · OULU-NPU Protocol II dataset audit",
        "Create one manifest row per video and verify counts, Protocol II membership, subject leakage, duplicates, "
        "annotations, frame coverage and full decodability.",
        "We check the dataset is exactly what the protocol promises *before* training anything. Each video is "
        "copied alone into temporary storage, inspected with FFmpeg, and deleted. Progress is saved every few "
        "videos, so a Colab disconnect only loses a few minutes.",
        ["Archives and the header indexes from notebook 00."],
        ["`QPAD_Project/manifests/oulu_manifest.csv` and `.parquet`", "`QPAD_Project/reports/dataset_audit.json` / `.md`",
         "`QPAD_Project/logs/nb01.log`", "`QPAD_Project/manifests/audit_progress.csv` (resume file)"],
        "0 (Checkpoint 1)")]
    cells += setup("nb01")
    cells += [
        md("## 1. Index-level checks (fast)\nNames, labels, counts, leakage and annotation presence come from the "
           "archive indexes alone."),
        code('''
from qpad.pipeline import build_video_manifest, dry_run_subset
from qpad.manifest import validate_structure, checks_to_markdown
from IPython.display import Markdown
VIDEOS, PROTOCOL_LISTS = build_video_manifest(paths, cfgs["dataset"])
STRUCT = validate_structure(VIDEOS, cfgs["dataset"], PROTOCOL_LISTS or None)
display(Markdown(checks_to_markdown(STRUCT)))
print("video members:", len(VIDEOS), "| Protocol II:", int(VIDEOS["in_protocol"].sum()))
display(VIDEOS[VIDEOS.in_protocol].groupby(["split", "class"]).size().unstack())
'''),
        md("## 2. Pre-flight\nRead this before a full run (~2,970 videos)."),
        code('''
from qpad.audit import estimate_audit_hours
TARGET = dry_run_subset(VIDEOS, DRY_RUN_VIDEOS_PER_CLASS) if DRY_RUN else VIDEOS[VIDEOS.in_protocol]
acfg = cfgs["dataset"]["audit"]
assert len(TARGET) > 0, "nothing to audit"
assert TARGET["parse_error"].eq("").all(), "fix filename parse errors first"
preflight("per-video audit (extract one video -> ffprobe -> ffmpeg decode -> delete)", len(TARGET),
          [paths.archive(k) for k in ("train", "dev", "test")],
          [OUT.manifests / "oulu_manifest.csv", OUT.reports / "dataset_audit.md"], LOCAL_WORK_ROOT, acfg,
          RANDOM_SEED, DRY_RUN,
          f"~{estimate_audit_hours(len(TARGET), TARGET.file_size_bytes.mean() / 1e6, workers=acfg['workers'])} h "
          "(assumes ~60 MB/s Drive reads)")
'''),
        md("## 3. Run the audit (resumable)\nRe-running this cell skips videos already recorded in "
           "`audit_progress.csv`. Failed videos are kept and reported; set `RETRY_FAILED = True` to retry them."),
        code('''
from qpad.pipeline import run_dataset_audit
RESULT = run_dataset_audit(paths, cfgs["dataset"], DRY_RUN, DRY_RUN_VIDEOS_PER_CLASS, RETRY_FAILED, logger,
                           tqdm_progress("audit"))
print("OVERALL STATUS:", RESULT["status"])
'''),
        code('''
display(Markdown(checks_to_markdown(RESULT["checks"])))
M = RESULT["manifest"]
print("failed videos:", int((M["status"] == "failed").sum()))
display(M[M["status"] == "failed"][["video_id", "error"]].head(20))
print("resolutions:", RESULT["summary"]["resolution_counts"])
print("frame counts:", RESULT["summary"]["frame_count"], "| duration (s):", RESULT["summary"]["duration_s"])
print("annotation frame bases:", M.loc[M.in_protocol, "ann_frame_base"].value_counts(dropna=False).to_dict())
'''),
        code('''
# --- Final summary -----------------------------------------------------------------------------
print("Status:", RESULT["status"], "(PASS requires every required check to run and pass)")
print("Manifest:", OUT.manifests / "oulu_manifest.csv")
print("Report  :", OUT.reports / "dataset_audit.md")
if DRY_RUN:
    print("DRY RUN: status INCOMPLETE is expected. Set DRY_RUN = False for the real audit.")
'''),
        footer("02_preprocess_oulu.ipynb"),
    ]
    notebook("01_dataset_audit.ipynb", cells)


# ---------------------------------------------------------------------------------------
# 02 — preprocessing
# ---------------------------------------------------------------------------------------
def nb02():
    cells = [header(
        "02 · Deterministic eye-based preprocessing",
        "Select fixed frames, align faces using the supplied eye coordinates, crop to 224×224 RGB, and store "
        "compact restart-safe shards per split.",
        "Every model must see exactly the same pixels. We pick 20 evenly spaced frames per video (plus 5 "
        "calibration frames per training video), rotate/scale each frame so the eyes land on fixed positions, "
        "and save the crops in ~50-video shards. A shard is only counted as done after it is verified on Drive.",
        ["A PASSED audit manifest from notebook 01 (dry runs may use the dry-run manifest)."],
        ["`QPAD_Project/processed/<split>/<split>_shard_XXXX.tar` + `.json` + `.frames.csv`",
         "`QPAD_Project/processed/processed_manifest_<split>.csv`",
         "`QPAD_Project/processed/preprocessing_<split>_metadata.json`"],
        "0 (Checkpoint 1)")]
    cells += setup("nb02", '''SPLITS_TO_PROCESS = ["train"]   # process ONE split at a time: "train", then "dev", then "test"
if os.environ.get("QPAD_SPLITS_TO_PROCESS"):   # test mode only
    SPLITS_TO_PROCESS = os.environ["QPAD_SPLITS_TO_PROCESS"].split(",")
SOURCE_MODE = "stream"            # "stream": read single videos from the TAR on Drive (no big copy)
                                  # "copy_local": copy ONE split TAR to /content first (faster, needs ~40 GB)
SHOW_FACES = False                # face thumbnails are private; keep False unless you need visual QC''')
    cells += [
        md("## 1. Storage format (why shards?)\n"
           "Each video becomes one compressed NumPy file with its aligned crops and metadata (frame numbers, eye "
           "coordinates, affine matrix, crop bounds). Fifty videos form one TAR shard, so Drive holds ~60 shard "
           "files instead of ~25,000 images. Normalization is **not** baked into the pixels: every backend calls the "
           "same `qpad.preprocessing.to_model_input` (ImageNet mean/std) at load time."),
        code('''
from qpad.pipeline import load_verified_manifest, preprocessing_targets
from qpad.annotations import resolve_dataset_frame_base
pcfg = cfgs["preprocessing"]
print(json.dumps({k: pcfg[k] for k in ("output_size", "alignment", "frames", "normalization")}, indent=1))
MANIFEST, AUDIT = load_verified_manifest(paths, DRY_RUN)
print("audit status:", AUDIT["overall_status"])
BASE = resolve_dataset_frame_base(cfgs["dataset"]["annotations"]["frame_index_base"],
                                  MANIFEST.loc[MANIFEST.in_protocol & MANIFEST.status.eq("ok"), "ann_frame_base"])
print("annotation frame numbering base:", BASE)
for s in SPLITS_TO_PROCESS:
    assert s in ("train", "dev", "test")
    t = preprocessing_targets(MANIFEST, s, DRY_RUN, DRY_RUN_VIDEOS_PER_CLASS)
    print(f"{s}: {len(t)} videos -> {len(t) * pcfg['frames']['evaluation_per_video']} evaluation frames")
'''),
        md("## 2. Pre-flight and disk check"),
        code('''
from qpad.config import require_free_space
require_free_space(LOCAL_WORK_ROOT, 3.0)   # one shard + one video at a time
for s in SPLITS_TO_PROCESS:
    t = preprocessing_targets(MANIFEST, s, DRY_RUN, DRY_RUN_VIDEOS_PER_CLASS)
    preflight(f"preprocess {s}", len(t), [paths.archive(s)], [OUT.processed / s], LOCAL_WORK_ROOT,
              {"source_mode": SOURCE_MODE, **pcfg}, RANDOM_SEED, DRY_RUN,
              "about 2–4 s per video when streaming from Drive")
'''),
        md("## 3. Run (resumable, one split at a time)\nRe-running skips verified shards. Failures are recorded in "
           "each shard's JSON sidecar and in `processed_failures_<split>.csv`."),
        code('''
from qpad.pipeline import run_preprocessing, copy_archive_locally
RESULTS = {}
for s in SPLITS_TO_PROCESS:
    overrides = {}
    if SOURCE_MODE == "copy_local" and not DRY_RUN:
        overrides[s] = copy_archive_locally(paths.archive(s), LOCAL_WORK_ROOT / "archives")
    RESULTS[s] = run_preprocessing(paths, cfgs, s, DRY_RUN, DRY_RUN_VIDEOS_PER_CLASS, QPAD_CODE_DIR, overrides,
                                   RETRY_FAILED, logger, tqdm_progress(f"{s} shards"))
    print(s, {k: v for k, v in RESULTS[s].items() if k != "new_failures"})
    if overrides and RESULTS[s]["verified"]:
        overrides[s].unlink()      # remove the local TAR copy only after outputs are verified
        print("removed local copy", overrides[s])
'''),
        md("## 4. Quality control (no faces by default)\nWe verify geometrically that the eyes land on the target "
           "positions and report how much of each crop falls outside the original frame."),
        code('''
import numpy as np, pandas as pd
from qpad.alignment import apply_affine
for s in SPLITS_TO_PROCESS:
    f = pd.read_csv(OUT.processed / f"processed_manifest_{s}.csv", dtype={"video_id": str})
    m = np.stack(f["affine"].map(json.loads).map(np.array).to_list()).reshape(-1, 2, 3)
    left = np.einsum("nij,nj->ni", m[:, :, :2], f[["eye_left_x", "eye_left_y"]].to_numpy()) + m[:, :, 2]
    err = np.abs(left - np.array(pcfg["alignment"]["target_image_left_eye_xy"])).max()
    print(f"{s}: {f.video_id.nunique()} videos, {len(f)} frames; max eye-placement error {err:.2e} px; "
          f"outside-frame fraction p99 = {f.outside_fraction.quantile(0.99):.3f}")
    display(f.groupby("class")[["brightness", "contrast", "sharpness", "outside_fraction"]].describe().T.head(12))
if SHOW_FACES:
    # PRIVATE: shows identifiable faces; output stays in this Colab session only. Clear outputs before sharing.
    import matplotlib.pyplot as plt
    from qpad.preprocessing import iter_shard
    shard = sorted((OUT.processed / SPLITS_TO_PROCESS[0]).glob("*.tar"))[0]
    key, arr = next(iter_shard(shard))
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for ax, img in zip(axes, arr["images"][:5]):
        ax.imshow(img); ax.axis("off")
    fig.suptitle(f"Private QC: aligned crops of {key}")
    plt.show()
'''),
        code('''
# --- Final summary -----------------------------------------------------------------------------
for s, r in RESULTS.items():
    print(f"{s}: verified={r['verified']} frames={r['n_frames']} failed={r['n_failed']} problems={r['problems']}")
print("Next: change SPLITS_TO_PROCESS to the next split and re-run, until train, dev and test are all verified.")
'''),
        footer("03_eda.ipynb"),
    ]
    notebook("02_preprocess_oulu.ipynb", cells)


# ---------------------------------------------------------------------------------------
# 03 — EDA
# ---------------------------------------------------------------------------------------
def nb03():
    cells = [header(
        "03 · Exploratory data analysis",
        "Reproducible univariate and bivariate figures and tables of the Protocol II data for the checkpoint report.",
        "We describe the data before modelling: how many videos of each class, subject, phone and session; how long "
        "videos are; and how bright, contrasty and sharp the aligned faces are for each class.",
        ["Audit manifest (notebook 01).", "Processed manifests for train and dev (notebook 02) for pixel statistics."],
        ["`QPAD_Project/results/eda/*.png` and `*.pdf` (300 dpi + vector)", "`QPAD_Project/results/eda/captions.json`",
         "`class_counts.csv`, `phone_session_counts.csv`, `resolution_table.csv`, `quality_by_class.csv`"],
        "0 (Checkpoint 1)")]
    cells += setup("nb03")
    cells += [
        md("## 1. Generate every figure\nPixel statistics use **train and development only**; test videos are counted "
           "in metadata tables but their pixels are not summarised, so no design decision can be influenced by test data."),
        code('''
from qpad.pipeline import run_eda
EDA = run_eda(paths, DRY_RUN)
print("skipped:", EDA["skipped"] or "nothing")
'''),
        code('''
from IPython.display import Image, display, Markdown
for f in EDA["figures"]:
    display(Image(filename=f["png"], width=640))
    display(Markdown(f"*{f['caption']}*"))
'''),
        code('''
import pandas as pd
eda_dir = Path(EDA["eda_dir"])
for name in ("class_counts.csv", "phone_session_counts.csv", "resolution_table.csv", "quality_by_class.csv"):
    p = eda_dir / name
    if p.is_file():
        print(name); display(pd.read_csv(p).head(20))
'''),
        code('''
# --- Final summary -----------------------------------------------------------------------------
print(len(EDA["figures"]), "figures saved to", EDA["eda_dir"])
'''),
        footer("04_train_deeppixbis_fp32.ipynb"),
    ]
    notebook("03_eda.ipynb", cells)


# ---------------------------------------------------------------------------------------
# 04 — training
# ---------------------------------------------------------------------------------------
def nb04():
    cells = [header(
        "04 · Train the FP32 DeepPixBiS baseline",
        "Train DeepPixBiS once on the Protocol II training split, select the epoch on development loss, and freeze it.",
        "The network looks at a 224×224 face and outputs a 14×14 grid of 'realness' values; training pushes the grid "
        "towards 1 for real faces and 0 for attacks. The development split only decides which epoch to keep. "
        "After this notebook the weights never change again — every INT8 model comes from this one checkpoint.",
        ["Processed train shards (train frames) and dev shards (evaluation frames)."],
        ["`QPAD_Project/models/deeppixbis_best_fp32.pt`, `deeppixbis_last_fp32.pt`, `training_config.json`",
         "`QPAD_Project/results/training_history.csv`, `training_curves.png`",
         "`deeppixbis_best_fp32.pt.FROZEN.json` (SHA-256 of the frozen checkpoint)"],
        "0 (Checkpoint 1)")]
    cells += setup("nb04", '''DRY_RUN_MAX_BATCHES = 3          # dry run: a few batches, 1 epoch
FREEZE_CHECKPOINT = False        # set True only after inspecting the curves of the FULL run''')
    cells += [
        md("## 1. Model and training recipe\nReference: George & Marcel, *Deep Pixel-wise Binary Supervision for Face "
           "Presentation Attack Detection*, ICB 2019 (Idiap `bob.paper.deep_pix_bis_pad`). Architecture: the first 8 "
           "blocks of ImageNet DenseNet-161 → 1×1 conv + sigmoid (14×14 map) → linear + sigmoid (binary output). "
           "Loss 0.5·pixel BCE + 0.5·binary BCE. See README *Deviations* for what is our own choice."),
        code('''
import torch
DEVICE = os.environ.get("QPAD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
if os.environ.get("QPAD_TEST_NO_PRETRAINED"):
    cfgs["training"]["model"]["pretrained_imagenet"] = False   # test mode only: no weight download
if os.environ.get("QPAD_TEST_NUM_WORKERS"):
    cfgs["training"]["data"]["num_workers"] = int(os.environ["QPAD_TEST_NUM_WORKERS"])
tcfg = cfgs["training"]
print("device:", DEVICE)
if DEVICE == "cpu" and not DRY_RUN:
    raise RuntimeError("A full training run needs a GPU runtime (Runtime -> Change runtime type -> GPU).")
if tcfg["runtime"]["deterministic"]:
    print("NOTE: deterministic mode can be slower; some CUDA ops only warn if no deterministic version exists.")
print(json.dumps(tcfg, indent=1))
'''),
        code('''
from qpad.pipeline import processed_split_dir
preflight("DeepPixBiS FP32 training", int(tcfg["schedule"]["epochs"]),
          [processed_split_dir(paths, DRY_RUN, "train"), processed_split_dir(paths, DRY_RUN, "dev")],
          [OUT.models, OUT.results], LOCAL_WORK_ROOT, tcfg, tcfg["runtime"]["seed"], DRY_RUN,
          "full run: roughly 1-3 min per epoch on a T4/L4 for ~21,600 training frames")
'''),
        md("## 2. Train (resumes automatically from `deeppixbis_last_fp32.pt`)"),
        code('''
from qpad.pipeline import run_training
TRAIN = run_training(paths, cfgs, DRY_RUN, DEVICE, QPAD_CODE_DIR,
                     max_batches=DRY_RUN_MAX_BATCHES if DRY_RUN else None,
                     epochs_override=1 if DRY_RUN else None, logger=logger,
                     progress_factory=lambda d: tqdm_progress(d))
display(TRAIN["history"])
'''),
        code('''
from IPython.display import Image
if TRAIN["figure"]:
    display(Image(filename=TRAIN["figure"]["png"], width=640)); print(TRAIN["figure"]["caption"])
s = TRAIN["state"]
print(f"best epoch {s.best_epoch} | best dev loss {s.best_dev_loss:.5f} | epochs run {s.epoch}")
'''),
        md("## 3. Freeze the selected checkpoint\nFreezing records the SHA-256 of `deeppixbis_best_fp32.pt`. Later "
           "notebooks refuse to run if the file changes, and training refuses to overwrite a frozen checkpoint."),
        code('''
from qpad.training import freeze_checkpoint, BEST_NAME
best = OUT.models / BEST_NAME
if DRY_RUN:
    print("dry run: freezing the DRY-RUN checkpoint so notebook 05 can be dry-run:", freeze_checkpoint(best))
elif FREEZE_CHECKPOINT:
    print("FROZEN:", freeze_checkpoint(best, {"best_epoch": TRAIN["state"].best_epoch}))
else:
    print("Inspect the curves, then set FREEZE_CHECKPOINT = True and re-run this cell.")
'''),
        code('''
# --- Final summary -----------------------------------------------------------------------------
print("checkpoint:", OUT.models / BEST_NAME, "| frozen:", (OUT.models / (BEST_NAME + ".FROZEN.json")).is_file())
print("history   :", OUT.results / "training_history.csv")
'''),
        footer("05_threshold_and_fp32_evaluation.ipynb"),
    ]
    notebook("04_train_deeppixbis_fp32.ipynb", cells)


# ---------------------------------------------------------------------------------------
# 05 — threshold and evaluation
# ---------------------------------------------------------------------------------------
def nb05():
    cells = [header(
        "05 · Development threshold and FP32 test evaluation",
        "Select the FP32 EER threshold on development scores only, freeze it, then evaluate the test set once.",
        "A score turns into a decision with a cutoff: `score >= threshold` means bona fide. We choose the cutoff "
        "where development APCER and BPCER are closest. Only after it is saved do we look at the test set, and the "
        "code refuses to re-evaluate the test set with a different threshold.",
        ["Frozen checkpoint (notebook 04).", "Processed dev and test evaluation frames (notebook 02)."],
        ["`QPAD_Project/results/fp32/{dev,test}_{frame,video}_scores.csv`",
         "`threshold.json`, `dev_threshold_candidates.csv`, `metrics.json`, `test_errors_by_*.csv`, figures",
         "Report assets in `reports/checkpoint1/{tables,figures}` (repository copy, aggregate results only)"],
        "0 (Checkpoint 1)")]
    cells += setup("nb05", '''CONFIRM_THRESHOLD_FROZEN = False   # full run: set True after inspecting the dev threshold
UPDATE_REPORT_ASSETS = True''')
    cells += [
        md("## 1. Score development videos and select the threshold\nRule: minimise |APCER − BPCER|; ties → lowest "
           "ACER; still tied → higher threshold (more secure). Counts are compared with exact integers."),
        code('''
import torch, pandas as pd
DEVICE = os.environ.get("QPAD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
from qpad.pipeline import select_fp32_threshold
THR = select_fp32_threshold(paths, cfgs, DRY_RUN, DEVICE, QPAD_CODE_DIR, tqdm_progress("dev frames"))
t = THR["threshold"]
print(f"threshold = {t['threshold']:.6f}")
print(f"dev APCER = {t['apcer_errors']}/{t['n_attacks']} = {100 * t['apcer']:.2f}%")
print(f"dev BPCER = {t['bpcer_errors']}/{t['n_bona_fide']} = {100 * t['bpcer']:.2f}%")
print(f"dev ACER  = {100 * t['acer']:.2f}%")
display(THR["table"][THR["table"].selected | (THR["table"].abs_diff_key <= THR["table"].abs_diff_key.min())].head(10))
'''),
        code('''
from IPython.display import Image
for f in THR["figures"]:
    display(Image(filename=f["png"], width=640)); print(f["caption"])
'''),
        md("## 2. Test evaluation (once)\nThe threshold file is hashed into a lock. Changing the threshold after "
           "this cell has run raises an error."),
        code('''
from qpad.pipeline import evaluate_fp32_test
if not DRY_RUN and not CONFIRM_THRESHOLD_FROZEN:
    raise RuntimeError("Inspect the development threshold, then set CONFIRM_THRESHOLD_FROZEN = True.")
EVAL = evaluate_fp32_test(paths, cfgs, DRY_RUN, DEVICE, QPAD_CODE_DIR, progress=tqdm_progress("test frames"))
m = EVAL["metrics"]["test"]
for k in ("print_apcer", "replay_apcer", "apcer_pooled", "apcer_max_pai", "bpcer"):
    print(f"{k:14s} {m[k]['text']}")
print(f"ACER (pooled) {100 * m['acer']:.2f}%  | ACER (worst PAI, OULU convention) {100 * m['acer_max_pai']:.2f}%")
print(f"EER (test scores, descriptive) {100 * m['eer']['value']:.2f}%  | AUROC {m['auroc']:.4f}")
print("confusion:", m["confusion"])
'''),
        code('''
res_dir = Path(EVAL["results_dir"])
for by in ("phone", "session", "phone_session"):
    print("errors by", by); display(pd.read_csv(res_dir / f"test_errors_by_{by}.csv"))
for f in EVAL["figures"]:
    display(Image(filename=f["png"], width=560)); print(f["caption"])
'''),
        md("## 3. Sanity reference (not a target)\nGeorge & Marcel (ICB 2019, Table 2) report DeepPixBiS on OULU-NPU "
           "Protocol 2 as APCER 11.39 %, BPCER 0.56 %, ACER 5.97 %, with worst-PAI APCER and a development-EER "
           "threshold. A very different result means: inspect preprocessing, labels, loss, sampling and score "
           "aggregation before continuing."),
        code('''
REFERENCE_PUBLISHED = {"apcer_max_pai_pct": 11.39, "bpcer_pct": 0.56, "acer_max_pai_pct": 5.97}   # George & Marcel 2019, Table 2
ours = {"apcer_max_pai_pct": m["apcer_max_pai"]["percent"], "bpcer_pct": m["bpcer"]["percent"],
        "acer_max_pai_pct": 100 * m["acer_max_pai"]}
display(pd.DataFrame([REFERENCE_PUBLISHED, ours], index=["DeepPixBiS (ICB 2019)", "this run"]))
'''),
        code('''
# --- Report assets (aggregate results only; no images of faces) --------------------------------
from qpad.reporting import build_report_assets
from qpad.reproducibility import git_commit
if UPDATE_REPORT_ASSETS and not DRY_RUN:
    w = build_report_assets(OUT.results, OUT.reports, QPAD_CODE_DIR / "reports" / "checkpoint1", git_commit(QPAD_CODE_DIR))
    print(json.dumps(w, indent=1))
else:
    print("dry run: report assets not updated")
'''),
        code('''
# --- Final summary -----------------------------------------------------------------------------
print("threshold:", res_dir / "threshold.json")
print("metrics  :", res_dir / "metrics.json")
print("lock     :", res_dir / "test_evaluation_lock.json")
'''),
        footer("06_export_onnx_tensorrt_validation.ipynb"),
    ]
    notebook("05_threshold_and_fp32_evaluation.ipynb", cells)


def main() -> None:
    for f in (nb00, nb01, nb02, nb03, nb04, nb05):
        f()
    print("notebooks written to", NB_DIR)


if __name__ == "__main__":
    main()
