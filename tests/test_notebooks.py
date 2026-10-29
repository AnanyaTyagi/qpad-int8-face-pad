"""Notebook validity checks and dry-run execution against the synthetic dataset."""

import json
import os
import re
from pathlib import Path

import pytest

from conftest import REPO, requires_ffmpeg

NB_DIR = REPO / "notebooks"
NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))
ALLOWED_DRIVE = ("/content/drive/MyDrive/QPAD_Project", "/content/drive\"")


def test_notebooks_00_to_07_exist():
    names = [p.name for p in NOTEBOOKS]
    assert len(names) == 8 and names[0].startswith("00_") and names[-1].startswith("07_")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_structure(path):
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    src = "\n".join(c.source for c in nb.cells)
    assert nb.cells[0].cell_type == "markdown" and nb.cells[0].source.startswith("# ")
    for required in ("drive.mount(", "DRY_RUN = True", "DRY_RUN_VIDEOS_PER_CLASS = 2", "FORCE_REBUILD = False",
                     "RANDOM_SEED = 42", "set_global_seed", "environment_info", "Expected inputs", "Expected outputs"):
        assert required in src, f"{path.name} lacks {required!r}"
    for m in re.finditer(r"/content/drive[^\s\"')]*", src):
        frag = m.group(0)
        assert frag == "/content/drive" or frag.startswith(("/content/drive/MyDrive/dataset",
                                                             "/content/drive/MyDrive/QPAD_Project")), frag
    assert "/Users/" not in src
    assert not re.search(r"(ghp_|github_pat_|hf_[A-Za-z0-9]{10}|password\s*=|api[_-]?key\s*=|token\s*=)", src, re.I)
    for c in nb.cells:
        if c.cell_type == "code":
            compile(c.source.replace("display(", "print("), path.name, "exec")
            assert not c.get("outputs"), "notebooks must be committed without outputs"


def _run(nb_path: Path, env: dict, workdir: Path, allow_error_name: str | None = None):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(nb_path, as_version=4)
    old = dict(os.environ)
    os.environ.update(env)
    try:
        client = nbclient.NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(workdir)}})
        try:
            client.execute()
        except nbclient.exceptions.CellExecutionError as exc:
            if allow_error_name and allow_error_name in str(exc):
                return nb, str(exc)
            raise
    finally:
        os.environ.clear()
        os.environ.update(old)
    return nb, None


@requires_ffmpeg
@pytest.mark.slow
def test_execute_checkpoint_notebooks_dry_run(synthetic, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ipykernel")
    env = {"QPAD_CODE_DIR": str(REPO), "QPAD_SKIP_DRIVE_MOUNT": "1", "QPAD_SKIP_PIP": "1",
           "QPAD_DATASET_CONFIG": str(synthetic["config_path"]),
           "QPAD_PROJECT_DRIVE_ROOT": str(tmp_path / "drive" / "QPAD_Project"),
           "QPAD_LOCAL_WORK_ROOT": str(tmp_path / "content" / "qpad_work"), "QPAD_FORCE_DRY_RUN": "1",
           "QPAD_TEST_NO_PRETRAINED": "1", "QPAD_TEST_NUM_WORKERS": "0", "QPAD_DEVICE": "cpu",
           "QPAD_SPLITS_TO_PROCESS": "train,dev,test", "MPLBACKEND": "Agg"}
    for name in ("00_colab_setup_and_archive_inventory", "01_dataset_audit", "02_preprocess_oulu", "03_eda",
                 "04_train_deeppixbis_fp32", "05_threshold_and_fp32_evaluation"):
        _run(NB_DIR / f"{name}.ipynb", env, tmp_path)
    dry = tmp_path / "drive" / "QPAD_Project" / "_dry_run"
    audit = json.loads((dry / "reports" / "dataset_audit.json").read_text())
    assert audit["overall_status"] == "INCOMPLETE"      # a dry run never claims PASS
    m = json.loads((dry / "results" / "fp32" / "metrics.json").read_text())
    assert m["dry_run"] is True and m["test"]["bpcer"]["total"] == 2
    assert not (tmp_path / "drive" / "QPAD_Project" / "results" / "fp32").exists()   # full-run area untouched
    # Notebook 06 without TensorRT: ONNX equivalence runs, then the STOP rule halts before INT8.
    nb, err = _run(NB_DIR / "06_export_onnx_tensorrt_validation.ipynb", env, tmp_path, "EquivalenceError")
    assert err is not None
    eq = json.loads((dry / "results" / "export" / "equivalence.json").read_text())
    assert eq["pytorch_vs_onnx"]["passed"] and eq["pytorch_vs_onnx"]["max_abs_diff"] <= 1e-5
