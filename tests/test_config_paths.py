from pathlib import Path

from qpad.config import ProjectPaths, load_config, require_free_space, set_global_seed


def test_authoritative_paths(real_cfg, monkeypatch):
    for v in ("QPAD_DATASET_ROOT", "QPAD_PROJECT_DRIVE_ROOT", "QPAD_LOCAL_WORK_ROOT"):
        monkeypatch.delenv(v, raising=False)
    p = ProjectPaths.from_config(real_cfg)
    assert p.dataset_root == Path("/content/drive/MyDrive/QPAD_Project/OULU_NPU/raw")
    assert p.archive("train") == Path("/content/drive/MyDrive/QPAD_Project/OULU_NPU/raw/Train_files-002.tar")
    assert p.archive("dev").name == "Dev_files-001.tar"
    assert p.archive("test").name == "Test_files-003.tar"
    assert p.archive("metadata").name == "OULU-NPU-20261010T032906Z-1-004.zip"
    assert p.project_drive_root == Path("/content/drive/MyDrive/QPAD_Project")
    assert p.local_work_root == Path("/content/qpad_work")
    assert p.manifests == Path("/content/drive/MyDrive/QPAD_Project/manifests")
    assert not p.overrides


def test_env_override_is_reported(real_cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("QPAD_DATASET_ROOT", str(tmp_path / "with space"))
    p = ProjectPaths.from_config(real_cfg)
    assert p.dataset_root == tmp_path / "with space"
    assert "WARNING" in p.describe()


def test_load_config_all_files(repo_root):
    cfg = load_config(repo_root / "configs")
    assert {"dataset", "preprocessing", "training", "int8"} <= set(cfg)


def test_seed_reproducible():
    import numpy as np

    set_global_seed(123)
    a = np.random.rand(3)
    set_global_seed(123)
    assert (a == np.random.rand(3)).all()


def test_require_free_space(tmp_path):
    assert require_free_space(tmp_path, 0.0) >= 0
    import pytest

    with pytest.raises(RuntimeError):
        require_free_space(tmp_path, 1e9)
