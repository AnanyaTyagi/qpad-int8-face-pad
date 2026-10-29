"""Shared pytest fixtures: import path and a session-scoped synthetic dataset."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from qpad.config import load_yaml  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def real_cfg() -> dict:
    return load_yaml(REPO / "configs" / "dataset.yaml")


@pytest.fixture(scope="session")
def pre_cfg() -> dict:
    return load_yaml(REPO / "configs" / "preprocessing.yaml")


@pytest.fixture(scope="session")
def int8_cfg() -> dict:
    return load_yaml(REPO / "configs" / "int8_experiments.yaml")


def _have_ffmpeg() -> bool:
    return shutil.which("ffprobe") is not None and shutil.which("ffmpeg") is not None


requires_ffmpeg = pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg/ffprobe not installed")


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    pytest.importorskip("cv2")
    from synthetic_oulu import build_synthetic_dataset

    root = tmp_path_factory.mktemp("synthetic")
    return build_synthetic_dataset(root)
