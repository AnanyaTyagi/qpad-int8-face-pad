"""Configuration loading, authoritative paths, logging and seeding."""

from __future__ import annotations

import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

# Environment variables that may override the authoritative Colab paths.
# They exist ONLY so the notebooks can be executed against a synthetic dataset
# in automated tests. Every override is printed loudly by `ProjectPaths.describe`.
ENV_OVERRIDES = {
    "dataset_root": "QPAD_DATASET_ROOT",
    "project_drive_root": "QPAD_PROJECT_DRIVE_ROOT",
    "local_work_root": "QPAD_LOCAL_WORK_ROOT",
}
ENV_DATASET_CONFIG = "QPAD_DATASET_CONFIG"

CONFIG_FILES = {
    "dataset": "dataset.yaml",
    "preprocessing": "preprocessing.yaml",
    "training": "training.yaml",
    "int8": "int8_experiments.yaml",
}


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load one YAML file into a dictionary."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping")
    return data


def load_config(config_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Load every QPAD configuration file from ``config_dir``.

    ``QPAD_DATASET_CONFIG`` may point to an alternative dataset YAML (used only
    by the synthetic-data tests); the override is recorded under ``_overrides``.
    """
    config_dir = Path(config_dir)
    cfg: dict[str, Any] = {}
    overrides: list[str] = []
    for key, fname in CONFIG_FILES.items():
        path = config_dir / fname
        if key == "dataset" and os.environ.get(ENV_DATASET_CONFIG):
            path = Path(os.environ[ENV_DATASET_CONFIG])
            overrides.append(f"dataset config <- {path}")
        cfg[key] = load_yaml(path)
        cfg[key]["_source"] = str(path)
    cfg["_overrides"] = overrides
    return cfg


@dataclass(frozen=True)
class ProjectPaths:
    """All filesystem locations used by the project.

    The defaults are the authoritative Google Drive / Colab paths from
    ``configs/dataset.yaml``. Paths are ``pathlib.Path`` objects, so
    paths containing spaces never need manual quoting.
    """

    dataset_root: Path
    project_drive_root: Path
    local_work_root: Path
    archive_names: Mapping[str, str]
    overrides: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, dataset_cfg: Mapping[str, Any]) -> "ProjectPaths":
        p = dataset_cfg["paths"]
        values: dict[str, Path] = {}
        overrides: list[str] = []
        for key, env in ENV_OVERRIDES.items():
            if os.environ.get(env):
                values[key] = Path(os.environ[env])
                overrides.append(f"{key} <- ${env} = {values[key]}")
            else:
                values[key] = Path(p[key])
        return cls(
            dataset_root=values["dataset_root"],
            project_drive_root=values["project_drive_root"],
            local_work_root=values["local_work_root"],
            archive_names=dict(p["archives"]),
            overrides=tuple(overrides),
        )

    # ---- archives -----------------------------------------------------------------
    def archive(self, key: str) -> Path:
        """Path of the ``train``/``dev``/``test``/``metadata`` archive."""
        return self.dataset_root / self.archive_names[key]

    # ---- Drive outputs ------------------------------------------------------------
    @property
    def manifests(self) -> Path:
        return self.project_drive_root / "manifests"

    @property
    def reports(self) -> Path:
        return self.project_drive_root / "reports"

    @property
    def logs(self) -> Path:
        return self.project_drive_root / "logs"

    @property
    def processed(self) -> Path:
        return self.project_drive_root / "processed"

    @property
    def models(self) -> Path:
        return self.project_drive_root / "models"

    @property
    def results(self) -> Path:
        return self.project_drive_root / "results"

    @property
    def engines(self) -> Path:
        return self.project_drive_root / "engines"

    # ---- local scratch ------------------------------------------------------------
    @property
    def local_tmp(self) -> Path:
        return self.local_work_root / "tmp"

    @property
    def local_cache(self) -> Path:
        return self.local_work_root / "cache"

    def drive_dirs(self) -> list[Path]:
        return [self.manifests, self.reports, self.logs, self.processed,
                self.models, self.results, self.engines]

    def ensure_output_dirs(self) -> None:
        """Create project output folders (never touches the dataset folder)."""
        for d in self.drive_dirs() + [self.local_tmp, self.local_cache]:
            d.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        lines = [
            f"DATASET_ROOT       = {self.dataset_root}",
            *(f"  {k:<9}-> {self.archive(k)}" for k in self.archive_names),
            f"PROJECT_DRIVE_ROOT = {self.project_drive_root}",
            f"LOCAL_WORK_ROOT    = {self.local_work_root}",
        ]
        if self.overrides:
            lines.append("WARNING: path overrides active (test mode only):")
            lines.extend(f"  {o}" for o in self.overrides)
        return "\n".join(lines)


def get_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to stdout and, optionally, to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_qpad", False) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh._qpad = True  # type: ignore[attr-defined]
        logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
                   for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def set_global_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy and (if installed) PyTorch. Returns what was set."""
    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    info: dict[str, Any] = {"seed": seed, "python": True, "numpy": True, "torch": False,
                            "deterministic": deterministic}
    try:
        import torch
    except ImportError:
        return info
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Required by cuBLAS for deterministic matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    info["torch"] = True
    return info


def disk_free_gb(path: Path | str) -> float:
    """Free space (GB) on the filesystem containing ``path`` (or its nearest existing parent)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def require_free_space(path: Path | str, needed_gb: float) -> float:
    """Raise if fewer than ``needed_gb`` GB are free at ``path``; return the free GB."""
    free = disk_free_gb(path)
    if free < needed_gb:
        raise RuntimeError(
            f"Only {free:.1f} GB free at {path}; {needed_gb:.1f} GB required. "
            "Free space or switch to streaming mode before continuing.")
    return free
