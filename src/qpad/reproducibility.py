"""Provenance helpers: hashes, software/hardware versions and experiment metadata."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

HASH_CHUNK_BYTES = 8 * 1024 * 1024

TRACKED_PACKAGES = ("numpy", "pandas", "torch", "torchvision", "cv2", "onnx",
                    "onnxruntime", "tensorrt", "modelopt", "scipy", "matplotlib", "yaml")


def file_sha256(path: Path | str) -> str:
    """SHA-256 hex digest of a file, read in chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_commit(repo_root: Path | str, ignore: Iterable[str] = ()) -> dict[str, Any]:
    """Current git commit and dirty flag, or ``None`` values when unavailable.

    ``ignore``: repo-relative paths whose changes do not count as dirty (e.g. generated report assets).
    """
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True,
                                 text=True, check=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--", ".", *(f":(exclude){p}" for p in ignore))
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def package_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed versions of the tracked packages (``None`` if missing)."""
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in packages:
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - any import failure means "not available"
            versions[name] = None
            continue
        versions[name] = str(getattr(mod, "__version__", "unknown"))
    return versions


def gpu_info() -> dict[str, Any]:
    """GPU model, driver and CUDA information from PyTorch and ``nvidia-smi``."""
    info: dict[str, Any] = {"cuda_available": False}
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=20)
        info["nvidia_smi"] = [line.strip() for line in out.stdout.splitlines() if line.strip()]
        if info["nvidia_smi"]:
            info["driver_version"] = info["nvidia_smi"][0].split(",")[1].strip()
    except (OSError, subprocess.SubprocessError):
        info["nvidia_smi"] = None
    return info


def environment_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpu": gpu_info(),
    }


def write_json(path: Path | str, obj: Any) -> Path:
    """Write JSON atomically (temp file + rename) so an interrupted write never corrupts it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def write_experiment_metadata(
    out_path: Path | str,
    stage: str,
    repo_root: Path | str,
    config: Mapping[str, Any] | None = None,
    seeds: Mapping[str, int] | None = None,
    inputs: Mapping[str, Path | str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``experiment_metadata.json`` tracing a result to code, config, data and environment.

    ``inputs`` maps a role (e.g. ``manifest``, ``checkpoint``, ``threshold``) to a file;
    each existing file is recorded with its SHA-256.
    """
    hashed: dict[str, Any] = {}
    for role, p in (inputs or {}).items():
        p = Path(p)
        hashed[role] = {"path": str(p), "sha256": file_sha256(p) if p.is_file() else None,
                        "exists": p.exists()}
    meta = {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_commit(repo_root),
        "seeds": dict(seeds or {}),
        "config": {k: v for k, v in (config or {}).items()},
        "inputs": hashed,
        "environment": environment_info(),
        "extra": dict(extra or {}),
    }
    write_json(out_path, meta)
    return meta
