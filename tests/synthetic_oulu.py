"""Build a tiny synthetic OULU-NPU-like dataset (TAR + ZIP archives) for tests.

It mimics the real layout assumptions that the code must *discover*:
videos ``<phone>_<session>_<subject>_<pai>.avi`` with same-stem ``.txt`` eye
annotations inside per-split TAR files, and Protocol_2 list files inside a
metadata ZIP. No real biometric data is involved.

Protocol scale is reduced (6 subjects, 2 phones, 1 session) through a derived
dataset YAML; the membership *rule* is unchanged.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
N_FRAMES = 30
W, H = 160, 120
EYES = ((60.0, 50.0), (100.0, 50.0))
SYN_SPLITS = {"train": [1, 2], "dev": [3, 4], "test": [5, 6]}
SYN_PHONES = [1, 2]
SYN_SESSIONS = [1]


def synthetic_dataset_cfg(dataset_root: Path) -> dict[str, Any]:
    """Real dataset.yaml with the protocol scaled down (rule and PAI codes unchanged)."""
    cfg = yaml.safe_load((REPO / "configs" / "dataset.yaml").read_text())
    cfg["paths"]["dataset_root"] = str(dataset_root)
    cfg["filename"]["phones"] = SYN_PHONES
    cfg["filename"]["sessions"] = SYN_SESSIONS
    for split, (lo, hi) in {k: (v[0], v[-1]) for k, v in SYN_SPLITS.items()}.items():
        cfg["protocol"]["splits"][split]["subjects"] = [lo, hi]
    per = len(SYN_PHONES) * len(SYN_SESSIONS) * 2  # subjects per split = 2
    cfg["protocol"]["expected_counts"] = {s: {c: per for c in cfg["classes"]} for s in SYN_SPLITS}
    cfg["protocol"]["expected_total"] = per * 3 * 3
    cfg["data_exceptions"] = {}  # the real dataset's documented exceptions do not apply to synthetic data
    cfg["audit"]["workers"] = 2
    cfg["audit"]["checkpoint_every"] = 5
    return cfg


def _frame(rng: np.random.Generator, cls: str, t: int) -> tuple[np.ndarray, tuple[float, ...]]:
    img = np.full((H, W, 3), 110, np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    face = ((xx - 80) / 38.0) ** 2 + ((yy - 62) / 50.0) ** 2 < 1
    img[face] = (190, 160, 140)
    if cls == "print":
        img[(xx + yy) % 4 == 0] //= 2           # halftone-like texture
    elif cls == "replay":
        img[(xx // 3) % 2 == 0] = np.clip(img[(xx // 3) % 2 == 0].astype(int) + 40, 0, 255)  # moire stripes
    noise = rng.integers(-8, 9, size=img.shape)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    dx = float(np.sin(t / 5.0) * 2)
    (x1, y1), (x2, y2) = EYES
    for (ex, ey) in ((x1 + dx, y1), (x2 + dx, y2)):
        img[int(ey) - 2:int(ey) + 3, int(ex) - 3:int(ex) + 4] = (20, 20, 20)
    return img, (x1 + dx, y1, x2 + dx, y2)


def write_video_and_annotation(path_avi: Path, cls: str, seed: int, n_frames: int = N_FRAMES) -> str:
    import cv2

    rng = np.random.default_rng(seed)
    vw = cv2.VideoWriter(str(path_avi), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H))
    lines = []
    for t in range(n_frames):
        rgb, (x1, y1, x2, y2) = _frame(rng, cls, t)
        vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        lines.append(f"{t + 1}, {x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}")  # 1-based frame numbers
    vw.release()
    return "\n".join(lines) + "\n"


def build_synthetic_dataset(root: Path, corrupt: str | None = None, nested_protocols: bool = True) -> dict[str, Any]:
    """Create archives under ``root/dataset biometric`` (path with a space, like the real one).

    ``corrupt``: ``None``, ``"missing_annotation"`` (drop one annotation) or
    ``"bad_video"`` (truncate one video) for negative tests.
    """
    ds_root = root / "dataset biometric"
    ds_root.mkdir(parents=True, exist_ok=True)
    cfg = synthetic_dataset_cfg(ds_root)
    work = root / "_build"
    work.mkdir(exist_ok=True)
    cls_of = {1: "bona_fide", 2: "print", 3: "print", 4: "replay", 5: "replay"}
    folder = {"train": "Train_files", "dev": "Dev_files", "test": "Test_files"}
    proto_lines: dict[str, list[str]] = {s: [] for s in SYN_SPLITS}
    seed = 0
    for split, subjects in SYN_SPLITS.items():
        tar_path = ds_root / cfg["paths"]["archives"][split]
        with tarfile.open(tar_path, "w") as tf:
            for subject in subjects:
                for phone in SYN_PHONES:
                    for session in SYN_SESSIONS:
                        for pai in (1, 2, 3, 4, 5):
                            vid = f"{phone}_{session}_{subject:02d}_{pai}"
                            avi = work / f"{vid}.avi"
                            ann = write_video_and_annotation(avi, cls_of[pai], seed)
                            seed += 1
                            data = avi.read_bytes()
                            if corrupt == "bad_video" and vid == "1_1_01_1":
                                data = data[: len(data) // 3]
                            ti = tarfile.TarInfo(f"{folder[split]}/{vid}.avi")
                            ti.size = len(data)
                            tf.addfile(ti, io.BytesIO(data))
                            if not (corrupt == "missing_annotation" and vid == "1_1_01_1"):
                                ab = ann.encode()
                                ti = tarfile.TarInfo(f"{folder[split]}/{vid}.txt")
                                ti.size = len(ab)
                                tf.addfile(ti, io.BytesIO(ab))
                            if pai in cfg["protocol"]["splits"][split]["pai_codes"]:
                                proto_lines[split].append(("+1," if pai == 1 else "-1,") + vid)
                            avi.unlink()
    zip_path = ds_root / cfg["paths"]["archives"]["metadata"]
    names = {"train": "Train.txt", "dev": "Dev.txt", "test": "Test.txt"}
    with zipfile.ZipFile(zip_path, "w") as zf:
        if nested_protocols:  # real layout: OULU-NPU/Protocols.tar inside the ZIP
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as pt:
                for split, lines in proto_lines.items():
                    data = ("\n".join(lines) + "\n").encode()
                    ti = tarfile.TarInfo(f"Protocols/Protocol_2/{names[split]}")
                    ti.size = len(data)
                    pt.addfile(ti, io.BytesIO(data))
            zf.writestr("OULU-NPU/Protocols.tar", buf.getvalue())
            zf.writestr("OULU-NPU/Baseline.tar", b"")
        else:
            for split, lines in proto_lines.items():
                zf.writestr(f"OULU-NPU/Protocols/Protocol_2/{names[split]}", "\n".join(lines) + "\n")
        zf.writestr("OULU-NPU/Readme.pdf", b"synthetic")
    cfg_path = root / "dataset_synthetic.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return {"dataset_root": ds_root, "config_path": cfg_path, "cfg": cfg}
