"""Loading processed shards into a local memory-mapped frame store and PyTorch datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .preprocessing import ROLE_NAMES, iter_shard, to_model_input


@dataclass
class FrameStore:
    """Aligned uint8 RGB frames of one split/role plus a row-aligned index table."""

    images: np.ndarray        # (N, S, S, 3) uint8, memory-mapped from local disk
    index: pd.DataFrame       # one row per image: video_id, frame_idx, class, label, subject, ...

    def __len__(self) -> int:
        return len(self.index)


def materialize(drive_split_dir: Path, split: str, role: str, local_cache: Path,
                force: bool = False) -> FrameStore:
    """Unpack all frames with ``role`` (eval/train/calib) of ``split`` into ``local_cache``.

    Shards are read directly from Drive (sequential reads of a few hundred files);
    frames are written to one ``.npy`` memmap so training does not hold the split
    in RAM. Rows are ordered by (video_id, frame_idx) for determinism.
    """
    bit = ROLE_NAMES[role]
    out_dir = local_cache / f"{split}_{role}"
    img_path, idx_path = out_dir / "images.npy", out_dir / "index.csv"
    if img_path.is_file() and idx_path.is_file() and not force:
        return FrameStore(np.load(img_path, mmap_mode="r"), pd.read_csv(idx_path, dtype={"video_id": str}))
    frames_meta = []
    sidecars = sorted(drive_split_dir.glob(f"{split}_shard_*.json"))
    if not sidecars:
        raise FileNotFoundError(f"no processed shards for {split} in {drive_split_dir}")
    for sc in sidecars:
        csv = drive_split_dir / f"{json.loads(sc.read_text())['shard']}.frames.csv"
        if csv.stat().st_size > 1:
            frames_meta.append(pd.read_csv(csv, dtype={"video_id": str}))
    meta = pd.concat(frames_meta, ignore_index=True)
    meta = meta[(meta["role_mask"] & bit) > 0].sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    if meta.empty:
        raise ValueError(f"no frames with role {role} in split {split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    first_shard = drive_split_dir / f"{meta['shard'].iloc[0]}.tar"
    size = next(iter_shard(first_shard))[1]["images"].shape[1]
    tmp_img = out_dir / "images.tmp.npy"
    mm = np.lib.format.open_memmap(tmp_img, mode="w+", dtype=np.uint8, shape=(len(meta), size, size, 3))
    row_of = {(v, int(f)): i for i, (v, f) in enumerate(zip(meta["video_id"], meta["frame_idx"]))}
    filled = np.zeros(len(meta), dtype=bool)
    for shard in sorted(meta["shard"].unique()):
        for key, arr in iter_shard(drive_split_dir / f"{shard}.tar"):
            vid = key[:-4]
            for pos, fidx in enumerate(arr["frame_idx"]):
                i = row_of.get((vid, int(fidx)))
                if i is not None:
                    mm[i] = arr["images"][pos]
                    filled[i] = True
    if not filled.all():
        raise IOError(f"{(~filled).sum()} frames listed in CSVs were not found in shards")
    mm.flush()
    del mm
    tmp_img.replace(img_path)
    meta.to_csv(idx_path, index=False)
    return FrameStore(np.load(img_path, mmap_mode="r"), meta)


def video_index(store: FrameStore) -> pd.DataFrame:
    """One row per video with its metadata and frame count."""
    cols = [c for c in ("video_id", "split", "class", "label", "subject", "phone", "session", "pai_code")
            if c in store.index]
    g = store.index.groupby("video_id", sort=True)
    out = g[cols[1:]].first().reset_index()
    out["n_frames"] = g.size().values
    return out


def make_torch_dataset(store: FrameStore, norm_cfg: Mapping[str, Any], augment: Mapping[str, Any] | None = None):
    """A ``torch.utils.data.Dataset`` returning (image tensor, label, row index)."""
    import torch
    from torch.utils.data import Dataset

    labels = store.index["label"].to_numpy(dtype=np.float32)

    class _FrameDataset(Dataset):
        def __len__(self) -> int:
            return len(store)

        def __getitem__(self, i: int):
            img = np.asarray(store.images[i]).astype(np.float32)
            if augment and augment.get("horizontal_flip") and torch.rand(()) < 0.5:
                img = img[:, ::-1]
            if augment and augment.get("color_jitter"):
                # Brightness/contrast jitter in pixel space, before normalization.
                s = float(augment.get("color_jitter_strength", 0.1))
                contrast = 1.0 + float(torch.rand(()) * 2 - 1) * s
                brightness = float(torch.rand(()) * 2 - 1) * s * 255.0
                img = np.clip((img - img.mean()) * contrast + img.mean() + brightness, 0, 255)
            x = torch.from_numpy(to_model_input(np.ascontiguousarray(img[None]), norm_cfg)[0])
            return x, torch.tensor(labels[i]), i

    return _FrameDataset()


def seeded_loader(dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int):
    """DataLoader with a seeded generator and per-worker seeds (reproducible order and augmentation)."""
    import random

    import torch
    from torch.utils.data import DataLoader

    def _init(worker_id: int) -> None:
        ws = torch.initial_seed() % 2**32
        np.random.seed(ws)
        random.seed(ws)

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      worker_init_fn=_init, generator=g, pin_memory=torch.cuda.is_available(),
                      drop_last=False, persistent_workers=False)
