"""FP32 DeepPixBiS training with resumable checkpoints and dev-set model selection."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .checkpointing import copy_verified
from .reproducibility import file_sha256, write_json

BEST_NAME = "deeppixbis_best_fp32.pt"
LAST_NAME = "deeppixbis_last_fp32.pt"
FROZEN_SUFFIX = ".FROZEN.json"


@dataclass
class TrainState:
    epoch: int = 0                       # last completed epoch (0 = none)
    best_dev_loss: float = float("inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def build_optimizer(model, opt_cfg: Mapping[str, Any]):
    import torch

    if opt_cfg["name"].lower() != "adam":
        raise ValueError(f"unsupported optimizer {opt_cfg['name']}")
    return torch.optim.Adam(model.parameters(), lr=float(opt_cfg["lr"]), weight_decay=float(opt_cfg["weight_decay"]))


def _rng_state() -> dict[str, Any]:
    import torch

    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: Mapping[str, Any]) -> None:
    import torch

    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def run_epoch(model, loader, loss_fn, device: str, optimizer=None, scaler=None, amp: bool = False,
              max_batches: int | None = None, progress: Callable[[int, int], None] | None = None
              ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """One pass over ``loader``. Trains if ``optimizer`` is given, else evaluates.

    Returns mean losses plus per-sample frame scores (map mean) and dataset row indices.
    """
    import torch

    from .deeppixbis import map_to_frame_score

    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "pix_loss": 0.0, "bin_loss": 0.0}
    n_seen = 0
    scores, rows = [], []
    n_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for b, (x, y, idx) in enumerate(loader):
            if b >= n_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.startswith("cuda")):
                out_map, out_bin = model(x)
            loss, l_pix, l_bin = loss_fn(out_map, out_bin, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            bs = x.shape[0]
            n_seen += bs
            totals["loss"] += float(loss.detach()) * bs
            totals["pix_loss"] += float(l_pix.detach()) * bs
            totals["bin_loss"] += float(l_bin.detach()) * bs
            scores.append(map_to_frame_score(out_map.detach().float()).cpu().numpy())
            rows.append(idx.numpy())
            if progress:
                progress(b + 1, n_batches)
    means = {k: v / max(n_seen, 1) for k, v in totals.items()}
    return means, np.concatenate(scores) if scores else np.zeros(0), np.concatenate(rows) if rows else np.zeros(0, int)


def save_checkpoint_to_drive(obj: Mapping[str, Any], local_path: Path, drive_path: Path) -> str:
    import torch

    local_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(obj), local_path)
    return copy_verified(local_path, drive_path)


def fit(model, train_loader, dev_loader, dev_index: pd.DataFrame, cfg: Mapping[str, Any], local_dir: Path,
        drive_models_dir: Path, device: str, logger: Any = None, max_batches: int | None = None,
        epochs_override: int | None = None, progress_factory: Callable[[str], Callable] | None = None) -> TrainState:
    """Train with dev-loss model selection, early stopping and per-epoch resumable checkpoints.

    ``dev_index`` is the FrameStore index of the dev frames (for video-level diagnostics).
    The test split is never passed to this function.
    """
    import torch

    from .deeppixbis import DeepPixBiSLoss
    from .metrics import auroc

    best_drive = drive_models_dir / BEST_NAME
    if (best_drive.with_name(BEST_NAME + FROZEN_SUFFIX)).exists():
        raise RuntimeError(f"{best_drive} is FROZEN; training would modify the frozen baseline.")
    rt, sched = cfg["runtime"], cfg["schedule"]
    epochs = int(epochs_override or sched["epochs"])
    loss_fn = DeepPixBiSLoss(cfg["loss"]["beta"])
    optimizer = build_optimizer(model, cfg["optimizer"])
    amp = bool(rt.get("amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.startswith("cuda"))
    state = TrainState()
    last_drive = drive_models_dir / LAST_NAME
    if last_drive.is_file():
        ck = torch.load(last_drive, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        if ck.get("scaler_state"):
            scaler.load_state_dict(ck["scaler_state"])
        state = TrainState(**ck["train_state"])
        _set_rng_state(ck["rng_state"])
        gen = getattr(train_loader, "generator", None)
        if gen is not None and ck.get("loader_generator_state") is not None:
            gen.set_state(ck["loader_generator_state"])
        if logger:
            logger.info("resumed from %s at epoch %d (best epoch %d, dev loss %.5f)", last_drive, state.epoch,
                        state.best_epoch, state.best_dev_loss)
    patience = int(sched["early_stopping_patience"])
    labels_dev = dev_index["label"].to_numpy()
    while state.epoch < epochs and state.epochs_without_improvement < patience:
        ep = state.epoch + 1
        t0 = time.time()
        pf = progress_factory(f"epoch {ep} train") if progress_factory else None
        tr, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler, amp, max_batches, pf)
        pf = progress_factory(f"epoch {ep} dev") if progress_factory else None
        dv, dscores, drows = run_epoch(model, dev_loader, loss_fn, device, None, None, amp, max_batches, pf)
        vid = pd.DataFrame({"video_id": dev_index["video_id"].to_numpy()[drows], "score": dscores,
                            "label": labels_dev[drows]}).groupby("video_id").agg(score=("score", "mean"),
                                                                                label=("label", "first"))
        dev_auc = auroc(vid["score"].to_numpy(), vid["label"].to_numpy() == 1) if vid["label"].nunique() == 2 else float("nan")
        improved = dv["loss"] < state.best_dev_loss
        row = {"epoch": ep, "train_loss": tr["loss"], "train_pix_loss": tr["pix_loss"], "train_bin_loss": tr["bin_loss"],
               "dev_loss": dv["loss"], "dev_pix_loss": dv["pix_loss"], "dev_bin_loss": dv["bin_loss"],
               "dev_video_auroc": dev_auc, "seconds": time.time() - t0, "improved": bool(improved)}
        state.history.append(row)
        state.epoch = ep
        if improved:
            state.best_dev_loss, state.best_epoch, state.epochs_without_improvement = dv["loss"], ep, 0
            save_checkpoint_to_drive({"model_state": model.state_dict(), "epoch": ep, "dev_loss": dv["loss"],
                                      "config": dict(cfg)}, local_dir / BEST_NAME, best_drive)
        else:
            state.epochs_without_improvement += 1
        save_checkpoint_to_drive({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                                  "scaler_state": scaler.state_dict(), "train_state": state.__dict__,
                                  "rng_state": _rng_state(), "config": dict(cfg),
                                  # the DataLoader's private generator drives shuffling and worker seeds
                                  "loader_generator_state": (train_loader.generator.get_state()
                                                             if getattr(train_loader, "generator", None) is not None
                                                             else None)},
                                 local_dir / LAST_NAME, last_drive)
        if logger:
            logger.info("epoch %d: train %.5f dev %.5f dev-AUROC %.4f %s", ep, tr["loss"], dv["loss"], dev_auc,
                        "(best)" if improved else "")
    return state


def freeze_checkpoint(best_path: Path, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Record the SHA-256 of the selected checkpoint; later stages verify it before use."""
    info = {"checkpoint": best_path.name, "sha256": file_sha256(best_path), "frozen": True, **dict(extra or {})}
    write_json(best_path.with_name(best_path.name + FROZEN_SUFFIX), info)
    return info


def load_frozen_model(best_path: Path, device: str = "cpu"):
    """Load the frozen FP32 checkpoint (hash verified), in eval mode with gradients disabled."""
    import torch

    from .deeppixbis import DeepPixBiS

    frozen = best_path.with_name(best_path.name + FROZEN_SUFFIX)
    if not frozen.is_file():
        raise FileNotFoundError(f"{frozen} missing: freeze the checkpoint in notebook 04 first")
    expected = json.loads(frozen.read_text())["sha256"]
    actual = file_sha256(best_path)
    if actual != expected:
        raise RuntimeError(f"checkpoint hash {actual} != frozen hash {expected}")
    model = DeepPixBiS(pretrained=False)
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, expected
