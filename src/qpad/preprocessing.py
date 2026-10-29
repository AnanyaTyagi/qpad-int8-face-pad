"""Deterministic frame selection, alignment, quality statistics and sharded storage.

Storage format (justified in the README): each video becomes one compressed
``.npz`` holding its aligned uint8 RGB crops plus metadata. Videos are grouped
into fixed, deterministic shards of ``shard_size_videos``; each shard is one
uncompressed ``.tar`` (the npz members are already compressed) with a JSON
sidecar and a per-frame CSV. A shard is the unit of resumption: it is built in
``/content``, verified, copied to Drive with a checksum, and only then counted
as done. This keeps Drive at a few hundred files instead of tens of thousands.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .alignment import align_face
from .annotations import parse_eye_annotation, to_zero_based, valid_eye_mask
from .archive_io import read_member_bytes, temporary_member
from .audit import annotation_member, video_member
from .checkpointing import copy_verified
from .reproducibility import file_sha256, write_json
from .video_io import read_frames

ROLE_EVAL = 1
ROLE_TRAIN = 2
ROLE_CALIB = 4
ROLE_NAMES = {"eval": ROLE_EVAL, "train": ROLE_TRAIN, "calib": ROLE_CALIB}


class InsufficientFramesError(ValueError):
    """A video has fewer valid annotated frames than the sampling plan needs."""


# ---------------------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------------------
def select_evenly_spaced(candidates: Sequence[int], n: int) -> list[int]:
    """Pick ``n`` evenly spaced items from sorted ``candidates`` (first and last included).

    Positions are ``round_half_up(linspace(0, len-1, n))``, which is unique
    whenever ``len(candidates) >= n``. Deterministic; no randomness.
    """
    cand = sorted(int(c) for c in candidates)
    if n <= 0:
        return []
    if len(cand) < n:
        raise InsufficientFramesError(f"need {n} frames, only {len(cand)} valid")
    if n == 1:
        return [cand[len(cand) // 2]]
    pos = np.floor(np.linspace(0, len(cand) - 1, n) + 0.5).astype(int)
    out = [cand[p] for p in pos]
    assert len(set(out)) == n, "evenly spaced selection produced duplicates"
    return out


def plan_frames(valid_frame_idx: Sequence[int], split: str, frames_cfg: Mapping[str, Any],
                n_eval: int | None = None) -> dict[int, int]:
    """Return ``{frame_idx: role_bitmask}`` for one video.

    * every split: ``evaluation_per_video`` evenly spaced frames (role EVAL), or ``n_eval``
      for a documented reduced-frame video (then ALL valid frames are used and the count
      must match exactly; not allowed for the training split)
    * train split: ``training_per_video`` evenly spaced frames (role TRAIN) and an
      evenly spaced ``calibration_per_video`` subset of them (role CALIB)
    """
    if frames_cfg.get("training_sampling", "evenly_spaced") != "evenly_spaced":
        raise ValueError(f"unsupported training_sampling {frames_cfg['training_sampling']!r}")
    roles: dict[int, int] = {}
    n_default = int(frames_cfg["evaluation_per_video"])
    if n_eval is not None and n_eval != n_default:
        if split == "train":
            raise ValueError("reduced-frame exceptions are not allowed in the training split")
        if len(valid_frame_idx) != n_eval:
            raise InsufficientFramesError(f"documented {n_eval} valid frames, found {len(valid_frame_idx)}")
    for f in select_evenly_spaced(valid_frame_idx, n_default if n_eval is None else n_eval):
        roles[f] = roles.get(f, 0) | ROLE_EVAL
    if split == "train":
        train = select_evenly_spaced(valid_frame_idx, int(frames_cfg["training_per_video"]))
        for f in train:
            roles[f] = roles.get(f, 0) | ROLE_TRAIN
        for f in select_evenly_spaced(train, int(frames_cfg["calibration_per_video"])):
            roles[f] = roles.get(f, 0) | ROLE_CALIB
    return dict(sorted(roles.items()))


# ---------------------------------------------------------------------------------------
# Pixels
# ---------------------------------------------------------------------------------------
def frame_quality(crop_rgb: np.ndarray, laplacian_ksize: int = 3) -> dict[str, float]:
    """Brightness (mean grey), contrast (grey std) and sharpness (variance of Laplacian)."""
    import cv2

    grey = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(grey, cv2.CV_64F, ksize=laplacian_ksize)
    return {"brightness": float(grey.mean()), "contrast": float(grey.std()), "sharpness": float(lap.var())}


def to_model_input(images_nhwc_uint8: np.ndarray, norm_cfg: Mapping[str, Any]) -> np.ndarray:
    """THE single normalization used by every backend: uint8 NHWC RGB -> float32 NCHW."""
    x = images_nhwc_uint8.astype(np.float32) / float(norm_cfg["scale"])
    mean = np.asarray(norm_cfg["mean"], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.asarray(norm_cfg["std"], dtype=np.float32).reshape(1, 1, 1, 3)
    x = (x - mean) / std
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


@dataclass
class VideoSample:
    video_id: str
    images: np.ndarray          # (n, S, S, 3) uint8 RGB
    frame_idx: np.ndarray       # (n,) 0-based decoded frame indices
    frame_number: np.ndarray    # (n,) original annotation frame numbers
    roles: np.ndarray           # (n,) role bitmask
    eyes: np.ndarray            # (n, 4) image-left x, y, image-right x, y
    matrices: np.ndarray        # (n, 2, 3)
    crop_bounds: np.ndarray     # (n, 4)
    outside_fraction: np.ndarray
    quality: list[dict[str, float]]


def process_video(video_path: Path, ann_table: pd.DataFrame, frame_base: int, n_frames: int, split: str,
                  video_id: str, pre_cfg: Mapping[str, Any], min_eye_distance_px: float,
                  excluded_frames: set[int] | None = None, n_eval: int | None = None) -> VideoSample:
    """Select, decode, align and measure the planned frames of one local video file.

    ``excluded_frames`` (documented damaged frames) are removed from the candidates before
    selection; ``n_eval`` is the documented evaluation-frame count of a reduced-frame video.
    """
    t = to_zero_based(ann_table, frame_base)
    t = t[valid_eye_mask(t, min_eye_distance_px) & (t["frame_idx"] >= 0) & (t["frame_idx"] < n_frames)]
    if excluded_frames:
        t = t[~t["frame_idx"].isin(sorted(excluded_frames))]
    t = t.drop_duplicates("frame_idx", keep="first").set_index("frame_idx")
    plan = plan_frames(t.index.tolist(), split, pre_cfg["frames"], n_eval)
    frames = read_frames(video_path, plan.keys())
    acfg, size = pre_cfg["alignment"], int(pre_cfg["output_size"])
    imgs, eyes, mats, bounds, outside, quality = [], [], [], [], [], []
    for fidx in plan:
        r = t.loc[fidx]
        res = align_face(frames[fidx], (r.x1, r.y1), (r.x2, r.y2), size, acfg["target_image_left_eye_xy"],
                         acfg["target_image_right_eye_xy"], acfg["interpolation"], int(acfg["border_value"]))
        imgs.append(res.crop)
        eyes.append([*res.eye_left, *res.eye_right])
        mats.append(res.matrix)
        bounds.append(res.crop_bounds)
        outside.append(res.outside_fraction)
        quality.append(frame_quality(res.crop, int(pre_cfg["storage"]["laplacian_ksize"])))
    idx = np.array(list(plan.keys()), dtype=np.int64)
    return VideoSample(video_id=video_id, images=np.stack(imgs).astype(np.uint8), frame_idx=idx,
                       frame_number=t.loc[idx, "frame"].to_numpy(dtype=np.int64),
                       roles=np.array(list(plan.values()), dtype=np.int8), eyes=np.array(eyes),
                       matrices=np.stack(mats), crop_bounds=np.array(bounds),
                       outside_fraction=np.array(outside), quality=quality)


def encode_sample(s: VideoSample, compressed: bool = True) -> bytes:
    buf = io.BytesIO()
    save = np.savez_compressed if compressed else np.savez
    save(buf, images=s.images, frame_idx=s.frame_idx, frame_number=s.frame_number, roles=s.roles,
         eyes=s.eyes, matrices=s.matrices, crop_bounds=s.crop_bounds, outside_fraction=s.outside_fraction,
         video_id=np.array(s.video_id))
    return buf.getvalue()


def decode_sample(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sample_frame_rows(s: VideoSample, meta: Mapping[str, Any], shard: str) -> list[dict[str, Any]]:
    rows = []
    for i in range(len(s.frame_idx)):
        rows.append({
            **meta, "shard": shard, "sample_key": f"{s.video_id}.npz", "position": i,
            "frame_idx": int(s.frame_idx[i]), "frame_number": int(s.frame_number[i]),
            "role_mask": int(s.roles[i]), "is_eval": bool(s.roles[i] & ROLE_EVAL),
            "is_train": bool(s.roles[i] & ROLE_TRAIN), "is_calib": bool(s.roles[i] & ROLE_CALIB),
            "eye_left_x": s.eyes[i, 0], "eye_left_y": s.eyes[i, 1],
            "eye_right_x": s.eyes[i, 2], "eye_right_y": s.eyes[i, 3],
            "affine": json.dumps([round(float(v), 8) for v in s.matrices[i].ravel()]),
            "crop_x0": s.crop_bounds[i, 0], "crop_y0": s.crop_bounds[i, 1],
            "crop_x1": s.crop_bounds[i, 2], "crop_y1": s.crop_bounds[i, 3],
            "outside_fraction": float(s.outside_fraction[i]), **s.quality[i],
        })
    return rows


# ---------------------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------------------
def plan_shards(video_ids: Sequence[str], split: str, shard_size: int) -> list[tuple[str, list[str]]]:
    """Deterministic shard assignment: sort IDs, chunk into fixed-size groups."""
    ids = sorted(video_ids)
    return [(f"{split}_shard_{i // shard_size:04d}", ids[i:i + shard_size]) for i in range(0, len(ids), shard_size)]


def write_shard_tar(path: Path, members: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tarfile.open(tmp, "w") as tf:
        for key in sorted(members):
            info = tarfile.TarInfo(name=key)
            info.size = len(members[key])
            info.mtime = 0  # deterministic archive bytes
            tf.addfile(info, io.BytesIO(members[key]))
    os.replace(tmp, path)


def iter_shard(path: Path) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    with tarfile.open(path, "r:") as tf:
        for ti in tf:
            if ti.isfile():
                fh = tf.extractfile(ti)
                assert fh is not None
                yield ti.name, decode_sample(fh.read())


def verify_shard(path: Path, expected_keys: Sequence[str]) -> None:
    """Open every member, decode it, and check the key set matches exactly."""
    seen = []
    for key, arrays in iter_shard(path):
        if arrays["images"].ndim != 4 or arrays["images"].dtype != np.uint8:
            raise IOError(f"{path}:{key} has invalid image array")
        seen.append(key)
    if sorted(seen) != sorted(expected_keys):
        raise IOError(f"{path}: members {sorted(seen)[:3]}... != expected {sorted(expected_keys)[:3]}...")


def shard_is_complete(drive_dir: Path, shard: str, retry_failed: bool) -> bool:
    sidecar = drive_dir / f"{shard}.json"
    if not sidecar.is_file():
        return False
    meta = json.loads(sidecar.read_text())
    tar = drive_dir / f"{shard}.tar"
    if not tar.is_file() or tar.stat().st_size != meta.get("size_bytes"):
        return False
    if file_sha256(tar) != meta.get("sha256"):
        return False
    return not (retry_failed and meta.get("failed"))


def preprocess_split(manifest: pd.DataFrame, split: str, archive_paths: Mapping[str, Path], drive_dir: Path,
                     local_dir: Path, pre_cfg: Mapping[str, Any], dataset_cfg: Mapping[str, Any], frame_base: int,
                     retry_failed: bool = False, logger: Any = None,
                     progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Process every video of ``split`` in ``manifest`` shard by shard (resumable)."""
    vids = manifest[(manifest["split"] == split) & manifest["in_protocol"]].set_index("video_id", drop=False)
    shards = plan_shards(vids.index.tolist(), split, int(pre_cfg["storage"]["shard_size_videos"]))
    drive_dir.mkdir(parents=True, exist_ok=True)
    tmp_videos = local_dir / "videos"
    from .data_exceptions import DataExceptions

    min_eye = float(dataset_cfg["annotations"]["min_eye_distance_px"])
    ex = DataExceptions.from_cfg(dataset_cfg)
    written, skipped, failures = 0, 0, []
    for si, (shard, ids) in enumerate(shards):
        if shard_is_complete(drive_dir, shard, retry_failed):
            skipped += 1
            if progress:
                progress(si + 1, len(shards))
            continue
        members: dict[str, bytes] = {}
        rows: list[dict[str, Any]] = []
        failed: dict[str, str] = {}
        for vid in ids:
            row = vids.loc[vid].to_dict()
            try:
                amember = annotation_member(row)
                if amember is None:
                    raise FileNotFoundError("no annotation file")
                ann = parse_eye_annotation(read_member_bytes(archive_paths[amember.archive], amember)
                                           .decode("utf-8", errors="replace"))
                with temporary_member(archive_paths[row["archive_key"]], video_member(row), tmp_videos) as local:
                    sample = process_video(local, ann.table, frame_base, int(row["frame_count"]), split, vid,
                                           pre_cfg, min_eye, ex.excluded_frames(vid),
                                           ex.reduced_frame_videos.get(vid))
                members[f"{vid}.npz"] = encode_sample(sample, bool(pre_cfg["storage"]["npz_compressed"]))
                meta = {k: row[k] for k in ("video_id", "split", "class", "label", "subject", "phone",
                                            "session", "pai_code")}
                rows.extend(sample_frame_rows(sample, meta, shard))
            except Exception as exc:  # noqa: BLE001 - recorded, written to the sidecar, reported
                failed[vid] = f"{type(exc).__name__}: {exc}"
                if logger:
                    logger.warning("preprocess failure %s: %s", vid, failed[vid])
        local_tar = local_dir / f"{shard}.tar"
        write_shard_tar(local_tar, members)
        verify_shard(local_tar, list(members))
        digest = copy_verified(local_tar, drive_dir / f"{shard}.tar")
        frames_csv = local_dir / f"{shard}.frames.csv"
        pd.DataFrame(rows).to_csv(frames_csv, index=False)
        copy_verified(frames_csv, drive_dir / f"{shard}.frames.csv")
        # The sidecar is written last: its presence marks the shard complete.
        write_json(drive_dir / f"{shard}.json", {
            "shard": shard, "split": split, "videos": ids, "ok": sorted(k[:-4] for k in members),
            "failed": failed, "sha256": digest, "size_bytes": local_tar.stat().st_size, "n_frames": len(rows)})
        local_tar.unlink()
        frames_csv.unlink()
        failures.extend({"video_id": k, "error": v, "shard": shard} for k, v in failed.items())
        written += 1
        if progress:
            progress(si + 1, len(shards))
    return {"split": split, "n_shards": len(shards), "written": written, "skipped_complete": skipped,
            "new_failures": failures}


def collect_split(drive_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate per-shard frame CSVs and sidecar failures for one split."""
    frames, fails = [], []
    for sidecar in sorted(drive_dir.glob(f"{split}_shard_*.json")):
        meta = json.loads(sidecar.read_text())
        csv = drive_dir / f"{meta['shard']}.frames.csv"
        if csv.is_file() and csv.stat().st_size > 1:
            frames.append(pd.read_csv(csv, dtype={"video_id": str}))
        fails.extend({"video_id": k, "error": v, "shard": meta["shard"]} for k, v in meta["failed"].items())
    fdf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return fdf, pd.DataFrame(fails, columns=["video_id", "error", "shard"])


def verify_split_outputs(drive_dir: Path, split: str, expected_ids: Sequence[str], frames_cfg: Mapping[str, Any],
                         full_hash: bool = True, eval_counts: Mapping[str, int] | None = None,
                         excluded: Mapping[str, Sequence[int]] | None = None) -> list[str]:
    """Return a list of problems (empty == verified): hashes, coverage and per-video frame counts."""
    problems: list[str] = []
    frames, fails = collect_split(drive_dir, split)
    for sidecar in sorted(drive_dir.glob(f"{split}_shard_*.json")):
        meta = json.loads(sidecar.read_text())
        tar = drive_dir / f"{meta['shard']}.tar"
        if not tar.is_file():
            problems.append(f"missing {tar.name}")
        elif full_hash and file_sha256(tar) != meta["sha256"]:
            problems.append(f"hash mismatch {tar.name}")
    got = set(frames["video_id"]) if len(frames) else set()
    missing = sorted(set(expected_ids) - got)
    if missing:
        problems.append(f"{len(missing)} videos without processed frames, e.g. {missing[:5]}")
    if len(frames):
        n_eval = frames[frames["is_eval"]].groupby("video_id").size()
        want = n_eval.index.map(lambda v: eval_counts[v] if eval_counts is not None
                                else int(frames_cfg["evaluation_per_video"]))
        bad = n_eval[n_eval.to_numpy() != np.asarray(want)]
        if len(bad):
            problems.append(f"{len(bad)} videos with wrong evaluation-frame count, e.g. {bad.index.tolist()[:5]}")
        for vid, idx in (excluded or {}).items():
            hit = frames[(frames["video_id"] == vid) & frames["frame_idx"].isin(list(idx))]
            if len(hit):
                problems.append(f"{vid}: documented damaged frame(s) {sorted(hit['frame_idx'].unique())} were selected")
        if split == "train":
            n_cal = frames[frames["is_calib"]].groupby("video_id").size()
            if (n_cal != int(frames_cfg["calibration_per_video"])).any():
                problems.append("wrong calibration-frame count for some train videos")
            n_tr = frames[frames["is_train"]].groupby("video_id").size()
            if (n_tr != int(frames_cfg["training_per_video"])).any():
                problems.append("wrong training-frame count for some train videos")
    if len(fails):
        problems.append(f"{len(fails)} failed videos recorded in sidecars")
    return problems
