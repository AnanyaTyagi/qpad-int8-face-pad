"""Stage-level orchestration used by the notebooks (kept here so it is testable).

DRY_RUN convention: every stage run with ``dry_run=True`` writes ONLY under
``<PROJECT_DRIVE_ROOT>/_dry_run/`` and reads its inputs from there first,
falling back to the full-run location. A dry run can therefore never overwrite
or be mistaken for a full result.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from .archive_io import (MemberRecord, check_archives_exist, classify_member, get_or_build_index, read_member_bytes,
                         stem_pairing, summarize_members)
from .audit import merge_manifest, normalize_manifest_types, run_audit, write_audit_reports
from .config import ProjectPaths, disk_free_gb
from .manifest import (build_annotation_lookup, build_video_table, parse_protocol_text, protocol_file_split,
                       validate_audit, validate_structure, overall_status)
from .reproducibility import write_json

DRY_RUN_DIRNAME = "_dry_run"
MANIFEST_CSV = "oulu_manifest.csv"


# ---------------------------------------------------------------------------------------
# Dry-run aware paths
# ---------------------------------------------------------------------------------------
def output_paths(paths: ProjectPaths, dry_run: bool) -> ProjectPaths:
    """Where a stage writes: the real project root, or ``<root>/_dry_run``."""
    if not dry_run:
        return paths
    return dataclasses.replace(paths, project_drive_root=paths.project_drive_root / DRY_RUN_DIRNAME,
                               local_work_root=paths.local_work_root / DRY_RUN_DIRNAME)


def find_input(paths: ProjectPaths, dry_run: bool, attr: str, relative: str) -> Path:
    """Locate an input produced by an earlier stage (dry-run copy first when in dry-run mode)."""
    candidates = [getattr(output_paths(paths, True), attr) / relative] if dry_run else []
    candidates.append(getattr(paths, attr) / relative)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"required input not found; looked in: {[str(c) for c in candidates]}")


def index_path(paths: ProjectPaths, key: str) -> Path:
    return paths.manifests / f"archive_index_{key}.json.gz"


NESTED_DIRNAME = "nested_archives"
SPLIT_KEYS = ("train", "dev", "test")


def nested_archive_paths(paths: ProjectPaths) -> dict[str, Path]:
    """Small archives that were found inside the metadata ZIP and extracted (e.g. ``Protocols.tar``)."""
    d = paths.manifests / NESTED_DIRNAME
    return {f"nested:{p.name}": p for p in sorted(d.glob("*.tar"))} if d.is_dir() else {}


def all_archive_paths(paths: ProjectPaths) -> dict[str, Path]:
    return {**{k: paths.archive(k) for k in paths.archive_names}, **nested_archive_paths(paths)}


def expand_nested_protocol_archives(paths: ProjectPaths, metadata_records: list[MemberRecord],
                                    dataset_cfg: Mapping[str, Any]) -> dict[str, Path]:
    """Extract TAR members of the metadata ZIP whose name mentions a protocol token.

    The real OULU-NPU metadata ZIP nests ``Protocols.tar`` (and ``Baseline.tar``,
    which holds precomputed baseline features and is not needed). Only the
    protocol archive is extracted, into ``manifests/nested_archives/``.
    """
    from .archive_io import extract_member

    tokens = [t.lower() for t in dataset_cfg["discovery"]["protocol_path_tokens"]]
    d = paths.manifests / NESTED_DIRNAME
    for rec in metadata_records:
        if rec.is_file and rec.suffix == ".tar" and any(t in rec.name.lower() for t in tokens):
            dst = d / rec.basename
            if not (dst.is_file() and dst.stat().st_size == rec.size):
                tmp = extract_member(paths.archive("metadata"), rec, d)
                tmp.replace(dst)
    return nested_archive_paths(paths)


# ---------------------------------------------------------------------------------------
# Notebook 00: inventory
# ---------------------------------------------------------------------------------------
def archive_inventory(paths: ProjectPaths, dataset_cfg: Mapping[str, Any], force: bool = False,
                      logger: Any = None) -> dict[str, Any]:
    """Index all four archives (headers only), summarise layout and save ``archive_inventory.json``."""
    archives = all_archive_paths(paths)
    check_archives_exist(archives)
    disc = dataset_cfg["discovery"]
    exts = (disc["video_extensions"], disc["annotation_extensions"], disc["protocol_path_tokens"])
    inv: dict[str, Any] = {"archives": {}, "local_free_gb": disk_free_gb(paths.local_work_root)}
    indexes: dict[str, list[MemberRecord]] = {}
    meta_recs = get_or_build_index(archives["metadata"], index_path(paths, "metadata"), force=force)
    archives.update(expand_nested_protocol_archives(paths, meta_recs, dataset_cfg))
    for key, path in archives.items():
        if logger:
            logger.info("indexing %s (%s)", key, path)
        recs = get_or_build_index(path, index_path(paths, key.replace(":", "_")), force=force)
        indexes[key] = recs
        inv["archives"][key] = {"path": str(path), "compressed_bytes": path.stat().st_size,
                                **summarize_members(recs, *exts), "stem_pairing": stem_pairing(recs, *exts)}
    proto = protocol_files(indexes, dataset_cfg)
    inv["protocol_files"] = {k: [r.name for _, r in v] for k, v in proto.items()}
    lookup = build_annotation_lookup({k: indexes[k] for k in SPLIT_KEYS}, dataset_cfg)
    vids = sum(inv["archives"][k]["kinds"].get("video", 0) for k in ("train", "dev", "test"))
    inv["discovery"] = {
        "video_members_in_split_archives": vids,
        "annotation_stems_anywhere": len(lookup),
        "annotation_locations": sorted({k for v in lookup.values() for k, _ in v}),
        "total_uncompressed_bytes": int(sum(a["total_uncompressed_bytes"] for a in inv["archives"].values())),
    }
    write_json(paths.manifests / "archive_inventory.json", inv)
    return inv


def protocol_files(indexes: Mapping[str, list[MemberRecord]], dataset_cfg: Mapping[str, Any]
                   ) -> dict[str, list[tuple[str, MemberRecord]]]:
    """Protocol list files of the configured protocol, grouped by split."""
    disc = dataset_cfg["discovery"]
    name = dataset_cfg["protocol"]["name"]
    out: dict[str, list[tuple[str, MemberRecord]]] = {}
    for key, recs in indexes.items():
        for r in recs:
            if classify_member(r, disc["video_extensions"], disc["annotation_extensions"],
                               disc["protocol_path_tokens"]) != "protocol":
                continue
            split = protocol_file_split(r.name, name)
            if split:
                out.setdefault(split, []).append((key, r))
    return out


def load_indexes(paths: ProjectPaths) -> dict[str, list[MemberRecord]]:
    return {k: get_or_build_index(p, index_path(paths, k.replace(":", "_"))) for k, p in all_archive_paths(paths).items()}


# ---------------------------------------------------------------------------------------
# Notebook 01: audit
# ---------------------------------------------------------------------------------------
def dry_run_subset(videos: pd.DataFrame, n_per_class: int) -> pd.DataFrame:
    """Deterministic subset: the first ``n_per_class`` protocol videos (by ID) of each split x class."""
    proto = videos[videos["in_protocol"]].sort_values("video_id")
    return proto.groupby(["split", "class"], group_keys=False).head(n_per_class).reset_index(drop=True)


def build_video_manifest(paths: ProjectPaths, dataset_cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, dict]:
    """Index-derived per-video table (all archives) and the official protocol lists."""
    indexes = load_indexes(paths)
    lookup = build_annotation_lookup({k: indexes[k] for k in SPLIT_KEYS}, dataset_cfg)
    videos = build_video_table({k: indexes[k] for k in ("train", "dev", "test")}, dataset_cfg, lookup)
    archives = all_archive_paths(paths)
    lists: dict[str, list[str]] = {}
    for split, files in protocol_files(indexes, dataset_cfg).items():
        for key, rec in files:
            lists.setdefault(split, []).extend(
                parse_protocol_text(read_member_bytes(archives[key], rec).decode("utf-8", errors="replace")))
    return videos, lists


def run_dataset_audit(paths: ProjectPaths, dataset_cfg: Mapping[str, Any], dry_run: bool, n_per_class: int,
                      retry_failed: bool = False, logger: Any = None,
                      progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Audit Protocol videos (all, or a dry-run subset) and write manifest + reports."""
    out = output_paths(paths, dry_run)
    out.ensure_output_dirs()
    acfg = dataset_cfg["audit"]
    videos, lists = build_video_manifest(paths, dataset_cfg)
    structure = validate_structure(videos, dataset_cfg, lists or None)
    proto = videos[videos["in_protocol"]]
    target = dry_run_subset(videos, n_per_class) if dry_run else proto
    records = run_audit(target, all_archive_paths(paths), out.local_tmp / "audit", out.manifests / "audit_progress.csv",
                        dataset_cfg, bool(acfg["full_decode"]), bool(acfg["compute_checksum"]), int(acfg["workers"]),
                        int(acfg["checkpoint_every"]), retry_failed, progress, logger)
    manifest = merge_manifest(videos, records)
    audited = manifest[manifest["in_protocol"]]
    content = validate_audit(audited, dataset_cfg, bool(acfg["full_decode"]), int(len(proto)))
    checks = structure + content
    manifest.to_csv(out.manifests / MANIFEST_CSV, index=False)
    try:
        manifest.to_parquet(out.manifests / "oulu_manifest.parquet", index=False)
    except ImportError as exc:  # pyarrow missing: CSV remains authoritative
        if logger:
            logger.warning("parquet not written: %s", exc)
    context = {"dry_run": dry_run, "n_audited": int(len(records)), "full_decode": bool(acfg["full_decode"]),
               "checksum": bool(acfg["compute_checksum"]), "dataset_config": dataset_cfg.get("_source")}
    summary = write_audit_reports(checks, manifest, out.reports / "dataset_audit.json", out.reports / "dataset_audit.md",
                                  context, int(acfg["min_valid_annotated_frames"]), dataset_cfg)
    return {"manifest": manifest, "checks": checks, "summary": summary, "status": overall_status(checks),
            "output_root": str(out.project_drive_root)}


def load_verified_manifest(paths: ProjectPaths, dry_run: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the audit manifest and report; a full run requires the audit to have PASSED."""
    man = find_input(paths, dry_run, "manifests", MANIFEST_CSV)
    rep = man.parent.parent / "reports" / "dataset_audit.json"
    summary = json.loads(rep.read_text())
    if not dry_run and summary["overall_status"] != "PASS":
        raise RuntimeError(f"dataset audit status is {summary['overall_status']}; a full run requires PASS. "
                           f"Inspect {rep.with_suffix('.md')}.")
    df = normalize_manifest_types(pd.read_csv(man, dtype={"video_id": str}))
    return df, summary


# ---------------------------------------------------------------------------------------
# Notebook 02: preprocessing
# ---------------------------------------------------------------------------------------
def preprocessing_targets(manifest: pd.DataFrame, split: str, dry_run: bool, n_per_class: int,
                          dataset_cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Videos to preprocess for ``split`` (all Protocol videos, or the dry-run subset)."""
    proto = manifest[manifest["in_protocol"]]
    target = dry_run_subset(manifest, n_per_class) if dry_run else proto
    target = target[target["split"] == split]
    from .data_exceptions import DataExceptions

    status = target.get("status", pd.Series("ok", index=target.index))
    ok_damaged = target["video_id"].isin(list(DataExceptions.from_cfg(dataset_cfg).damaged_frames)) \
        if dataset_cfg is not None else False
    not_ok = target[(status != "ok") & ~ok_damaged]
    if len(not_ok):
        raise RuntimeError(f"{len(not_ok)} {split} videos have no successful audit record "
                           f"(e.g. {not_ok['video_id'].tolist()[:5]}); resolve them in notebook 01 first.")
    return target


def run_preprocessing(paths: ProjectPaths, cfgs: Mapping[str, Any], split: str, dry_run: bool, n_per_class: int,
                      repo_root: Path, archive_overrides: Mapping[str, Path] | None = None,
                      retry_failed: bool = False, logger: Any = None,
                      progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Preprocess one split shard-by-shard and write its processed manifest and metadata."""
    from .annotations import resolve_dataset_frame_base
    from .preprocessing import collect_split, preprocess_split, verify_split_outputs
    from .reproducibility import write_experiment_metadata

    dcfg, pcfg = cfgs["dataset"], cfgs["preprocessing"]
    out = output_paths(paths, dry_run)
    out.ensure_output_dirs()
    manifest, _ = load_verified_manifest(paths, dry_run)
    target = preprocessing_targets(manifest, split, dry_run, n_per_class, dcfg)
    base = resolve_dataset_frame_base(dcfg["annotations"]["frame_index_base"],
                                      manifest.loc[manifest["in_protocol"] & manifest["status"].eq("ok"), "ann_frame_base"])
    archives = {**all_archive_paths(paths), **dict(archive_overrides or {})}
    drive_dir = out.processed / split
    local_dir = out.local_work_root / "preprocess" / split
    res = preprocess_split(target, split, archives, drive_dir, local_dir, pcfg, dcfg, base, retry_failed, logger, progress)
    frames, fails = collect_split(drive_dir, split)
    frames.to_csv(out.processed / f"processed_manifest_{split}.csv", index=False)
    fails.to_csv(out.processed / f"processed_failures_{split}.csv", index=False)
    from .data_exceptions import DataExceptions, expected_eval_counts

    problems = verify_split_outputs(drive_dir, split, target["video_id"].tolist(), pcfg["frames"],
                                    eval_counts=expected_eval_counts(cfgs),
                                    excluded=DataExceptions.from_cfg(dcfg).damaged_frames)
    write_experiment_metadata(out.processed / f"preprocessing_{split}_metadata.json", f"preprocess_{split}", repo_root,
                              config={"preprocessing": pcfg, "annotation_frame_base": base},
                              inputs={"manifest": find_input(paths, dry_run, "manifests", MANIFEST_CSV)},
                              extra={"dry_run": dry_run, "n_videos": int(len(target)), "problems": problems,
                                     "result": {k: v for k, v in res.items() if k != "new_failures"}})
    return {**res, "frame_base": base, "n_frames": int(len(frames)), "n_failed": int(len(fails)),
            "problems": problems, "verified": not problems, "drive_dir": str(drive_dir)}


def copy_archive_locally(src: Path, local_dir: Path, margin_gb: float = 5.0) -> Path:
    """Optional faster mode: copy ONE split archive to /content (size-verified)."""
    import shutil

    from .config import require_free_space

    need = src.stat().st_size / 1e9 + margin_gb
    require_free_space(local_dir, need)
    local_dir.mkdir(parents=True, exist_ok=True)
    dst = local_dir / src.name
    if not (dst.is_file() and dst.stat().st_size == src.stat().st_size):
        tmp = dst.with_name(dst.name + ".partial")
        shutil.copyfile(src, tmp)
        if tmp.stat().st_size != src.stat().st_size:
            raise IOError(f"size mismatch copying {src}")
        tmp.replace(dst)
    return dst


# ---------------------------------------------------------------------------------------
# Notebook 03: EDA
# ---------------------------------------------------------------------------------------
PIXEL_EDA_SPLITS = ("train", "dev")  # test pixels are not summarised, to keep test data out of design decisions


def run_eda(paths: ProjectPaths, dry_run: bool) -> dict[str, Any]:
    """Generate all EDA figures and tables under ``results/eda``."""
    from . import plotting as P

    out = output_paths(paths, dry_run)
    eda_dir = out.results / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    manifest, audit = load_verified_manifest(paths, dry_run)
    proto = manifest[manifest["in_protocol"]].copy()
    figs, skipped = [], []
    figs.append(P.fig_class_balance(proto, eda_dir))
    figs.append(P.fig_subjects_per_split(proto, eda_dir))
    figs.append(P.fig_by_factor(proto, "phone", "Phone", eda_dir))
    figs.append(P.fig_by_factor(proto, "session", "Session", eda_dir))
    # Every video whose properties were measured, including documented exceptions such as 6_1_12_1 (one damaged
    # frame, status != "ok" but kept in the study).
    audited = proto[proto["frame_count"].notna()] if "frame_count" in proto else proto.iloc[:0]
    if len(audited):
        figs.append(P.fig_hist_by_class(audited, "frame_count", "Frames per video", "Video length in frames by class",
                                        "frame_count_distribution",
                                        "Distribution of decoded frame counts per video for every Protocol II "
                                        "video.", eda_dir))
        figs.append(P.fig_hist_by_class(audited, "duration_s", "Duration (s)", "Video duration by class",
                                        "duration_distribution",
                                        "Distribution of video durations reported by ffprobe for each class.", eda_dir))
        res = (audited["width"].astype("Int64").astype(str) + "x" + audited["height"].astype("Int64").astype(str))
        res_tab = res.groupby([audited["split"], audited["class"], res]).size().rename("videos").reset_index()
        res_tab.columns = ["split", "class", "resolution", "videos"]
        res_tab.to_csv(eda_dir / "resolution_table.csv", index=False)
    else:
        skipped.append("frame/duration/resolution: no audited videos")
    fail_counts = {c["name"]: c["n_violations"] for c in audit["checks"] if c["status"] != "not_checked"}
    figs.append(P.fig_failures(fail_counts, eda_dir))
    tab = proto.groupby(["split", "class"]).size().unstack(fill_value=0)
    tab["total"] = tab.sum(axis=1)
    tab.to_csv(eda_dir / "class_counts.csv")
    proto.groupby(["split", "phone", "session"]).size().rename("videos").reset_index().to_csv(
        eda_dir / "phone_session_counts.csv", index=False)
    # Pixel statistics: per-video means over the 20 evaluation crops (one point per independent video).
    pix = []
    for split in PIXEL_EDA_SPLITS:
        p = out.processed / f"processed_manifest_{split}.csv"
        if not p.is_file():
            p = paths.processed / f"processed_manifest_{split}.csv"
        if p.is_file() and p.stat().st_size > 1:
            f = pd.read_csv(p, dtype={"video_id": str})
            f = f[f["is_eval"]]
            pix.append(f.groupby(["video_id", "split", "class"], as_index=False)[
                ["brightness", "contrast", "sharpness"]].mean())
        else:
            skipped.append(f"pixel statistics for {split}: processed manifest not found (run notebook 02)")
    if pix:
        v = pd.concat(pix, ignore_index=True)
        v.to_csv(eda_dir / "per_video_quality.csv", index=False)
        v.groupby("class")[["brightness", "contrast", "sharpness"]].describe().to_csv(eda_dir / "quality_by_class.csv")
        note = " (train + development videos; test excluded)"
        figs.append(P.fig_box_by_class(v, "brightness", "Mean grey level (0-255)", "Face-crop brightness by class" ,
                                       "brightness_by_class", "Per-video mean brightness of aligned face crops"
                                       + note + "; class differences show cues a detector could exploit.", eda_dir))
        figs.append(P.fig_box_by_class(v, "contrast", "Grey-level standard deviation", "Face-crop contrast by class",
                                       "contrast_by_class", "Per-video mean contrast of aligned face crops" + note + ".",
                                       eda_dir))
        figs.append(P.fig_box_by_class(v, "sharpness", "Variance of Laplacian (log scale)",
                                       "Face-crop sharpness by class", "sharpness_by_class",
                                       "Per-video mean sharpness of aligned face crops" + note +
                                       "; recapture through print or screen often changes high-frequency detail.",
                                       eda_dir, log_y=True))
        figs.append(P.fig_scatter_by_class(v, "brightness", "sharpness", "Mean grey level (0-255)",
                                           "Variance of Laplacian (log scale)", "Brightness versus sharpness by class",
                                           "brightness_vs_sharpness", "Each point is one video" + note +
                                           "; overlap between classes indicates that simple image statistics alone "
                                           "cannot separate attacks from bona fide presentations.", eda_dir, log_y=True))
    return {"figures": figs, "skipped": skipped, "eda_dir": str(eda_dir)}


# ---------------------------------------------------------------------------------------
# Notebook 04: training
# ---------------------------------------------------------------------------------------
def processed_split_dir(paths: ProjectPaths, dry_run: bool, split: str) -> Path:
    return find_input(paths, dry_run, "processed", split)


def run_training(paths: ProjectPaths, cfgs: Mapping[str, Any], dry_run: bool, device: str, repo_root: Path,
                 max_batches: int | None = None, epochs_override: int | None = None, logger: Any = None,
                 progress_factory: Callable[[str], Callable] | None = None) -> dict[str, Any]:
    """Train DeepPixBiS on the train split, selecting the epoch on development loss."""
    import torch

    from .config import set_global_seed
    from .datasets import make_torch_dataset, materialize, seeded_loader
    from .deeppixbis import DeepPixBiS
    from .plotting import fig_training_curves
    from .reproducibility import write_experiment_metadata
    from .training import fit

    tcfg, pcfg = cfgs["training"], cfgs["preprocessing"]
    out = output_paths(paths, dry_run)
    out.ensure_output_dirs()
    if dry_run:  # a dry run always starts fresh (only files under _dry_run are touched)
        for f in out.models.glob("deeppixbis_*_fp32.pt*"):
            f.unlink()
    seed = int(tcfg["runtime"]["seed"])
    seed_info = set_global_seed(seed, bool(tcfg["runtime"]["deterministic"]))
    train_store = materialize(processed_split_dir(paths, dry_run, "train"), "train", "train", out.local_cache)
    dev_store = materialize(processed_split_dir(paths, dry_run, "dev"), "dev", "eval", out.local_cache)
    if set(train_store.index["split"]) != {"train"} or set(dev_store.index["split"]) != {"dev"}:
        raise RuntimeError("split contamination: training data must be train-only and validation dev-only")
    norm = pcfg["normalization"]
    dcfg = tcfg["data"]
    train_ds = make_torch_dataset(train_store, norm, dcfg["augmentation"])
    dev_ds = make_torch_dataset(dev_store, norm, None)
    train_loader = seeded_loader(train_ds, int(dcfg["batch_size"]), True, seed, int(dcfg["num_workers"]))
    dev_loader = seeded_loader(dev_ds, int(dcfg["batch_size"]), False, seed, int(dcfg["num_workers"]))
    model = DeepPixBiS(pretrained=bool(tcfg["model"]["pretrained_imagenet"])).to(device)
    write_json(out.models / "training_config.json", {"training": tcfg, "preprocessing": pcfg, "seed_info": seed_info,
                                                     "device": device, "dry_run": dry_run,
                                                     "n_train_frames": len(train_store), "n_dev_frames": len(dev_store),
                                                     "max_batches": max_batches, "epochs_override": epochs_override})
    state = fit(model, train_loader, dev_loader, dev_store.index, tcfg, out.local_work_root / "models", out.models,
                device, logger, max_batches, epochs_override, progress_factory)
    hist = pd.DataFrame(state.history)
    out.results.mkdir(parents=True, exist_ok=True)
    hist.to_csv(out.results / "training_history.csv", index=False)
    fig = fig_training_curves(hist, out.results) if len(hist) else None
    write_experiment_metadata(out.models / "training_metadata.json", "train_fp32", repo_root, config=dict(tcfg),
                              seeds={"training": seed},
                              inputs={"best_checkpoint": out.models / "deeppixbis_best_fp32.pt"},
                              extra={"best_epoch": state.best_epoch, "best_dev_loss": state.best_dev_loss,
                                     "epochs_run": state.epoch, "torch": torch.__version__})
    return {"state": state, "history": hist, "figure": fig, "models_dir": str(out.models)}


# ---------------------------------------------------------------------------------------
# Notebook 05: threshold + FP32 evaluation
# ---------------------------------------------------------------------------------------
def score_split(model, paths: ProjectPaths, cfgs: Mapping[str, Any], dry_run: bool, split: str, device: str,
                progress: Callable[[int, int], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frame and video scores of the 20 saved evaluation frames per video."""
    from .datasets import materialize
    from .scoring import aggregate_video_scores, score_store, torch_backend

    out = output_paths(paths, dry_run)
    store = materialize(processed_split_dir(paths, dry_run, split), split, "eval", out.local_cache)
    frames = score_store(store, torch_backend(model, device), cfgs["preprocessing"]["normalization"], 64, progress)
    from .data_exceptions import expected_eval_counts

    videos = aggregate_video_scores(frames, expected_eval_counts(cfgs))
    return frames, videos


def select_fp32_threshold(paths: ProjectPaths, cfgs: Mapping[str, Any], dry_run: bool, device: str, repo_root: Path,
                          progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Score the development split and freeze the EER threshold (development data only)."""
    from .plotting import fig_score_hist, fig_threshold_curve
    from .reproducibility import file_sha256, write_experiment_metadata
    from .thresholds import select_eer_threshold
    from .training import BEST_NAME, load_frozen_model

    out = output_paths(paths, dry_run)
    res_dir = out.results / "fp32"
    res_dir.mkdir(parents=True, exist_ok=True)
    model, ck_hash = load_frozen_model(find_input(paths, dry_run, "models", BEST_NAME), device)
    frames, videos = score_split(model, paths, cfgs, dry_run, "dev", device, progress)
    frames.to_csv(res_dir / "dev_frame_scores.csv", index=False)
    videos.to_csv(res_dir / "dev_video_scores.csv", index=False)
    thr, table = select_eer_threshold(videos["score"].to_numpy(), (videos["class"] == "bona_fide").to_numpy())
    table.to_csv(res_dir / "dev_threshold_candidates.csv", index=False)
    info = {**thr.as_dict(), "rule": "min |APCER-BPCER|; tie -> min ACER; tie -> higher threshold",
            "decision": "score >= threshold -> bona fide", "split": "dev", "checkpoint_sha256": ck_hash,
            "dev_video_scores_sha256": file_sha256(res_dir / "dev_video_scores.csv"), "dry_run": dry_run}
    write_json(res_dir / "threshold.json", info)
    figs = [fig_threshold_curve(table, thr.threshold, res_dir),
            fig_score_hist(videos, thr.threshold, "Development", res_dir, "dev_score_hist")]
    write_experiment_metadata(res_dir / "threshold_metadata.json", "fp32_threshold", repo_root,
                              inputs={"checkpoint": find_input(paths, dry_run, "models", BEST_NAME),
                                      "dev_scores": res_dir / "dev_video_scores.csv"}, extra=info)
    return {"threshold": info, "table": table, "videos": videos, "figures": figs}


def evaluate_fp32_test(paths: ProjectPaths, cfgs: Mapping[str, Any], dry_run: bool, device: str, repo_root: Path,
                       force: bool = False, progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Evaluate the test split ONCE with the frozen development threshold."""
    from .metrics import error_breakdown, pad_metrics
    from .plotting import fig_roc, fig_score_hist
    from .reproducibility import file_sha256, write_experiment_metadata
    from .training import BEST_NAME, load_frozen_model

    out = output_paths(paths, dry_run)
    res_dir = out.results / "fp32"
    thr_path = res_dir / "threshold.json"
    if not thr_path.is_file():
        raise FileNotFoundError("threshold.json missing: select the development threshold first")
    thr = json.loads(thr_path.read_text())
    lock = res_dir / "test_evaluation_lock.json"
    thr_hash = file_sha256(thr_path)
    if lock.is_file() and not force and not dry_run:
        prev = json.loads(lock.read_text())
        if prev["threshold_sha256"] != thr_hash:
            raise RuntimeError("the test set was already evaluated with a DIFFERENT threshold; re-selecting the "
                               "threshold after seeing test results is not allowed. Inspect test_evaluation_lock.json.")
    model, ck_hash = load_frozen_model(find_input(paths, dry_run, "models", BEST_NAME), device)
    if ck_hash != thr["checkpoint_sha256"]:
        raise RuntimeError("checkpoint differs from the one used to select the threshold")
    frames, videos = score_split(model, paths, cfgs, dry_run, "test", device, progress)
    frames.to_csv(res_dir / "test_frame_scores.csv", index=False)
    videos.to_csv(res_dir / "test_video_scores.csv", index=False)
    t = float(thr["threshold"])
    metrics = {"test": pad_metrics(videos, t), "dev": pad_metrics(pd.read_csv(res_dir / "dev_video_scores.csv",
                                                                              dtype={"video_id": str}), t),
               "threshold": thr, "checkpoint_sha256": ck_hash, "dry_run": dry_run}
    write_json(res_dir / "metrics.json", metrics)
    for by in (["phone"], ["session"], ["phone", "session"]):
        error_breakdown(videos, t, by).to_csv(res_dir / f"test_errors_by_{'_'.join(by)}.csv", index=False)
    figs = [fig_score_hist(videos, t, "Test", res_dir, "test_score_hist"), fig_roc(videos, "Test", res_dir, "test_roc")]
    write_json(lock, {"threshold_sha256": thr_hash, "checkpoint_sha256": ck_hash,
                      "test_video_scores_sha256": file_sha256(res_dir / "test_video_scores.csv")})
    write_experiment_metadata(res_dir / "test_metadata.json", "fp32_test", repo_root,
                              inputs={"threshold": thr_path, "test_scores": res_dir / "test_video_scores.csv",
                                      "checkpoint": find_input(paths, dry_run, "models", BEST_NAME)},
                              extra={"dry_run": dry_run})
    return {"metrics": metrics, "videos": videos, "figures": figs, "results_dir": str(res_dir)}
