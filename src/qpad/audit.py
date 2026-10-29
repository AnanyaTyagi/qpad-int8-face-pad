"""Phase 1 dataset audit: per-video inspection and report writing.

One video at a time is extracted from its TAR into a temporary directory,
probed with ffprobe, fully decoded with ffmpeg (optional), optionally hashed,
and deleted. Results go to a resumable store so the audit survives Colab
disconnections.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from .annotations import annotation_summary, parse_eye_annotation
from .archive_io import MemberRecord, read_member_bytes, temporary_member
from .checkpointing import ResumableRecordStore
from .manifest import CheckResult, checks_to_markdown, overall_status
from .reproducibility import file_sha256, write_json
from .video_io import ffmpeg_full_decode, ffprobe_video

AUDIT_FIELDS = ("frame_count", "width", "height", "fps", "duration_s", "codec", "decode_status",
                "decode_error", "sha256", "ann_rows", "ann_lines", "ann_malformed_rows",
                "ann_malformed_examples", "ann_valid_rows", "ann_min_frame", "ann_max_frame",
                "ann_duplicate_frames", "ann_frame_base", "ann_valid_in_range", "status", "error")


def video_member(row: Mapping[str, Any]) -> MemberRecord:
    return MemberRecord(archive=str(row["archive"]), name=str(row["member_path"]),
                        size=int(row["file_size_bytes"]), is_file=True,
                        offset_data=None if pd.isna(row["member_offset"]) else int(row["member_offset"]))


def annotation_member(row: Mapping[str, Any]) -> MemberRecord | None:
    if not row.get("annotation_path"):
        return None
    off = row.get("annotation_offset")
    return MemberRecord(archive=str(row["annotation_archive_key"]), name=str(row["annotation_path"]),
                        size=int(row["annotation_size"]), is_file=True,
                        offset_data=None if off is None or pd.isna(off) else int(off))


def audit_one_video(row: Mapping[str, Any], archive_paths: Mapping[str, Path], tmp_dir: Path,
                    dataset_cfg: Mapping[str, Any], full_decode: bool, checksum: bool) -> dict[str, Any]:
    """Inspect one video; never raises — errors are returned in ``status``/``error``."""
    rec: dict[str, Any] = {"video_id": row["video_id"]}
    rec.update({k: None for k in AUDIT_FIELDS})
    errors: list[str] = []
    try:
        vmember = video_member(row)
        with temporary_member(archive_paths[row["archive_key"]], vmember, tmp_dir) as local:
            probe = ffprobe_video(local)
            rec.update(probe)
            if full_decode:
                rec["decode_status"], rec["decode_error"] = ffmpeg_full_decode(local)
            else:
                rec["decode_status"], rec["decode_error"] = "not_run", ""
            if checksum:
                rec["sha256"] = file_sha256(local)
    except Exception as exc:  # noqa: BLE001 - recorded per video, reported as failure
        errors.append(f"video: {type(exc).__name__}: {exc}")
        rec["decode_status"] = rec["decode_status"] or "failed"
    try:
        amember = annotation_member(row)
        if amember is None:
            raise FileNotFoundError("no annotation file")
        text = read_member_bytes(archive_paths[amember.archive], amember).decode("utf-8", errors="replace")
        parsed = parse_eye_annotation(text)
        rec.update(annotation_summary(parsed, int(rec["frame_count"] or 0),
                                      float(dataset_cfg["annotations"]["min_eye_distance_px"])))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"annotation: {type(exc).__name__}: {exc}")
    if rec["decode_status"] == "failed":
        errors.append(f"decode: {rec['decode_error']}")
    rec["status"] = "failed" if errors else "ok"
    rec["error"] = " | ".join(errors)
    return rec


def run_audit(videos: pd.DataFrame, archive_paths: Mapping[str, Path], tmp_dir: Path, store_path: Path,
              dataset_cfg: Mapping[str, Any], full_decode: bool, checksum: bool, workers: int,
              flush_every: int, retry_failed: bool = False,
              progress: Callable[[int, int], None] | None = None, logger: Any = None) -> pd.DataFrame:
    """Audit every row of ``videos`` not already completed in ``store_path``; return all records."""
    store = ResumableRecordStore(store_path, key="video_id", flush_every=flush_every)
    todo_ids = store.pending(videos["video_id"].tolist(), retry_failed=retry_failed)
    rows = videos.set_index("video_id", drop=False).loc[todo_ids].to_dict(orient="records")
    if logger:
        logger.info("audit: %d done previously, %d to process", len(videos) - len(rows), len(rows))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(audit_one_video, r, archive_paths, tmp_dir, dataset_cfg, full_decode,
                                   checksum): r["video_id"] for r in rows}
            for fut in as_completed(futures):
                rec = fut.result()
                store.add(rec)
                done += 1
                if rec["status"] != "ok" and logger:
                    logger.warning("audit failure %s: %s", rec["video_id"], rec["error"])
                if progress:
                    progress(done, len(rows))
    finally:
        store.flush()
    return store.to_frame()


NUMERIC_AUDIT_FIELDS = ("frame_count", "width", "height", "fps", "duration_s", "ann_rows", "ann_lines",
                        "ann_malformed_rows", "ann_valid_rows", "ann_min_frame", "ann_max_frame",
                        "ann_duplicate_frames", "ann_valid_in_range")


def _base_to_str(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = str(v).strip()
    return {"0.0": "0", "1.0": "1"}.get(s, s)


def normalize_manifest_types(df: pd.DataFrame) -> pd.DataFrame:
    """Restore column types after a CSV round trip (numbers numeric, frame base as '0'/'1'/'ambiguous')."""
    out = df.copy()
    for c in NUMERIC_AUDIT_FIELDS:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "ann_frame_base" in out:
        out["ann_frame_base"] = out["ann_frame_base"].map(_base_to_str)
    for c in ("video_id",):
        out[c] = out[c].astype(str)
    if "in_protocol" in out:
        out["in_protocol"] = out["in_protocol"].astype(str).str.lower().eq("true")
    return out


def merge_manifest(videos: pd.DataFrame, audit_records: pd.DataFrame) -> pd.DataFrame:
    """Join index-derived rows with audit fields (videos without a record keep NaNs)."""
    a = audit_records.copy()
    a["video_id"] = a["video_id"].astype(str)
    keep = [c for c in a.columns if c in AUDIT_FIELDS or c == "video_id"]
    out = videos.merge(a[keep], on="video_id", how="left")
    out["label"] = out["class"].map({"bona_fide": 1, "print": 0, "replay": 0})
    return normalize_manifest_types(out)


def videos_failing_checks(proto: pd.DataFrame, min_valid_frames: int) -> pd.DataFrame:
    """Every audited Protocol video failing any per-video check, with the reasons."""
    if "status" not in proto:
        return pd.DataFrame(columns=["video_id", "split", "class", "reasons"])
    a = proto[proto["status"].isin(["ok", "failed"])]
    reasons = pd.DataFrame({
        "processing_error": a["status"] != "ok",
        "decode_error": a["decode_status"].isin(["failed"]),
        "malformed_annotation": a["ann_malformed_rows"].fillna(1) > 0,
        f"fewer_than_{min_valid_frames}_valid_annotated_frames": a["ann_valid_in_range"].fillna(0) < min_valid_frames,
    }, index=a.index)
    bad = reasons.any(axis=1)
    out = a.loc[bad, ["video_id", "split", "class", "frame_count", "ann_valid_in_range", "error"]].copy()
    out["reasons"] = reasons[bad].apply(lambda r: ";".join(c for c, v in r.items() if v), axis=1)
    return out.reset_index(drop=True)


def annotate_documented(failing: pd.DataFrame, dataset_cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Add the documented handling of each failing video (empty if undocumented)."""
    from .data_exceptions import DataExceptions

    ex = DataExceptions.from_cfg(dataset_cfg)

    def handling(vid: str) -> str:
        if vid in ex.damaged_frames:
            return f"keep; frames {list(ex.damaged_frames[vid])} never selected"
        if vid in ex.reduced_frame_videos:
            return f"keep; scored on all {ex.reduced_frame_videos[vid]} valid annotated frames"
        return ""

    out = failing.copy()
    out["documented_handling"] = out["video_id"].map(handling) if len(out) else []
    return out


def write_audit_reports(checks: Sequence[CheckResult], manifest: pd.DataFrame, json_path: Path, md_path: Path,
                        context: Mapping[str, Any], min_valid_frames: int = 20,
                        dataset_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write the machine-readable JSON and the human-readable Markdown audit reports."""
    proto = manifest[manifest["in_protocol"]]
    status = overall_status(checks)

    def _stat(col: str) -> dict[str, float] | None:
        if col not in proto or proto[col].dropna().empty:
            return None
        s = pd.to_numeric(proto[col], errors="coerce").dropna()
        return {"min": float(s.min()), "median": float(s.median()), "max": float(s.max())}

    summary = {
        "overall_status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "context": dict(context),
        "n_video_members_all_archives": int(len(manifest)),
        "n_protocol_videos": int(len(proto)),
        "n_outside_protocol": int((~manifest["in_protocol"]).sum()),
        "counts": {s: proto[proto["split"] == s]["class"].value_counts().to_dict() for s in ("train", "dev", "test")},
        "resolution_counts": (proto["width"].astype("Int64").astype(str) + "x" +
                              proto["height"].astype("Int64").astype(str)).value_counts().to_dict()
        if "width" in proto else {},
        "frame_count": _stat("frame_count"),
        "duration_s": _stat("duration_s"),
        "fps": _stat("fps"),
        "checks": [c.as_dict() for c in checks],
        "failed_videos": proto.loc[proto.get("status", pd.Series(dtype=str)) == "failed",
                                   ["video_id", "error"]].to_dict(orient="records") if "status" in proto else [],
    }
    failing = videos_failing_checks(proto, min_valid_frames)
    if dataset_cfg is not None:
        failing = annotate_documented(failing, dataset_cfg)
    summary["videos_failing_checks"] = failing.to_dict(orient="records")
    write_json(json_path, summary)
    md = [
        "# OULU-NPU Protocol II dataset audit", "",
        f"**Overall status: {status}**", "",
        "`PASS` requires every required check to have run and passed. `INCOMPLETE` means at least one "
        "required check was not run (for example in DRY_RUN mode).", "",
        "## Context", "", *[f"- {k}: {v}" for k, v in context.items()], "",
        "## Counts (Protocol II)", "",
        "| Split | Bona fide | Print | Replay | Total |", "|---|---:|---:|---:|---:|",
    ]
    for s in ("train", "dev", "test"):
        c = summary["counts"][s]
        md.append(f"| {s} | {c.get('bona_fide', 0)} | {c.get('print', 0)} | {c.get('replay', 0)} | {sum(c.values())} |")
    md += ["", f"Video members found in all archives: {summary['n_video_members_all_archives']} "
           f"({summary['n_outside_protocol']} outside Protocol II, e.g. Printer 2/Display 2 in train/dev).", "",
           "## Checks", "", checks_to_markdown(checks), "",
           "## Video properties (Protocol II)", "",
           f"- Resolutions: {summary['resolution_counts']}",
           f"- Frame count: {summary['frame_count']}",
           f"- Duration (s): {summary['duration_s']}",
           f"- FPS: {summary['fps']}", "",
           f"## Failed videos ({len(summary['failed_videos'])})", ""]
    md += [f"- `{f['video_id']}`: {f['error']}" for f in summary["failed_videos"][:200]]
    md += ["", f"## Videos failing any per-video check ({len(failing)})", ""]
    md += [f"- `{r['video_id']}` ({r['split']}/{r['class']}): {r['reasons']} "
           f"(frames {r['frame_count']}, valid annotated {r['ann_valid_in_range']}) — handling: "
           f"{r.get('documented_handling') or 'UNDOCUMENTED'}" for r in summary["videos_failing_checks"]]
    md += ["", "## Limitations to record in the report", "",
           "- Annotation eye coordinates are taken as supplied; their accuracy is not independently verified.",
           "- Frame counts come from ffprobe packet decoding; OpenCV decoding is re-checked during preprocessing."]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def estimate_audit_hours(n_videos: int, mean_size_mb: float, read_mb_s: float = 60.0,
                         decode_s_per_video: float = 2.0, workers: int = 4) -> float:
    """Rough wall-clock estimate shown before a full run (I/O-bound assumption)."""
    per_video = mean_size_mb / read_mb_s + decode_s_per_video
    return math.ceil(n_videos * per_video / max(1, workers) / 360) / 10
