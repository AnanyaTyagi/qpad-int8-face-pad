"""Inspect and extract individual members of very large TAR/ZIP archives.

Design: a TAR file is scanned once header-by-header (the tar module seeks past
member payloads, so 40 GB are not read). Each member's byte offset is cached in
an index file on Drive. Later stages extract a single video by seeking straight
to that offset, so no stage ever needs the whole archive in ``/content``.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import os
import re
import tarfile
import uuid
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Sequence

COPY_CHUNK_BYTES = 8 * 1024 * 1024
SAMPLE_NAMES_PER_KIND = 8


@dataclass(frozen=True)
class MemberRecord:
    """One archive member. ``offset_data`` is the payload byte offset (TAR only)."""

    archive: str
    name: str
    size: int
    is_file: bool
    offset_data: int | None = None

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.name).suffix.lower()

    @property
    def stem(self) -> str:
        return PurePosixPath(self.name).stem

    @property
    def basename(self) -> str:
        return PurePosixPath(self.name).name


# ---------------------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------------------
def index_tar(tar_path: Path | str, progress: Callable[[int], None] | None = None) -> list[MemberRecord]:
    """List all members of an *uncompressed* TAR without reading member payloads."""
    tar_path = Path(tar_path)
    records: list[MemberRecord] = []
    with tarfile.open(tar_path, mode="r:") as tf:
        for i, ti in enumerate(tf):
            records.append(MemberRecord(archive=tar_path.name, name=ti.name, size=int(ti.size),
                                        is_file=ti.isfile(), offset_data=int(ti.offset_data)))
            if progress is not None and i % 500 == 0:
                progress(i)
    return records


def peek_tar(tar_path: Path | str, max_members: int) -> list[MemberRecord]:
    """Read only the first ``max_members`` headers (quick layout check for DRY_RUN)."""
    tar_path = Path(tar_path)
    out: list[MemberRecord] = []
    with tarfile.open(tar_path, mode="r:") as tf:
        while len(out) < max_members:
            ti = tf.next()
            if ti is None:
                break
            out.append(MemberRecord(archive=tar_path.name, name=ti.name, size=int(ti.size), is_file=ti.isfile(),
                                    offset_data=int(ti.offset_data)))
    return out


def index_zip(zip_path: Path | str) -> list[MemberRecord]:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        return [MemberRecord(archive=zip_path.name, name=zi.filename, size=int(zi.file_size),
                             is_file=not zi.is_dir()) for zi in zf.infolist()]


def _archive_fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"name": path.name, "size": st.st_size, "mtime_ns": int(st.st_mtime_ns)}


def save_index(records: Sequence[MemberRecord], archive_path: Path, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"archive": _archive_fingerprint(archive_path), "members": [asdict(r) for r in records]}
    tmp = index_path.with_name(index_path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, index_path)


def load_index(archive_path: Path, index_path: Path) -> list[MemberRecord] | None:
    """Load a cached index if it matches the archive's size and mtime, else ``None``."""
    if not index_path.is_file():
        return None
    with gzip.open(index_path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("archive") != _archive_fingerprint(archive_path):
        return None
    return [MemberRecord(**m) for m in payload["members"]]


def get_or_build_index(archive_path: Path | str, index_path: Path | str,
                       force: bool = False) -> list[MemberRecord]:
    """Return the cached member index for a TAR or ZIP, building it if needed."""
    archive_path, index_path = Path(archive_path), Path(index_path)
    if not force:
        cached = load_index(archive_path, index_path)
        if cached is not None:
            return cached
    if zipfile.is_zipfile(archive_path):
        records = index_zip(archive_path)
    else:
        records = index_tar(archive_path)
    save_index(records, archive_path, index_path)
    return records


# ---------------------------------------------------------------------------------------
# Extraction of single members
# ---------------------------------------------------------------------------------------
def _safe_local_name(member_name: str) -> str:
    """A flat, collision-free local filename (never trusts archive paths)."""
    base = re.sub(r"[^A-Za-z0-9._-]", "_", PurePosixPath(member_name).name) or "member"
    return f"{uuid.uuid4().hex[:8]}_{base}"


def read_member_bytes(archive_path: Path | str, rec: MemberRecord) -> bytes:
    """Read one member fully into memory (use for small files such as annotations)."""
    archive_path = Path(archive_path)
    if rec.offset_data is None:
        with zipfile.ZipFile(archive_path) as zf:
            return zf.read(rec.name)
    with archive_path.open("rb") as fh:
        fh.seek(rec.offset_data)
        data = fh.read(rec.size)
    if len(data) != rec.size:
        raise IOError(f"Short read for {rec.name}: {len(data)} of {rec.size} bytes")
    return data


def extract_member(archive_path: Path | str, rec: MemberRecord, dest_dir: Path | str) -> Path:
    """Copy one member to ``dest_dir`` in chunks and verify its size."""
    archive_path, dest_dir = Path(archive_path), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / _safe_local_name(rec.name)
    written = 0
    if rec.offset_data is None:
        with zipfile.ZipFile(archive_path) as zf, zf.open(rec.name) as src, dest.open("wb") as out:
            for chunk in iter(lambda: src.read(COPY_CHUNK_BYTES), b""):
                out.write(chunk)
                written += len(chunk)
    else:
        with archive_path.open("rb") as src, dest.open("wb") as out:
            src.seek(rec.offset_data)
            remaining = rec.size
            while remaining > 0:
                chunk = src.read(min(COPY_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                remaining -= len(chunk)
    if written != rec.size:
        dest.unlink(missing_ok=True)
        raise IOError(f"Extracted {written} of {rec.size} bytes for {rec.name}")
    return dest


@contextlib.contextmanager
def temporary_member(archive_path: Path | str, rec: MemberRecord, tmp_dir: Path | str) -> Iterator[Path]:
    """Extract one member to a temporary file and always delete it afterwards."""
    path = extract_member(archive_path, rec, tmp_dir)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------
# Discovery / inventory
# ---------------------------------------------------------------------------------------
def classify_member(rec: MemberRecord, video_exts: Sequence[str], annotation_exts: Sequence[str],
                    protocol_tokens: Sequence[str]) -> str:
    """Classify a member as ``video``, ``protocol``, ``annotation``, ``directory`` or ``other``."""
    if not rec.is_file:
        return "directory"
    lower = rec.name.lower()
    if rec.suffix in {e.lower() for e in video_exts}:
        return "video"
    if rec.suffix in {e.lower() for e in annotation_exts}:
        if any(tok in lower for tok in protocol_tokens):
            return "protocol"
        return "annotation"
    return "other"


def summarize_members(records: Sequence[MemberRecord], video_exts: Sequence[str],
                      annotation_exts: Sequence[str], protocol_tokens: Sequence[str]) -> dict[str, Any]:
    """Counts, extensions, top-level layout, sizes and representative names."""
    kinds: Counter[str] = Counter()
    exts: Counter[str] = Counter()
    dirs: Counter[str] = Counter()
    bytes_by_kind: Counter[str] = Counter()
    samples: dict[str, list[str]] = {}
    for rec in records:
        kind = classify_member(rec, video_exts, annotation_exts, protocol_tokens)
        kinds[kind] += 1
        bytes_by_kind[kind] += rec.size
        if rec.is_file:
            exts[rec.suffix or "<none>"] += 1
            parent = str(PurePosixPath(rec.name).parent)
            dirs[parent] += 1
        bucket = samples.setdefault(kind, [])
        if len(bucket) < SAMPLE_NAMES_PER_KIND:
            bucket.append(rec.name)
    return {
        "n_members": len(records),
        "n_files": sum(r.is_file for r in records),
        "total_uncompressed_bytes": int(sum(r.size for r in records if r.is_file)),
        "kinds": dict(kinds),
        "bytes_by_kind": dict(bytes_by_kind),
        "extensions": dict(exts.most_common()),
        "directories": dict(dirs.most_common(20)),
        "n_directories_with_files": len(dirs),
        "sample_names": samples,
    }


def stem_pairing(records: Sequence[MemberRecord], video_exts: Sequence[str],
                 annotation_exts: Sequence[str], protocol_tokens: Sequence[str]) -> dict[str, Any]:
    """How many videos have an annotation member with the same filename stem."""
    videos = {r.stem for r in records if classify_member(r, video_exts, annotation_exts, protocol_tokens) == "video"}
    anns = {r.stem for r in records if classify_member(r, video_exts, annotation_exts, protocol_tokens) == "annotation"}
    return {"videos": len(videos), "annotations": len(anns), "videos_with_same_stem_annotation": len(videos & anns),
            "videos_without_annotation_examples": sorted(videos - anns)[:SAMPLE_NAMES_PER_KIND],
            "annotations_without_video_examples": sorted(anns - videos)[:SAMPLE_NAMES_PER_KIND]}


def list_nearby(path: Path | str, max_entries: int = 30) -> dict[str, list[str]]:
    """List the nearest existing ancestor directories of ``path`` (for diagnosing bad paths).

    Only the immediate entries of up to two ancestor directories are listed; the
    Drive is never searched recursively.
    """
    path = Path(path)
    found: dict[str, list[str]] = {}
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    for d in [p, p.parent][:2]:
        if d.is_dir():
            try:
                found[str(d)] = sorted(x.name for x in d.iterdir())[:max_entries]
            except OSError as exc:
                found[str(d)] = [f"<cannot list: {exc}>"]
    return found


def check_archives_exist(archives: dict[str, Path]) -> None:
    """Raise a FileNotFoundError naming every missing archive and nearby folders."""
    missing = {k: p for k, p in archives.items() if not Path(p).is_file()}
    if not missing:
        return
    lines = ["The following archives were not found:"]
    lines += [f"  {k}: {p}" for k, p in missing.items()]
    first = next(iter(missing.values()))
    lines.append("Nearby folders (to help fix the path):")
    for d, entries in list_nearby(first).items():
        lines.append(f"  {d}/")
        lines += [f"    {e}" for e in entries]
    raise FileNotFoundError("\n".join(lines))
