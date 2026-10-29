"""OULU-NPU filename parsing, Protocol II membership and manifest validation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .archive_io import MemberRecord, classify_member

SPLITS = ("train", "dev", "test")
VIDEO_ID_TOKEN = re.compile(r"(?<![\d_])(\d+_\d+_\d+_\d+)(?![\d_])")


# ---------------------------------------------------------------------------------------
# Filenames and labels
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OuluId:
    phone: int
    session: int
    subject: int
    pai: int

    @property
    def video_id(self) -> str:
        return f"{self.phone}_{self.session}_{self.subject:02d}_{self.pai}"


def parse_oulu_name(name: str, dataset_cfg: Mapping[str, Any]) -> OuluId:
    """Parse ``<phone>_<session>_<subject>_<pai>`` from a filename or stem.

    Raises ``ValueError`` for any malformed name or out-of-range field; callers
    record the error instead of skipping the file.
    """
    fcfg = dataset_cfg["filename"]
    stem = PurePosixPath(name).stem
    m = re.match(fcfg["pattern"], stem)
    if not m:
        raise ValueError(f"filename '{stem}' does not match {fcfg['pattern']}")
    oid = OuluId(int(m["phone"]), int(m["session"]), int(m["subject"]), int(m["pai"]))
    lo, hi = fcfg["subjects"]
    if oid.phone not in fcfg["phones"]:
        raise ValueError(f"'{stem}': phone {oid.phone} not in {fcfg['phones']}")
    if oid.session not in fcfg["sessions"]:
        raise ValueError(f"'{stem}': session {oid.session} not in {fcfg['sessions']}")
    if not lo <= oid.subject <= hi:
        raise ValueError(f"'{stem}': subject {oid.subject} outside [{lo}, {hi}]")
    if oid.pai not in fcfg["pai_codes"]:
        raise ValueError(f"'{stem}': PAI code {oid.pai} not in {fcfg['pai_codes']}")
    return oid


def pai_class(pai: int, dataset_cfg: Mapping[str, Any]) -> str:
    """Map a PAI code to ``bona_fide``/``print``/``replay``."""
    mapping = {int(k): v for k, v in dataset_cfg["pai_to_class"].items()}
    if pai not in mapping:
        raise ValueError(f"unknown PAI code {pai}")
    return mapping[pai]


def subject_split(subject: int, dataset_cfg: Mapping[str, Any]) -> str | None:
    """Protocol split owning ``subject`` (``None`` if no split claims it)."""
    for split, spec in dataset_cfg["protocol"]["splits"].items():
        lo, hi = spec["subjects"]
        if lo <= subject <= hi:
            return split
    return None


def protocol_split(oid: OuluId, dataset_cfg: Mapping[str, Any]) -> str | None:
    """Split this video belongs to under the configured protocol, or ``None`` if excluded."""
    split = subject_split(oid.subject, dataset_cfg)
    if split is None:
        return None
    return split if oid.pai in dataset_cfg["protocol"]["splits"][split]["pai_codes"] else None


def expected_protocol_ids(dataset_cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Enumerate every video the membership rule says must exist."""
    fcfg = dataset_cfg["filename"]
    rows = []
    for split, spec in dataset_cfg["protocol"]["splits"].items():
        lo, hi = spec["subjects"]
        for subject in range(lo, hi + 1):
            for phone in fcfg["phones"]:
                for session in fcfg["sessions"]:
                    for pai in spec["pai_codes"]:
                        oid = OuluId(phone, session, subject, pai)
                        rows.append({"video_id": oid.video_id, "split": split,
                                     "class": pai_class(pai, dataset_cfg)})
    return pd.DataFrame(rows)


def derived_counts(dataset_cfg: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    exp = expected_protocol_ids(dataset_cfg)
    table = exp.groupby(["split", "class"]).size()
    return {s: {c: int(table.get((s, c), 0)) for c in dataset_cfg["classes"]} for s in SPLITS}


# ---------------------------------------------------------------------------------------
# Protocol files shipped with the dataset (cross-check only)
# ---------------------------------------------------------------------------------------
def parse_protocol_text(text: str) -> list[str]:
    """Extract video IDs from a protocol list such as ``+1,1_1_01_1`` lines."""
    ids = []
    for line in text.splitlines():
        m = VIDEO_ID_TOKEN.search(line)
        if m:
            ids.append(m.group(1))
    return ids


def protocol_file_split(name: str, protocol_name: str) -> str | None:
    """Return the split of a protocol file belonging to ``protocol_name`` (e.g. ``Protocol_2``)."""
    lower = name.lower()
    if protocol_name.lower() not in lower.replace(" ", "_"):
        return None
    base = PurePosixPath(lower).stem
    for split, tokens in {"train": ("train",), "dev": ("dev",), "test": ("test",)}.items():
        if any(base.startswith(t) for t in tokens):
            return split
    return None


# ---------------------------------------------------------------------------------------
# Building the per-video table from archive indexes
# ---------------------------------------------------------------------------------------
def build_video_table(indexes: Mapping[str, Sequence[MemberRecord]], dataset_cfg: Mapping[str, Any],
                      annotation_lookup: Mapping[str, list[tuple[str, MemberRecord]]]) -> pd.DataFrame:
    """One row per *video member* found in the split archives (all PAI codes).

    ``indexes`` maps ``train``/``dev``/``test`` to the archive's member records.
    ``annotation_lookup`` maps a filename stem to the annotation member(s) found
    anywhere (split archives or metadata ZIP) as ``(archive_key, record)``.
    """
    disc = dataset_cfg["discovery"]
    rows: list[dict[str, Any]] = []
    for archive_split, records in indexes.items():
        for rec in records:
            if classify_member(rec, disc["video_extensions"], disc["annotation_extensions"],
                               disc["protocol_path_tokens"]) != "video":
                continue
            row: dict[str, Any] = {
                "video_id": rec.stem, "archive_key": archive_split, "archive": rec.archive,
                "member_path": rec.name, "member_offset": rec.offset_data, "file_size_bytes": rec.size,
                "parse_error": "",
            }
            try:
                oid = parse_oulu_name(rec.name, dataset_cfg)
                row.update(video_id=oid.video_id, phone=oid.phone, session=oid.session,
                           subject=oid.subject, pai_code=oid.pai, **{"class": pai_class(oid.pai, dataset_cfg)})
                psplit = protocol_split(oid, dataset_cfg)
                row["split"] = psplit or ""
                row["in_protocol"] = psplit is not None
                row["subject_split"] = subject_split(oid.subject, dataset_cfg) or ""
            except ValueError as exc:
                row.update(parse_error=str(exc), in_protocol=False, split="", subject_split="")
            anns = annotation_lookup.get(rec.stem, [])
            row["annotation_candidates"] = len(anns)
            if anns:
                akey, arec = anns[0]
                row.update(annotation_archive_key=akey, annotation_path=arec.name,
                           annotation_offset=arec.offset_data, annotation_size=arec.size)
            else:
                row.update(annotation_archive_key="", annotation_path="", annotation_offset=None,
                           annotation_size=None)
            rows.append(row)
    return pd.DataFrame(rows)


def build_annotation_lookup(indexes: Mapping[str, Sequence[MemberRecord]], dataset_cfg: Mapping[str, Any]
                            ) -> dict[str, list[tuple[str, MemberRecord]]]:
    """Map filename stem -> annotation members across all given archives."""
    disc = dataset_cfg["discovery"]
    lookup: dict[str, list[tuple[str, MemberRecord]]] = {}
    for key, records in indexes.items():
        for rec in records:
            if classify_member(rec, disc["video_extensions"], disc["annotation_extensions"],
                               disc["protocol_path_tokens"]) == "annotation":
                lookup.setdefault(rec.stem, []).append((key, rec))
    return lookup


# ---------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    status: str            # "pass" | "fail" | "not_checked"
    required: bool
    details: str
    n_violations: int = 0
    examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _result(name: str, violations: Iterable[str], details: str, required: bool = True,
            checked: bool = True) -> CheckResult:
    v = sorted(set(map(str, violations)))
    status = "not_checked" if not checked else ("pass" if not v else "fail")
    return CheckResult(name, status, required, details, len(v), v[:10])


def validate_structure(videos: pd.DataFrame, dataset_cfg: Mapping[str, Any],
                       protocol_lists: Mapping[str, list[str]] | None = None) -> list[CheckResult]:
    """Checks that need only archive indexes (names, counts, leakage, annotations present)."""
    out: list[CheckResult] = []
    stated = dataset_cfg["protocol"]["expected_counts"]
    derived = derived_counts(dataset_cfg)
    out.append(_result("stated_counts_match_membership_rule",
                       [f"{s}/{c}: stated {stated[s][c]} vs rule {derived[s][c]}"
                        for s in SPLITS for c in dataset_cfg["classes"] if stated[s][c] != derived[s][c]],
                       "expected_counts in dataset.yaml equal the subject x phone x session x PAI rule"))

    out.append(_result("filenames_parse", videos.loc[videos["parse_error"] != "", "member_path"],
                       "every video member matches <phone>_<session>_<subject>_<pai> with valid ranges"))

    dup = videos["video_id"][videos["video_id"].duplicated(keep=False)]
    out.append(_result("no_duplicate_video_ids", dup, "each video ID appears once across all archives"))

    parsed = videos[videos["parse_error"] == ""]
    mism = parsed[parsed["subject_split"] != parsed["archive_key"]]
    out.append(_result("archive_matches_subject_split", mism["video_id"],
                       "each archive contains only subjects of its own split"))

    subj = parsed.groupby("subject")["archive_key"].nunique()
    out.append(_result("no_subject_leakage", subj[subj > 1].index.astype(str),
                       "no subject appears in more than one split archive"))

    proto = parsed[parsed["in_protocol"]]
    for split, spec in dataset_cfg["protocol"]["splits"].items():
        lo, hi = spec["subjects"]
        got = set(proto.loc[proto["split"] == split, "subject"].astype(int))
        want = set(range(lo, hi + 1))
        out.append(_result(f"subjects_{split}",
                           [f"missing {s}" for s in sorted(want - got)] + [f"extra {s}" for s in sorted(got - want)],
                           f"{split} contains exactly subjects {lo}-{hi}"))

    expected = expected_protocol_ids(dataset_cfg)
    missing = set(expected["video_id"]) - set(proto["video_id"])
    out.append(_result("no_missing_protocol_videos", missing,
                       f"all {len(expected)} Protocol videos exist in the archives"))
    total_expected = int(dataset_cfg["protocol"]["expected_total"])
    out.append(_result("protocol_total", [] if proto["video_id"].nunique() == total_expected
                       else [f"found {proto['video_id'].nunique()} != {total_expected}"],
                       f"exactly {total_expected} Protocol videos"))

    counts = proto.groupby(["split", "class"]).size()
    bad = [f"{s}/{c}: {int(counts.get((s, c), 0))} != {stated[s][c]}"
           for s in SPLITS for c in dataset_cfg["classes"] if int(counts.get((s, c), 0)) != stated[s][c]]
    out.append(_result("split_class_counts", bad, "split x class counts equal the expected table"))

    out.append(_result("annotation_member_exists", proto.loc[proto["annotation_candidates"] < 1, "video_id"],
                       "every Protocol video has an annotation file with the same stem"))
    out.append(_result("annotation_member_unique", proto.loc[proto["annotation_candidates"] > 1, "video_id"],
                       "no Protocol video has more than one candidate annotation file", required=False))

    if protocol_lists:
        viol = []
        for split, ids in protocol_lists.items():
            ours = set(proto.loc[proto["split"] == split, "video_id"])
            theirs = set(ids)
            viol += [f"{split}: only in protocol file {v}" for v in sorted(theirs - ours)]
            viol += [f"{split}: only in membership rule {v}" for v in sorted(ours - theirs)]
        out.append(_result("official_protocol_files_agree", viol,
                           "membership rule agrees with the official Protocol_2 list files", required=False))
    else:
        out.append(_result("official_protocol_files_agree", [], "official Protocol_2 list files not found",
                           required=False, checked=False))
    return out


def validate_audit(manifest: pd.DataFrame, dataset_cfg: Mapping[str, Any], full_decode_requested: bool,
                   n_expected: int) -> list[CheckResult]:
    """Checks that need the per-video audit fields (decoding and annotation content)."""
    from .data_exceptions import DataExceptions

    out: list[CheckResult] = []
    acfg = dataset_cfg["audit"]
    ex = DataExceptions.from_cfg(dataset_cfg)
    audited = manifest[manifest["status"].isin(["ok", "failed"])] if "status" in manifest else manifest.iloc[:0]
    complete = len(audited) == n_expected
    # An unfinished audit (dry run or interrupted) is INCOMPLETE, not a data failure.
    out.append(CheckResult("every_protocol_video_audited", "pass" if complete else "not_checked", True,
                           f"every Protocol video was inspected (audited {len(audited)} of {n_expected})"))
    # A documented damaged-frame video may fail ONLY through its decode error.
    damaged = audited["video_id"].isin(list(ex.damaged_frames))
    only_decode = audited["error"].fillna("").astype(str).str.split(" | ", regex=False).map(
        lambda parts: all(p.startswith("decode:") for p in parts if p))
    out.append(_result("audit_errors", audited.loc[(audited["status"] != "ok") & ~(damaged & only_decode), "video_id"],
                       "no video failed extraction, probing or annotation parsing (documented damaged-frame "
                       "videos may fail only by their decode error)", checked=complete))
    out.append(_result("ffprobe_metadata",
                       audited.loc[~(audited["frame_count"].fillna(0).astype(float) > 0), "video_id"],
                       "ffprobe reports a positive frame count for every video", checked=complete))
    out.append(_result("full_decode", audited.loc[(audited["decode_status"] != "ok") & ~damaged, "video_id"],
                       "complete FFmpeg decode without errors (except documented damaged frames)",
                       checked=complete and full_decode_requested))
    out.append(_result("annotation_parse", audited.loc[audited["ann_malformed_rows"].fillna(1) > 0, "video_id"],
                       "annotation files contain only well-formed 'frame, x1, y1, x2, y2' rows",
                       checked=complete))
    min_valid = int(acfg["min_valid_annotated_frames"])
    valid = audited["ann_valid_in_range"].fillna(0)
    documented_reduced = audited["video_id"].map(ex.reduced_frame_videos)
    out.append(_result("annotation_coverage",
                       audited.loc[(valid < min_valid) & ~(documented_reduced == valid), "video_id"],
                       f">= {min_valid} valid annotated frames inside each video's frame range "
                       "(or exactly the documented count for reduced-frame videos)", checked=complete))
    viol = []
    by_id = audited.set_index("video_id")
    for vid in ex.damaged_frames:
        if vid not in by_id.index:
            viol.append(f"{vid}: listed as damaged but not audited")
        elif by_id.loc[vid, "decode_status"] == "ok":
            viol.append(f"{vid}: listed as damaged but decodes cleanly (remove the exception)")
    for vid, n in ex.reduced_frame_videos.items():
        if vid not in by_id.index:
            viol.append(f"{vid}: listed as reduced-frame but not audited")
        else:
            if by_id.loc[vid, "split"] == "train":
                viol.append(f"{vid}: reduced-frame exceptions are not allowed in the training split")
            if int(by_id.loc[vid, "ann_valid_in_range"]) != n:
                viol.append(f"{vid}: documented {n} valid frames, audit found {int(by_id.loc[vid, 'ann_valid_in_range'])}")
            if n >= min_valid:
                viol.append(f"{vid}: documented count {n} is not below {min_valid}")
    out.append(_result("documented_exceptions_match_data", viol,
                       f"every data exception in dataset.yaml matches the audited data "
                       f"({len(ex.damaged_frames)} damaged-frame, {len(ex.reduced_frame_videos)} reduced-frame)",
                       checked=complete))
    bases = audited["ann_frame_base"].fillna("").astype(str)
    resolved = bases[bases.isin(["0", "1"])]
    conflict = resolved.nunique() > 1
    out.append(_result("annotation_frame_base_consistent",
                       ([f"bases seen: {sorted(resolved.unique())}"] if conflict else [])
                       + list(audited.loc[~bases.isin(["0", "1"]), "video_id"]),
                       "annotation frame numbering is consistently 0- or 1-based", checked=complete))
    return out


def overall_status(checks: Sequence[CheckResult]) -> str:
    """``PASS`` only when every required check ran and passed."""
    req = [c for c in checks if c.required]
    if any(c.status == "fail" for c in req):
        return "FAIL"
    if any(c.status == "not_checked" for c in req):
        return "INCOMPLETE"
    return "PASS"


def checks_to_markdown(checks: Sequence[CheckResult]) -> str:
    lines = ["| Check | Status | Required | Violations | Details | Examples |",
             "|---|---|---|---:|---|---|"]
    for c in checks:
        ex = "; ".join(c.examples[:3]).replace("|", "/")
        lines.append(f"| {c.name} | {c.status.upper()} | {'yes' if c.required else 'no'} | "
                     f"{c.n_violations} | {c.details} | {ex} |")
    return "\n".join(lines)
