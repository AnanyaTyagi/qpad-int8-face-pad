import io
import tarfile
import zipfile

import pytest

from qpad.archive_io import (check_archives_exist, extract_member, get_or_build_index, index_tar, index_zip,
                             read_member_bytes, stem_pairing, summarize_members, temporary_member)

EXTS = ([".avi"], [".txt"], ["protocol"])


def _make_tar(path, members):
    with tarfile.open(path, "w") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def test_index_and_extract_by_offset(tmp_path):
    members = {"Train_files/1_1_01_1.avi": b"A" * 5000, "Train_files/1_1_01_1.txt": b"1, 2, 3, 4, 5\n"}
    tar = tmp_path / "with space.tar"
    _make_tar(tar, members)
    recs = index_tar(tar)
    assert [r.name for r in recs] == list(members)
    for r in recs:
        assert read_member_bytes(tar, r) == members[r.name]
    out = extract_member(tar, recs[0], tmp_path / "x")
    assert out.read_bytes() == members[recs[0].name]


def test_temporary_member_is_deleted(tmp_path):
    tar = tmp_path / "a.tar"
    _make_tar(tar, {"v.avi": b"123"})
    rec = index_tar(tar)[0]
    with temporary_member(tar, rec, tmp_path / "tmp") as p:
        assert p.exists()
    assert not p.exists()


def test_index_cache_roundtrip_and_invalidation(tmp_path):
    tar = tmp_path / "a.tar"
    _make_tar(tar, {"v.avi": b"123"})
    idx = tmp_path / "idx.json.gz"
    first = get_or_build_index(tar, idx)
    assert get_or_build_index(tar, idx) == first
    _make_tar(tar, {"v.avi": b"123", "w.avi": b"4567"})
    assert len(get_or_build_index(tar, idx)) == 2  # size changed -> rebuilt


def test_zip_index_and_summary(tmp_path):
    z = tmp_path / "m.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("OULU/Protocols/Protocol_2/Train.txt", "+1,1_1_01_1\n")
        zf.writestr("OULU/readme.md", "x")
    recs = index_zip(z)
    s = summarize_members(recs, *EXTS)
    assert s["kinds"]["protocol"] == 1 and s["kinds"]["other"] == 1
    assert read_member_bytes(z, recs[0]) == b"+1,1_1_01_1\n"


def test_stem_pairing(tmp_path):
    tar = tmp_path / "a.tar"
    _make_tar(tar, {"d/1_1_01_1.avi": b"1", "d/1_1_01_1.txt": b"x", "d/1_1_01_2.avi": b"2"})
    pairing = stem_pairing(index_tar(tar), *EXTS)
    assert pairing["videos_with_same_stem_annotation"] == 1
    assert pairing["videos_without_annotation_examples"] == ["1_1_01_2"]


def test_missing_archive_error_lists_nearby(tmp_path):
    (tmp_path / "dataset biometric").mkdir()
    (tmp_path / "dataset biometric" / "other.tar").write_bytes(b"")
    with pytest.raises(FileNotFoundError) as e:
        check_archives_exist({"train": tmp_path / "dataset biometric" / "Train_files-002.tar"})
    assert "other.tar" in str(e.value)
