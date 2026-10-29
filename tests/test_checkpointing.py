import pytest

from qpad.checkpointing import ResumableRecordStore, copy_verified, remove_local_after_verified_copy


def test_store_resumes_and_keeps_failures(tmp_path):
    p = tmp_path / "drive" / "progress.csv"
    s = ResumableRecordStore(p, "video_id", flush_every=2)
    s.add({"video_id": "a", "status": "ok", "x": 1})
    assert not p.exists()                    # not flushed yet
    s.add({"video_id": "b", "status": "failed", "x": 2})
    assert p.exists()                        # flushed after 2 records
    s.add({"video_id": "c", "status": "ok", "x": 3})   # lost on "crash" (not flushed)
    s2 = ResumableRecordStore(p, "video_id")
    assert len(s2) == 2
    assert s2.pending(["a", "b", "c"]) == ["c"]
    assert s2.pending(["a", "b", "c"], retry_failed=True) == ["b", "c"]
    assert s2.failed()["video_id"].tolist() == ["b"]


def test_store_rejects_incomplete_record(tmp_path):
    with pytest.raises(ValueError):
        ResumableRecordStore(tmp_path / "p.csv", "video_id").add({"video_id": "a"})


def test_copy_verified_and_safe_delete(tmp_path):
    src = tmp_path / "local" / "f.bin"
    src.parent.mkdir()
    src.write_bytes(b"abc" * 1000)
    dst = tmp_path / "drive dir" / "f.bin"
    h = copy_verified(src, dst)
    assert dst.read_bytes() == src.read_bytes() and not (dst.parent / "f.bin.partial").exists()
    with pytest.raises(IOError):
        remove_local_after_verified_copy(src, dst, "0" * 64)
    assert src.exists()
    remove_local_after_verified_copy(src, dst, h)
    assert not src.exists()
