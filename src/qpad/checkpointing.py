"""Restart-safe record stores and safe copying to Google Drive.

Colab can disconnect at any time. Long loops therefore append one record per
item to a :class:`ResumableRecordStore`, which is flushed atomically to Drive
every few items and reloaded on restart so finished items are skipped.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .reproducibility import file_sha256


class ResumableRecordStore:
    """A keyed table of per-item results persisted as CSV.

    * ``status`` column: ``"ok"`` or ``"failed"`` (failures are kept, never dropped).
    * :meth:`pending` returns keys that still need work; failed keys are retried
      only when ``retry_failed=True``.
    """

    def __init__(self, path: Path | str, key: str, flush_every: int = 25) -> None:
        self.path = Path(path)
        self.key = key
        self.flush_every = max(1, int(flush_every))
        self._records: dict[str, dict[str, Any]] = {}
        self._unflushed = 0
        if self.path.is_file():
            df = pd.read_csv(self.path, dtype={key: str})
            for rec in df.to_dict(orient="records"):
                self._records[str(rec[key])] = rec

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, k: str) -> bool:
        return str(k) in self._records

    def status(self, k: str) -> str | None:
        rec = self._records.get(str(k))
        return None if rec is None else str(rec.get("status"))

    def pending(self, keys: Iterable[str], retry_failed: bool = False) -> list[str]:
        out = []
        for k in keys:
            st = self.status(k)
            if st is None or (retry_failed and st != "ok"):
                out.append(k)
        return out

    def add(self, record: Mapping[str, Any]) -> None:
        if self.key not in record or "status" not in record:
            raise ValueError(f"record must contain '{self.key}' and 'status'")
        self._records[str(record[self.key])] = dict(record)
        self._unflushed += 1
        if self._unflushed >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Atomically rewrite the CSV (write temp file, then rename)."""
        if not self._records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        self.to_frame().to_csv(tmp, index=False)
        os.replace(tmp, self.path)
        self._unflushed = 0

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(list(self._records.values()))
        return df.sort_values(self.key, kind="mergesort").reset_index(drop=True)

    def failed(self) -> pd.DataFrame:
        df = self.to_frame() if self._records else pd.DataFrame(columns=[self.key, "status"])
        return df[df["status"] != "ok"] if len(df) else df


def copy_verified(src: Path | str, dst: Path | str) -> str:
    """Copy ``src`` to ``dst`` via a temporary name and verify the SHA-256 matches.

    Returns the digest. The destination only appears under its final name after
    verification, so a half-written file is never mistaken for a finished one.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copyfile(src, tmp)
    src_hash, tmp_hash = file_sha256(src), file_sha256(tmp)
    if src_hash != tmp_hash:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Checksum mismatch copying {src} -> {dst}")
    os.replace(tmp, dst)
    return src_hash


def remove_local_after_verified_copy(local: Path | str, remote: Path | str, expected_sha256: str) -> None:
    """Delete ``local`` only if ``remote`` exists and has the expected digest."""
    remote = Path(remote)
    if not remote.is_file():
        raise FileNotFoundError(f"Refusing to delete {local}: {remote} does not exist")
    actual = file_sha256(remote)
    if actual != expected_sha256:
        raise IOError(f"Refusing to delete {local}: remote digest {actual} != {expected_sha256}")
    Path(local).unlink(missing_ok=True)
