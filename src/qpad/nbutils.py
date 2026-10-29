"""Small helpers for notebook display: progress bars and pre-flight summaries."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import disk_free_gb


def tqdm_progress(desc: str) -> Callable[[int, int], None]:
    """A ``progress(done, total)`` callback backed by a tqdm bar."""
    from tqdm.auto import tqdm

    bar: dict[str, Any] = {}

    def cb(done: int, total: int) -> None:
        if "b" not in bar:
            bar["b"] = tqdm(total=total, desc=desc)
        bar["b"].n = done
        bar["b"].refresh()
        if done >= total:
            bar["b"].close()

    return cb


def df_h(path: Path | str) -> str:
    """Output of ``df -h`` for the filesystem holding ``path`` (argument passed safely, no shell)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    if shutil.which("df") is None:
        return f"{disk_free_gb(p):.1f} GB free at {p}"
    return subprocess.run(["df", "-h", str(p)], capture_output=True, text=True).stdout


def preflight(title: str, n_items: int, sources: Sequence[Path | str], destinations: Sequence[Path | str],
              local_root: Path, config: Mapping[str, Any], seed: int | None, dry_run: bool,
              estimate: str | None = None) -> None:
    """Print everything that must be checked before a long operation starts."""
    bar = "=" * 78
    print(bar)
    print(f"PRE-FLIGHT: {title}   [{'DRY RUN' if dry_run else 'FULL RUN'}]")
    print(bar)
    print(f"Items to process : {n_items}")
    print("Sources          :", *[f"\n  {s}" for s in sources])
    print("Destinations     :", *[f"\n  {d}" for d in destinations])
    print(f"Local free space : {disk_free_gb(local_root):.1f} GB at {local_root}")
    print(df_h(local_root).strip())
    if estimate:
        print(f"Estimate         : {estimate}")
    print(f"Seed             : {seed}")
    print("Configuration    :")
    print(json.dumps(config, indent=2, default=str)[:3000])
    print(bar)
