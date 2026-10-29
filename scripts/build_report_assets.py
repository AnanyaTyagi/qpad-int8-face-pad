"""Regenerate report tables/macros/figures from result files.

Usage (Colab or locally):
    python scripts/build_report_assets.py --project-root "/content/drive/MyDrive/QPAD_Project"

Only aggregate tables and plots are copied into reports/checkpoint1; no faces, no videos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qpad.reporting import build_report_assets  # noqa: E402
from qpad.reproducibility import git_commit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, required=True, help="QPAD_Project folder (results + reports)")
    args = ap.parse_args()
    written = build_report_assets(args.project_root / "results", args.project_root / "reports",
                                  REPO / "reports" / "checkpoint1",
                                  # the generated assets themselves do not make the code state "dirty"
                                  git_commit(REPO, ignore=("reports/checkpoint1/tables", "reports/checkpoint1/figures")))
    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()
