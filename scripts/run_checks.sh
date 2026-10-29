#!/usr/bin/env bash
# Repository self-check: unit + synthetic end-to-end tests, byte-compilation, notebook regeneration.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/build_notebooks.py
python -m compileall -q src scripts tests
python -m pytest -q
