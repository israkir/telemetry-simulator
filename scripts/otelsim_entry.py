#!/usr/bin/env python3
"""
Thin entry-point wrapper for building a single-file otelsim binary with PyInstaller.

Usage (from repo root):
  venv/bin/pyinstaller -F -n otelsim-linux-arm64 scripts/otelsim_entry.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Ensure the src/ directory is on sys.path so simulator package is importable."""
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def main() -> None:
    """Delegate to simulator.cli.main after preparing import path."""
    _ensure_src_on_path()
    from simulator.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()

