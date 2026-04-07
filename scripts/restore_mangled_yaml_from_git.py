#!/usr/bin/env python3
"""Restore YAML files mangled by yaml.dump() using git HEAD text + semantic edits from current file.

Reads committed formatting, reapplies string changes discovered by diffing safe_load(current) vs safe_load(HEAD).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _git_show(relpath: str) -> str:
    out = subprocess.run(
        ["git", "show", f"HEAD:{relpath}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def _walk_strings(
    a: Any, b: Any, key_path: str = ""
) -> list[tuple[str, str]]:
    """Return (old, new) pairs for differing string leaves at matching structure paths."""
    if type(a) is not type(b):
        return []
    if isinstance(a, dict):
        keys = set(a) | set(b)
        out: list[tuple[str, str]] = []
        for k in sorted(keys):
            if k not in a or k not in b:
                continue
            out.extend(_walk_strings(a[k], b[k], f"{key_path}.{k}"))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return []
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(_walk_strings(x, y, f"{key_path}[{i}]"))
        return out
    if isinstance(a, str) and isinstance(b, str) and a != b:
        return [(a, b)]
    return []


def restore_file(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    good_text = _git_show(rel)
    bad_text = path.read_text(encoding="utf-8")
    try:
        good_data = yaml.safe_load(good_text)
        bad_data = yaml.safe_load(bad_text)
    except yaml.YAMLError:
        return False
    if not isinstance(good_data, dict) or not isinstance(bad_data, dict):
        return False

    pairs = _walk_strings(good_data, bad_data)
    if not pairs:
        return False

    # Longest `old` first to reduce accidental substring collisions.
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    out = good_text
    for old, new in pairs:
        if old == new:
            continue
        if old not in out:
            continue
        out = out.replace(old, new)
    if out == good_text:
        return False
    path.write_text(out, encoding="utf-8")
    return True


def main() -> int:
    # Files whose first line is `name:` lost the manifest header (yaml.dump).
    definitions = ROOT / "resource" / "scenarios" / "definitions"
    mangled: list[Path] = []
    for path in sorted(definitions.rglob("*.yaml")):
        first = path.read_text(encoding="utf-8").splitlines()[:1]
        if first and first[0].startswith("name:"):
            mangled.append(path)

    n = 0
    for path in mangled:
        try:
            if restore_file(path):
                print(f"restored {path.relative_to(ROOT)}")
                n += 1
        except subprocess.CalledProcessError:
            print(f"skip (not in HEAD?): {path}", file=sys.stderr)
    print(f"done, {n} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
