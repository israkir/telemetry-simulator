#!/usr/bin/env python3
"""Prefix claim ticket tokens with CLAIM- (e.g. PH-9201 -> CLAIM-PH-9201) across scenario YAML.

Skips tokens embedded in QT-, POL-, APT-, PAY-, and avoids double-prefixing CLAIM-.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = ROOT / "resource" / "scenarios" / "definitions"

# Same family as prefix_claim_ids_in_scenarios.py — product claim tickets only.
CLAIM_TOKEN = re.compile(
    r"(?<!QT-)(?<!POL-)(?<!APT-)(?<!PAY-)(?<!CLAIM-)"
    r"\b((?:PH|EL|AP|HA|EV|TV)(?:-[A-Z0-9]+)+)\b"
)


DUP_CLAIM = re.compile(r"(?i)\bclaim\s+CLAIM-")


def transform(text: str) -> str:
    out = CLAIM_TOKEN.sub(lambda m: f"CLAIM-{m.group(1)}", text)
    out = DUP_CLAIM.sub("CLAIM-", out)
    return out


def process_file(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    out = transform(raw)
    if out == raw:
        return False
    path.write_text(out, encoding="utf-8")
    return True


def main() -> int:
    changed: list[Path] = []
    for path in sorted(DEFINITIONS.rglob("*.yaml")):
        if process_file(path):
            changed.append(path)
    for p in changed:
        print(f"updated {p.relative_to(ROOT)}")
    print(f"done, {len(changed)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
