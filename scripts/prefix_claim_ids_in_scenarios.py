#!/usr/bin/env python3
"""Prefix claim IDs in scenario YAML with 'claim' / 'claim ID' style wording.

Skips machine fields (claim_id, reference_id values in JSON) and non-claim IDs (QT-, POL-, APT-, SLOT-).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = ROOT / "resource" / "scenarios" / "definitions"

# Claim ticket prefixes used in scenarios (not quotes, policies, appointments, slots).
# Negative lookbehinds: do not match inside QT-AP-…, POL-AP-…, APT-PH-…, PAY-PH-…, etc.
CLAIM_ID = re.compile(
    r"(?<!QT-)(?<!POL-)(?<!APT-)(?<!PAY-)\b((?:PH|EL|AP|HA|EV|TV)(?:-[A-Z0-9]+)+)\b"
)

# Whole string is only a claim id token (for trigger_quote).
CLAIM_ONLY = re.compile(
    r"^((?:PH|EL|AP|HA|EV|TV)(?:-[A-Z0-9]+)+)$"
)


def _claim_already_prefixed(text: str, id_start: int) -> bool:
    prefix = text[max(0, id_start - 80) : id_start]
    if re.search(
        r"(?:^|[\s,;:(\[\{'\"]|—)(?:claim|ticket)\s+ID\s+$",
        prefix,
        re.I | re.M,
    ):
        return True
    if re.search(r"(?:^|[\s,;:(\[\{'\"]|—)claim\s+$", prefix, re.I | re.M):
        return True
    # Underscore tool names: new_claim HA-… / cancel_claim PH-… (already claim-scoped).
    if re.search(r"(?:new_claim|cancel_claim)\s+$", prefix, re.I | re.M):
        return True
    return False


def _is_machine_json_value(text: str, id_start: int) -> bool:
    """True if this ID is the value of claim_id / reference_id / etc. in JSON/YAML."""
    window = text[max(0, id_start - 120) : id_start]
    if re.search(
        r'"(?:claim_id|reference_id|quote_id|policy_id|appointment_id|slot_id|payment_id)"\s*:\s*"\s*$',
        window,
    ):
        return True
    if re.search(
        r"(?:claim_id|reference_id|quote_id|policy_id|appointment_id|slot_id|payment_id):\s*\"\s*$",
        window,
    ):
        return True
    return False


def prefix_claim_ids_in_prose(text: str) -> str:
    """Insert 'claim ' before claim IDs when not already prefixed or in machine fields."""

    def repl(m: re.Match[str]) -> str:
        cid = m.group(1)
        start = m.start()
        if _is_machine_json_value(text, start):
            return m.group(0)
        if _claim_already_prefixed(text, start):
            return cid
        return f"claim {cid}"

    return CLAIM_ID.sub(repl, text)


def fix_trigger_quote_value(val: str) -> str:
    v = val.strip()
    if not v:
        return val
    if v.lower().startswith("claim id ") or v.lower().startswith("ticket id "):
        return val
    if v.lower().startswith("claim ") and CLAIM_ONLY.match(
        v.split(None, 1)[1] if len(v.split(None, 1)) > 1 else ""
    ):
        return val
    m = re.match(r"^confirm\s+((?:PH|EL|AP|HA|EV|TV)(?:-[A-Z0-9]+)+)$", v, re.I)
    if m:
        return f"confirm claim {m.group(1)}"
    if CLAIM_ONLY.match(v):
        return f"claim {v}"
    return val


def process_file(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    out = raw

    def sub_trigger_field(field: str, inner: re.Match[str]) -> str:
        val = inner.group(1)
        if field == "trigger_quote":
            new_val = fix_trigger_quote_value(val)
        else:
            new_val = prefix_claim_ids_in_prose(val)
        return f'"{field}":"{new_val}"'

    for field in ("trigger_quote", "trigger_summary"):
        pattern = re.compile(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"')
        out = pattern.sub(lambda m, f=field: sub_trigger_field(f, m), out)

    out = prefix_claim_ids_in_prose(out)

    if out != raw:
        path.write_text(out, encoding="utf-8")
        return True
    return False


def main() -> int:
    changed = []
    for path in sorted(DEFINITIONS.rglob("*.yaml")):
        if process_file(path):
            changed.append(path)
    for p in changed:
        print(f"updated {p.relative_to(ROOT)}")
    print(f"done, {len(changed)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

