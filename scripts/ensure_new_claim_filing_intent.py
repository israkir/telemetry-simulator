#!/usr/bin/env python3
"""Ensure new_claim user messages explicitly express filing a claim (file/start/open claim wording).

Patches scenario YAML in place without round-tripping through yaml.dump (preserves comments and formatting).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = ROOT / "resource" / "scenarios" / "definitions"

_FILING_OK = re.compile(
    r"(?:"
    r"\bI want to (?:file|start) a claim\b|"
    r"\bI need to (?:file|start) a claim\b|"
    r"\bI(?:'d)? like to (?:file|start) a claim\b|"
    r"\b(?:file|start|submit)\s+a\s+claim\b|"
    r"\bopen\s+(?:a|an|my)\s+[\w\s-]{0,48}claim\b|"
    r"\bopen\s+a\s+claim\b|"
    r"\bstart\s+[\w\s-]{0,32}claim\b|"
    r"\bmake\s+a\s+claim\b|"
    r"\blog\s+a\s+claim\b|"
    r"\bnew\s+claim\b|"
    r"\bneed\s+a\s+claim\b|"
    r"\bwant\s+a\s+claim\b"
    r")",
    re.IGNORECASE,
)

PREFIX = "I want to file a claim — "


def needs_prefix(text: str) -> bool:
    if not text or not str(text).strip():
        return False
    return _FILING_OK.search(text) is None


def _patch_tool_plan_json(tpl: str) -> tuple[str, bool]:
    if "new_claim" not in tpl:
        return tpl, False
    m = re.search(r"\{[\s\S]*\}", tpl)
    if not m:
        return tpl, False
    try:
        parsed = json.loads(m.group())
    except json.JSONDecodeError:
        return tpl, False
    plan = parsed.get("tool_plan")
    if not isinstance(plan, list):
        return tpl, False
    changed = False
    for item in plan:
        if item.get("tool_name") != "new_claim":
            continue
        tq = item.get("trigger_quote") or ""
        ts = item.get("trigger_summary") or ""
        if needs_prefix(tq):
            item["trigger_quote"] = (PREFIX + tq.lstrip()) if tq.strip() else "file a claim"
            changed = True
        if needs_prefix(ts):
            item["trigger_summary"] = (PREFIX + ts.lstrip()) if ts.strip() else "Customer files a claim"
            changed = True
    if not changed:
        return tpl, False
    new_json = json.dumps(parsed, ensure_ascii=False)
    return re.sub(r"\{[\s\S]*\}", new_json, tpl, count=1), True


def _collect_turn_replacements(turn: dict) -> list[tuple[str, str]]:
    """Return (old, new) pairs to apply to file text for this turn."""
    if "new_claim" not in (turn.get("tool_chain") or []):
        return []
    raw = turn.get("vendor.enduser.request.raw")
    if not raw or not isinstance(raw, str):
        return []

    pairs: list[tuple[str, str]] = []
    if needs_prefix(raw):
        new_raw = PREFIX + raw.lstrip()
        pairs.append((raw, new_raw))
        for ev in turn.get("span_events") or []:
            attrs = ev.get("attributes") or {}
            for key in ("vendor.enduser.request.raw", "vendor.agent.tool_selection.input.raw"):
                v = attrs.get(key)
                if v == raw:
                    pairs.append((raw, new_raw))
            tpl = attrs.get("vendor.agent.tool_selection.tool.plan")
            if isinstance(tpl, str):
                if raw in tpl:
                    pairs.append((tpl, tpl.replace(raw, new_raw)))
                else:
                    new_tpl, ch = _patch_tool_plan_json(tpl)
                    if ch:
                        pairs.append((tpl, new_tpl))
    else:
        for ev in turn.get("span_events") or []:
            attrs = ev.get("attributes") or {}
            tpl = attrs.get("vendor.agent.tool_selection.tool.plan")
            if isinstance(tpl, str):
                new_tpl, ch = _patch_tool_plan_json(tpl)
                if ch:
                    pairs.append((tpl, new_tpl))

    # Deduplicate while preserving order (first wins for same old).
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for o, n in pairs:
        if o in seen:
            continue
        seen.add(o)
        out.append((o, n))
    return out


def process_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    if not isinstance(data, dict):
        return False

    all_pairs: list[tuple[str, str]] = []
    for eu in data.get("endusers") or []:
        for turn in eu.get("turns") or []:
            all_pairs.extend(_collect_turn_replacements(turn))

    if not all_pairs:
        return False

    # Longest old string first to avoid partial overlaps.
    all_pairs.sort(key=lambda p: len(p[0]), reverse=True)
    out = text
    for old, new in all_pairs:
        if old == new or old not in out:
            continue
        out = out.replace(old, new)
    if out == text:
        return False
    path.write_text(out, encoding="utf-8")
    return True


def main() -> int:
    n = 0
    for path in sorted(DEFINITIONS.rglob("*.yaml")):
        if process_file(path):
            print(f"updated {path.relative_to(ROOT)}")
            n += 1
    print(f"done, {n} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
