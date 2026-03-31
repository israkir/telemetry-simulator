#!/usr/bin/env python3
"""Extract unique enduser prompts from scenario YAML, grouped by interaction context.

Context is derived from each turn's ``tool_chain``: tool names are mapped to coarse
buckets (new claim, payment, scheduling, cancellation, claim status, etc.). Turns
that invoke multiple tools are grouped under a composite key made from the sorted
unique bucket labels for that turn.

Usage:
  ./scripts/extract_user_prompts_by_context.py
  ./scripts/extract_user_prompts_by_context.py --format json --output prompts.json
  ./scripts/extract_user_prompts_by_context.py --provenance
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

# Map MCP tool names to human-oriented context buckets (order not significant).
TOOL_TO_CONTEXT: dict[str, str] = {
    "new_claim": "new_claim",
    "claim_status": "claim_status",
    "pay": "payment",
    "cancel_claim": "cancellation",
    "get_available_slots": "scheduling",
    "update_appointment": "scheduling",
    "upload_documents": "document_upload",
    "choose_insurance": "insurance",
    "buy_insurance": "insurance",
    "cancel_product": "product_cancellation",
}


def _default_definitions_dir(repo_root: Path) -> Path:
    return repo_root / "resource" / "scenarios" / "definitions"


def _normalize_prompt(text: str) -> str:
    return " ".join(text.split()).strip()


def _turn_labels(tool_chain: list[str] | None) -> list[str]:
    if not tool_chain:
        return ["no_tool"]
    labels: set[str] = set()
    for tool in tool_chain:
        labels.add(TOOL_TO_CONTEXT.get(tool, tool))
    return sorted(labels)


def _context_key(labels: list[str]) -> str:
    return " + ".join(labels)


def _iter_turns(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    endusers = data.get("endusers") or []
    if not isinstance(endusers, list):
        return out
    for eu in endusers:
        if not isinstance(eu, dict):
            continue
        eid = str(eu.get("id", ""))
        for turn in eu.get("turns") or []:
            if isinstance(turn, dict):
                out.append((eid, turn))
    return out


def _prompt_from_turn(turn: dict[str, Any]) -> str:
    raw = turn.get("vendor.enduser.request.raw") or turn.get("request_raw") or ""
    return _normalize_prompt(str(raw))


def _tool_chain_from_turn(turn: dict[str, Any]) -> list[str]:
    tc = turn.get("tool_chain")
    if not isinstance(tc, list):
        return []
    return [str(x) for x in tc]


def collect_prompts(
    definitions_dir: Path,
    *,
    with_provenance: bool,
) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, list[dict[str, Any]]]],
]:
    """Return (context_key -> sorted unique prompts, provenance: context -> prompt -> meta rows)."""
    prompts_by_context: dict[str, set[str]] = defaultdict(set)
    provenance: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    by_path: list[Path] = sorted(definitions_dir.rglob("*.yaml"))
    for path in by_path:
        if "conventions" in path.parts and path.name == "semconv.yaml":
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict) or "endusers" not in data:
            continue
        scenario_name = str(data.get("name", path.stem))
        rel = path.relative_to(definitions_dir) if path.is_relative_to(definitions_dir) else path

        for enduser_id, turn in _iter_turns(data):
            prompt = _prompt_from_turn(turn)
            if not prompt:
                continue
            labels = _turn_labels(_tool_chain_from_turn(turn))
            key = _context_key(labels)
            prompts_by_context[key].add(prompt)
            if with_provenance:
                provenance[key][prompt].append(
                    {
                        "file": str(rel),
                        "scenario": scenario_name,
                        "enduser_id": enduser_id,
                        "turn_index": turn.get("turn_index"),
                        "tool_chain": _tool_chain_from_turn(turn),
                    }
                )

    sorted_prompts = {k: sorted(v) for k, v in sorted(prompts_by_context.items())}
    sorted_prov = {
        ctx: {p: rows for p, rows in sorted(per_prompt.items())}
        for ctx, per_prompt in sorted(provenance.items())
    }
    return sorted_prompts, sorted_prov


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--definitions-dir",
        type=Path,
        default=None,
        help="Scenario definitions root (default: <repo>/resource/scenarios/definitions)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write to this file instead of stdout",
    )
    parser.add_argument(
        "--provenance",
        action="store_true",
        help="Include per-prompt source refs in JSON (file, scenario, enduser, turn)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    definitions_dir = args.definitions_dir or _default_definitions_dir(repo_root)
    if not definitions_dir.is_dir():
        print(f"Definitions directory not found: {definitions_dir}", file=sys.stderr)
        return 1

    prompts, prov = collect_prompts(definitions_dir, with_provenance=args.provenance)

    if args.format == "json":
        payload: dict[str, Any] = {"by_context": prompts}
        if args.provenance:
            payload["provenance_by_context"] = prov
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    else:
        if args.provenance:
            print("--provenance only applies to --format json", file=sys.stderr)
            return 2
        import io

        buf = io.StringIO()
        for ctx, plist in prompts.items():
            buf.write(f"## {ctx} ({len(plist)} unique prompt(s))\n")
            for i, p in enumerate(plist, start=1):
                buf.write(f"{i}. {p}\n")
        text = buf.getvalue()

    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
