#!/usr/bin/env python3
"""Emit resource/scenarios/definitions/all_definitions_trace_spec.yaml from all scenario YAMLs.

Slim trace list: scenario, user_input, mcp_server, turn_index, per-tool parameters,
status/error, and rough ms latencies (see catalog_metadata.latency_statistics_note).

Usage:
  source .venv/bin/activate && python scripts/generate_all_definitions_trace_spec.py
"""

from __future__ import annotations

import sys
import yaml
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFINITIONS_DIR = REPO_ROOT / "resource" / "scenarios" / "definitions"
OUTPUT_PATH = DEFINITIONS_DIR / "all_definitions_trace_spec.yaml"

SKIP_NAMES = frozenset({"all_definitions_trace_spec.yaml"})

# Rough per-tool RTT bands (ms), inclusive range for deterministic pick — not calibrated data.
BASE_MS = {
    "new_claim": (142, 268),
    "claim_status": (88, 195),
    "update_appointment": (96, 214),
    "cancel_claim": (74, 188),
    "upload_documents": (105, 240),
    "choose_insurance": (91, 205),
    "buy_insurance": (118, 256),
    "cancel_product": (81, 198),
    "pay": (165, 412),
    "get_available_slots": (124, 289),
}


def jitter(key: str, lo: int, hi: int) -> int:
    h = int(sha256(key.encode()).hexdigest(), 16)
    return lo + (h % (hi - lo + 1))


def latency_for(tool: str, key: str) -> int:
    lo, hi = BASE_MS.get(tool, (100, 250))
    return jitter(key, lo, hi)


def tool_status_error(tool: str, res: dict | None) -> tuple[str, str | None]:
    """Return (status, error_message_or_null)."""
    if not isinstance(res, dict):
        return "unknown", "missing_or_invalid_result"
    st = str(res.get("status", "")).lower()
    if st in ("rejected", "failed", "error", "declined"):
        return "error", st
    if tool == "pay":
        if st in ("not_captured", "failed", "declined"):
            return "error", st or "payment_not_completed"
        if st in ("captured", "authorized"):
            return "success", None
    if tool == "cancel_claim" and st == "rejected":
        return "error", "rejected"
    return "success", None


def attempt_latencies(
    tool: str, scenario_key: str, attempts_meta: list | None
) -> tuple[list[dict] | None, int | None]:
    if not attempts_meta:
        return None, None
    out: list[dict] = []
    total = 0
    for i, att in enumerate(attempts_meta):
        key = f"{scenario_key}|{tool}|attempt{i+1}"
        base = latency_for(tool, key)
        if isinstance(att, dict) and att.get("outcome") == "fail":
            base = int(base * 1.85)
        out.append(
            {
                "attempt_index": i + 1,
                "outcome": att.get("outcome") if isinstance(att, dict) else str(att),
                "error.type": (att.get("error.type") if isinstance(att, dict) else None),
                "latency_ms": base,
            }
        )
        total += base
    return out, total


def main() -> int:
    files = sorted(
        p
        for p in DEFINITIONS_DIR.rglob("*.yaml")
        if p.name not in SKIP_NAMES and "conventions" not in p.parts
    )
    catalog: list[dict] = []

    for path in files:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as e:
            print(f"skip {path.relative_to(DEFINITIONS_DIR)}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict) or "endusers" not in data:
            continue
        scenario_name = data.get("name", path.stem)
        for eu in data.get("endusers") or []:
            if not isinstance(eu, dict):
                continue
            mcp = eu.get("mcp_server") or data.get("mcp_server")
            for turn in eu.get("turns") or []:
                if not isinstance(turn, dict):
                    continue
                raw = turn.get("vendor.enduser.request.raw") or turn.get("request_raw") or ""
                if isinstance(raw, str):
                    raw = raw.strip()
                tc = turn.get("tool_chain")
                if not isinstance(tc, list):
                    tc = []
                args = turn.get("gen_ai.tool.call.arguments") or {}
                results = turn.get("gen_ai.tool.call.result") or {}
                retries = turn.get("mcp_tool_retries") or {}
                euid = str(eu.get("id", ""))

                tools_out: list[dict] = []
                for idx, tool in enumerate(tc):
                    tkey = f"{scenario_name}|{euid}|{turn.get('turn_index')}|{tool}|{idx}"
                    res = results.get(tool) if isinstance(results, dict) else None
                    status, err = tool_status_error(tool, res if isinstance(res, dict) else None)
                    entry: dict = {
                        "tool": tool,
                        "input_parameters": args.get(tool) if isinstance(args, dict) else None,
                        "status": status,
                    }
                    if err is not None:
                        entry["error"] = err
                    rmeta = retries.get(tool) if isinstance(retries, dict) else None
                    if isinstance(rmeta, dict) and rmeta.get("attempts"):
                        alist, total_ms = attempt_latencies(
                            tool,
                            f"{scenario_name}|{euid}|{turn.get('turn_index')}",
                            rmeta["attempts"],
                        )
                        if alist:
                            entry["retry_attempts"] = alist
                            if total_ms is not None:
                                entry["latency_total_ms"] = total_ms
                        else:
                            entry["latency_ms"] = latency_for(tool, tkey)
                    else:
                        entry["latency_ms"] = latency_for(tool, tkey)
                    tools_out.append(entry)

                catalog.append(
                    {
                        "scenario": scenario_name,
                        "user_input": raw,
                        "mcp_server": mcp,
                        "turn_index": turn.get("turn_index"),
                        "tools": tools_out,
                    }
                )

    # Avoid literal ': ' mid-string (PyYAML can split plain scalars on colon+space).
    latency_note = (
        "Rough illustrative milliseconds per tool; not sampled from a distribution and not tied to "
        "scenario latency_profiles. Each latency_ms is SHA-256(key) modulo span inside fixed per-tool "
        "[low, high] bands; key is scenario|enduser_id|turn_index|tool|chain_index. Retry rows expand to "
        "retry_attempts with the same rule per attempt; failed attempts multiply the draw by 1.85. "
        "latency_total_ms sums attempts. Stable across regenerations for fixtures; not an estimate of "
        "production mean, p50, or p95."
    )
    header = {
        "title": "Trace specification — all scenario definitions (slim)",
        "generated_from": "resource/scenarios/definitions",
        "trace_count": len(catalog),
        "latency_statistics_note": latency_note,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("# AUTO-GENERATED — do not hand-edit; regenerate via scripts/generate_all_definitions_trace_spec.py\n\n")
        yaml.safe_dump(
            {"catalog_metadata": header, "traces": catalog},
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )
    print(f"Wrote {OUTPUT_PATH} ({len(catalog)} traces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
