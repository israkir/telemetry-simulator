"""
Tenant from environment only (aligned with data-plane collector).

Simulator prefers TELEMETRY_TENANT_UUIDS / TELEMETRY_TENANT_WEIGHTS when set (e.g. in container);
otherwise TENANT_UUID (single or comma-separated) and optional TENANT_WEIGHTS.
"""

import os


def get_default_tenant_ids() -> list[str]:
    """Tenant IDs for generation: from TELEMETRY_TENANT_UUIDS or TENANT_UUID env (required)."""
    raw = (
        os.environ.get("TELEMETRY_TENANT_UUIDS", "").strip()
        or os.environ.get("TENANT_UUID", "").strip()
    )
    if not raw:
        raise SystemExit(
            "TENANT_UUID or TELEMETRY_TENANT_UUIDS must be set (aligns with data-plane collector)."
        )
    return [t.strip() for t in raw.split(",") if t.strip()]


def _default_tenant_weights(n: int) -> list[float]:
    """Realistic skewed distribution: first tenant gets most traffic, then decay."""
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    # Skew: ~50%, ~28%, ~15%, ~7% for 4; generalizes via 2^(-i) then normalize
    weights = [0.5 ** (i + 1) for i in range(n)]
    total = sum(weights)
    return [w / total for w in weights]


def get_tenant_distribution() -> dict[str, float]:
    """Tenant IDs and weights from env. Uses TELEMETRY_TENANT_WEIGHTS or TENANT_WEIGHTS if set, else realistic default."""
    ids = get_default_tenant_ids()
    raw_weights = (
        os.environ.get("TELEMETRY_TENANT_WEIGHTS", "").strip()
        or os.environ.get("TENANT_WEIGHTS", "").strip()
    )
    if raw_weights:
        parts = [p.strip() for p in raw_weights.split(",") if p.strip()]
        if len(parts) != len(ids):
            raise SystemExit(
                f"TENANT_WEIGHTS must have {len(ids)} comma-separated values (one per tenant)."
            )
        try:
            weights = [float(x) for x in parts]
        except ValueError:
            raise SystemExit("TENANT_WEIGHTS must be comma-separated numbers.") from None
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise SystemExit("TENANT_WEIGHTS must be non-negative and sum to a positive number.")
        total = sum(weights)
        return dict(zip(ids, (w / total for w in weights), strict=True))
    return dict(zip(ids, _default_tenant_weights(len(ids)), strict=True))
