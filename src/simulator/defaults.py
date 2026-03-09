"""
Default tenant for the simulator.

By default, the tenant id comes from config: scenarios/config/config.yaml tenants map.
When TELEMETRY_SIMULATOR_TENANT_ID is set (e.g. via CLI --tenant-id), that value is used
as the default tenant id instead.
The default tenant id is used when a scenario has no explicit context.
"""

from .config import get_default_tenant_id


def get_default_tenant_ids() -> list[str]:
    """Return the single default tenant id from config (first tenant)."""
    return [get_default_tenant_id()]
