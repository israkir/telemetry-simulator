"""
Default tenant for the simulator.

Tenant identity comes only from config: scenarios/config/config.yaml tenants map.
The default tenant id is the first tenant in config (used when a scenario has no context).
"""

from .config import get_default_tenant_id


def get_default_tenant_ids() -> list[str]:
    """Return the single default tenant id from config (first tenant)."""
    return [get_default_tenant_id()]
