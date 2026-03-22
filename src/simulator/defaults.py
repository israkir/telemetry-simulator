"""Default values for tenant IDs and config paths."""

from .config import get_default_tenant_id, load_config


def get_default_tenant_ids() -> list[str]:
    """Return list of tenant IDs from config (for display). Empty if no config."""
    config = load_config()
    tenants = config.get("tenants") or {}
    ids: list[str] = []
    for v in tenants.values():
        if isinstance(v, dict) and "id" in v:
            ids.append(v["id"])
        else:
            ids.append(str(v))
    default = get_default_tenant_id(config)
    if default and default not in ids:
        ids.insert(0, default)
    return ids
