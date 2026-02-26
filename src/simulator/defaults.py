"""
Tenant defaults for the simulator.

Tenant identity is driven by the shared `scenarios_config.yaml` file:

- Primary source: `tenant.id` in scenarios_config.yaml
- Fallback when config or id is missing: `"test-tenant-001"`

All telemetry is generated for a **single** tenant id by default.
"""

from .config import SCENARIOS_CONFIG_PATH, load_yaml

_DEFAULT_TENANT_ID = "test-tenant-001"


def _load_tenant_id_from_config() -> str:
    """
    Load the default tenant id from scenarios_config.yaml.

    Config shape:

    tenant:
      id: "9cafa427-504f-4bb7-a09f-ec1f5524facf"
      display_name: "Toro Insurance"
    """
    data = load_yaml(SCENARIOS_CONFIG_PATH)
    tenant_data = data.get("tenant")
    if isinstance(tenant_data, dict):
        tenant_id = tenant_data.get("id")
        if isinstance(tenant_id, str) and tenant_id.strip():
            return tenant_id.strip()

    return _DEFAULT_TENANT_ID


def get_default_tenant_ids() -> list[str]:
    """Return the single default tenant id from scenarios_config.yaml (or fallback)."""
    return [_load_tenant_id_from_config()]


def get_tenant_distribution() -> dict[str, float]:
    """
    Return tenant distribution for generation.

    Currently a single tenant with weight 1.0, using the id from scenarios_config.yaml
    or the `"test-tenant-001"` fallback when not configured.
    """
    tenant_id = _load_tenant_id_from_config()
    return {tenant_id: 1.0}
