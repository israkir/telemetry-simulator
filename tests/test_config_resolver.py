import pytest

from simulator.config import load_config
from simulator.scenarios.config_resolver import resolve_context


def test_tenant_id_override_takes_precedence_over_tenant_key() -> None:
    config = load_config()
    resolved = resolve_context(
        config,
        tenant_key="toro",
        default_tenant_id="OVERRIDE_TENANT_ID",
    )
    assert resolved.tenant_id == "OVERRIDE_TENANT_ID"


def test_tenant_id_resolution_fails_without_tenant_key_and_default() -> None:
    config = load_config()
    with pytest.raises(ValueError, match="Unable to resolve tenant id"):
        resolve_context(config, tenant_key=None, default_tenant_id=None)

