"""
Configuration for vendor-agnostic telemetry simulation.

Allows projects to use the simulator with their own attribute namespace by setting
VENDOR (e.g. "acme" for acme.session.id, acme.turn.status, etc.).
Default prefix is "vendor"; override with VENDOR for your project.

Resource attributes and resource.schemaUrl are loaded only from scenarios_config.yaml
(resource.attributes, resource.schema_url); env vars are not used for these.
"""

import os
from pathlib import Path
from typing import Any

import yaml


def _get_attr_prefix() -> str:
    """Vendor attribute prefix (e.g. vendor, acme). Default: vendor."""
    return (os.environ.get("VENDOR") or "vendor").strip().lower() or "vendor"


def _get_vendor_name() -> str:
    """Display name for validation messages. Default: capitalize(prefix)."""
    custom = os.environ.get("TELEMETRY_SIMULATOR_VENDOR_NAME", "").strip()
    if custom:
        return custom
    prefix = _get_attr_prefix()
    return prefix.capitalize() if prefix else "Vendor"


# Module-level config; read from env at import time so CLI and tests can override via env.
ATTR_PREFIX = _get_attr_prefix()
VENDOR_NAME = _get_vendor_name()


def attr(suffix: str) -> str:
    """Return full attribute name with configured prefix (e.g. attr('session.id') -> 'vendor.session.id')."""
    if not suffix:
        return ATTR_PREFIX
    return f"{ATTR_PREFIX}.{suffix}" if ATTR_PREFIX else suffix


def span_name(suffix: str) -> str:
    """Return full span name with configured vendor prefix (e.g. span_name('a2a.orchestrate') -> 'vendor.a2a.orchestrate')."""
    if not suffix:
        return ATTR_PREFIX
    return f"{ATTR_PREFIX}.{suffix}" if ATTR_PREFIX else suffix


def schema_version_attr() -> str:
    """Attribute key for schema version (prefix.schema.version or schema.version)."""
    return attr("schema.version")


_SCENARIOS_CONFIG_PATH = Path(__file__).resolve().parent / "scenarios" / "scenarios_config.yaml"
SCENARIOS_CONFIG_PATH = _SCENARIOS_CONFIG_PATH

# Keys in resource.attributes that are prefix-relative (expanded with attr() when loading from YAML).
_PREFIX_RELATIVE_KEYS = frozenset({"module", "component", "otel.source"})


def load_yaml(path: Path, default: Any = None) -> Any:
    """Load YAML file; return default on missing file or parse error."""
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        with path.open(encoding="utf-8") as f:
            data: Any = yaml.safe_load(f)
    except Exception:
        return default
    return data if isinstance(data, dict) else default


def _load_resource_config() -> tuple[str, dict[str, str]]:
    """Load resource.schema_url and resource.attributes from scenarios_config.yaml. Returns (schema_url, attributes)."""
    default_url = "https://example.com/otel/schema/1.0.0"
    default_attrs: dict[str, str] = {}
    data = load_yaml(_SCENARIOS_CONFIG_PATH)
    if not data:
        return default_url, default_attrs
    resource = data.get("resource")
    if not isinstance(resource, dict):
        return default_url, default_attrs
    schema_url = resource.get("schema_url")
    if isinstance(schema_url, str) and schema_url.strip():
        default_url = schema_url.strip()
    raw = resource.get("attributes")
    if not isinstance(raw, dict):
        return default_url, default_attrs
    for k, v in raw.items():
        if isinstance(v, str) and k not in _PREFIX_RELATIVE_KEYS:
            default_attrs[k] = v
    # Expand prefix-relative keys so they are stored under full attribute names
    for rel_key in _PREFIX_RELATIVE_KEYS:
        if rel_key in raw and isinstance(raw[rel_key], str):
            default_attrs[attr(rel_key)] = raw[rel_key]
    return default_url, default_attrs


def resource_schema_url() -> str:
    """Schema URL for the OTEL resource (resource.schemaUrl). From scenarios_config.yaml only."""
    yaml_url, _ = _load_resource_config()
    return yaml_url


def resource_attributes(tenant_id: str) -> dict[str, str]:
    """
    Build resource attributes per OTEL resource spec.

    Values are loaded only from scenarios_config.yaml (resource.attributes).
    prefix.tenant.id is set from the given tenant_id (scenario context).
    """
    _, yaml_attrs = _load_resource_config()
    attrs: dict[str, str] = dict(yaml_attrs)
    attrs[attr("tenant.id")] = tenant_id
    # Normalize otel.source to allowed values only
    otel_src = attrs.get(attr("otel.source"), "propagated")
    if otel_src not in ("internal", "propagated"):
        attrs[attr("otel.source")] = "propagated"
    return attrs
