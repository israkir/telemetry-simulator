"""
Configuration for vendor-agnostic telemetry simulation.

Allows projects to use the simulator with their own attribute namespace by setting
VENDOR (e.g. "acme" for acme.session.id, acme.turn.status, etc.).
Default prefix is "vendor"; override with VENDOR for your project.

Resource attributes and resource.schemaUrl are loaded from config/resource.yaml.
Config and scenario files live outside src/ under resource/ (resource/config/,
resource/scenarios/). When running from source, resource/ at project root is used.
When the package is installed, set TELEMETRY_SIMULATOR_ROOT to a directory
containing config/ and scenarios/ (e.g. the project's resource/ folder).
"""

import os
from pathlib import Path
from typing import Any

import yaml


def get_resources_root() -> Path:
    """Return the root directory for config and scenario resources.

    Resolution order:
    1. TELEMETRY_SIMULATOR_ROOT env var (must contain config/ and scenarios/)
    2. resource/ under directory containing pyproject.toml (when running from source)
    3. simulator/resources/ next to this package (when installed; set TELEMETRY_SIMULATOR_ROOT if not present)
    """
    env_root = os.environ.get("TELEMETRY_SIMULATOR_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if p.is_dir():
            return p
    # Walk up from this file (e.g. .../src/simulator/config.py) looking for pyproject.toml
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate / "resource"
    # Installed package: no bundled resources; caller should set TELEMETRY_SIMULATOR_ROOT
    return Path(__file__).resolve().parent / "resources"


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


# Semantic conventions: canonical allowed values from conventions/semconv.yaml.
# Use these for error paths and happy-path flows so emitted data is semantically correct.
SEMCONV_ERROR_TYPE_VALUES = (
    "timeout",
    "unavailable",
    "invalid_arguments",
    "tool_error",
    "protocol_error",
)
SEMCONV_STEP_OUTCOME_VALUES = ("success", "fail", "skipped")


_RESOURCES_ROOT = get_resources_root()
_CONFIG_PATH = _RESOURCES_ROOT / "config" / "config.yaml"
CONFIG_PATH = _CONFIG_PATH
_RESOURCE_CONFIG_PATH = _RESOURCES_ROOT / "config" / "resource.yaml"
# Default semantic-conventions path when SEMCONV / --semconv not set.
DEFAULT_SEMCONV_PATH = _RESOURCES_ROOT / "scenarios" / "conventions" / "semconv.yaml"

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


def _expand_resource_attributes(raw: dict[str, str]) -> dict[str, str]:
    """Expand prefix-relative keys (module, component, otel.source) and return full attribute dict."""
    result: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, str) and k not in _PREFIX_RELATIVE_KEYS:
            result[k] = v
    for rel_key in _PREFIX_RELATIVE_KEYS:
        if rel_key in raw and isinstance(raw[rel_key], str):
            result[attr(rel_key)] = raw[rel_key]
    return result


def _load_resource_config() -> tuple[str, dict[str, str], dict[str, str]]:
    """Load resource from config/resource.yaml. Returns (schema_url, control_plane_attrs, data_plane_attrs)."""
    default_url = "https://example.com/otel/schema/1.0.0"
    empty_attrs: dict[str, str] = {}
    data = load_yaml(_RESOURCE_CONFIG_PATH)
    if not isinstance(data, dict):
        return default_url, empty_attrs.copy(), empty_attrs.copy()
    # Expected shape: control_plane and data_plane sections
    if "control_plane" not in data or "data_plane" not in data:
        return default_url, empty_attrs.copy(), empty_attrs.copy()
    schema_url = data.get("schema_url") if isinstance(data.get("schema_url"), str) else default_url
    if isinstance(schema_url, str) and schema_url.strip():
        default_url = schema_url.strip()
    cp_block = data.get("control_plane")
    dp_block = data.get("data_plane")
    cp_raw = cp_block.get("attributes") if isinstance(cp_block, dict) else None
    dp_raw = dp_block.get("attributes") if isinstance(dp_block, dict) else None
    cp_attrs = (
        _expand_resource_attributes(cp_raw) if isinstance(cp_raw, dict) else empty_attrs.copy()
    )
    dp_attrs = (
        _expand_resource_attributes(dp_raw) if isinstance(dp_raw, dict) else empty_attrs.copy()
    )
    return default_url, cp_attrs, dp_attrs


def resource_schema_url() -> str:
    """Schema URL for the OTEL resource (resource.schemaUrl). From config/resource.yaml."""
    yaml_url, _, _ = _load_resource_config()
    return yaml_url


def get_default_tenant_id(config_path: Path | None = None) -> str:
    """
    Return the first tenant id from config (tenants map).
    Used by defaults and config_resolver; avoids circular imports.
    """
    path = config_path or _CONFIG_PATH
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError("Config is missing or invalid")
    tenants = data.get("tenants") or {}
    if not isinstance(tenants, dict) or not tenants:
        raise ValueError("Config has no tenants; at least one tenant is required")
    for t in tenants.values():
        if isinstance(t, dict):
            tid = t.get("id")
            if isinstance(tid, str) and tid.strip():
                return tid.strip()
    raise ValueError("Config has no tenants; at least one tenant is required")


# Data-plane component values (resource attribute); per semconv allowed_values.
DATA_PLANE_COMPONENT_VALUES = (
    "orchestrator",
    "planning",
    "retrieval",
    "llm",
    "mcp_client",
    "tool_recommender",
)


def resource_attributes(
    tenant_id: str,
    component: str | None = None,
) -> dict[str, str]:
    """
    Build resource attributes per OTEL resource spec.

    Control-plane spans: use control_plane config (component=None).
    Data-plane spans: use data_plane config and set prefix.component to the given value.
    Values are loaded from config/resource.yaml (control_plane / data_plane sections).
    prefix.tenant.id is set from the given tenant_id (scenario context).
    """
    _, cp_attrs, dp_attrs = _load_resource_config()
    base = dp_attrs if component is not None else cp_attrs
    attrs: dict[str, str] = dict(base)
    attrs[attr("tenant.id")] = tenant_id
    if component is not None:
        attrs[attr("component")] = component
    # Normalize otel.source to allowed values only
    otel_src = attrs.get(attr("otel.source"), "propagated")
    if otel_src not in ("internal", "propagated"):
        attrs[attr("otel.source")] = "propagated"
    return attrs
