"""
Configuration for vendor-agnostic telemetry simulation.

Allows projects to use the simulator with their own attribute namespace by setting
TELEMETRY_SIMULATOR_ATTR_PREFIX (e.g. "acme" for acme.session.id, acme.turn.status, etc.).
Default prefix is "vendor"; override with TELEMETRY_SIMULATOR_ATTR_PREFIX for your project.
"""

import os


def _get_attr_prefix() -> str:
    """Vendor attribute prefix (e.g. vendor, acme). Default: vendor."""
    return (
        os.environ.get("TELEMETRY_SIMULATOR_ATTR_PREFIX") or "vendor"
    ).strip().lower() or "vendor"


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


def resource_schema_url() -> str:
    """Schema URL for the OTEL resource (resource.schemaUrl)."""
    return os.environ.get("TELEMETRY_SIMULATOR_RESOURCE_SCHEMA_URL", "").strip() or ""


def _env(key: str, default: str) -> str:
    """Get env var with default."""
    return (os.environ.get(key, "").strip() or default).strip()


def resource_attributes(tenant_id: str) -> dict[str, str]:
    """
    Build resource attributes per OTEL resource spec.

    Required: service.name, service.version, prefix.module, prefix.component,
    prefix.tenant.id, prefix.otel.source.
    Recommended: service.instance.id, deployment.environment.name.
    """
    otel_src = _env("TELEMETRY_SIMULATOR_OTEL_SOURCE", "propagated")
    if otel_src not in ("internal", "propagated"):
        otel_src = "propagated"
    attrs: dict[str, str] = {
        "service.name": _env("TELEMETRY_SIMULATOR_SERVICE_NAME", "telemetry-simulator"),
        "service.version": _env("TELEMETRY_SIMULATOR_SERVICE_VERSION", "1.0.0"),
        attr("module"): _env("TELEMETRY_SIMULATOR_MODULE", "data-plane"),
        attr("component"): _env("TELEMETRY_SIMULATOR_COMPONENT", "simulator"),
        attr("tenant.id"): tenant_id,
        attr("otel.source"): otel_src,
    }
    instance_id = _env("SERVICE_INSTANCE_ID", "")
    if instance_id:
        attrs["service.instance.id"] = instance_id
    # deployment.environment.name is resource-level only (recommended by spec).
    attrs["deployment.environment.name"] = _env("DEPLOYMENT_ENVIRONMENT", "") or _env(
        "TELEMETRY_SIMULATOR_DEPLOYMENT_ENVIRONMENT", "development"
    )
    return attrs
