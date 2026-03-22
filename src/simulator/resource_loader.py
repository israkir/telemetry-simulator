"""
Load OTEL resource presets from resource/config/resource.yaml.

Attribute keys use vendor.* in YAML; at runtime substitute with the configured prefix (``simulator.config.ATTR_PREFIX``, set via ``--vendor``).
"""

from pathlib import Path
from typing import Any

import yaml

from . import config as sim_config


def _default_resource_path() -> Path:
    """Default path to resource.yaml."""
    # Try repo-root resolution first (relative to this file), then fall back to cwd.
    # Expected structure: <repo>/src/simulator/resource_loader.py -> <repo>/resource/config/resource.yaml
    repo_root = Path(__file__).resolve().parent.parent.parent
    for base in [repo_root, Path.cwd()]:
        candidate = base / "resource" / "config" / "resource.yaml"
        if candidate.exists():
            return candidate
    return repo_root / "resource" / "config" / "resource.yaml"


def _substitute_prefix(attrs: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Replace 'vendor.' with prefix in attribute keys."""
    out: dict[str, Any] = {}
    for k, v in attrs.items():
        if k.startswith("vendor."):
            out[f"{prefix}.{k[7:]}"] = v
        else:
            out[k] = v
    return out


def load_resource_presets(config_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """
    Load control-plane and data-plane resource presets.

    Returns dict with keys "control-plane" and "data-plane", each with
    "attributes" (prefix-substituted) for use as OTEL Resource.
    """
    if config_dir:
        path = Path(config_dir) / "resource.yaml"
    else:
        path = _default_resource_path()
    if not path.exists():
        return {
            "control-plane": {"attributes": {}},
            "data-plane": {"attributes": {}},
        }
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    prefix = sim_config.ATTR_PREFIX
    result: dict[str, dict[str, Any]] = {}
    for key in ("control-plane", "data-plane"):
        preset = data.get(key) or {}
        attrs = preset.get("attributes") or {}
        result[key] = {"attributes": _substitute_prefix(attrs, prefix)}
    return result
