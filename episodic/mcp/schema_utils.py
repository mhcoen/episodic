"""
MCP Schema Utilities.

Normalize JSON Schemas from MCP tool discovery for consistent fingerprinting.
Handles the mcp-gsuite bug where 'required' is a boolean instead of an array.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a JSON Schema for consistent fingerprinting.

    Fixes:
    - Strips 'required' fields that are booleans (mcp-gsuite bug)
    - Sorts keys deterministically
    - Removes 'description' fields (cosmetic, not structural)
    """
    if not isinstance(schema, dict):
        return schema

    result = {}
    for key in sorted(schema.keys()):
        value = schema[key]

        # Strip 'required' if it's a boolean instead of an array
        if key == "required" and isinstance(value, bool):
            continue

        # Strip description fields (cosmetic)
        if key == "description":
            continue

        # Recurse into nested dicts
        if isinstance(value, dict):
            result[key] = normalize_schema(value)
        elif isinstance(value, list):
            result[key] = [
                normalize_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    return result


def canonical_json(obj: Any) -> str:
    """Produce canonical JSON (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def schema_fingerprint(schema: Dict[str, Any]) -> str:
    """
    Compute SHA-256 fingerprint of a normalized JSON Schema.

    The fingerprint is used for allowlist lookup in the security pipeline.
    """
    normalized = normalize_schema(schema)
    canonical = canonical_json(normalized)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
