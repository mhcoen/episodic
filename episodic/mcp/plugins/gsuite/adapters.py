"""Adapter bridge for the gsuite plugin.

Imports existing adapters from episodic.mcp.adapters and re-exports
them as the plugin's adapter_map and tool_map.
"""

from typing import Any, Dict

from episodic.mcp.adapters import ARGUMENT_ADAPTERS
from episodic.mcp.dispatch_types import DEFAULT_INTENT_MAPPING


def get_adapter_map() -> Dict[str, Any]:
    """Return intent -> adapter class mapping for gsuite intents."""
    return dict(ARGUMENT_ADAPTERS)


def get_tool_map() -> Dict[str, Dict[str, Any]]:
    """Return intent -> tool mapping for gsuite intents."""
    return dict(DEFAULT_INTENT_MAPPING)
