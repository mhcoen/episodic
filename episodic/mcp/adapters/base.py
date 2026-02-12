"""
Base Argument Adapter.

Translates UtilityQuery.args (CFG-normalized) to
MCP tool argument dicts (server-specific schema).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from episodic.mcp.dispatch_types import MCPResolution


DEFAULT_ACCOUNT = "michaelbot2718@gmail.com"


class ArgumentAdapter:
    """
    Base class for argument adapters.

    One adapter per intent→tool mapping. Adapters translate
    CFG-normalized args to MCP tool-specific argument dicts.
    """

    def adapt(
        self,
        query_args: Dict[str, Any],
        resolution: MCPResolution,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Translate CFG args to MCP tool args."""
        raise NotImplementedError

    def _resolve_account(
        self,
        query_args: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Resolve the account (email) to use for the MCP call."""
        if config and "default_account" in config:
            return config["default_account"]
        return query_args.get("account", DEFAULT_ACCOUNT)
