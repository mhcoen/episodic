"""Google Workspace plugin for episodic MCP.

Side-effect-free registration: imports from existing locations and
re-exports as a PluginRegistration. Called by the plugin registry
during register_all().
"""

from episodic.mcp.plugins._protocol import PluginRegistration

from .manifest import GSUITE_MANIFEST
from .tokens import GSUITE_TOKENS
from .grammar import get_grammar_rules, get_arg_extractors
from .adapters import get_adapter_map, get_tool_map
from .slash import GSUITE_SLASH_COMMANDS
from .help import show_calendar_email_help
from .extraction_contrib import build_extraction_contribution


def register() -> PluginRegistration:
    """Register the gsuite plugin. Side-effect-free."""
    return PluginRegistration(
        name="gsuite",
        manifest=GSUITE_MANIFEST,
        slash_commands=GSUITE_SLASH_COMMANDS,
        tokens=GSUITE_TOKENS,
        grammar_rules=get_grammar_rules(),
        tool_map=get_tool_map(),
        adapter_map=get_adapter_map(),
        help_fn=show_calendar_email_help,
        help_category="Calendar & Email",
        extraction_contribution=build_extraction_contribution(),
        arg_extractors=get_arg_extractors(),
    )
