"""Plugin protocol types.

Dataclasses defining the contract between plugins and the registry:
ServerManifest, SlashCommand, TokenDefinition, PluginRegistration.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from episodic.utility.voice.grammar_types import GrammarRule


@dataclass(frozen=True)
class ServerManifest:
    """Describes the MCP server a plugin connects to."""
    server_id: str
    display_name: str
    command: str                       # e.g. "npx"
    args: List[str] = field(default_factory=list)
    env_vars: List[str] = field(default_factory=list)  # Required env vars
    connect_policy: str = "manual"     # "manual", "on-demand", "startup"


@dataclass(frozen=True)
class SlashCommand:
    """A slash command contributed by a plugin."""
    name: str                          # e.g. "/cal"
    aliases: List[str] = field(default_factory=list)
    category: str = ""                 # Help category
    description: str = ""
    domain: str = ""                   # Extraction domain scope
    completions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TokenDefinition:
    """A lexer token contributed by a plugin."""
    word: str
    token_kind: str


@dataclass
class PluginRegistration:
    """Everything a plugin contributes to the system."""
    name: str
    manifest: ServerManifest
    slash_commands: List[SlashCommand] = field(default_factory=list)
    tokens: List[TokenDefinition] = field(default_factory=list)
    grammar_rules: List[GrammarRule] = field(default_factory=list)
    tool_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    adapter_map: Dict[str, Any] = field(default_factory=dict)
    help_fn: Optional[Callable[[], str]] = None
    help_category: Optional[str] = None
    extraction_contribution: Optional[Any] = None  # PluginExtractionContribution
    arg_extractors: Dict[str, Callable] = field(default_factory=dict)
