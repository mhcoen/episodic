"""MCP Plugin Registry.

Manages plugin discovery, registration, and state lifecycle.
Plugins transition through states:
  DISCOVERED -> REGISTERED -> CONNECTED -> ACTIVE -> DISCONNECTED
  Any state -> DISABLED
"""

from __future__ import annotations

import enum
import logging
from typing import Callable, Dict, List, Optional

from ._protocol import PluginRegistration, SlashCommand
from ._token_registry import TokenRegistry

logger = logging.getLogger(__name__)


class PluginState(enum.Enum):
    """Lifecycle state of a plugin."""
    DISCOVERED = "discovered"
    REGISTERED = "registered"
    CONNECTED = "connected"
    ACTIVE = "active"
    DISCONNECTED = "disconnected"
    DISABLED = "disabled"


# Valid state transitions
_TRANSITIONS: Dict[PluginState, frozenset] = {
    PluginState.DISCOVERED: frozenset({PluginState.REGISTERED, PluginState.DISABLED}),
    PluginState.REGISTERED: frozenset({PluginState.CONNECTED, PluginState.ACTIVE, PluginState.DISABLED}),
    PluginState.CONNECTED: frozenset({PluginState.ACTIVE, PluginState.DISCONNECTED, PluginState.DISABLED}),
    PluginState.ACTIVE: frozenset({PluginState.DISCONNECTED, PluginState.DISABLED}),
    PluginState.DISCONNECTED: frozenset({PluginState.CONNECTED, PluginState.DISABLED}),
    PluginState.DISABLED: frozenset({PluginState.DISCOVERED}),
}


class _PluginEntry:
    """Internal bookkeeping for a registered plugin."""
    __slots__ = ("name", "state", "registration", "register_fn")

    def __init__(
        self,
        name: str,
        state: PluginState,
        register_fn: Optional[Callable[[], PluginRegistration]] = None,
        registration: Optional[PluginRegistration] = None,
    ) -> None:
        self.name = name
        self.state = state
        self.register_fn = register_fn
        self.registration = registration


class PluginRegistry:
    """Central registry for MCP plugins.

    Thread-safe enough for single-threaded async (no locks needed).
    Call register_all() to discover and register all known plugins.
    Idempotent — safe to call multiple times.
    """

    def __init__(self) -> None:
        self._plugins: Dict[str, _PluginEntry] = {}
        self._slash_commands: Dict[str, str] = {}  # cmd_name -> plugin_name
        self._token_registry = TokenRegistry()
        self._initialized: bool = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def token_registry(self) -> TokenRegistry:
        return self._token_registry

    def discover(
        self,
        name: str,
        register_fn: Callable[[], PluginRegistration],
    ) -> None:
        """Register a plugin factory. Does not call register_fn yet."""
        if name in self._plugins:
            return  # Idempotent
        self._plugins[name] = _PluginEntry(
            name=name,
            state=PluginState.DISCOVERED,
            register_fn=register_fn,
        )

    def register(self, name: str) -> PluginRegistration:
        """Call a discovered plugin's register_fn, index its contributions."""
        entry = self._plugins.get(name)
        if entry is None:
            raise KeyError(f"Plugin '{name}' not discovered")
        if entry.state not in (PluginState.DISCOVERED, PluginState.DISABLED):
            if entry.registration is not None:
                return entry.registration
            raise ValueError(
                f"Plugin '{name}' in state {entry.state.value}, "
                f"cannot register"
            )

        if entry.register_fn is None:
            raise ValueError(f"Plugin '{name}' has no register function")

        reg = entry.register_fn()
        entry.registration = reg

        # Index slash commands
        for sc in reg.slash_commands:
            self._slash_commands[sc.name] = name
            for alias in sc.aliases:
                self._slash_commands[alias] = name

        # Register tokens
        if reg.tokens:
            warnings = self._token_registry.register_plugin(name, reg.tokens)
            for w in warnings:
                logger.warning(w)

        self._transition(name, PluginState.REGISTERED)
        return reg

    def register_all(self) -> None:
        """Discover and register all known plugins. Idempotent."""
        if self._initialized:
            return
        self._initialized = True

        # Auto-discover built-in plugins
        try:
            from episodic.mcp.plugins.gsuite import register as gsuite_register
            self.discover("gsuite", gsuite_register)
        except ImportError:
            logger.debug("gsuite plugin not available")

        # Register all discovered plugins
        for name in list(self._plugins.keys()):
            entry = self._plugins[name]
            if entry.state == PluginState.DISCOVERED:
                try:
                    self.register(name)
                except Exception:
                    logger.exception("Failed to register plugin '%s'", name)

    def get(self, name: str) -> Optional[PluginRegistration]:
        """Get a plugin's registration, or None."""
        entry = self._plugins.get(name)
        if entry is None:
            return None
        return entry.registration

    def get_state(self, name: str) -> Optional[PluginState]:
        """Get a plugin's current state."""
        entry = self._plugins.get(name)
        return entry.state if entry else None

    def registered(self) -> List[PluginRegistration]:
        """Return all registered plugins."""
        return [
            e.registration
            for e in self._plugins.values()
            if e.registration is not None
        ]

    def names(self) -> List[str]:
        """Return all plugin names."""
        return list(self._plugins.keys())

    def states(self) -> Dict[str, PluginState]:
        """Return name -> state for all plugins."""
        return {name: e.state for name, e in self._plugins.items()}

    def has_slash_command(self, cmd: str) -> bool:
        """Check if any plugin owns this slash command."""
        return cmd in self._slash_commands

    def get_slash_command(self, cmd: str) -> Optional[SlashCommand]:
        """Look up a slash command across all plugins."""
        plugin_name = self._slash_commands.get(cmd)
        if plugin_name is None:
            return None
        reg = self.get(plugin_name)
        if reg is None:
            return None
        for sc in reg.slash_commands:
            if sc.name == cmd or cmd in sc.aliases:
                return sc
        return None

    def get_plugin_for_command(self, cmd: str) -> Optional[str]:
        """Get plugin name that owns a slash command."""
        return self._slash_commands.get(cmd)

    def transition(self, name: str, to_state: PluginState) -> None:
        """Public state transition."""
        self._transition(name, to_state)

    def _transition(self, name: str, to_state: PluginState) -> None:
        """Transition a plugin to a new state."""
        entry = self._plugins.get(name)
        if entry is None:
            raise KeyError(f"Plugin '{name}' not found")
        valid = _TRANSITIONS.get(entry.state, frozenset())
        if to_state not in valid:
            raise ValueError(
                f"Invalid transition: {entry.state.value} -> {to_state.value} "
                f"for plugin '{name}'"
            )
        entry.state = to_state


# Module-level singleton
_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Get or create the global plugin registry singleton."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


def reset_plugin_registry() -> None:
    """Reset the global registry. For testing only."""
    global _registry
    _registry = None
