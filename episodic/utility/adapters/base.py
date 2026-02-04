"""
Base Protocol for Service Adapters.

Defines the common interface all adapters must implement.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Dict, Any, List, Optional


class AdapterStatus(Enum):
    """Status of a service adapter."""
    READY = "ready"
    NOT_CONFIGURED = "not_configured"
    NOT_AUTHENTICATED = "not_authenticated"
    UNAVAILABLE = "unavailable"


@dataclass
class CommandSchema:
    """Schema for a single adapter command."""
    name: str
    description: str
    args: Dict[str, str]  # arg_name -> type ("str", "int", "bool", etc.)
    required_args: List[str]
    mutating: bool
    requires_auth: bool


@dataclass
class AdapterResult:
    """Result from adapter execution."""
    status: str  # "ok" | "error"
    payload: Dict[str, Any]
    speech_text: str
    display_text: str
    side_effects: List[str] = field(default_factory=list)
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    @classmethod
    def ok(
        cls,
        display: str,
        speech: Optional[str] = None,
        side_effects: Optional[List[str]] = None,
        **payload
    ) -> "AdapterResult":
        """Create a successful result."""
        return cls(
            status="ok",
            payload=payload,
            speech_text=speech or display,
            display_text=display,
            side_effects=side_effects or [],
        )

    @classmethod
    def error(cls, error_type: str, message: str) -> "AdapterResult":
        """Create an error result."""
        return cls(
            status="error",
            payload={},
            speech_text=message,
            display_text=f"Error: {message}",
            error_type=error_type,
            error_message=message,
        )


class ServiceAdapter(Protocol):
    """Protocol for external service integrations."""

    name: str
    display_name: str
    commands: List[str]

    def describe(self) -> Dict[str, CommandSchema]:
        """Return schema for all commands."""
        ...

    def status(self) -> AdapterStatus:
        """Check adapter health and auth status."""
        ...

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply configuration (API keys, preferences)."""
        ...

    def authenticate(self) -> bool:
        """Perform authentication flow if needed. Returns success."""
        ...

    def execute(self, command: str, args: Dict[str, Any]) -> AdapterResult:
        """Execute a command. Must be configured and authenticated first."""
        ...

    def is_playing(self) -> bool:
        """Check if adapter is currently playing media."""
        ...

    def stop(self) -> None:
        """Stop playback (for system stop command)."""
        ...


class AdapterRegistry:
    """Central registry for all service adapters."""

    def __init__(self):
        self._adapters: Dict[str, ServiceAdapter] = {}
        self._command_map: Dict[str, str] = {}  # command -> adapter_name

    def register(self, adapter: ServiceAdapter) -> None:
        """Register an adapter and map its commands."""
        self._adapters[adapter.name] = adapter
        for cmd in adapter.commands:
            key = f"{adapter.name}:{cmd}"
            self._command_map[key] = adapter.name

    def unregister(self, name: str) -> None:
        """Unregister an adapter."""
        if name in self._adapters:
            adapter = self._adapters.pop(name)
            for cmd in adapter.commands:
                key = f"{name}:{cmd}"
                self._command_map.pop(key, None)

    def get_adapter(self, name: str) -> Optional[ServiceAdapter]:
        """Get adapter by name."""
        return self._adapters.get(name)

    def route_command(self, adapter_name: str, command: str) -> Optional[ServiceAdapter]:
        """Get adapter that handles this command."""
        return self._adapters.get(adapter_name)

    def list_adapters(self) -> List[ServiceAdapter]:
        """List all registered adapters."""
        return list(self._adapters.values())

    def status_all(self) -> Dict[str, AdapterStatus]:
        """Get status of all adapters."""
        return {name: adapter.status() for name, adapter in self._adapters.items()}

    def get_playing_adapters(self) -> List[ServiceAdapter]:
        """Get all adapters currently playing."""
        return [a for a in self._adapters.values() if a.is_playing()]

    def stop_all(self) -> None:
        """Stop all adapters."""
        for adapter in self._adapters.values():
            try:
                adapter.stop()
            except Exception:
                pass
