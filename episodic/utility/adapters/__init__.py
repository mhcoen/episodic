"""
Service Adapters for Utility Commands.

Adapters handle external integrations: radio streams, Spotify, calendar, etc.
Each adapter implements the ServiceAdapter protocol.
"""

from .base import (
    ServiceAdapter,
    AdapterStatus,
    AdapterResult,
    CommandSchema,
    AdapterRegistry,
)
from .radio import RadioAdapter

__all__ = [
    "ServiceAdapter",
    "AdapterStatus",
    "AdapterResult",
    "CommandSchema",
    "AdapterRegistry",
    "RadioAdapter",
]
