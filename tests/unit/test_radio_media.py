"""
Tests for Radio Adapter and Media handlers.

Tests cover:
1. Radio adapter (station matching, play/stop/list)
2. Media handlers (play, pause, stop, status, volume)
3. Adapter registry
"""

import pytest
import sqlite3
from typing import Dict, Any, List

from episodic.utility.types import UtilityQuery, ResultStatus
from episodic.utility.adapters.base import (
    AdapterRegistry,
    AdapterStatus,
    AdapterResult,
    CommandSchema,
)
from episodic.utility.adapters.radio import (
    RadioAdapter,
    NullRadioAdapter,
    DEFAULT_STATIONS,
)
from episodic.utility.handlers.media import (
    handle_media_play,
    handle_media_pause,
    handle_media_stop,
    handle_media_status,
    handle_volume_up,
    handle_volume_down,
    handle_volume_set,
    handle_radio_list,
    dispatch_media_command,
)
from episodic.utility.dispatcher import create_utility_query


@pytest.fixture
def null_radio():
    """Create a NullRadioAdapter for testing."""
    return NullRadioAdapter()


@pytest.fixture
def adapter_registry(null_radio):
    """Create an adapter registry with null radio."""
    registry = AdapterRegistry()
    registry.register(null_radio)
    return registry


class TestAdapterRegistry:
    """Tests for AdapterRegistry."""

    def test_register_adapter(self, null_radio):
        """Can register an adapter."""
        registry = AdapterRegistry()
        registry.register(null_radio)

        assert registry.get_adapter("radio") is not None
        assert registry.get_adapter("radio").name == "radio"

    def test_unregister_adapter(self, null_radio):
        """Can unregister an adapter."""
        registry = AdapterRegistry()
        registry.register(null_radio)
        registry.unregister("radio")

        assert registry.get_adapter("radio") is None

    def test_list_adapters(self, null_radio):
        """Can list all adapters."""
        registry = AdapterRegistry()
        registry.register(null_radio)

        adapters = registry.list_adapters()
        assert len(adapters) == 1
        assert adapters[0].name == "radio"

    def test_status_all(self, null_radio):
        """Can get status of all adapters."""
        registry = AdapterRegistry()
        registry.register(null_radio)

        statuses = registry.status_all()
        assert "radio" in statuses
        assert statuses["radio"] == AdapterStatus.READY


class TestNullRadioAdapter:
    """Tests for NullRadioAdapter (used in testing)."""

    def test_status(self, null_radio):
        """Null adapter is always ready."""
        assert null_radio.status() == AdapterStatus.READY

    def test_authenticate(self, null_radio):
        """Authentication always succeeds."""
        assert null_radio.authenticate() is True

    def test_describe(self, null_radio):
        """Returns command schemas."""
        schemas = null_radio.describe()
        assert "play" in schemas
        assert "stop" in schemas
        assert "list" in schemas

    def test_play_station(self, null_radio):
        """Can play a station."""
        result = null_radio.execute("play", {"station": "wbez"})

        assert result.status == "ok"
        assert "WBEZ" in result.display_text
        assert null_radio.is_playing()

    def test_play_station_not_found(self, null_radio):
        """Play fails for unknown station."""
        result = null_radio.execute("play", {"station": "unknown123"})

        assert result.status == "error"
        assert result.error_type == "station_not_found"

    def test_play_partial_match(self, null_radio):
        """Can play station by partial name match."""
        result = null_radio.execute("play", {"station": "chicago"})

        assert result.status == "ok"
        assert null_radio.is_playing()

    def test_stop(self, null_radio):
        """Can stop playback."""
        null_radio.execute("play", {"station": "npr"})
        result = null_radio.execute("stop", {})

        assert result.status == "ok"
        assert not null_radio.is_playing()

    def test_list_stations(self, null_radio):
        """Can list stations."""
        result = null_radio.execute("list", {})

        assert result.status == "ok"
        assert "stations" in result.payload
        assert len(result.payload["stations"]) > 0

    def test_status_not_playing(self, null_radio):
        """Status when not playing."""
        result = null_radio.execute("status", {})

        assert result.status == "ok"
        assert result.payload["playing"] is False

    def test_status_while_playing(self, null_radio):
        """Status while playing."""
        null_radio.execute("play", {"station": "npr"})
        result = null_radio.execute("status", {})

        assert result.status == "ok"
        assert result.payload["playing"] is True


class TestRadioAdapterStationMatching:
    """Tests for radio station matching logic."""

    def test_exact_key_match(self, null_radio):
        """Exact key match works."""
        station = null_radio._find_station("wbez")
        assert station is not None
        assert station["name"] == "WBEZ Chicago"

    def test_partial_key_match(self, null_radio):
        """Partial key match works."""
        station = null_radio._find_station("wbe")
        assert station is not None
        assert "WBEZ" in station["name"]

    def test_name_match(self, null_radio):
        """Station name match works."""
        station = null_radio._find_station("chicago")
        assert station is not None

    def test_case_insensitive(self, null_radio):
        """Matching is case insensitive."""
        station = null_radio._find_station("WBEZ")
        assert station is not None

    def test_no_match(self, null_radio):
        """Returns None for no match."""
        station = null_radio._find_station("xyz123notastation")
        assert station is None


class TestMediaHandlers:
    """Tests for media command handlers."""

    def test_media_play_radio(self, adapter_registry):
        """Can play radio via media_play."""
        query = create_utility_query("media", "media_play", args={
            "source": "radio",
            "station": "npr",
        })
        result = handle_media_play(query, adapter_registry)

        assert result.status == ResultStatus.OK
        assert "NPR" in result.display_text

    def test_media_play_infer_radio(self, adapter_registry):
        """Infers radio source when station specified."""
        query = create_utility_query("media", "media_play", args={
            "station": "wbez",
        })
        result = handle_media_play(query, adapter_registry)

        assert result.status == ResultStatus.OK

    def test_media_play_no_source(self):
        """Play fails when no source or adapters."""
        registry = AdapterRegistry()
        query = create_utility_query("media", "media_play", args={})
        result = handle_media_play(query, registry)

        assert result.status == ResultStatus.ERROR

    def test_media_stop(self, adapter_registry, null_radio):
        """Can stop media."""
        # Start playing
        null_radio.execute("play", {"station": "npr"})

        query = create_utility_query("media", "media_stop")
        result = handle_media_stop(query, adapter_registry)

        assert result.status == ResultStatus.OK
        assert not null_radio.is_playing()

    def test_media_stop_nothing_playing(self, adapter_registry):
        """Stop when nothing playing."""
        query = create_utility_query("media", "media_stop")
        result = handle_media_stop(query, adapter_registry)

        assert result.status == ResultStatus.OK
        assert "Nothing playing" in result.display_text

    def test_media_status_playing(self, adapter_registry, null_radio):
        """Status shows what's playing."""
        null_radio.execute("play", {"station": "wbez"})

        query = create_utility_query("media", "media_status")
        result = handle_media_status(query, adapter_registry)

        assert result.status == ResultStatus.OK
        assert result.data["playing"] is True

    def test_media_status_not_playing(self, adapter_registry):
        """Status when nothing playing."""
        query = create_utility_query("media", "media_status")
        result = handle_media_status(query, adapter_registry)

        assert result.status == ResultStatus.OK
        assert "Nothing playing" in result.display_text

    def test_radio_list(self, adapter_registry):
        """Can list radio stations via handler."""
        query = create_utility_query("media", "radio_list")
        result = handle_radio_list(query, adapter_registry)

        assert result.status == ResultStatus.OK
        assert "stations" in result.data

    def test_dispatch_media_command(self, adapter_registry):
        """Media dispatcher routes correctly."""
        query = create_utility_query("media", "media_play", args={
            "source": "radio",
            "station": "jazz",
        })
        result = dispatch_media_command(query, adapter_registry)

        assert result.status == ResultStatus.OK

    def test_dispatch_media_unknown_command(self, adapter_registry):
        """Media dispatcher handles unknown commands."""
        query = create_utility_query("media", "unknown_command")
        result = dispatch_media_command(query, adapter_registry)

        assert result.status == ResultStatus.ERROR


class TestVolumeControls:
    """Tests for volume control handlers."""

    def test_volume_up(self, adapter_registry):
        """Volume up works."""
        query = create_utility_query("media", "volume_up", args={"delta": 10})
        result = handle_volume_up(query, adapter_registry)

        assert result.status == ResultStatus.OK

    def test_volume_down(self, adapter_registry):
        """Volume down works."""
        query = create_utility_query("media", "volume_down", args={"delta": 10})
        result = handle_volume_down(query, adapter_registry)

        assert result.status == ResultStatus.OK

    def test_volume_set(self, adapter_registry):
        """Volume set works."""
        query = create_utility_query("media", "volume_set", args={"level": 50})
        result = handle_volume_set(query, adapter_registry)

        assert result.status == ResultStatus.OK
        assert result.data["volume"] == 50

    def test_volume_set_missing_level(self, adapter_registry):
        """Volume set requires level."""
        query = create_utility_query("media", "volume_set", args={})
        result = handle_volume_set(query, adapter_registry)

        assert result.status == ResultStatus.ERROR


class TestDefaultStations:
    """Tests for default station registry."""

    def test_default_stations_exist(self):
        """Default stations are defined."""
        assert len(DEFAULT_STATIONS) > 0

    def test_stations_have_required_fields(self):
        """Each station has name and url."""
        for key, station in DEFAULT_STATIONS.items():
            assert "name" in station
            assert "url" in station
            assert station["url"].startswith("http")

    def test_npr_station_exists(self):
        """NPR station is available."""
        assert "npr" in DEFAULT_STATIONS
        assert "NPR" in DEFAULT_STATIONS["npr"]["name"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
