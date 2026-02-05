"""
Tests for DataRefreshScheduler.
"""

import time
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from episodic.utility.data_refresh import (
    DataRefreshScheduler,
    RefreshConfig,
    get_data_refresh_scheduler,
)
from episodic.utility.providers.base import RefreshResult


class MockProvider:
    """Mock data provider for testing."""

    name = "mock"
    refresh_interval_s = 60

    def __init__(self):
        self.refresh_count = 0
        self.last_args = None
        self.should_fail = False

    def refresh(self, args):
        self.refresh_count += 1
        self.last_args = args

        if self.should_fail:
            return RefreshResult(
                success=False,
                cache_key="mock:test",
                payload=None,
                error="Mock error",
                next_refresh_s=120,
            )

        return RefreshResult(
            success=True,
            cache_key="mock:test",
            payload={"data": "test"},
            error=None,
            next_refresh_s=60,
        )


class TestDataRefreshScheduler:
    """Tests for DataRefreshScheduler."""

    def test_register_provider(self):
        """Test registering a provider."""
        scheduler = DataRefreshScheduler()
        provider = MockProvider()

        scheduler.register("test_job", provider, 300, args={"category": "general"})

        status = scheduler.status()
        assert "test_job" in status["jobs"]
        assert status["jobs"]["test_job"]["provider"] == "mock"
        assert status["jobs"]["test_job"]["interval_s"] == 300

    def test_unregister_provider(self):
        """Test unregistering a provider."""
        scheduler = DataRefreshScheduler()
        provider = MockProvider()

        scheduler.register("test_job", provider, 300)
        scheduler.unregister("test_job")

        status = scheduler.status()
        assert "test_job" not in status["jobs"]

    def test_start_stop(self):
        """Test starting and stopping the scheduler."""
        scheduler = DataRefreshScheduler()
        provider = MockProvider()

        scheduler.register("test_job", provider, 300)

        assert not scheduler.is_running()

        scheduler.start()
        assert scheduler.is_running()

        scheduler.stop()
        assert not scheduler.is_running()

    def test_initial_refresh_on_start(self):
        """Test that providers are refreshed immediately on start."""
        scheduler = DataRefreshScheduler()
        provider = MockProvider()

        scheduler.register("test_job", provider, 300, args={"key": "value"})
        scheduler.start()

        # Give the thread time to run initial refresh
        time.sleep(0.2)

        scheduler.stop()

        assert provider.refresh_count >= 1
        assert provider.last_args == {"key": "value"}

    def test_refresh_updates_config(self):
        """Test that successful refresh updates config."""
        scheduler = DataRefreshScheduler()
        provider = MockProvider()

        scheduler.register("test_job", provider, 300)
        scheduler.start()

        time.sleep(0.2)
        scheduler.stop()

        config = scheduler._configs.get("test_job")
        assert config is not None
        assert config.last_refresh is not None
        assert config.error_count == 0

    def test_error_increments_count(self):
        """Test that errors increment error count."""
        scheduler = DataRefreshScheduler()
        provider = MockProvider()
        provider.should_fail = True

        scheduler.register("test_job", provider, 300)
        scheduler.start()

        time.sleep(0.2)
        scheduler.stop()

        config = scheduler._configs.get("test_job")
        assert config is not None
        assert config.error_count >= 1

    def test_exponential_backoff(self):
        """Test exponential backoff on errors."""
        scheduler = DataRefreshScheduler()
        provider = MockProvider()
        provider.should_fail = True

        scheduler.register("test_job", provider, 60)

        # Manually trigger refresh to increment error count
        scheduler._do_refresh("test_job")

        config = scheduler._configs.get("test_job")
        initial_interval = config.refresh_interval_s

        # Another failure should increase interval
        scheduler._do_refresh("test_job")

        # With error_count=2, backoff should be 60 * 4 = 240
        assert config.refresh_interval_s > initial_interval

    def test_backoff_max_cap(self):
        """Test that backoff is capped at 2 hours."""
        scheduler = DataRefreshScheduler()
        provider = MockProvider()
        provider.should_fail = True

        scheduler.register("test_job", provider, 60)

        # Manually set high error count
        config = scheduler._configs.get("test_job")
        config.error_count = 10

        scheduler._do_refresh("test_job")

        # Should be capped at 7200 (2 hours)
        assert config.refresh_interval_s <= 7200

    def test_status_output(self):
        """Test status output format."""
        scheduler = DataRefreshScheduler()
        provider = MockProvider()

        scheduler.register("test_job", provider, 300, args={"category": "general"})

        status = scheduler.status()

        assert "running" in status
        assert "jobs" in status
        assert "test_job" in status["jobs"]

        job_status = status["jobs"]["test_job"]
        assert job_status["provider"] == "mock"
        assert job_status["interval_s"] == 300
        assert job_status["error_count"] == 0

    def test_singleton_instance(self):
        """Test that get_data_refresh_scheduler returns singleton."""
        # Reset singleton for test
        import episodic.utility.data_refresh as module
        module._scheduler = None

        s1 = get_data_refresh_scheduler()
        s2 = get_data_refresh_scheduler()

        assert s1 is s2


class TestRefreshConfig:
    """Tests for RefreshConfig dataclass."""

    def test_default_values(self):
        """Test default values for RefreshConfig."""
        config = RefreshConfig(
            provider_name="test",
            refresh_interval_s=300,
            args={"key": "value"},
        )

        assert config.last_refresh is None
        assert config.error_count == 0

    def test_custom_values(self):
        """Test custom values for RefreshConfig."""
        now = datetime.now()
        config = RefreshConfig(
            provider_name="test",
            refresh_interval_s=600,
            args={"key": "value"},
            last_refresh=now,
            error_count=3,
        )

        assert config.last_refresh == now
        assert config.error_count == 3
