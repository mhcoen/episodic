"""
Data Refresh Scheduler.

Background daemon thread for pre-fetching provider data.
Ensures queries have instant cached responses.
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Any

from .providers.base import DataProvider

logger = logging.getLogger(__name__)


@dataclass
class RefreshConfig:
    """Configuration for a scheduled refresh job."""

    provider_name: str
    refresh_interval_s: int
    args: Dict[str, Any]
    last_refresh: Optional[datetime] = None
    error_count: int = 0


class DataRefreshScheduler:
    """
    Background scheduler for pre-fetching provider data.

    Runs as a daemon thread, checking each registered provider
    and calling refresh() when due. Uses exponential backoff on errors.
    """

    def __init__(self):
        self._providers: Dict[str, DataProvider] = {}
        self._configs: Dict[str, RefreshConfig] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def register(
        self,
        key: str,
        provider: DataProvider,
        refresh_interval_s: int,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a provider for background refresh.

        Args:
            key: Unique key for this refresh job (e.g., "news_general")
            provider: The DataProvider instance
            refresh_interval_s: Seconds between refreshes
            args: Arguments to pass to provider.refresh()
        """
        with self._lock:
            self._providers[key] = provider
            self._configs[key] = RefreshConfig(
                provider_name=provider.name,
                refresh_interval_s=refresh_interval_s,
                args=args or {},
            )
            logger.debug(f"Registered refresh job: {key} every {refresh_interval_s}s")

    def unregister(self, key: str) -> None:
        """Unregister a refresh job."""
        with self._lock:
            self._providers.pop(key, None)
            self._configs.pop(key, None)

    def start(self) -> None:
        """Start the background refresh thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="data-refresh-scheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info("DataRefreshScheduler started")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the scheduler gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("DataRefreshScheduler stopped")

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        """Main scheduler loop."""
        # Initial refresh for all providers
        self._refresh_all()

        while not self._stop_event.is_set():
            self._check_and_refresh()
            # Check every 30 seconds
            self._stop_event.wait(timeout=30)

    def _refresh_all(self) -> None:
        """Refresh all registered providers immediately."""
        with self._lock:
            keys = list(self._configs.keys())

        for key in keys:
            if self._stop_event.is_set():
                break
            self._do_refresh(key)

    def _check_and_refresh(self) -> None:
        """Check each provider and refresh if due."""
        now = datetime.now()

        with self._lock:
            items = list(self._configs.items())

        for key, config in items:
            if self._stop_event.is_set():
                break

            # Check if refresh is due
            if config.last_refresh is None:
                # Never refreshed - do it now
                self._do_refresh(key)
            else:
                elapsed = (now - config.last_refresh).total_seconds()
                if elapsed >= config.refresh_interval_s:
                    self._do_refresh(key)

    def _do_refresh(self, key: str) -> None:
        """Execute a single refresh operation."""
        with self._lock:
            provider = self._providers.get(key)
            config = self._configs.get(key)

        if not provider or not config:
            return

        try:
            logger.debug(f"Refreshing {key}")
            result = provider.refresh(config.args)

            with self._lock:
                config = self._configs.get(key)
                if config:
                    config.last_refresh = datetime.now()
                    if result.success:
                        config.error_count = 0
                        config.refresh_interval_s = result.next_refresh_s
                        logger.debug(f"Refresh {key} succeeded, next in {result.next_refresh_s}s")
                    else:
                        config.error_count += 1
                        # Exponential backoff: double interval, max 2 hours
                        backoff = min(
                            config.refresh_interval_s * (2 ** config.error_count),
                            7200,
                        )
                        config.refresh_interval_s = backoff
                        logger.warning(f"Refresh {key} failed: {result.error}, backoff {backoff}s")

        except Exception as e:
            logger.exception(f"Refresh {key} exception: {e}")
            with self._lock:
                config = self._configs.get(key)
                if config:
                    config.error_count += 1
                    config.last_refresh = datetime.now()

    def status(self) -> Dict[str, Any]:
        """Return scheduler status for diagnostics."""
        with self._lock:
            jobs = {}
            for key, config in self._configs.items():
                jobs[key] = {
                    "provider": config.provider_name,
                    "interval_s": config.refresh_interval_s,
                    "last_refresh": config.last_refresh.isoformat() if config.last_refresh else None,
                    "error_count": config.error_count,
                }

        return {
            "running": self.is_running(),
            "jobs": jobs,
        }


# Singleton instance
_scheduler: Optional[DataRefreshScheduler] = None


def get_data_refresh_scheduler() -> DataRefreshScheduler:
    """Get the singleton DataRefreshScheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = DataRefreshScheduler()
    return _scheduler
