"""Tests for episodic.mcp.server module."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestBuildHealthResponse:
    """Tests for _build_health_response."""

    def test_health_response_keys(self):
        from episodic.mcp.server import _build_health_response
        with patch("episodic.mcp.server._get_node_count", return_value=42):
            resp = _build_health_response()
        assert "status" in resp
        assert "version" in resp
        assert "uptime_seconds" in resp
        assert "pid" in resp
        assert "node_count" in resp

    def test_health_response_status_ok(self):
        from episodic.mcp.server import _build_health_response
        with patch("episodic.mcp.server._get_node_count", return_value=0):
            resp = _build_health_response()
        assert resp["status"] == "ok"

    def test_health_response_pid_is_current(self):
        from episodic.mcp.server import _build_health_response
        with patch("episodic.mcp.server._get_node_count", return_value=0):
            resp = _build_health_response()
        assert resp["pid"] == os.getpid()

    def test_health_response_version(self):
        from episodic.mcp.server import _build_health_response
        from episodic.mcp import __version__
        with patch("episodic.mcp.server._get_node_count", return_value=0):
            resp = _build_health_response()
        assert resp["version"] == __version__

    def test_health_response_node_count(self):
        from episodic.mcp.server import _build_health_response
        with patch("episodic.mcp.server._get_node_count", return_value=99):
            resp = _build_health_response()
        assert resp["node_count"] == 99

    def test_health_response_uptime(self):
        import episodic.mcp.server as srv
        srv._start_time = time.time() - 60
        with patch("episodic.mcp.server._get_node_count", return_value=0):
            resp = srv._build_health_response()
        assert resp["uptime_seconds"] >= 59


class TestPidfileLifecycle:
    """Tests for write_pidfile and remove_pidfile."""

    def test_write_and_read_pidfile(self, tmp_path):
        from episodic.mcp.server import write_pidfile, remove_pidfile

        pidfile = tmp_path / "mcp-server.pid"
        with patch("episodic.mcp.server._get_pidfile_path", return_value=pidfile):
            write_pidfile(51983)
            assert pidfile.exists()
            data = json.loads(pidfile.read_text())
            assert data["pid"] == os.getpid()
            assert data["port"] == 51983
            assert "started_at" in data

            remove_pidfile()
            assert not pidfile.exists()

    def test_remove_nonexistent_pidfile(self, tmp_path):
        from episodic.mcp.server import remove_pidfile

        pidfile = tmp_path / "mcp-server.pid"
        with patch("episodic.mcp.server._get_pidfile_path", return_value=pidfile):
            # Should not raise
            remove_pidfile()


class TestGetNodeCount:
    """Tests for _get_node_count."""

    def test_returns_zero_when_no_db(self, tmp_path):
        from episodic.mcp.server import _get_node_count
        with patch.dict(os.environ, {"EPISODIC_DB_PATH": str(tmp_path / "nope.db")}):
            assert _get_node_count() == 0

    def test_returns_count_from_db(self, tmp_path):
        import sqlite3
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO nodes VALUES ('a')")
        conn.execute("INSERT INTO nodes VALUES ('b')")
        conn.commit()
        conn.close()

        from episodic.mcp.server import _get_node_count
        with patch.dict(os.environ, {"EPISODIC_DB_PATH": str(db_path)}):
            assert _get_node_count() == 2
