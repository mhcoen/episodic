"""Tests for episodic.mcp.lifecycle module."""

import json
import os
import signal
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from episodic.mcp.lifecycle import (
    get_pidfile_path,
    read_pidfile,
    is_process_alive,
    check_port_available,
    get_server_status,
    start_server,
    stop_server,
    _clean_stale_pidfile,
)


class TestPidfileOperations:
    """Tests for pidfile read/write."""

    def test_get_pidfile_path_default(self):
        with patch.dict(os.environ, {}, clear=False):
            # Remove EPISODIC_DATA_DIR if set
            env = os.environ.copy()
            env.pop("EPISODIC_DATA_DIR", None)
            with patch.dict(os.environ, env, clear=True):
                path = get_pidfile_path()
                assert path.name == "mcp-server.pid"
                assert ".episodic" in str(path)

    def test_get_pidfile_path_custom_dir(self):
        with patch.dict(os.environ, {"EPISODIC_DATA_DIR": "/custom/data"}):
            path = get_pidfile_path()
            assert str(path) == "/custom/data/mcp-server.pid"

    def test_read_pidfile_not_exists(self, tmp_path):
        with patch("episodic.mcp.lifecycle.get_pidfile_path",
                    return_value=tmp_path / "nope.pid"):
            assert read_pidfile() is None

    def test_read_pidfile_valid(self, tmp_path):
        pidfile = tmp_path / "mcp-server.pid"
        data = {"pid": 12345, "port": 51983, "started_at": 1000.0}
        pidfile.write_text(json.dumps(data))

        with patch("episodic.mcp.lifecycle.get_pidfile_path",
                    return_value=pidfile):
            result = read_pidfile()
            assert result["pid"] == 12345
            assert result["port"] == 51983

    def test_read_pidfile_invalid_json(self, tmp_path):
        pidfile = tmp_path / "mcp-server.pid"
        pidfile.write_text("not json")

        with patch("episodic.mcp.lifecycle.get_pidfile_path",
                    return_value=pidfile):
            assert read_pidfile() is None

    def test_read_pidfile_missing_keys(self, tmp_path):
        pidfile = tmp_path / "mcp-server.pid"
        pidfile.write_text(json.dumps({"pid": 1}))  # missing port

        with patch("episodic.mcp.lifecycle.get_pidfile_path",
                    return_value=pidfile):
            assert read_pidfile() is None


class TestProcessDetection:
    """Tests for is_process_alive."""

    def test_current_process_is_alive(self):
        assert is_process_alive(os.getpid()) is True

    def test_nonexistent_pid(self):
        # PID 2^30 is unlikely to exist
        assert is_process_alive(1073741824) is False

    def test_zero_pid(self):
        # PID 0 should raise an error (it sends to process group)
        # On macOS/Linux, kill(0, 0) checks own process group
        # We just check it doesn't crash
        result = is_process_alive(0)
        assert isinstance(result, bool)


class TestPortCheck:
    """Tests for check_port_available."""

    def test_available_port(self):
        # Port 0 lets the OS choose — always available
        assert check_port_available("127.0.0.1", 0) is True

    def test_unavailable_port(self):
        import socket
        # Bind and listen so SO_REUSEADDR won't allow rebinding on Linux
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            # Port is in use
            assert check_port_available("127.0.0.1", port) is False


class TestGetServerStatus:
    """Tests for get_server_status."""

    def test_stopped_no_pidfile(self, tmp_path):
        with patch("episodic.mcp.lifecycle.get_pidfile_path",
                    return_value=tmp_path / "nope.pid"):
            status, data = get_server_status()
            assert status == "stopped"
            assert data is None

    def test_stale_pidfile(self, tmp_path):
        pidfile = tmp_path / "mcp-server.pid"
        data = {"pid": 1073741824, "port": 51983, "started_at": 1000.0}
        pidfile.write_text(json.dumps(data))

        with patch("episodic.mcp.lifecycle.get_pidfile_path",
                    return_value=pidfile):
            status, result = get_server_status()
            assert status == "stale"
            assert result["pid"] == 1073741824

    def test_running_with_health(self, tmp_path):
        pidfile = tmp_path / "mcp-server.pid"
        data = {"pid": os.getpid(), "port": 51983, "started_at": 1000.0}
        pidfile.write_text(json.dumps(data))

        health_resp = {"status": "ok", "version": "0.1.0",
                       "uptime_seconds": 30, "pid": os.getpid(),
                       "node_count": 5}

        with patch("episodic.mcp.lifecycle.get_pidfile_path",
                    return_value=pidfile), \
             patch("episodic.mcp.lifecycle.health_check",
                    return_value=health_resp):
            status, result = get_server_status()
            assert status == "running"
            assert result["node_count"] == 5

    def test_running_no_health(self, tmp_path):
        pidfile = tmp_path / "mcp-server.pid"
        data = {"pid": os.getpid(), "port": 51983, "started_at": 1000.0}
        pidfile.write_text(json.dumps(data))

        with patch("episodic.mcp.lifecycle.get_pidfile_path",
                    return_value=pidfile), \
             patch("episodic.mcp.lifecycle.health_check",
                    return_value=None):
            status, result = get_server_status()
            assert status == "running"
            assert result["pid"] == os.getpid()


class TestStartServer:
    """Tests for start_server with mocked subprocess."""

    def test_start_when_already_running(self, tmp_path):
        with patch("episodic.mcp.lifecycle.get_server_status",
                    return_value=("running", {"pid": 123, "port": 51983})):
            success, msg = start_server(51983, "127.0.0.1")
            assert success is False
            assert "already running" in msg

    def test_start_port_in_use(self, tmp_path):
        with patch("episodic.mcp.lifecycle.get_server_status",
                    return_value=("stopped", None)), \
             patch("episodic.mcp.lifecycle.check_port_available",
                    return_value=False):
            success, msg = start_server(51983, "127.0.0.1")
            assert success is False
            assert "already in use" in msg

    def test_start_success(self, tmp_path):
        health_resp = {"status": "ok", "pid": 99}
        mock_proc = MagicMock()
        mock_proc.pid = 99
        mock_proc.poll.return_value = None

        with patch("episodic.mcp.lifecycle.get_server_status",
                    return_value=("stopped", None)), \
             patch("episodic.mcp.lifecycle.check_port_available",
                    return_value=True), \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch("episodic.mcp.lifecycle.health_check",
                    return_value=health_resp), \
             patch("time.sleep"):
            success, msg = start_server(51983, "127.0.0.1")
            assert success is True
            assert "99" in msg

    def test_start_process_exits_immediately(self, tmp_path):
        mock_proc = MagicMock()
        mock_proc.pid = 99
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with patch("episodic.mcp.lifecycle.get_server_status",
                    return_value=("stopped", None)), \
             patch("episodic.mcp.lifecycle.check_port_available",
                    return_value=True), \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch("time.sleep"):
            success, msg = start_server(51983, "127.0.0.1")
            assert success is False
            assert "exited immediately" in msg

    def test_start_cleans_stale_pidfile(self, tmp_path):
        """Starting when stale should clean up before spawning."""
        health_resp = {"status": "ok", "pid": 99}
        mock_proc = MagicMock()
        mock_proc.pid = 99
        mock_proc.poll.return_value = None

        with patch("episodic.mcp.lifecycle.get_server_status",
                    return_value=("stale", {"pid": 1, "port": 51983})), \
             patch("episodic.mcp.lifecycle._clean_stale_pidfile") as mock_clean, \
             patch("episodic.mcp.lifecycle.check_port_available",
                    return_value=True), \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch("episodic.mcp.lifecycle.health_check",
                    return_value=health_resp), \
             patch("time.sleep"):
            success, msg = start_server(51983, "127.0.0.1")
            assert success is True
            mock_clean.assert_called_once()


class TestStopServer:
    """Tests for stop_server."""

    def test_stop_no_pidfile(self, tmp_path):
        with patch("episodic.mcp.lifecycle.read_pidfile", return_value=None):
            success, msg = stop_server()
            assert success is False
            assert "not running" in msg

    def test_stop_stale_pidfile(self, tmp_path):
        with patch("episodic.mcp.lifecycle.read_pidfile",
                    return_value={"pid": 1073741824, "port": 51983}), \
             patch("episodic.mcp.lifecycle.is_process_alive",
                    return_value=False), \
             patch("episodic.mcp.lifecycle._clean_stale_pidfile") as mock_clean:
            success, msg = stop_server()
            assert success is True
            assert "stale" in msg.lower()
            mock_clean.assert_called_once()

    def test_stop_sends_sigterm(self, tmp_path):
        with patch("episodic.mcp.lifecycle.read_pidfile",
                    return_value={"pid": 12345, "port": 51983}), \
             patch("episodic.mcp.lifecycle.is_process_alive",
                    side_effect=[True, False]), \
             patch("os.kill") as mock_kill, \
             patch("episodic.mcp.lifecycle._clean_stale_pidfile"), \
             patch("time.sleep"):
            success, msg = stop_server()
            assert success is True
            mock_kill.assert_called_once_with(12345, signal.SIGTERM)


class TestCleanStalePidfile:
    """Tests for _clean_stale_pidfile."""

    def test_clean_existing(self, tmp_path):
        pidfile = tmp_path / "mcp-server.pid"
        pidfile.write_text("{}")

        with patch("episodic.mcp.lifecycle.get_pidfile_path",
                    return_value=pidfile):
            _clean_stale_pidfile()
            assert not pidfile.exists()

    def test_clean_nonexistent(self, tmp_path):
        with patch("episodic.mcp.lifecycle.get_pidfile_path",
                    return_value=tmp_path / "nope.pid"):
            # Should not raise
            _clean_stale_pidfile()
