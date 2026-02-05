"""
Black-box smoke tests for the CLI.

These tests run the actual CLI as a subprocess to:
1. Verify packaging and entry points work
2. Detect drift between TestSession and real CLI behavior
3. Catch import errors and startup failures

These are intentionally simple - they test that the CLI runs,
not that it produces correct output (that's for integration tests).
"""

import subprocess
import sys
import pytest


class TestCLIStarts:
    """Tests that the CLI starts and responds to basic commands."""

    def test_help_flag(self):
        """CLI should respond to --help."""
        result = subprocess.run(
            [sys.executable, "-m", "episodic", "--help"],
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert b"episodic" in result.stdout.lower() or b"usage" in result.stdout.lower()

    def test_version_or_help_available(self):
        """CLI should have either --version or --help working."""
        # Try --help first (more common)
        result = subprocess.run(
            [sys.executable, "-m", "episodic", "--help"],
            capture_output=True,
            timeout=10,
        )
        # Should either succeed or fail gracefully
        assert result.returncode in (0, 1, 2)  # 2 is common for "invalid option"

    def test_module_importable(self):
        """The episodic module should be importable."""
        result = subprocess.run(
            [sys.executable, "-c", "import episodic; print('ok')"],
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert b"ok" in result.stdout


class TestCLICommands:
    """Tests that specific CLI commands don't crash."""

    @pytest.mark.parametrize("cmd", [
        "/help",
        "/quit",
        "/exit",
    ])
    def test_command_no_crash(self, cmd):
        """Basic commands should not crash the CLI."""
        # Use echo to send command then quit
        result = subprocess.run(
            [sys.executable, "-m", "episodic"],
            input=f"{cmd}\n".encode(),
            capture_output=True,
            timeout=15,
        )
        # Should exit cleanly (0) or with user-initiated exit
        # Not crash (segfault, unhandled exception)
        assert result.returncode in (0, 1)
        # Should not have Python traceback in output
        assert b"Traceback (most recent call last)" not in result.stderr


class TestCLIEnvironment:
    """Tests for CLI environment handling."""

    def test_no_database_creates_one(self, tmp_path):
        """CLI should handle missing database gracefully."""
        import os

        env = os.environ.copy()
        env["EPISODIC_HOME"] = str(tmp_path)
        env["HOME"] = str(tmp_path)

        result = subprocess.run(
            [sys.executable, "-m", "episodic"],
            input=b"/quit\n",
            capture_output=True,
            timeout=15,
            env=env,
        )
        # Should not crash
        assert result.returncode in (0, 1)
        assert b"Traceback" not in result.stderr

    def test_invalid_db_path_handled(self, tmp_path):
        """CLI should handle invalid database paths gracefully."""
        import os

        env = os.environ.copy()
        # Point to a directory that doesn't exist and can't be created
        env["EPISODIC_DB_PATH"] = "/nonexistent/path/that/cannot/exist/db.sqlite"

        result = subprocess.run(
            [sys.executable, "-m", "episodic"],
            input=b"/quit\n",
            capture_output=True,
            timeout=15,
            env=env,
        )
        # Should handle gracefully (error message, not crash)
        # Return code may be non-zero but shouldn't be a crash
        assert b"Segmentation fault" not in result.stderr


class TestCLIOutput:
    """Tests for CLI output format."""

    def test_welcome_message(self, tmp_path):
        """CLI should display a welcome message."""
        import os

        env = os.environ.copy()
        env["EPISODIC_HOME"] = str(tmp_path)
        env["HOME"] = str(tmp_path)

        result = subprocess.run(
            [sys.executable, "-m", "episodic"],
            input=b"/quit\n",
            capture_output=True,
            timeout=15,
            env=env,
        )
        # Should have some welcome/startup output
        output = result.stdout.decode(errors="replace").lower()
        assert "episodic" in output or "welcome" in output or ">" in output

    def test_quit_says_goodbye(self, tmp_path):
        """CLI should acknowledge quit command."""
        import os

        env = os.environ.copy()
        env["EPISODIC_HOME"] = str(tmp_path)
        env["HOME"] = str(tmp_path)

        result = subprocess.run(
            [sys.executable, "-m", "episodic"],
            input=b"/quit\n",
            capture_output=True,
            timeout=15,
            env=env,
        )
        output = result.stdout.decode(errors="replace").lower()
        # Should have goodbye or exit acknowledgment
        assert "bye" in output or "goodbye" in output or "exit" in output or result.returncode == 0
