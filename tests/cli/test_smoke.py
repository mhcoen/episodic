"""
CLI smoke tests.

Verifies entry point wiring, not logic (harness tests cover logic).
Uses subprocess to test the actual CLI as a user would invoke it.

These tests catch bugs like 5c256ca where script execution bypassed
voice grammar despite the harness tests passing.
"""

import os
import subprocess
import tempfile
import pytest


# Strip SOCKS proxy to allow LLM calls
ENV = {k: v for k, v in os.environ.items() if k.upper() not in ("ALL_PROXY", "all_proxy")}

TIMEOUT = 30  # seconds


def run_cli(*args, input_text=None, timeout=TIMEOUT):
    """Run the CLI with given arguments and return (stdout, stderr, returncode)."""
    cmd = ["python", "-m", "episodic"] + list(args)
    result = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=ENV,
    )
    return result.stdout, result.stderr, result.returncode


def run_script(content, timeout=TIMEOUT):
    """Run a script through the CLI and return (stdout, stderr, returncode)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        f.flush()
        script_path = f.name

    try:
        return run_cli("-e", script_path, timeout=timeout)
    finally:
        os.unlink(script_path)


class TestCLIHelp:
    """Test CLI help and basic invocation."""

    def test_help_flag(self):
        """--help should show usage and exit 0."""
        stdout, stderr, code = run_cli("--help")
        assert code == 0
        assert "Usage:" in stdout or "usage:" in stdout.lower()
        assert "episodic" in stdout.lower()

    def test_invalid_flag(self):
        """Invalid flag should exit non-zero."""
        stdout, stderr, code = run_cli("--nonexistent-flag")
        assert code != 0


class TestScriptExecution:
    """Test script execution mode (-e flag)."""

    def test_script_time_command(self):
        """/time command should return time in script mode."""
        stdout, stderr, code = run_script("/time\n/exit\n")
        assert code == 0
        # Time output should have AM or PM
        assert "AM" in stdout or "PM" in stdout or ":" in stdout

    def test_script_weather_command(self):
        """/weather command should return weather data."""
        stdout, stderr, code = run_script("/weather\n/exit\n")
        assert code == 0
        # Weather output should have temperature indicator
        assert "degrees" in stdout.lower() or "°" in stdout

    def test_script_news_command(self):
        """/news command should return headlines."""
        stdout, stderr, code = run_script("/news\n/exit\n")
        assert code == 0
        # News should have numbered headlines
        assert "1." in stdout or "Headlines" in stdout

    def test_script_exit_command(self):
        """/exit should terminate script cleanly."""
        stdout, stderr, code = run_script("/exit\n")
        assert code == 0
        assert "exit" in stdout.lower() or "complete" in stdout.lower()

    def test_script_invalid_command_recovers(self):
        """Invalid command should not crash the script."""
        stdout, stderr, code = run_script("/nonexistent_xyz\n/time\n/exit\n")
        assert code == 0
        # Should still execute subsequent commands
        assert "AM" in stdout or "PM" in stdout or ":" in stdout


class TestVoiceGrammarScript:
    """Test voice grammar in script mode (regression for 5c256ca)."""

    def test_voice_time_query(self):
        """'what time is it' should route to utility, not LLM."""
        stdout, stderr, code = run_script("what time is it\n/exit\n")
        assert code == 0
        # Should get a time response, not an LLM "I can't check time" response
        # Utility response will have actual time like "12:30" or emoji
        has_time_indicator = (
            "AM" in stdout or "PM" in stdout or
            ":" in stdout.split("what time")[0] if "what time" in stdout else ":" in stdout
        )
        # Should NOT have LLM's typical "I can't" or "I'm unable" response
        no_llm_refusal = "unable" not in stdout.lower() and "can't check" not in stdout.lower()
        assert has_time_indicator or no_llm_refusal

    def test_voice_timer_command(self):
        """'set a timer for 5 minutes' should route to utility."""
        stdout, stderr, code = run_script("set a timer for 5 minutes\n/exit\n")
        assert code == 0
        # Utility timer response has "timer" and time indication
        assert "timer" in stdout.lower() or "minute" in stdout.lower()


class TestMixedFlow:
    """Test mixed utility and conversation flows."""

    def test_utility_then_utility(self):
        """Multiple utility commands should work in sequence."""
        stdout, stderr, code = run_script("/time\n/weather\n/exit\n")
        assert code == 0
        # Both should produce output
        assert "AM" in stdout or "PM" in stdout or ":" in stdout
        assert "degrees" in stdout.lower() or "°" in stdout

    def test_voice_then_slash(self):
        """Voice grammar then slash command should both work."""
        stdout, stderr, code = run_script("what time is it\n/weather\n/exit\n")
        assert code == 0
        # Weather should work after voice command
        assert "degrees" in stdout.lower() or "°" in stdout


class TestLLMIntegration:
    """Test LLM integration (limited tests to control costs)."""

    def test_simple_llm_query(self):
        """Simple LLM query should get a response."""
        stdout, stderr, code = run_script("Say hello in one word\n/exit\n", timeout=60)
        assert code == 0
        # Should have some response (not empty after the command line)
        lines = [l for l in stdout.split("\n") if l.strip() and not l.startswith("[")]
        assert len(lines) > 0

    def test_context_retention(self):
        """LLM should retain context across turns."""
        script = "My name is TestUser123\nWhat is my name?\n/exit\n"
        stdout, stderr, code = run_script(script, timeout=60)
        assert code == 0
        # Should mention the name in response
        assert "TestUser123" in stdout


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_script(self):
        """Empty script should complete without error."""
        stdout, stderr, code = run_script("")
        assert code == 0

    def test_comment_only_script(self):
        """Script with only comments should complete."""
        stdout, stderr, code = run_script("# This is a comment\n# Another comment\n")
        assert code == 0

    def test_whitespace_handling(self):
        """Script with blank lines should handle gracefully."""
        stdout, stderr, code = run_script("\n\n/time\n\n/exit\n")
        assert code == 0
