#!/usr/bin/env python3
"""Test HuggingFace API key and model access."""

import os
import subprocess
import sys
import tempfile

import pytest


def test_huggingface_api_integration():
    """Run a CLI script against a HuggingFace model."""
    if os.environ.get("EPISODIC_RUN_HF_TESTS") != "1":
        pytest.skip("Set EPISODIC_RUN_HF_TESTS=1 to run HuggingFace CLI tests")

    test_script = """
/init --erase
/model chat huggingface/tiiuae/falcon-7b-instruct
Tell me a fun fact in 10 words or less
/exit
"""

    with tempfile.TemporaryDirectory() as temp_home:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_script)
            script_path = f.name

        try:
            env = os.environ.copy()
            env["EPISODIC_HOME"] = temp_home
            env["EPISODIC_DB_PATH"] = os.path.join(temp_home, "episodic.db")
            env["HOME"] = temp_home

            result = subprocess.run(
                [sys.executable, "-m", "episodic", "--execute", script_path],
                capture_output=True,
                text=True,
                timeout=60,
                env=env
            )

            assert result.returncode == 0
        finally:
            os.unlink(script_path)
