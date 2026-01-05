#!/usr/bin/env python3
"""Test topic detection without actual LLM responses."""

import os
import subprocess
import sys
import tempfile

import pytest


def test_topic_detection_without_llm():
    """Run a CLI script that exercises topic detection with LLM skipped."""
    if os.environ.get("EPISODIC_RUN_TOPIC_DETECTION") != "1":
        pytest.skip("Set EPISODIC_RUN_TOPIC_DETECTION=1 to run this CLI workflow")

    test_script = """
/init --erase
/set automatic_topic_detection true
/set min_messages_before_topic_change 4
/set show_topics true
/set skip_llm_response true
/model chat gpt-3.5-turbo
/model detection gpt-3.5-turbo

# Start conversation
Hello, tell me about Mars
What's the atmosphere like?
How long to get there?
Is it cold?

# Topic change - should be detected here
How to make pasta carbonara?
What ingredients?
What cheese to use?
How long to cook?

# Another topic change
What is machine learning?
How do neural networks work?
What is backpropagation?
What is deep learning?

/topics list
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
                timeout=30,
                env=env
            )

            assert "Error: cannot access local variable 'get_ancestry'" not in result.stdout
            assert "Error: cannot access local variable 'get_ancestry'" not in result.stderr
            assert result.returncode == 0
        finally:
            os.unlink(script_path)
