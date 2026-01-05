#!/usr/bin/env python3
"""Test which HuggingFace models work with chat format."""

import os
import subprocess
import sys
import tempfile

import pytest


def test_hf_chat_models():
    """Run a CLI script against a set of HuggingFace chat models."""
    if os.environ.get("EPISODIC_RUN_HF_TESTS") != "1":
        pytest.skip("Set EPISODIC_RUN_HF_TESTS=1 to run HuggingFace CLI tests")

    models_to_test = [
        "huggingface/tiiuae/falcon-7b-instruct",
        "huggingface/tiiuae/falcon-40b-instruct",
        "huggingface/tiiuae/falcon-180B-chat",
        "huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
        "huggingface/mistralai/Mistral-7B-Instruct-v0.2",
        "huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1",
        "huggingface/google/flan-t5-xxl",
        "huggingface/bigscience/bloom",
        "huggingface/EleutherAI/gpt-neox-20b",
        "huggingface/stabilityai/stablelm-tuned-alpha-7b"
    ]

    with tempfile.TemporaryDirectory() as temp_home:
        env = os.environ.copy()
        env["EPISODIC_HOME"] = temp_home
        env["EPISODIC_DB_PATH"] = os.path.join(temp_home, "episodic.db")
        env["HOME"] = temp_home

        for model in models_to_test:
            test_script = f"""
/init --erase
/model chat {model}
Say 'Hello' in 5 words or less
/exit
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_script)
                script_path = f.name

            try:
                result = subprocess.run(
                    [sys.executable, "-m", "episodic", "--execute", script_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=env
                )
                assert result.returncode == 0
            finally:
                os.unlink(script_path)
