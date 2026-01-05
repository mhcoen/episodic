#!/usr/bin/env python3
"""Test new HuggingFace models."""

import os
import subprocess
import sys
import tempfile

import pytest


def test_new_hf_models():
    """Run a CLI script against a set of new HuggingFace chat models."""
    if os.environ.get("EPISODIC_RUN_HF_TESTS") != "1":
        pytest.skip("Set EPISODIC_RUN_HF_TESTS=1 to run HuggingFace CLI tests")

    models_to_test = [
        ("huggingface/Qwen/Qwen2.5-7B-Instruct", "Qwen 2.5 7B"),
        ("huggingface/meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B"),
        ("huggingface/mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B v0.3"),
        ("huggingface/tiiuae/Falcon3-7B-Instruct", "Falcon 3 7B"),
        ("huggingface/deepseek-ai/deepseek-llm-7b-chat", "DeepSeek 7B"),
        ("huggingface/01-ai/Yi-1.5-6B-Chat", "Yi 1.5 6B"),
        ("huggingface/GeneZC/MiniChat-2-3B", "MiniChat 2 3B")
    ]

    with tempfile.TemporaryDirectory() as temp_home:
        env = os.environ.copy()
        env["EPISODIC_HOME"] = temp_home
        env["EPISODIC_DB_PATH"] = os.path.join(temp_home, "episodic.db")
        env["HOME"] = temp_home

        for model, _ in models_to_test:
            test_script = f"""
/init --erase
/model chat {model}
Say hello in 5 words
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
