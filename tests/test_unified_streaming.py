#!/usr/bin/env python3
"""
Unit tests for unified streaming behavior.
"""

import unittest
from unittest.mock import patch

from episodic.config import config
from episodic.unified_streaming import unified_stream_response


class TestUnifiedStreaming(unittest.TestCase):
    """Test unified streaming edge cases."""

    def setUp(self):
        self.original_stream_rate = config.get("stream_rate")

    def tearDown(self):
        if self.original_stream_rate is not None:
            config.set("stream_rate", self.original_stream_rate)
        else:
            config.delete("stream_rate")

    @patch("episodic.unified_streaming.typer.secho")
    @patch("episodic.unified_streaming.typer.echo")
    @patch("episodic.unified_streaming.process_stream_response")
    def test_keyboard_interrupt_returns_partial(self, mock_process, mock_echo, mock_secho):
        """KeyboardInterrupt should return partial response with marker."""
        def gen():
            yield "Hello"
            raise KeyboardInterrupt

        config.set("stream_rate", 0)
        mock_process.return_value = gen()

        result = unified_stream_response(
            stream_generator=iter(()),
            model="test-model",
            preserve_formatting=False
        )

        self.assertIn("Hello", result)
        self.assertTrue(result.endswith("[Response interrupted by user]"))

    @patch("episodic.unified_streaming.process_stream_response")
    def test_stream_exception_propagates(self, mock_process):
        """Unexpected stream errors should propagate."""
        def gen():
            yield "Hello"
            raise RuntimeError("boom")

        config.set("stream_rate", 0)
        mock_process.return_value = gen()

        with self.assertRaises(RuntimeError):
            unified_stream_response(
                stream_generator=iter(()),
                model="test-model",
                preserve_formatting=False
            )
