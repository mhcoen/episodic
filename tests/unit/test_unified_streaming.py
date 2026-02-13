#!/usr/bin/env python3
"""
Unit tests for unified streaming behavior.
"""

import unittest
from unittest.mock import patch

from episodic.config import config
from episodic.unified_streaming import (
    unified_stream_response,
    unified_stream_text,
    _strip_inline_source_citations_for_tts,
)


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

    def test_strip_inline_source_citations_for_tts(self):
        text = "Great pick [Source 2]. Also see this [Sources 1, 3]."
        cleaned = _strip_inline_source_citations_for_tts(text)
        self.assertEqual(cleaned, "Great pick . Also see this .")

    @patch("episodic.unified_streaming.secho_color")
    @patch("episodic.unified_streaming.typer.echo")
    def test_osc8_hyperlink_sequence_preserved_in_streamer(self, mock_echo, mock_secho_color):
        """OSC8 hyperlink escape sequence should survive unified streaming path."""
        config.set("stream_rate", 0)
        osc8 = "\033]8;;https://example.com\033\\https://example.com\033]8;;\033\\"

        returned = unified_stream_text(
            f"Sources:\n{osc8}",
            model="test-model",
            preserve_formatting=False,
            enable_tts=False,
        )

        self.assertIn(osc8, returned)
        printed_chunks = "".join(call.args[0] for call in mock_secho_color.call_args_list if call.args)
        self.assertIn(osc8, printed_chunks)
