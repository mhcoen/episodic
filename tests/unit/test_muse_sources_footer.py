"""Unit tests for Muse source URL footer display helpers."""

from episodic.conversation_pipeline_llm import (
    _format_clickable_url,
    _select_muse_source_urls,
)


def test_select_muse_source_urls_top_three(monkeypatch):
    from episodic.config import config

    monkeypatch.setattr(config, "get", lambda k, d=None: "top-three" if k == "muse_sources" else d)

    web_context = {
        "results": [
            {"url": "https://a.example"},
            {"url": "https://b.example"},
            {"url": "https://c.example"},
            {"url": "https://d.example"},
        ]
    }

    assert _select_muse_source_urls(web_context) == [
        "https://a.example",
        "https://b.example",
        "https://c.example",
    ]


def test_select_muse_source_urls_first_only(monkeypatch):
    from episodic.config import config

    monkeypatch.setattr(config, "get", lambda k, d=None: "first-only" if k == "muse_sources" else d)

    web_context = {
        "results": [
            {"url": "https://a.example"},
            {"url": "https://b.example"},
        ]
    }

    assert _select_muse_source_urls(web_context) == ["https://a.example"]


def test_select_muse_source_urls_all_relevant_dedupes(monkeypatch):
    from episodic.config import config

    monkeypatch.setattr(config, "get", lambda k, d=None: "all-relevant" if k == "muse_sources" else d)

    web_context = {
        "results": [
            {"url": "https://a.example"},
            {"url": "https://a.example"},
            {"url": "https://b.example"},
        ]
    }

    assert _select_muse_source_urls(web_context) == [
        "https://a.example",
        "https://b.example",
    ]


def test_format_clickable_url_plain_when_not_tty(monkeypatch):
    from episodic import conversation_pipeline_llm as llm_pipeline

    class _Stdout:
        @staticmethod
        def isatty():
            return False

    monkeypatch.setattr(llm_pipeline, "sys", type("S", (), {"stdout": _Stdout()})())
    assert _format_clickable_url("https://example.com") == "https://example.com"


def test_format_clickable_url_osc8_when_tty(monkeypatch):
    from episodic import conversation_pipeline_llm as llm_pipeline

    class _Stdout:
        @staticmethod
        def isatty():
            return True

    monkeypatch.setattr(llm_pipeline, "sys", type("S", (), {"stdout": _Stdout()})())
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_OSC8", raising=False)
    monkeypatch.delenv("EPISODIC_NO_OSC8", raising=False)

    out = _format_clickable_url("https://example.com")
    assert "]8;;https://example.com" in out


def test_format_clickable_url_respects_opt_out(monkeypatch):
    from episodic import conversation_pipeline_llm as llm_pipeline

    class _Stdout:
        @staticmethod
        def isatty():
            return True

    monkeypatch.setattr(llm_pipeline, "sys", type("S", (), {"stdout": _Stdout()})())
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("EPISODIC_NO_OSC8", "1")

    assert _format_clickable_url("https://example.com") == "https://example.com"


def test_format_clickable_url_plain_on_apple_terminal(monkeypatch):
    from episodic import conversation_pipeline_llm as llm_pipeline

    class _Stdout:
        @staticmethod
        def isatty():
            return True

    monkeypatch.setattr(llm_pipeline, "sys", type("S", (), {"stdout": _Stdout()})())
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    monkeypatch.delenv("NO_OSC8", raising=False)
    monkeypatch.delenv("EPISODIC_NO_OSC8", raising=False)

    assert _format_clickable_url("https://example.com") == "https://example.com"
