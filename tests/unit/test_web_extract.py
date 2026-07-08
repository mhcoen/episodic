from unittest.mock import patch

import pytest

from episodic.web_extract import fetch_page_content_sync, _is_usable_extracted_content


@pytest.fixture(autouse=True)
def _allow_fetch(monkeypatch):
    # These tests exercise the fetch/extract logic with a mocked HTTP layer.
    # The SSRF guard (covered by test_web_extract_ssrf.py) does live DNS and
    # would reject the non-resolving example hosts before the mock runs.
    monkeypatch.setattr("episodic.web_extract._is_safe_public_url", lambda url: True)


def test_fetch_failure_silent_when_web_debug_off():
    with patch("episodic.web_extract.debug_enabled", return_value=False), \
         patch("requests.get", side_effect=AttributeError("boom")), \
         patch("episodic.web_extract.typer.secho") as mock_secho:
        result = fetch_page_content_sync("https://chicago.eater.com/restaurants")

    assert result is None
    assert mock_secho.call_count == 0


def test_fetch_failure_logs_when_web_debug_on():
    with patch("episodic.web_extract.debug_enabled", side_effect=lambda c: c == "web"), \
         patch("requests.get", side_effect=AttributeError("boom")), \
         patch("episodic.web_extract.typer.secho") as mock_secho:
        result = fetch_page_content_sync("https://chicago.eater.com/restaurants")

    assert result is None
    assert mock_secho.call_count == 1
    message = mock_secho.call_args.args[0]
    assert "Failed to fetch chicago.eater.com: AttributeError: boom" in message


def test_fetch_failure_logs_trace_when_muse_debug_on():
    with patch("episodic.web_extract.debug_enabled", side_effect=lambda c: c == "muse"), \
         patch("requests.get", side_effect=AttributeError("boom")), \
         patch("episodic.web_extract.typer.secho") as mock_secho:
        result = fetch_page_content_sync("https://chicago.eater.com/restaurants")

    assert result is None
    assert mock_secho.call_count >= 2
    first_message = mock_secho.call_args_list[0].args[0]
    second_message = mock_secho.call_args_list[1].args[0]
    assert "Failed to fetch chicago.eater.com: AttributeError: boom" in first_message
    assert "Traceback" in second_message


def test_usable_content_rejects_placeholder():
    assert _is_usable_extracted_content("Could not extract content from this page.") is False


def test_usable_content_accepts_meaningful_short_text_with_config():
    with patch("episodic.web_extract.config.get", side_effect=lambda k, d=None: 10 if k == "muse_extract_min_chars" else d):
        assert _is_usable_extracted_content("Event: Live jazz at 8pm") is True


def test_sync_fetch_prefers_trafilatura_when_available():
    class _Resp:
        status_code = 200
        text = "<html><body>placeholder</body></html>"

    with patch("requests.get", return_value=_Resp()), \
         patch("episodic.web_extract._extract_with_trafilatura", return_value="Event list with times and venues"), \
         patch("episodic.web_extract._sanitize_soup"), \
         patch("episodic.web_extract.config.get", side_effect=lambda k, d=None: {
             "muse_sanitize_html": True,
             "muse_extract_min_chars": 20,
         }.get(k, d)):
        result = fetch_page_content_sync("https://example.com/events")

    assert result == "Event list with times and venues"
