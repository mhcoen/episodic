from unittest.mock import patch

from episodic.web_extract import fetch_page_content_sync


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
