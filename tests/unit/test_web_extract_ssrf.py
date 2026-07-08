"""SSRF guard and TLS-host matching for web_extract."""

import pytest

from episodic import web_extract as w


class TestSafeUrlGuard:
    @pytest.mark.parametrize("url", [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://127.0.0.1/",
        "http://localhost/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "file:///etc/passwd",
        "ftp://example.com/",
        "http://[::1]/",
        "not-a-url",
        "",
    ])
    def test_unsafe_urls_rejected(self, url):
        assert w._is_safe_public_url(url) is False
        assert w._reject_unsafe_url(url) is True

    def test_public_url_allowed(self, monkeypatch):
        # Mock DNS so the test doesn't depend on network: a public IP passes.
        def fake_getaddrinfo(host, port, *a, **k):
            return [(2, 1, 6, "", ("93.184.216.34", port))]  # public address
        monkeypatch.setattr(w.socket, "getaddrinfo", fake_getaddrinfo)
        assert w._is_safe_public_url("https://example.com/") is True

    def test_private_ip_rejected_even_if_dns_says_so(self, monkeypatch):
        # A public-looking host that resolves to a private IP (DNS rebinding
        # style) is still rejected.
        def fake_getaddrinfo(host, port, *a, **k):
            return [(2, 1, 6, "", ("10.0.0.7", port))]
        monkeypatch.setattr(w.socket, "getaddrinfo", fake_getaddrinfo)
        assert w._is_safe_public_url("https://sneaky.example/") is False


class TestRelaxedTlsHostMatching:
    def test_substring_url_not_matched(self):
        # The classic bug: an attacker URL that merely contains a trusted
        # domain as a substring must NOT opt into relaxed TLS.
        assert w._hostname_matches(
            "http://evil.example/?ref=weather.com", w._RELAXED_TLS_HOSTS) is False

    def test_real_host_matched(self):
        assert w._hostname_matches(
            "https://www.weather.com/today", w._RELAXED_TLS_HOSTS) is True
        assert w._hostname_matches(
            "https://weather.gov/", w._RELAXED_TLS_HOSTS) is True

    def test_lookalike_domain_not_matched(self):
        assert w._hostname_matches(
            "https://weather.com.evil.test/", w._RELAXED_TLS_HOSTS) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
