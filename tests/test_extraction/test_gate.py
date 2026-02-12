"""Tests for the keyword gate (episodic.mcp.extraction.gate)."""

import pytest

from episodic.mcp.extraction.gate import _normalize_token, _tokenize, matched_domains


class TestTokenization:
    def test_lowercase(self):
        tokens = _tokenize("CHECK MY CALENDAR")
        assert all(t == t.lower() for t in tokens)

    def test_punctuation_stripped(self):
        tokens = _tokenize("What's on my calendar?")
        assert "?" not in "".join(tokens)
        assert "'" not in "".join(tokens)

    def test_unicode_normalization(self):
        # NFKC normalizes full-width characters
        tokens = _tokenize("\uff43\uff41\uff4c\uff45\uff4e\uff44\uff41\uff52")  # "calendar" in fullwidth
        assert "calendar" in tokens

    def test_filler_removal(self):
        tokens = _tokenize("umm check my uhh email")
        assert "umm" not in tokens
        assert "uhh" not in tokens
        assert "check" in tokens
        assert "email" in tokens

    def test_extra_whitespace(self):
        tokens = _tokenize("  check   my   calendar  ")
        assert tokens == _tokenize("check my calendar")

    def test_empty_input(self):
        assert _tokenize("") == []
        assert _tokenize("   ") == []


class TestNormalization:
    def test_plural_s(self):
        assert _normalize_token("appointments") == "appointment"
        assert _normalize_token("emails") == "email"
        assert _normalize_token("meetings") == "meeting"

    def test_plural_es(self):
        assert _normalize_token("schedules") == "schedule"
        assert _normalize_token("reserves") == "reserve"

    def test_exceptions_not_stripped(self):
        assert _normalize_token("busy") == "busy"
        assert _normalize_token("free") == "free"
        assert _normalize_token("is") == "is"

    def test_short_words_unchanged(self):
        assert _normalize_token("to") == "to"
        assert _normalize_token("my") == "my"


class TestKeywordMatching:
    def test_single_keyword_calendar(self):
        assert "calendar" in matched_domains("check my calendar")

    def test_single_keyword_email(self):
        assert "email" in matched_domains("check my email")

    def test_plural_keyword_matches(self):
        assert "calendar" in matched_domains("I have two meetings tomorrow")

    def test_no_keywords_empty_set(self):
        assert matched_domains("hello how are you") == set()

    def test_chat_no_domain_words(self):
        assert matched_domains("the weather is nice today") == set()

    def test_cross_domain(self):
        domains = matched_domains("email Bob about tomorrow's meeting")
        assert "email" in domains
        assert "calendar" in domains

    def test_empty_input(self):
        assert matched_domains("") == set()
        assert matched_domains("   ") == set()

    def test_none_like_empty(self):
        # Edge case: only whitespace
        assert matched_domains("\t\n") == set()


class TestPhraseMatching:
    def test_exact_phrase(self):
        assert "calendar" in matched_domains("am I free Thursday?")

    def test_phrase_with_intervening_words(self):
        # "set up a meeting" with "quick" intervening — within 5-token window
        assert "calendar" in matched_domains("set up a quick meeting")

    def test_phrase_outside_window(self):
        # Phrase tokens too far apart (> 5 token window)
        result = matched_domains("set blah blah blah blah blah up a meeting")
        # "set" and "up" are 6 tokens apart, outside window
        # But "meeting" is a keyword, so calendar still matches
        assert "calendar" in result

    def test_email_phrase(self):
        assert "email" in matched_domains("I need to follow up on that")

    def test_whats_on_my_phrase(self):
        assert "calendar" in matched_domains("what's on my schedule today")


class TestSpeechArtifacts:
    def test_umm_and_uhh(self):
        assert "email" in matched_domains("umm check my uhh email")

    def test_extra_whitespace(self):
        assert "calendar" in matched_domains("  check   my   calendar  ")

    def test_all_caps(self):
        assert "calendar" in matched_domains("CHECK MY CALENDAR TOMORROW")

    def test_mixed_case(self):
        assert "email" in matched_domains("Check My Email Please")
