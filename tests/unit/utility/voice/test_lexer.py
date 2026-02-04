"""
Comprehensive Lexer Tests.

Tests for episodic/utility/voice/lexer.py including:
- Maximal munch (longest match semantics)
- Multi-word token recognition
- Single-word token classification
- Number token handling
- Token boundary cases
"""

import pytest
from episodic.utility.voice.lexer import Lexer, Token


# =============================================================================
# Multi-word Token Tests (Maximal Munch)
# =============================================================================

class TestLexerMultiwordTokens:
    """Test multi-word token recognition with maximal munch."""

    @pytest.mark.parametrize("text,first_token_kind,first_token_value", [
        ("what is the time", "QUERY", "what is"),
        ("tell me the weather", "QUERY", "tell me"),
        ("show me the forecast", "QUERY", "show me"),
        ("give me the news", "QUERY", "give me"),
        ("how is the weather", "QUERY", "how is"),
    ])
    def test_query_markers(self, text, first_token_kind, first_token_value):
        lexer = Lexer()
        tokens = lexer.tokenize(text)
        assert tokens[0].kind == first_token_kind
        assert tokens[0].value == first_token_value

    @pytest.mark.parametrize("text,first_token_kind,first_token_value", [
        ("turn on the radio", "ACTION_ON", "turn on"),
        ("turn off the music", "ACTION_OFF", "turn off"),
        ("put on some jazz", "ACTION_ON", "put on"),
    ])
    def test_action_phrases(self, text, first_token_kind, first_token_value):
        lexer = Lexer()
        tokens = lexer.tokenize(text)
        assert tokens[0].kind == first_token_kind
        assert tokens[0].value == first_token_value

    @pytest.mark.parametrize("text,expected_kind", [
        ("do not disturb", "KW_DND"),
        ("wake me up at 7", "KW_ALARM"),
    ])
    def test_multiword_keywords(self, text, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(text)
        # First token should be the multiword match
        assert any(t.kind == expected_kind for t in tokens)

    @pytest.mark.parametrize("text,expected_kind", [
        ("at 7 in the morning", "AMPM"),
        ("at 3 in the afternoon", "AMPM"),
        ("at 8 in the evening", "AMPM"),
        ("at 10 at night", "AMPM"),
    ])
    def test_time_of_day_phrases(self, text, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(text)
        assert any(t.kind == expected_kind for t in tokens)

    @pytest.mark.parametrize("text,expected_kind", [
        ("quarter past 7", "TIME_QUARTER_PAST"),
        ("quarter to 8", "TIME_QUARTER_TO"),
        ("half past 6", "TIME_HALF_PAST"),
    ])
    def test_time_patterns(self, text, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(text)
        assert any(t.kind == expected_kind for t in tokens)


class TestLexerMaximalMunchOrder:
    """Test that maximal munch prefers longer matches."""

    def test_what_is_over_what(self):
        """'what is' should match before 'what' alone."""
        lexer = Lexer()
        tokens = lexer.tokenize("what is the time")
        assert tokens[0].kind == "QUERY"
        assert tokens[0].value == "what is"
        # NOT: tokens[0].value == "what"

    def test_turn_on_over_turn(self):
        """'turn on' should match before 'turn' alone."""
        lexer = Lexer()
        tokens = lexer.tokenize("turn on the lights")
        assert tokens[0].kind == "ACTION_ON"
        assert tokens[0].value == "turn on"


# =============================================================================
# Single-word Token Tests
# =============================================================================

class TestLexerQueryMarkers:
    """Test single-word query markers."""

    @pytest.mark.parametrize("word,expected_kind", [
        ("what", "QUERY"),
        ("when", "QUERY"),
        ("where", "QUERY"),
        ("how", "QUERY"),
        ("which", "QUERY"),
    ])
    def test_query_words(self, word, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(word)
        assert tokens[0].kind == expected_kind


class TestLexerActionVerbs:
    """Test action verb classification."""

    @pytest.mark.parametrize("word,expected_kind", [
        ("set", "ACTION_SET"),
        ("start", "ACTION_SET"),
        ("create", "ACTION_SET"),
        ("cancel", "ACTION_CANCEL"),
        ("delete", "ACTION_CANCEL"),
        ("remove", "ACTION_CANCEL"),
        ("play", "ACTION_PLAY"),
        ("pause", "ACTION_PAUSE"),
        ("resume", "ACTION_RESUME"),
        ("stop", "ACTION_STOP"),
        ("skip", "ACTION_SKIP"),
        ("add", "ACTION_ADD"),
        ("remind", "ACTION_REMIND"),
    ])
    def test_action_verbs(self, word, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(word)
        assert tokens[0].kind == expected_kind


class TestLexerDomainKeywords:
    """Test domain keyword classification."""

    @pytest.mark.parametrize("word,expected_kind", [
        ("timer", "KW_TIMER"),
        ("timers", "KW_TIMER"),
        ("alarm", "KW_ALARM"),
        ("alarms", "KW_ALARM"),
        ("reminder", "KW_REMIND"),
        ("reminders", "KW_REMIND"),
        ("weather", "KW_WEATHER"),
        ("forecast", "KW_WEATHER"),
        ("temperature", "KW_WEATHER"),
        ("news", "KW_NEWS"),
        ("headlines", "KW_NEWS"),
        ("note", "KW_NOTE"),
        ("notes", "KW_NOTE"),
        ("time", "KW_TIME"),
        ("clock", "KW_TIME"),
        ("date", "KW_DATE"),
        ("day", "KW_DATE"),
        ("radio", "KW_RADIO"),
        ("station", "KW_RADIO"),
        ("music", "KW_MUSIC"),
        ("song", "KW_MUSIC"),
    ])
    def test_domain_keywords(self, word, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(word)
        assert tokens[0].kind == expected_kind


class TestLexerTimeUnits:
    """Test time unit classification."""

    @pytest.mark.parametrize("word,expected_kind", [
        ("second", "TIME_UNIT"),
        ("seconds", "TIME_UNIT"),
        ("sec", "TIME_UNIT"),
        ("secs", "TIME_UNIT"),
        ("minute", "TIME_UNIT"),
        ("minutes", "TIME_UNIT"),
        ("min", "TIME_UNIT"),
        ("mins", "TIME_UNIT"),
        ("hour", "TIME_UNIT"),
        ("hours", "TIME_UNIT"),
        ("hr", "TIME_UNIT"),
        ("hrs", "TIME_UNIT"),
    ])
    def test_time_units(self, word, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(word)
        assert tokens[0].kind == expected_kind


class TestLexerDurationWords:
    """Test duration word classification."""

    @pytest.mark.parametrize("word,expected_kind", [
        ("couple", "DURATION_WORD"),
        ("few", "DURATION_WORD"),
        ("half", "DURATION_WORD"),
        ("quarter", "DURATION_WORD"),
    ])
    def test_duration_words(self, word, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(word)
        assert tokens[0].kind == expected_kind


class TestLexerRelativeTime:
    """Test relative time word classification."""

    @pytest.mark.parametrize("word,expected_kind", [
        ("today", "RELATIVE_DAY"),
        ("tomorrow", "RELATIVE_DAY"),
        ("tonight", "RELATIVE_DAY"),
        ("morning", "RELATIVE_DAY"),
        ("afternoon", "RELATIVE_DAY"),
        ("evening", "RELATIVE_DAY"),
    ])
    def test_relative_time_words(self, word, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(word)
        assert tokens[0].kind == expected_kind


class TestLexerAMPM:
    """Test AM/PM classification."""

    @pytest.mark.parametrize("word,expected_kind", [
        ("am", "AMPM"),
        ("pm", "AMPM"),
    ])
    def test_ampm(self, word, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(word)
        assert tokens[0].kind == expected_kind


class TestLexerPrepositions:
    """Test preposition classification."""

    @pytest.mark.parametrize("word,expected_kind", [
        ("at", "PREP_AT"),
        ("in", "PREP_IN"),
        ("for", "PREP_FOR"),
        ("on", "PREP_ON"),
        ("to", "PREP_TO"),
        ("from", "PREP_FROM"),
    ])
    def test_prepositions(self, word, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(word)
        assert tokens[0].kind == expected_kind


class TestLexerArticles:
    """Test article classification."""

    @pytest.mark.parametrize("word,expected_kind", [
        ("a", "ARTICLE"),
        ("an", "ARTICLE"),
        ("the", "ARTICLE"),
    ])
    def test_articles(self, word, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(word)
        assert tokens[0].kind == expected_kind


class TestLexerPoliteness:
    """Test politeness word classification."""

    @pytest.mark.parametrize("word,expected_kind", [
        ("please", "POLITENESS"),
        ("could", "POLITENESS"),
        ("would", "POLITENESS"),
    ])
    def test_politeness(self, word, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(word)
        assert tokens[0].kind == expected_kind


class TestLexerConjunctions:
    """Test conjunction classification."""

    @pytest.mark.parametrize("word,expected_kind", [
        ("and", "CONJ"),
        ("or", "CONJ"),
        ("then", "CONJ"),
        ("also", "CONJ"),
    ])
    def test_conjunctions(self, word, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(word)
        assert tokens[0].kind == expected_kind


class TestLexerNamedMarkers:
    """Test named/labeled markers."""

    @pytest.mark.parametrize("word,expected_kind", [
        ("called", "NAMED"),
        ("named", "NAMED"),
        ("labeled", "NAMED"),
    ])
    def test_named_markers(self, word, expected_kind):
        lexer = Lexer()
        tokens = lexer.tokenize(word)
        assert tokens[0].kind == expected_kind


# =============================================================================
# Number Token Tests
# =============================================================================

class TestLexerNumbers:
    """Test number token classification."""

    @pytest.mark.parametrize("text", [
        "5", "10", "30", "100", "1234",
    ])
    def test_digit_strings(self, text):
        lexer = Lexer()
        tokens = lexer.tokenize(text)
        assert tokens[0].kind == "NUMBER"

    def test_numbers_in_context(self):
        lexer = Lexer()
        tokens = lexer.tokenize("timer 10 minutes")
        assert tokens[1].kind == "NUMBER"
        assert tokens[1].value == "10"


# =============================================================================
# Unknown Word Tests
# =============================================================================

class TestLexerUnknownWords:
    """Test unknown word classification."""

    @pytest.mark.parametrize("word", [
        "npr", "jazz", "chicken", "roast", "call", "mom",
    ])
    def test_unknown_words_are_WORD(self, word):
        lexer = Lexer()
        tokens = lexer.tokenize(word)
        assert tokens[0].kind == "WORD"


# =============================================================================
# Token Position Tests
# =============================================================================

class TestLexerTokenPositions:
    """Test token position tracking."""

    def test_positions_are_word_indices(self):
        lexer = Lexer()
        tokens = lexer.tokenize("set a timer for 10 minutes")
        # set=0, a=1, timer=2, for=3, 10=4, minutes=5
        assert tokens[0].position == 0  # set
        assert tokens[1].position == 1  # a
        assert tokens[2].position == 2  # timer
        assert tokens[3].position == 3  # for
        assert tokens[4].position == 4  # 10
        assert tokens[5].position == 5  # minutes

    def test_multiword_advances_position(self):
        lexer = Lexer()
        tokens = lexer.tokenize("what is the time")
        # "what is" consumes positions 0,1
        assert tokens[0].position == 0  # what is
        assert tokens[1].position == 2  # the


# =============================================================================
# Full Sentence Tests
# =============================================================================

class TestLexerFullSentences:
    """Test tokenization of full sentences."""

    def test_timer_command(self):
        lexer = Lexer()
        tokens = lexer.tokenize("set a timer for 10 minutes")
        kinds = [t.kind for t in tokens]
        assert kinds == ["ACTION_SET", "ARTICLE", "KW_TIMER", "PREP_FOR", "NUMBER", "TIME_UNIT"]

    def test_time_query(self):
        lexer = Lexer()
        tokens = lexer.tokenize("what is the time")
        kinds = [t.kind for t in tokens]
        assert "QUERY" in kinds
        assert "KW_TIME" in kinds

    def test_alarm_command(self):
        lexer = Lexer()
        tokens = lexer.tokenize("set an alarm for 7 am")
        kinds = [t.kind for t in tokens]
        assert "ACTION_SET" in kinds
        assert "KW_ALARM" in kinds
        assert "AMPM" in kinds

    def test_weather_query(self):
        lexer = Lexer()
        tokens = lexer.tokenize("what is the weather")
        kinds = [t.kind for t in tokens]
        assert "QUERY" in kinds
        assert "KW_WEATHER" in kinds
