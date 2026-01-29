"""Unit tests for MQL lexer."""

import pytest
from episodic.query import tokenize, TokenKind, KEYWORD_MAP


class TestLexerBasics:
    """Basic lexer functionality tests."""

    def test_empty_input(self):
        """Empty input should produce only EOF."""
        result = tokenize("")
        assert len(result.tokens) == 1
        assert result.tokens[0].kind == TokenKind.EOF

    def test_single_word(self):
        """Single word should tokenize correctly."""
        result = tokenize("hello")
        assert len(result.tokens) == 2
        assert result.tokens[0].kind == TokenKind.WORD
        assert result.tokens[0].lexeme == "hello"
        assert result.tokens[0].normalized == "hello"
        assert result.tokens[1].kind == TokenKind.EOF

    def test_word_span(self):
        """Word span should be correct."""
        result = tokenize("hello")
        assert result.tokens[0].span == (0, 5)

    def test_word_with_leading_space(self):
        """Leading space should be skipped."""
        result = tokenize("  hello")
        assert result.tokens[0].span == (2, 7)

    def test_token_indices(self):
        """Token indices should be sequential."""
        result = tokenize("hello world")
        assert result.tokens[0].index == 0
        assert result.tokens[1].index == 1
        assert result.tokens[2].index == 2  # EOF


class TestWordPattern:
    """Tests for WORD token pattern."""

    def test_apostrophe_in_word(self):
        """Apostrophe should stay in WORD: don't -> single token."""
        result = tokenize("don't")
        assert len(result.tokens) == 2
        assert result.tokens[0].kind == TokenKind.WORD
        assert result.tokens[0].lexeme == "don't"
        assert result.tokens[0].normalized == "don't"

    def test_hyphen_in_word(self):
        """Hyphen should stay in WORD: Ralph-Wiggum -> single token."""
        result = tokenize("Ralph-Wiggum")
        assert len(result.tokens) == 2
        assert result.tokens[0].kind == TokenKind.WORD
        assert result.tokens[0].lexeme == "Ralph-Wiggum"

    def test_possessive_apostrophe(self):
        """Possessive apostrophe: Fenway's -> single token."""
        result = tokenize("Fenway's")
        assert len(result.tokens) == 2
        assert result.tokens[0].kind == TokenKind.WORD
        assert result.tokens[0].lexeme == "Fenway's"

    def test_underscore_in_word(self):
        """Underscore should stay in WORD: foo_bar -> single token."""
        result = tokenize("foo_bar")
        assert len(result.tokens) == 2
        assert result.tokens[0].kind == TokenKind.WORD
        assert result.tokens[0].lexeme == "foo_bar"

    def test_single_char_word(self):
        """Single character should be valid WORD."""
        result = tokenize("a")
        assert result.tokens[0].kind == TokenKind.WORD
        assert result.tokens[0].lexeme == "a"

    def test_single_digit_word(self):
        """Single digit should become NUMBER."""
        result = tokenize("5")
        assert result.tokens[0].kind == TokenKind.NUMBER
        assert result.tokens[0].lexeme == "5"


class TestPunctuation:
    """Tests for punctuation tokenization."""

    def test_colon(self):
        """Colon should be COLON token."""
        result = tokenize("topic:")
        assert result.tokens[1].kind == TokenKind.COLON
        assert result.tokens[1].lexeme == ":"

    def test_comma(self):
        """Comma should be COMMA token."""
        result = tokenize("a,b")
        assert result.tokens[1].kind == TokenKind.COMMA

    def test_question(self):
        """Question mark should be QUESTION token."""
        result = tokenize("what?")
        assert result.tokens[1].kind == TokenKind.QUESTION

    def test_lparen(self):
        """Left paren should be LPAREN token."""
        result = tokenize("(test")
        assert result.tokens[0].kind == TokenKind.LPAREN

    def test_rparen(self):
        """Right paren should be RPAREN token."""
        result = tokenize("test)")
        assert result.tokens[1].kind == TokenKind.RPAREN

    def test_standalone_dash(self):
        """Standalone dash should be DASH token."""
        result = tokenize("- foo")
        assert result.tokens[0].kind == TokenKind.DASH


class TestLeadingDash:
    """Tests for leading dash handling (CRITICAL)."""

    def test_leading_dash_lex_error(self):
        """Leading dash before alphanumeric should be LEX_ERROR."""
        result = tokenize("-foo")
        assert result.has_error
        assert result.tokens[0].kind == TokenKind.LEX_ERROR
        assert result.error_code == "leading_dash"

    def test_trailing_dash_standalone(self):
        """Trailing standalone dash should be DASH token."""
        result = tokenize("foo -")
        assert result.tokens[1].kind == TokenKind.DASH
        assert not result.has_error


class TestQuotedStrings:
    """Tests for quoted string tokenization."""

    def test_double_quoted(self):
        """Double quoted string should be QUOTED token."""
        result = tokenize('"hello world"')
        assert len(result.tokens) == 2
        assert result.tokens[0].kind == TokenKind.QUOTED
        assert result.tokens[0].lexeme == '"hello world"'
        assert result.tokens[0].normalized == "hello world"

    def test_single_quoted(self):
        """Single quoted string should be QUOTED token."""
        result = tokenize("'hello'")
        assert result.tokens[0].kind == TokenKind.QUOTED
        assert result.tokens[0].normalized == "hello"

    def test_unclosed_quote_lex_error(self):
        """Unclosed quote should produce LEX_ERROR."""
        result = tokenize('"unclosed')
        assert result.has_error
        assert result.tokens[0].kind == TokenKind.LEX_ERROR
        assert result.error_code == "lex_error_unclosed_quote"

    def test_empty_quoted_string(self):
        """Empty quoted string should work."""
        result = tokenize('""')
        assert result.tokens[0].kind == TokenKind.QUOTED
        assert result.tokens[0].normalized == ""


class TestISODate:
    """Tests for ISO date tokenization."""

    def test_iso_date(self):
        """ISO date YYYY-MM-DD should be ISO_DATE token."""
        result = tokenize("2026-01-25")
        assert result.tokens[0].kind == TokenKind.ISO_DATE
        assert result.tokens[0].lexeme == "2026-01-25"

    def test_iso_date_in_context(self):
        """ISO date in context should tokenize correctly."""
        result = tokenize("on 2026-01-25 coffee")
        assert result.tokens[0].kind == TokenKind.KW_ON
        assert result.tokens[1].kind == TokenKind.ISO_DATE
        assert result.tokens[2].kind == TokenKind.WORD


class TestNumbers:
    """Tests for number tokenization."""

    def test_number(self):
        """Number should be NUMBER token."""
        result = tokenize("42")
        assert result.tokens[0].kind == TokenKind.NUMBER
        assert result.tokens[0].lexeme == "42"

    def test_multi_digit_number(self):
        """Multi-digit number should be single token."""
        result = tokenize("12345")
        assert result.tokens[0].kind == TokenKind.NUMBER
        assert result.tokens[0].lexeme == "12345"


class TestKeywords:
    """Tests for keyword recognition."""

    def test_mode_keyword_browse(self):
        """'browse' should be KW_MODE."""
        result = tokenize("browse")
        assert result.tokens[0].kind == TokenKind.KW_MODE
        assert result.tokens[0].normalized == "browse"

    def test_mode_keyword_show(self):
        """'show' should normalize to browse."""
        result = tokenize("show")
        assert result.tokens[0].kind == TokenKind.KW_MODE
        assert result.tokens[0].normalized == "browse"

    def test_segment_keyword(self):
        """'topic' should be KW_SEGMENT."""
        result = tokenize("topic")
        assert result.tokens[0].kind == TokenKind.KW_SEGMENT

    def test_speaker_keyword_i(self):
        """'I' should be KW_SPEAKER with normalized 'user'."""
        result = tokenize("I")
        assert result.tokens[0].kind == TokenKind.KW_SPEAKER
        assert result.tokens[0].normalized == "user"

    def test_speaker_keyword_you(self):
        """'you' should be KW_SPEAKER with normalized 'assistant'."""
        result = tokenize("you")
        assert result.tokens[0].kind == TokenKind.KW_SPEAKER
        assert result.tokens[0].normalized == "assistant"

    def test_speaker_keyword_we(self):
        """'we' should be KW_SPEAKER with normalized 'both'."""
        result = tokenize("we")
        assert result.tokens[0].kind == TokenKind.KW_SPEAKER
        assert result.tokens[0].normalized == "both"

    def test_have_keyword(self):
        """'have' should be KW_HAVE."""
        result = tokenize("have")
        assert result.tokens[0].kind == TokenKind.KW_HAVE

    def test_has_keyword(self):
        """'has' should be KW_HAVE with normalized 'have'."""
        result = tokenize("has")
        assert result.tokens[0].kind == TokenKind.KW_HAVE
        assert result.tokens[0].normalized == "have"

    def test_time_keyword(self):
        """'time' should be KW_TIME."""
        result = tokenize("time")
        assert result.tokens[0].kind == TokenKind.KW_TIME

    def test_discourse_keyword_before(self):
        """'before' should be KW_DISCOURSE."""
        result = tokenize("before")
        assert result.tokens[0].kind == TokenKind.KW_DISCOURSE

    def test_discourse_keyword_previously(self):
        """'previously' should be KW_DISCOURSE."""
        result = tokenize("previously")
        assert result.tokens[0].kind == TokenKind.KW_DISCOURSE

    def test_ever_keyword(self):
        """'ever' should be KW_EVER."""
        result = tokenize("ever")
        assert result.tokens[0].kind == TokenKind.KW_EVER

    def test_keyword_case_insensitive(self):
        """Keywords should be case-insensitive."""
        result = tokenize("BROWSE")
        assert result.tokens[0].kind == TokenKind.KW_MODE
        assert result.tokens[0].lexeme == "BROWSE"
        assert result.tokens[0].normalized == "browse"


class TestUnknownCharacters:
    """Tests for unknown character handling (CRITICAL)."""

    def test_unknown_char_lex_error(self):
        """Unknown character should emit LEX_ERROR, not silently skip."""
        result = tokenize("foo@bar")
        assert result.has_error
        # Find the @ token
        at_token = [t for t in result.tokens if t.lexeme == "@"]
        assert len(at_token) == 1
        assert at_token[0].kind == TokenKind.LEX_ERROR

    def test_at_sign_lex_error(self):
        """@ should produce LEX_ERROR."""
        result = tokenize("@browse")
        assert result.has_error
        assert result.tokens[0].kind == TokenKind.LEX_ERROR
        assert result.error_code == "unknown_char"

    def test_hash_lex_error(self):
        """# should produce LEX_ERROR."""
        result = tokenize("#tag")
        assert result.has_error
        assert result.tokens[0].kind == TokenKind.LEX_ERROR

    def test_multiple_unknown_chars(self):
        """Multiple unknown chars should all be LEX_ERROR."""
        result = tokenize("@#$")
        assert result.has_error
        assert sum(1 for t in result.tokens if t.kind == TokenKind.LEX_ERROR) == 3


class TestKeywordMapCoverage:
    """Tests for KEYWORD_MAP coverage."""

    def test_all_keywords_in_map(self):
        """All documented keywords should be in KEYWORD_MAP."""
        expected_keywords = [
            # Mode
            "browse", "show", "list", "display", "summarize", "summary", "answer",
            # Segment
            "topic", "topics", "segment", "segments",
            # Speaker
            "i", "me", "my", "you", "your", "we", "us", "our", "user", "assistant",
            # Time relative
            "yesterday", "today", "last", "this", "week", "month", "year", "ago", "days", "day",
            # Time noun
            "time",
            # Discussion
            "discussed", "discuss", "talked", "talk", "mentioned", "mention",
            "brought", "bring", "said", "say", "asked", "ask",
            # Query openers
            "when", "where", "did", "have", "has", "ever",
            # Scope
            "in", "within", "on", "about",
            # Deictic
            "earlier", "previous",
            # Discourse
            "before", "previously", "already",
        ]
        for kw in expected_keywords:
            assert kw in KEYWORD_MAP, f"Missing keyword: {kw}"
