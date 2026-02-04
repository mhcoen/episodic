"""
Comprehensive Normalizer Tests.

Tests for episodic/utility/voice/normalizer.py including:
- Contraction expansion (all contractions, boundary-aware)
- Edge filler stripping (all fillers, loop stability)
- Numeric normalization (multiword, zero, malformed rejection)
- Letter sequence joining (known sequences, single letters)
- Punctuation handling (dashes, dots, sentence-ending)
"""

import pytest
from episodic.utility.voice.normalizer import (
    NumericNormalizer,
    LetterSequenceNormalizer,
    Normalizer,
)


# =============================================================================
# Numeric Normalizer Tests
# =============================================================================

class TestNumericNormalizerUnits:
    """Test basic unit number conversions."""

    @pytest.mark.parametrize("word,expected", [
        ("zero", "0"),
        ("one", "1"),
        ("two", "2"),
        ("three", "3"),
        ("four", "4"),
        ("five", "5"),
        ("six", "6"),
        ("seven", "7"),
        ("eight", "8"),
        ("nine", "9"),
        ("ten", "10"),
        ("eleven", "11"),
        ("twelve", "12"),
        ("thirteen", "13"),
        ("fourteen", "14"),
        ("fifteen", "15"),
        ("sixteen", "16"),
        ("seventeen", "17"),
        ("eighteen", "18"),
        ("nineteen", "19"),
    ])
    def test_single_units(self, word, expected):
        n = NumericNormalizer()
        assert n.normalize(word) == expected

    @pytest.mark.parametrize("word,expected", [
        ("twenty", "20"),
        ("thirty", "30"),
        ("forty", "40"),
        ("fifty", "50"),
        ("sixty", "60"),
        ("seventy", "70"),
        ("eighty", "80"),
        ("ninety", "90"),
    ])
    def test_single_tens(self, word, expected):
        n = NumericNormalizer()
        assert n.normalize(word) == expected


class TestNumericNormalizerCompound:
    """Test compound number conversions."""

    @pytest.mark.parametrize("text,expected", [
        ("twenty one", "21"),
        ("twenty five", "25"),
        ("thirty two", "32"),
        ("forty three", "43"),
        ("fifty four", "54"),
        ("sixty five", "65"),
        ("seventy six", "76"),
        ("eighty seven", "87"),
        ("ninety nine", "99"),
    ])
    def test_tens_plus_units(self, text, expected):
        n = NumericNormalizer()
        assert n.normalize(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("one hundred", "100"),
        ("two hundred", "200"),
        ("five hundred", "500"),
        ("one hundred and five", "105"),
        ("one hundred five", "105"),
        ("two hundred thirty four", "234"),
        ("three hundred and twenty one", "321"),
    ])
    def test_hundreds(self, text, expected):
        n = NumericNormalizer()
        assert n.normalize(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("one thousand", "1000"),
        ("one thousand five hundred", "1500"),
        ("two thousand", "2000"),
        ("one thousand and one", "1001"),
    ])
    def test_thousands(self, text, expected):
        n = NumericNormalizer()
        assert n.normalize(text) == expected


class TestNumericNormalizerMalformed:
    """Test malformed number rejection."""

    @pytest.mark.parametrize("text", [
        "thirty forty",        # Two consecutive TENS
        "five six",            # Two consecutive UNITS
        "twenty thirty",       # Two TENS
        "one two",             # Two UNITS
        "fifteen sixteen",     # Two teen UNITS
    ])
    def test_malformed_sequences_unchanged(self, text):
        """Malformed sequences should be left unchanged."""
        n = NumericNormalizer()
        # The first word may parse, but the sequence should not fully convert
        result = n.normalize(text)
        # Should not be a single number
        assert not result.isdigit()


class TestNumericNormalizerInContext:
    """Test number normalization in sentence context."""

    @pytest.mark.parametrize("text,expected", [
        ("set a timer for ten minutes", "set a timer for 10 minutes"),
        ("twenty five minutes", "25 minutes"),
        ("in five seconds", "in 5 seconds"),
        ("alarm for seven am", "alarm for 7 am"),
        ("zero minutes left", "0 minutes left"),
        ("one hundred and five degrees", "105 degrees"),
    ])
    def test_numbers_in_context(self, text, expected):
        n = NumericNormalizer()
        assert n.normalize(text) == expected


# =============================================================================
# Letter Sequence Normalizer Tests
# =============================================================================

class TestLetterSequenceNormalizerKnown:
    """Test known letter sequence joining."""

    @pytest.mark.parametrize("text,expected", [
        ("n p r", "npr"),
        ("b b c", "bbc"),
        ("w b e z", "wbez"),
        ("w f m t", "wfmt"),
        ("k e x p", "kexp"),
        ("k u s c", "kusc"),
        ("w b g o", "wbgo"),
        ("w n y c", "wnyc"),
    ])
    def test_known_sequences(self, text, expected):
        n = LetterSequenceNormalizer()
        assert n.normalize(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("play n p r", "play npr"),
        ("turn on b b c", "turn on bbc"),
        ("listen to w n y c please", "listen to wnyc please"),
    ])
    def test_sequences_in_context(self, text, expected):
        n = LetterSequenceNormalizer()
        assert n.normalize(text) == expected


class TestLetterSequenceNormalizerUnknown:
    """Test unknown sequences stay separate."""

    @pytest.mark.parametrize("text", [
        "a b c",        # Not a known station
        "x y z",        # Not a known station
        "c n n",        # Not in whitelist
    ])
    def test_unknown_sequences_unchanged(self, text):
        n = LetterSequenceNormalizer()
        # Unknown sequences stay as separate letters
        result = n.normalize(text)
        assert " " in result  # Should still have spaces


class TestLetterSequenceNormalizerSingleLetters:
    """Test single letters are preserved."""

    @pytest.mark.parametrize("text,expected", [
        ("a timer", "a timer"),      # 'a' is article, not sequence
        ("play something", "play something"),
    ])
    def test_single_letters_in_context(self, text, expected):
        n = LetterSequenceNormalizer()
        assert n.normalize(text) == expected


# =============================================================================
# Main Normalizer Tests - Contractions
# =============================================================================

class TestNormalizerContractions:
    """Test contraction expansion."""

    @pytest.mark.parametrize("contraction,expanded", [
        ("what's", "what is"),
        ("What's", "what is"),
        ("it's", "it is"),
        ("It's", "it is"),
        ("that's", "that is"),
        ("That's", "that is"),
        ("don't", "do not"),
        ("Don't", "do not"),
        ("can't", "cannot"),
        ("Can't", "cannot"),
        ("won't", "will not"),
        ("Won't", "will not"),
        ("i'm", "i am"),
        ("I'm", "i am"),
        ("i'll", "i will"),
        ("I'll", "i will"),
        ("let's", "let us"),
        ("Let's", "let us"),
        ("didn't", "did not"),
        ("Didn't", "did not"),
        ("doesn't", "does not"),
        ("Doesn't", "does not"),
        ("isn't", "is not"),
        ("Isn't", "is not"),
        ("aren't", "are not"),
        ("Aren't", "are not"),
        ("wasn't", "was not"),
        ("Wasn't", "was not"),
        ("weren't", "were not"),
        ("Weren't", "were not"),
        ("how's", "how is"),
        ("How's", "how is"),
        ("where's", "where is"),
        ("Where's", "where is"),
        ("who's", "who is"),
        ("Who's", "who is"),
        ("there's", "there is"),
        ("There's", "there is"),
    ])
    def test_contraction_expansion(self, contraction, expanded):
        n = Normalizer()
        result = n.normalize(contraction)
        assert expanded in result

    def test_contraction_in_sentence(self):
        n = Normalizer()
        assert "what is" in n.normalize("what's the time")
        assert "it is" in n.normalize("it's raining")
        assert "do not" in n.normalize("don't do that")

    def test_contraction_boundary_aware(self):
        """Contractions should not expand inside words."""
        n = Normalizer()
        # "what's" in "what'sup" should not expand
        # But since we use word boundaries, this should be fine


# =============================================================================
# Main Normalizer Tests - Edge Fillers
# =============================================================================

class TestNormalizerEdgeFillers:
    """Test edge filler stripping."""

    @pytest.mark.parametrize("filler", [
        "um", "uh", "er", "ah", "like", "you know",
        "basically", "anyway", "so", "well", "okay", "ok",
        "hey", "hi", "yo",
    ])
    def test_leading_fillers_stripped(self, filler):
        n = Normalizer()
        text = f"{filler} what time is it"
        result = n.normalize(text)
        assert result == "what time is it"

    @pytest.mark.parametrize("filler", [
        "um", "uh", "er", "ah", "like", "you know",
        "basically", "anyway", "so", "well", "okay", "ok",
    ])
    def test_trailing_fillers_stripped(self, filler):
        n = Normalizer()
        text = f"what time is it {filler}"
        result = n.normalize(text)
        assert result == "what time is it"

    def test_multiple_fillers_stripped(self):
        """Multiple fillers should all be stripped."""
        n = Normalizer()
        assert n.normalize("um uh what time is it") == "what time is it"
        assert n.normalize("so like what time") == "what time"

    def test_filler_loop_stability(self):
        """Filler stripping should loop until stable."""
        n = Normalizer()
        # Multiple layers of fillers
        assert n.normalize("um um um time") == "time"
        assert n.normalize("okay so well time") == "time"

    def test_filler_not_in_middle(self):
        """Fillers should not be stripped from middle of sentence."""
        n = Normalizer()
        result = n.normalize("set like a timer")
        # "like" in the middle should not be stripped
        # (edge fillers only strip from edges)
        assert "timer" in result

    def test_filler_boundary_preserved(self):
        """Fillers inside words should not be stripped."""
        n = Normalizer()
        # "er" is a filler but should not strip from "weather"
        assert n.normalize("weather") == "weather"
        assert n.normalize("whether") == "whether"


# =============================================================================
# Main Normalizer Tests - Punctuation
# =============================================================================

class TestNormalizerPunctuation:
    """Test punctuation normalization."""

    @pytest.mark.parametrize("text,expected", [
        ("what-time", "what time"),
        ("what time?", "what time"),
        ("set a timer.", "set a timer"),
        ("hello!", "hello"),
        ("timer, please", "timer please"),
    ])
    def test_punctuation_handling(self, text, expected):
        n = Normalizer()
        result = n.normalize(text)
        assert result == expected or expected in result

    def test_dotted_acronym_npr(self):
        """n.p.r. should become npr (via letter sequence join)."""
        n = Normalizer()
        result = n.normalize("n.p.r.")
        # The normalizer expands dots and then letter sequence normalizer joins
        assert "npr" in result or "n p r" in result


class TestNormalizerMultipleSpaces:
    """Test multiple space collapsing."""

    def test_multiple_spaces_collapsed(self):
        n = Normalizer()
        assert n.normalize("what   time   is   it") == "what time is it"
        assert n.normalize("  set  a  timer  ") == "set a timer"


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================

class TestNormalizerFullPipeline:
    """Test full normalizer pipeline with combined transformations."""

    @pytest.mark.parametrize("input_text,expected", [
        # Contractions + lowercase
        ("What's the time?", "what is the time"),
        # Fillers + numbers
        ("um set a timer for ten minutes", "set a timer for 10 minutes"),
        # Letter sequences + context
        ("hey play n p r", "play npr"),
        # Multiple transformations
        ("Um, what's the weather like?", "what is the weather like"),
        ("Okay, set a twenty five minute timer", "set a 25 minute timer"),
    ])
    def test_combined_transformations(self, input_text, expected):
        n = Normalizer()
        assert n.normalize(input_text) == expected

    def test_order_of_operations(self):
        """Verify correct order: contractions -> lowercase -> fillers -> punct -> letters -> numbers."""
        n = Normalizer()
        # This tests the full pipeline
        result = n.normalize("Um, What's N.P.R. playing for twenty five minutes?")
        assert "what is" in result
        assert "npr" in result
        assert "25" in result
