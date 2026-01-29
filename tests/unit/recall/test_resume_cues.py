"""
Tests for resume cue detection.
"""

import pytest

from episodic.recall.resume_cues import has_resume_cues, RESUME_CUE_PATTERNS


class TestHasResumeCues:
    """Tests for has_resume_cues function."""

    def test_back_to_pattern(self):
        """Test 'back to' resume cue."""
        assert has_resume_cues("Back to that Python thing") is True
        assert has_resume_cues("Let's go back to the API discussion") is True

    def test_continuing_pattern(self):
        """Test 'continuing' resume cue."""
        assert has_resume_cues("Continuing with the retry logic") is True

    def test_that_thing_pattern(self):
        """Test 'that X thing' resume cue."""
        assert has_resume_cues("What about that Python thing?") is True
        assert has_resume_cues("Remember that retry thing?") is True

    def test_where_were_we_pattern(self):
        """Test 'where were we' resume cue."""
        assert has_resume_cues("Where were we with the API?") is True

    def test_anyway_pattern(self):
        """Test 'anyway' resume cue."""
        assert has_resume_cues("Anyway, about Python") is True

    def test_resume_pattern(self):
        """Test 'resume' resume cue."""
        assert has_resume_cues("Let's resume our discussion") is True

    def test_picking_up_pattern(self):
        """Test 'picking up' resume cue."""
        assert has_resume_cues("Picking up where we left off") is True

    def test_anaphoric_with_question(self):
        """Test anaphoric reference + question pattern."""
        # "that" + question mark triggers detection
        assert has_resume_cues("What did you say about that?") is True
        assert has_resume_cues("Can we finish that?") is True

    def test_no_resume_cues(self):
        """Test that normal queries don't trigger false positives."""
        assert has_resume_cues("Tell me about Python retry patterns") is False
        assert has_resume_cues("How do I use tenacity?") is False
        assert has_resume_cues("What is a good retry strategy?") is False

    def test_case_insensitive(self):
        """Test that detection is case-insensitive."""
        assert has_resume_cues("BACK TO Python") is True
        assert has_resume_cues("back to python") is True
        assert has_resume_cues("Back To Python") is True

    def test_patterns_list_not_empty(self):
        """Sanity check that we have patterns defined."""
        assert len(RESUME_CUE_PATTERNS) > 0
