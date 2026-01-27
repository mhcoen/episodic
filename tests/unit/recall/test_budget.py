"""
Tests for recall/budget.py

Tests intent-to-budget mapping.
"""

import pytest
from episodic.recall.budget import (
    map_parser_output_to_budget,
    IntentClass,
    RecallBudget,
    get_budget_description
)
from episodic.recall.expansion import Tier


class TestBudgetMappingMatrix:
    """Test the complete mapping matrix from parser output to budget."""
    
    @pytest.mark.parametrize("query_form,broadness,speaker,expected_intent,expected_topics,expected_statements,expected_tier", [
        # when_we -> conversation_recall_locate
        ("when_we", False, None, IntentClass.CONVERSATION_RECALL_LOCATE, 2, 1, Tier.B),
        ("when_we", True, None, IntentClass.CONVERSATION_RECALL_LOCATE, 2, 1, Tier.B),  # broadness only affects horizon
        
        # what_we -> conversation_recall_summarize
        ("what_we", False, None, IntentClass.CONVERSATION_RECALL_SUMMARIZE, 2, 1, Tier.B),
        ("what_we", True, None, IntentClass.CONVERSATION_RECALL_SUMMARIZE, 2, 1, Tier.B),
        
        # have_we -> existence_check (always, regardless of broadness)
        ("have_we", False, None, IntentClass.EXISTENCE_CHECK, 1, 2, Tier.A),
        ("have_we", True, None, IntentClass.EXISTENCE_CHECK, 1, 2, Tier.A),
        
        # did_speaker -> statement_recall
        ("did_speaker", False, None, IntentClass.STATEMENT_RECALL, 1, 3, Tier.A),
        ("did_speaker", True, None, IntentClass.STATEMENT_RECALL, 1, 3, Tier.A),
        ("did_speaker", False, "user", IntentClass.STATEMENT_RECALL, 0, 4, Tier.A),  # Speaker modifier
        ("did_speaker", False, "assistant", IntentClass.STATEMENT_RECALL, 0, 4, Tier.A),
    ])
    def test_budget_mapping_matrix(self, query_form, broadness, speaker, 
                                   expected_intent, expected_topics, expected_statements, expected_tier):
        """Parametrized test for intent-to-budget mapping."""
        budget = map_parser_output_to_budget(
            query_form=query_form,
            has_broadness_cue=broadness,
            speaker=speaker,
            mode=None
        )
        
        assert budget.intent == expected_intent
        assert budget.max_topics == expected_topics
        assert budget.max_statements == expected_statements
        assert budget.topic_tier == expected_tier
    
    def test_browse_mode_mapping(self):
        """Explicit browse mode maps correctly."""
        budget = map_parser_output_to_budget(
            query_form=None,
            has_broadness_cue=False,
            speaker=None,
            mode="browse"
        )
        
        assert budget.intent == IntentClass.BROWSE
        assert budget.max_topics == 3
        assert budget.max_statements == 0


class TestBroadnessModifier:
    """Test that broadness cue affects horizon, not intent."""
    
    def test_broadness_enables_broad_horizon(self):
        """has_broadness_cue=True sets broad_horizon=True."""
        budget_narrow = map_parser_output_to_budget("when_we", False, None)
        budget_broad = map_parser_output_to_budget("when_we", True, None)
        
        assert budget_narrow.broad_horizon is False
        assert budget_broad.broad_horizon is True
    
    def test_broadness_increases_overfetch(self):
        """has_broadness_cue=True increases overfetch multiplier."""
        budget_narrow = map_parser_output_to_budget("when_we", False, None)
        budget_broad = map_parser_output_to_budget("when_we", True, None)
        
        assert budget_broad.overfetch_multiplier > budget_narrow.overfetch_multiplier
    
    def test_broadness_does_not_change_intent(self):
        """Broadness cue does not change the intent class."""
        for query_form in ["when_we", "what_we", "have_we", "did_speaker"]:
            budget_narrow = map_parser_output_to_budget(query_form, False, None)
            budget_broad = map_parser_output_to_budget(query_form, True, None)
            
            assert budget_narrow.intent == budget_broad.intent


class TestSpeakerModifier:
    """Test that speaker filter modifies budget appropriately."""
    
    def test_speaker_reduces_topic_budget(self):
        """Speaker filter reduces topic budget (lexical-only clusters less well)."""
        budget_both = map_parser_output_to_budget("when_we", False, None)
        budget_user = map_parser_output_to_budget("when_we", False, "user")
        
        assert budget_user.max_topics < budget_both.max_topics
    
    def test_speaker_increases_statement_budget(self):
        """Speaker filter increases statement budget."""
        budget_both = map_parser_output_to_budget("when_we", False, None)
        budget_user = map_parser_output_to_budget("when_we", False, "user")
        
        assert budget_user.max_statements > budget_both.max_statements
    
    def test_speaker_downgrades_tier(self):
        """Speaker filter downgrades tier from B to A."""
        budget_both = map_parser_output_to_budget("when_we", False, None)
        budget_user = map_parser_output_to_budget("when_we", False, "user")
        
        # when_we normally gets Tier.B
        assert budget_both.topic_tier == Tier.B
        # With speaker filter, downgrade to Tier.A
        assert budget_user.topic_tier == Tier.A


class TestEmphasisFlags:
    """Test emphasis flags in budget."""
    
    def test_locate_intent_emphasizes_timestamps(self):
        """CONVERSATION_RECALL_LOCATE emphasizes timestamps."""
        budget = map_parser_output_to_budget("when_we", False, None)
        
        assert budget.emphasize_timestamps is True
        assert budget.emphasize_summary is False
    
    def test_summarize_intent_emphasizes_summary(self):
        """CONVERSATION_RECALL_SUMMARIZE emphasizes summary."""
        budget = map_parser_output_to_budget("what_we", False, None)
        
        assert budget.emphasize_timestamps is False
        assert budget.emphasize_summary is True


class TestBudgetDescription:
    """Test human-readable budget description."""
    
    def test_budget_description_includes_all_fields(self):
        """Budget description includes intent, topics, tier, horizon."""
        budget = map_parser_output_to_budget("when_we", True, None)
        desc = get_budget_description(budget)
        
        assert "CONVERSATION_RECALL_LOCATE" in desc
        assert "Topics:" in desc
        assert "Tier" in desc
        assert "broad" in desc  # Because broadness_cue=True
