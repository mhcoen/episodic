"""
Tests for recall/formatting.py

Tests block formatting with provenance labels.
"""

import pytest
from . import create_test_db, create_hits


class TestFormatBlocksProvenance:
    """Test that formatted blocks contain required provenance labels."""
    
    def test_format_blocks_contains_provenance_labels(self, tmp_path):
        """Output includes required headers for Conversation/Statement with topic name and match counts."""
        from episodic.recall.promotion import promote_hits_to_topics
        from episodic.recall.ranking import rank_topics, get_top_topics, get_top_statements
        from episodic.recall.expansion import expand_topic, Tier, ExpansionConfig
        from episodic.recall.budget import map_parser_output_to_budget
        from episodic.recall.formatting import format_recall_result
        
        conn, _ = create_test_db(tmp_path)
        
        # Create hits
        hits = create_hits(
            ['node_1', 'node_2', 'node_5'],
            [0.9, 0.85, 0.7]
        )
        
        promotion = promote_hits_to_topics(conn, hits)
        ranking = rank_topics(promotion)
        
        # Get top topics and expand
        top_topics = get_top_topics(ranking, 2)
        
        config = ExpansionConfig()
        expansions = [
            expand_topic(conn, t, Tier.B, config) for t in top_topics
        ]
        
        # Get statement hits
        statement_hits = get_top_statements(
            ranking, n=2, 
            exclude_topic_ids=[t.topic_id for t in top_topics]
        )
        
        # Build score map
        topic_scores = {t.topic_id: (t.best_hit, t.hit_count) for t in top_topics}
        
        budget = map_parser_output_to_budget("when_we", False, None)
        
        formatted = format_recall_result(
            conn, expansions, statement_hits, topic_scores, budget
        )
        
        # Convert to string
        output = formatted.to_context_string(budget)
        
        # Check conversation block provenance
        assert '[Conversation:' in output
        assert 'topic-alpha' in output or 'topic-beta' in output
        assert 'matches' in output
        
    def test_conversation_block_header_format(self, tmp_path):
        """Conversation block header has topic name, date, match count, score."""
        from episodic.recall.formatting import ConversationBlock, _format_conversation_block
        from episodic.recall.expansion import ExpandedExchange
        from episodic.recall.budget import map_parser_output_to_budget
        
        block = ConversationBlock(
            topic_id=1,
            topic_name="test-topic",
            date_range="Jan 15-16",
            hit_count=3,
            best_score=0.85,
            is_compressed=False,
            summary=None,
            exchanges=[
                ExpandedExchange(
                    node_id="node_1",
                    content="Test content",
                    role="user",
                    is_anchor=True,
                    position=0
                )
            ]
        )
        
        budget = map_parser_output_to_budget("when_we", False, None)
        output = _format_conversation_block(block, budget)
        
        assert '[Conversation: test-topic' in output
        assert 'Jan 15-16' in output  # Timestamps emphasized for when_we
        assert '3 matches' in output
        assert 'best=0.85' in output
    
    def test_statement_block_header_format(self, tmp_path):
        """Statement block header has topic name (if any), timestamp, score."""
        from episodic.recall.formatting import StatementBlock, _format_statement_block
        
        block = StatementBlock(
            exchange_id="node_5",
            topic_name="other-topic",
            timestamp="Jan 17",
            score=0.72,
            user_content="User question here",
            assistant_content="Assistant response here"
        )
        
        output = _format_statement_block(block)
        
        assert '[Statement' in output
        assert 'from other-topic' in output
        assert 'Jan 17' in output
        assert 'score=0.72' in output
        assert 'User:' in output
        assert 'Assistant:' in output
    
    def test_compressed_topic_includes_summary(self, tmp_path):
        """Compressed topic block includes summary text."""
        from episodic.recall.formatting import ConversationBlock, _format_conversation_block
        from episodic.recall.expansion import ExpandedExchange
        from episodic.recall.budget import map_parser_output_to_budget
        
        block = ConversationBlock(
            topic_id=1,
            topic_name="compressed-topic",
            date_range="Jan 10",
            hit_count=2,
            best_score=0.8,
            is_compressed=True,
            summary="This is the compressed summary of the topic.",
            exchanges=[
                ExpandedExchange(
                    node_id="node_1",
                    content="Anchor exchange",
                    role="user",
                    is_anchor=True,
                    position=0
                )
            ]
        )
        
        budget = map_parser_output_to_budget("what_we", False, None)
        output = _format_conversation_block(block, budget)
        
        assert 'Summary:' in output
        assert 'compressed summary of the topic' in output
        assert 'Relevant exchanges:' in output


class TestFormattedRecallContextString:
    """Test FormattedRecall.to_context_string() method."""
    
    def test_empty_result_returns_empty_string(self):
        """Empty result produces empty string."""
        from episodic.recall.formatting import FormattedRecall
        
        result = FormattedRecall(
            conversation_blocks=[],
            statement_blocks=[],
            total_exchanges=0
        )
        
        assert result.to_context_string() == ""
    
    def test_conversation_blocks_before_statements(self, tmp_path):
        """Conversation blocks appear before statement blocks."""
        from episodic.recall.formatting import (
            FormattedRecall, ConversationBlock, StatementBlock
        )
        from episodic.recall.expansion import ExpandedExchange
        
        conv_block = ConversationBlock(
            topic_id=1,
            topic_name="conv-topic",
            date_range="Jan 15",
            hit_count=1,
            best_score=0.9,
            is_compressed=False,
            summary=None,
            exchanges=[ExpandedExchange("n1", "content", "user", True, 0)]
        )
        
        stmt_block = StatementBlock(
            exchange_id="n5",
            topic_name=None,
            timestamp="Jan 16",
            score=0.7,
            user_content="stmt user content",
            assistant_content="stmt assistant content"
        )
        
        result = FormattedRecall(
            conversation_blocks=[conv_block],
            statement_blocks=[stmt_block],
            total_exchanges=2
        )
        
        output = result.to_context_string()
        
        conv_pos = output.find('[Conversation:')
        stmt_pos = output.find('[Statement')
        
        assert conv_pos < stmt_pos
