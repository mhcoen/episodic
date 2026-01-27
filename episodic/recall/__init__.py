"""
Recall module for Episodic.

Implements conversation recall: exchange hits → topic promotion → ranking → expansion → formatting.

Includes ambiguity detection: when retrieval candidates form multiple competitive clusters,
returns an AMBIGUOUS result for the caller to handle disambiguation.

Main entry point: recall(conn, query, query_form) -> RecallResult
"""

from .pipeline import recall, RecallResult, RecallResultKind, SemanticHit
from .promotion import promote_hits_to_topics, PromotedHit, PromotionResult
from .ranking import rank_topics, RankedTopic, RankingResult, get_top_topics, get_top_statements
from .expansion import expand_topic, Tier, TopicExpansion, ExpandedExchange, ExpansionConfig, DEFAULT_CONFIG
from .budget import (
    map_parser_output_to_budget,
    RecallBudget,
    IntentClass,
    get_budget_description
)
from .formatting import (
    format_recall_result,
    FormattedRecall,
    ConversationBlock,
    StatementBlock
)
from .cli_integration import (
    handle_recall_query,
    get_recall_context_for_llm
)
from .ambiguity import (
    AmbiguityConfig,
    AmbiguityResult,
    ClusterOption,
    ambiguity_detect,
    format_disambiguation_prompt,
)

__all__ = [
    # Main pipeline
    'recall',
    'RecallResult',
    'RecallResultKind',
    'SemanticHit',

    # Ambiguity detection
    'AmbiguityConfig',
    'AmbiguityResult',
    'ClusterOption',
    'ambiguity_detect',
    'format_disambiguation_prompt',

    # Promotion
    'promote_hits_to_topics',
    'PromotedHit',
    'PromotionResult',

    # Ranking
    'rank_topics',
    'RankedTopic',
    'RankingResult',
    'get_top_topics',
    'get_top_statements',

    # Expansion
    'expand_topic',
    'Tier',
    'TopicExpansion',
    'ExpandedExchange',
    'ExpansionConfig',
    'DEFAULT_CONFIG',

    # Budget
    'map_parser_output_to_budget',
    'RecallBudget',
    'IntentClass',
    'get_budget_description',

    # Formatting
    'format_recall_result',
    'FormattedRecall',
    'ConversationBlock',
    'StatementBlock',

    # CLI Integration
    'handle_recall_query',
    'get_recall_context_for_llm',
]
