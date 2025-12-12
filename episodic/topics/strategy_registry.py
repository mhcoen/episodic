"""
Strategy registry for pluggable topic detection.

This module provides a factory for creating topic strategies by name,
enabling configuration-driven strategy selection.
"""

from typing import Dict, Any, Type, Optional
import logging

from episodic.topics.strategy import TopicStrategy, NullStrategy
from episodic.config import config

logger = logging.getLogger(__name__)

# Registry of available strategies
_STRATEGY_REGISTRY: Dict[str, Type[TopicStrategy]] = {
    'null': NullStrategy,
}


def register_strategy(name: str, strategy_class: Type[TopicStrategy]) -> None:
    """
    Register a strategy class under a name.

    Args:
        name: Name to register the strategy under (used in config)
        strategy_class: The strategy class to register
    """
    _STRATEGY_REGISTRY[name] = strategy_class
    logger.debug(f"Registered topic strategy: {name}")


def get_strategy(
    name: Optional[str] = None,
    strategy_params: Optional[Dict[str, Any]] = None
) -> TopicStrategy:
    """
    Get a strategy instance by name.

    Args:
        name: Strategy name (if None, uses config value)
        strategy_params: Strategy-specific parameters (if None, uses config value)

    Returns:
        Instantiated strategy

    Raises:
        ValueError: If strategy name is not registered
    """
    # Get from config if not provided
    if name is None:
        name = config.get('topic_strategy', 'default')

    if strategy_params is None:
        strategy_params = config.get('topic_strategy_params', {})

    # Lazy import strategies to avoid circular imports
    _ensure_strategies_registered()

    if name not in _STRATEGY_REGISTRY:
        available = ', '.join(_STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown topic strategy: '{name}'. "
            f"Available strategies: {available}"
        )

    strategy_class = _STRATEGY_REGISTRY[name]
    return strategy_class(strategy_params)


def list_strategies() -> Dict[str, str]:
    """
    List all registered strategies with descriptions.

    Returns:
        Dict mapping strategy names to their descriptions
    """
    _ensure_strategies_registered()

    result = {}
    for name, strategy_class in _STRATEGY_REGISTRY.items():
        # Get description from docstring or class name
        doc = strategy_class.__doc__
        if doc:
            # Get first line of docstring
            description = doc.strip().split('\n')[0]
        else:
            description = strategy_class.__name__
        result[name] = description

    return result


def _ensure_strategies_registered() -> None:
    """
    Ensure all built-in strategies are registered.

    Called lazily to avoid circular imports at module load time.
    """
    if 'dual_window' not in _STRATEGY_REGISTRY:
        try:
            from episodic.topics.strategies.dual_window_strategy import DualWindowStrategy
            register_strategy('dual_window', DualWindowStrategy)
        except ImportError as e:
            logger.warning(f"Could not import DualWindowStrategy: {e}")

    if 'keyword' not in _STRATEGY_REGISTRY:
        try:
            from episodic.topics.strategies.keyword_strategy import KeywordStrategy
            register_strategy('keyword', KeywordStrategy)
        except ImportError as e:
            logger.warning(f"Could not import KeywordStrategy: {e}")

    if 'relative_embedding' not in _STRATEGY_REGISTRY:
        try:
            from episodic.topics.strategies.relative_embedding_strategy import RelativeEmbeddingStrategy
            register_strategy('relative_embedding', RelativeEmbeddingStrategy)
        except ImportError as e:
            logger.warning(f"Could not import RelativeEmbeddingStrategy: {e}")

    if 'neural' not in _STRATEGY_REGISTRY:
        try:
            from episodic.topics.strategies.neural_strategy import NeuralStrategy
            register_strategy('neural', NeuralStrategy)
        except ImportError as e:
            logger.warning(f"Could not import NeuralStrategy: {e}")

    if 'ensemble' not in _STRATEGY_REGISTRY:
        try:
            from episodic.topics.strategies.ensemble_strategy import EnsembleStrategy
            register_strategy('ensemble', EnsembleStrategy)
        except ImportError as e:
            logger.warning(f"Could not import EnsembleStrategy: {e}")

    if 'cusum' not in _STRATEGY_REGISTRY:
        try:
            from episodic.topics.strategies.cusum_strategy import CUSUMStrategy
            register_strategy('cusum', CUSUMStrategy)
        except ImportError as e:
            logger.warning(f"Could not import CUSUMStrategy: {e}")

    if 'delta' not in _STRATEGY_REGISTRY:
        try:
            from episodic.topics.strategies.delta_strategy import DeltaStrategy
            register_strategy('delta', DeltaStrategy)
        except ImportError as e:
            logger.warning(f"Could not import DeltaStrategy: {e}")

    if 'speech_act' not in _STRATEGY_REGISTRY:
        try:
            from episodic.topics.strategies.speech_act_strategy import SpeechActStrategy
            register_strategy('speech_act', SpeechActStrategy)
        except ImportError as e:
            logger.warning(f"Could not import SpeechActStrategy: {e}")

    if 'time_aware' not in _STRATEGY_REGISTRY:
        try:
            from episodic.topics.strategies.time_aware_strategy import TimeAwareStrategy
            register_strategy('time_aware', TimeAwareStrategy)
        except ImportError as e:
            logger.warning(f"Could not import TimeAwareStrategy: {e}")

    if 'commitment' not in _STRATEGY_REGISTRY:
        try:
            from episodic.topics.strategies.commitment_strategy import CommitmentPolicyStrategy
            register_strategy('commitment', CommitmentPolicyStrategy)
        except ImportError as e:
            logger.warning(f"Could not import CommitmentPolicyStrategy: {e}")

    if 'summary_probe' not in _STRATEGY_REGISTRY:
        try:
            from episodic.topics.strategies.summary_probe_strategy import SummaryProbeStrategy
            register_strategy('summary_probe', SummaryProbeStrategy)
        except ImportError as e:
            logger.warning(f"Could not import SummaryProbeStrategy: {e}")

    if 'default' not in _STRATEGY_REGISTRY:
        try:
            from episodic.topics.strategies.default_strategy import DefaultStrategy
            register_strategy('default', DefaultStrategy)
        except ImportError as e:
            logger.warning(f"Could not import DefaultStrategy: {e}")


# Singleton instance for convenience
_current_strategy: Optional[TopicStrategy] = None


def get_current_strategy() -> TopicStrategy:
    """
    Get the currently configured strategy (cached).

    Returns:
        The current strategy instance
    """
    global _current_strategy

    if _current_strategy is None:
        _current_strategy = get_strategy()

    return _current_strategy


def reset_strategy() -> None:
    """
    Reset the cached strategy.

    Call this if configuration changes and you need a new strategy instance.
    """
    global _current_strategy
    _current_strategy = None
