"""
Parameter handlers for settings command.

This module contains logic for handling different parameter types.
"""

import typer
from typing import Optional

from episodic.config import config
from episodic.configuration import (
    get_system_color
)
from episodic.llm import enable_cache, disable_cache
from episodic.debug_system import debug_system
from episodic.constants import (
    WEB_SEARCH_PROVIDERS, COLOR_MODES, COMPRESSION_METHODS, PORCUPINE_KEYWORDS,
    TOPIC_GRANULARITY_LEVELS
)


def handle_boolean_param(param: str, value: str) -> bool:
    """Handle boolean parameter setting."""
    bool_val = value.lower() in ['true', '1', 'yes', 'on']
    config.set(param, bool_val)
    
    # Special handling for certain params
    if param == 'use_context_cache':
        if bool_val:
            enable_cache()
            typer.secho("✅ Context caching enabled", fg=get_system_color())
        else:
            disable_cache()
            typer.secho("✅ Context caching disabled", fg=get_system_color())
    else:
        typer.secho(f"✅ Set {param} = {bool_val}", fg=get_system_color())
    
    return True


def handle_integer_param(param: str, value: str, min_val: Optional[int] = None, 
                        max_val: Optional[int] = None) -> bool:
    """Handle integer parameter setting."""
    try:
        int_val = int(value)
        if min_val is not None and int_val < min_val:
            typer.secho(f"Value must be at least {min_val}", fg="red")
            return False
        if max_val is not None and int_val > max_val:
            typer.secho(f"Value must be at most {max_val}", fg="red")
            return False
            
        config.set(param, int_val)
        typer.secho(f"✅ Set {param} = {int_val}", fg=get_system_color())
        return True
    except ValueError:
        typer.secho(f"Invalid integer value: {value}", fg="red")
        return False


def handle_float_param(param: str, value: str, min_val: Optional[float] = None,
                      max_val: Optional[float] = None) -> bool:
    """Handle float parameter setting."""
    try:
        float_val = float(value)
        if min_val is not None and float_val < min_val:
            typer.secho(f"Value must be at least {min_val}", fg="red")
            return False
        if max_val is not None and float_val > max_val:
            typer.secho(f"Value must be at most {max_val}", fg="red")
            return False
            
        config.set(param, float_val)
        typer.secho(f"✅ Set {param} = {float_val}", fg=get_system_color())
        return True
    except ValueError:
        typer.secho(f"Invalid float value: {value}", fg="red")
        return False


def handle_string_param(param: str, value: str, valid_values: Optional[list] = None) -> bool:
    """Handle string parameter setting."""
    if valid_values and value not in valid_values:
        typer.secho(f"Invalid value. Must be one of: {', '.join(valid_values)}", fg="red")
        return False

    config.set(param, value)
    typer.secho(f"✅ Set {param} = {value}", fg=get_system_color())
    return True


def handle_rag_embedding_model(value: str) -> bool:
    """Handle RAG embedding model setting with restart warning."""
    old_value = config.get("rag_embedding_model", "all-MiniLM-L6-v2")
    if old_value == value:
        typer.secho(f"rag_embedding_model already set to {value}", fg=get_system_color())
        return True

    config.set("rag_embedding_model", value)
    typer.secho(f"✅ Set rag_embedding_model = {value}", fg=get_system_color())
    typer.secho("⚠️  Restart required for this change to take effect", fg="yellow")
    typer.secho("⚠️  Existing RAG documents will need to be re-indexed", fg="yellow")
    return True


def handle_topic_granularity(value: str) -> bool:
    """Handle topic granularity setting for neural segmentation."""
    if value not in TOPIC_GRANULARITY_LEVELS:
        typer.secho(f"Invalid granularity. Must be one of: {', '.join(TOPIC_GRANULARITY_LEVELS)}", fg="red")
        return False

    config.set("topic_granularity", value)

    # Clear strategy cache so next detection uses new granularity
    from episodic.topics.strategy_registry import reset_strategy
    reset_strategy()

    # Show what threshold this maps to
    from episodic.topics.calibration import GRANULARITY_LEVELS
    threshold = GRANULARITY_LEVELS.get(value, 0.5)

    typer.secho(f"✅ Set topic_granularity = {value}", fg=get_system_color())
    typer.secho(f"   Neural threshold: {threshold} (more boundaries at lower values)", fg=get_system_color(), dim=True)
    return True


def _clear_strategy_cache_after_set(param: str, value, msg: str) -> bool:
    """Set a config value and clear strategy cache (for strategy-affecting settings)."""
    config.set(param, value)
    from episodic.topics.strategy_registry import reset_strategy
    reset_strategy()
    typer.secho(msg, fg=get_system_color())
    return True


def handle_min_messages_before_topic_change(value: str) -> bool:
    """Handle min_messages_before_topic_change (commit gate, not detection gate)."""
    try:
        int_value = int(value)
        if int_value < 2 or int_value > 50:
            typer.secho("Value must be between 2 and 50", fg="red")
            return False
        return _clear_strategy_cache_after_set(
            'min_messages_before_topic_change', int_value,
            f"✅ Set min_messages_before_topic_change = {int_value}"
        )
    except ValueError:
        typer.secho("Value must be an integer", fg="red")
        return False


def handle_drift_suspect_threshold(value: str) -> bool:
    """Handle drift_suspect_threshold for hybrid topic detection."""
    try:
        float_value = float(value)
        if float_value < 0.0 or float_value > 1.0:
            typer.secho("Value must be between 0.0 and 1.0", fg="red")
            return False
        return _clear_strategy_cache_after_set(
            'drift_suspect_threshold', float_value,
            f"✅ Set drift_suspect_threshold = {float_value}"
        )
    except ValueError:
        typer.secho("Value must be a number", fg="red")
        return False


def handle_use_drift_trigger(value: str) -> bool:
    """Handle use_drift_trigger for hybrid topic detection."""
    bool_value = value.lower() in ('true', '1', 'yes', 'on')
    return _clear_strategy_cache_after_set(
        'use_drift_trigger', bool_value,
        f"✅ Set use_drift_trigger = {bool_value}"
    )


def handle_suspect_threshold(value: str) -> bool:
    """Handle suspect_threshold for neural SUSPECT entry.

    This threshold controls when high neural confidence triggers SUSPECT state.
    Setting to 1.0 disables neural-triggered SUSPECT (drift-only mode).
    Typical values: 0.7-0.9 for balanced detection.
    """
    try:
        float_value = float(value)
        if float_value < 0.0 or float_value > 1.0:
            typer.secho("Value must be between 0.0 and 1.0", fg="red")
            return False
        return _clear_strategy_cache_after_set(
            'suspect_threshold', float_value,
            f"✅ Set suspect_threshold = {float_value}"
        )
    except ValueError:
        typer.secho("Value must be a number", fg="red")
        return False


def handle_neural_commit_drift_threshold(value: str) -> bool:
    """Handle neural_commit_drift_threshold for neural commit drift gate.

    For neural-triggered SUSPECT, require drift >= this threshold to COMMIT.
    Prevents subtopic changes (like carbonara within pasta) from creating boundaries.
    Set to 'none' or 'null' to disable the gate.
    Typical values: 0.6-0.8 for balanced filtering.
    """
    if value.lower() in ('none', 'null'):
        return _clear_strategy_cache_after_set(
            'neural_commit_drift_threshold', None,
            "✅ Disabled neural commit drift gate"
        )
    try:
        float_value = float(value)
        if float_value < 0.0 or float_value > 1.0:
            typer.secho("Value must be between 0.0 and 1.0 (or 'none' to disable)", fg="red")
            return False
        return _clear_strategy_cache_after_set(
            'neural_commit_drift_threshold', float_value,
            f"✅ Set neural_commit_drift_threshold = {float_value}"
        )
    except ValueError:
        typer.secho("Value must be a number or 'none' to disable", fg="red")
        return False


def handle_topic_temperature(value: str) -> bool:
    """Handle topic temperature setting for neural calibration."""
    try:
        temp = float(value)
        if temp <= 0:
            typer.secho("Temperature must be positive", fg="red")
            return False
        if temp > 5.0:
            typer.secho("Temperature > 5.0 is extreme. Use with caution.", fg="yellow")

        config.set("topic_temperature", temp)
        typer.secho(f"✅ Set topic_temperature = {temp}", fg=get_system_color())

        if temp < 1.0:
            typer.secho("   T < 1.0: Sharper predictions (more confident)", fg=get_system_color(), dim=True)
        elif temp > 1.0:
            typer.secho("   T > 1.0: Softer predictions (more uncertain)", fg=get_system_color(), dim=True)
        else:
            typer.secho("   T = 1.0: Default (no scaling)", fg=get_system_color(), dim=True)

        return True
    except ValueError:
        typer.secho(f"Invalid temperature value: {value}", fg="red")
        return False


def handle_topic_strategy(value: str) -> bool:
    """Handle topic strategy setting."""
    from episodic.topics.strategy_registry import list_strategies

    # Get available strategies
    available = list(list_strategies().keys())

    if value not in available:
        typer.secho(f"Invalid strategy. Must be one of: {', '.join(available)}", fg="red")
        return False

    config.set("topic_strategy", value)

    # Reset cached strategy so next detection uses the new one
    from episodic.topics.strategy_registry import reset_strategy
    reset_strategy()

    typer.secho(f"✅ Set topic_strategy = {value}", fg=get_system_color())

    # Show description
    descriptions = list_strategies()
    if value in descriptions:
        typer.secho(f"   {descriptions[value]}", fg=get_system_color(), dim=True)

    return True


def handle_enable_topic_reactivation(value: str) -> bool:
    """Handle enable_topic_reactivation with eager centroid and embedding checks.

    When enabling topic reactivation, check if any topics lack centroids
    and compute them. Also check for missing embeddings which are required
    for the reactivation probe to work.
    """
    bool_val = value.lower() in ['true', '1', 'yes', 'on']
    config.set('enable_topic_reactivation', bool_val)

    if bool_val:
        typer.secho("✅ Set enable_topic_reactivation = True", fg=get_system_color())

        # Check for topics missing centroids and compute them
        try:
            from episodic.recall.centroid import backfill_centroids
            from episodic.db import get_recent_topics
            from episodic.db_connection import get_connection

            # Count topics needing centroids
            topics = get_recent_topics(limit=1000)
            with get_connection() as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM topics t
                    LEFT JOIN topic_centroids tc ON t.start_node_id = tc.start_node_id
                    WHERE tc.centroid_medoid_exchange_id IS NULL
                """)
                missing_count = cursor.fetchone()[0]

            if missing_count > 0:
                typer.secho(f"   Computing centroids for {missing_count} topic(s)...", fg=get_system_color(), dim=True)
                updated = backfill_centroids()
                typer.secho(f"   ✓ Computed {updated} centroid(s)", fg=get_system_color(), dim=True)
            else:
                typer.secho("   All topics have centroids", fg=get_system_color(), dim=True)

        except Exception as e:
            typer.secho(f"   ⚠️  Could not compute centroids: {e}", fg="yellow")

        # Check for missing embeddings (required for reactivation probe)
        # Use incremental backfill (O(new_nodes) instead of O(total_nodes))
        try:
            from episodic.maintenance.backfill_conversation_embeddings import (
                needs_incremental_backfill,
                backfill_embeddings_incremental
            )

            needs_it, estimated_count = needs_incremental_backfill()
            if needs_it:
                typer.secho(f"   Indexing ~{estimated_count} new conversation(s)...", fg=get_system_color(), dim=True)
                report = backfill_embeddings_incremental()
                if report.newly_indexed > 0:
                    typer.secho(f"   ✓ Indexed {report.newly_indexed} conversation(s)", fg=get_system_color(), dim=True)
                else:
                    typer.secho("   All conversations have embeddings", fg=get_system_color(), dim=True)
            else:
                typer.secho("   All conversations have embeddings", fg=get_system_color(), dim=True)

        except Exception as e:
            typer.secho(f"   ⚠️  Could not check/backfill embeddings: {e}", fg="yellow")
            typer.secho(f"   Reactivation may not work until embeddings are indexed", fg="yellow", dim=True)
    else:
        typer.secho("✅ Set enable_topic_reactivation = False", fg=get_system_color())

    return True


def handle_list_param(param: str, value: str, valid_values: Optional[list] = None) -> bool:
    """Handle list parameter setting (comma-separated values)."""
    # Parse comma-separated values
    if ',' in value:
        values = [v.strip() for v in value.split(',')]
    else:
        # Single value becomes a list
        values = [value.strip()]
    
    # Validate if allowed values specified
    if valid_values:
        invalid = [v for v in values if v not in valid_values]
        if invalid:
            typer.secho(f"Invalid values: {', '.join(invalid)}", fg="red")
            typer.secho(f"Allowed: {', '.join(valid_values)}", fg="red")
            return False
    
    config.set(param, values)

    # Special handling for web search providers - reset the global manager
    if param == 'web_search_providers':
        import episodic.web_search
        episodic.web_search._web_search_manager = None
        typer.secho(f"✅ Set {param} = {values}", fg=get_system_color())
        typer.secho("  Web search manager reset to use new providers", fg=get_system_color(), dim=True)
    else:
        typer.secho(f"✅ Set {param} = {values}", fg=get_system_color())

    return True


def handle_depth_param(value: str) -> int:
    """Handle depth parameter specifically."""
    try:
        depth = int(value)
        if depth < 1:
            typer.secho("Depth must be at least 1", fg="red")
            return None
        return depth
    except ValueError:
        typer.secho(f"Invalid depth value: {value}", fg="red")
        return None


def handle_semdepth_param(value: str) -> int:
    """Handle semdepth parameter specifically."""
    try:
        semdepth = int(value)
        if semdepth < 0:
            typer.secho("Semdepth must be non-negative", fg="red")
            return None
        return semdepth
    except ValueError:
        typer.secho(f"Invalid semdepth value: {value}", fg="red")
        return None


def handle_debug_param(value: str) -> bool:
    """Handle debug parameter with named category support."""
    from episodic.debug_system import debug_set
    
    # Use centralized debug_set function
    result = debug_set(value)
    
    # Check if it was an error message
    if result.startswith("Invalid debug categories:"):
        typer.secho(f"❌ {result}", fg="red")
        return False
    else:
        typer.secho(f"✅ {result}", fg=get_system_color())
        
        # Update config based on current state - only set legacy debug flag when 'all' is enabled
        enabled = debug_system.get_enabled()
        config.set('debug', 'all' in enabled)
        return True


def handle_special_params(param: str, value: str, context_depth: int, semdepth: int) -> tuple:
    """
    Handle special parameters that don't follow standard patterns.
    
    Returns: (handled, new_context_depth, new_semdepth)
    """
    if param == "depth":
        new_depth = handle_depth_param(value)
        if new_depth is not None:
            config.set('context_depth', new_depth)
            typer.secho(f"✅ Set context depth = {new_depth}", fg=get_system_color())
            return True, new_depth, semdepth
        return True, context_depth, semdepth
        
    elif param == "semdepth":
        new_semdepth = handle_semdepth_param(value)
        if new_semdepth is not None:
            config.set('semantic_depth', new_semdepth)
            typer.secho(f"✅ Set semantic depth = {new_semdepth}", fg=get_system_color())
            return True, context_depth, new_semdepth
        return True, context_depth, semdepth
        
    elif param == "cache":
        return handle_boolean_param('use_context_cache', value), context_depth, semdepth
        
    elif param == "debug":
        return handle_debug_param(value), context_depth, semdepth
        
    return False, context_depth, semdepth


# Parameter definitions for easy lookup
PARAM_HANDLERS = {
    # Boolean parameters
    # Note: 'debug' is handled in handle_special_params for category support
    'benchmark': lambda v: handle_boolean_param('benchmark', v),
    'benchmark_display': lambda v: handle_boolean_param('benchmark_display', v),
    'wrap': lambda v: handle_boolean_param('text_wrap', v),
    'text_wrap': lambda v: handle_boolean_param('text_wrap', v),
    'show_cost': lambda v: handle_boolean_param('show_cost', v),
    'topic_change_info': lambda v: handle_boolean_param('topic_change_info', v),
    'vi_mode': lambda v: handle_boolean_param('vi_mode', v),
    'automatic_topic_detection': lambda v: handle_boolean_param('automatic_topic_detection', v),
    'show_topics': lambda v: handle_boolean_param('show_topics', v),
    'show_drift': lambda v: handle_boolean_param('show_drift', v),
    'analyze_topic_boundaries': lambda v: handle_boolean_param('analyze_topic_boundaries', v),
    'auto_compress_topics': lambda v: handle_boolean_param('auto_compress_topics', v),
    'show_model_list': lambda v: handle_boolean_param('show_model_list', v),
    'use_sliding_window_detection': lambda v: handle_boolean_param('use_sliding_window_detection', v),
    'use_hybrid_topic_detection': lambda v: handle_boolean_param('use_hybrid_topic_detection', v),
    'stream_responses': lambda v: handle_boolean_param('stream_responses', v),
    'stream_constant_rate': lambda v: handle_boolean_param('stream_constant_rate', v),
    'stream_natural_rhythm': lambda v: handle_boolean_param('stream_natural_rhythm', v),
    'rag_enabled': lambda v: handle_boolean_param('rag_enabled', v),
    'rag_auto_search': lambda v: handle_boolean_param('rag_auto_search', v),
    'rag_show_citations': lambda v: handle_boolean_param('rag_show_citations', v),
    'web_search_enabled': lambda v: handle_boolean_param('web_search_enabled', v),
    'web_search_fallback_enabled': lambda v: handle_boolean_param('web_search_fallback_enabled', v),
    'muse_mode': lambda v: handle_boolean_param('muse_mode', v),
    
    # Integer parameters
    'wrap_width': lambda v: handle_integer_param('wrap_width', v, 40, 200),
    'compression_length': lambda v: handle_integer_param('compression_length', v, 100, 10000),
    'compression_queue_max_topics': lambda v: handle_integer_param('compression_queue_max_topics', v, 1, 100),
    'min_messages_before_topic_change': handle_min_messages_before_topic_change,
    'drift_suspect_threshold': handle_drift_suspect_threshold,
    'use_drift_trigger': handle_use_drift_trigger,
    'suspect_threshold': handle_suspect_threshold,
    'neural_commit_drift_threshold': handle_neural_commit_drift_threshold,
    'rag_max_results': lambda v: handle_integer_param('rag_max_results', v, 1, 10),
    'web_search_max_results': lambda v: handle_integer_param('web_search_max_results', v, 1, 20),
    'web_search_fallback_cache_minutes': lambda v: handle_integer_param('web_search_fallback_cache_minutes', v, 0, 60),
    
    # Float parameters
    'stream_rate': lambda v: handle_float_param('stream_rate', v, 1.0, 100.0),
    'drift_threshold': lambda v: handle_float_param('drift_threshold', v, 0.0, 1.0),
    'voice_tts_speed': lambda v: handle_float_param('voice_tts_speed', v, 0.5, 2.0),
    'voice_idle_timeout': lambda v: handle_integer_param('voice_idle_timeout', v, 0, 600),
    'voice_wake_word': lambda v: handle_string_param('voice_wake_word', v, PORCUPINE_KEYWORDS),
    'voice_wake_word_sensitivity': lambda v: handle_float_param('voice_wake_word_sensitivity', v, 0.0, 1.0),
    'voice_wake_word_enabled': lambda v: handle_boolean_param('voice_wake_word_enabled', v),
    'porcupine_access_key': lambda v: handle_string_param('porcupine_access_key', v),

    # String parameters with validation
    'color_mode': lambda v: handle_string_param('color_mode', v, COLOR_MODES),
    'compression_method': lambda v: handle_string_param('compression_method', v, COMPRESSION_METHODS),
    'drift_embedding_provider': lambda v: handle_string_param('drift_embedding_provider', v),
    'drift_embedding_model': lambda v: handle_string_param('drift_embedding_model', v),
    'web_search_providers': lambda v: handle_list_param('web_search_providers', v, WEB_SEARCH_PROVIDERS),
    # muse_detail removed - use /detail command instead to avoid duplication
    
    # Model parameters (special handling needed)
    'compression_model': lambda v: handle_string_param('compression_model', v),
    'topic_detection_model': lambda v: handle_string_param('topic_detection_model', v),

    # RAG embedding model (requires restart and re-indexing)
    'rag_embedding_model': handle_rag_embedding_model,

    # Topic segmentation settings
    'topic_granularity': handle_topic_granularity,
    'topic_temperature': handle_topic_temperature,
    'topic_strategy': handle_topic_strategy,

    # Topic reactivation (with eager centroid computation)
    'enable_topic_reactivation': handle_enable_topic_reactivation,

    # Reasoning control (for models that support it: GPT-5.2, Nemotron, Qwen3, etc.)
    'reasoning': lambda v: handle_boolean_param('reasoning_enabled', v),
    'reasoning_enabled': lambda v: handle_boolean_param('reasoning_enabled', v),
    'reasoning_effort': lambda v: handle_string_param('reasoning_effort', v, ['minimal', 'low', 'medium', 'high']),
    'reasoning_verbosity': lambda v: handle_string_param('reasoning_verbosity', v, ['low', 'medium', 'high']),
}
