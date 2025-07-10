"""
Parameter handlers for settings command.

This module contains logic for handling different parameter types.
"""

import typer
from typing import Optional, Any

from episodic.config import config
from episodic.configuration import (
    DEFAULT_COLOR_MODE,
    get_text_color, get_system_color
)
from episodic.llm import enable_cache, disable_cache
from episodic.param_mappings import normalize_param_name


def handle_boolean_param(param: str, value: str) -> bool:
    """Handle boolean parameter setting."""
    bool_val = value.lower() in ['true', '1', 'yes', 'on']
    config[param] = bool_val
    
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
            
        config[param] = int_val
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
            
        config[param] = float_val
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
        
    config[param] = value
    typer.secho(f"✅ Set {param} = {value}", fg=get_system_color())
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


def handle_special_params(param: str, value: str, context_depth: int, semdepth: int) -> tuple:
    """
    Handle special parameters that don't follow standard patterns.
    
    Returns: (handled, new_context_depth, new_semdepth)
    """
    if param == "depth":
        new_depth = handle_depth_param(value)
        if new_depth is not None:
            config['context_depth'] = new_depth
            typer.secho(f"✅ Set context depth = {new_depth}", fg=get_system_color())
            return True, new_depth, semdepth
        return True, context_depth, semdepth
        
    elif param == "semdepth":
        new_semdepth = handle_semdepth_param(value)
        if new_semdepth is not None:
            config['semantic_depth'] = new_semdepth
            typer.secho(f"✅ Set semantic depth = {new_semdepth}", fg=get_system_color())
            return True, context_depth, new_semdepth
        return True, context_depth, semdepth
        
    elif param == "cache":
        return handle_boolean_param('use_context_cache', value), context_depth, semdepth
        
    return False, context_depth, semdepth


# Parameter definitions for easy lookup
PARAM_HANDLERS = {
    # Boolean parameters
    'debug': lambda v: handle_boolean_param('debug', v),
    'benchmark': lambda v: handle_boolean_param('benchmark', v),
    'benchmark_display': lambda v: handle_boolean_param('benchmark_display', v),
    'wrap': lambda v: handle_boolean_param('text_wrap', v),
    'text_wrap': lambda v: handle_boolean_param('text_wrap', v),
    'show_costs': lambda v: handle_boolean_param('show_costs', v),
    'topic_change_info': lambda v: handle_boolean_param('topic_change_info', v),
    'vi_mode': lambda v: handle_boolean_param('vi_mode', v),
    'automatic_topic_detection': lambda v: handle_boolean_param('automatic_topic_detection', v),
    'show_topics': lambda v: handle_boolean_param('show_topics', v),
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
    'muse_mode': lambda v: handle_boolean_param('muse_mode', v),
    
    # Integer parameters
    'wrap_width': lambda v: handle_integer_param('wrap_width', v, 40, 200),
    'compression_length': lambda v: handle_integer_param('compression_length', v, 100, 10000),
    'compression_queue_max_topics': lambda v: handle_integer_param('compression_queue_max_topics', v, 1, 100),
    'min_messages_before_topic_change': lambda v: handle_integer_param('min_messages_before_topic_change', v, 2, 50),
    'rag_max_results': lambda v: handle_integer_param('rag_max_results', v, 1, 10),
    'web_search_max_results': lambda v: handle_integer_param('web_search_max_results', v, 1, 20),
    
    # Float parameters
    'stream_rate': lambda v: handle_float_param('stream_rate', v, 1.0, 100.0),
    'drift_threshold': lambda v: handle_float_param('drift_threshold', v, 0.0, 1.0),
    
    # String parameters with validation
    'color_mode': lambda v: handle_string_param('color_mode', v, ['full', 'basic', 'none']),
    'compression_method': lambda v: handle_string_param('compression_method', v, 
                                                      ['tiered', 'simple', 'extractive']),
    'drift_embedding_provider': lambda v: handle_string_param('drift_embedding_provider', v),
    'drift_embedding_model': lambda v: handle_string_param('drift_embedding_model', v),
    'web_search_provider': lambda v: handle_string_param('web_search_provider', v, ['duckduckgo']),
    
    # Model parameters (special handling needed)
    'compression_model': lambda v: handle_string_param('compression_model', v),
    'topic_detection_model': lambda v: handle_string_param('topic_detection_model', v),
}