"""
Settings and configuration commands for the Episodic CLI.

This module is a refactored version that delegates to specialized modules.
"""

import typer
from typing import Optional

from episodic.config import config
from episodic.configuration import get_text_color, get_system_color, get_heading_color
from episodic.conversation import conversation_manager
from episodic.param_mappings import normalize_param_name, get_display_name
from episodic.param_display import get_canonical_name, get_display_name as get_short_name, get_param_description

from .settings_display import display_all_settings
from .settings_handlers import (
    PARAM_HANDLERS, handle_special_params
)

# Global variables for settings
default_context_depth = 5
default_semdepth = 2


def set(param: Optional[str] = None, value: Optional[str] = None):
    """Configure various parameters."""
    global default_context_depth, default_semdepth

    # Initialize from config if needed
    default_context_depth = config.get('context_depth', 5)
    default_semdepth = config.get('semantic_depth', 2)

    # Handle 'all' keyword to show all parameters
    if param == "all":
        display_all_settings(default_context_depth, default_semdepth)
        return

    # If no param provided, show commonly changed settings
    if param is None:
        typer.secho("Commonly Changed Settings:", fg=get_heading_color(), bold=True)
        
        # Core settings with descriptions
        settings_to_show = [
            ("debug", config.get('debug', False)),
            ("cost", config.get('show_costs', False)), 
            ("topics", config.get('show_topics', False)),
            ("streaming", config.get('stream_responses', True)),
            ("depth", f"{default_context_depth} messages"),
            ("wrap", config.get('text_wrap', True)),
        ]
        
        for setting_name, value in settings_to_show:
            desc = get_param_description(setting_name)
            padding = ' ' * max(1, 20 - len(setting_name))
            typer.secho(f"  {setting_name}:{padding}{value}", fg=get_system_color(), nl=False)
            if desc:
                typer.secho(f" - {desc}", fg=get_text_color(), dim=True)
            else:
                typer.echo()
        
        # Mode settings
        muse_mode = config.get('muse_mode', False)
        desc = get_param_description("muse-mode")
        typer.secho(f"  muse-mode:        {muse_mode}", fg=get_system_color(), nl=False)
        if desc:
            typer.secho(f" - {desc}", fg=get_text_color(), dim=True)
        else:
            typer.echo()
            
        if muse_mode:
            muse_settings = [
                ("muse-style", config.get('muse_style', 'standard')),
                ("muse-detail", config.get('muse_detail', 'moderate')),
            ]
            for setting_name, value in muse_settings:
                desc = get_param_description(setting_name)
                padding = ' ' * max(1, 20 - len(setting_name))
                typer.secho(f"  {setting_name}:{padding}{value}", fg=get_system_color(), nl=False)
                if desc:
                    typer.secho(f" - {desc}", fg=get_text_color(), dim=True)
                else:
                    typer.echo()
        
        # Feature toggles
        feature_settings = [
            ("rag-enabled", config.get('rag_enabled', False)),
            ("web-enabled", config.get('web_search_enabled', False)),
        ]
        
        for setting_name, value in feature_settings:
            desc = get_param_description(setting_name)
            padding = ' ' * max(1, 20 - len(setting_name))
            typer.secho(f"  {setting_name}:{padding}{value}", fg=get_system_color(), nl=False)
            if desc:
                typer.secho(f" - {desc}", fg=get_text_color(), dim=True)
            else:
                typer.echo()
        
        typer.secho("\nUse '/set all' to see all settings", fg=get_text_color())
        typer.secho("Use '/set <param> <value>' to change a setting", fg=get_text_color())
        typer.secho("Both short names (debug) and long names (show-debug) work", fg=get_text_color(), dim=True)
        return

    # If value not provided, show current value
    if value is None:
        # First try alias resolution, then fallback to normal resolution
        canonical_param = get_canonical_name(param)
        normalized = normalize_param_name(canonical_param)
        if normalized:
            current_value = config.get(normalized, "Not set")
            display_name = get_short_name(normalized) or get_display_name(normalized)
            typer.secho(f"{display_name}: {current_value}", fg=get_system_color())
        else:
            typer.secho(f"Unknown parameter: {param}", fg="red")
        return

    # Normalize the parameter name (try alias first)
    original_param = param
    canonical_param = get_canonical_name(param)
    param = normalize_param_name(canonical_param)
    
    if not param:
        typer.secho(f"Unknown parameter: {original_param}", fg="red")
        typer.secho("Use '/config-docs' to see available parameters", fg=get_text_color())
        return

    # Handle special parameters first
    handled, new_depth, new_semdepth = handle_special_params(
        original_param, value, default_context_depth, default_semdepth
    )
    
    if handled:
        default_context_depth = new_depth
        default_semdepth = new_semdepth
        return

    # Check if we have a handler for this parameter
    if param in PARAM_HANDLERS:
        PARAM_HANDLERS[param](value)
    else:
        # Generic string parameter
        config.set(param, value)
        typer.secho(f"✅ Set {param} = {value}", fg=get_system_color())


def verify():
    """Verify and display current configuration."""
    typer.secho("Configuration Verification", fg=get_heading_color(), bold=True)
    typer.secho("=" * 50, fg=get_heading_color())
    
    # Model configuration
    typer.secho("\nModel Configuration:", fg=get_heading_color())
    model = config.get("model", "Not set")
    typer.secho(f"  Primary model: {model}", fg=get_system_color())
    
    compression_model = config.get("compression_model", "Not set")
    if compression_model == "Not set":
        compression_model = f"{model} (using primary)"
    typer.secho(f"  Compression model: {compression_model}", fg=get_system_color())
    
    topic_model = config.get("topic_detection_model", "ollama/llama3")
    typer.secho(f"  Topic detection model: {topic_model}", fg=get_system_color())
    
    # Core settings
    typer.secho("\nCore Settings:", fg=get_heading_color())
    context_depth = config.get("context_depth", default_context_depth)
    typer.secho(f"  Context depth: {context_depth} messages", fg=get_system_color())
    
    cache_enabled = config.get("use_context_cache", True)
    typer.secho(f"  Context caching: {'Enabled' if cache_enabled else 'Disabled'}", 
                fg=get_system_color())
    
    # Topic detection
    typer.secho("\nTopic Detection:", fg=get_heading_color())
    auto_topics = config.get("automatic_topic_detection", True)
    typer.secho(f"  Automatic detection: {'Enabled' if auto_topics else 'Disabled'}", 
                fg=get_system_color())
    
    if auto_topics:
        min_messages = config.get("min_messages_before_topic_change", 8)
        typer.secho(f"  Min messages before change: {min_messages}", fg=get_system_color())
        
        detection_method = "Standard LLM"
        if config.get("use_hybrid_topic_detection"):
            detection_method = "Hybrid (LLM + Drift)"
        elif config.get("use_sliding_window_detection"):
            detection_method = "Sliding Window"
        typer.secho(f"  Detection method: {detection_method}", fg=get_system_color())
    
    # Display settings
    typer.secho("\nDisplay Settings:", fg=get_heading_color())
    color_mode = config.get("color_mode", "full")
    typer.secho(f"  Color mode: {color_mode}", fg=get_system_color())
    
    wrap_enabled = config.get("text_wrap", True)
    wrap_width = config.get("wrap_width", 80)
    typer.secho(f"  Text wrapping: {'Enabled' if wrap_enabled else 'Disabled'} (width: {wrap_width})", 
                fg=get_system_color())
    
    # Features status
    typer.secho("\nFeatures Status:", fg=get_heading_color())
    features = [
        ("RAG (Knowledge Base)", config.get("rag_enabled", False)),
        ("Web Search", config.get("web_search_enabled", False)),
        ("Muse Mode", config.get("muse_mode", False)),
        ("Response Streaming", config.get("stream_responses", True)),
        ("Cost Tracking", config.get("show_costs", False)),
        ("Debug Mode", config.get("debug", False)),
        ("Benchmarking", config.get("benchmark", False))
    ]
    
    for feature, enabled in features:
        status = "Enabled" if enabled else "Disabled"
        color = "green" if enabled else "yellow"
        typer.secho(f"  {feature}: {status}", fg=color)
    
    typer.secho("\n" + "=" * 50, fg=get_heading_color())


def model_params(param_set: Optional[str] = None):
    """Display or set model-specific parameters."""
    if param_set is None:
        # Show all parameter sets
        typer.secho("Model Parameter Sets:", fg=get_heading_color(), bold=True)
        
        # Main conversation parameters
        typer.secho("\nMain conversation parameters:", fg=get_system_color())
        main_params = config.get("main_params", {})
        if main_params:
            for key, value in main_params.items():
                typer.secho(f"  {key}: {value}", fg=get_text_color())
        else:
            typer.secho("  (using defaults)", fg=get_text_color())
        
        # Topic detection parameters
        typer.secho("\nTopic detection parameters:", fg=get_system_color())
        topic_params = config.get("topic_params", {})
        if topic_params:
            for key, value in topic_params.items():
                typer.secho(f"  {key}: {value}", fg=get_text_color())
        else:
            typer.secho("  (using defaults)", fg=get_text_color())
            
        # Compression parameters
        typer.secho("\nCompression parameters:", fg=get_system_color())
        compression_params = config.get("compression_params", {})
        if compression_params:
            for key, value in compression_params.items():
                typer.secho(f"  {key}: {value}", fg=get_text_color())
        else:
            typer.secho("  (using defaults)", fg=get_text_color())
            
        typer.secho("\nUse '/model-params <set>' to configure specific parameter sets", 
                   fg=get_text_color())
        return
    
    # Configure specific parameter set
    valid_sets = ["main", "topic", "compression"]
    if param_set not in valid_sets:
        typer.secho(f"Invalid parameter set: {param_set}", fg="red")
        typer.secho(f"Valid sets: {', '.join(valid_sets)}", fg=get_text_color())
        return
        
    # Interactive configuration would go here
    typer.secho(f"Configuring {param_set} parameters...", fg=get_system_color())
    typer.secho("(Interactive configuration not yet implemented)", fg="yellow")


def config_docs():
    """Display configuration documentation."""
    typer.secho("Configuration Parameters", fg=get_heading_color(), bold=True)
    typer.secho("=" * 70, fg=get_heading_color())
    
    sections = {
        "Core Settings": [
            ("depth", "Number of conversation exchanges to include in context", "5"),
            ("semdepth", "Semantic analysis depth", "2"),
            ("cache", "Enable/disable prompt caching", "true"),
            ("debug", "Enable debug output", "false"),
            ("benchmark", "Enable performance benchmarking", "false"),
        ],
        "Display Settings": [
            ("color-mode", "Color output mode (full/basic/none)", "full"),
            ("wrap", "Enable text wrapping", "true"),
            ("wrap-width", "Text wrapping width in characters", "80"),
            ("show-costs", "Show token costs after responses", "false"),
            ("vi-mode", "Enable vi keybindings in prompt", "false"),
        ],
        "Topic Detection": [
            ("automatic-topic-detection", "Enable automatic topic detection", "true"),
            ("topic-detection-model", "Model for topic detection", "ollama/llama3"),
            ("min-messages-before-topic-change", "Messages required before topic change", "8"),
            ("show-topics", "Show topic evolution in responses", "false"),
        ],
        "Compression": [
            ("compression-model", "Model for compression", "gpt-3.5-turbo"),
            ("compression-method", "Compression algorithm (tiered/simple/extractive)", "tiered"),
            ("compression-length", "Maximum compressed text length", "2000"),
            ("auto-compress-topics", "Automatically compress closed topics", "true"),
        ],
        "Advanced Features": [
            ("use-sliding-window-detection", "Enable sliding window topic detection", "false"),
            ("use-hybrid-topic-detection", "Enable hybrid topic detection", "false"),
            ("drift-threshold", "Semantic drift threshold (0.0-1.0)", "0.9"),
            ("drift-embedding-model", "Model for semantic embeddings", "paraphrase-mpnet-base-v2"),
        ],
        "RAG Settings": [
            ("rag-enabled", "Enable RAG (Retrieval Augmented Generation)", "false"),
            ("rag-auto-search", "Automatically search knowledge base", "true"),
            ("rag-max-results", "Maximum RAG search results", "3"),
        ],
        "Web Search": [
            ("web-search-enabled", "Enable web search integration", "false"),
            ("web-search-provider", "Search provider (duckduckgo)", "duckduckgo"),
            ("muse-mode", "Enable muse mode (web-enhanced responses)", "false"),
        ]
    }
    
    for section, params in sections.items():
        typer.secho(f"\n{section}:", fg=get_heading_color())
        for param, desc, default in params:
            typer.secho(f"  {param:<35} {desc:<40} [{default}]", fg=get_text_color())
    
    typer.secho("\n" + "=" * 70, fg=get_heading_color())
    typer.secho("\nUsage: /set <parameter> <value>", fg=get_system_color())
    typer.secho("Example: /set debug true", fg=get_text_color())


def cost():
    """Display session cost information."""
    if not conversation_manager:
        typer.secho("No active conversation session", fg="yellow")
        return
        
    costs = conversation_manager.get_session_costs()
    
    typer.secho("\n💰 Session Costs", fg=get_heading_color(), bold=True)
    typer.secho("=" * 40, fg=get_heading_color())
    
    # Token counts
    typer.secho(f"Input tokens:  {costs['total_input_tokens']:>10,}", fg=get_system_color())
    typer.secho(f"Output tokens: {costs['total_output_tokens']:>10,}", fg=get_system_color())
    typer.secho(f"Total tokens:  {costs['total_tokens']:>10,}", fg=get_system_color())
    
    # Cost
    typer.secho("─" * 40, fg=get_text_color())
    typer.secho(f"Total cost:    ${costs['total_cost_usd']:>10.4f}", fg="green", bold=True)
    
    # API stats if available
    if 'api_stats' in costs and costs['api_stats']:
        typer.secho("\nAPI Usage by Operation:", fg=get_heading_color())
        for operation, stats in costs['api_stats'].items():
            if stats['count'] > 0:
                typer.secho(f"\n  {operation}:", fg=get_system_color())
                typer.secho(f"    Calls: {stats['count']}", fg=get_text_color())
                typer.secho(f"    Cost:  ${stats['total_cost']:.4f}", fg=get_text_color())
                if stats['operations']:
                    for op, op_stats in stats['operations'].items():
                        if op_stats['count'] > 0:
                            typer.secho(f"      - {op}: {op_stats['count']} calls, ${op_stats['total_cost']:.4f}", 
                                       fg=get_text_color())
    
    typer.secho("=" * 40, fg=get_heading_color())