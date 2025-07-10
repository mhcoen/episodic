"""
Display functions for settings command.

This module handles displaying various configuration settings.
"""

import typer
from typing import List, Tuple

from episodic.config import config
from episodic.configuration import (
    DEFAULT_COLOR_MODE,
    get_text_color, get_system_color, get_heading_color
)


def display_setting(setting: str, value: str, description: str):
    """Display a single setting with consistent formatting."""
    padding = ' ' * max(1, 30 - len(f"{setting}: {value}") - 2)
    typer.secho(f"  {setting}: {value}{padding}", fg=get_system_color(), bold=True, nl=False)
    typer.secho(description, fg=get_text_color())


def display_section(title: str, settings: List[Tuple[str, str, str]]):
    """Display a section of settings."""
    typer.secho(f"\n{title}:", fg=get_heading_color())
    for setting, value, desc in settings:
        display_setting(setting, value, desc)


def display_all_settings(context_depth: int, semdepth: int):
    """Display all configuration settings."""
    typer.secho("All Settings:", fg=get_heading_color(), bold=True)
    
    # Core settings
    core_settings = [
        ("depth", str(context_depth), "Messages of conversation history"),
        ("semdepth", str(semdepth), "Semantic analysis depth"),
        ("cache", str(config.get('use_context_cache', True)), "Enable prompt caching"),
        ("debug", str(config.get('debug', False)), "Debug output mode"),
        ("benchmark", str(config.get('benchmark', False)), "Performance benchmarking"),
        ("benchmark-display", str(config.get('benchmark_display', False)), "Show benchmark results"),
    ]
    display_section("Core", core_settings)
    
    # Display settings
    display_settings = [
        ("color-mode", config.get('color_mode', DEFAULT_COLOR_MODE), "Color output (full/basic/none)"),
        ("wrap", str(config.get('text_wrap', True)), "Wrap long lines"),
        ("wrap-width", str(config.get('wrap_width', 80)), "Wrapping width in characters"),
        ("show-costs", str(config.get('show_costs', False)), "Show costs after each response"),
        ("show-topic-change-info", str(config.get('topic_change_info', True)), "Show topic change messages"),
        ("vi-mode", str(config.get('vi_mode', False)), "Vi keybindings in prompt"),
    ]
    display_section("Display", display_settings)
    
    # Compression settings
    compression_settings = [
        ("compression-model", config.get('compression_model', 'gpt-3.5-turbo'), "Model for compression"),
        ("compression-method", config.get('compression_method', 'tiered'), "Compression algorithm"),
        ("compression-length", str(config.get('compression_length', 2000)), "Max compressed text length"),
        ("auto-compress-topics", str(config.get('auto_compress_topics', True)), "Auto-compress closed topics"),
        ("compression-queue-max", str(config.get('compression_queue_max_topics', 10)), "Max queued compressions"),
    ]
    display_section("Compression", compression_settings)
    
    # Topic detection settings
    topic_settings = [
        ("automatic-topic-detection", str(config.get('automatic_topic_detection', True)), "Enable topic detection"),
        ("topic-detection-model", config.get('topic_detection_model', 'ollama/llama3'), "Model for detection"),
        ("min-messages-before-topic-change", str(config.get('min_messages_before_topic_change', 8)), "Messages before topic change"),
        ("show-topics", str(config.get('show_topics', False)), "Show topic evolution"),
        ("analyze-topic-boundaries", str(config.get('analyze_topic_boundaries', True)), "Use boundary analysis"),
    ]
    display_section("Topic Detection", topic_settings)
    
    # Advanced settings
    advanced_settings = [
        ("show-model-list", str(config.get('show_model_list', True)), "Show model selection list"),
        ("use-sliding-window-detection", str(config.get('use_sliding_window_detection', False)), "Window-based detection"),
        ("use-hybrid-topic-detection", str(config.get('use_hybrid_topic_detection', False)), "Hybrid detection"),
        ("drift-embedding-provider", config.get('drift_embedding_provider', 'sentence-transformers'), "Embedding provider"),
        ("drift-embedding-model", config.get('drift_embedding_model', 'paraphrase-mpnet-base-v2'), "Embedding model"),
    ]
    display_section("Advanced", advanced_settings)
    
    # LLM streaming settings
    streaming_settings = [
        ("stream-responses", str(config.get('stream_responses', True)), "Enable response streaming"),
        ("stream-rate", str(config.get('stream_rate', 15.0)), "Stream rate (chars/sec)"),
        ("stream-constant-rate", str(config.get('stream_constant_rate', False)), "Use constant rate"),
        ("stream-natural-rhythm", str(config.get('stream_natural_rhythm', False)), "Natural typing rhythm"),
    ]
    display_section("Streaming", streaming_settings)
    
    # RAG settings
    rag_settings = [
        ("rag-enabled", str(config.get('rag_enabled', False)), "Enable RAG system"),
        ("rag-auto-search", str(config.get('rag_auto_search', True)), "Auto-search on queries"),
        ("rag-max-results", str(config.get('rag_max_results', 3)), "Max search results"),
        ("rag-show-citations", str(config.get('rag_show_citations', True)), "Show source citations"),
    ]
    display_section("RAG (Knowledge Base)", rag_settings)
    
    # Web search settings
    web_settings = [
        ("web-search-enabled", str(config.get('web_search_enabled', False)), "Enable web search"),
        ("web-search-provider", config.get('web_search_provider', 'duckduckgo'), "Search provider"),
        ("web-search-max-results", str(config.get('web_search_max_results', 5)), "Max search results"),
        ("muse-mode", str(config.get('muse_mode', False)), "Muse mode (web synthesis)"),
    ]
    display_section("Web Search", web_settings)
    
    typer.echo()  # Add spacing after display