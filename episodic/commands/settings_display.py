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
from episodic.param_display import get_display_name, get_param_description


def display_setting(setting: str, value: str, description: str):
    """Display a single setting with consistent formatting."""
    from episodic.text_formatter import unified_format_and_display
    
    # Format as bulleted list with key:value
    content = f"  • **{setting}**: {value} - {description}"
    unified_format_and_display(content, instant=True)


def display_section(title: str, settings: List[Tuple[str, str, str]]):
    """Display a section of settings."""
    from episodic.text_formatter import unified_format_and_display
    
    # Format section header and content together
    content = f"\n## {title}\n"
    for setting, value, desc in settings:
        content += f"• **{setting}**: {value} - {desc}\n"
    
    unified_format_and_display(content, instant=True)


def display_all_settings(context_depth: int, semdepth: int):
    """Display all configuration settings with short names and 4 model contexts."""
    from episodic.text_formatter import unified_format_and_display
    
    # Build complete settings content using markdown formatting
    content = """# 🤖 Models
Use '/model' to view and manage all 4 model contexts
• **/model**: Show current models
• **/model list**: Show available models
• **/model chat <name>**: Set conversation model

"""
    
    # Core settings with short names where available
    core_settings = [
        ("depth", str(context_depth), "Conversation history depth (messages)"),
        ("debug", str(config.get('debug', False)), get_param_description("debug") or "Debug output mode"),
        ("cost", str(config.get('show_costs', False)), get_param_description("cost") or "Show token costs"),
        ("streaming", str(config.get('stream_responses', True)), get_param_description("streaming") or "Response streaming"),
        ("topics", str(config.get('show_topics', False)), get_param_description("topics") or "Show topic transitions"),
        ("wrap", str(config.get('text_wrap', True)), get_param_description("wrap") or "Text wrapping"),
    ]
    
    content += "## 💬 Core Settings\n"
    for setting, value, desc in core_settings:
        content += f"• **{setting}**: {value} - {desc}\n"
    content += "\n"
    
    # Feature toggles with short names
    feature_settings = [
        ("muse-mode", str(config.get('muse_mode', False)), get_param_description("muse-mode") or "Web search synthesis mode"),
        ("rag-enabled", str(config.get('rag_enabled', False)), get_param_description("rag-enabled") or "Knowledge base search"),
        ("web-enabled", str(config.get('web_search_enabled', False)), get_param_description("web-enabled") or "Web search functionality"),
        ("auto-topics", str(config.get('automatic_topic_detection', True)), get_param_description("auto-topics") or "Automatic topic detection"),
    ]
    
    content += "## 🎛️ Feature Toggles\n"
    for setting, value, desc in feature_settings:
        content += f"• **{setting}**: {value} - {desc}\n"
    content += "\n"
    
    # Muse mode configuration (only show if enabled)
    if config.get('muse_mode', False):
        muse_settings = [
            ("muse-style", config.get('muse_style', 'standard'), get_param_description("muse-style") or "Response length"),
            ("muse-detail", config.get('muse_detail', 'moderate'), get_param_description("muse-detail") or "Detail level"),
            ("muse-format", config.get('muse_format', 'mixed'), "Output format style"),
            ("muse-sources", config.get('muse_sources', 'top-three'), "Source selection strategy"),
        ]
        content += "## 🎭 Muse Configuration\n"
        for setting, value, desc in muse_settings:
            content += f"• **{setting}**: {value} - {desc}\n"
        content += "\n"
    
    # Topic detection (detailed)
    topic_settings = [
        ("topic-threshold", str(config.get('min_messages_before_topic_change', 8)), get_param_description("topic-threshold") or "Messages before topic change"),
        ("sliding-window", str(config.get('use_sliding_window_detection', False)), "Window-based detection"),
        ("hybrid-topics", str(config.get('use_hybrid_topic_detection', False)), "Hybrid detection method"),
    ]
    
    content += "## 📑 Topic Detection\n"
    for setting, value, desc in topic_settings:
        content += f"• **{setting}**: {value} - {desc}\n"
    content += "\n"
    
    # Knowledge base (only show if enabled)  
    if config.get('rag_enabled', False):
        rag_settings = [
            ("rag-auto", str(config.get('rag_auto_search', True)), "Automatic knowledge search"),
            ("rag-results", str(config.get('rag_max_results', 3)), "Maximum search results"),
            ("rag-citations", str(config.get('rag_show_citations', True)), "Show source citations"),
        ]
        content += "## 📚 Knowledge Base\n"
        for setting, value, desc in rag_settings:
            content += f"• **{setting}**: {value} - {desc}\n"
        content += "\n"
    
    # Web search (only show if enabled)
    if config.get('web_search_enabled', False):
        web_settings = [
            ("web-provider", config.get('web_search_provider', 'duckduckgo'), "Search provider"),
            ("web-results", str(config.get('web_search_max_results', 5)), "Maximum search results"),
        ]
        content += "## 🌐 Web Search\n"
        for setting, value, desc in web_settings:
            content += f"• **{setting}**: {value} - {desc}\n"
        content += "\n"
    
    # Usage hints
    content += """## 💡 Quick Tips
• **/set debug true**: Enable debug output
• **/set muse-mode true**: Enable web search mode
• **/model**: View/manage all models
• **/mset chat.temperature**: Set model parameters
"""
    
    # Display using unified formatting
    unified_format_and_display(content, instant=True)