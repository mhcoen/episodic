"""
Response style commands for Episodic.

This module handles the global response style setting that applies to all modes
(chat, RAG-enhanced responses, and muse synthesis).
"""

import typer
from episodic.config import config
from episodic.configuration import get_system_color, get_text_color
from episodic.color_utils import secho_color
from episodic.prompt_manager import get_prompt_manager


# Style definitions with token limits and descriptions
STYLE_DEFINITIONS = {
    "concise": {
        "description": "Brief, direct responses (1-2 sentences when possible)",
        "max_tokens": 500
    },
    "standard": {
        "description": "Clear, well-structured responses with appropriate detail",
        "max_tokens": 1000
    },
    "comprehensive": {
        "description": "Thorough, detailed responses with examples and context",
        "max_tokens": 2000
    },
    "custom": {
        "description": "Use model-specific max_tokens settings for fine control",
        "max_tokens": None
    }
}

# Format definitions with descriptions
FORMAT_DEFINITIONS = {
    "paragraph": {
        "description": "Flowing prose in paragraph form with markdown headers"
    },
    "bulleted": {
        "description": "Bullet points and lists for all information"
    },
    "mixed": {
        "description": "Mix of paragraphs and bullet points as appropriate"
    },
    "academic": {
        "description": "Formal academic style with proper citations"
    }
}


def style_command(style: str = None):
    """Set or show the global response style for all modes."""
    
    if style is None:
        # Show current style
        current_style = config.get("response_style", "standard")
        secho_color(f"Current response style: ", fg=get_system_color(), bold=True, nl=False)
        secho_color(f"{current_style}", fg="bright_green", bold=True)
        
        if current_style in STYLE_DEFINITIONS:
            desc = STYLE_DEFINITIONS[current_style]["description"]
            secho_color(f"  {desc}", fg=get_text_color())
            
            max_tokens = STYLE_DEFINITIONS[current_style]["max_tokens"]
            if max_tokens:
                secho_color(f"  Max tokens: {max_tokens}", fg=get_text_color(), dim=True)
            else:
                secho_color(f"  Uses model-specific token limits", fg=get_text_color(), dim=True)
        
        typer.echo("")
        secho_color("Available styles:", fg=get_system_color(), bold=True)
        for style_name, info in STYLE_DEFINITIONS.items():
            marker = "→" if style_name == current_style else " "
            is_current = style_name == current_style
            secho_color(f"{marker} {style_name}: {info['description']}", fg=get_text_color(), bold=is_current)
        
        typer.echo("")
        secho_color("Usage: /style [concise|standard|comprehensive|custom]", fg=get_text_color(), dim=True)
        return
    
    # Validate and set style
    style = style.lower()
    if style not in STYLE_DEFINITIONS:
        available = ", ".join(STYLE_DEFINITIONS.keys())
        secho_color(f"Invalid style: {style}", fg="red")
        secho_color(f"Available styles: {available}", fg=get_text_color())
        return
    
    # Set the new style
    config.set("response_style", style)
    info = STYLE_DEFINITIONS[style]
    
    secho_color("✓ Response style set to ", fg="green", bold=True, nl=False)
    secho_color(f"{style}", fg="bright_green", bold=True)
    secho_color(f"  {info['description']}", fg=get_text_color())
    
    if info["max_tokens"]:
        secho_color(f"  Max tokens: {info['max_tokens']}", fg=get_text_color(), dim=True)
    else:
        secho_color(f"  Uses model-specific token limits", fg=get_text_color(), dim=True)
    
    secho_color("This style applies to all modes: chat, RAG-enhanced, and muse synthesis", fg=get_text_color(), dim=True)


def get_style_prompt(has_rag=False, rag_length=0, has_web=False) -> str:
    """
    Get the combined style and format prompt for the current settings, adapted to context.
    
    Args:
        has_rag: Whether RAG context is available
        rag_length: Length of RAG context in characters
        has_web: Whether web search context is available
        
    Returns:
        Prompt string to prepend to the user query
    """
    current_style = config.get("response_style", "standard")
    
    if current_style not in STYLE_DEFINITIONS:
        current_style = "standard"
    
    # Load style prompt from file
    prompt_manager = get_prompt_manager()
    style_prompt = prompt_manager.get(f"style/{current_style}")
    if not style_prompt:
        # Fallback if file not found
        style_prompt = "Provide a clear, natural response with appropriate detail."
    
    format_prompt = get_format_prompt()
    
    # Combine style and format instructions
    combined_prompt = f"{style_prompt.strip()} {format_prompt}"
    
    # Adapt prompt based on available context
    if has_rag and rag_length < 200:  # Small RAG context
        return f"{combined_prompt} Base your response primarily on the provided context, expanding only where directly relevant."
    elif has_rag:  # Substantial RAG context
        return f"{combined_prompt} Use the provided context as your primary source, supplementing with your knowledge as appropriate."
    elif has_web:  # Web search context
        return f"{combined_prompt} Synthesize information from the provided web search results."
    else:
        return combined_prompt


def get_style_max_tokens() -> int:
    """Get the max_tokens value for the current style, or None for custom."""
    current_style = config.get("response_style", "standard")
    
    if current_style not in STYLE_DEFINITIONS:
        current_style = "standard"
    
    return STYLE_DEFINITIONS[current_style]["max_tokens"]


def format_command(format_type: str = None):
    """Set or show the global response format for all modes."""
    
    if format_type is None:
        # Show current format
        current_format = config.get("response_format", "mixed")
        secho_color(f"Current response format: ", fg=get_system_color(), bold=True, nl=False)
        secho_color(f"{current_format}", fg="bright_green", bold=True)
        
        if current_format in FORMAT_DEFINITIONS:
            desc = FORMAT_DEFINITIONS[current_format]["description"]
            secho_color(f"  {desc}", fg=get_text_color())
        
        typer.echo("")
        secho_color("Available formats:", fg=get_system_color(), bold=True)
        for format_name, info in FORMAT_DEFINITIONS.items():
            marker = "→" if format_name == current_format else " "
            is_current = format_name == current_format
            secho_color(f"{marker} {format_name}: {info['description']}", fg=get_text_color(), bold=is_current)
        
        typer.echo("")
        secho_color("Usage: /format [paragraph|bulleted|mixed|academic]", fg=get_text_color(), dim=True)
        return
    
    # Validate and set format
    format_type = format_type.lower()
    if format_type not in FORMAT_DEFINITIONS:
        available = ", ".join(FORMAT_DEFINITIONS.keys())
        secho_color(f"Invalid format: {format_type}", fg="red")
        secho_color(f"Available formats: {available}", fg=get_text_color())
        return
    
    # Set the new format
    config.set("response_format", format_type)
    info = FORMAT_DEFINITIONS[format_type]
    
    secho_color("✓ Response format set to ", fg="green", bold=True, nl=False)
    secho_color(f"{format_type}", fg="bright_green", bold=True)
    secho_color(f"  {info['description']}", fg=get_text_color())
    
    secho_color("This format applies to all modes: chat, RAG-enhanced, and muse synthesis", fg=get_text_color(), dim=True)


def get_format_prompt() -> str:
    """Get the format prompt for the current response format."""
    current_format = config.get("response_format", "mixed")
    
    if current_format not in FORMAT_DEFINITIONS:
        current_format = "mixed"
    
    # Load format prompt from file
    prompt_manager = get_prompt_manager()
    format_prompt = prompt_manager.get(f"format/{current_format}")
    if not format_prompt:
        # Fallback if file not found
        format_prompt = "Use a natural mix of paragraphs and bullet points as appropriate."
    
    return format_prompt.strip()


def handle_style(args: list):
    """Handle /style command."""
    if not args:
        style_command()
    else:
        style_command(args[0])


def handle_format(args: list):
    """Handle /format command."""
    if not args:
        format_command()
    else:
        format_command(args[0])