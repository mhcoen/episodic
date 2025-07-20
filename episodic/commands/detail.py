"""
Detail level command for controlling response information depth.

This command works in both regular chat and muse mode to control
how much detail is included in responses.
"""

from typing import Optional
import typer

from episodic.config import config
from episodic.configuration import get_text_color, get_llm_color


# Detail level definitions
DETAIL_LEVELS = {
    'minimal': {
        'name': 'Minimal',
        'description': 'Essential facts only',
        'color': 'bright_blue'
    },
    'moderate': {
        'name': 'Moderate',
        'description': 'Facts with relevant context',
        'color': 'bright_cyan'
    },
    'detailed': {
        'name': 'Detailed',
        'description': 'Facts, context, and explanations',
        'color': 'bright_green'
    },
    'maximum': {
        'name': 'Maximum',
        'description': 'All information, nuances, and edge cases',
        'color': 'bright_yellow'
    }
}


def detail(level: Optional[str] = None):
    """Set or display the current detail level for responses."""
    if level is None:
        # Display current detail level
        current_detail = config.get('response_detail', 'moderate')
        detail_info = DETAIL_LEVELS.get(current_detail, DETAIL_LEVELS['moderate'])
        
        typer.secho(f"\nCurrent detail level: ", nl=False, fg=get_text_color())
        typer.secho(f"{detail_info['name']}", fg=detail_info['color'], bold=True)
        typer.secho(f"{detail_info['description']}", fg=get_text_color())
        
        # Show all available levels
        typer.secho("\nAvailable detail levels:", fg=get_text_color())
        for key, info in DETAIL_LEVELS.items():
            prefix = "  • " if key == current_detail else "    "
            typer.secho(f"{prefix}{info['name'].lower()}", nl=False, fg=info['color'])
            typer.secho(f" - {info['description']}", fg=get_text_color())
    else:
        # Set new detail level
        level_lower = level.lower()
        if level_lower not in DETAIL_LEVELS:
            typer.secho(f"Invalid detail level: {level}", fg="red")
            typer.secho("Valid options: minimal, moderate, detailed, maximum", fg="red")
            return
        
        # Update both response_detail and muse_detail for consistency
        config.set('response_detail', level_lower)
        config.set('muse_detail', level_lower)  # Keep muse_detail in sync
        
        detail_info = DETAIL_LEVELS[level_lower]
        typer.secho(f"Detail level set to: ", nl=False, fg=get_text_color())
        typer.secho(f"{detail_info['name']}", fg=detail_info['color'], bold=True)
        typer.secho(f"{detail_info['description']}", fg=get_text_color())


def get_detail_prompt() -> str:
    """Get the current detail prompt to include in LLM requests."""
    current_detail = config.get('response_detail', 'moderate')
    
    # Load detail prompt from file
    from episodic.prompt_manager import get_prompt_manager
    prompt_manager = get_prompt_manager()
    
    detail_prompt = prompt_manager.get(f"detail/{current_detail}")
    if not detail_prompt:
        # Fallback if file not found
        detail_map = {
            'minimal': 'Include only essential facts without elaboration.',
            'moderate': 'Include facts with relevant context for understanding.',
            'detailed': 'Include facts, context, and clear explanations.',
            'maximum': 'Include all available information, nuances, and edge cases.'
        }
        detail_prompt = detail_map.get(current_detail, detail_map['moderate'])
    
    return detail_prompt.strip()