"""Cost display utilities for showing token usage and costs."""

import typer
from episodic.config import config
from episodic.configuration import get_text_color, format_cost


def display_response_cost(cost_info: dict, model: str = None) -> None:
    """
    Display cost information after a response.
    
    Args:
        cost_info: Dictionary with token counts and cost
        model: The model name (if not provided, gets from config)
    """
    if not cost_info:
        return
        
    # Get model if not provided
    if not model:
        model = config.get("model", "")
    
    # Format the display
    input_tokens = cost_info.get('input_tokens', 0)
    output_tokens = cost_info.get('output_tokens', 0)
    total_tokens = cost_info.get('total_tokens', 0)
    cost_usd = cost_info.get('cost_usd', 0.0)
    
    # Build the cost line
    parts = []
    if input_tokens or output_tokens:
        parts.append(f"Input: {input_tokens:,} tokens")
        parts.append(f"Output: {output_tokens:,} tokens")
        parts.append(f"{total_tokens:,} total")
    
    # Add cost or HF tier info
    if model.startswith("huggingface/"):
        parts.append("Free tier: ~30K/mo | Pro: $9/mo")
    elif cost_usd > 0:
        parts.append(f"Cost: {format_cost(cost_usd)}")
    else:
        parts.append("Cost: $0.00")
    
    # Display the line
    if parts:
        cost_line = " • ".join(parts)
        typer.secho(f"[{cost_line}]", fg=get_text_color(), dim=True)