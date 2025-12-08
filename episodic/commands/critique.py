"""
Critique command - have another LLM critique the last response.
"""

import typer
from typing import Optional

from episodic.config import config
from episodic.configuration import get_system_color, get_heading_color
from episodic.color_utils import secho_color
from episodic.db import get_head, get_node
from episodic.llm import _execute_llm_query, get_model_string
from episodic.unified_streaming import unified_stream_response


CRITIQUE_SYSTEM_PROMPT = """You are a thoughtful critic reviewing an AI assistant's response. Analyze the response for:
- Factual accuracy and potential errors
- Logical gaps or questionable assumptions
- Missing nuances or oversimplifications
- Areas that could be improved or expanded

Be specific and constructive. If the response is solid, acknowledge what works well.
Keep your critique focused and concise."""


def critique_command(target: Optional[str] = None):
    """
    Critique the last assistant response using the critic model.

    Usage:
        /critique          # Critique the last assistant response
        /critique <id>     # Critique a specific response by short_id
    """
    # Get the target node
    if target:
        # Look up by short_id (get_node handles both id and short_id)
        node = get_node(target)
        if not node:
            typer.secho(f"Node '{target}' not found.", fg="red")
            return
    else:
        # Get the current head (last response)
        head_id = get_head()
        if not head_id:
            typer.secho("No conversation history to critique.", fg="red")
            return
        node = get_node(head_id)

    if not node:
        typer.secho("Could not find node to critique.", fg="red")
        return

    # Make sure it's an assistant response
    if node.get('role') != 'assistant':
        typer.secho("Can only critique assistant responses.", fg="yellow")
        typer.secho(f"Node '{node.get('short_id')}' is a {node.get('role')} message.", fg="yellow")
        return

    # Get the parent (user message that prompted this response)
    parent_id = node.get('parent_id')
    user_message = None
    if parent_id:
        parent_node = get_node(parent_id)
        if parent_node and parent_node.get('role') == 'user':
            user_message = parent_node.get('content', '')

    if not user_message:
        typer.secho("Could not find the user message that prompted this response.", fg="yellow")
        user_message = "[User message not available]"

    assistant_response = node.get('content', '')

    # Get the critic model
    critic_model = config.get('critic_model', 'anthropic/claude-opus-4-5-20251101')

    # Build the messages for the critique
    messages = [
        {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
        {"role": "user", "content": f"**User's question:**\n{user_message}\n\n**Assistant's response:**\n{assistant_response}"}
    ]

    # Display header
    model_display = get_model_string(critic_model)
    typer.echo()
    secho_color(f"🔍 Critique ({model_display}):", fg=get_heading_color(), bold=True)
    typer.echo()

    # Execute the critique with streaming
    # Use minimal parameters to avoid conflicts (e.g., Anthropic doesn't allow temp + top_p)
    # Use generous max_tokens for thorough critique (ignores chat max_tokens setting)
    try:
        stream_generator, _ = _execute_llm_query(
            messages=messages,
            model=critic_model,
            stream=True,
            temperature=0.7,
            top_p=None,
            max_tokens=1000
        )

        # Stream the response
        unified_stream_response(
            stream_generator=stream_generator,
            model=critic_model
        )

    except Exception as e:
        error_str = str(e).lower()
        if "not_found_error" in error_str or "model:" in error_str:
            typer.secho(f"Model not found: {critic_model}", fg="red")
            typer.secho("The model ID may have changed - check the provider's documentation.", fg="yellow")
            typer.secho("Use '/model critic <model>' to set a valid model.", fg="yellow")
        elif "api_key" in error_str or "authentication" in error_str:
            provider = critic_model.split("/")[0] if "/" in critic_model else "unknown"
            typer.secho(f"Authentication error for {provider}.", fg="red")
            typer.secho(f"Check your API key configuration.", fg="yellow")
        elif "invalid_request" in error_str or "cannot both be specified" in error_str:
            typer.secho(f"Invalid request to {critic_model}.", fg="red")
            typer.secho("Try a different model with '/model critic <model>'.", fg="yellow")
        else:
            typer.secho(f"Error getting critique: {e}", fg="red")
        return

    typer.echo()
