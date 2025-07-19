"""
Reflection and multi-step prompting command for Episodic.

This module provides the /reflect command that enables multi-step thinking
and reflection on complex problems.
"""

import typer
from typing import Optional, List, Dict, Any, Tuple
from episodic.config import config
from episodic.configuration import (
    get_heading_color, get_text_color, get_system_color,
    get_error_color, get_warning_color, get_success_color,
    get_llm_color
)
from episodic.llm import query_llm
from episodic.unified_streaming import unified_stream_response
from episodic.debug_utils import debug_print
import json


def reflection_command(query: Optional[str] = None, steps: int = 3):
    """
    Enable multi-step reflection on a problem.
    
    Usage:
        /reflect                           # Enable reflection mode for next message
        /reflect "problem to solve"        # Reflect on specific problem
        /reflect "problem" --steps 5       # Custom number of reflection steps
    
    This command enables the LLM to think through problems step-by-step,
    showing its reasoning process and potentially correcting itself.
    """
    if query is None:
        # Enable reflection mode for next message
        config.set("reflection_mode", True)
        config.set("reflection_steps", steps)
        typer.secho("\n🤔 Reflection mode enabled", fg=get_system_color())
        typer.secho(f"   The next message will use {steps}-step reflection", fg=get_text_color())
        typer.secho("   Use /reflect off to disable", fg=get_text_color(), dim=True)
        return
    
    # Handle turning off reflection mode
    if query.lower() == "off":
        config.set("reflection_mode", False)
        typer.secho("\n✅ Reflection mode disabled", fg=get_success_color())
        return
    
    # Perform reflection on the given query
    perform_reflection(query, steps)


def perform_reflection(query: str, num_steps: int = 3) -> str:
    """
    Perform multi-step reflection on a query.
    
    Args:
        query: The problem or question to reflect on
        num_steps: Number of reflection steps
        
    Returns:
        The final response after reflection
    """
    model = config.get('model', 'gpt-3.5-turbo')
    stream_enabled = config.get("stream_responses", True)
    
    typer.secho(f"\n🤔 Reflecting on: {query}", fg=get_heading_color(), bold=True)
    typer.secho(f"   Using {num_steps} reflection steps", fg=get_text_color(), dim=True)
    typer.echo()
    
    # Store all steps for context
    reflection_history = []
    
    for step in range(num_steps):
        typer.secho(f"\n📍 Step {step + 1}/{num_steps}:", fg=get_system_color(), bold=True)
        
        # Build the reflection prompt
        if step == 0:
            # Initial analysis
            prompt = f"""Please analyze this problem step by step:

Problem: {query}

Provide a thorough initial analysis, breaking down the problem into components and identifying key considerations."""
        elif step == num_steps - 1:
            # Final synthesis
            prompt = f"""Based on your previous analysis, provide a final, comprehensive answer to the original problem.

Original problem: {query}

Previous analysis:
{_format_reflection_history(reflection_history)}

Now synthesize your thoughts into a clear, actionable response."""
        else:
            # Middle reflection steps
            prompt = f"""Reflect on your previous analysis and consider:
1. What aspects might you have missed?
2. Are there any errors in your reasoning?
3. What additional perspectives should be considered?

Original problem: {query}

Your analysis so far:
{_format_reflection_history(reflection_history)}

Provide additional insights or corrections."""
        
        # Query the LLM
        if stream_enabled:
            try:
                # Get streaming response
                stream_tuple = query_llm(prompt, model=model, stream=True)
                stream_gen = stream_tuple[0] if isinstance(stream_tuple, tuple) else stream_tuple
                
                # Stream the response
                response = unified_stream_response(
                    stream_generator=stream_gen,
                    model=model,
                    color=get_llm_color()
                )
            except Exception as e:
                if config.get('debug', False):
                    debug_print(f"Streaming error: {e}")
                # Fallback to non-streaming
                result = query_llm(prompt, model=model, stream=False)
                response = result[0] if isinstance(result, tuple) else result
                typer.secho(response, fg=get_llm_color())
        else:
            # Non-streaming response
            result = query_llm(prompt, model=model, stream=False)
            response = result[0] if isinstance(result, tuple) else result
            typer.secho(response, fg=get_llm_color())
        
        # Store this step
        reflection_history.append({
            "step": step + 1,
            "type": _get_step_type(step, num_steps),
            "response": response
        })
        
        # Add a separator between steps (except after the last one)
        if step < num_steps - 1:
            typer.echo()
            typer.secho("─" * 50, fg=get_text_color(), dim=True)
    
    # Show completion message
    typer.echo()
    typer.secho("✅ Reflection complete", fg=get_success_color(), bold=True)
    
    # Return the final response
    return reflection_history[-1]["response"]


def _get_step_type(step: int, total_steps: int) -> str:
    """Get a descriptive type for the reflection step."""
    if step == 0:
        return "Initial Analysis"
    elif step == total_steps - 1:
        return "Final Synthesis"
    else:
        return f"Reflection {step}"


def _format_reflection_history(history: List[Dict[str, Any]]) -> str:
    """Format reflection history for inclusion in prompts."""
    formatted = []
    for item in history:
        formatted.append(f"Step {item['step']} ({item['type']}):\n{item['response']}")
    return "\n\n".join(formatted)


def handle_reflection_in_conversation(user_input: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Modify messages to include reflection instructions if reflection mode is enabled.
    
    This function is called from the conversation flow to inject reflection
    instructions when reflection mode is active.
    
    Args:
        user_input: The user's input
        messages: The conversation messages
        
    Returns:
        Modified messages with reflection instructions if enabled
    """
    if not config.get("reflection_mode", False):
        return messages
    
    # Reflection mode is enabled
    num_steps = config.get("reflection_steps", 3)
    
    # Disable reflection mode after use (one-shot)
    config.set("reflection_mode", False)
    
    # Find the user message and enhance it with reflection instructions
    reflection_prompt = f"""Please approach this problem using multi-step reflection:

1. First, analyze the problem thoroughly
2. Then, reflect on your analysis and identify any gaps or errors
3. Finally, synthesize your thoughts into a comprehensive response

Show your thinking process step by step.

User's request: {user_input}"""
    
    # Replace the last user message with the enhanced prompt
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] = reflection_prompt
    
    return messages