"""
Summary command for the Episodic CLI.
"""

import typer
from typing import Optional
from episodic.db import get_recent_nodes
from episodic.llm import query_llm
from episodic.config import config
from episodic.configuration import (
    get_heading_color, get_text_color, get_system_color, get_llm_color
)
from episodic.benchmark import benchmark_operation
from episodic.conversation import conversation_manager, wrapped_llm_print
from episodic.prompt_manager import load_prompt


def summary(count: Optional[int] = None, length: Optional[str] = None):
    """Summarize the recent conversation or entire history.
    
    Args:
        count: Number of exchanges to summarize (or 'all' or 'loaded')
        length: Summary length style (brief/short/standard/detailed/bulleted)
    """
    # Determine how many messages to summarize
    if count is None:
        # Default to last 10 exchanges (20 nodes)
        num_nodes = 20
        summary_type = "recent conversation"
    elif isinstance(count, str) and count.lower() == "all":
        # Summarize everything
        num_nodes = None
        summary_type = "entire conversation"
    elif isinstance(count, str) and count.lower() == "loaded":
        # Summarize the last loaded conversation
        if not conversation_manager.last_loaded_start_id:
            typer.secho("No conversation has been loaded yet", fg="red")
            return
        # Will handle this case specially below
        num_nodes = None
        summary_type = "loaded conversation"
    else:
        try:
            # Summarize last N exchanges (2N nodes)
            num_exchanges = int(count)
            num_nodes = num_exchanges * 2
            summary_type = f"last {num_exchanges} exchanges"
        except ValueError:
            typer.secho("Invalid count. Use a number or 'all'", fg="red")
            return
    
    # Get the nodes
    with benchmark_operation("Fetch nodes for summary"):
        if summary_type == "loaded conversation":
            # Get nodes between loaded start and end
            from episodic.db_nodes import get_descendants, get_node
            
            # Get all descendants of the start node
            descendants = get_descendants(conversation_manager.last_loaded_start_id)
            
            # Filter to only include nodes up to and including the end node
            nodes = []
            # First add the start node itself
            start_node = get_node(conversation_manager.last_loaded_start_id)
            if start_node:
                nodes.append(start_node)
            
            # Then add descendants up to the end node
            for node in descendants:
                nodes.append(node)
                if node['id'] == conversation_manager.last_loaded_end_id:
                    break
        elif num_nodes is None:
            nodes = get_recent_nodes(limit=1000)  # Reasonable limit for "all"
        else:
            nodes = get_recent_nodes(limit=num_nodes)
    
    if not nodes:
        typer.secho("No conversation history to summarize", fg=get_system_color())
        return
    
    # Build conversation text
    conversation_parts = []
    for node in reversed(nodes):  # Show in chronological order
        if node.get('content') and node.get('role'):
            if node['role'] == 'user':
                conversation_parts.append(f"User: {node['content']}")
            elif node['role'] == 'assistant':
                conversation_parts.append(f"Assistant: {node['content']}")
    
    if not conversation_parts:
        typer.secho("No conversation content to summarize", fg=get_system_color())
        return
    
    conversation_text = "\n\n".join(conversation_parts)
    
    # Count words/tokens
    word_count = len(conversation_text.split())
    
    typer.secho(f"\n📝 Summarizing {summary_type} ({len(nodes)} messages, ~{word_count} words)...", 
               fg=get_heading_color())
    
    # Load appropriate prompt template based on length
    length = length or "standard"  # Default to standard if not specified
    valid_lengths = ["brief", "short", "standard", "detailed", "bulleted"]
    
    if length not in valid_lengths:
        typer.secho(f"Invalid length '{length}'. Choose from: {', '.join(valid_lengths)}", fg="red")
        return
    
    # Load the prompt template
    prompt_data = load_prompt(f"summary_{length}")
    if prompt_data and prompt_data.get('content'):
        # Use the loaded template
        prompt_template = prompt_data['content']
        prompt = prompt_template.replace('{conversation_text}', conversation_text)
    else:
        # Fallback to default prompt
        typer.secho(f"Warning: Could not load prompt template for '{length}', using default", 
                   fg="yellow")
        prompt = f"""Please provide a concise summary of the following conversation. 
Focus on:
1. Main topics discussed
2. Key questions asked and answers provided  
3. Any decisions made or conclusions reached
4. Overall flow and progression of the conversation

Conversation to summarize:

{conversation_text}

Please structure the summary clearly with sections if there are multiple distinct topics."""
    
    # Get summary from LLM
    try:
        with benchmark_operation("Generate summary"):
            # Use streaming for the summary
            if config.get("stream_responses", True):
                typer.secho("\n🤖 Summary: ", fg=get_llm_color(), bold=True, nl=False)
                
                # Use the LLM with streaming
                from episodic.llm import _execute_llm_query
                model = config.get("model", "gpt-3.5-turbo")
                
                # Adjust system message based on length type
                system_messages = {
                    "brief": "You are an assistant that creates extremely concise summaries. Be as brief as possible.",
                    "short": "You are an assistant that creates short, compact summaries. Keep it concise.",
                    "standard": "You are an assistant that creates clear, well-structured summaries.",
                    "detailed": "You are an assistant that creates comprehensive, detailed summaries with clear organization.",
                    "bulleted": "You are an assistant that creates well-organized bullet-point summaries."
                }
                
                messages = [
                    {"role": "system", "content": system_messages.get(length, "You are a helpful assistant that creates clear, concise summaries.")},
                    {"role": "user", "content": prompt}
                ]
                
                stream_generator, _ = _execute_llm_query(
                    messages, 
                    model=model,
                    temperature=0.7,
                    stream=True
                )
                
                # Use unified streaming for consistent formatting
                from episodic.unified_streaming import unified_stream_response
                display_response = unified_stream_response(stream_generator, model, prefix="📝 ")
                
                # Calculate cost info for streaming response
                from litellm import token_counter, cost_per_token
                
                # Count tokens in the response
                output_tokens = token_counter(model=model, text=display_response)
                
                # Count tokens in the prompt (more accurate than estimation)
                prompt_tokens = token_counter(model=model, text=prompt)
                system_tokens = token_counter(model=model, text="You are a helpful assistant that creates clear, concise summaries.")
                input_tokens = prompt_tokens + system_tokens
                
                # Calculate actual cost
                total_cost = sum(cost_per_token(
                    model=model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens
                ))
                
                cost_info = {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'cost_usd': total_cost
                }
                
                # Show cost if enabled
                if config.get("show_cost", False) and cost_info and cost_info.get('cost_usd', 0) > 0:
                    typer.secho(f"\n💰 Summary cost: ${cost_info['cost_usd']:.4f}", 
                               fg=get_text_color(), dim=True)
            else:
                # Non-streaming response
                # Adjust system message based on length type
                system_messages = {
                    "brief": "You are an assistant that creates extremely concise summaries. Be as brief as possible.",
                    "short": "You are an assistant that creates short, compact summaries. Keep it concise.",
                    "standard": "You are an assistant that creates clear, well-structured summaries.",
                    "detailed": "You are an assistant that creates comprehensive, detailed summaries with clear organization.",
                    "bulleted": "You are an assistant that creates well-organized bullet-point summaries."
                }
                
                summary_text, cost_info = query_llm(
                    prompt,
                    model=config.get("model", "gpt-3.5-turbo"),
                    system_message=system_messages.get(length, "You are a helpful assistant that creates clear, concise summaries.")
                )
                
                typer.secho("\n🤖 Summary:", fg=get_llm_color(), bold=True)
                wrapped_llm_print(summary_text)
                
                # Show cost if enabled
                if config.get("show_cost", False) and cost_info:
                    total_cost = cost_info.get('cost_usd', 0)
                    if total_cost > 0:
                        typer.secho(f"\n💰 Summary cost: ${total_cost:.4f}", 
                                   fg=get_text_color(), dim=True)
        
        # Update session costs
        if 'cost_info' in locals() and cost_info:
            conversation_manager.session_costs["total_input_tokens"] += cost_info.get("input_tokens", 0)
            conversation_manager.session_costs["total_output_tokens"] += cost_info.get("output_tokens", 0)
            conversation_manager.session_costs["total_tokens"] += cost_info.get("total_tokens", 0)
            conversation_manager.session_costs["total_cost_usd"] += cost_info.get("cost_usd", 0.0)
        
    except Exception as e:
        typer.secho(f"\nError generating summary: {str(e)}", fg="red")
        return