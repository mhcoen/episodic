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


def summary(count: Optional[int] = None):
    """Summarize the recent conversation or entire history."""
    # Determine how many messages to summarize
    if count is None:
        # Default to last 10 exchanges (20 nodes)
        num_nodes = 20
        summary_type = "recent conversation"
    elif isinstance(count, str) and count.lower() == "all":
        # Summarize everything
        num_nodes = None
        summary_type = "entire conversation"
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
        if num_nodes is None:
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
    
    # Create summary prompt
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
                from episodic.llm import _execute_llm_query, process_stream_response
                model = config.get("model", "gpt-3.5-turbo")
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that creates clear, concise summaries."},
                    {"role": "user", "content": prompt}
                ]
                
                stream_generator, _ = _execute_llm_query(
                    messages, 
                    model=model,
                    temperature=0.7,
                    stream=True
                )
                
                # Process the stream with StreamingFormatter
                from episodic.text_formatter import StreamingFormatter
                formatter = StreamingFormatter(base_color=get_llm_color())
                
                full_response_parts = []
                for chunk in process_stream_response(stream_generator, model):
                    formatter.process_chunk(chunk)
                    full_response_parts.append(chunk)
                
                formatter.finish()
                display_response = ''.join(full_response_parts)
                
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
                summary_text, cost_info = query_llm(
                    prompt,
                    model=config.get("model", "gpt-3.5-turbo"),
                    system_message="You are a helpful assistant that creates clear, concise summaries."
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