"""
Compression commands for the Episodic CLI.
"""

import typer
from typing import Optional, List, Dict
from episodic.db import get_node, get_ancestry
from episodic.compression import compression_manager
from episodic.llm import query_llm
from episodic.llm_manager import llm_manager
from episodic.configuration import (
    get_heading_color, get_text_color, get_system_color
)
from episodic.benchmark import benchmark_operation
from episodic.conversation import conversation_manager


def compress(
    strategy: str = typer.Option("simple", "--strategy", "-s", 
                                help="Compression strategy: simple, keymoments"),
    node_id: Optional[str] = typer.Option(None, "--node", "-n", 
                                         help="Specific node to compress from"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", 
                               help="Show what would be compressed without doing it")
):
    """Compress the conversation history from current or specified node."""
    from episodic.db import get_node_by_short_id
    
    # Determine which node to compress from
    if node_id:
        # Use specified node
        target_node = get_node_by_short_id(node_id) or get_node(node_id)
        if not target_node:
            typer.secho(f"Error: Node '{node_id}' not found", fg="red", err=True)
            return
        target_id = target_node['id']
    else:
        # Use current node
        target_id = conversation_manager.get_current_node_id()
        if not target_id:
            typer.secho("No current node set. Use '/head <node>' or specify --node", 
                       fg="yellow")
            return
        target_node = get_node(target_id)
    
    # Get the branch to compress
    with benchmark_operation("Fetch ancestry for compression"):
        nodes = get_ancestry(target_id)
    
    if len(nodes) < 3:
        typer.secho("Not enough history to compress (need at least 3 nodes)", 
                   fg="yellow")
        return
    
    typer.secho(f"\n📦 Compressing {len(nodes)} nodes from branch ending at {target_node['short_id']}...", 
               fg=get_heading_color())
    
    # Choose compression strategy
    if strategy == "simple":
        result = _compress_branch_simple(nodes, dry_run)
    elif strategy == "keymoments":
        result = _compress_branch_key_moments(nodes, dry_run)
    else:
        typer.secho(f"Unknown strategy: {strategy}", fg="red")
        return
    
    if dry_run:
        typer.secho("\n🔍 Dry run - no changes made", fg="yellow")
        typer.secho("\nWould produce this summary:", fg=get_text_color())
        typer.secho("─" * 60, fg=get_text_color())
        typer.echo(result)
        typer.secho("─" * 60, fg=get_text_color())
    else:
        typer.secho(f"\n✅ Compression complete", fg=get_system_color())
        # In a real implementation, we might store this summary somewhere


def _compress_branch_simple(nodes: List[Dict], dry_run: bool = False) -> str:
    """Simple compression strategy - just summarize the conversation."""
    # Build conversation text
    conversation_parts = []
    for node in reversed(nodes):  # Chronological order
        if node.get('message'):
            conversation_parts.append(f"User: {node['message']}")
        if node.get('response'):
            conversation_parts.append(f"Assistant: {node['response']}")
    
    conversation_text = "\n\n".join(conversation_parts)
    
    # Create compression prompt
    prompt = f"""Summarize this conversation concisely, capturing the main topics discussed and key conclusions reached:

{conversation_text}

Provide a clear, structured summary that someone could read to understand what was discussed."""
    
    if not dry_run:
        with benchmark_operation("LLM compression"):
            summary, _ = query_llm(prompt, system_message="You are a helpful assistant that creates concise summaries.")
            return summary
    else:
        return f"[Would compress {len(nodes)} nodes into a summary]"


def _compress_branch_key_moments(nodes: List[Dict], dry_run: bool = False) -> str:
    """Key moments compression - identify and preserve important exchanges."""
    # Build conversation with node IDs
    conversation_parts = []
    for i, node in enumerate(reversed(nodes)):  # Chronological order
        if node.get('message'):
            conversation_parts.append(f"[{i}] User: {node['message']}")
        if node.get('response'):
            conversation_parts.append(f"[{i}] Assistant: {node['response']}")
    
    conversation_text = "\n\n".join(conversation_parts)
    
    # Create key moments prompt
    prompt = f"""Analyze this conversation and identify the KEY MOMENTS - the most important exchanges that capture the essence of the discussion.

{conversation_text}

Identify 3-5 key moments by their numbers [N] and explain why each is important. Then provide a brief summary connecting these moments."""
    
    if not dry_run:
        with benchmark_operation("LLM key moments analysis"):
            analysis, _ = query_llm(prompt, system_message="You are a helpful assistant that identifies key moments in conversations.")
            return analysis
    else:
        return f"[Would identify key moments from {len(nodes)} nodes]"


def compression_stats():
    """Show compression statistics."""
    stats = compression_manager.get_stats()
    
    typer.secho("\n📊 Compression Statistics", fg=get_heading_color(), bold=True)
    typer.secho("─" * 40, fg=get_heading_color())
    
    typer.echo(f"Topics compressed: {stats['total_compressed']}")
    typer.echo(f"Failed compressions: {stats['failed_compressions']}")
    typer.echo(f"Total words saved: {stats['total_words_saved']:,}")
    typer.echo(f"Compression ratio: {stats.get('compression_ratio', 0):.1f}x")
    
    if stats['total_compressed'] > 0:
        avg_saved = stats['total_words_saved'] / stats['total_compressed']
        typer.echo(f"Avg words saved per topic: {avg_saved:.0f}")
    
    # Show queue status
    queue_info = compression_manager.get_queue_info()
    if queue_info:
        typer.secho("\n📋 Queue Status", fg=get_heading_color())
        typer.echo(f"Pending jobs: {len(queue_info)}")
        if queue_info:
            typer.echo(f"Next topic: {queue_info[0]['topic']}")
    
    # Show if compression is running
    if compression_manager.running:
        typer.secho("\n✅ Compression worker is running", fg=get_system_color())
    else:
        typer.secho("\n⚠️  Compression worker is not running", fg="yellow")


def compression_queue():
    """Show pending compression jobs."""
    queue_info = compression_manager.get_queue_info()
    
    if not queue_info:
        typer.secho("No pending compression jobs", fg=get_system_color())
        return
    
    typer.secho(f"\n📋 Compression Queue ({len(queue_info)} jobs)", 
               fg=get_heading_color(), bold=True)
    typer.secho("─" * 60, fg=get_heading_color())
    
    for i, job in enumerate(queue_info):
        typer.secho(f"\n[{i+1}] {job['topic']}", fg=get_text_color(), bold=True)
        typer.echo(f"    Priority: {job['priority']}")
        typer.echo(f"    Attempts: {job['attempts']}")
        typer.echo(f"    Created: {job['created_at']}")
        
        # Show node range if possible
        start_node = get_node(job['start_node_id'])
        end_node = get_node(job['end_node_id'])
        if start_node and end_node:
            typer.echo(f"    Range: {start_node['short_id']} → {end_node['short_id']}")


def api_call_stats():
    """Display LLM API call statistics by thread."""
    stats = llm_manager.get_call_stats()
    
    typer.secho("\n📊 LLM API Call Statistics", fg=get_heading_color(), bold=True)
    typer.secho("=" * 50, fg=get_heading_color())
    
    typer.echo(f"Total API calls: {stats['total_calls']}")
    typer.echo(f"Total time: {stats['total_time']:.2f}s")
    
    if stats['by_thread']:
        typer.echo("\nBy thread:")
        for thread_id, thread_stats in stats['by_thread'].items():
            typer.echo(f"  Thread {thread_id}: {thread_stats['calls']} calls, {thread_stats['time']:.2f}s")
    else:
        typer.echo("No API calls recorded yet.")


def reset_api_stats():
    """Reset LLM API call statistics."""
    llm_manager.reset_stats()
    typer.secho("✅ API call statistics reset", fg=get_system_color())