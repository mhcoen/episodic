"""
Topic management commands for the Episodic CLI.
"""

import typer
from typing import Optional, List
from datetime import datetime
from episodic.db import get_recent_topics, get_node, get_ancestry
from episodic.topics import TopicManager, count_nodes_in_topic
from episodic.configuration import (
    get_heading_color, get_text_color, get_system_color, get_llm_color
)
from episodic.benchmark import benchmark_operation


def topics(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of topics to show"),
    all: bool = typer.Option(False, "--all", "-a", help="Show all topics"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information")
):
    """List conversation topics."""
    from episodic.conversation import wrapped_text_print
    
    with benchmark_operation("List topics"):
        # Get topics
        if all:
            topic_list = get_recent_topics(limit=None)
        else:
            topic_list = get_recent_topics(limit=limit)
        
        if not topic_list:
            typer.secho("No topics found yet. Topics are created as conversations progress.", 
                       fg=get_system_color())
            return
        
        typer.secho(f"\n📑 Conversation Topics ({len(topic_list)} total)", 
                   fg=get_heading_color(), bold=True)
        typer.secho("=" * 70, fg=get_heading_color())
        
        for i, topic in enumerate(topic_list):
            # Topic header
            status = "✓" if topic['end_node_id'] else "○"
            typer.secho(f"\n[{i+1}] {status} ", fg=get_heading_color(), bold=True, nl=False)
            typer.secho(topic['name'], fg=get_text_color(), bold=True)
            
            # Time range
            if topic.get('created_at'):
                try:
                    created = datetime.fromisoformat(topic['created_at'].replace('Z', '+00:00'))
                    time_str = created.strftime("%Y-%m-%d %H:%M")
                    typer.secho(f"    Created: {time_str}", fg=get_text_color(), dim=True)
                except:
                    pass
            
            # Node range
            typer.secho(f"    Range: ", fg=get_text_color(), nl=False)
            start_node = get_node(topic['start_node_id'])
            end_node = get_node(topic['end_node_id']) if topic['end_node_id'] else None
            
            if start_node:
                typer.secho(f"{start_node['short_id']}", fg=get_system_color(), nl=False)
            else:
                typer.secho(f"{topic['start_node_id'][:8]}...", fg=get_system_color(), nl=False)
            
            typer.secho(" → ", fg=get_text_color(), nl=False)
            
            if end_node:
                typer.secho(f"{end_node['short_id']}", fg=get_system_color(), nl=False)
            else:
                typer.secho("ongoing", fg="yellow", nl=False)
            
            # Message count
            if start_node:
                ancestry = get_ancestry(topic['end_node_id'] or topic['start_node_id'])
                node_count = count_nodes_in_topic(topic['start_node_id'], topic['end_node_id'])
                typer.secho(f" ({node_count} messages)", fg=get_text_color(), dim=True)
            else:
                typer.echo("")
            
            # Confidence score if available
            if topic.get('confidence'):
                # Handle both numeric and string confidence values
                confidence = topic['confidence']
                if isinstance(confidence, (int, float)):
                    confidence_pct = int(confidence * 100)
                    typer.secho(f"    Confidence: {confidence_pct}%", fg=get_text_color(), dim=True)
                elif isinstance(confidence, str) and confidence != 'detected':
                    typer.secho(f"    Confidence: {confidence}", fg=get_text_color(), dim=True)
            
            # Show preview if verbose
            if verbose and start_node:
                typer.secho("    Preview:", fg=get_text_color(), dim=True)
                # Get first few nodes in topic
                nodes_in_topic = []
                for node in ancestry:
                    if is_node_in_topic_range(node['id'], topic, ancestry):
                        nodes_in_topic.append(node)
                        if len(nodes_in_topic) >= 2:  # Show first 2 exchanges
                            break
                
                for node in reversed(nodes_in_topic):  # Show in chronological order
                    if node.get('message'):
                        msg = node['message'].replace('\n', ' ')[:60]
                        if len(node['message']) > 60:
                            msg += "..."
                        typer.secho(f"      💬 {msg}", fg=get_text_color())
                    
                    if node.get('response'):
                        resp = node['response'].replace('\n', ' ')[:60]
                        if len(node['response']) > 60:
                            resp += "..."
                        typer.secho(f"      🤖 {resp}", fg=get_llm_color())
        
        typer.secho("\n" + "=" * 70, fg=get_heading_color())
        
        # Show tips
        if not all and len(topic_list) == limit:
            typer.secho(f"💡 Showing {limit} most recent topics. Use --all to see all topics.", 
                       fg=get_text_color(), dim=True)


def compress_current_topic():
    """Compress the current active topic if it meets requirements."""
    from episodic.topics import TopicManager
    from episodic.compression import queue_topic_for_compression
    from episodic.config import config
    
    tm = TopicManager()
    current_topic = tm.get_current_topic()
    
    if not current_topic:
        typer.secho("No active topic found.", fg=get_system_color())
        return
    
    if not current_topic.get('end_node_id'):
        typer.secho("Cannot compress ongoing topic. Topic must be closed first.", 
                   fg="yellow")
        return
    
    # Check if topic is large enough
    ancestry = get_ancestry(current_topic['end_node_id'])
    node_count = count_nodes_in_topic(current_topic, ancestry)
    min_nodes = config.get('compression_min_nodes', 10)
    
    if node_count < min_nodes:
        typer.secho(f"Topic too small for compression ({node_count} < {min_nodes} messages)", 
                   fg="yellow")
        return
    
    # Queue for compression
    typer.secho(f"Queuing topic '{current_topic['name']}' for compression...", 
               fg=get_system_color())
    
    success = queue_topic_for_compression(
        current_topic['start_node_id'],
        current_topic['end_node_id'],
        current_topic['name']
    )
    
    if success:
        typer.secho("✅ Topic queued for compression", fg=get_system_color())
    else:
        typer.secho("Topic already queued or being compressed", fg="yellow")


def is_node_in_topic_range(node_id: str, topic: dict, ancestry: List[dict]) -> bool:
    """Check if a node is within the topic's range."""
    # Find positions in ancestry
    start_pos = None
    end_pos = None
    node_pos = None
    
    for i, node in enumerate(ancestry):
        if node['id'] == topic['start_node_id']:
            start_pos = i
        if topic.get('end_node_id') and node['id'] == topic['end_node_id']:
            end_pos = i
        if node['id'] == node_id:
            node_pos = i
    
    if start_pos is None or node_pos is None:
        return False
    
    if end_pos is None:
        # Open-ended topic
        return node_pos <= start_pos
    else:
        # Closed topic
        return end_pos <= node_pos <= start_pos