"""
Command to rename ongoing-discussion topics.
"""

import typer
from typing import Optional
from episodic.db import get_recent_topics, get_ancestry, update_topic_name
from episodic.topics import extract_topic_ollama, build_conversation_segment
from episodic.configuration import get_system_color, get_heading_color


def rename_ongoing_topics():
    """Find and rename all ongoing-discussion topics."""
    # Get all topics
    topics = get_recent_topics(limit=100)
    
    # Match both patterns: 'ongoing-discussion-' and 'ongoing-' followed by numbers
    ongoing_topics = [t for t in topics if t['name'].startswith('ongoing-')]
    
    if not ongoing_topics:
        typer.secho("No ongoing topics found.", fg=get_system_color())
        return
    
    typer.secho(f"\n📝 Found {len(ongoing_topics)} ongoing topics to rename", fg=get_heading_color())
    
    renamed_count = 0
    for topic in ongoing_topics:
        try:
            # Get the conversation for this topic
            # For ongoing topics (end_node_id is NULL), use the current head
            if topic['end_node_id'] is None:
                from episodic.db import get_head
                current_head = get_head()
                if not current_head:
                    typer.secho(f"  ⚠️  Could not find current head for ongoing topic {topic['name']}", fg="yellow")
                    continue
                end_ancestry = get_ancestry(current_head)
            else:
                end_ancestry = get_ancestry(topic['end_node_id'])
            
            if not end_ancestry:
                typer.secho(f"  ⚠️  Could not get ancestry for topic {topic['name']}", fg="yellow")
                continue
            
            # Build the segment for this topic
            segment_nodes = []
            in_segment = False
            for node in end_ancestry:
                if node['id'] == topic['start_node_id']:
                    in_segment = True
                if in_segment:
                    segment_nodes.append(node)
                # For ongoing topics, include all nodes after start
                if topic['end_node_id'] is not None and node['id'] == topic['end_node_id']:
                    break
            
            if len(segment_nodes) < 2:
                typer.secho(f"  ⚠️  Topic {topic['name']} has too few nodes", fg="yellow")
                continue
            
            # Build conversation text
            segment = build_conversation_segment(segment_nodes, max_length=2000)
            
            # Extract topic name
            topic_name, _ = extract_topic_ollama(segment)
            
            if topic_name and topic_name != topic['name']:
                # Update the topic name
                rows = update_topic_name(topic['name'], topic['start_node_id'], topic_name)
                if rows > 0:
                    typer.secho(f"  ✅ Renamed '{topic['name']}' → '{topic_name}'", fg="green")
                    renamed_count += 1
                else:
                    typer.secho(f"  ⚠️  Failed to rename {topic['name']}", fg="yellow")
            else:
                typer.secho(f"  ℹ️  Could not extract name for {topic['name']}", fg="yellow")
                
        except Exception as e:
            typer.secho(f"  ❌ Error processing {topic['name']}: {e}", fg="red")
    
    typer.secho(f"\n✅ Renamed {renamed_count} topics", fg=get_system_color())