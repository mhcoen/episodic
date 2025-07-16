"""Markdown export functionality for conversations."""

import os
from datetime import datetime
from typing import List, Tuple, Optional
from episodic.db_nodes import get_node, get_ancestry
from episodic.db_topics import get_recent_topics
from episodic.topics import count_nodes_in_topic

def export_topics_to_markdown(
    topic_spec: str, 
    filename: Optional[str] = None,
    export_dir: str = "exports"
) -> str:
    """Export specified topics to markdown file."""
    # Parse topic specification
    topics_to_export = get_topics_by_spec(topic_spec)
    
    if not topics_to_export:
        raise ValueError(f"No topics found for specification: {topic_spec}")
    
    # Get nodes for each topic
    topic_nodes_list = []
    for topic in topics_to_export:
        nodes = get_nodes_for_topic(topic)
        if nodes:
            topic_nodes_list.append((topic, nodes))
    
    # Generate filename if not provided
    if not filename:
        first_topic = topics_to_export[0]
        date_str = datetime.now().strftime("%Y-%m-%d")
        topic_name = first_topic['name'].lower().replace(' ', '-')
        filename = f"{topic_name}-{date_str}.md"
    
    # Ensure .md extension
    if not filename.endswith('.md'):
        filename += '.md'
    
    # Format to markdown
    markdown_content = format_conversation_to_markdown(topic_nodes_list)
    
    # Save to file
    os.makedirs(export_dir, exist_ok=True)
    filepath = os.path.join(export_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return filepath

def format_conversation_to_markdown(
    topic_nodes_list: List[Tuple[dict, List[dict]]]
) -> str:
    """Format topics and nodes to markdown."""
    lines = []
    
    # Build title from topic names
    topic_names = [t['name'] for t, _ in topic_nodes_list]
    if len(topic_names) == 1:
        title = topic_names[0]
    elif len(topic_names) <= 3:
        title = " and ".join(topic_names)
    else:
        title = f"{', '.join(topic_names[:2])}, and {len(topic_names)-2} more topics"
    
    # Header
    lines.append(f"# {title}")
    lines.append(f"*{datetime.now().strftime('%B %d, %Y')}*")
    
    # Get initial model/style from first assistant node
    model = 'unknown'
    if topic_nodes_list and topic_nodes_list[0][1]:
        # Find the first assistant node to get the model
        for node in topic_nodes_list[0][1]:
            if node.get('role') == 'assistant' and node.get('model'):
                model = node.get('model')
                # If there's a provider, prepend it
                if node.get('provider'):
                    model = f"{node.get('provider')}/{model}"
                break
        # TODO: Get actual style/format from config
        lines.append(f"*Model: {model} • Style: standard • Format: mixed*")
    
    lines.append("")
    
    # Process each topic
    for topic, nodes in topic_nodes_list:
        lines.append(f"## {topic['name']}")
        
        # Time range
        if nodes:
            start_time = nodes[0].get('created_at', '')
            end_time = nodes[-1].get('created_at', '')
            # TODO: Format times nicely
            lines.append(f"*{len(nodes)} messages*")
        
        lines.append("")
        
        current_model = None
        for node in nodes:
            # Track model changes
            node_model = node.get('model')
            if node_model and node_model != current_model:
                if current_model is not None:
                    lines.append(f"*Model changed to {node_model}*")
                    lines.append("")
                current_model = node_model
            
            # Format message
            role = node.get('role', 'user')
            content = node.get('content', '')
            
            # Determine speaker label
            if role == 'user':
                speaker = 'You'
            elif role == 'assistant':
                # Check for special sources in future
                speaker = 'LLM'
            else:
                speaker = 'System'
            
            # Handle special content markers
            if "[Response interrupted" in content:
                lines.append(f"**{speaker}**: [Response interrupted by user]")
                lines.append("")
                content = content.replace("[Response interrupted by user]", "").strip()
                if content:
                    lines.append(content)
            else:
                lines.append(f"**{speaker}**: {content}")
            
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Footer
    lines.append("<!-- ")
    lines.append(f"Exported from Episodic on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("Note: Re-importing will create new nodes, not reuse existing ones")
    lines.append("-->")
    
    return "\n".join(lines)

def get_topics_by_spec(spec: str) -> List[dict]:
    """Get topics based on specification."""
    if spec == "current":
        # Get current topic from conversation manager
        from episodic.conversation import conversation_manager
        
        # Get the most recent ongoing topic
        topics = get_recent_topics(limit=10)
        for topic in reversed(topics):  # Check from newest to oldest
            if topic.get('end_node_id') is None:  # Ongoing topic
                return [topic]
        
        # If no ongoing topic, get the most recent one
        if topics:
            return [topics[-1]]
        else:
            raise ValueError("No topics found")
    
    elif spec == "all":
        return get_recent_topics(limit=None)
    
    else:
        # Parse numeric specifications
        topic_numbers = parse_topic_numbers(spec)
        
        # Get topics using the same limit as /topics display (10)
        # to ensure numbering matches what user sees
        display_topics = get_recent_topics(limit=10)
        
        # If user requests a topic number beyond 10, get all topics
        max_requested = max(topic_numbers) if topic_numbers else 0
        if max_requested > 10:
            display_topics = get_recent_topics(limit=None)
        
        # Filter to requested topics (1-indexed)
        selected = []
        for num in topic_numbers:
            if 0 < num <= len(display_topics):
                selected.append(display_topics[num-1])
        
        return selected

def parse_topic_numbers(spec: str) -> List[int]:
    """Parse topic specification into list of numbers."""
    try:
        if '-' in spec:
            # Range like "1-3"
            start, end = spec.split('-', 1)
            return list(range(int(start), int(end) + 1))
        elif ',' in spec:
            # List like "1,3,5"
            return [int(x.strip()) for x in spec.split(',')]
        else:
            # Single number
            return [int(spec)]
    except ValueError:
        raise ValueError(f"Invalid topic specification: '{spec}'")

def get_nodes_for_topic(topic: dict) -> List[dict]:
    """Get all nodes belonging to a topic."""
    from episodic.db_nodes import get_ancestry, get_head, get_children
    
    # Get end node (or head if ongoing)
    end_node_id = topic.get('end_node_id')
    if not end_node_id:
        # Ongoing topic - use current head
        head_id = get_head()
        if head_id:
            end_node_id = head_id
        else:
            return []
    
    # Get full ancestry
    ancestry = get_ancestry(end_node_id)
    
    # Filter to nodes in this topic
    topic_nodes = []
    in_topic = False
    
    for node in ancestry:
        if node['id'] == topic['start_node_id']:
            in_topic = True
        
        if in_topic:
            topic_nodes.append(node)
        
        if node['id'] == topic.get('end_node_id'):
            break
    
    # Check if we need to add an assistant response after the end node
    # This handles the case where topics end on user messages
    if topic_nodes and topic_nodes[-1].get('role') == 'user':
        children = get_children(topic.get('end_node_id'))
        for child in children:
            if child.get('role') == 'assistant':
                topic_nodes.append(child)
                break  # Only add the first assistant response
    
    return topic_nodes