"""Markdown import functionality for conversations."""

import re
from typing import List, Dict, Tuple, Optional
from episodic.db_nodes import insert_node
from episodic.db_topics import store_topic

def import_markdown_file(
    filepath: str,
    conversation_manager
) -> str:
    """Import markdown file into current conversation."""
    # Parse the file
    parsed_data = parse_markdown_file(filepath)
    
    if not parsed_data['topics']:
        raise ValueError("No conversation content found in file")
    
    # Create nodes continuing from current position
    last_node_id = create_nodes_from_markdown(
        parsed_data,
        conversation_manager
    )
    
    return last_node_id

def parse_markdown_file(filepath: str) -> Dict:
    """Parse markdown file into structured conversation data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    topics = parse_topics(content)
    
    return {
        'topics': topics,
        'metadata': extract_metadata(content)
    }

def parse_topics(content: str) -> List[Dict]:
    """Parse markdown content into topics with messages."""
    topics = []
    current_topic = None
    current_messages = []
    
    lines = content.splitlines()
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Skip empty lines
        if not line.strip():
            i += 1
            continue
        
        # Skip metadata lines (but not bold text like **You**)
        if (line.strip().startswith('*') and not line.strip().startswith('**')) or line.strip().startswith('<!--'):
            i += 1
            continue
        
        # Main title (# Title) - skip it
        if line.startswith('# '):
            i += 1
            continue
        
        # Topic header (## Topic Name)
        if line.startswith('## '):
            # Save previous topic if exists
            if current_topic and current_messages:
                topics.append({
                    'name': current_topic,
                    'messages': current_messages
                })
            
            current_topic = line[3:].strip()
            current_messages = []
            i += 1
            continue
        
        # Horizontal rule - skip
        if line.strip() == '---':
            i += 1
            continue
        
        # Try to parse as a message
        speaker_match = detect_speaker_pattern(line)
        if speaker_match:
            role, content_start = speaker_match
            
            # Collect content (may be on same line or following lines)
            content_parts = []
            if content_start:
                content_parts.append(content_start)
            
            # Look ahead for continuation lines
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                
                # Stop at next speaker or structural element
                if (not next_line.strip() or 
                    detect_speaker_pattern(next_line) or
                    next_line.startswith('#') or
                    next_line.strip() == '---' or
                    next_line.strip().startswith('*')):
                    break
                
                content_parts.append(next_line)
                j += 1
            
            # Update position to continue after the message
            i = j
            
            # Create message
            content = '\n'.join(content_parts).strip()
            if content:
                current_messages.append({
                    'role': role,
                    'content': content
                })
            continue  # Skip the i += 1 at the end
        
        i += 1
    
    # Don't forget the last topic
    if current_topic and current_messages:
        topics.append({
            'name': current_topic,
            'messages': current_messages
        })
    
    return topics

def detect_speaker_pattern(line: str) -> Optional[Tuple[str, str]]:
    """Detect speaker patterns and return (role, remaining_content)."""
    patterns = [
        (r'\*\*You\*\*:\s*(.*)', 'user'),
        (r'\*\*LLM\*\*:\s*(.*)', 'assistant'),
        (r'\*\*User\*\*:\s*(.*)', 'user'),
        (r'\*\*Assistant\*\*:\s*(.*)', 'assistant'),
        (r'\*\*Human\*\*:\s*(.*)', 'user'),
        (r'\*\*AI\*\*:\s*(.*)', 'assistant'),
        (r'\*\*System\*\*:\s*(.*)', 'system'),
        # Also support without bold
        (r'You:\s*(.*)', 'user'),
        (r'User:\s*(.*)', 'user'),
        (r'Human:\s*(.*)', 'user'),
        (r'Assistant:\s*(.*)', 'assistant'),
        (r'AI:\s*(.*)', 'assistant'),
        (r'LLM:\s*(.*)', 'assistant'),
    ]
    
    for pattern, role in patterns:
        if match := re.match(pattern, line.strip()):
            content = match.group(1) if match.lastindex else ''
            return (role, content)
    
    return None

def create_nodes_from_markdown(
    parsed_data: Dict,
    conversation_manager
) -> str:
    """Create new nodes from parsed markdown data."""
    # Start from current conversation position
    parent_id = conversation_manager.current_node_id
    last_node_id = parent_id
    first_node_id = None  # Track the first node we create
    
    for topic in parsed_data['topics']:
        topic_start_node = None
        
        for message in topic['messages']:
            # Always create new node
            node_id, short_id = insert_node(
                content=message['content'],
                parent_id=parent_id,
                role=message['role']
            )
            
            # Track the very first node we create
            if first_node_id is None:
                first_node_id = node_id
            
            if topic_start_node is None:
                topic_start_node = node_id
            
            parent_id = node_id
            last_node_id = node_id
        
        # Create topic if we have nodes
        if topic_start_node:
            store_topic(
                name=topic['name'],
                start_node_id=topic_start_node,
                end_node_id=last_node_id,
                confidence='imported'
            )
    
    # Store the loaded range in conversation manager
    if first_node_id and conversation_manager:
        conversation_manager.last_loaded_start_id = first_node_id
        conversation_manager.last_loaded_end_id = last_node_id
    
    return last_node_id

def extract_metadata(content: str) -> Dict:
    """Extract any metadata from the markdown."""
    metadata = {}
    
    # Look for common metadata patterns
    lines = content.splitlines()
    for line in lines[:10]:  # Check first 10 lines
        if line.startswith('*') and line.endswith('*'):
            # Potential metadata line
            inner = line.strip('*').strip()
            if 'Model:' in inner:
                metadata['model'] = inner
            elif 'Date:' in inner:
                metadata['date'] = inner
    
    return metadata