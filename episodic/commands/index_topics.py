"""
Manual topic indexing command using sliding window analysis.
"""

import typer
from typing import List, Dict, Any, Tuple
from episodic.db import get_ancestry, get_head, store_topic, get_recent_topics, update_topic_end_node
from episodic.topics_hybrid import HybridTopicDetector
from episodic.config import config
from episodic.configuration import get_text_color, get_system_color, get_heading_color


def index_topics(
    window_size: int = typer.Argument(5, help="Size of sliding window for comparison"),
    apply: bool = typer.Option(False, "--apply", "-a", help="Apply detected topics (otherwise just preview)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed scores for each position")
):
    """
    Manually detect topics using sliding window analysis across the entire conversation.
    
    Compares windows of n user messages to detect topic transitions.
    """
    # Get current conversation head
    head = get_head()
    if not head:
        typer.secho("No conversation found.", fg=get_system_color())
        return
    
    # Get full ancestry (all nodes from root to current)
    ancestry = get_ancestry(head)
    
    # Extract only user messages with their positions
    user_messages = []
    for i, node in enumerate(reversed(ancestry)):  # Reverse to get chronological order
        if node.get('role') == 'user' and node.get('content'):
            user_messages.append({
                'position': i,
                'node_id': node['id'],
                'short_id': node.get('short_id', '??'),
                'content': node['content']
            })
    
    if len(user_messages) < 2:
        typer.secho("Not enough user messages for topic detection.", fg=get_system_color())
        return
    
    typer.secho(f"\n🔍 Analyzing {len(user_messages)} user messages with window size {window_size}", 
               fg=get_heading_color(), bold=True)
    typer.secho("=" * 80, fg=get_heading_color())
    
    # Initialize detector
    detector = HybridTopicDetector()
    
    # Store results
    detection_results = []
    
    # Sliding window analysis
    for i in range(len(user_messages) - 1):
        # Determine window boundaries
        # Window A: Previous messages (including current)
        window_a_start = max(0, i - window_size + 1)
        window_a_end = i + 1
        
        # Window B: Following messages
        window_b_start = i + 1
        window_b_end = min(len(user_messages), i + 1 + window_size)
        
        # Skip if window B would be empty
        if window_b_start >= len(user_messages):
            continue
        
        # Extract window contents
        window_a = user_messages[window_a_start:window_a_end]
        window_b = user_messages[window_b_start:window_b_end]
        
        # Skip if either window is empty
        if not window_a or not window_b:
            continue
        
        # Create combined text for each window
        window_a_text = " ".join([msg['content'] for msg in window_a])
        window_b_text = " ".join([msg['content'] for msg in window_b])
        
        # Use drift calculator to compare windows
        node_a = {"message": window_a_text}
        node_b = {"message": window_b_text}
        
        drift_score = detector.drift_calculator.calculate_drift(node_a, node_b)
        
        # Also check for keyword transitions in the boundary message
        boundary_msg = user_messages[i]['content']
        keyword_results = detector.transition_detector.detect_transition_keywords(boundary_msg)
        
        # Combine scores (simplified - using mostly drift for window comparison)
        combined_score = (drift_score * 0.7) + (keyword_results['explicit_transition'] * 0.3)
        
        result = {
            'position': i,
            'node_id': user_messages[i]['node_id'],
            'short_id': user_messages[i]['short_id'],
            'drift_score': drift_score,
            'keyword_score': keyword_results['explicit_transition'],
            'combined_score': combined_score,
            'window_a_size': len(window_a),
            'window_b_size': len(window_b),
            'message_preview': user_messages[i]['content'][:50] + '...',
            'transition_phrase': keyword_results.get('found_phrase'),
            'is_boundary': combined_score >= config.get('hybrid_topic_threshold', 0.55)
        }
        
        detection_results.append(result)
        
        # Show progress
        if verbose or result['is_boundary']:
            status = "🔄 TOPIC CHANGE" if result['is_boundary'] else "  continuation"
            color = "green" if result['is_boundary'] else "dim"
            
            typer.secho(f"\n[{result['short_id']}] {status}", fg=color, bold=result['is_boundary'])
            typer.secho(f"Message: {result['message_preview']}", fg=get_text_color())
            typer.secho(f"Windows: {result['window_a_size']} msgs ← | → {result['window_b_size']} msgs", 
                       fg=get_text_color(), dim=True)
            
            if verbose:
                typer.secho(f"Scores: drift={result['drift_score']:.3f}, keyword={result['keyword_score']:.3f}, combined={result['combined_score']:.3f}", 
                           fg=get_text_color(), dim=True)
                if result['transition_phrase']:
                    typer.secho(f"Transition phrase: '{result['transition_phrase']}'", 
                               fg=get_text_color(), dim=True)
    
    # Find topic boundaries
    boundaries = [r for r in detection_results if r['is_boundary']]
    
    typer.secho("\n" + "=" * 80, fg=get_heading_color())
    typer.secho(f"\n📊 Detected {len(boundaries)} topic boundaries", fg=get_heading_color(), bold=True)
    
    if not boundaries:
        typer.secho("No topic changes detected with current threshold.", fg=get_system_color())
        return
    
    # Show proposed topic structure
    typer.secho("\nProposed topics:", fg=get_heading_color())
    
    # Add start of conversation as first boundary
    topic_ranges = []
    start_node = user_messages[0]
    
    for i, boundary in enumerate(boundaries):
        # End previous topic at the message before the boundary
        end_idx = user_messages.index(next(m for m in user_messages if m['node_id'] == boundary['node_id']))
        if end_idx > 0:
            end_node = user_messages[end_idx - 1]
            topic_ranges.append({
                'start': start_node,
                'end': end_node,
                'message_count': end_idx - user_messages.index(start_node) + 1
            })
        
        # Start new topic at boundary
        start_node = next(m for m in user_messages if m['node_id'] == boundary['node_id'])
    
    # Add final topic
    if start_node != user_messages[-1]:
        topic_ranges.append({
            'start': start_node,
            'end': user_messages[-1],
            'message_count': len(user_messages) - user_messages.index(start_node)
        })
    
    # Display proposed topics
    for i, topic in enumerate(topic_ranges):
        typer.secho(f"\nTopic {i+1}:", fg=get_text_color(), bold=True)
        typer.secho(f"  Range: [{topic['start']['short_id']}] → [{topic['end']['short_id']}]", 
                   fg=get_text_color())
        typer.secho(f"  Messages: {topic['message_count']} user messages", fg=get_text_color())
        typer.secho(f"  Starts with: {topic['start']['content'][:60]}...", 
                   fg=get_text_color(), dim=True)
    
    # Apply topics if requested
    if apply:
        typer.secho("\n✅ Applying detected topics...", fg=get_system_color())
        
        # Clear existing topics (optional - could prompt user)
        existing_topics = get_recent_topics(limit=None)
        if existing_topics:
            typer.secho(f"Note: This will replace {len(existing_topics)} existing topics", 
                       fg="yellow")
        
        # TODO: Implement actual topic creation/update logic
        # This would involve:
        # 1. Clearing or updating existing topics
        # 2. Creating new topics with proper boundaries
        # 3. Extracting topic names for each range
        
        typer.secho("Topic application not yet implemented.", fg="yellow")
    else:
        typer.secho("\n💡 Use --apply to create these topics", fg=get_text_color(), dim=True)