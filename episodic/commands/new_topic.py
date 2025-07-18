"""
Command to explicitly start a new topic/conversation.

This helps users clearly separate different conversations or work sessions.
"""

import typer
from typing import Optional
from datetime import datetime

from episodic.config import config
from episodic.configuration import get_system_color, get_text_color
from episodic.conversation import conversation_manager
from episodic.db import insert_node, get_recent_topics, store_topic


def new_command(topic_name: Optional[str] = None):
    """
    Start a new topic/conversation.
    
    Usage:
        /new                    # Start a fresh topic
        /new "Project Planning" # Start a new topic with a specific name
    """
    try:
        # Get current node ID before we add the system message
        current_node_id = conversation_manager.get_current_node_id()
        
        # Check if there's an ongoing topic to end
        recent_topics = get_recent_topics(limit=1)
        if recent_topics and recent_topics[0]['end_node_id'] is None:
            # End the current topic at the current node
            old_topic = recent_topics[0]
            from episodic.db_topics import update_topic_end
            update_topic_end(old_topic['id'], current_node_id)
            typer.secho(f"✓ Ended topic: {old_topic['name']}", fg=get_text_color(), dim=True)
        
        # Create a system message to mark the boundary
        system_message = "--- New conversation started ---"
        if topic_name:
            system_message = f"--- New conversation: {topic_name} ---"
            
        # Add a system node to mark the transition
        node_data = {
            'node_type': 'system',
            'model': 'system',
            'content': system_message,
            'parent_id': current_node_id,
            'tokens_input': 0,
            'tokens_output': 0,
            'cost': 0.0
        }
        
        system_node_id = insert_node(node_data)
        conversation_manager.update_head(system_node_id)
        
        # Create the new topic starting after the system message
        # The topic will officially start with the user's next message
        if topic_name:
            # Pre-create the topic with the given name
            store_topic(topic_name, system_node_id, None)
            typer.secho(f"\n🆕 Started new topic: {topic_name}", fg=get_system_color(), bold=True)
        else:
            typer.secho("\n🆕 Started fresh conversation", fg=get_system_color(), bold=True)
            typer.secho("The next message will begin a new topic.", fg=get_text_color(), dim=True)
            
        typer.secho("You can now start chatting about a new subject.", fg=get_text_color())
        
        # In simple mode, remind about save
        if config.get("interface_mode", "advanced") == "simple":
            typer.secho("\n💡 Remember to /save your previous conversation if needed!", 
                       fg=get_text_color(), dim=True)
        
    except Exception as e:
        typer.secho(f"Error starting new topic: {str(e)}", fg="red")
        if config.get("debug"):
            import traceback
            typer.secho(traceback.format_exc(), fg="red")


def clear_command():
    """
    Alias for /new - start a fresh conversation.
    
    Usage:
        /clear    # Start fresh
    """
    new_command()