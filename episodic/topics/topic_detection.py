"""
Topic change detection functionality.

This module handles detecting when conversation topics change.
"""

import json
import logging
from typing import Optional, List, Dict, Any, Tuple

import typer

from episodic.config import config
from episodic.llm import query_llm
from episodic.prompt_manager import PromptManager
from episodic.benchmark import benchmark_resource

# Set up logging
logger = logging.getLogger(__name__)


def detect_topic_change_separately(
    recent_messages: List[Dict[str, Any]], 
    new_message: str,
    current_topic: Optional[Tuple[str, str]] = None,
    prompt_manager: Optional[PromptManager] = None
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Detect if the topic has changed by analyzing recent messages and the new message.
    
    This function runs separately from the main conversation flow, using a focused
    LLM call to determine if the topic has shifted.
    
    Args:
        recent_messages: List of recent conversation nodes (3-5 messages)
        new_message: The new user message to analyze
        current_topic: Optional tuple of (topic_name, start_node_id) for the current topic
        prompt_manager: Optional PromptManager instance (will create if not provided)
        
    Returns:
        Tuple of (topic_changed: bool, new_topic_name: Optional[str], cost_info: Optional[Dict])
    """
    if not prompt_manager:
        prompt_manager = PromptManager()
        
    try:
        # Build context from recent messages
        use_v2_prompt = config.get("topic_detection_v2", False)
        
        if use_v2_prompt:
            # V2 prompt expects only user messages in a specific format
            context_parts = []
            user_messages = [msg for msg in recent_messages if msg.get('role') == 'user']
            # Take the last 3-5 user messages
            for msg in user_messages[-5:]:
                content = msg.get('content', '').strip()
                if content:
                    context_parts.append(f"- User: {content}")
            context = "\n".join(context_parts)
        else:
            # Original format includes both user and assistant messages
            context_parts = []
            # Take the last 4 messages and reverse to get chronological order
            messages_for_context = list(reversed(recent_messages[-4:]))
            for msg in messages_for_context:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '').strip()
                if content:
                    context_parts.append(f"{role}: {content}")
            context = "\n".join(context_parts)
        
        # Load topic detection prompt
        # Use simplified prompt for ollama models
        topic_model = config.get("topic_detection_model", "ollama/llama3")
        use_v3_prompt = config.get("topic_detection_v3", True)  # Default to v3
        
        if "ollama" in topic_model.lower() and prompt_manager.get("topic_detection_ollama"):
            prompt_name = "topic_detection_ollama"
        elif use_v3_prompt and prompt_manager.get("topic_detection_v3"):
            prompt_name = "topic_detection_v3"
        else:
            prompt_name = "topic_detection_v2" if use_v2_prompt else "topic_detection"
        
        topic_detection_prompt_content = prompt_manager.get(prompt_name)
        
        if topic_detection_prompt_content:
            # Use the loaded prompt template
            prompt_template = topic_detection_prompt_content
            # Format new message based on prompt version
            if use_v2_prompt:
                formatted_new_message = new_message  # V2 expects just the message
            else:
                formatted_new_message = f"user: {new_message}"  # V1 expects role prefix
            
            prompt = prompt_template.format(
                recent_conversation=context,
                new_message=formatted_new_message
            )
            
            if config.get("debug"):
                typer.echo(f"   Context preview: {context[:200]}...")
                typer.echo(f"   Prompt length: {len(prompt)} chars")
                typer.echo(f"   Full prompt:\n{prompt}\n   ---End prompt---")
        else:
            # Fallback to default prompt if file not found
            prompt = _get_fallback_detection_prompt(context, new_message)

        # Get the topic detection model from config (default to ollama/llama3)
        topic_model = config.get("topic_detection_model", "ollama/llama3")
        
        if config.get("debug"):
            typer.echo(f"\n🔍 DEBUG: Topic change detection")
            typer.echo(f"   Model: {topic_model}")
            typer.echo(f"   Recent messages: {len(recent_messages)}")
            typer.echo(f"   New message preview: {new_message[:100]}...")
        
        # Use configured model for detection with topic parameters
        topic_params = config.get_model_params('topic', model=topic_model)
        
        if config.get("debug"):
            typer.echo(f"   Topic params being used: {topic_params}")
        
        with benchmark_resource("LLM Call", f"topic detection - {topic_model}"):
            response, cost_info = query_llm(
                prompt, 
                model=topic_model,
                **topic_params
            )
        
        if response:
            return _parse_detection_response(response, cost_info)
    
    except Exception as e:
        logger.warning(f"Topic change detection error: {e}")
        if config.get("debug"):
            typer.echo(f"⚠️  Topic detection error: {e}")
        return False, None, None


def _get_fallback_detection_prompt(context: str, new_message: str) -> str:
    """Get fallback topic detection prompt."""
    return f"""Analyze if there is a MAJOR topic change in this conversation.

IMPORTANT RULES:
1. Only detect changes when switching to a COMPLETELY DIFFERENT domain of knowledge
2. Continuing to ask about the same subject, even with variations, is NOT a topic change
3. Default to "No" unless you are absolutely certain the topic has changed dramatically

Previous conversation:
{context}

New user message:
user: {new_message}

Has the topic changed to a COMPLETELY DIFFERENT subject?

Examples that should be "No":
- Asking more questions about the same subject
- Requesting clarification or more details
- Asking for examples of the same thing
- Variations on the same theme
- Related or connected topics

Only answer "Yes" for dramatic shifts like:
- Technical discussion → Personal life
- Science → Entertainment
- Programming → Food/Cooking
- Math → Travel plans

Respond with a JSON object containing only an "answer" field with value "Yes" or "No"."""


def _parse_detection_response(response: str, cost_info: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Parse the topic detection response."""
    response = response.strip()
    
    # Always log the raw response for debugging
    logger.info(f"Topic detection raw response: '{response}'")
    
    if config.get("debug"):
        typer.echo(f"   LLM response: {response}")
        typer.echo(f"   Response type: {type(response)}")
        typer.echo(f"   Response length: {len(response)}")
    
    try:
        # Parse JSON response
        result = json.loads(response)
        
        # Support both simple and intent-based formats
        if "intent" in result and "shift" in result:
            # New intent-based format
            intent = result.get("intent", "UNKNOWN")
            shift = result.get("shift", "NO")
            
            if config.get("debug"):
                typer.echo(f"   Parsed intent-based JSON: intent={intent}, shift={shift}")
            
            # Validate intent
            valid_intents = ["JUST_COMMENT", "DEVELOP_TOPIC", "INTRODUCE_TOPIC", "CHANGE_TOPIC"]
            if intent not in valid_intents:
                typer.echo(f"   ⚠️  Invalid intent: {intent}")
            
            if shift == "YES":
                if config.get("debug"):
                    typer.echo(f"   ✅ Topic change detected (intent: {intent})")
                return True, None, cost_info
            else:
                if config.get("debug"):
                    typer.echo(f"   ➡️ Continuing same topic (intent: {intent})")
                return False, None, cost_info
        
        else:
            # Simple format (backward compatibility)
            answer = result.get("answer", "No")
            
            if config.get("debug"):
                typer.echo(f"   Parsed simple JSON: answer={answer}")
            
            if answer == "Yes":
                if config.get("debug"):
                    typer.echo(f"   ✅ Topic change detected")
                return True, None, cost_info
            else:
                if config.get("debug"):
                    typer.echo(f"   ➡️ Continuing same topic")
                return False, None, cost_info
                
    except json.JSONDecodeError as e:
        # Fallback to more flexible parsing if JSON parsing fails
        if config.get("debug"):
            typer.echo(f"⚠️  Topic detection JSON parsing failed: {e}")
            typer.echo(f"   Raw response: {response}")
            typer.echo(f"   Falling back to text parsing")
        
        return _parse_detection_response_fallback(response, cost_info)


def _parse_detection_response_fallback(response: str, cost_info: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Fallback parsing for topic detection response."""
    # Try to extract answer from various formats
    response_lower = response.lower().strip()
    
    # Check for various positive responses
    if any(pattern in response_lower for pattern in ["yes", '"yes"', "'yes'", ": yes", "answer is yes"]):
        if config.get("debug"):
            typer.echo(f"   ✅ Topic change detected (fallback: found 'yes' in response)")
        return True, None, cost_info
    
    # Check for various negative responses  
    if any(pattern in response_lower for pattern in ["no", '"no"', "'no'", ": no", "answer is no"]):
        if config.get("debug"):
            typer.echo(f"   ➡️ Continuing same topic (fallback: found 'no' in response)")
        return False, None, cost_info
    
    # Try to fix common JSON errors and re-parse
    try:
        # Fix missing quotes around property names
        fixed_response = response.replace('{answer:', '{"answer":')
        fixed_response = fixed_response.replace('{shift:', '{"shift":')
        fixed_response = fixed_response.replace('{intent:', '{"intent":')
        fixed_response = fixed_response.replace(', answer:', ', "answer":')
        fixed_response = fixed_response.replace(', shift:', ', "shift":')
        fixed_response = fixed_response.replace(', intent:', ', "intent":')
        
        result = json.loads(fixed_response)
        
        # Check for answer in the fixed JSON
        if "answer" in result:
            answer = result.get("answer", "No")
            if answer == "Yes":
                if config.get("debug"):
                    typer.echo(f"   ✅ Topic change detected (fixed JSON)")
                return True, None, cost_info
            else:
                if config.get("debug"):
                    typer.echo(f"   ➡️ Continuing same topic (fixed JSON)")
                return False, None, cost_info
    except:
        pass
    
    # Default to no change if we can't parse the response
    if config.get("debug"):
        typer.echo(f"   ➡️ Continuing same topic (fallback: couldn't parse response)")
    return False, None, cost_info