"""
This module provides integration with OpenAI's API for the Episodic conversation system.
It handles sending queries to the API and processing the responses.
"""

import os
import openai
from typing import Dict, List, Optional, Any, Union

# Initialize the OpenAI client with API key from environment variable
def get_openai_client():
    """
    Initialize and return an OpenAI client using the API key from environment variables.
    
    Returns:
        OpenAI client object
    
    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is not set
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        )
    return openai.OpenAI(api_key=api_key)

def query_llm(
    prompt: str, 
    model: str = "gpt-3.5-turbo", 
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    """
    Send a query to the OpenAI API and return the response.
    
    Args:
        prompt: The user's query to send to the LLM
        model: The OpenAI model to use (default: gpt-3.5-turbo)
        system_message: The system message to set the context for the LLM
        temperature: Controls randomness (0-1, lower is more deterministic)
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        The LLM's response as a string
        
    Raises:
        Exception: If there's an error communicating with the OpenAI API
    """
    try:
        client = get_openai_client()
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error querying OpenAI API: {str(e)}")

def query_with_context(
    prompt: str,
    context_messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo",
    system_message: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    """
    Send a query to the OpenAI API with conversation context and return the response.
    
    Args:
        prompt: The user's query to send to the LLM
        context_messages: List of previous messages in the conversation
                         [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        model: The OpenAI model to use (default: gpt-3.5-turbo)
        system_message: The system message to set the context for the LLM
        temperature: Controls randomness (0-1, lower is more deterministic)
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        The LLM's response as a string
        
    Raises:
        Exception: If there's an error communicating with the OpenAI API
    """
    try:
        client = get_openai_client()
        
        # Prepare messages with system message first, then context, then the new prompt
        messages = [{"role": "system", "content": system_message}]
        messages.extend(context_messages)
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error querying OpenAI API with context: {str(e)}")