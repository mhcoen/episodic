#!/usr/bin/env python3
"""Test script to verify streaming functionality"""

from episodic.llm import query_with_context, process_stream_response
from episodic.config import config
import typer

def test_streaming():
    """Test the streaming functionality"""
    print("Testing streaming responses...\n")
    
    # Enable streaming
    config.set("stream_responses", True)
    
    # Create a simple test query
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 10 slowly, with a brief pause between each number."}
    ]
    
    # Test streaming
    print("Streaming response:")
    print("-" * 50)
    
    from episodic.llm import _execute_llm_query
    stream, _ = _execute_llm_query(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=200,
        stream=True
    )
    
    # Process and display the stream
    for chunk in process_stream_response(stream, "gpt-4o-mini"):
        print(chunk, end="", flush=True)
    
    print("\n" + "-" * 50)
    print("\nStreaming test complete!")
    
    # Now test non-streaming for comparison
    print("\n\nNon-streaming response:")
    print("-" * 50)
    
    response, cost_info = _execute_llm_query(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=200,
        stream=False
    )
    
    print(response)
    print("-" * 50)
    print(f"\nCost info: {cost_info}")

if __name__ == "__main__":
    test_streaming()