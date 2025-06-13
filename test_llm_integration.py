"""
Test script to demonstrate the LLM integration in Episodic.
This script requires an OpenAI API key to be set in the environment.

Usage:
    python test_llm_integration.py
"""

import os
import sys
from episodic.db import initialize_db, get_node, get_ancestry, get_head
from episodic.llm import query_llm, query_with_context

def test_llm_integration():
    """
    Test the LLM integration by sending a query and checking the response.
    """
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)

    print("== Testing LLM Integration ==")

    # Initialize the database
    print("\n== Initializing DB ==")
    initialize_db()

    # Test simple query
    print("\n== Testing simple query ==")
    try:
        prompt = "What is the capital of France?"
        print(f"Query: {prompt}")

        response, cost_info = query_llm(prompt)
        print(f"Response: {response}")
        print(f"Cost info: {cost_info}")

        print("\nTest passed: Successfully queried the LLM.")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)

    # Test query with context
    print("\n== Testing query with context ==")
    try:
        context_messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]

        prompt = "Tell me more about this city."
        print(f"Context: {context_messages}")
        print(f"Query: {prompt}")

        response, cost_info = query_with_context(prompt, context_messages)
        print(f"Response: {response}")
        print(f"Cost info: {cost_info}")

        print("\nTest passed: Successfully queried the LLM with context.")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)

    print("\n== All tests passed ==")

if __name__ == "__main__":
    test_llm_integration()
