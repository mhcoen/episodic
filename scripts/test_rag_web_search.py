#!/usr/bin/env python3
"""
Interactive test script for RAG and web search functionality.

This script provides a comprehensive test of the new RAG and web search features,
including document indexing, searching, and context enhancement.
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from episodic.db import initialize_db
from episodic.config import config
from episodic.cli import execute_script
import typer


def print_section(title: str):
    """Print a section header."""
    typer.echo()
    typer.secho(f"{'='*60}", fg="cyan")
    typer.secho(f"{title:^60}", fg="cyan", bold=True)
    typer.secho(f"{'='*60}", fg="cyan")
    typer.echo()


def test_rag_features():
    """Test RAG functionality."""
    print_section("Testing RAG Features")
    
    # Create test content
    test_files = {
        "python_basics.txt": """Python Programming Basics

Python is a high-level, interpreted programming language known for its simplicity 
and readability. It was created by Guido van Rossum and first released in 1991.

Key features of Python:
1. Easy to learn and use
2. Extensive standard library
3. Cross-platform compatibility
4. Strong community support
5. Versatile - used in web development, data science, AI, and more

Python uses indentation to define code blocks, making it visually clean.
""",
        "episodic_readme.txt": """Episodic Documentation

Episodic is a conversational DAG-based memory agent that stores conversations
as a directed acyclic graph. Each node represents a message exchange.

Key features:
- Persistent memory with SQLite storage
- Automatic topic detection using semantic analysis
- Multi-provider LLM support via LiteLLM
- Real-time visualization of conversation flow
- Cost tracking and API monitoring

The system uses a sliding window approach for topic detection, calculating
semantic drift between conversation segments.
""",
        "web_development.txt": """Modern Web Development

Web development has evolved significantly with frameworks like React, Vue, and Angular.
Modern web apps are built using component-based architectures and state management.

Key concepts:
- Single Page Applications (SPAs)
- REST APIs and GraphQL
- Responsive design
- Progressive Web Apps (PWAs)
- Server-side rendering (SSR)

JavaScript remains the foundation, but TypeScript adds type safety.
"""
    }
    
    # Write test files
    os.makedirs("test_documents", exist_ok=True)
    for filename, content in test_files.items():
        filepath = os.path.join("test_documents", filename)
        with open(filepath, 'w') as f:
            f.write(content)
        typer.secho(f"✅ Created test file: {filepath}", fg="green")
    
    # Create test script for RAG
    rag_script = """# RAG Feature Test Script

# Initialize and enable RAG
/init --erase
/rag on

# Index the test documents
/index test_documents/python_basics.txt
/index test_documents/episodic_readme.txt
/index test_documents/web_development.txt

# List indexed documents
/docs list

# Test searching
/search Python programming
/search topic detection
/search React framework

# Test context enhancement with questions
Tell me about Python's history
How does Episodic detect topics?
What are Progressive Web Apps?

# Show RAG statistics
/rag

# Test document management
/docs show 1
/docs remove test_documents/web_development.txt
/docs list

# Test with a query that won't match well
What is quantum computing used for?

# Show final stats
/rag
"""
    
    script_path = "test_rag_features.txt"
    with open(script_path, 'w') as f:
        f.write(rag_script)
    
    typer.secho(f"\n📝 Created RAG test script: {script_path}", fg="yellow")
    
    if typer.confirm("\nRun RAG test script?"):
        typer.secho("\n🚀 Running RAG tests...\n", fg="cyan")
        execute_script(script_path)
    
    # Cleanup
    if typer.confirm("\nClean up test files?"):
        import shutil
        shutil.rmtree("test_documents", ignore_errors=True)
        os.remove(script_path)
        typer.secho("✅ Cleaned up test files", fg="green")


def test_web_search_features():
    """Test web search functionality."""
    print_section("Testing Web Search Features")
    
    # Create web search test script
    web_script = """# Web Search Feature Test Script

# Initialize and enable web search
/init --erase
/websearch on

# Test direct web search
/websearch Python 3.12 new features
/ws latest AI developments 2024
/websearch climate change solutions

# Test configuration
/websearch config
/websearch stats

# Enable auto-enhancement
/set web_search_auto_enhance true

# Ask questions that should trigger web search
What are the latest features in Python 3.12?
Tell me about recent breakthroughs in quantum computing
What's the current status of electric vehicle adoption?

# Check cache performance
/websearch stats

# Test rate limiting (if we hit the limit)
/websearch OpenAI GPT-5 news
/websearch Meta Llama 3 updates
/websearch Google Gemini features

# Final statistics
/websearch stats
"""
    
    script_path = "test_web_search.txt"
    with open(script_path, 'w') as f:
        f.write(web_script)
    
    typer.secho(f"\n📝 Created web search test script: {script_path}", fg="yellow")
    
    if typer.confirm("\nRun web search test script?"):
        typer.secho("\n🚀 Running web search tests...\n", fg="cyan")
        execute_script(script_path)
    
    # Cleanup
    if typer.confirm("\nClean up test script?"):
        os.remove(script_path)
        typer.secho("✅ Cleaned up test script", fg="green")


def test_rag_web_integration():
    """Test RAG and web search integration."""
    print_section("Testing RAG + Web Search Integration")
    
    # Create integration test script
    integration_script = """# RAG + Web Search Integration Test

# Initialize with both features
/init --erase
/rag on
/websearch on

# Configure for integration
/set rag_auto_search true
/set web_search_auto_enhance true
/set web_search_index_results true

# Index some local content
/index --text "Episodic uses ChromaDB for vector storage and semantic search."
/index --text "The sliding window algorithm detects topic changes using embedding drift."

# Ask questions that use both local and web knowledge
How does Episodic's vector storage compare to other solutions?
What are the latest developments in vector databases?

# Ask about something not in our knowledge base
Tell me about the James Webb Space Telescope's recent discoveries

# Check what got indexed from web results
/docs list

# Search for web-sourced content
/search telescope discoveries

# Final statistics
/rag
/websearch stats
"""
    
    script_path = "test_integration.txt"
    with open(script_path, 'w') as f:
        f.write(integration_script)
    
    typer.secho(f"\n📝 Created integration test script: {script_path}", fg="yellow")
    
    if typer.confirm("\nRun integration test script?"):
        typer.secho("\n🚀 Running integration tests...\n", fg="cyan")
        execute_script(script_path)
    
    # Cleanup
    if typer.confirm("\nClean up test script?"):
        os.remove(script_path)
        typer.secho("✅ Cleaned up test script", fg="green")


def run_unit_tests():
    """Run the unit tests."""
    print_section("Running Unit Tests")
    
    typer.secho("Running RAG integration tests...", fg="yellow")
    os.system("python -m pytest tests/integration/test_rag_integration.py -v")
    
    typer.echo()
    typer.secho("Running web search integration tests...", fg="yellow")
    os.system("python -m pytest tests/integration/test_web_search_integration.py -v")


def main():
    """Main test runner."""
    typer.secho("\n🧪 Episodic RAG & Web Search Test Suite", fg="cyan", bold=True)
    typer.secho("=" * 50, fg="cyan")
    
    while True:
        typer.echo("\nSelect test to run:")
        typer.echo("1. Test RAG features")
        typer.echo("2. Test web search features")
        typer.echo("3. Test RAG + web search integration")
        typer.echo("4. Run unit tests")
        typer.echo("5. Run all tests")
        typer.echo("0. Exit")
        
        choice = typer.prompt("\nEnter choice", type=int)
        
        if choice == 0:
            typer.secho("\n👋 Goodbye!", fg="green")
            break
        elif choice == 1:
            test_rag_features()
        elif choice == 2:
            test_web_search_features()
        elif choice == 3:
            test_rag_web_integration()
        elif choice == 4:
            run_unit_tests()
        elif choice == 5:
            test_rag_features()
            test_web_search_features()
            test_rag_web_integration()
            run_unit_tests()
        else:
            typer.secho("Invalid choice!", fg="red")


if __name__ == "__main__":
    main()