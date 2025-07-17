#!/usr/bin/env python3
"""
Test script for tab completion functionality.
Tests the EpisodicCompleter without needing to run the full CLI.
"""

from prompt_toolkit import prompt
from prompt_toolkit.completion import Completion
from prompt_toolkit.document import Document

# Add episodic to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from episodic.cli_completer import EpisodicCompleter


def test_completions(line: str, cursor_pos: int = None):
    """Test what completions are generated for a given input line."""
    if cursor_pos is None:
        cursor_pos = len(line)
    
    completer = EpisodicCompleter()
    document = Document(line, cursor_pos)
    
    completions = list(completer.get_completions(document, None))
    
    print(f"\nInput: '{line}' (cursor at {cursor_pos})")
    print(f"Found {len(completions)} completions:")
    
    for comp in completions[:10]:  # Show first 10
        print(f"  - '{comp.text}' (start: {comp.start_position})")
        if comp.display:
            print(f"    Display: {comp.display}")
        if comp.display_meta:
            print(f"    Meta: {comp.display_meta}")


def test_interactive():
    """Run an interactive test with actual tab completion."""
    completer = EpisodicCompleter()
    
    print("Interactive tab completion test")
    print("Type commands and press TAB to see completions")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            # Use prompt_toolkit with our completer
            user_input = prompt("> ", completer=completer)
            
            if user_input.strip():
                print(f"You entered: {user_input}")
                
                # Parse and show what the completer understood
                if user_input.startswith('/'):
                    parts = user_input.split()
                    print(f"  Command: {parts[0][1:]}")
                    if len(parts) > 1:
                        print(f"  Args: {parts[1:]}")
            
    except KeyboardInterrupt:
        print("\nExiting...")


def run_tests():
    """Run a series of completion tests."""
    print("=== Tab Completion Tests ===")
    
    # Test command completion
    test_completions("/mo")
    test_completions("/set")
    test_completions("/")
    
    # Test alias completion
    test_completions("/ex")  # Should show /export alias
    test_completions("/im")  # Should show /import alias
    
    # Test model command completion
    test_completions("/model ")
    test_completions("/model ch")
    test_completions("/model chat ")
    test_completions("/model chat gpt")
    
    # Test set command completion
    test_completions("/set ")
    test_completions("/set deb")
    test_completions("/set debug ")
    
    # Test unified command completion
    test_completions("/topics ")
    test_completions("/topics ren")
    
    # Test file completion (current directory)
    test_completions("/import ")
    test_completions("/export ")
    
    # Test mset completion
    test_completions("/mset ")
    test_completions("/mset chat.")
    test_completions("/mset chat.temp")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test tab completion")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run interactive test")
    
    args = parser.parse_args()
    
    if args.interactive:
        test_interactive()
    else:
        run_tests()
        print("\n\nRun with --interactive to test actual tab completion")