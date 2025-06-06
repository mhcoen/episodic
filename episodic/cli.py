"""
Episodic Interactive Shell

This module provides an interactive command-line interface for the Episodic project
using prompt_toolkit. It allows users to interact with the conversation DAG through
a shell-like interface that maintains state between commands.

Features:
- Command history (persisted between sessions)
- Auto-completion for commands and arguments
- Syntax highlighting
- Help documentation for all commands
- State management between commands

Usage:
    episodic-shell

Or from Python:
    from episodic.cli import main
    main()
"""

import os
import sys
import shlex
import webbrowser
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers import BashLexer

from episodic.db import (
    insert_node, get_node, get_ancestry, initialize_db, 
    resolve_node_ref, get_head, set_head, database_exists
)
from episodic.llm import query_llm, query_with_context
from episodic.visualization import visualize_dag


def parse_command_line(text: str) -> List[str]:
    """
    Parse a command line without requiring quotation marks.

    This function splits the input by command and flags (--), treating everything
    between flags (or after the command until the first flag) as a single argument,
    regardless of spaces.

    Args:
        text: The command line text to parse

    Returns:
        A list of arguments, with the first element being the command
    """
    # Strip leading/trailing whitespace
    text = text.strip()

    if not text:
        return []

    # Split by whitespace to get the command
    parts = text.split(None, 1)
    command = parts[0]

    # If there's no additional text, return just the command
    if len(parts) == 1:
        return [command]

    # Get the rest of the text
    rest = parts[1].strip()

    # Initialize the result with the command
    result = [command]

    # Find all flag positions
    flag_positions = [(m.start(), m.group()) for m in re.finditer(r'\s--[\w-]+', rest)]

    # If there are no flags, the entire rest is one argument
    if not flag_positions:
        result.append(rest)
        return result

    # Process the text before the first flag
    if flag_positions[0][0] > 0:
        result.append(rest[:flag_positions[0][0]].strip())

    # Process each flag and the text until the next flag
    for i, (pos, flag) in enumerate(flag_positions):
        # Add the flag
        flag = flag.strip()
        result.append(flag)

        # Calculate the start and end of the text after this flag
        start = pos + len(flag)
        end = flag_positions[i+1][0] if i+1 < len(flag_positions) else len(rest)

        # Add the text after this flag and before the next flag (or end)
        if start < end:
            result.append(rest[start:end].strip())

    return result


class EpisodicCompleter(Completer):
    """
    Custom completer for Episodic commands and arguments.

    Provides context-aware completion for commands and their arguments.
    """

    def __init__(self):
        # Define all available commands
        self.commands = {
            'init': {
                'help': 'Initialize the database',
                'args': []
            },
            'add': {
                'help': 'Add a new node with content',
                'args': ['--parent']
            },
            'show': {
                'help': 'Show a specific node',
                'args': []  # Node ID is a positional argument
            },
            'ancestry': {
                'help': 'Show the ancestry of a node',
                'args': []  # Node ID is a positional argument
            },
            'query': {
                'help': 'Query an LLM and store the result',
                'args': ['--model', '--system', '--parent']
            },
            'chat': {
                'help': 'Chat with an LLM using conversation history',
                'args': ['--model', '--system', '--context-depth']
            },
            'visualize': {
                'help': 'Create an interactive visualization of the conversation DAG',
                'args': ['--output', '--no-browser']
            },
            'help': {
                'help': 'Show help for a command or list all commands',
                'args': []
            },
            'exit': {
                'help': 'Exit the shell',
                'args': []
            },
            'quit': {
                'help': 'Exit the shell',
                'args': []
            }
        }

        # Special completions for specific arguments
        self.arg_completions = {
            '--model': ['gpt-3.5-turbo', 'gpt-4'],
            '--parent': ['HEAD', 'HEAD~1', 'HEAD~2']  # Will be dynamically updated
        }

    def get_completions(self, document, complete_event):
        # Get the text before the cursor
        text_before_cursor = document.text_before_cursor

        # Split the input using our custom parser
        words = parse_command_line(text_before_cursor) if text_before_cursor else []

        # If we're at the start of a new word or there are no words yet
        if not text_before_cursor or text_before_cursor[-1].isspace():
            word_before_cursor = ''
            words.append('')
        else:
            word_before_cursor = words[-1]
            words[-1] = ''  # For checking the command context

        # If we're completing the command itself
        if len(words) == 1:
            for command in self.commands:
                if command.startswith(word_before_cursor):
                    help_text = self.commands[command]['help']
                    yield Completion(
                        command, 
                        start_position=-len(word_before_cursor),
                        display=command,
                        display_meta=help_text
                    )

        # If we're completing an argument for a command
        elif len(words) > 1 and words[0] in self.commands:
            command = words[0]

            # If the word starts with '--', it's an argument flag
            if word_before_cursor.startswith('--'):
                for arg in self.commands[command]['args']:
                    if arg.startswith(word_before_cursor):
                        yield Completion(
                            arg, 
                            start_position=-len(word_before_cursor),
                            display=arg
                        )

            # If the previous word is an argument that has specific completions
            elif len(words) > 2 and words[-2] in self.arg_completions:
                arg = words[-2]
                for value in self.arg_completions[arg]:
                    if value.startswith(word_before_cursor):
                        yield Completion(
                            value, 
                            start_position=-len(word_before_cursor),
                            display=value
                        )


class EpisodicShell:
    """
    Interactive shell for the Episodic project.

    Provides a command-line interface with history, auto-completion, and state management.
    """

    def __init__(self):
        # Set up history file in user's home directory
        history_file = os.path.join(str(Path.home()), '.episodic_history')

        # Create the prompt session with history and auto-completion
        self.session = PromptSession(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=EpisodicCompleter(),
            lexer=PygmentsLexer(BashLexer),
            style=Style.from_dict({
                'prompt': 'ansigreen bold',
            }),
            message=HTML('<ansigreen>episodic</ansigreen>> '),
        )

        # Initialize state
        self.current_node_id = None
        self.default_model = "gpt-3.5-turbo"
        self.default_system = "You are a helpful assistant."
        self.default_context_depth = 5

        # Command handlers
        self.handlers = {
            'init': self.handle_init,
            'add': self.handle_add,
            'show': self.handle_show,
            'ancestry': self.handle_ancestry,
            'query': self.handle_query,
            'chat': self.handle_chat,
            'visualize': self.handle_visualize,
            'help': self.handle_help,
            'exit': self.handle_exit,
            'quit': self.handle_exit,
        }

    def run(self):
        """Run the interactive shell."""
        print("Welcome to the Episodic interactive shell.")
        print("Type 'help' to see available commands or 'exit' to quit.")

        # Try to get the current head node
        try:
            self.current_node_id = get_head()
            if self.current_node_id:
                print(f"Current node: {self.current_node_id}")
        except:
            print("No database found. Use 'init' to create a new database.")

        while True:
            try:
                # Get input from the user
                text = self.session.prompt()

                # Skip empty input
                if not text.strip():
                    continue

                # Parse the input without requiring quotation marks
                args = parse_command_line(text)
                command = args[0].lower() if args else ""

                # Execute the command
                if command in self.handlers:
                    self.handlers[command](args[1:])
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' to see available commands.")

            except KeyboardInterrupt:
                # Handle Ctrl+C
                print("^C")
                continue
            except EOFError:
                # Handle Ctrl+D
                print("^D")
                self.handle_exit([])
                break
            except Exception as e:
                print(f"Error: {str(e)}")

    def handle_init(self, args):
        """Initialize the database."""
        if database_exists():
            try:
                # Print the question and use input() instead of self.session.prompt()
                print("The database already exists.\nDo you want to erase it? (yes/no): ", end="", flush=True)
                response = input().strip().lower()
                if response in ["yes", "y"]:
                    initialize_db(erase=True)
                    print("Database has been reinitialized.")
                else:
                    print("Database initialization cancelled.")
            except (KeyboardInterrupt, EOFError):
                print("\nDatabase initialization cancelled.")
        else:
            initialize_db()
            print("Database initialized.")

    def handle_add(self, args):
        """Add a new node with content."""
        if not args:
            print("Error: Content required")
            return

        # Parse arguments
        content = args[0]
        parent = None

        # Check for --parent flag
        if len(args) > 2 and args[1] == "--parent":
            parent = resolve_node_ref(args[2])

        # Insert the node
        node_id = insert_node(content, parent)
        set_head(node_id)
        self.current_node_id = node_id
        print(f"Added node {node_id}")

    def handle_show(self, args):
        """Show a specific node."""
        if not args:
            # If no node ID is provided, use the current node
            if not self.current_node_id:
                print("No current node. Specify a node ID or use 'add' to create a node.")
                return
            node_id = self.current_node_id
        else:
            node_id = resolve_node_ref(args[0])

        # Get and display the node
        node = get_node(node_id)
        if node:
            print(f"Node ID: {node['id']}")
            print(f"Parent: {node['parent_id']}")
            print(f"Message: {node['content']}")
        else:
            print("Node not found.")

    def handle_ancestry(self, args):
        """Show the ancestry of a node."""
        if not args:
            # If no node ID is provided, use the current node
            if not self.current_node_id:
                print("No current node. Specify a node ID or use 'add' to create a node.")
                return
            node_id = self.current_node_id
        else:
            node_id = resolve_node_ref(args[0])

        # Get and display the ancestry
        ancestry = get_ancestry(node_id)
        for ancestor in ancestry:
            print(f"{ancestor['id']}: {ancestor['content']}")

    def handle_query(self, args):
        """Query an LLM and store the result."""
        if not args:
            print("Error: Prompt required")
            return

        try:
            # Parse arguments
            prompt = args[0]
            model = self.default_model
            system = self.default_system
            parent = None

            # Process optional arguments
            i = 1
            while i < len(args):
                if args[i] == "--model" and i + 1 < len(args):
                    model = args[i + 1]
                    i += 2
                elif args[i] == "--system" and i + 1 < len(args):
                    system = args[i + 1]
                    i += 2
                elif args[i] == "--parent" and i + 1 < len(args):
                    parent = resolve_node_ref(args[i + 1])
                    i += 2
                else:
                    i += 1

            # If no parent is specified, use the current node
            if parent is None and self.current_node_id:
                parent = self.current_node_id

            # Store the user query as a node
            query_node_id = insert_node(prompt, parent)
            print(f"Added query node {query_node_id}")

            # Query the LLM
            response = query_llm(
                prompt=prompt,
                model=model,
                system_message=system
            )

            # Store the LLM response as a node with the query as its parent
            response_node_id = insert_node(response, query_node_id)
            print(f"Added response node {response_node_id}")

            # Update the current node
            self.current_node_id = response_node_id
            set_head(response_node_id)

            # Display the response
            print("\nLLM Response:")
            print(response)

        except Exception as e:
            print(f"Error: {str(e)}")

    def handle_chat(self, args):
        """Chat with an LLM using conversation history."""
        if not args:
            print("Error: Prompt required")
            return

        try:
            # Parse arguments
            prompt = args[0]
            model = self.default_model
            system = self.default_system
            context_depth = self.default_context_depth

            # Process optional arguments
            i = 1
            while i < len(args):
                if args[i] == "--model" and i + 1 < len(args):
                    model = args[i + 1]
                    i += 2
                elif args[i] == "--system" and i + 1 < len(args):
                    system = args[i + 1]
                    i += 2
                elif args[i] == "--context-depth" and i + 1 < len(args):
                    context_depth = int(args[i + 1])
                    i += 2
                else:
                    i += 1

            # Get the current head node
            head_id = self.current_node_id or get_head()
            if not head_id:
                # If there's no head node, we need to check if the database exists
                if database_exists():
                    # Database exists but no messages yet
                    print("No conversation history found. Use 'add' to add a message first.")
                    return
                else:
                    # Database doesn't exist yet
                    print("No conversation history found. Use 'init' to initialize the database and 'add' to add a message.")
                    return

            # Get the ancestry of the head node to use as context
            ancestry = get_ancestry(head_id)

            # Limit the context to the specified depth
            context_ancestry = ancestry[-context_depth:] if context_depth > 0 else ancestry

            # Convert the ancestry to the format expected by the LLM
            context_messages = []
            for i, node in enumerate(context_ancestry):
                # Skip the first node if it's a system message or has no parent
                if i == 0 and node['parent_id'] is None:
                    continue

                # Alternate between user and assistant roles
                role = "user" if i % 2 == 0 else "assistant"
                context_messages.append({"role": role, "content": node['content']})

            # Store the user query as a node
            query_node_id = insert_node(prompt, head_id)
            print(f"Added query node {query_node_id}")

            # Query the LLM with context
            response = query_with_context(
                prompt=prompt,
                context_messages=context_messages,
                model=model,
                system_message=system
            )

            # Store the LLM response as a node with the query as its parent
            response_node_id = insert_node(response, query_node_id)
            print(f"Added response node {response_node_id}")

            # Update the current node
            self.current_node_id = response_node_id
            set_head(response_node_id)

            # Display the response
            print("\nLLM Response:")
            print(response)

        except Exception as e:
            print(f"Error: {str(e)}")

    def handle_visualize(self, args):
        """Create an interactive visualization of the conversation DAG."""
        try:
            # Parse arguments
            output = None
            no_browser = False

            # Process optional arguments
            i = 0
            while i < len(args):
                if args[i] == "--output" and i + 1 < len(args):
                    output = args[i + 1]
                    i += 2
                elif args[i] == "--no-browser":
                    no_browser = True
                    i += 1
                else:
                    i += 1

            # Generate the visualization
            output_path = visualize_dag(output)
            if output_path and not no_browser:
                print(f"Opening visualization in browser: {output_path}")
                webbrowser.open(f"file://{os.path.abspath(output_path)}")
            elif output_path:
                print(f"Visualization saved to: {output_path}")

        except Exception as e:
            print(f"Error generating visualization: {str(e)}")

    def handle_help(self, args):
        """Show help for a command or list all commands."""
        if not args:
            # Show general help
            print("Available commands:")
            for cmd, info in sorted(self.session.completer.commands.items()):
                print(f"  {cmd:<12} - {info['help']}")
            print("\nType 'help <command>' for more information on a specific command.")
        else:
            # Show help for a specific command
            command = args[0].lower()
            if command in self.session.completer.commands:
                print(f"{command} - {self.session.completer.commands[command]['help']}")

                # Show arguments if any
                args_list = self.session.completer.commands[command]['args']
                if args_list:
                    print("\nArguments:")
                    for arg in args_list:
                        print(f"  {arg}")
            else:
                print(f"Unknown command: {command}")

    def handle_exit(self, args):
        """Exit the shell."""
        print("Goodbye!")
        sys.exit(0)


def main():
    """
    Main entry point for the Episodic interactive shell.

    This function creates and runs an instance of the EpisodicShell.
    """
    shell = EpisodicShell()
    shell.run()


if __name__ == "__main__":
    main()
