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
from prompt_toolkit import print_formatted_text
from pygments.lexers import BashLexer

from episodic.db import (
    insert_node, get_node, get_ancestry, initialize_db, 
    resolve_node_ref, get_head, set_head, database_exists,
    get_recent_nodes
)
from episodic.llm import query_llm, query_with_context
from litellm import cost_per_token
from episodic.llm_config import get_current_provider
from episodic.visualization import visualize_dag
from episodic.prompt_manager import PromptManager
from episodic.config import config


def parse_command_line(text: str) -> List[str]:
    """
    Parse a command line without requiring quotation marks.

    This function splits the input by command and flags (--), treating everything
    between flags (or after the command until the first flag) as a single argument,
    regardless of spaces.

    The function is designed to handle flags that appear in the middle of text without
    requiring proper whitespace separation. For example, the command:
    "chat Tell me my name --context-depth 10"
    will be correctly parsed as:
    ["chat", "Tell me my name", "--context-depth", "10"]

    This allows users to type natural language queries without worrying about
    quoting arguments that contain spaces.

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
    # Use a more robust regex that looks for '--' followed by word characters or hyphens
    # This will match flags even if they appear in the middle of text without proper whitespace
    flag_positions = []
    for m in re.finditer(r'(^|\s)--[\w-]+', rest):
        # We want to capture just the flag part (including the --), not any preceding whitespace
        start = m.start()
        if rest[start:start+1].isspace():
            start += 1  # Skip the whitespace
        flag_positions.append((start, rest[start:m.end()]))

    # If there are no flags, the entire rest is one argument
    if not flag_positions:
        result.append(rest)
        return result

    # Process the text before the first flag
    if flag_positions[0][0] > 0:
        result.append(rest[:flag_positions[0][0]].strip())

    # Process each flag and the text until the next flag
    for i, (pos, flag) in enumerate(flag_positions):
        # Add the flag (no need to strip as we've already handled whitespace)
        result.append(flag)

        # Calculate the start and end of the text after this flag
        start = pos + len(flag)
        end = flag_positions[i+1][0] if i+1 < len(flag_positions) else len(rest)

        # Add the text after this flag and before the next flag (or end)
        if start < end:
            # Extract and clean the text between this flag and the next one
            between_text = rest[start:end].strip()
            if between_text:
                result.append(between_text)

    return result


class EpisodicCompleter(Completer):
    """
    Custom completer for Episodic commands and arguments.

    Provides context-aware completion for commands and their arguments.
    """

    def __init__(self):
        # Define all available commands
        self.commands = {
            'dbinit': {
                'help': 'Initialize the database',
                'args': []
            },
            'add': {
                'help': 'Add a new node with content',
                'args': ['--parent']
            },
            'head': {
                'help': 'Display current node or change to specified node',
                'args': []  # Node ID is an optional positional argument
            },
            'print': {
                'help': 'Print node info (defaults to current node)',
                'args': []  # Node ID is an optional positional argument
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
                'args': ['--output', '--no-browser', '--port']
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
            },
            'list': {
                'help': 'List recent nodes',
                'args': ['--count']
            },
            'prompts': {
                'help': 'Manage system prompts',
                'args': []  # Subcommands are handled in the handler
            },
            'talk': {
                'help': 'Enter talk mode for seamless conversation with colored output',
                'args': ['--model', '--system', '--context-depth']
            },
            'debug': {
                'help': 'Set the debug flag to enable/disable debug output',
                'args': []
            },
            'llm': {
                'help': 'Manage LLM models',
                'args': ['list', 'model', 'add-local', 'local']
            },
            'set': {
                'help': 'Configure various parameters',
                'args': ['cost', 'depth']
            }
        }

        # Special completions for specific arguments
        self.arg_completions = {
            '--model': ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'],
            '--parent': ['HEAD', 'HEAD~1', 'HEAD~2'],  # Will be dynamically updated
            'debug': ['on', 'off', 'true', 'false'],
            'llm': ['list', 'model', 'add-local', 'local'],
            'set': ['cost', 'depth'],
            'cost': ['on', 'off', 'true', 'false'],
            'depth': ['3', '5', '10', '20', '50']
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
            # The last word might be incomplete, so we need to get it from the original text
            # rather than from the parsed words (which might have been split differently)
            word_before_cursor = text_before_cursor.split()[-1]
            # For checking the command context, we need to remove the last word
            if len(words) > 0:
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
        self.default_model = "gpt-4o-mini"

        # Initialize session cost tracking
        self.session_costs = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0
        }

        # Get the system message from the active prompt
        try:
            manager = PromptManager()
            self.default_system = manager.get_active_prompt_content(config.get)
        except Exception:
            # Fallback to default if there's an error
            self.default_system = "You are a helpful assistant."

        self.default_context_depth = 5

        # Command handlers
        self.handlers = {
            'dbinit': self.handle_db_init,
            'add': self.handle_add,
            'head': self.handle_head,
            'print': self.handle_print,
            'show': self.handle_show,
            'ancestry': self.handle_ancestry,
            'query': self.handle_query,
            'chat': self.handle_chat,
            'talk': self.handle_talk,
            'visualize': self.handle_visualize,
            'help': self.handle_help,
            'exit': self.handle_exit,
            'quit': self.handle_exit,
            'list': self.handle_list,
            'prompts': self.handle_prompts,
            'debug': self.handle_debug,
            'llm': self.handle_llm_providers,
            'set': self.handle_set,
        }

    def run(self):
        """Run the interactive shell."""
        print("Welcome to the Episodic interactive shell.")
        print("Type 'help' to see available commands or 'exit' to quit.")
        print("You can also use the '/' prefix for commands, e.g., '/help'.")

        # Print the active prompt
        try:
            manager = PromptManager()
            active_prompt = config.get("active_prompt", "default")
            metadata = manager.get_metadata(active_prompt)
            description = f" - {metadata.get('description')}" if metadata and 'description' in metadata else ''
            print(f"Active prompt: {active_prompt}{description}")
        except Exception as e:
            print("Active prompt: default")

        # Try to get the current head node
        try:
            self.current_node_id = get_head()
            if self.current_node_id:
                # Get the node to access its short ID
                node = get_node(self.current_node_id)
                if node:
                    print(f"Current node: {node['short_id']} (UUID: {node['id']})")
                else:
                    print(f"Current node: {self.current_node_id}")
        except:
            print("No database found. Use 'init' to create a new database.")

        # Print the current model
        try:
            from episodic.llm_config import get_current_provider, get_default_model, set_default_model
            # Get the current model
            current_model = get_default_model()
            # Set the default model to ensure proper initialization
            try:
                set_default_model(current_model)
                # Update the default model in the shell
                self.default_model = current_model
            except ValueError:
                # If there's an error setting the model, just continue
                pass
            # Get the current provider and model after initialization
            current_provider = get_current_provider()
            current_model = get_default_model()
            print(f"Current model: {current_model} (Provider: {current_provider})")
        except Exception as e:
            # If there's an error, just skip printing the model
            pass

        while True:
            try:
                # Get input from the user
                text = self.session.prompt()

                # Skip empty input
                if not text.strip():
                    continue

                # Check if the input starts with a slash (/)
                if text.startswith('/'):
                    # Remove the slash and get the command
                    text = text[1:].strip()

                # Parse the input without requiring quotation marks
                args = parse_command_line(text)
                command = args[0].lower() if args else ""

                # Execute the command
                if command in self.handlers:
                    self.handlers[command](args[1:])
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' or '/help' to see available commands.")

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

    def handle_db_init(self, args):
        """Initialize the database."""
        if database_exists():
            try:
                # Print the question and use input() instead of self.session.prompt()
                print("The database already exists.\nDo you want to erase it? (yes/no): ", end="", flush=True)
                response = input().strip().lower()
                if response in ["yes", "y"]:
                    result = initialize_db(erase=True)
                    if result:
                        root_node_id, root_short_id = result
                        self.current_node_id = root_node_id
                        # Reset the active prompt to default
                        config.set("active_prompt", "default")
                        # Update the default system message
                        try:
                            manager = PromptManager()
                            self.default_system = manager.get_active_prompt_content(config.get)
                        except Exception:
                            # Fallback to default if there's an error
                            self.default_system = "You are a helpful assistant."
                        print(f"Database has been reinitialized with a default root node (ID: {root_short_id}, UUID: {root_node_id}).")
                        print("Prompt has been restored to default.")
                    else:
                        print("Database has been reinitialized.")
                else:
                    print("Database initialization cancelled.")
            except (KeyboardInterrupt, EOFError):
                print("\nDatabase initialization cancelled.")
        else:
            result = initialize_db()
            if result:
                root_node_id, root_short_id = result
                self.current_node_id = root_node_id
                # Reset the active prompt to default for new databases too
                config.set("active_prompt", "default")
                # Update the default system message
                try:
                    manager = PromptManager()
                    self.default_system = manager.get_active_prompt_content(config.get)
                except Exception:
                    # Fallback to default if there's an error
                    self.default_system = "You are a helpful assistant."
                print(f"Database initialized with a default root node (ID: {root_short_id}, UUID: {root_node_id}).")
                print("Prompt has been set to default.")
            else:
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

        # Insert the node with "user" role
        node_id, short_id = insert_node(content, parent, role="user")
        set_head(node_id)
        self.current_node_id = node_id
        print(f"Added node {short_id} (UUID: {node_id})")

    def handle_print(self, args):
        """Print node info (defaults to current node)."""
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
            print(f"Node ID: {node['short_id']} (UUID: {node['id']})")
            if node['parent_id']:
                parent = get_node(node['parent_id'])
                parent_short_id = parent['short_id'] if parent else "Unknown"
                print(f"Parent: {parent_short_id} (UUID: {node['parent_id']})")
            else:
                print(f"Parent: None")
            print(f"Role: {node['role'] or 'None'}")
            print(f"Message: {node['content']}")
        else:
            print("Node not found.")

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
            print(f"Node ID: {node['short_id']} (UUID: {node['id']})")
            if node['parent_id']:
                parent = get_node(node['parent_id'])
                parent_short_id = parent['short_id'] if parent else "Unknown"
                print(f"Parent: {parent_short_id} (UUID: {node['parent_id']})")
            else:
                print(f"Parent: None")
            print(f"Role: {node['role'] or 'None'}")
            print(f"Message: {node['content']}")
        else:
            print("Node not found.")

    def handle_head(self, args):
        """Display current node or change to specified node."""
        if not args:
            # If no node ID is provided, display the current node's info
            if not self.current_node_id:
                print("No current node. Specify a node ID or use 'add' to create a node.")
                return

            # Get and display the node using a slightly different format to indicate it's the current node
            node = get_node(self.current_node_id)
            if node:
                print(f"Current node: {node['short_id']} (UUID: {node['id']})")
                if node['parent_id']:
                    parent = get_node(node['parent_id'])
                    parent_short_id = parent['short_id'] if parent else "Unknown"
                    print(f"Parent: {parent_short_id} (UUID: {node['parent_id']})")
                else:
                    print(f"Parent: None")
                print(f"Role: {node['role'] or 'None'}")
                print(f"Message: {node['content']}")
            else:
                print("Node not found.")
            return

        # Resolve the node ID
        node_id = resolve_node_ref(args[0])

        # Verify that the node exists
        node = get_node(node_id)
        if not node:
            print(f"Error: Node not found: {args[0]}")
            return

        # Update the current node
        self.current_node_id = node_id
        set_head(node_id)

        # Display confirmation
        print(f"Current node changed to: {node['short_id']} (UUID: {node['id']})")

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
            print(f"{ancestor['short_id']} (UUID: {ancestor['id']}): {ancestor['content']}")

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

            # Store the user query as a node with "user" role
            query_node_id, query_short_id = insert_node(prompt, parent, role="user")
            print(f"Added query node {query_short_id} (UUID: {query_node_id})")

            # Query the LLM
            response, cost_info = query_llm(
                prompt=prompt,
                model=model,
                system_message=system
            )

            # Store the LLM response as a node with the query as its parent and "assistant" role
            response_node_id, response_short_id = insert_node(response, query_node_id, role="assistant")
            print(f"Added response node {response_short_id} (UUID: {response_node_id})")

            # Update the current node
            self.current_node_id = response_node_id
            set_head(response_node_id)

            # Display the response with model information
            print("\nLLM Response:")
            # Get the current provider to display along with the model
            from episodic.llm_config import get_current_provider
            provider = get_current_provider()

            # Display model info with cost information on the same line if enabled
            if config.get("show_cost", False):
                print(f"\033[36m🤖 {provider}/{model}: ({cost_info['input_tokens']}_in + {cost_info['output_tokens']}_out = {cost_info['total_tokens']}_tokens ${cost_info['cost_usd']:.6f} USD)\033[0m")
            else:
                print(f"\033[36m🤖 {provider}/{model}:\033[0m")

            print(response)

            # Update session cost totals
            self.session_costs["total_input_tokens"] += cost_info["input_tokens"]
            self.session_costs["total_output_tokens"] += cost_info["output_tokens"]
            self.session_costs["total_tokens"] += cost_info["total_tokens"]
            self.session_costs["total_cost_usd"] += cost_info["cost_usd"]

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
                    # Database exists but no messages yet - this should be rare now with the implicit root node
                    print("No conversation history found. This is unusual since initialization should create a root node.")
                    print("Try reinitializing the database with 'init' or add a message with 'add'.")
                    return
                else:
                    # Database doesn't exist yet
                    print("No database found. Please initialize the database with 'init' command first.")
                    print("Initialization will create a default root node that can be used for conversation.")
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

                # Use the stored role if available, otherwise fall back to alternating roles
                role = node.get('role')
                if role is None:
                    # Fallback to alternating roles if role is not stored
                    role = "user" if i % 2 == 0 else "assistant"
                context_messages.append({"role": role, "content": node['content']})

            # Store the user query as a node with "user" role
            query_node_id, query_short_id = insert_node(prompt, head_id, role="user")
            print(f"Added query node {query_short_id} (UUID: {query_node_id})")

            # Query the LLM with context
            response, cost_info = query_with_context(
                prompt=prompt,
                context_messages=context_messages,
                model=model,
                system_message=system
            )

            # Store the LLM response as a node with the query as its parent and "assistant" role
            response_node_id, response_short_id = insert_node(response, query_node_id, role="assistant")
            print(f"Added response node {response_short_id} (UUID: {response_node_id})")

            # Update the current node
            self.current_node_id = response_node_id
            set_head(response_node_id)

            # Display the response with model information
            print("\nLLM Response:")
            # Get the current provider to display along with the model
            from episodic.llm_config import get_current_provider
            provider = get_current_provider()

            # Display model info with cost information on the same line if enabled
            if config.get("show_cost", False):
                print(f"\033[36m🤖 {provider}/{model}: ({cost_info['input_tokens']}_in + {cost_info['output_tokens']}_out = {cost_info['total_tokens']}_tokens ${cost_info['cost_usd']:.6f} USD)\033[0m")
            else:
                print(f"\033[36m🤖 {provider}/{model}:\033[0m")

            print(response)

            # Update session cost totals
            self.session_costs["total_input_tokens"] += cost_info["input_tokens"]
            self.session_costs["total_output_tokens"] += cost_info["output_tokens"]
            self.session_costs["total_tokens"] += cost_info["total_tokens"]
            self.session_costs["total_cost_usd"] += cost_info["cost_usd"]

        except Exception as e:
            print(f"Error: {str(e)}")

    def handle_talk(self, args):
        """
        Enter a conversation mode for seamless interaction with the LLM.
        Exit by typing 'done', 'exit', or pressing Ctrl+D.
        """
        try:
            # Parse arguments
            model = self.default_model
            system = self.default_system
            context_depth = self.default_context_depth

            # Process optional arguments
            i = 0
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
                if database_exists():
                    print("No conversation history found. Try initializing the database with 'init' or adding a message with 'add'.")
                    return
                else:
                    print("No database found. Please initialize the database with 'init' command first.")
                    return

            print("\nEntering talk mode. Type 'done', 'exit', or press Ctrl+D to return to the main CLI.\n")

            # Create a custom prompt session for the talk mode
            from prompt_toolkit import PromptSession
            from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

            talk_session = PromptSession(
                message=HTML("<ansigreen>> </ansigreen>"),
                history=self.session.history,
                auto_suggest=AutoSuggestFromHistory(),
            )

            while True:
                try:
                    # Get user input
                    user_input = talk_session.prompt()

                    # Check for exit commands
                    if user_input.lower() in ["done", "exit"]:
                        print("Exiting talk mode.")
                        break

                    # Skip empty input
                    if not user_input.strip():
                        continue

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

                        # Use the stored role if available, otherwise fall back to alternating roles
                        role = node.get('role')
                        if role is None:
                            # Fallback to alternating roles if role is not stored
                            role = "user" if i % 2 == 0 else "assistant"
                        context_messages.append({"role": role, "content": node['content']})

                    # Store the user query as a node with "user" role
                    query_node_id, query_short_id = insert_node(user_input, head_id, role="user")

                    # Query the LLM with context
                    response, cost_info = query_with_context(
                        prompt=user_input,
                        context_messages=context_messages,
                        model=model,
                        system_message=system
                    )

                    # Store the LLM response as a node with the query as its parent and "assistant" role
                    response_node_id, response_short_id = insert_node(response, query_node_id, role="assistant")

                    # Update the current node and head
                    self.current_node_id = response_node_id
                    set_head(response_node_id)

                    # Display the response with colored formatting
                    print("")  # Empty line before response
                    # Get the current provider to display along with the model
                    from episodic.llm_config import get_current_provider
                    provider = get_current_provider()

                    # Display model info with cost information on the same line if enabled
                    if config.get("show_cost", False):
                        print(f"\033[36m🤖 {provider}/{model}: ({cost_info['input_tokens']}_in + {cost_info['output_tokens']}_out = {cost_info['total_tokens']}_tokens ${cost_info['cost_usd']:.6f} USD)\033[0m")
                    else:
                        print(f"\033[36m🤖 {provider}/{model}:\033[0m")

                    print(f"\033[33m{response}\033[0m")

                    # Update session cost totals
                    self.session_costs["total_input_tokens"] += cost_info["input_tokens"]
                    self.session_costs["total_output_tokens"] += cost_info["output_tokens"]
                    self.session_costs["total_tokens"] += cost_info["total_tokens"]
                    self.session_costs["total_cost_usd"] += cost_info["cost_usd"]

                    print("")  # Empty line after response

                    # Update head_id for the next iteration
                    head_id = response_node_id

                except KeyboardInterrupt:
                    print("\n^C")
                    continue
                except EOFError:
                    print("\n^D")
                    print("Exiting talk mode.")
                    break
                except Exception as e:
                    print("")  # Empty line before error
                    print(f"\033[31mError: {str(e)}\033[0m")

        except Exception as e:
            print(f"Error: {str(e)}")

    def handle_visualize(self, args):
        """Create an interactive visualization of the conversation DAG."""
        try:
            # Parse arguments
            output = None
            no_browser = False
            port = 5000  # Default port

            # Process optional arguments
            i = 0
            while i < len(args):
                if args[i] == "--output" and i + 1 < len(args):
                    output = args[i + 1]
                    i += 2
                elif args[i] == "--no-browser":
                    no_browser = True
                    i += 1
                elif args[i] == "--port" and i + 1 < len(args):
                    try:
                        port = int(args[i + 1])
                        i += 2
                    except ValueError:
                        print(f"Error: Invalid port value: {args[i + 1]}")
                        return
                else:
                    i += 1

            # Generate the visualization
            output_path = visualize_dag(output)

            # If interactive mode is requested, start the server with the specified port
            if not no_browser:
                from episodic.server import start_server, stop_server
                server_url = start_server(server_port=port)
                print(f"Starting visualization server at {server_url}")
                print("Press Ctrl+C when done to stop the server.")
                webbrowser.open(server_url)

                try:
                    # Keep the server running until the user presses Ctrl+C
                    while True:
                        import time
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nStopping server...")
                    stop_server()
                    print("Server stopped.")
            elif output_path:
                print(f"Visualization saved to: {output_path}")
                print(f"Opening visualization in browser: {output_path}")
                webbrowser.open(f"file://{os.path.abspath(output_path)}")

        except Exception as e:
            print(f"Error generating visualization: {str(e)}")

    def handle_help(self, args):
        """Show help for a command or list all commands."""
        if not args:
            # Show general help
            print("Available commands:")
            for cmd, info in sorted(self.session.completer.commands.items()):
                print(f"  {cmd:<12} - {info['help']}")
            print("\nCommands can also be prefixed with a slash, e.g., '/help', '/exit'.")
            print("Type 'help <command>' or '/help <command>' for more information on a specific command.")
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

    def handle_list(self, args):
        """List recent nodes."""
        # Default count is 5
        count = 5

        # Parse arguments
        i = 0
        while i < len(args):
            if args[i] == "--count" and i + 1 < len(args):
                try:
                    count = int(args[i + 1])
                    i += 2
                except ValueError:
                    print(f"Error: Invalid count value: {args[i + 1]}")
                    return
            else:
                i += 1

        # Get recent nodes
        try:
            nodes = get_recent_nodes(count)

            if not nodes:
                print("No nodes found in the database.")
                return

            print(f"Recent nodes (showing {len(nodes)} of {count} requested):")
            for node in nodes:
                # Truncate content for display
                content = node['content']
                if len(content) > 50:
                    content = content[:47] + "..."

                # Display node information
                print(f"{node['short_id']} (UUID: {node['id']}): {content}")

        except Exception as e:
            print(f"Error retrieving recent nodes: {str(e)}")

    def handle_prompts(self, args):
        """Manage system prompts."""
        # Create a prompt manager instance
        manager = PromptManager()

        if not args:
            # If no subcommand is provided, show help
            print("Usage: prompts <subcommand>")
            print("\nAvailable subcommands:")
            print("  list - List all available prompts")
            print("  use <name> - Set the active prompt")
            print("  show [<name>] - Show the content of a prompt (defaults to active prompt)")
            print("  id - Show the current active prompt")
            print("  reload - Reload prompts from disk")
            return

        # Get the first word as the subcommand
        subcommand_parts = args[0].lower().split()
        subcommand = subcommand_parts[0]

        # If the subcommand has additional words, treat them as arguments
        if len(subcommand_parts) > 1:
            # Add the remaining words as separate arguments
            args = [subcommand] + subcommand_parts[1:] + args[1:]
        else:
            # Keep the original args but with the first element replaced by the subcommand
            args[0] = subcommand

        if subcommand == "list":
            # List all available prompts
            prompts = manager.list()
            if not prompts:
                print("No prompts found.")
                return

            print("Available prompts:")
            for name in prompts:
                metadata = manager.get_metadata(name)
                description = metadata.get('description', '') if metadata else ''
                print(f"  - {name}: {description}")

        elif subcommand == "use":
            # Check if a name is provided
            if len(args) < 2:
                print("Error: Prompt name required")
                print("Usage: prompts use <name>")
                return

            name = args[1]

            # Set the active prompt
            if name not in manager.list():
                print(f"Prompt '{name}' not found.")
                return

            # Store the active prompt name in config
            config.set("active_prompt", name)

            # Display confirmation and description if available
            metadata = manager.get_metadata(name)
            description = f" - {metadata.get('description')}" if metadata and 'description' in metadata else ''
            print(f"Now using prompt: {name}{description}")

        elif subcommand == "show":
            # Determine which prompt to show
            name = args[1] if len(args) > 1 else config.get("active_prompt", "default")

            # Get the prompt content
            prompt = manager.get(name)
            if not prompt:
                print(f"Prompt '{name}' not found.")
                return

            # Display the prompt information
            metadata = manager.get_metadata(name)
            description = f" - {metadata.get('description')}" if metadata and 'description' in metadata else ''

            print(f"--- Prompt: {name}{description} ---")
            print(prompt)

        elif subcommand == "id":
            # Show the current active prompt
            active_prompt = config.get("active_prompt", "default")
            metadata = manager.get_metadata(active_prompt)
            description = f" - {metadata.get('description')}" if metadata and 'description' in metadata else ''
            print(f"Current active prompt: {active_prompt}{description}")

        elif subcommand == "reload":
            # Reload prompts from disk
            manager.reload()
            print(f"Reloaded {len(manager.list())} prompts.")

        else:
            print(f"Unknown subcommand: {subcommand}")
            print("Available subcommands: list, use, show, id, reload")

    def handle_exit(self, args):
        """Exit the shell."""
        # Display total cost information for the session
        if self.session_costs["total_tokens"] > 0:
            print("\nSession Cost Information:")
            print(f"Total input tokens: {self.session_costs['total_input_tokens']}")
            print(f"Total output tokens: {self.session_costs['total_output_tokens']}")
            print(f"Total tokens: {self.session_costs['total_tokens']}")
            print(f"Total cost: ${self.session_costs['total_cost_usd']:.6f} USD")

        print("Goodbye!")
        sys.exit(0)

    def handle_debug(self, args):
        """
        Set the debug flag to enable/disable debug output.

        Usage:
            debug on|true    - Enable debug output
            debug off|false  - Disable debug output
            debug            - Show current debug status
        """
        if not args:
            # Show current debug status
            debug_enabled = config.get("debug", False)
            print(f"Debug mode is currently {'enabled' if debug_enabled else 'disabled'}")
            return

        value = args[0].lower()
        if value in ["on", "true"]:
            config.set("debug", True)
            print("Debug mode enabled")
        elif value in ["off", "false"]:
            config.set("debug", False)
            print("Debug mode disabled")
        else:
            print("Invalid argument. Use 'on', 'true', 'off', or 'false'")

    def handle_set(self, args):
        """
        Configure various parameters.

        Usage:
            set                  - Show all configurable parameters and their current values
            set cost on|off      - Enable/disable displaying cost information for LLM queries
            set depth <number>   - Set the default context depth for chat/talk commands
        """
        if not args:
            # Show all configurable parameters and their current values
            show_cost = config.get("show_cost", False)
            print(f"cost: {'on' if show_cost else 'off'} - Display cost information for LLM queries")
            print(f"depth: {self.default_context_depth} - Default context depth for chat/talk commands")
            return

        # Extract the parameter and any additional arguments
        param_parts = args[0].split()
        param = param_parts[0].lower()

        # If the parameter has additional parts, add them back to args
        if len(param_parts) > 1:
            args = [param] + param_parts[1:] + args[1:]

        # Handle 'cost' parameter
        if param == "cost":
            if len(args) < 2:
                show_cost = config.get("show_cost", False)
                print(f"Cost display is currently {'enabled' if show_cost else 'disabled'}")
                return

            value = args[1].lower()
            if value in ["on", "true"]:
                config.set("show_cost", True)
                print("Cost display enabled")
            elif value in ["off", "false"]:
                config.set("show_cost", False)
                print("Cost display disabled")
            else:
                print("Invalid value for cost. Use 'on', 'off', 'true', or 'false'")

        # Handle 'depth' parameter
        elif param == "depth":
            if len(args) < 2:
                print(f"Current default context depth: {self.default_context_depth}")
                return

            try:
                depth = int(args[1])
                if depth < 0:
                    print("Context depth must be a non-negative integer")
                else:
                    self.default_context_depth = depth
                    print(f"Default context depth set to {depth}")
            except ValueError:
                print("Invalid value for depth. Please provide a non-negative integer")

        # Handle unknown parameter
        else:
            print(f"Unknown parameter: {param}")
            print("Available parameters: cost, depth")
            print("Use 'set' without arguments to see all parameters and their current values")

    def handle_llm_providers(self, args):
        """
        Manage LLM models and view available models by provider.
        Usage:
          llm list                  - List all available models by provider
          llm model <model_name>    - Switch to a specific model
          llm add-local <name> <path> <backend> - Add a local model
          llm local <provider>      - Switch to a local provider (ollama or lmstudio)
        """
        from episodic.llm_config import (
            get_current_provider, set_current_provider, 
            get_available_providers, add_local_model,
            get_provider_models, get_provider_config,
            get_default_model, set_default_model,
            load_config, save_config,
            has_api_key, get_providers_with_api_keys,
            LOCAL_PROVIDERS
        )

        if not args:
            # Get and display the current model before showing usage information
            current_provider = get_current_provider()
            current_model = get_default_model()
            print(f"Current model: {current_model} (Provider: {current_provider})")
            print("")

            # Display usage information
            print("Usage:")
            print("  llm list                  - List all available models by provider")
            print("  llm model <model_name>    - Switch to a specific model")
            print("  llm add-local <name> <path> <backend> - Add a local model")
            print("  llm local <provider>      - Switch to a local provider (ollama or lmstudio)")
            return

        # Extract the subcommand and any additional arguments
        subcommand_parts = args[0].split()
        subcommand = subcommand_parts[0]

        # If the subcommand has additional parts, add them back to args
        if len(subcommand_parts) > 1:
            args = [subcommand] + subcommand_parts[1:] + args[1:]

        if subcommand == "list":
            # List all providers and their models
            current_provider = get_current_provider()
            current_model = get_default_model()
            providers = get_available_providers()

            # Get providers with API keys
            providers_with_api_keys = get_providers_with_api_keys()

            print(f"Current model: {current_model}")
            print(f"Available LLM models by provider:")
            for provider, details in providers.items():
                # Only mark the provider as current if it's the provider of the current model
                marker = "*" if provider == current_provider else " "

                # Check if the provider has an API key
                has_key = providers_with_api_keys.get(provider, False)
                api_key_status = ""
                if provider not in LOCAL_PROVIDERS and not has_key:
                    api_key_status = " (Not selectable - missing API key)"

                print(f"{marker} {provider}{api_key_status}:")

                # Print models for this provider
                models = details.get("models", [])
                if models:
                    if provider == "local":
                        # Local models have name and path
                        for model in models:
                            model_name = model.get("name")
                            # Mark the model as current if it's the current model
                            model_marker = "*" if model_name == current_model else " "
                            print(f"  {model_marker} {model_name} (path: {model.get('path')}, backend: {model.get('backend', 'llama.cpp')})")
                    else:
                        # Cloud models are just strings
                        for model in models:
                            # Mark the model as current if it's the current model
                            model_marker = "*" if model == current_model else " "

                            # Get cost information using LiteLLM's cost_per_token function
                            model_with_provider = f"{provider}/{model}"
                            cost_info = ""
                            try:
                                # Calculate cost for 1000 tokens (to get cost per 1K tokens)
                                input_cost, output_cost = cost_per_token(
                                    model=model_with_provider,
                                    prompt_tokens=1000,
                                    completion_tokens=1000
                                )
                                # Multiply by 1000 to get cost per 1K tokens
                                input_cost *= 1000
                                output_cost *= 1000
                                cost_info = f"(${input_cost:.4f}/1K input, ${output_cost:.4f}/1K output tokens)"
                            except Exception:
                                # If cost calculation fails, don't show cost information
                                pass

                            # Use a fixed width for the model name to align the cost information
                            print(f"  {model_marker} {model:<20}\t{cost_info}")
                else:
                    print("  (No models configured)")

                # Print additional provider details if available
                if provider == "lmstudio" and "api_base" in details:
                    print(f"  API Base: {details['api_base']}")

        elif subcommand == "model" and len(args) > 1:
            # Switch to a specific model
            model_name = args[1]
            try:
                set_default_model(model_name)
                print(f"Switched to model: {model_name}")
                # Update the default model in the shell
                self.default_model = model_name
            except ValueError as e:
                print(f"Error: {str(e)}")
                print("Use 'llm list' to see available models")

        elif subcommand == "switch" and len(args) > 1:
            # Redirect to model command for better user experience
            provider_or_model = args[1]

            # Check if it's a model name
            is_model = False
            model_name = provider_or_model
            providers = get_available_providers()

            for p_name, p_details in providers.items():
                models = p_details.get("models", [])
                if isinstance(models, list):
                    for model in models:
                        if (isinstance(model, dict) and model.get("name") == model_name) or \
                           (isinstance(model, str) and model == model_name):
                            is_model = True
                            break
                if is_model:
                    break

            if is_model:
                # It's a model name, so use the model command
                try:
                    set_default_model(model_name)
                    print(f"Switched to model: {model_name}")
                    # Update the default model in the shell
                    self.default_model = model_name
                except ValueError as e:
                    print(f"Error: {str(e)}")
                    print("Use 'llm list' to see available models")
            else:
                # It's not a model name, suggest using the model command instead
                print(f"The 'switch' command is deprecated. Please use 'llm model <model_name>' instead.")
                print(f"If you're trying to use a model from {provider_or_model}, use 'llm list' to see available models.")

        elif subcommand == "add-local" and len(args) > 3:
            # Add a local model
            name = args[1]
            path = args[2]
            backend = args[3]
            try:
                add_local_model(name, path, backend)
                print(f"Added local model: {name} (path: {path}, backend: {backend})")
            except Exception as e:
                print(f"Error adding local model: {str(e)}")

        elif subcommand == "local" and len(args) > 1:
            # Switch to a model from a local provider (ollama or lmstudio)
            local_provider = args[1]
            if local_provider in ["ollama", "lmstudio"]:
                try:
                    # Get the available models for this provider
                    provider_models = get_provider_models(local_provider)
                    if provider_models:
                        # Get the first model (handle both string and dict models)
                        if isinstance(provider_models[0], dict):
                            default_model = provider_models[0].get("name")
                        else:
                            default_model = provider_models[0]

                        # Use set_default_model to switch to this model
                        set_default_model(default_model)
                        # Update the default model in the shell
                        self.default_model = default_model

                        print(f"Switched to model: {default_model} from {local_provider}")

                        # Show available models for this provider
                        print(f"Available models from {local_provider}:")
                        for model in provider_models:
                            if isinstance(model, dict):
                                print(f"  - {model.get('name')}")
                            else:
                                print(f"  - {model}")
                    else:
                        print(f"No models available from {local_provider}")
                except ValueError as e:
                    print(f"Error: {str(e)}")
            else:
                print(f"Unknown local provider: {local_provider}")
                print("Available local providers: ollama, lmstudio")

        else:
            # For invalid subcommands or missing arguments, show a more helpful error message
            if len(args) > 0:
                print(f"Error: Invalid subcommand or missing arguments for '{args[0]}'.")
            else:
                print("Error: Invalid subcommand or missing arguments.")

            print("Usage:")
            print("  llm list                  - List all available models by provider")
            print("  llm model <model_name>    - Switch to a specific model")
            print("  llm add-local <name> <path> <backend> - Add a local model")
            print("  llm local <provider>      - Switch to a local provider (ollama or lmstudio)")


def main():
    """
    Main entry point for the Episodic interactive shell.

    This function creates and runs an instance of the EpisodicShell.
    """
    # Always set debug to False when starting the CLI
    config.set("debug", False)

    shell = EpisodicShell()
    shell.run()


def test_parse_command_line():
    """
    Test the parse_command_line function with various inputs.
    This function is for development and testing purposes only.
    """
    test_cases = [
        # Basic command with no arguments
        ("init", ["init"]),

        # Command with a simple argument
        ("add Hello", ["add", "Hello"]),

        # Command with a quoted argument (should be handled by shlex)
        ("add \"Hello, world!\"", ["add", "Hello, world!"]),

        # Command with a flag
        ("add --parent HEAD", ["add", "--parent", "HEAD"]),

        # Command with text and a flag
        ("add Hello --parent HEAD", ["add", "Hello", "--parent", "HEAD"]),

        # Command with text containing a flag-like pattern
        ("add Hello--world", ["add", "Hello--world"]),

        # Command with text and multiple flags
        ("add Hello --parent HEAD --model gpt-4", ["add", "Hello", "--parent", "HEAD", "--model", "gpt-4"]),

        # Command with text, a flag, more text, and another flag
        ("chat Tell me about Paris --model gpt-4 and its history --context-depth 10", 
         ["chat", "Tell me about Paris", "--model", "gpt-4 and its history", "--context-depth", "10"]),

        # The problematic case from the issue
        ("chat Tell me my name --context-depth 10", 
         ["chat", "Tell me my name", "--context-depth", "10"]),
    ]

    for i, (input_text, expected_output) in enumerate(test_cases):
        result = parse_command_line(input_text)
        if result == expected_output:
            print(f"Test {i+1} passed: {input_text}")
        else:
            print(f"Test {i+1} failed: {input_text}")
            print(f"  Expected: {expected_output}")
            print(f"  Got:      {result}")


if __name__ == "__main__":
    # Run tests if the --test flag is provided
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_parse_command_line()
    else:
        main()
