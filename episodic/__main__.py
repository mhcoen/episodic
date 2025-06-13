import argparse
import uuid
import os
import webbrowser
import sys
from episodic.db import insert_node, get_node, get_ancestry, initialize_db, resolve_node_ref, get_head, set_head, database_exists, get_recent_nodes
from episodic.llm import query_llm, query_with_context
from episodic.visualization import visualize_dag
from episodic.prompt_manager import PromptManager
from episodic.config import config

# This comment was added to demonstrate file editing capabilities

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("dbinit")

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("content", help="Message content")
    add_parser.add_argument("--parent", help="Parent node ID", default=None)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("node_id", help="Node ID to show")

    ancestry_parser = subparsers.add_parser("ancestry")
    ancestry_parser.add_argument("node_id", help="Node ID to trace ancestry")

    # Add new command for displaying current node or changing to a specified node
    head_parser = subparsers.add_parser("head")
    head_parser.add_argument("node_id", help="Node ID to make current", nargs='?')

    # Add new command for printing node info
    print_parser = subparsers.add_parser("print")
    print_parser.add_argument("node_id", help="Node ID to print (defaults to current node)", nargs='?')

    # Add new command for querying the LLM
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("prompt", help="Query to send to the LLM")
    query_parser.add_argument("--model", help="LLM model to use", default="gpt-3.5-turbo")
    query_parser.add_argument("--system", help="System message for the LLM (overrides active prompt)", default=None)
    query_parser.add_argument("--parent", help="Parent node ID", default=None)

    # Add new command for chatting with the LLM using conversation history
    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("prompt", help="Query to send to the LLM")
    chat_parser.add_argument("--model", help="LLM model to use", default="gpt-3.5-turbo")
    chat_parser.add_argument("--system", help="System message for the LLM (overrides active prompt)", default=None)
    chat_parser.add_argument("--context-depth", help="Number of ancestor nodes to include as context", type=int, default=5)

    # Add new command for visualizing the conversation DAG
    visualize_parser = subparsers.add_parser("visualize")
    visualize_parser.add_argument("--output", help="Path to save the HTML visualization", default=None)
    visualize_parser.add_argument("--no-browser", help="Don't open the visualization in a browser", action="store_true")
    visualize_parser.add_argument("--port", help="Port to use for the visualization server", type=int, default=5000)
    visualize_parser.add_argument("--native", help="Open the visualization in a native window instead of a browser", action="store_true")
    visualize_parser.add_argument("--width", help="Width of the native window", type=int, default=1000)
    visualize_parser.add_argument("--height", help="Height of the native window", type=int, default=800)

    # Add new command for listing recent nodes
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--count", help="Number of recent nodes to list", type=int, default=5)

    # Add new command for managing prompts
    prompts_parser = subparsers.add_parser("prompts")
    prompts_subparsers = prompts_parser.add_subparsers(dest="prompts_command")

    # Add subcommands for prompts
    prompts_list_parser = prompts_subparsers.add_parser("list", help="List all available prompts")

    prompts_use_parser = prompts_subparsers.add_parser("use", help="Set the active prompt")
    prompts_use_parser.add_argument("name", help="Name of the prompt to use")

    prompts_show_parser = prompts_subparsers.add_parser("show", help="Show the content of a prompt")
    prompts_show_parser.add_argument("name", help="Name of the prompt to show", nargs='?')

    prompts_reload_parser = prompts_subparsers.add_parser("reload", help="Reload prompts from disk")

    args = parser.parse_args()

    if args.command == "dbinit":
        if database_exists():
            print("Database already exists. Do you want to erase it? (yes/no)")
            response = input().strip().lower()
            if response in ["yes", "y"]:
                result = initialize_db(erase=True)
                if result:
                    root_node_id, root_short_id = result
                    print(f"Database has been reinitialized with a default root node (ID: {root_short_id}, UUID: {root_node_id}).")
                else:
                    print("Database has been reinitialized.")
            else:
                print("Database initialization cancelled.")
        else:
            result = initialize_db()
            if result:
                root_node_id, root_short_id = result
                print(f"Database initialized with a default root node (ID: {root_short_id}, UUID: {root_node_id}).")
            else:
                print("Database initialized.")
    elif args.command == "add":
        parent_id = resolve_node_ref(args.parent) if args.parent else None
        node_id, short_id = insert_node(args.content, parent_id)
        set_head(node_id)
        print(f"Added node {short_id} (UUID: {node_id})")
    elif args.command == "show":
        node_id = resolve_node_ref(args.node_id)
        node = get_node(node_id)
        if node:
            print(f"Node ID: {node['short_id']} (UUID: {node['id']})")
            if node['parent_id']:
                parent = get_node(node['parent_id'])
                parent_short_id = parent['short_id'] if parent else "Unknown"
                print(f"Parent: {parent_short_id} (UUID: {node['parent_id']})")
            else:
                print(f"Parent: None")
            print(f"Message: {node['content']}")
        else:
            print("Node not found.")
    elif args.command == "ancestry":
        node_id = resolve_node_ref(args.node_id)
        node = get_node(node_id)
        ancestry = get_ancestry(node_id)
        for ancestor in ancestry:
            print(f"{ancestor['short_id']} (UUID: {ancestor['id']}): {ancestor['content']}")
    elif args.command == "print":
        # Determine which node to print
        if not args.node_id:
            # If no node ID is provided, use the current head node
            node_id = get_head()
            if not node_id:
                print("No current node. Specify a node ID or use 'add' to create a node.")
                return
        else:
            # Resolve the provided node ID
            node_id = resolve_node_ref(args.node_id)

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
            print(f"Message: {node['content']}")
        else:
            print("Node not found.")
    elif args.command == "head":
        if not args.node_id:
            # If no node ID is provided, display the current node's info
            # This is equivalent to running the print command with no arguments
            head_id = get_head()
            if not head_id:
                print("No current node. Specify a node ID or use 'add' to create a node.")
                return

            # Get and display the node
            node = get_node(head_id)
            if node:
                # Use a slightly different format for head to indicate it's the current node
                print(f"Current node: {node['short_id']} (UUID: {node['id']})")
                if node['parent_id']:
                    parent = get_node(node['parent_id'])
                    parent_short_id = parent['short_id'] if parent else "Unknown"
                    print(f"Parent: {parent_short_id} (UUID: {node['parent_id']})")
                else:
                    print(f"Parent: None")
                print(f"Message: {node['content']}")
            else:
                print("Node not found.")
        else:
            # Resolve the node ID
            node_id = resolve_node_ref(args.node_id)

            # Verify that the node exists
            node = get_node(node_id)
            if not node:
                print(f"Error: Node not found: {args.node_id}")
                return

            # Update the current node
            set_head(node_id)

            # Display confirmation
            print(f"Current node changed to: {node['short_id']} (UUID: {node['id']})")
    elif args.command == "query":
        try:
            # Resolve parent ID if provided
            parent_id = resolve_node_ref(args.parent) if args.parent else None

            # Store the user query as a node
            query_node_id, query_short_id = insert_node(args.prompt, parent_id)
            print(f"Added query node {query_short_id} (UUID: {query_node_id})")

            # Get the system message from the active prompt if not provided
            system_message = args.system
            if system_message is None:
                # Create a prompt manager instance
                manager = PromptManager()
                system_message = manager.get_active_prompt_content(config.get)

            # Query the LLM
            response, cost_info = query_llm(
                prompt=args.prompt,
                model=args.model,
                system_message=system_message
            )
            # Display cost information if enabled
            if config.get("show_cost", False):
                print(f"({cost_info['input_tokens']}_in + {cost_info['output_tokens']}_out = {cost_info['total_tokens']}_tokens ${cost_info['cost_usd']:.6f} USD)")

            # Store the LLM response as a node with the query as its parent
            response_node_id, response_short_id = insert_node(response, query_node_id)
            print(f"Added response node {response_short_id} (UUID: {response_node_id})")

            # Display the response with model information
            print("\nLLM Response:")
            # Get the current provider to display along with the model
            from episodic.llm_config import get_current_provider
            provider = get_current_provider()
            print(f"\033[36m🤖 {provider}/{args.model}:\033[0m")
            print(response)

        except Exception as e:
            print(f"Error: {str(e)}")
    elif args.command == "chat":
        try:
            # Get the current head node
            head_id = get_head()
            if not head_id:
                if database_exists():
                    # Database exists but no messages yet - this should be rare now with the implicit root node
                    print("No conversation history found. This is unusual since initialization should create a root node.")
                    print("Try reinitializing the database with 'episodic init' or add a message with 'episodic add'.")
                else:
                    # Database doesn't exist yet
                    print("No database found. Please initialize the database with 'episodic init' command first.")
                    print("Initialization will create a default root node that can be used for conversation.")
                return

            # Get the ancestry of the head node to use as context
            ancestry = get_ancestry(head_id)

            # Limit the context to the specified depth
            context_ancestry = ancestry[-args.context_depth:] if args.context_depth > 0 else ancestry

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
            query_node_id, query_short_id = insert_node(args.prompt, head_id)
            print(f"Added query node {query_short_id} (UUID: {query_node_id})")

            # Get the system message from the active prompt if not provided
            system_message = args.system
            if system_message is None:
                # Create a prompt manager instance
                manager = PromptManager()
                system_message = manager.get_active_prompt_content(config.get)

            # Query the LLM with context
            response, cost_info = query_with_context(
                prompt=args.prompt,
                context_messages=context_messages,
                model=args.model,
                system_message=system_message
            )
            # Display cost information if enabled
            if config.get("show_cost", False):
                print(f"({cost_info['input_tokens']}_in + {cost_info['output_tokens']}_out = {cost_info['total_tokens']}_tokens ${cost_info['cost_usd']:.6f} USD)")

            # Store the LLM response as a node with the query as its parent
            response_node_id, response_short_id = insert_node(response, query_node_id)
            print(f"Added response node {response_short_id} (UUID: {response_node_id})")

            # Display the response with model information
            print("\nLLM Response:")
            # Get the current provider to display along with the model
            from episodic.llm_config import get_current_provider
            provider = get_current_provider()
            print(f"\033[36m🤖 {provider}/{args.model}:\033[0m")
            print(response)

        except Exception as e:
            print(f"Error: {str(e)}")
    elif args.command == "visualize":
        try:
            # If native mode is requested, use the native visualization
            if args.native:
                from episodic.gui import visualize_native_blocking
                print(f"Opening native visualization window (width: {args.width}, height: {args.height})...")
                print("The window will remain open until you close it.")
                print("Press Ctrl+C to stop the program if the window doesn't close properly.")
                try:
                    visualize_native_blocking(width=args.width, height=args.height, server_port=args.port)
                except KeyboardInterrupt:
                    print("\nProgram stopped by user.")
            else:
                output_path = visualize_dag(args.output)

                # If interactive mode is requested, start the server with the specified port
                if output_path and not args.no_browser:
                    from episodic.server import start_server, stop_server
                    server_url = start_server(server_port=args.port)
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
    elif args.command == "list":
        try:
            # Get recent nodes
            nodes = get_recent_nodes(args.count)

            if not nodes:
                print("No nodes found in the database.")
                return

            print(f"Recent nodes (showing {len(nodes)} of {args.count} requested):")
            for node in nodes:
                # Truncate content for display
                content = node['content']
                if len(content) > 50:
                    content = content[:47] + "..."

                # Display node information
                print(f"{node['short_id']} (UUID: {node['id']}): {content}")
        except Exception as e:
            print(f"Error retrieving recent nodes: {str(e)}")
    elif args.command == "prompts":
        # Create a prompt manager instance
        manager = PromptManager()

        if args.prompts_command == "list":
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

        elif args.prompts_command == "use":
            # Set the active prompt
            if args.name not in manager.list():
                print(f"Prompt '{args.name}' not found.")
                return

            # Store the active prompt name in config
            config.set("active_prompt", args.name)

            # Display confirmation and description if available
            metadata = manager.get_metadata(args.name)
            description = f" - {metadata.get('description')}" if metadata and 'description' in metadata else ''
            print(f"Now using prompt: {args.name}{description}")

        elif args.prompts_command == "show":
            # Determine which prompt to show
            name = args.name if args.name else config.get("active_prompt", "default")

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

        elif args.prompts_command == "reload":
            # Reload prompts from disk
            manager.reload()
            print(f"Reloaded {len(manager.list())} prompts.")

        else:
            # If no subcommand is provided, show help
            prompts_parser.print_help()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
