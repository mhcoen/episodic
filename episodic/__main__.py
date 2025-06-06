import argparse
import uuid
import os
import webbrowser
import sys
from episodic.db import insert_node, get_node, get_ancestry, initialize_db, resolve_node_ref, get_head, set_head, database_exists
from episodic.llm import query_llm, query_with_context
from episodic.visualization import visualize_dag

# This comment was added to demonstrate file editing capabilities

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init")

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("content", help="Message content")
    add_parser.add_argument("--parent", help="Parent node ID", default=None)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("node_id", help="Node ID to show")

    ancestry_parser = subparsers.add_parser("ancestry")
    ancestry_parser.add_argument("node_id", help="Node ID to trace ancestry")

    # Add new command for changing the current node
    goto_parser = subparsers.add_parser("goto")
    goto_parser.add_argument("node_id", help="Node ID to make current")

    # Add new command for querying the LLM
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("prompt", help="Query to send to the LLM")
    query_parser.add_argument("--model", help="LLM model to use", default="gpt-3.5-turbo")
    query_parser.add_argument("--system", help="System message for the LLM", default="You are a helpful assistant.")
    query_parser.add_argument("--parent", help="Parent node ID", default=None)

    # Add new command for chatting with the LLM using conversation history
    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("prompt", help="Query to send to the LLM")
    chat_parser.add_argument("--model", help="LLM model to use", default="gpt-3.5-turbo")
    chat_parser.add_argument("--system", help="System message for the LLM", default="You are a helpful assistant.")
    chat_parser.add_argument("--context-depth", help="Number of ancestor nodes to include as context", type=int, default=5)

    # Add new command for visualizing the conversation DAG
    visualize_parser = subparsers.add_parser("visualize")
    visualize_parser.add_argument("--output", help="Path to save the HTML visualization", default=None)
    visualize_parser.add_argument("--no-browser", help="Don't open the visualization in a browser", action="store_true")

    args = parser.parse_args()

    if args.command == "init":
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
    elif args.command == "goto":
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

            # Query the LLM
            response = query_llm(
                prompt=args.prompt,
                model=args.model,
                system_message=args.system
            )

            # Store the LLM response as a node with the query as its parent
            response_node_id, response_short_id = insert_node(response, query_node_id)
            print(f"Added response node {response_short_id} (UUID: {response_node_id})")

            # Display the response
            print("\nLLM Response:")
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

            # Query the LLM with context
            response = query_with_context(
                prompt=args.prompt,
                context_messages=context_messages,
                model=args.model,
                system_message=args.system
            )

            # Store the LLM response as a node with the query as its parent
            response_node_id, response_short_id = insert_node(response, query_node_id)
            print(f"Added response node {response_short_id} (UUID: {response_node_id})")

            # Display the response
            print("\nLLM Response:")
            print(response)

        except Exception as e:
            print(f"Error: {str(e)}")
    elif args.command == "visualize":
        try:
            output_path = visualize_dag(args.output)
            if output_path and not args.no_browser:
                print(f"Opening visualization in browser: {output_path}")
                webbrowser.open(f"file://{os.path.abspath(output_path)}")
            elif output_path:
                print(f"Visualization saved to: {output_path}")
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
