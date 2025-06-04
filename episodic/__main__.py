import argparse
import uuid
import os
from episodic.db import insert_node, get_node, get_ancestry, initialize_db, resolve_node_ref, get_head, set_head
from episodic.llm import query_llm, query_with_context

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

    args = parser.parse_args()

    if args.command == "init":
        initialize_db()
    elif args.command == "add":
        parent_id = resolve_node_ref(args.parent) if args.parent else None
        node_id = insert_node(args.content, parent_id)
        set_head(node_id)
        print(f"Added node {node_id}")
    elif args.command == "show":
        node_id = resolve_node_ref(args.node_id)
        node = get_node(node_id)
        if node:
            print(f"Node ID: {node['id']}")
            print(f"Parent: {node['parent_id']}")
            print(f"Message: {node['content']}")
        else:
            print("Node not found.")
    elif args.command == "ancestry":
        node_id = resolve_node_ref(args.node_id)
        node = get_node(node_id)
        ancestry = get_ancestry(node_id)
        for ancestor in ancestry:
            print(f"{ancestor['id']}: {ancestor['content']}")
    elif args.command == "query":
        try:
            # Resolve parent ID if provided
            parent_id = resolve_node_ref(args.parent) if args.parent else None

            # Store the user query as a node
            query_node_id = insert_node(args.prompt, parent_id)
            print(f"Added query node {query_node_id}")

            # Query the LLM
            response = query_llm(
                prompt=args.prompt,
                model=args.model,
                system_message=args.system
            )

            # Store the LLM response as a node with the query as its parent
            response_node_id = insert_node(response, query_node_id)
            print(f"Added response node {response_node_id}")

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
                print("No conversation history found. Initialize with 'episodic init' and add a message with 'episodic add'.")
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
            query_node_id = insert_node(args.prompt, head_id)
            print(f"Added query node {query_node_id}")

            # Query the LLM with context
            response = query_with_context(
                prompt=args.prompt,
                context_messages=context_messages,
                model=args.model,
                system_message=args.system
            )

            # Store the LLM response as a node with the query as its parent
            response_node_id = insert_node(response, query_node_id)
            print(f"Added response node {response_node_id}")

            # Display the response
            print("\nLLM Response:")
            print(response)

        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
