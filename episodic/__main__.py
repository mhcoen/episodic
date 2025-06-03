import argparse
import uuid
from episodic.db import insert_node, get_node, get_ancestry, initialize_db, resolve_node_ref, get_head, set_head

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
            print(f"Message: {node['message']}")
        else:
            print("Node not found.")
    elif args.command == "ancestry":
        node_id = resolve_node_ref(args.node_id)
        node = get_node(node_id)
        ancestry = get_ancestry(node_id)
        for ancestor in ancestry:
            print(f"{ancestor['id']}: {ancestor['message']}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
