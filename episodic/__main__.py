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
    # Import and use the new CLI
    from episodic.new_cli import main as new_main
    new_main()

if __name__ == "__main__":
    main()
