"""
Command modules for the Episodic CLI.

This package contains all the command implementations, organized by functionality.
"""

# Import all command functions for easy access
from .navigation import (
    init, add, show, print_node, head, ancestry, list as list_nodes
)
from .settings import set, verify, cost
from .topics import topics, compress_current_topic
from .compression import compress, compression_stats, compression_queue
from .visualization import visualize
from .prompts import prompts
from .summary import summary
from .utility import benchmark, help
from .model import handle_model

__all__ = [
    # Navigation commands
    'init', 'add', 'show', 'print_node', 'head', 'ancestry', 'list_nodes',
    # Settings commands
    'set', 'verify', 'cost',
    # Topic commands
    'topics', 'compress_current_topic',
    # Compression commands
    'compress', 'compression_stats', 'compression_queue',
    # Other commands
    'visualize', 'prompts', 'summary', 'benchmark', 'help',
    'handle_model'
]