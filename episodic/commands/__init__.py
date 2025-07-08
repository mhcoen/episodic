"""
Command modules for the Episodic CLI.

This package contains all the command implementations, organized by functionality.
"""

# Import all command functions for easy access
from .navigation import (
    init, add, show, print_node, head, ancestry, list as list_nodes
)
from .settings import set, verify, cost, model_params, config_docs
from .reset import reset, reset_all
from .topics import topics, compress_current_topic
from .topic_rename import rename_ongoing_topics
from .compression import compress, compression_stats, compression_queue, api_call_stats, reset_api_stats
from .visualization import visualize
from .prompts import prompts
from .summary import summary
from .utility import benchmark
from .help import help
from .model import handle_model

__all__ = [
    # Navigation commands
    'init', 'add', 'show', 'print_node', 'head', 'ancestry', 'list_nodes',
    # Settings commands
    'set', 'verify', 'cost', 'model_params', 'config_docs', 'reset', 'reset_all',
    # Topic commands
    'topics', 'compress_current_topic', 'rename_ongoing_topics',
    # Compression commands
    'compress', 'compression_stats', 'compression_queue', 'api_call_stats', 'reset_api_stats',
    # Other commands
    'visualize', 'prompts', 'summary', 'benchmark', 'help',
    'handle_model'
]