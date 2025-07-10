"""
Database operations for Episodic.

This module serves as the main interface for all database operations,
importing and re-exporting functionality from specialized modules.
For backward compatibility, all functions are available directly from
this module.
"""

# Connection management
from .db_connection import (
    get_db_path,
    get_connection,
    database_exists,
    # Re-export constants for backward compatibility
    DEFAULT_DB_PATH,
    DB_PATH
)

# ID generation
from .db_ids import (
    base36_encode,
    generate_short_id
)

# Node operations
from .db_nodes import (
    insert_node,
    get_node,
    get_ancestry,
    set_head,
    get_head,
    get_recent_nodes,
    get_all_nodes,
    get_descendants,
    get_children,
    delete_node,
    resolve_node_ref
)

# Topic operations
from .db_topics import (
    store_topic,
    get_recent_topics,
    get_all_topics,
    update_topic_end_node,
    update_topic_name
)

# Compression operations
from .db_compression import (
    store_compression,
    get_compression_stats
)

# Topic detection scoring
from .db_scoring import (
    store_topic_detection_scores,
    get_topic_detection_scores,
    store_manual_index_score,
    get_manual_index_scores,
    clear_manual_index_scores
)

# Database migrations
from .db_migrations import (
    initialize_db,
    migrate_to_short_ids,
    migrate_to_provider_model,
    migrate_topics_nullable_end,
    migrate_to_roles
)

# RAG operations
from .db_rag import (
    create_rag_tables
)

# Export all functions for backward compatibility
__all__ = [
    # Connection management
    'get_db_path',
    'get_connection',
    'database_exists',
    'DEFAULT_DB_PATH',
    'DB_PATH',
    
    # ID generation
    'base36_encode',
    'generate_short_id',
    
    # Node operations
    'insert_node',
    'get_node',
    'get_ancestry',
    'set_head',
    'get_head',
    'get_recent_nodes',
    'get_all_nodes',
    'get_descendants',
    'get_children',
    'delete_node',
    'resolve_node_ref',
    
    # Topic operations
    'store_topic',
    'get_recent_topics',
    'get_all_topics',
    'update_topic_end_node',
    'update_topic_name',
    
    # Compression operations
    'store_compression',
    'get_compression_stats',
    
    # Topic detection scoring
    'store_topic_detection_scores',
    'get_topic_detection_scores',
    'store_manual_index_score',
    'get_manual_index_scores',
    'clear_manual_index_scores',
    
    # Database migrations
    'initialize_db',
    'migrate_to_short_ids',
    'migrate_to_provider_model',
    'migrate_topics_nullable_end',
    'migrate_to_roles',
    
    # RAG operations
    'create_rag_tables'
]