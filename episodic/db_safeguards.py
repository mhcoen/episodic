"""
Database safeguards to prevent creating databases in the wrong location.
"""

import os
import sqlite3
import functools
from pathlib import Path


def ensure_not_in_project_root(func):
    """Decorator to ensure database operations don't create files in project root."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the project root (where .git directory is)
        current_path = Path(__file__).parent.parent
        
        # Check if we're trying to create a database in the project root
        if args and isinstance(args[0], str):
            db_path = Path(args[0]).resolve()
            project_root = current_path.resolve()
            
            # If the database would be created in the project directory, raise an error
            if db_path.parent == project_root or db_path.parent == project_root / "episodic":
                raise ValueError(
                    f"Attempted to create database in project directory: {db_path}\n"
                    f"Database should be created in ~/.episodic/ instead.\n"
                    f"Set EPISODIC_DB_PATH environment variable or use get_db_path()."
                )
        
        return func(*args, **kwargs)
    return wrapper


# Monkey-patch sqlite3.connect to add our safeguard
_original_sqlite3_connect = sqlite3.connect

@ensure_not_in_project_root
def _safe_sqlite3_connect(database, *args, **kwargs):
    """Safe version of sqlite3.connect that prevents creating DBs in project root."""
    return _original_sqlite3_connect(database, *args, **kwargs)

# Apply the patch
sqlite3.connect = _safe_sqlite3_connect


# Also safeguard ChromaDB to prevent it from creating directories in project root
try:
    import chromadb
    
    # Save original PersistentClient
    _original_persistent_client = chromadb.PersistentClient
    
    def _safe_persistent_client(path=None, *args, **kwargs):
        """Safe version of chromadb.PersistentClient that requires explicit path."""
        if path is None:
            raise ValueError(
                "ChromaDB PersistentClient requires an explicit path.\n"
                "Use path='~/.episodic/rag/chroma' or another appropriate location.\n"
                "Do not create ChromaDB databases in the project directory."
            )
        
        # Validate the path
        validate_db_path(path)
        
        return _original_persistent_client(path, *args, **kwargs)
    
    # Apply the patch
    chromadb.PersistentClient = _safe_persistent_client
    
except ImportError:
    # ChromaDB not installed, that's fine
    pass


def validate_db_path(db_path: str) -> str:
    """
    Validate that a database path is not in the project directory.
    
    Args:
        db_path: The database path to validate
        
    Returns:
        The validated path
        
    Raises:
        ValueError: If the path would create a database in the project directory
    """
    project_root = Path(__file__).parent.parent.resolve()
    resolved_path = Path(db_path).resolve()
    
    # Check if the path is in the project directory
    if resolved_path.parent == project_root or resolved_path.parent == project_root / "episodic":
        raise ValueError(
            f"Invalid database path: {db_path}\n"
            f"Database cannot be created in the project directory.\n"
            f"Use ~/.episodic/ or set EPISODIC_DB_PATH environment variable."
        )
    
    return db_path