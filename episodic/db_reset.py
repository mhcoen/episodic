"""
Database reset utilities for Episodic.

This module provides functions to completely reset the database subsystem,
which is necessary for operations like /init --erase in interactive mode.
"""

import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def reset_database_subsystem(db_path: Optional[str] = None) -> None:
    """
    Completely reset the database subsystem.
    
    This function:
    1. Closes all connection pools
    2. Clears any global connection state
    3. Removes database files if specified
    4. Waits for filesystem sync
    
    Args:
        db_path: If provided, remove this database file and associated files
    """
    # Import here to avoid circular imports
    from .db_connection import close_pool
    
    # Close all connections
    logger.info("Closing all database connections...")
    close_pool()
    
    # If db_path provided, remove database files
    if db_path and os.path.exists(db_path):
        logger.info(f"Removing database files at {db_path}")
        
        # Remove main database file
        try:
            os.remove(db_path)
        except Exception as e:
            logger.error(f"Failed to remove database file: {e}")
            
        # Remove associated files (WAL, SHM, journal)
        for suffix in ['-wal', '-shm', '-journal']:
            associated_file = db_path + suffix
            if os.path.exists(associated_file):
                try:
                    os.remove(associated_file)
                    logger.info(f"Removed {associated_file}")
                except Exception as e:
                    logger.error(f"Failed to remove {associated_file}: {e}")
    
    # Wait a moment for filesystem to sync
    time.sleep(0.1)
    
    logger.info("Database subsystem reset complete")