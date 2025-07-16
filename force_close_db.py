#!/usr/bin/env python3
"""Force close any database connections and remove lock files."""

import os
import glob
import time

def force_close_db():
    """Force close database by removing all lock files."""
    db_path = os.path.expanduser("~/.episodic/episodic.db")
    
    # Remove all database-related files
    patterns = [
        db_path,
        db_path + "-wal",
        db_path + "-shm",
        db_path + "-journal"
    ]
    
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                print(f"Removing: {file}")
                os.remove(file)
            except Exception as e:
                print(f"Could not remove {file}: {e}")
    
    # Wait a moment
    time.sleep(1)
    
    print("\nDatabase files cleaned up.")

if __name__ == "__main__":
    force_close_db()