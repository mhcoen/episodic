#\!/usr/bin/env python3
"""Migrate manual_index_scores table to use short IDs."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from episodic.db import get_connection

with get_connection() as conn:
    c = conn.cursor()
    
    # Drop old table if exists
    c.execute("DROP TABLE IF EXISTS manual_index_scores")
    
    # Create new table with short IDs
    c.execute("""
        CREATE TABLE IF NOT EXISTS manual_index_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_node_short_id TEXT NOT NULL UNIQUE,
            window_size INTEGER NOT NULL,
            
            -- Window information
            window_a_start_short_id TEXT,
            window_a_end_short_id TEXT,
            window_a_size INTEGER NOT NULL,
            window_b_start_short_id TEXT,
            window_b_end_short_id TEXT,
            window_b_size INTEGER NOT NULL,
            
            -- Scores
            drift_score REAL NOT NULL,
            keyword_score REAL NOT NULL,
            combined_score REAL NOT NULL,
            
            -- Detection result
            is_boundary BOOLEAN NOT NULL,
            transition_phrase TEXT,
            
            -- Metadata
            detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            threshold_used REAL
        )
    """)
    
    conn.commit()
    print("✅ Successfully migrated manual_index_scores table to use short IDs")

