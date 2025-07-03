#!/usr/bin/env python3
"""Check what's in the topic_detection_scores table"""

import sqlite3
import os

# Get database path
db_path = "episodic.db"
if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Get table info
print("Table schema for topic_detection_scores:")
c.execute("PRAGMA table_info(topic_detection_scores)")
columns = c.fetchall()
for col in columns:
    print(f"  {col[1]} {col[2]}")

# Count records
c.execute("SELECT COUNT(*) FROM topic_detection_scores")
count = c.fetchone()[0]
print(f"\nTotal records: {count}")

# Get sample records
print("\nSample records (first 5):")
c.execute("SELECT * FROM topic_detection_scores LIMIT 5")
rows = c.fetchall()

if rows:
    # Get column names
    c.execute("PRAGMA table_info(topic_detection_scores)")
    col_names = [col[1] for col in c.fetchall()]
    
    for row in rows:
        print("\n---")
        for i, val in enumerate(row):
            if val is not None:
                print(f"  {col_names[i]}: {val}")
else:
    print("  No records found")

# Check which columns have data
print("\n\nColumns with non-null values:")
for col in [col[1] for col in columns]:
    c.execute(f"SELECT COUNT(*) FROM topic_detection_scores WHERE {col} IS NOT NULL")
    count = c.fetchone()[0]
    if count > 0:
        print(f"  {col}: {count} non-null values")

conn.close()