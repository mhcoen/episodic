#!/usr/bin/env python3
"""Run database migrations for Episodic."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from episodic.db import get_connection
from episodic.migrations import run_migrations, get_pending_migrations
from episodic.migrations.m004_schema_cleanup import migration as m004

# List of all migrations in order
ALL_MIGRATIONS = [
    m004,
]


def main():
    """Run all pending migrations."""
    print("Episodic Database Migration Runner")
    print("=" * 50)
    
    with get_connection() as conn:
        # Check pending migrations
        pending = get_pending_migrations(conn, ALL_MIGRATIONS)
        
        if not pending:
            print("✅ Database is up to date!")
            return
        
        print(f"\nFound {len(pending)} pending migrations:")
        for m in pending:
            print(f"  - Version {m.version}: {m.description}")
        
        # Ask for confirmation
        response = input("\nProceed with migrations? (y/N): ")
        if response.lower() != 'y':
            print("Migrations cancelled.")
            return
        
        # Run migrations
        print("\nRunning migrations...")
        run_migrations(conn, ALL_MIGRATIONS)
        
        print("\n✅ All migrations completed successfully!")


if __name__ == "__main__":
    main()