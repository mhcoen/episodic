"""
Database migration system for Episodic.

This module provides a simple migration framework for managing database
schema changes in a controlled and versioned manner.
"""

import os
import sqlite3
import logging
from typing import List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Migration:
    """Base class for database migrations."""
    
    def __init__(self, version: int, description: str):
        self.version = version
        self.description = description
        self.applied_at: Optional[datetime] = None
    
    def up(self, conn: sqlite3.Connection):
        """Apply the migration."""
        raise NotImplementedError("Subclasses must implement up()")
    
    def down(self, conn: sqlite3.Connection):
        """Rollback the migration."""
        raise NotImplementedError("Subclasses must implement down()")


class MigrationRunner:
    """Manages and runs database migrations."""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._ensure_migration_table()
    
    def _ensure_migration_table(self):
        """Create the migration tracking table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def get_applied_migrations(self) -> List[int]:
        """Get list of already applied migration versions."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
        return [row[0] for row in cursor.fetchall()]
    
    def is_migration_applied(self, version: int) -> bool:
        """Check if a specific migration has been applied."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM schema_migrations WHERE version = ?", (version,))
        return cursor.fetchone() is not None
    
    def apply_migration(self, migration: Migration):
        """Apply a single migration."""
        if self.is_migration_applied(migration.version):
            logger.info(f"Migration {migration.version} already applied, skipping")
            return
        
        logger.info(f"Applying migration {migration.version}: {migration.description}")
        
        try:
            # Start transaction
            cursor = self.conn.cursor()
            
            # Apply the migration
            migration.up(self.conn)
            
            # Record the migration
            cursor.execute(
                "INSERT INTO schema_migrations (version, description) VALUES (?, ?)",
                (migration.version, migration.description)
            )
            
            self.conn.commit()
            logger.info(f"Migration {migration.version} applied successfully")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Migration {migration.version} failed: {e}")
            raise
    
    def rollback_migration(self, migration: Migration):
        """Rollback a single migration."""
        if not self.is_migration_applied(migration.version):
            logger.info(f"Migration {migration.version} not applied, skipping rollback")
            return
        
        logger.info(f"Rolling back migration {migration.version}: {migration.description}")
        
        try:
            # Start transaction
            cursor = self.conn.cursor()
            
            # Rollback the migration
            migration.down(self.conn)
            
            # Remove the migration record
            cursor.execute("DELETE FROM schema_migrations WHERE version = ?", (migration.version,))
            
            self.conn.commit()
            logger.info(f"Migration {migration.version} rolled back successfully")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Rollback of migration {migration.version} failed: {e}")
            raise


def get_pending_migrations(conn: sqlite3.Connection, migrations: List[Migration]) -> List[Migration]:
    """Get list of migrations that haven't been applied yet."""
    runner = MigrationRunner(conn)
    applied = set(runner.get_applied_migrations())
    
    pending = []
    for migration in sorted(migrations, key=lambda m: m.version):
        if migration.version not in applied:
            pending.append(migration)
    
    return pending


def run_migrations(conn: sqlite3.Connection, migrations: List[Migration]):
    """Run all pending migrations."""
    runner = MigrationRunner(conn)
    pending = get_pending_migrations(conn, migrations)
    
    if not pending:
        logger.info("No pending migrations")
        return
    
    logger.info(f"Found {len(pending)} pending migrations")
    
    for migration in pending:
        runner.apply_migration(migration)