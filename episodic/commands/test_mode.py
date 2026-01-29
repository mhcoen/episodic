"""
Test mode command for Episodic.

Provides /test command to switch between production and test databases,
enabling end-to-end testing of query understanding and retrieval.
"""

import typer
from pathlib import Path
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from episodic.configuration import get_system_color, get_text_color, get_success_color, get_warning_color
from episodic.config import config


# Database paths
TEST_DB_PATH = Path.home() / ".episodic" / "episodic_test.db"
PROD_DB_PATH = Path.home() / ".episodic" / "episodic.db"


def test_command(subcommand: Optional[str] = None):
    """
    Manage test mode for Episodic.
    
    Usage:
        /test             # Show current test mode status
        /test on          # Switch to test database
        /test off         # Switch to production database
        /test setup       # Initialize test DB with fixtures
        /test status      # Show test database status
        /test destroy     # Delete test database
    """
    if not subcommand:
        _show_status()
        return
        
    subcommand = subcommand.lower()
    
    if subcommand == "on":
        _enable_test_mode()
    elif subcommand == "off":
        _disable_test_mode()
    elif subcommand == "setup":
        _setup_test_fixtures()
    elif subcommand == "status":
        _show_detailed_status()
    elif subcommand == "destroy":
        _destroy_test_db()
    else:
        typer.secho(f"Unknown subcommand: {subcommand}", fg="red")
        typer.secho("Valid subcommands: on, off, setup, status, destroy", fg=get_text_color())


def _show_status():
    """Show current test mode status."""
    is_test_mode = config.get("test_mode", False)
    
    if is_test_mode:
        typer.secho("🧪 Test mode: ENABLED", fg=get_warning_color(), bold=True)
        typer.secho(f"   Database: {TEST_DB_PATH}", fg=get_text_color())
    else:
        typer.secho("📦 Test mode: DISABLED (using production)", fg=get_system_color())
        typer.secho(f"   Database: {PROD_DB_PATH}", fg=get_text_color())
    
    typer.echo()
    typer.secho("Commands:", fg=get_text_color())
    typer.secho("  /test on     - Switch to test database", fg=get_text_color())
    typer.secho("  /test off    - Switch to production database", fg=get_text_color())
    typer.secho("  /test setup  - Initialize test fixtures", fg=get_text_color())
    typer.secho("  /test status - Show detailed status", fg=get_text_color())


def _enable_test_mode():
    """Enable test mode (switch to test database)."""
    if not TEST_DB_PATH.exists():
        typer.secho("⚠️  Test database does not exist.", fg=get_warning_color())
        typer.secho("   Run '/test setup' first to create it.", fg=get_text_color())
        return
    
    config.set("test_mode", True)
    config.set("database_path", str(TEST_DB_PATH))
    
    typer.secho("🧪 Test mode ENABLED", fg=get_success_color(), bold=True)
    typer.secho(f"   Now using: {TEST_DB_PATH}", fg=get_text_color())
    typer.secho("   ⚠️  Note: You may need to restart for full effect.", fg=get_warning_color())


def _disable_test_mode():
    """Disable test mode (switch to production database)."""
    config.set("test_mode", False)
    config.set("database_path", str(PROD_DB_PATH))
    
    typer.secho("📦 Test mode DISABLED", fg=get_success_color(), bold=True)
    typer.secho(f"   Now using: {PROD_DB_PATH}", fg=get_text_color())
    typer.secho("   ⚠️  Note: You may need to restart for full effect.", fg=get_warning_color())


def _setup_test_fixtures():
    """Initialize test database with standard fixtures."""
    from episodic.test_fixtures import setup_test_environment
    
    typer.secho("🔧 Setting up test fixtures...", fg=get_system_color())
    
    # Use current time as reference (or fixed time for reproducibility)
    reference_time = datetime(2026, 1, 26, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
    
    try:
        manager = setup_test_environment(reference_time)
        
        # Get stats
        cursor = manager.conn.execute("SELECT COUNT(*) FROM nodes")
        node_count = cursor.fetchone()[0]
        
        cursor = manager.conn.execute("SELECT COUNT(*) FROM topics")
        topic_count = cursor.fetchone()[0]
        
        cursor = manager.conn.execute("SELECT name, end_node_id FROM topics")
        topics = cursor.fetchall()
        
        manager.cleanup()
        
        typer.secho(f"✅ Test database created: {TEST_DB_PATH}", fg=get_success_color())
        typer.secho(f"   Nodes: {node_count}", fg=get_text_color())
        typer.secho(f"   Topics: {topic_count}", fg=get_text_color())
        
        typer.echo()
        typer.secho("📚 Test topics:", fg=get_system_color())
        for name, end_id in topics:
            status = "ongoing" if end_id is None else "closed"
            typer.secho(f"   • {name} ({status})", fg=get_text_color())
        
        typer.echo()
        typer.secho("Reference time: 2026-01-26 12:00 UTC", fg=get_text_color())
        typer.secho("Temporal fixtures:", fg=get_text_color())
        typer.secho("   • yesterday: machine-learning-basics", fg=get_text_color())
        typer.secho("   • 3 days ago: python-asyncio", fg=get_text_color())
        typer.secho("   • last week: database-indexing", fg=get_text_color())
        typer.secho("   • last month: quantum-computing", fg=get_text_color())
        
        typer.echo()
        typer.secho("Run '/test on' to switch to test database.", fg=get_text_color())
        
    except Exception as e:
        typer.secho(f"❌ Failed to set up test fixtures: {e}", fg="red")
        if config.get("debug"):
            import traceback
            typer.secho(traceback.format_exc(), fg="red")


def _show_detailed_status():
    """Show detailed test database status."""
    typer.secho("\n🧪 Test Database Status", fg=get_system_color(), bold=True)
    typer.secho("─" * 50, fg=get_system_color())
    
    is_test_mode = config.get("test_mode", False)
    typer.secho(f"Test mode: {'ENABLED' if is_test_mode else 'DISABLED'}", 
                fg=get_warning_color() if is_test_mode else get_text_color())
    
    if TEST_DB_PATH.exists():
        import os
        stat = os.stat(TEST_DB_PATH)
        size_kb = stat.st_size / 1024
        mtime = datetime.fromtimestamp(stat.st_mtime)
        
        typer.secho(f"Test DB exists: {TEST_DB_PATH}", fg=get_success_color())
        typer.secho(f"   Size: {size_kb:.1f} KB", fg=get_text_color())
        typer.secho(f"   Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}", fg=get_text_color())
        
        # Try to read stats from DB
        try:
            import sqlite3
            conn = sqlite3.connect(str(TEST_DB_PATH))
            
            cursor = conn.execute("SELECT COUNT(*) FROM nodes")
            node_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM topics")
            topic_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT name, end_node_id FROM topics ORDER BY id")
            topics = cursor.fetchall()
            
            conn.close()
            
            typer.secho(f"   Nodes: {node_count}", fg=get_text_color())
            typer.secho(f"   Topics: {topic_count}", fg=get_text_color())
            
            if topics:
                typer.echo()
                typer.secho("Topics in test DB:", fg=get_system_color())
                for name, end_id in topics:
                    status = "ongoing" if end_id is None else "closed"
                    typer.secho(f"   • {name} ({status})", fg=get_text_color())
                    
        except Exception as e:
            typer.secho(f"   ⚠️  Could not read DB stats: {e}", fg=get_warning_color())
    else:
        typer.secho(f"Test DB does not exist: {TEST_DB_PATH}", fg=get_warning_color())
        typer.secho("   Run '/test setup' to create it.", fg=get_text_color())
    
    typer.echo()
    if PROD_DB_PATH.exists():
        import os
        stat = os.stat(PROD_DB_PATH)
        size_kb = stat.st_size / 1024
        typer.secho(f"Prod DB: {PROD_DB_PATH} ({size_kb:.1f} KB)", fg=get_text_color())
    else:
        typer.secho(f"Prod DB does not exist: {PROD_DB_PATH}", fg=get_warning_color())


def _destroy_test_db():
    """Delete the test database."""
    if not TEST_DB_PATH.exists():
        typer.secho("Test database does not exist.", fg=get_text_color())
        return
    
    # Confirm
    typer.secho("⚠️  This will delete the test database permanently.", fg=get_warning_color())
    confirm = typer.confirm("Are you sure?")
    
    if confirm:
        try:
            TEST_DB_PATH.unlink()
            # Also remove WAL files if present
            wal_path = TEST_DB_PATH.with_suffix(".db-wal")
            shm_path = TEST_DB_PATH.with_suffix(".db-shm")
            if wal_path.exists():
                wal_path.unlink()
            if shm_path.exists():
                shm_path.unlink()
                
            typer.secho("✅ Test database deleted.", fg=get_success_color())
            
            # Disable test mode if it was on
            if config.get("test_mode", False):
                config.set("test_mode", False)
                config.set("database_path", str(PROD_DB_PATH))
                typer.secho("   Test mode disabled.", fg=get_text_color())
                
        except Exception as e:
            typer.secho(f"❌ Failed to delete test database: {e}", fg="red")
    else:
        typer.secho("Cancelled.", fg=get_text_color())
