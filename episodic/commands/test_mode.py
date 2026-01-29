"""
Test mode command for Episodic.

Provides /test command to switch between production and test databases,
enabling end-to-end testing of query understanding and retrieval.

Features:
- Full isolation: Separate SQLite DB and ChromaDB directory
- Clone: Copy production database to test environment
- Clear: Wipe test environment completely
- Visual indicator: [TEST] prompt when in test mode
"""

import os
import shutil
import typer
from pathlib import Path
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from episodic.configuration import get_system_color, get_text_color, get_success_color, get_warning_color
from episodic.config import config


def is_test_mode() -> bool:
    """Check if test mode is currently active."""
    return config.get("test_mode", False)


# Test environment paths
TEST_BASE_DIR = Path.home() / ".episodic" / "test"
TEST_DB_PATH = TEST_BASE_DIR / "episodic.db"
TEST_CHROMA_PATH = TEST_BASE_DIR / "chroma"

# Production paths
PROD_DB_PATH = Path.home() / ".episodic" / "episodic.db"
PROD_CHROMA_PATH = Path.home() / ".episodic" / "rag" / "chroma"


def test_command(subcommand: Optional[str] = None):
    """
    Manage test mode for Episodic.

    Usage:
        /test             # Show current test mode status
        /test on          # Switch to test environment
        /test off         # Switch to production environment
        /test clone       # Copy production to test (DB + ChromaDB)
        /test clear       # Wipe test environment completely
        /test setup       # Initialize test DB with fixtures
        /test status      # Show test database status
        /test destroy     # Delete test database (alias for clear)
    """
    if not subcommand:
        _show_status()
        return

    subcommand = subcommand.lower()

    if subcommand == "on":
        _enable_test_mode()
    elif subcommand == "off":
        _disable_test_mode()
    elif subcommand == "clone":
        _clone_production_to_test()
    elif subcommand == "clear":
        _clear_test_environment()
    elif subcommand == "setup":
        _setup_test_fixtures()
    elif subcommand == "status":
        _show_detailed_status()
    elif subcommand == "destroy":
        _clear_test_environment()  # Alias for clear
    else:
        typer.secho(f"Unknown subcommand: {subcommand}", fg="red")
        typer.secho("Valid subcommands: on, off, clone, clear, setup, status", fg=get_text_color())


def _show_status():
    """Show current test mode status."""
    is_test_mode = config.get("test_mode", False)

    if is_test_mode:
        typer.secho("🧪 Test mode: ENABLED", fg=get_warning_color(), bold=True)
        typer.secho(f"   Database: {TEST_DB_PATH}", fg=get_text_color())
        typer.secho(f"   ChromaDB: {TEST_CHROMA_PATH}", fg=get_text_color())
    else:
        typer.secho("📦 Test mode: DISABLED (using production)", fg=get_system_color())
        typer.secho(f"   Database: {PROD_DB_PATH}", fg=get_text_color())
        typer.secho(f"   ChromaDB: {PROD_CHROMA_PATH}", fg=get_text_color())

    typer.echo()
    typer.secho("Commands:", fg=get_text_color())
    typer.secho("  /test on     - Switch to test environment", fg=get_text_color())
    typer.secho("  /test off    - Switch to production environment", fg=get_text_color())
    typer.secho("  /test clone  - Copy production to test", fg=get_text_color())
    typer.secho("  /test clear  - Wipe test environment", fg=get_text_color())
    typer.secho("  /test setup  - Initialize test fixtures", fg=get_text_color())
    typer.secho("  /test status - Show detailed status", fg=get_text_color())


def _enable_test_mode():
    """Enable test mode (switch to test environment)."""
    # Ensure test directory exists
    TEST_BASE_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_DB_PATH.exists():
        typer.secho("⚠️  Test database does not exist.", fg=get_warning_color())
        typer.secho("   Run '/test clone' to copy production, or '/test setup' for fixtures.", fg=get_text_color())
        return

    # Close existing connections before switching
    _reset_connections()

    # Set environment variable for database path
    os.environ["EPISODIC_DB_PATH"] = str(TEST_DB_PATH)
    os.environ["EPISODIC_CHROMA_PATH"] = str(TEST_CHROMA_PATH)

    config.set("test_mode", True)
    config.set("database_path", str(TEST_DB_PATH))
    config.set("chroma_path", str(TEST_CHROMA_PATH))

    typer.secho("🧪 Test mode ENABLED", fg=get_success_color(), bold=True)
    typer.secho(f"   Database: {TEST_DB_PATH}", fg=get_text_color())
    typer.secho(f"   ChromaDB: {TEST_CHROMA_PATH}", fg=get_text_color())
    typer.secho("   ⚠️  Note: Restart recommended for full isolation.", fg=get_warning_color())


def _disable_test_mode():
    """Disable test mode (switch to production environment)."""
    # Close existing connections before switching
    _reset_connections()

    # Clear environment variables
    os.environ.pop("EPISODIC_DB_PATH", None)
    os.environ.pop("EPISODIC_CHROMA_PATH", None)

    config.set("test_mode", False)
    config.set("database_path", str(PROD_DB_PATH))
    config.set("chroma_path", str(PROD_CHROMA_PATH))

    typer.secho("📦 Test mode DISABLED", fg=get_success_color(), bold=True)
    typer.secho(f"   Database: {PROD_DB_PATH}", fg=get_text_color())
    typer.secho(f"   ChromaDB: {PROD_CHROMA_PATH}", fg=get_text_color())
    typer.secho("   ⚠️  Note: Restart recommended for full isolation.", fg=get_warning_color())


def _reset_connections():
    """Reset database and ChromaDB connections."""
    try:
        from episodic.db_connection import close_pool
        close_pool()
    except Exception:
        pass

    try:
        import episodic.rag_collections as rag
        rag._multi_collection_rag = None
    except Exception:
        pass


def _clone_production_to_test():
    """Clone production database and ChromaDB to test environment."""
    # Check if production exists
    if not PROD_DB_PATH.exists():
        typer.secho("❌ Production database does not exist.", fg="red")
        return

    # Ensure test directory exists
    TEST_BASE_DIR.mkdir(parents=True, exist_ok=True)

    typer.secho("📋 Cloning production to test environment...", fg=get_system_color())

    # Copy database
    try:
        shutil.copy2(PROD_DB_PATH, TEST_DB_PATH)
        typer.secho(f"   ✓ Copied database: {TEST_DB_PATH}", fg=get_text_color())

        # Also copy WAL files if present
        for suffix in ["-wal", "-shm"]:
            wal_src = PROD_DB_PATH.with_suffix(PROD_DB_PATH.suffix + suffix)
            if wal_src.exists():
                wal_dst = TEST_DB_PATH.with_suffix(TEST_DB_PATH.suffix + suffix)
                shutil.copy2(wal_src, wal_dst)
    except Exception as e:
        typer.secho(f"   ❌ Failed to copy database: {e}", fg="red")
        return

    # Copy ChromaDB directory
    if PROD_CHROMA_PATH.exists():
        try:
            if TEST_CHROMA_PATH.exists():
                shutil.rmtree(TEST_CHROMA_PATH)
            shutil.copytree(PROD_CHROMA_PATH, TEST_CHROMA_PATH)
            typer.secho(f"   ✓ Copied ChromaDB: {TEST_CHROMA_PATH}", fg=get_text_color())
        except Exception as e:
            typer.secho(f"   ⚠️  Failed to copy ChromaDB: {e}", fg=get_warning_color())
    else:
        typer.secho("   ℹ  No production ChromaDB found (will be created on first use)", fg=get_text_color())

    typer.echo()
    typer.secho("✅ Clone complete!", fg=get_success_color())
    typer.secho("   Run '/test on' to switch to test environment.", fg=get_text_color())


def _clear_test_environment():
    """Clear all test environment data."""
    if not TEST_BASE_DIR.exists():
        typer.secho("Test environment does not exist.", fg=get_text_color())
        return

    # Confirm
    typer.secho("⚠️  This will delete all test data permanently:", fg=get_warning_color())
    typer.secho(f"   • {TEST_DB_PATH}", fg=get_text_color())
    typer.secho(f"   • {TEST_CHROMA_PATH}", fg=get_text_color())

    confirm = typer.confirm("Are you sure?")

    if not confirm:
        typer.secho("Cancelled.", fg=get_text_color())
        return

    # Disable test mode first if enabled
    if config.get("test_mode", False):
        _disable_test_mode()

    # Delete test directory
    try:
        shutil.rmtree(TEST_BASE_DIR)
        typer.secho("✅ Test environment cleared.", fg=get_success_color())
    except Exception as e:
        typer.secho(f"❌ Failed to clear test environment: {e}", fg="red")


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
    
    # ChromaDB info
    typer.echo()
    if TEST_CHROMA_PATH.exists():
        # Count files in ChromaDB directory
        chroma_files = sum(1 for _ in TEST_CHROMA_PATH.rglob("*") if _.is_file())
        typer.secho(f"Test ChromaDB: {TEST_CHROMA_PATH}", fg=get_success_color())
        typer.secho(f"   Files: {chroma_files}", fg=get_text_color())
    else:
        typer.secho(f"Test ChromaDB does not exist: {TEST_CHROMA_PATH}", fg=get_text_color(), dim=True)

    typer.echo()
    if PROD_DB_PATH.exists():
        stat = os.stat(PROD_DB_PATH)
        size_kb = stat.st_size / 1024
        typer.secho(f"Prod DB: {PROD_DB_PATH} ({size_kb:.1f} KB)", fg=get_text_color())
    else:
        typer.secho(f"Prod DB does not exist: {PROD_DB_PATH}", fg=get_warning_color())

    if PROD_CHROMA_PATH.exists():
        chroma_files = sum(1 for _ in PROD_CHROMA_PATH.rglob("*") if _.is_file())
        typer.secho(f"Prod ChromaDB: {PROD_CHROMA_PATH} ({chroma_files} files)", fg=get_text_color())


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
