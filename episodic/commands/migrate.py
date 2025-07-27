"""
Migration command for RAG collection separation.
"""

import typer
from typing import Optional

from episodic.configuration import get_text_color, get_system_color, get_error_color
from episodic.rag_migration import (
    check_migration_needed,
    migrate_to_multi_collection,
    rollback_migration
)


def migrate_command(action: Optional[str] = None, *args):
    """
    Migrate RAG data to new multi-collection system.
    
    Usage:
        /migrate              # Check migration status
        /migrate run          # Run migration
        /migrate dry-run      # Preview what would be migrated
        /migrate rollback     # Rollback migration (for testing)
    """
    if not action:
        # Check migration status
        if check_migration_needed():
            typer.secho("\n⚠️  Migration needed", fg=get_system_color())
            typer.secho("Your RAG data needs to be migrated to the new collection system.", fg=get_text_color())
            typer.secho("\nRun '/migrate dry-run' to preview the migration", fg=get_text_color())
            typer.secho("Run '/migrate run' to perform the migration", fg=get_text_color())
        else:
            typer.secho("\n✅ No migration needed", fg=get_system_color())
            typer.secho("Your RAG data is already using the multi-collection system.", fg=get_text_color())
        return
    
    if action == "run":
        # Run actual migration
        verbose = "--verbose" in args or "-v" in args
        success = migrate_to_multi_collection(dry_run=False, verbose=verbose)
        
        if success:
            typer.secho("\n🎉 Migration completed successfully!", fg=get_system_color())
            typer.secho("Your memory and document collections are now separate.", fg=get_text_color())
        else:
            typer.secho("\n❌ Migration failed", fg=get_error_color())
            typer.secho("Please check the error messages above.", fg=get_text_color())
    
    elif action == "dry-run":
        # Preview migration
        migrate_to_multi_collection(dry_run=True, verbose=True)
        typer.secho("\nThis was a dry run - no changes were made.", fg=get_text_color())
        typer.secho("Run '/migrate run' to perform the actual migration.", fg=get_text_color())
    
    elif action == "rollback":
        # Rollback migration (mainly for testing)
        if not typer.confirm("\n⚠️  Rollback migration? This will clear the new collections."):
            typer.secho("Rollback cancelled.", fg=get_text_color())
            return
        
        rollback_migration()
        typer.secho("\n✅ Migration rolled back", fg=get_system_color())
    
    else:
        typer.secho(f"Unknown action: {action}", fg=get_error_color())
        typer.secho("Available actions: run, dry-run, rollback", fg=get_text_color())