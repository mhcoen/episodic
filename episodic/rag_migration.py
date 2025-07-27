"""
Migration utilities for moving from single collection to multi-collection RAG system.
"""

import os
import warnings
import logging
from typing import Dict, List, Any
from datetime import datetime

# Suppress ChromaDB warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger('chromadb').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*telemetry.*")

import chromadb
from chromadb.utils import embedding_functions
import typer

from episodic.config import config
from episodic.configuration import get_text_color, get_system_color, get_success_color, get_error_color
from episodic.debug_utils import debug_print
from episodic.rag_utils import suppress_chromadb_telemetry


def check_migration_needed() -> bool:
    """Check if migration from old single collection is needed."""
    # Check if migration already completed
    if config.get("collection_migration_completed", False):
        return False
    
    # Check if old collection exists
    db_path = os.path.expanduser("~/.episodic/rag/chroma")
    if not os.path.exists(db_path):
        return False
    
    try:
        client = chromadb.PersistentClient(path=db_path)
        # Try to get the old collection
        client.get_collection(name="episodic_docs")
        return True
    except:
        return False


def count_documents_by_source() -> Dict[str, int]:
    """Count documents in old collection by source type."""
    db_path = os.path.expanduser("~/.episodic/rag/chroma")
    client = chromadb.PersistentClient(path=db_path)
    
    # Get embedding function
    embedding_model = config.get("rag_embedding_model", "all-MiniLM-L6-v2")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    
    try:
        old_collection = client.get_collection(
            name="episodic_docs",
            embedding_function=embedding_function
        )
        
        # Get all documents
        all_data = old_collection.get()
        
        # Count by source
        source_counts = {}
        if all_data['metadatas']:
            for metadata in all_data['metadatas']:
                source = metadata.get('source', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
        
        return source_counts
    except Exception as e:
        debug_print(f"Error counting documents: {e}", category="migration")
        return {}


def migrate_to_multi_collection(dry_run: bool = True, verbose: bool = False) -> bool:
    """
    Migrate from single collection to multi-collection system.
    
    Args:
        dry_run: If True, only show what would be migrated
        verbose: If True, show detailed progress
        
    Returns:
        True if migration successful, False otherwise
    """
    typer.secho("\n🔄 RAG Collection Migration", fg=get_system_color(), bold=True)
    typer.secho("─" * 50, fg=get_system_color())
    
    # Check if migration needed
    if not check_migration_needed():
        typer.secho("✅ No migration needed", fg=get_success_color())
        return True
    
    # Count documents
    source_counts = count_documents_by_source()
    if not source_counts:
        typer.secho("❌ Could not read old collection", fg=get_error_color())
        return False
    
    # Show what will be migrated
    typer.secho("\nDocuments to migrate:", fg=get_text_color())
    total_docs = 0
    for source, count in source_counts.items():
        typer.secho(f"  {source}: {count} documents", fg=get_text_color())
        total_docs += count
    typer.secho(f"\nTotal: {total_docs} documents", fg=get_text_color(), bold=True)
    
    if dry_run:
        typer.secho("\n🔍 DRY RUN - No changes will be made", fg=get_system_color())
        typer.secho("\nMigration plan:", fg=get_text_color())
        typer.secho("  • Conversation documents → episodic_conversation_memory", fg=get_text_color())
        typer.secho("  • Other documents → episodic_user_docs", fg=get_text_color())
        return True
    
    # Confirm migration
    if not typer.confirm(f"\nMigrate {total_docs} documents to new collections?"):
        typer.secho("Migration cancelled", fg=get_text_color())
        return False
    
    # Perform migration
    typer.secho("\nStarting migration...", fg=get_system_color())
    
    try:
        # Initialize clients
        db_path = os.path.expanduser("~/.episodic/rag/chroma")
        client = chromadb.PersistentClient(path=db_path)
        
        # Get embedding function
        embedding_model = config.get("rag_embedding_model", "all-MiniLM-L6-v2")
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Get old collection
        old_collection = client.get_collection(
            name="episodic_docs",
            embedding_function=embedding_function
        )
        
        # Create new collections
        from episodic.rag_collections import get_multi_collection_rag, CollectionType
        multi_rag = get_multi_collection_rag()
        
        # Get all data from old collection
        all_data = old_collection.get()
        
        if not all_data['ids']:
            typer.secho("No documents to migrate", fg=get_text_color())
            return True
        
        # Migrate documents
        migrated_conversation = 0
        migrated_user_docs = 0
        errors = 0
        
        with typer.progressbar(range(len(all_data['ids'])), label="Migrating") as progress:
            for i in progress:
                try:
                    doc_id = all_data['ids'][i]
                    document = all_data['documents'][i]
                    metadata = all_data['metadatas'][i] if all_data['metadatas'] else {}
                    
                    # Determine target collection
                    source = metadata.get('source', 'unknown')
                    if source == 'conversation':
                        collection_type = CollectionType.CONVERSATION
                        migrated_conversation += 1
                    else:
                        collection_type = CollectionType.USER_DOCS
                        migrated_user_docs += 1
                    
                    # Add to new collection
                    collection = multi_rag.get_collection(collection_type)
                    collection.add(
                        ids=[doc_id],
                        documents=[document],
                        metadatas=[metadata]
                    )
                    
                    if verbose:
                        typer.secho(f"  Migrated {doc_id[:8]} to {collection_type}", fg=get_text_color())
                    
                except Exception as e:
                    errors += 1
                    if verbose:
                        typer.secho(f"  Error migrating {doc_id[:8]}: {e}", fg=get_error_color())
        
        # Show results
        typer.secho("\n✅ Migration completed!", fg=get_success_color())
        typer.secho(f"  Conversation memories: {migrated_conversation}", fg=get_text_color())
        typer.secho(f"  User documents: {migrated_user_docs}", fg=get_text_color())
        if errors > 0:
            typer.secho(f"  Errors: {errors}", fg=get_error_color())
        
        # Mark migration as completed
        config.set("collection_migration_completed", True)
        config.save_setting("collection_migration_completed", True)
        
        # Ask about deleting old collection
        if typer.confirm("\nDelete old collection (episodic_docs)?"):
            try:
                client.delete_collection(name="episodic_docs")
                typer.secho("✅ Old collection deleted", fg=get_success_color())
            except Exception as e:
                typer.secho(f"⚠️  Could not delete old collection: {e}", fg=get_error_color())
        
        return True
        
    except Exception as e:
        typer.secho(f"\n❌ Migration failed: {e}", fg=get_error_color())
        return False


def rollback_migration():
    """Rollback the migration (for testing/recovery)."""
    # Reset migration flag
    config.set("collection_migration_completed", False)
    config.save_setting("collection_migration_completed", False)
    
    # Clear new collections
    from episodic.rag_collections import get_multi_collection_rag, CollectionType
    multi_rag = get_multi_collection_rag()
    
    conversation_cleared = multi_rag.clear_collection(CollectionType.CONVERSATION)
    user_docs_cleared = multi_rag.clear_collection(CollectionType.USER_DOCS)
    
    typer.secho(f"Rolled back migration:", fg=get_system_color())
    typer.secho(f"  Cleared {conversation_cleared} conversation memories", fg=get_text_color())
    typer.secho(f"  Cleared {user_docs_cleared} user documents", fg=get_text_color())
    typer.secho("  Reset migration flag", fg=get_text_color())