"""
Document management commands for Episodic.
Provides PDF loading and document-enhanced chat capabilities.
"""

import typer
from typing import Optional
import os

# Disable Chroma telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from episodic.configuration import get_system_color, get_text_color
from episodic.documents_poc import DocumentManagerPOC


class DocumentCommands:
    def __init__(self):
        self.doc_manager = DocumentManagerPOC()
        self.context_enabled = False
    
    def handle_load(self, filepath: str) -> None:
        """Load a PDF document into the conversation context."""
        success, message = self.doc_manager.load_pdf(filepath)
        
        if success:
            typer.secho(f"📄 {message}", fg=get_system_color())
            
            # Auto-enable context if this is the first document
            if not self.context_enabled and self.doc_manager.vectorstore:
                self.context_enabled = True
                typer.secho("✅ Document context automatically enabled", fg=get_system_color())
        else:
            typer.secho(f"❌ {message}", fg="red")
    
    def handle_docs(self, action: Optional[str] = None) -> None:
        """Manage loaded documents."""
        if not action or action == "status":
            # Show status
            status = self.doc_manager.get_status()
            typer.echo(status)
            
            if self.context_enabled:
                typer.secho("\n✅ Document context is ENABLED", fg=get_system_color())
            else:
                typer.secho("\n❌ Document context is DISABLED", fg=get_text_color())
                
        elif action == "clear":
            # Clear all documents
            self.doc_manager.vectorstore = None
            self.doc_manager.loaded_documents.clear()
            typer.secho("🗑️  Cleared all documents", fg=get_system_color())
            
        elif action == "enable":
            if not self.doc_manager.vectorstore:
                typer.secho("⚠️  No documents loaded yet", fg="yellow")
            else:
                self.context_enabled = True
                typer.secho("✅ Document context enabled", fg=get_system_color())
                
        elif action == "disable":
            self.context_enabled = False
            typer.secho("❌ Document context disabled", fg=get_system_color())
            
        else:
            typer.secho(f"Unknown action: {action}", fg="red")
            typer.echo("Available actions: status, clear, enable, disable")
    
    def handle_search(self, query: str) -> None:
        """Search loaded documents for relevant content."""
        if not self.doc_manager.enabled:
            typer.secho("Document features not available", fg="red")
            return
            
        if not self.doc_manager.vectorstore:
            typer.secho("No documents loaded. Use /load <pdf> first.", fg="red")
            return
        
        # Find relevant contexts
        contexts = self.doc_manager.find_relevant_context(query, k=5)
        
        if not contexts:
            typer.secho("No relevant content found.", fg=get_text_color())
            return
        
        typer.secho(f"\n🔍 Found {len(contexts)} relevant sections:\n", fg=get_system_color())
        
        for i, context in enumerate(contexts, 1):
            # Extract source info and content
            lines = context.split('\n', 1)
            source_info = lines[0] if len(lines) > 0 else ""
            content = lines[1] if len(lines) > 1 else context
            
            # Truncate content for display
            max_length = 200
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            typer.secho(f"{i}. {source_info}", fg=get_system_color())
            typer.echo(f"   {content}\n")
    
    def enhance_message_if_enabled(self, user_message: str) -> str:
        """Enhance a message with document context if enabled."""
        if self.context_enabled and self.doc_manager.enabled and self.doc_manager.vectorstore:
            enhanced = self.doc_manager.enhance_prompt_with_context(user_message)
            if enhanced != user_message:
                typer.secho("📄 Including relevant document context", fg=get_system_color())
            return enhanced
        return user_message


# Global instance
doc_commands = DocumentCommands()


# CLI command handlers
def handle_load_command(filepath: str):
    """Load a PDF document."""
    doc_commands.handle_load(filepath)


def handle_docs_command(action: Optional[str] = None):
    """Manage documents."""
    doc_commands.handle_docs(action)


def handle_search_command(query: str):
    """Search documents."""
    doc_commands.handle_search(query)