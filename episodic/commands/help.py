"""
Enhanced help command with both command listing and RAG-based documentation search.

This command shows available commands when used without arguments, and searches
documentation when given a query. RAG search works regardless of global RAG setting.
"""

import typer
import re
from typing import Optional
from episodic.config import config
from episodic.configuration import (
    get_heading_color, get_text_color, get_system_color,
    get_error_color, get_warning_color, get_success_color
)
# Import EpisodicRAG only when needed to avoid import errors
from episodic.commands.utility import help as show_commands_help
import os
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO


@contextmanager
def suppress_all_output():
    """Context manager to suppress all stdout and stderr output."""
    # Try to suppress output, but if it fails (e.g., no file descriptor), 
    # just continue without suppression
    try:
        with redirect_stdout(StringIO()):
            with redirect_stderr(StringIO()):
                yield
    except Exception:
        # If redirect fails, just yield without suppression
        yield



def _display_help_output(text: str, color: str):
    """Display help output with proper formatting and word wrapping."""
    import shutil
    import textwrap
    
    # Get terminal width for wrapping
    terminal_width = shutil.get_terminal_size().columns
    wrap_width = min(terminal_width - 2, 80)  # Leave some margin, cap at 80
    
    # For non-streaming output, we need to handle bold markers
    lines = text.split('\n')
    for line in lines:
        # Wrap long lines
        if len(line) > wrap_width:
            # First remove bold markers temporarily for accurate wrapping
            clean_line = line.replace('**', '')
            wrapped = textwrap.wrap(clean_line, width=wrap_width)
            
            # Now display each wrapped line with bold markers restored
            for wrapped_line in wrapped:
                if '**' not in line:
                    typer.secho(wrapped_line, fg=color)
                else:
                    # Restore and handle bold markers
                    # This is simplified - just displays without bold for wrapped lines
                    typer.secho(wrapped_line, fg=color)
        else:
            # Short lines - display normally with bold support
            if '**' not in line:
                typer.secho(line, fg=color)
            else:
                # Split by bold markers
                parts = re.split(r'(\*\*[^*]+\*\*)', line)
                for part in parts:
                    if part.startswith('**') and part.endswith('**'):
                        # This is bold text - remove markers and display bold
                        bold_text = part[2:-2]
                        typer.secho(bold_text, fg=color, bold=True, nl=False)
                    else:
                        # Regular text
                        typer.secho(part, fg=color, nl=False)
                typer.echo()  # Add newline at end




class HelpRAG:
    """Specialized RAG for help documentation."""
    
    def __init__(self):
        """Initialize with help-specific collection."""
        # Import here to avoid module-level import errors
        from episodic.rag import EpisodicRAG
        
        # Use composition instead of inheritance
        self.rag = EpisodicRAG()
        # Mark this as a help RAG to skip retrieval tracking
        self.rag._is_help_rag = True
        
        # Override the collection to use a help-specific one
        try:
            with suppress_all_output():
                self.collection = self.rag.client.get_or_create_collection(
                    name="episodic_help",
                    metadata={"description": "Episodic documentation for help system"}
                )
                # Update the rag's collection reference
                self.rag.collection = self.collection
        except Exception as e:
            typer.secho(f"Error creating help collection: {str(e)}", fg=get_error_color())
            raise
            
        self._indexed_docs = set()
    
    def ensure_help_docs_indexed(self):
        """Ensure help documentation is indexed."""
        help_docs = [
            "USER_GUIDE.md",
            "docs/cli-reference.md", 
            "docs/quick-reference.md",
            "docs/configuration.md",
            "README.md",
            "docs/features.md"
        ]
        
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        for doc in help_docs:
            doc_path = os.path.join(project_root, doc)
            if os.path.exists(doc_path) and doc not in self._indexed_docs:
                try:
                    # Check if already indexed by looking for the file path in metadata
                    try:
                        with suppress_all_output():
                            results = self.collection.get(
                                where={"source": doc_path},
                                limit=1
                            )
                        already_indexed = results['ids'] and len(results['ids']) > 0
                    except Exception:
                        # If query fails, assume not indexed
                        already_indexed = False
                    
                    if not already_indexed:
                        # Not indexed yet, index it
                        typer.secho(f"Indexing help documentation: {doc}...", fg=get_text_color(), dim=True)
                        # Read file content
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Add document with clean metadata
                        doc_metadata = {
                            'title': doc,
                            'type': 'help_documentation',
                            'doc_name': doc
                        }
                        doc_ids = self.rag.add_document(
                            content=content,
                            source=doc_path,
                            metadata=doc_metadata
                        )
                        success = doc_ids is not None and len(doc_ids) > 0
                        # message = f"Indexed {len(doc_ids)} chunks" if success else "Failed to index"
                        if success:
                            self._indexed_docs.add(doc)
                    else:
                        self._indexed_docs.add(doc)
                        
                except Exception as e:
                    typer.secho(f"Error checking/indexing {doc}: {str(e)}", fg=get_error_color())
    
    def search_help(self, query: str, n_results: int = 5) -> list:
        """Search help documentation."""
        # Ensure docs are indexed
        self.ensure_help_docs_indexed()
        
        # Search with help-specific prompt
        with suppress_all_output():
            results = self.rag.search(query, n_results=n_results)
        
        # Format results for help display
        formatted_results = []
        
        # Extract data from search results
        if results['results'] and len(results['results']) > 0:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            for i, result in enumerate(results['results']):
                content = result['content']
                metadata = result.get('metadata', {})
                score = result.get('relevance_score', 0)
                
                # Extract source file from metadata
                source = metadata.get('source', 'Unknown')
                source = source.replace(project_root + '/', '')
                
                formatted_results.append({
                    'content': content,
                    'source': source,
                    'score': score
                })
        
        return formatted_results


# Global help RAG instance (lazy initialization)
_help_rag = None


def get_help_rag():
    """Get or create the help RAG instance."""
    global _help_rag
    if _help_rag is None:
        _help_rag = HelpRAG()
    return _help_rag


def help(advanced: bool = False, query: Optional[str] = None):
    """
    Show help information or search documentation.
    
    Without arguments, shows available commands.
    With a query, searches documentation using RAG.
    
    Usage:
        /help                         # Show available commands
        /help all                     # Show all commands (advanced)
        /help chat                    # Show chat commands
        /help settings                # Show settings commands  
        /help search                  # Show search commands
        /help history                 # Show history commands
        /help How do I change models? # Search for model-related help
        /help What is the muse mode?  # Learn about muse mode
    """
    # Handle special cases
    if query:
        query_lower = query.lower()
        
        # Check for category help first
        categories = ["chat", "settings", "search", "history", "topics", "markdown"]
        if query_lower in categories:
            from episodic.cli_registry import show_category_help
            show_category_help(query_lower)
            return
            
        # Handle "/help all" to show all commands
        if query_lower == "all":
            # Check if we're in simple mode
            from episodic.commands.interface_mode import is_simple_mode
            if is_simple_mode():
                typer.secho("Advanced help is not available in simple mode.", fg=get_error_color())
                typer.secho("Type /advanced to switch to advanced mode first.", fg=get_warning_color())
                return
            show_commands_help(advanced=True)
            return
        
        # Otherwise, it's a documentation search query
        help_command(query)
        return
        
    # If no query, show basic command list and categories
    if not query and not advanced:
        # First show regular commands
        show_commands_help(advanced=False)
        
        # Then add help search info
        typer.secho("\n🔍 Documentation Search:", fg=get_heading_color(), bold=True)
        cmd = "/help <query>"
        padding = ' ' * max(1, 30 - len(cmd))
        typer.secho(f"  {cmd}{padding}", fg=get_system_color(), bold=True, nl=False)
        typer.secho("Search documentation", fg=get_text_color())
        typer.secho("\n  Examples:", fg=get_text_color(), dim=True)
        typer.secho("    /help How do I change models?", fg=get_system_color(), dim=True)
        typer.secho("    /help What is the muse mode?", fg=get_system_color(), dim=True)
        typer.secho("    /help configuration settings", fg=get_system_color(), dim=True)
        return
    
    if not query and advanced:
        show_commands_help(advanced=True)
        return
    
    # If we have a query, do RAG search
    try:
        help_command(query)
    except ImportError as e:
        if "chromadb" in str(e):
            typer.secho("\n⚠️  Documentation search requires ChromaDB.", fg=get_warning_color())
            typer.secho("Install with: pip install chromadb sentence-transformers", fg=get_text_color())
        else:
            raise
    except Exception as e:
        # Catch all other errors and provide fallback
        typer.secho(f"\n⚠️  Error with documentation search: {str(e)}", fg=get_warning_color())
        typer.secho("Showing all commands instead:", fg=get_text_color())
        typer.echo()
        show_commands_help(advanced=True)


def help_command(query: str):
    # Handle common queries directly for accuracy
    query_lower = query.lower()
    
    # Interface mode queries
    if any(phrase in query_lower for phrase in ['advanced mode', 'change to advanced', 'switch to advanced', 'enable advanced']):
        typer.secho("\n🔍 Searching documentation for: " + query, fg=get_heading_color())
        typer.echo()
        typer.secho("To switch to advanced mode, use:", fg=get_system_color())
        typer.secho("  /advanced", fg=get_system_color(), bold=True)
        return
    
    if any(phrase in query_lower for phrase in ['simple mode', 'change to simple', 'switch to simple', 'enable simple']):
        typer.secho("\n🔍 Searching documentation for: " + query, fg=get_heading_color())
        typer.echo()
        typer.secho("To switch to simple mode, use:", fg=get_system_color())
        typer.secho("  /simple", fg=get_system_color(), bold=True)
        return
    """
    Search Episodic documentation using RAG.
    
    Uses the main chat flow with RAG to provide synthesized answers.
    """
    if not query:
        # Show help topics
        typer.secho("\n📚 Episodic Help System", fg=get_heading_color(), bold=True)
        typer.secho("─" * 50, fg=get_heading_color())
        
        typer.secho("\nSearch the documentation:", fg=get_text_color())
        typer.secho("  /help <query>", fg=get_system_color())
        
        typer.secho("\nExample queries:", fg=get_text_color())
        examples = [
            ("/help change models", "How to change language models"),
            ("/help muse mode", "Learn about web search mode"),
            ("/help topic detection", "Understanding topic detection"),
            ("/help rag commands", "RAG and document commands"),
            ("/help configuration", "Configuration options"),
            ("/help keyboard shortcuts", "Interactive mode shortcuts"),
        ]
        for cmd, desc in examples:
            padding = ' ' * max(1, 30 - len(cmd) - 4)  # -4 for "    " indent
            typer.secho(f"    {cmd}{padding}", fg=get_system_color(), dim=True, nl=False)
            typer.secho(desc, fg=get_text_color(), dim=True)
        
        typer.secho("\nDocumentation indexed:", fg=get_text_color())
        typer.secho("  • USER_GUIDE.md - Complete user guide", fg=get_text_color(), dim=True)
        typer.secho("  • CLIReference.md - Command reference", fg=get_text_color(), dim=True)
        typer.secho("  • QUICK_REFERENCE.md - Quick command guide", fg=get_text_color(), dim=True)
        typer.secho("  • CONFIG_REFERENCE.md - Configuration guide", fg=get_text_color(), dim=True)
        return
    
    # Check if ChromaDB is available
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        typer.secho("\n⚠️  Documentation search requires ChromaDB and sentence-transformers.", fg=get_warning_color())
        typer.secho("Install with: pip install chromadb sentence-transformers", fg=get_text_color())
        typer.secho("\nAlternatively, browse the documentation files directly:", fg=get_text_color())
        typer.secho("  • USER_GUIDE.md", fg=get_text_color(), dim=True)
        typer.secho("  • docs/CLIReference.md", fg=get_text_color(), dim=True)
        typer.secho("  • QUICK_REFERENCE.md", fg=get_text_color(), dim=True)
        return
    
    # Initialize help RAG
    try:
        help_rag = get_help_rag()
    except Exception as e:
        typer.secho(f"\n⚠️  Error initializing help system: {str(e)}", fg=get_warning_color())
        return
    
    # Ensure docs are indexed
    help_rag.ensure_help_docs_indexed()
    
    # Save current RAG state
    original_rag_enabled = config.get('rag_enabled', False)
    original_web_search_enabled = config.get('web_search_enabled', False)
    original_collection = None
    
    try:
        # Temporarily use help collection for RAG
        from episodic.rag import get_rag_system
        rag_system = get_rag_system()
        if rag_system:
            original_collection = rag_system.collection
            rag_system.collection = help_rag.collection
        
        # Enable RAG temporarily and disable web search for help
        config.set('rag_enabled', True)
        config.set('web_search_enabled', False)
        
        # Create a simple, direct prompt - context will be added FIRST by RAG
        help_prompt = f"""Answer this Episodic CLI question: {query}

Instructions:
- Use the documentation context provided above to answer
- If you see relevant commands in the context, use them
- Commands in Episodic always start with /
- Be helpful and answer the user's intent

Format: No markdown code blocks. Indent commands with 2 spaces. Be concise."""
        
        typer.secho(f"\n🔍 Searching documentation for: {query}", fg=get_heading_color())
        
        # Search help docs directly for better relevance
        search_terms = query
        
        # For interface mode questions, search more specifically
        if 'advanced' in query.lower():
            # Search for the actual command
            search_terms = "/advanced command switch mode"
        elif 'simple' in query.lower():
            search_terms = "/simple command switch mode"
        
        try:
            with suppress_all_output():
                # Use help_rag for better search
                search_results = help_rag.search_help(search_terms, n_results=5)
                
                # Build context from search results
                context_parts = []
                for result in search_results[:3]:  # Use top 3 results
                    context_parts.append(result['content'])
                
                # Debug: Show what we found
                if config.get('debug', False):
                    typer.secho(f"\nDebug: Found {len(search_results)} results", fg=get_text_color())
                    for i, result in enumerate(search_results[:3]):
                        typer.secho(f"Result {i+1} preview: {result['content'][:100]}...", fg=get_text_color())
                
                # Create enhanced prompt with context
                if context_parts:
                    context = "\n\n".join(context_parts)
                    enhanced_prompt = f"Documentation Context:\n{context}\n\n{help_prompt}"
                else:
                    enhanced_prompt = help_prompt
        except Exception:
            # Fallback to regular enhancement
            enhanced_prompt = rag_system.enhance_with_context(help_prompt, include_web=False)
        
        # sources_used is not returned by enhance_with_context
        sources_used = None
        
        if sources_used and config.get('debug', False):
            typer.secho(f"📚 Found relevant documentation from: {', '.join(sources_used)}", 
                      fg=get_text_color(), dim=True)
        
        # Query the LLM with the enhanced prompt
        from episodic.llm import query_llm
        from episodic.unified_streaming import unified_stream_response
        
        # Get the model to use
        model = config.get('model', 'gpt-3.5-turbo')
        
        # Add a blank line before the answer
        typer.echo()
        
        # Check if streaming is enabled
        if config.get('stream_responses', True):
            try:
                # Query with streaming - returns (generator, None) tuple
                stream_tuple = query_llm(enhanced_prompt, model=model, stream=True)
                # Extract the generator from the tuple
                stream_gen = stream_tuple[0] if isinstance(stream_tuple, tuple) else stream_tuple
                
                # Stream the response - use regular streaming with wrapping
                # Don't use format preservation as it prevents proper word wrapping
                import shutil
                terminal_width = shutil.get_terminal_size().columns
                wrap_width = min(terminal_width - 2, 80)  # Leave margin, cap at 80
                
                response_text = unified_stream_response(
                    stream_gen,
                    model=model,
                    color=get_system_color(),
                    wrap_width=wrap_width,
                    preserve_formatting=False  # Ensure wrapping is enabled
                )
                
            except Exception as stream_error:
                if config.get('debug', False):
                    typer.secho(f"Streaming error: {stream_error}", fg=get_warning_color())
                # Fallback to non-streaming
                result = query_llm(enhanced_prompt, model=model, stream=False)
                response_text = result[0] if isinstance(result, tuple) else result
                _display_help_output(response_text, get_system_color())
        else:
            # Query without streaming
            result = query_llm(enhanced_prompt, model=model, stream=False)
            # Extract response text from tuple
            response_text = result[0] if isinstance(result, tuple) else result
            _display_help_output(response_text, get_system_color())
        
    except Exception as e:
        typer.secho(f"\n⚠️  Error getting help: {str(e)}", fg=get_warning_color())
        typer.secho("Try browsing the documentation files directly.", fg=get_text_color())
    
    finally:
        # Restore original RAG state
        config.set('rag_enabled', original_rag_enabled)
        config.set('web_search_enabled', original_web_search_enabled)
        if 'original_show_citations' in locals() and 'original_show_citations' in dir():
            config.set('rag_show_citations', original_show_citations)
        if rag_system and original_collection:
            rag_system.collection = original_collection


def help_reindex():
    """
    Reindex all help documentation files.
    
    This command clears the existing help index and re-indexes all documentation
    files listed in HELP_INDEXED_FILES.md. Useful after documentation updates.
    """
    typer.secho("\n📚 Reindexing Help Documentation", fg=get_heading_color(), bold=True)
    typer.secho("─" * 50, fg=get_heading_color())
    
    # Check if ChromaDB is available
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        typer.secho("\n⚠️  Documentation indexing requires ChromaDB and sentence-transformers.", fg=get_warning_color())
        typer.secho("Install with: pip install chromadb sentence-transformers", fg=get_text_color())
        return
    
    try:
        # Get or create help RAG
        help_rag = get_help_rag()
        
        # Clear existing index
        typer.secho("\nClearing existing help index...", fg=get_text_color())
        try:
            with suppress_all_output():
                # Delete and recreate the collection
                help_rag.rag.client.delete_collection(name="episodic_help")
                help_rag.collection = help_rag.rag.client.create_collection(
                    name="episodic_help",
                    metadata={"description": "Episodic documentation for help system"}
                )
                help_rag.rag.collection = help_rag.collection
        except Exception as e:
            # Collection might not exist, that's okay
            if config.get('debug', False):
                typer.secho(f"Note: {str(e)}", fg=get_text_color(), dim=True)
        
        # Clear the indexed docs set
        help_rag._indexed_docs.clear()
        
        # Define help docs
        help_docs = [
            "USER_GUIDE.md",
            "docs/cli-reference.md", 
            "docs/quick-reference.md",
            "docs/configuration.md",
            "README.md",
            "docs/features.md"
        ]
        
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        indexed_count = 0
        total_chunks = 0
        
        typer.secho("\nIndexing documentation files:", fg=get_text_color())
        
        for doc in help_docs:
            doc_path = os.path.join(project_root, doc)
            if os.path.exists(doc_path):
                typer.secho(f"\n  📄 {doc}", fg=get_system_color(), bold=True)
                
                try:
                    # Read file content
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Get file size for display
                    file_size = len(content)
                    typer.secho(f"     Size: {file_size:,} characters", fg=get_text_color(), dim=True)
                    
                    # Add document with clean metadata
                    doc_metadata = {
                        'title': doc,
                        'type': 'help_documentation',
                        'doc_name': doc
                    }
                    
                    doc_ids = help_rag.rag.add_document(
                        content=content,
                        source=doc_path,
                        metadata=doc_metadata
                    )
                    
                    if doc_ids:
                        chunks = len(doc_ids)
                        total_chunks += chunks
                        indexed_count += 1
                        typer.secho(f"     ✓ Indexed {chunks} chunks", fg=get_success_color())
                        help_rag._indexed_docs.add(doc)
                    else:
                        typer.secho(f"     ✗ Failed to index", fg=get_error_color())
                        
                except Exception as e:
                    typer.secho(f"     ✗ Error: {str(e)}", fg=get_error_color())
            else:
                typer.secho(f"\n  ⚠️  {doc} - File not found", fg=get_warning_color())
        
        # Summary
        typer.secho("\n" + "─" * 50, fg=get_heading_color())
        
        if indexed_count == 0:
            typer.secho(f"\n❌ Reindexing Failed!", fg="red", bold=True)
            typer.secho(f"   • Files indexed: {indexed_count}/{len(help_docs)}", fg=get_error_color())
            typer.secho(f"   • Total chunks: {total_chunks}", fg=get_error_color())
            typer.secho(f"   • All files failed to index due to errors above", fg=get_error_color())
            typer.secho(f"\n⚠️  The help search will not work until indexing succeeds.", fg=get_warning_color())
        elif indexed_count < len(help_docs):
            typer.secho(f"\n⚠️  Reindexing Partially Complete!", fg="yellow", bold=True)
            typer.secho(f"   • Files indexed: {indexed_count}/{len(help_docs)}", fg=get_warning_color())
            typer.secho(f"   • Total chunks: {total_chunks}", fg=get_text_color())
            typer.secho(f"   • Some files failed to index (see errors above)", fg=get_warning_color())
        else:
            typer.secho(f"\n✅ Reindexing Complete!", fg="green", bold=True)
            typer.secho(f"   • Files indexed: {indexed_count}/{len(help_docs)}", fg=get_text_color())
            typer.secho(f"   • Total chunks: {total_chunks}", fg=get_text_color())
        
        typer.secho(f"   • Collection: episodic_help", fg=get_text_color())
        
        if config.get('rag_preserve_formatting', True):
            typer.secho(f"   • Format preservation: Enabled", fg=get_text_color())
        
        if indexed_count > 0:
            typer.secho("\nYou can now search documentation with:", fg=get_text_color())
            typer.secho("  /help <query>", fg=get_system_color())
        
    except Exception as e:
        typer.secho(f"\n❌ Error during reindexing: {str(e)}", fg=get_error_color())
        if config.get('debug', False):
            import traceback
            traceback.print_exc()
