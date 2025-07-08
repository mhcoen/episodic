"""
Enhanced help command with both command listing and RAG-based documentation search.

This command shows available commands when used without arguments, and searches
documentation when given a query. RAG search works regardless of global RAG setting.
"""

import typer
from typing import Optional
from episodic.config import config
from episodic.configuration import get_heading_color, get_text_color, get_system_color
from functools import wraps
# Import EpisodicRAG only when needed to avoid import errors
from episodic.unified_streaming import unified_stream_response
from episodic.commands.utility import help as show_commands_help
import os
import re


def _clean_help_output(text: str) -> str:
    """Remove markdown code block markers from help output."""
    # Remove ```bash and ``` markers
    text = re.sub(r'```bash\s*\n', '  ', text)
    text = re.sub(r'```\s*\n', '', text)
    # Also remove standalone ``` on their own lines
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
    return text


def _display_help_output(text: str, color: str):
    """Display help output with proper formatting for bold text."""
    lines = text.split('\n')
    for line in lines:
        _process_line_for_display(line, color)


def _process_line_for_display(line: str, color: str, newline: bool = True):
    """Process a line of text to handle markdown bold markers and display it."""
    import re
    
    # Check if line contains bold markers
    if '**' not in line:
        typer.secho(line, fg=color, nl=newline)
        return
    
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
    
    if newline:
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
            self.collection = self.rag.client.get_or_create_collection(
                name="episodic_help",
                metadata={"description": "Episodic documentation for help system"}
            )
            # Update the rag's collection reference
            self.rag.collection = self.collection
        except Exception as e:
            typer.secho(f"Error creating help collection: {str(e)}", fg="red")
            raise
            
        self._indexed_docs = set()
    
    def ensure_help_docs_indexed(self):
        """Ensure help documentation is indexed."""
        help_docs = [
            "USER_GUIDE.md",
            "docs/CLIReference.md", 
            "QUICK_REFERENCE.md",
            "CONFIG_REFERENCE.md",
            "README.md",
            "docs/LLMProviders.md",
            "docs/WebSearchProviders.md",
            "docs/WEB_SYNTHESIS.md"
        ]
        
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        for doc in help_docs:
            doc_path = os.path.join(project_root, doc)
            if os.path.exists(doc_path) and doc not in self._indexed_docs:
                try:
                    # Check if already indexed by looking for the file path in metadata
                    try:
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
                        message = f"Indexed {len(doc_ids)} chunks" if success else "Failed to index"
                        if success:
                            self._indexed_docs.add(doc)
                    else:
                        self._indexed_docs.add(doc)
                        
                except Exception as e:
                    typer.secho(f"Error checking/indexing {doc}: {str(e)}", fg="red")
    
    def search_help(self, query: str, n_results: int = 5) -> list:
        """Search help documentation."""
        # Ensure docs are indexed
        self.ensure_help_docs_indexed()
        
        # Search with help-specific prompt
        results = self.rag.search(query, n_results=n_results)
        
        # Format results for help display
        formatted_results = []
        
        # Extract data from search results
        if results['documents'] and len(results['documents']) > 0:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            for i in range(len(results['documents'])):
                content = results['documents'][i]
                metadata = results['metadatas'][i] if results['metadatas'] and i < len(results['metadatas']) else {}
                distance = results['distances'][i] if results['distances'] and i < len(results['distances']) else 0
                
                # Extract source file from metadata
                source = metadata.get('source', 'Unknown')
                source = source.replace(project_root + '/', '')
                
                formatted_results.append({
                    'content': content,
                    'source': source,
                    'score': 2.0 - distance  # Convert L2 distance to similarity-like score
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
        /help                        # Show available commands
        /help --all                 # Show all commands (advanced)
        /help change models          # Search for model-related help
        /help muse mode             # Learn about muse mode
        /help topic detection       # Get help on topic detection
        /help configuration         # Find configuration options
    """
    # If no query, show command list
    if not query and not advanced:
        # First show regular commands
        show_commands_help(advanced=False)
        
        # Then add help search info
        typer.secho("\n🔍 Documentation Search:", fg=get_heading_color(), bold=True)
        typer.secho("  /help <query>             ", fg=get_system_color(), bold=True, nl=False)
        typer.secho("Search documentation", fg=get_text_color())
        typer.secho("\n  Examples:", fg=get_text_color(), dim=True)
        typer.secho("    /help change models", fg=get_system_color(), dim=True)
        typer.secho("    /help muse mode", fg=get_system_color(), dim=True)
        typer.secho("    /help configuration", fg=get_system_color(), dim=True)
        return
    
    if not query and advanced:
        show_commands_help(advanced=True)
        return
    
    # If we have a query, do RAG search
    try:
        help_command(query)
    except ImportError as e:
        if "chromadb" in str(e):
            typer.secho("\n⚠️  Documentation search requires ChromaDB.", fg="yellow")
            typer.secho("Install with: pip install chromadb sentence-transformers", fg=get_text_color())
        else:
            raise


def help_command(query: str):
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
        typer.secho("  /help change models         ", fg=get_system_color(), nl=False)
        typer.secho("# How to change language models", fg=get_text_color(), dim=True)
        typer.secho("  /help muse mode            ", fg=get_system_color(), nl=False)
        typer.secho("# Learn about web search mode", fg=get_text_color(), dim=True)
        typer.secho("  /help topic detection      ", fg=get_system_color(), nl=False)
        typer.secho("# Understanding topic detection", fg=get_text_color(), dim=True)
        typer.secho("  /help rag commands         ", fg=get_system_color(), nl=False)
        typer.secho("# RAG and document commands", fg=get_text_color(), dim=True)
        typer.secho("  /help configuration        ", fg=get_system_color(), nl=False)
        typer.secho("# Configuration options", fg=get_text_color(), dim=True)
        typer.secho("  /help keyboard shortcuts   ", fg=get_system_color(), nl=False)
        typer.secho("# Interactive mode shortcuts", fg=get_text_color(), dim=True)
        
        typer.secho("\nDocumentation indexed:", fg=get_text_color())
        typer.secho("  • USER_GUIDE.md - Complete user guide", fg=get_text_color(), dim=True)
        typer.secho("  • CLIReference.md - Command reference", fg=get_text_color(), dim=True)
        typer.secho("  • QUICK_REFERENCE.md - Quick command guide", fg=get_text_color(), dim=True)
        typer.secho("  • CONFIG_REFERENCE.md - Configuration guide", fg=get_text_color(), dim=True)
        return
    
    # Check if ChromaDB is available
    try:
        import chromadb
        import sentence_transformers
    except ImportError:
        typer.secho("\n⚠️  Documentation search requires ChromaDB and sentence-transformers.", fg="yellow")
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
        typer.secho(f"\n⚠️  Error initializing help system: {str(e)}", fg="yellow")
        return
    
    # Ensure docs are indexed
    help_rag.ensure_help_docs_indexed()
    
    # Save current RAG state
    original_rag_enabled = config.get('rag_enabled', False)
    original_collection = None
    
    try:
        # Temporarily use help collection for RAG
        from episodic.rag import get_rag_system
        rag_system = get_rag_system()
        if rag_system:
            original_collection = rag_system.collection
            rag_system.collection = help_rag.collection
        
        # Enable RAG temporarily
        config.set('rag_enabled', True)
        
        # Create a help-specific prompt that will use RAG enhancement
        help_prompt = f"""Please provide a clear, practical answer to this question about using Episodic: {query}

Important: Format commands by indenting them with 2 spaces, but do NOT use markdown code blocks (no ```bash or ``` markers). Just indent the commands."""
        
        typer.secho(f"\n🔍 Searching documentation for: {query}", fg=get_heading_color())
        
        # Enhance the prompt with RAG context
        enhanced_prompt, sources_used = rag_system.enhance_with_context(help_prompt)
        
        if sources_used and config.get('debug', False):
            typer.secho(f"📚 Found relevant documentation from: {', '.join(sources_used)}", 
                      fg=get_text_color(), dim=True)
        
        # Query the LLM with the enhanced prompt
        from episodic.llm import query_llm
        from episodic.unified_streaming import unified_stream_response
        
        # Get the model to use
        model = config.get('model', 'gpt-3.5-turbo')
        
        # Display the response header
        typer.secho("\n📚 Answer:", fg=get_heading_color(), bold=True)
        typer.secho("─" * 50, fg=get_heading_color())
        
        # Check if streaming is enabled
        if config.get('stream_responses', True):
            try:
                # Query with streaming - returns (generator, None) tuple
                stream_tuple = query_llm(enhanced_prompt, model=model, stream=True)
                # Extract the generator from the tuple
                stream_gen = stream_tuple[0] if isinstance(stream_tuple, tuple) else stream_tuple
                
                # Stream directly with cleaning
                from episodic.llm import process_stream_response
                
                response_parts = []
                buffer = ""
                llm_color = config.get('llm_color', 'green')
                
                for chunk in process_stream_response(stream_gen, model):
                    buffer += chunk
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        
                        # Skip markdown code block markers
                        if line.strip() == '```bash':
                            typer.secho('  ', fg=llm_color, nl=False)
                            response_parts.append('  ')
                            continue
                        elif line.strip() == '```':
                            continue
                        
                        # Process the line for markdown bold markers
                        processed_line = _process_line_for_display(line, llm_color)
                        response_parts.append(line + '\n')
                
                # Print any remaining buffer
                if buffer and buffer.strip() not in ['```bash', '```']:
                    _process_line_for_display(buffer, llm_color, newline=False)
                    response_parts.append(buffer)
                
                # Final newline if needed
                if response_parts and not response_parts[-1].endswith('\n'):
                    typer.echo()
                
                response_text = ''.join(response_parts)
                
            except Exception as stream_error:
                if config.get('debug', False):
                    typer.secho(f"Streaming error: {stream_error}", fg="yellow")
                # Fallback to non-streaming
                result = query_llm(enhanced_prompt, model=model, stream=False)
                response_text = result[0] if isinstance(result, tuple) else result
                # Clean up markdown code blocks before displaying
                response_text = _clean_help_output(response_text)
                _display_help_output(response_text, config.get('llm_color', 'green'))
        else:
            # Query without streaming
            result = query_llm(enhanced_prompt, model=model, stream=False)
            # Extract response text from tuple
            response_text = result[0] if isinstance(result, tuple) else result
            # Clean up markdown code blocks before displaying
            response_text = _clean_help_output(response_text)
            _display_help_output(response_text, config.get('llm_color', 'green'))
        
    except Exception as e:
        typer.secho(f"\n⚠️  Error getting help: {str(e)}", fg="yellow")
        typer.secho("Try browsing the documentation files directly.", fg=get_text_color())
    
    finally:
        # Restore original RAG state
        config.set('rag_enabled', original_rag_enabled)
        if 'original_show_citations' in locals():
            config.set('rag_show_citations', original_show_citations)
        if rag_system and original_collection:
            rag_system.collection = original_collection