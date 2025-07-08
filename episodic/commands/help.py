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


class HelpRAG:
    """Specialized RAG for help documentation."""
    
    def __init__(self):
        """Initialize with help-specific collection."""
        # Import here to avoid module-level import errors
        from episodic.rag import EpisodicRAG
        
        # Use composition instead of inheritance
        self.rag = EpisodicRAG()
        
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
    
    This uses RAG to search indexed documentation regardless of the global RAG setting.
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
    
    # Search documentation
    typer.secho(f"\n🔍 Searching documentation for: {query}", fg=get_heading_color())
    results = help_rag.search_help(query, n_results=3)
    
    if not results:
        typer.secho("No relevant documentation found.", fg="yellow")
        typer.secho("Try different search terms or check the full documentation.", fg=get_text_color(), dim=True)
        return
    
    # Synthesize a helpful answer using the LLM
    typer.secho("\n💭 Generating answer...", fg=get_text_color(), dim=True)
    
    # Build context from search results, limiting each to reasonable size
    context_parts = []
    max_chars_per_result = 1000  # Limit each result to avoid token limits
    
    for result in results:
        content = result['content']
        if len(content) > max_chars_per_result:
            # Try to find a good breaking point
            content = content[:max_chars_per_result]
            last_period = content.rfind('.')
            last_newline = content.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > max_chars_per_result * 0.7:  # If we found a good break point
                content = content[:break_point + 1]
            content += "..."
        
        context_parts.append(f"From {result['source']}:\n{content}\n")
    
    context = "\n---\n".join(context_parts)
    
    # Create synthesis prompt
    synthesis_prompt = f"""Based on the following documentation excerpts, provide a clear and helpful answer to the user's question: "{query}"

Documentation context:
{context}

Please provide a concise, practical answer that directly addresses the user's question. Focus on:
1. What they need to know
2. The specific commands or steps to use
3. Any important configuration options
4. Brief examples if helpful

Keep the answer focused and actionable."""

    # Get the synthesis model or use main model
    from episodic.llm import query_llm
    synthesis_model = config.get('synthesis_model', config.get('model', 'gpt-3.5-turbo'))
    
    try:
        # Query the LLM for synthesis
        response_data = query_llm(synthesis_prompt, model=synthesis_model, stream=False)
        
        # Extract the actual response text
        if isinstance(response_data, tuple):
            response_text = response_data[0]
        else:
            response_text = response_data
        
        # Display the synthesized answer
        typer.secho("\n📚 Answer:", fg=get_heading_color(), bold=True)
        typer.secho("─" * 50, fg=get_heading_color())
        
        # Display the response with color
        typer.secho(response_text, fg=config.get('llm_color', 'green'))
        
        # Show sources
        typer.secho("\n📄 Sources:", fg=get_text_color(), dim=True)
        seen_sources = set()
        for result in results:
            if result['source'] not in seen_sources:
                typer.secho(f"  • {result['source']}", fg=get_text_color(), dim=True)
                seen_sources.add(result['source'])
                
    except Exception as e:
        # Fallback to showing raw results if synthesis fails
        typer.secho(f"\n⚠️  Could not generate answer: {str(e)}", fg="yellow")
        typer.secho("\nShowing raw documentation excerpts:", fg=get_text_color())
        
        for i, result in enumerate(results):
            typer.secho(f"\n📄 From {result['source']}:", fg=get_system_color(), bold=True)
            typer.secho("─" * 50, fg=get_system_color())
            
            # Show truncated content
            content = result['content']
            if len(content) > 500:
                content = content[:500] + "..."
            typer.echo(content)
            
            if i < len(results) - 1:
                typer.secho("\n" + "─" * 50, fg=get_text_color(), dim=True)