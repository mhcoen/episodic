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
    max_width = config.get('wrap_width', 80)
    wrap_width = min(terminal_width - 2, max_width)  # Leave some margin
    
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
    """Specialized RAG for help documentation with dedicated collection.

    Uses a completely separate ChromaDB collection to avoid mixing help docs
    with conversation history or user documents.
    """

    def __init__(self):
        """Initialize with dedicated help-only collection."""
        from pathlib import Path
        import chromadb
        from episodic.rag_utils import SilentSentenceTransformerEmbeddingFunction

        # Use a separate ChromaDB directory for help docs
        persist_dir = Path.home() / ".episodic" / "help_chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_dir))

        # Use sentence transformers for embeddings (silent version suppresses tqdm progress bars)
        self.embedding_function = SilentSentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Get or create dedicated help collection
        # Handle case where collection exists with different embedding function config
        collection_recreated = False
        try:
            self.collection = self.client.get_or_create_collection(
                name="episodic_help_docs",
                embedding_function=self.embedding_function,
                metadata={"description": "Episodic help documentation"}
            )
        except ValueError as e:
            if "Embedding function conflict" in str(e):
                # Collection exists with incompatible embedding function - delete and recreate
                self.client.delete_collection(name="episodic_help_docs")
                self.collection = self.client.create_collection(
                    name="episodic_help_docs",
                    embedding_function=self.embedding_function,
                    metadata={"description": "Episodic help documentation"}
                )
                collection_recreated = True
            else:
                raise

        self._indexed_docs = set()
        self._load_indexed_docs()

        # If collection was recreated due to conflict, it's now empty - trigger reindex
        if collection_recreated and self.collection.count() == 0:
            self._needs_reindex = True
        else:
            self._needs_reindex = False

    def _load_indexed_docs(self):
        """Load which docs have been indexed from the collection."""
        try:
            results = self.collection.get()
            if results and 'metadatas' in results:
                for metadata in results['metadatas']:
                    if metadata and 'doc_name' in metadata:
                        self._indexed_docs.add(metadata['doc_name'])
        except Exception:
            self._indexed_docs = set()

    def _add_document(self, content: str, source: str, metadata: dict) -> tuple:
        """Add a document to the help collection with chunking."""
        import hashlib

        # Chunk size and overlap for help docs - use config values
        chunk_size = config.get('rag_chunk_size', 500)
        overlap = config.get('rag_chunk_overlap', 100)

        # Simple chunking
        chunks = []
        if len(content) <= chunk_size:
            chunks.append((content, {"start": 0, "end": len(content)}))
        else:
            start = 0
            while start < len(content):
                end = min(start + chunk_size, len(content))
                # Try to break at a newline or space
                if end < len(content):
                    last_newline = content.rfind('\n', start, end)
                    if last_newline > start + chunk_size // 2:
                        end = last_newline
                    else:
                        last_space = content.rfind(' ', start, end)
                        if last_space > start:
                            end = last_space

                chunk_text = content[start:end].strip()
                if chunk_text:
                    chunks.append((chunk_text, {"start": start, "end": end}))

                start = end - overlap if end < len(content) else end
                if start <= chunks[-1][1]["start"]:
                    start = chunks[-1][1]["end"]

        # Generate IDs and prepare data
        doc_id = hashlib.sha256(source.encode()).hexdigest()[:16]
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        chunk_texts = [c[0] for c in chunks]
        chunk_metadatas = []

        for i, (text, pos) in enumerate(chunks):
            meta = {**metadata, "source": source, "chunk_index": i, **pos}
            chunk_metadatas.append(meta)

        # Add to collection
        self.collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            metadatas=chunk_metadatas
        )

        return doc_id, len(chunks)

    def _search(self, query: str, n_results: int = 5) -> dict:
        """Search the help collection."""
        try:
            # Debug: check collection count
            count = self.collection.count()
            if config.get("debug"):
                typer.secho(f"[Help RAG] Collection has {count} documents", fg=get_text_color())

            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

            if config.get("debug"):
                typer.secho(f"[Help RAG] Query returned {len(results.get('ids', [[]])[0])} results", fg=get_text_color())

            if not results or not results['ids'] or not results['ids'][0]:
                return {'results': []}

            formatted = []
            for i in range(len(results['ids'][0])):
                doc = results['documents'][0][i] if results['documents'] else ""
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if 'distances' in results else 0
                # For normalized vectors with L2 distance: cosine_sim = 1 - (dist^2 / 2)
                # This converts L2 distance back to cosine similarity (0 to 1 range)
                score = max(0, 1 - (distance ** 2 / 2))

                formatted.append({
                    'content': doc,
                    'metadata': metadata,
                    'relevance_score': score
                })

            return {'results': formatted}
        except Exception as e:
            # Always show search errors for debugging
            typer.secho(f"[Help RAG] Search error: {e}", fg=get_error_color())
            return {'results': []}

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
                        try:
                            with open(doc_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except OSError as e:
                            typer.secho(f"Failed to read {doc}: {e}", fg=get_error_color())
                            continue

                        # Add document with clean metadata
                        doc_metadata = {
                            'title': doc,
                            'type': 'help_documentation',
                            'doc_name': doc
                        }

                        # Use dedicated help collection
                        doc_id, chunks = self._add_document(
                            content=content,
                            source=doc_path,
                            metadata=doc_metadata
                        )

                        if doc_id:
                            self._indexed_docs.add(doc)
                    else:
                        self._indexed_docs.add(doc)
                        
                except Exception as e:
                    typer.secho(f"Error checking/indexing {doc}: {str(e)}", fg=get_error_color())
    
    def search_help(self, query: str, n_results: int = 5) -> list:
        """Search help documentation using dedicated help collection."""
        # Ensure docs are indexed
        self.ensure_help_docs_indexed()

        # Check if collection is empty (docs may not have been found)
        if self.collection.count() == 0:
            typer.secho("\n⚠️  Help documentation index is empty.", fg=get_warning_color())
            typer.secho("Run '/help reindex' to rebuild the index.", fg=get_text_color())
            return []

        # Search dedicated help collection (no filtering needed - all results are help docs)
        results = self._search(query, n_results=n_results)

        # Format results for help display
        formatted_results = []
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if results['results']:
            for result in results['results']:
                metadata = result.get('metadata', {})
                content = result['content']
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

    def clear_collection(self):
        """Clear all documents from the help collection."""
        try:
            # Delete and recreate the collection
            self.client.delete_collection(name="episodic_help_docs")
            self.collection = self.client.create_collection(
                name="episodic_help_docs",
                embedding_function=self.embedding_function,
                metadata={"description": "Episodic help documentation"}
            )
            self._indexed_docs.clear()
            return True
        except Exception as e:
            if config.get("debug"):
                typer.secho(f"[Help RAG] Clear error: {e}", fg=get_error_color())
            return False


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
        categories = ["chat", "settings", "search", "history", "topics", "markdown", "voice"]
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

    # Copy/clipboard queries
    if any(phrase in query_lower for phrase in ['copy', 'clipboard', 'copy response', 'copy to clipboard']):
        typer.secho("\n📋 Copy Command", fg=get_heading_color(), bold=True)
        typer.echo()
        typer.secho("Copy conversation content to your system clipboard.", fg=get_text_color())
        typer.echo()
        typer.secho("Usage:", fg=get_text_color())
        typer.secho("  /copy              ", fg=get_system_color(), bold=True, nl=False)
        typer.secho("Copy last assistant response", fg=get_text_color())
        typer.secho("  /copy <node_id>    ", fg=get_system_color(), bold=True, nl=False)
        typer.secho("Copy specific node content", fg=get_text_color())
        typer.echo()
        typer.secho("Cross-platform support: macOS, Linux (X11/Wayland), Windows/WSL", fg=get_text_color(), dim=True)
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
    
    # Handle "reindex" subcommand
    if query_lower == "reindex":
        help_reindex()
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

    typer.secho(f"\n🔍 Searching documentation for: {query}", fg=get_heading_color())

    # Search help docs directly
    search_terms = query

    # For interface mode questions, search more specifically
    if 'advanced' in query.lower():
        search_terms = "/advanced command switch mode"
    elif 'simple' in query.lower():
        search_terms = "/simple command switch mode"

    try:
        # Ensure docs are indexed
        help_rag.ensure_help_docs_indexed()

        # Search help documentation
        search_results = help_rag.search_help(search_terms, n_results=5)

        # Debug: Show what we found
        if config.get('debug', False):
            typer.secho(f"\nDebug: Found {len(search_results)} results", fg=get_text_color())
            for i, result in enumerate(search_results):
                score = result.get('score', 0)
                typer.secho(f"Result {i+1} (score: {score:.3f}): {result['content'][:100]}...", fg=get_text_color())

        # Check if we have relevant results
        # Score is now cosine similarity (0 to 1), where 0.25+ indicates reasonable semantic match
        RELEVANCE_THRESHOLD = 0.25
        relevant_results = [r for r in search_results if r.get('score', 0) >= RELEVANCE_THRESHOLD]

        if not relevant_results:
            # No relevant results found - show helpful message
            typer.echo()
            typer.secho(f"❌ I couldn't find information on '{query}' in the documentation.", fg=get_warning_color())
            typer.echo()
            typer.secho("Try these options:", fg=get_text_color())
            typer.secho("  • Use /help all to see all available commands", fg=get_system_color())
            typer.secho("  • Try /help <category> for topic-specific help (chat, settings, topics, etc.)", fg=get_system_color())
            typer.secho("  • Browse documentation files directly:", fg=get_system_color())
            typer.secho("    - USER_GUIDE.md", fg=get_text_color(), dim=True)
            typer.secho("    - docs/cli-reference.md", fg=get_text_color(), dim=True)
            typer.secho("    - docs/quick-reference.md", fg=get_text_color(), dim=True)
            typer.secho("  • If documentation was recently updated: /help reindex", fg=get_system_color(), dim=True)
            return

        # Synthesize answer using LLM (like original implementation)
        import shutil
        terminal_width = shutil.get_terminal_size().columns
        wrap_width = min(terminal_width - 2, 80)

        # Gather context from search results
        context_parts = []
        sources = set()
        for result in relevant_results[:5]:
            content = result['content']
            source = result.get('source', 'Unknown')
            source_name = source.split('/')[-1] if '/' in source else source
            sources.add(source_name)
            context_parts.append(content)

        context = "\n\n".join(context_parts)

        # Build prompt with strict context-only instruction
        help_prompt = f"""You are the Episodic CLI help system. Answer ONLY using the documentation provided below.

DOCUMENTATION:
{context}

QUESTION: {query}

RULES:
1. ONLY use information from the DOCUMENTATION above - never use external knowledge
2. If the documentation doesn't contain the answer, say "This isn't covered in the documentation"
3. Be concise - 2-3 short paragraphs max
4. NEVER use markdown code blocks (no ``` ever)
5. Show commands on their own line, indented with 2 spaces, like:
  /muse
  /chat"""

        typer.echo()

        try:
            from episodic.llm import query_llm
            from episodic.unified_streaming import unified_stream_response, unified_stream_text

            # Use main chat model for help synthesis, with fallback to ollama/phi4
            model = config.get("model")
            if not model or model == "test-model":
                model = "ollama/phi4"

            if config.get('stream_responses', True):
                # Stream the response
                stream_tuple = query_llm(help_prompt, model=model, stream=True)
                stream_gen = stream_tuple[0] if isinstance(stream_tuple, tuple) else stream_tuple

                unified_stream_response(
                    stream_gen,
                    model=model,
                    color=get_system_color(),
                    wrap_width=wrap_width,
                    enable_tts=False  # Don't speak help output
                )
            else:
                # Non-streaming
                result = query_llm(help_prompt, model=model, stream=False)
                response_text = result[0] if isinstance(result, tuple) else result
                unified_stream_text(
                    response_text,
                    model=model,
                    color=get_system_color(),
                    wrap_width=wrap_width,
                    enable_tts=False  # Don't speak help output
                )

            # Show sources
            typer.echo()
            typer.secho("─" * min(50, wrap_width), fg=get_text_color(), dim=True)
            typer.secho("📚 Sources: ", fg=get_heading_color(), nl=False)
            typer.secho(", ".join(sorted(sources)), fg=get_system_color(), dim=True)

        except Exception as e:
            if config.get('debug'):
                typer.secho(f"LLM synthesis failed: {e}", fg=get_warning_color())
                import traceback
                traceback.print_exc()
            typer.secho("Error getting synthesized answer. Try browsing docs directly.", fg=get_warning_color())

    except Exception as e:
        typer.secho(f"\n⚠️  Error searching documentation: {str(e)}", fg=get_warning_color())
        if config.get('debug', False):
            import traceback
            traceback.print_exc()
        typer.secho("Try browsing the documentation files directly.", fg=get_text_color())


def help_reindex():
    """
    Reindex all help documentation files.

    This command clears the existing help index and re-indexes all documentation
    files into a dedicated help-only ChromaDB collection. Useful after docs updates.
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
        # Clear the global help RAG instance to force fresh initialization
        global _help_rag
        _help_rag = None

        # Get or create help RAG (creates fresh instance)
        help_rag = get_help_rag()

        # Clear the dedicated help collection
        typer.secho("\nClearing existing help index...", fg=get_text_color())
        help_rag.clear_collection()

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
                    try:
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except OSError as e:
                        typer.secho(f"     ✗ Failed to read: {e}", fg=get_error_color())
                        continue

                    # Get file size for display
                    file_size = len(content)
                    typer.secho(f"     Size: {file_size:,} characters", fg=get_text_color(), dim=True)

                    # Add document with clean metadata
                    doc_metadata = {
                        'title': doc,
                        'type': 'help_documentation',
                        'doc_name': doc
                    }

                    # Use dedicated help collection (has its own chunking)
                    doc_id, chunks = help_rag._add_document(
                        content=content,
                        source=doc_path,
                        metadata=doc_metadata
                    )

                    if doc_id:
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
        
        typer.secho(f"   • Collection: episodic_help_docs (dedicated)", fg=get_text_color())
        
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
