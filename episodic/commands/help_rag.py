"""Help documentation RAG search (dedicated ChromaDB collection).

Split out of commands/help.py to keep that module under the size limit.
"""

import os
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO

import typer

from episodic.config import config
from episodic.configuration import (
    get_heading_color, get_text_color, get_system_color,
    get_error_color, get_warning_color, get_success_color,
)


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


