"""
Hermetic ChromaDB test fixtures.

Provides isolated Chroma clients and collections for tests that need
real ChromaDB functionality without sharing state or file locks.

Usage:
    @pytest.mark.serial
    class TestWithChroma:
        def test_something(self, hermetic_chroma):
            # hermetic_chroma.client - PersistentClient in tmp_path
            # hermetic_chroma.collection - unique collection per test
            # hermetic_chroma.add_doc("content") - helper
            pass
"""

import os
import uuid
import shutil
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Suppress ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings


class DummyEmbeddingFunction:
    """
    Deterministic embedding function for tests.

    Returns fixed-dimension vectors that vary by input hash,
    ensuring consistent results without loading ML models.
    """

    EMBEDDING_DIM = 384  # Matches all-MiniLM-L6-v2

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        """Generate deterministic embeddings from text hashes."""
        results = []
        for text in input_texts:
            # Create deterministic vector from text hash
            text_hash = hash(text) & 0xFFFFFFFF
            vec = []
            for i in range(self.EMBEDDING_DIM):
                # Generate pseudo-random float from hash
                val = ((text_hash * (i + 1)) % 1000) / 1000.0
                vec.append(val - 0.5)  # Center around 0
            results.append(vec)
        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.__call__([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return self.__call__(texts)

    def is_legacy(self) -> bool:
        """Return True to indicate legacy embedding function."""
        return True


@dataclass
class HermeticChromaContext:
    """
    Container for hermetic Chroma test resources.

    Attributes:
        client: PersistentClient isolated to tmp_path
        collection: Uniquely named collection for this test
        persist_dir: Path to the Chroma data directory
        collection_name: Unique name of the collection
    """
    client: chromadb.ClientAPI
    collection: Any
    persist_dir: str
    collection_name: str

    def add_doc(
        self,
        content: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a document to the test collection.

        Args:
            content: Document text
            doc_id: Optional ID (generates UUID if not provided)
            metadata: Optional metadata dict

        Returns:
            Document ID
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        meta = metadata or {}
        meta.setdefault("source", "test")

        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[meta]
        )
        return doc_id

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search the test collection.

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            Chroma query results dict
        """
        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

    def count(self) -> int:
        """Return document count in collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from collection."""
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)


@pytest.fixture
def hermetic_chroma(tmp_path):
    """
    Provide a fully isolated ChromaDB environment for a single test.

    Features:
    - Unique persist directory (tmp_path)
    - Unique collection name (uuid4 hex)
    - Deterministic embedding function
    - Explicit cleanup with client close

    Yields:
        HermeticChromaContext with client, collection, and helpers
    """
    # Create unique persist directory within tmp_path
    persist_dir = str(tmp_path / "chroma")
    os.makedirs(persist_dir, exist_ok=True)

    # Create unique collection name
    collection_name = f"test_{uuid.uuid4().hex}"

    # Create isolated client
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    # Create collection with dummy embeddings
    embedding_fn = DummyEmbeddingFunction()
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"description": "Test collection"}
    )

    context = HermeticChromaContext(
        client=client,
        collection=collection,
        persist_dir=persist_dir,
        collection_name=collection_name
    )

    yield context

    # Explicit cleanup
    try:
        # Delete collection first
        client.delete_collection(collection_name)
    except Exception:
        pass

    try:
        # Close client to release file locks
        # Note: PersistentClient may not have close(), but we try
        if hasattr(client, 'close'):
            client.close()
        elif hasattr(client, '_client') and hasattr(client._client, 'close'):
            client._client.close()
    except Exception:
        pass

    # Force cleanup of persist directory
    try:
        shutil.rmtree(persist_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def hermetic_chroma_with_rag(tmp_path, monkeypatch):
    """
    Provide hermetic Chroma with episodic RAG system integration.

    This fixture patches the RAG singleton to use an isolated Chroma
    client, enabling tests of the full RAG pipeline with isolation.

    Yields:
        HermeticChromaContext with episodic RAG patched to use it
    """
    import episodic.rag as rag_module
    import episodic.rag_collections as collections_module

    # Create hermetic Chroma environment
    persist_dir = str(tmp_path / "chroma")
    os.makedirs(persist_dir, exist_ok=True)
    collection_name = f"test_{uuid.uuid4().hex}"

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    embedding_fn = DummyEmbeddingFunction()
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"description": "Test RAG collection"}
    )

    context = HermeticChromaContext(
        client=client,
        collection=collection,
        persist_dir=persist_dir,
        collection_name=collection_name
    )

    # Reset RAG singletons
    rag_module._rag_system = None
    if hasattr(collections_module, '_multi_collection_rag'):
        collections_module._multi_collection_rag = None

    # Patch embedding function
    monkeypatch.setattr(
        'episodic.rag_utils.SilentSentenceTransformerEmbeddingFunction',
        DummyEmbeddingFunction
    )

    yield context

    # Reset singletons
    rag_module._rag_system = None
    if hasattr(collections_module, '_multi_collection_rag'):
        collections_module._multi_collection_rag = None

    # Cleanup
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    try:
        if hasattr(client, 'close'):
            client.close()
        elif hasattr(client, '_client') and hasattr(client._client, 'close'):
            client._client.close()
    except Exception:
        pass

    try:
        shutil.rmtree(persist_dir, ignore_errors=True)
    except Exception:
        pass


# Conftest hook to run serial tests sequentially
def pytest_configure(config):
    """Register the serial marker."""
    config.addinivalue_line(
        "markers",
        "serial: mark test to run serially (avoid parallel execution)"
    )
