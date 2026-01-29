"""
RAG-based Memory System for Episodic - SQLite Integration
Uses existing SQLite messages with ChromaDB for vector search.

This module is a thin wrapper that delegates ChromaDB operations to
rag_collections.py while retaining indexing logic and search formatting.
"""
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timezone

from episodic.db import get_recent_nodes
from episodic.config import config
from episodic.debug_utils import debug_print


class SQLiteMemoryRAG:
    """Memory system that uses existing SQLite data with ChromaDB search.

    Delegates all ChromaDB operations to rag_collections for unified persistence.
    Uses durable status tracking in SQLite for indexing state.
    """

    def __init__(self):
        """Initialize the memory RAG system using shared rag_collections."""
        # Lazy initialization of rag_collections to avoid import cycles
        self._rag = None
        self._collection = None

    @property
    def rag(self):
        """Lazily initialize the multi-collection RAG system."""
        if self._rag is None:
            from episodic.rag_collections import get_multi_collection_rag
            self._rag = get_multi_collection_rag()
        return self._rag

    @property
    def collection(self):
        """Get the CONVERSATION collection from rag_collections."""
        if self._collection is None:
            from episodic.rag_collections import CollectionType
            self._collection = self.rag.get_collection(CollectionType.CONVERSATION)
        return self._collection

    def _should_index(self, node_id: str) -> bool:
        """Check if a node needs indexing via durable status."""
        from episodic.db import should_index
        return should_index(node_id, 'conversation')

    def index_recent_conversations(self, limit: int = 100):
        """Index recent conversations that haven't been indexed yet."""
        nodes = get_recent_nodes(limit=limit)

        new_indexed = 0
        for node in nodes:
            if node['role'] == 'user' and self._should_index(node['id']):
                # Get the assistant response
                assistant_node = None
                for check_node in nodes:
                    if (check_node['role'] == 'assistant' and
                        check_node.get('parent_id') == node['id']):
                        assistant_node = check_node
                        break

                if assistant_node:
                    self.index_exchange(node, assistant_node)
                    new_indexed += 1

        if new_indexed > 0:
            debug_print(f"Indexed {new_indexed} new conversations", category="memory")

        return new_indexed

    def index_exchange(
        self,
        user_node: Dict,
        assistant_node: Dict,
        topic_start_node_id: Optional[str] = None
    ):
        """Index a single user-assistant exchange using rag_collections.

        Tracks indexing status in SQLite for durability.
        All operations are synchronous (ChromaDB and SQLite).

        Args:
            user_node: The user message node
            assistant_node: The assistant response node
            topic_start_node_id: Optional topic identifier for filtering anchors
        """
        from episodic.db import update_indexing_status
        from episodic.rag_collections import CollectionType

        node_id = user_node['id']

        try:
            # Create document text
            doc_text = f"User: {user_node['content']}\nAssistant: {assistant_node['content']}"

            # Create metadata
            metadata = {
                'user_id': node_id,
                'assistant_id': assistant_node['id'],
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'user_content': user_node['content'][:500],
                'assistant_content': assistant_node['content'][:500],
                'source': 'conversation'
            }

            # Add topic_start_node_id if provided (for anchor retrieval filtering)
            if topic_start_node_id:
                metadata['topic_start_node_id'] = topic_start_node_id

            # Use rag_collections add_document with explicit doc_id for deduplication
            self.rag.add_document(
                content=doc_text,
                source='conversation',
                metadata=metadata,
                collection_type=CollectionType.CONVERSATION,
                chunk=False,
                doc_id=node_id
            )

            # Track success in durable status
            update_indexing_status(node_id, 'conversation', status='ok')

        except Exception as e:
            # Track failure in durable status
            try:
                update_indexing_status(node_id, 'conversation', status='failed', error=str(e))
            except Exception as status_err:
                debug_print(f"Failed to track indexing status: {status_err}", category="memory")
            raise  # Re-raise so caller knows it failed

    def search_memories(self, query: str, limit: int = 3) -> List[Dict]:
        """Search memories using vector similarity via rag_collections."""
        debug_print(f"search_memories called with query: {query[:50]}...", category="memory")

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )

            if not results or not results['ids'] or not results['ids'][0]:
                debug_print("ChromaDB returned 0 results", category="memory")
                return []

            # Format results
            memories = []
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else 0

                memories.append({
                    'user_content': metadata.get('user_content', ''),
                    'assistant_content': metadata.get('assistant_content', ''),
                    'timestamp': metadata.get('timestamp', ''),
                    'relevance_score': max(0, 1 - distance),
                    'user_id': metadata.get('user_id', ''),
                    'assistant_id': metadata.get('assistant_id', '')
                })

            debug_print(f"ChromaDB returned {len(memories)} results", category="memory")
            return sorted(memories, key=lambda x: x['relevance_score'], reverse=True)

        except Exception as e:
            debug_print(f"Search error: {e}", category="memory")
            return []

    def format_for_context(self, memories: List[Dict]) -> Optional[str]:
        """Format memories for injection into prompt context."""
        if not memories:
            return None

        context_parts = ["[Relevant context from past conversations:]"]

        for i, memory in enumerate(memories, 1):
            try:
                ts = datetime.fromisoformat(memory['timestamp'].replace('Z', '+00:00'))
                time_str = ts.strftime("%b %d")
            except Exception:
                time_str = "Recently"

            context_parts.append(
                f"\n{i}. {time_str}:\n"
                f"   User: {memory['user_content']}\n"
                f"   Assistant: {memory['assistant_content']}"
            )

        return "\n".join(context_parts)


# Global instance (lazy initialization)
memory_rag = SQLiteMemoryRAG()


# Integration function for conversation.py
async def enhance_with_memory_context(user_input: str) -> Optional[str]:
    """Check if memory context should be added and return it.

    Note: This function uses the legacy is_query_referential detection.
    Prefer using detect_recall_intent from rag_memory_smart.py for new code.
    """
    from episodic.rag_memory_smart import detect_recall_intent

    # Use unified detector
    should_retrieve, confidence, reason = detect_recall_intent(user_input)

    if not should_retrieve or confidence < 0.5:
        return None

    # Search for relevant memories
    memories = memory_rag.search_memories(user_input, limit=2)

    if not memories:
        return None

    # Only use memories with good relevance
    relevant_memories = [m for m in memories if m['relevance_score'] > 0.7]

    if not relevant_memories:
        return None

    # Format for context
    context = memory_rag.format_for_context(relevant_memories)

    if config.get("debug") and context:
        debug_print(f"Found relevant context ({reason}, confidence: {confidence:.2f})", category="memory")

    return context


# Test function
async def test_memory_rag():
    """Test the SQLite memory RAG system"""
    print("=== Testing SQLite Memory RAG ===\n")

    # Index recent conversations
    print("Indexing recent conversations...")
    count = await memory_rag.index_recent_conversations(50)
    print(f"Indexed {count} new exchanges\n")

    # Test searches
    test_queries = [
        "virtual environment python",
        "What was that command?",
        "How do I activate it?"
    ]

    for query in test_queries:
        print(f"Query: '{query}'")

        # Search
        memories = memory_rag.search_memories(query)
        if memories:
            print(f"Found {len(memories)} memories:")
            for mem in memories:
                print(f"  - Score: {mem['relevance_score']:.3f}")
                print(f"    User: {mem['user_content'][:60]}...")
        else:
            print("No memories found")
        print()


if __name__ == "__main__":
    asyncio.run(test_memory_rag())
