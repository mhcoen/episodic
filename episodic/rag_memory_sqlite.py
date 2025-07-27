"""
RAG-based Memory System for Episodic - SQLite Integration
Uses existing SQLite messages with ChromaDB for vector search
"""
import os
# Disable ChromaDB telemetry to avoid errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import hashlib

from episodic.db import get_recent_nodes
from episodic.config import config


class SQLiteMemoryRAG:
    """Memory system that uses existing SQLite data with ChromaDB search"""
    
    def __init__(self):
        # Create separate collection for conversation memories
        persist_dir = Path.home() / ".episodic" / "memory_chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        
        # Use sentence transformers for embeddings (free, local)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create memory collection
        try:
            self.collection = self.client.get_collection(
                name="episodic_conversation_memory",
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.client.create_collection(
                name="episodic_conversation_memory",
                embedding_function=self.embedding_function
            )
        
        self.indexed_ids = set()
        self._load_indexed_ids()
    
    def _load_indexed_ids(self):
        """Load which node IDs have been indexed"""
        try:
            # Get all indexed IDs from ChromaDB
            results = self.collection.get()
            if results and 'ids' in results:
                self.indexed_ids = set(results['ids'])
        except:
            self.indexed_ids = set()
    
    def _should_index(self, node_id: str) -> bool:
        """Check if a node should be indexed"""
        return node_id not in self.indexed_ids
    
    async def index_recent_conversations(self, limit: int = 100):
        """Index recent conversations that haven't been indexed yet"""
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
                    await self.index_exchange(node, assistant_node)
                    new_indexed += 1
        
        if new_indexed > 0:
            print(f"[Memory] Indexed {new_indexed} new conversations")
        
        return new_indexed
    
    async def index_exchange(self, user_node: Dict, assistant_node: Dict):
        """Index a single user-assistant exchange"""
        # Create document text
        doc_text = f"User: {user_node['content']}\nAssistant: {assistant_node['content']}"
        
        # Create metadata - use current time since nodes don't have timestamps
        metadata = {
            'user_id': user_node['id'],
            'assistant_id': assistant_node['id'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_content': user_node['content'][:500],  # First 500 chars
            'assistant_content': assistant_node['content'][:500]
        }
        
        # Add to ChromaDB
        self.collection.add(
            ids=[user_node['id']],
            documents=[doc_text],
            metadatas=[metadata]
        )
        
        self.indexed_ids.add(user_node['id'])
    
    def search_memories(self, query: str, limit: int = 3) -> List[Dict]:
        """Search memories using vector similarity"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            if not results or not results['ids'] or not results['ids'][0]:
                return []
            
            # Format results
            memories = []
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else 0
                
                memories.append({
                    'user_content': metadata['user_content'],
                    'assistant_content': metadata['assistant_content'],
                    'timestamp': metadata['timestamp'],
                    'relevance_score': max(0, 1 - distance),  # Convert distance to similarity (0-1)
                    'user_id': metadata['user_id'],
                    'assistant_id': metadata['assistant_id']
                })
            
            return sorted(memories, key=lambda x: x['relevance_score'], reverse=True)
        
        except Exception as e:
            if config.get("debug"):
                print(f"[Memory] Search error: {e}")
            return []
    
    def format_for_context(self, memories: List[Dict]) -> Optional[str]:
        """Format memories for injection into prompt context"""
        if not memories:
            return None
        
        context_parts = ["[Relevant context from past conversations:]"]
        
        for i, memory in enumerate(memories, 1):
            # Parse timestamp
            try:
                ts = datetime.fromisoformat(memory['timestamp'].replace('Z', '+00:00'))
                time_str = ts.strftime("%b %d")
            except:
                time_str = "Recently"
            
            context_parts.append(
                f"\n{i}. {time_str}:\n"
                f"   User: {memory['user_content']}\n"
                f"   Assistant: {memory['assistant_content']}"
            )
        
        return "\n".join(context_parts)
    
    def is_query_referential(self, query: str) -> Tuple[bool, float]:
        """Detect if a query references past conversations"""
        # Referential markers
        referential_phrases = [
            "we discussed", "we talked about", "you mentioned", "you said",
            "remember when", "last time", "previously", "earlier",
            "what was that", "you told me", "we covered", "go back to",
            "that command", "that example", "that explanation"
        ]
        
        query_lower = query.lower()
        
        # Check for explicit referential language
        for phrase in referential_phrases:
            if phrase in query_lower:
                return True, 0.9  # High confidence
        
        # Check for implicit references (questions without context)
        implicit_markers = ["what was", "which one", "that one", "those"]
        for marker in implicit_markers:
            if query_lower.startswith(marker):
                return True, 0.7  # Medium confidence
        
        # Check if it's a follow-up question (very short, no context)
        if len(query.split()) <= 5 and "?" in query:
            # Could be referential
            return True, 0.5  # Low confidence
        
        return False, 0.0


# Global instance
memory_rag = SQLiteMemoryRAG()


# Integration function for conversation.py
async def enhance_with_memory_context(user_input: str) -> Optional[str]:
    """Check if memory context should be added and return it"""
    # Check if query seems referential
    is_ref, confidence = memory_rag.is_query_referential(user_input)
    
    if not is_ref or confidence < 0.5:
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
        print(f"[Memory] Found relevant context (confidence: {confidence:.2f})")
    
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
        
        # Check if referential
        is_ref, conf = memory_rag.is_query_referential(query)
        print(f"Referential: {is_ref} (confidence: {conf:.2f})")
        
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