"""
Proof of Concept: RAG-based Memory System for Episodic
Goal: See memory working in 2 hours
"""
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path
from episodic.config import config
from episodic.debug_utils import debug_print


class MinimalMemoryRAG:
    """Quick and dirty memory system - just prove it works!"""
    
    def __init__(self):
        # Start simple - just use a list!
        self.memories = []
        self.message_count = 0
        
        # Store memories in a JSON file for persistence
        self.memory_file = Path.home() / ".episodic" / "poc_memories.json"
        self.memory_file.parent.mkdir(exist_ok=True)
        self._load_memories()
        
    def _load_memories(self):
        """Load existing memories from disk"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.memories = data.get('memories', [])
                    self.message_count = data.get('message_count', 0)
                debug_print(f"Loaded {len(self.memories)} existing memories", category="memory")
            except:
                debug_print("Starting with fresh memory", category="memory")
    
    def _save_memories(self):
        """Persist memories to disk"""
        with open(self.memory_file, 'w') as f:
            json.dump({
                'memories': self.memories,
                'message_count': self.message_count,
                'last_saved': datetime.now().isoformat()
            }, f, indent=2)
    
    async def on_message(self, user_msg: str, assistant_msg: str):
        """Called after each message exchange"""
        self.message_count += 1
        
        # Create a memory entry
        memory = {
            'timestamp': datetime.now().isoformat(),
            'exchange_num': self.message_count,
            'user': user_msg,
            'assistant': assistant_msg
        }
        
        # Store it
        self.memories.append(memory)
        self._save_memories()
        
        # Show progress
        debug_print(f"Stored exchange #{self.message_count}", category="memory")
        
        # Every 5 messages, test retrieval
        if self.message_count % 5 == 0:
            debug_print(f"Checkpoint! {len(self.memories)} memories stored", category="memory")
            test_results = self.simple_search("our conversation")
            if test_results:
                debug_print(f"Sample retrieval: Found {len(test_results)} relevant memories", category="memory")
    
    def simple_search(self, query: str, limit: int = 3) -> List[Dict]:
        """Super simple keyword search - just to prove it works"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score each memory by keyword overlap
        scored_memories = []
        for memory in self.memories:
            text = f"{memory['user']} {memory['assistant']}".lower()
            text_words = set(text.split())
            
            # Simple scoring: how many query words appear?
            score = len(query_words.intersection(text_words))
            if score > 0:
                scored_memories.append((score, memory))
        
        # Return top matches
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in scored_memories[:limit]]
    
    def search_and_format(self, query: str) -> Optional[str]:
        """Search memories and format for prompt injection"""
        results = self.simple_search(query)
        
        if not results:
            return None
            
        # Format as context
        context_parts = ["[Retrieved memories:]"]
        for i, memory in enumerate(results, 1):
            timestamp = memory['timestamp'].split('T')[0]  # Just date
            context_parts.append(
                f"\n{i}. From {timestamp}:\n"
                f"   User: {memory['user'][:100]}...\n"
                f"   Assistant: {memory['assistant'][:100]}..."
            )
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """Get memory system statistics"""
        return {
            'total_memories': len(self.memories),
            'total_exchanges': self.message_count,
            'memory_size_kb': len(json.dumps(self.memories)) / 1024,
            'oldest_memory': self.memories[0]['timestamp'] if self.memories else None,
            'newest_memory': self.memories[-1]['timestamp'] if self.memories else None
        }


# Global instance for easy testing
memory_system = MinimalMemoryRAG()


# Test functions for proof of concept
async def test_memory_system():
    """Quick test to see it working"""
    print("\n=== Memory System POC Test ===\n")
    
    # Simulate some conversations
    test_exchanges = [
        ("What's the best Python web framework?", "Django and FastAPI are both excellent choices."),
        ("Tell me about FastAPI", "FastAPI is a modern, fast web framework for building APIs."),
        ("How do I install it?", "You can install FastAPI with: pip install fastapi uvicorn"),
        ("What about Django?", "Django is a full-featured web framework with batteries included."),
        ("Which one should I choose?", "Choose FastAPI for APIs, Django for full web applications."),
        ("What was that install command again?", "The command was: pip install fastapi uvicorn")
    ]
    
    # Process each exchange
    for user, assistant in test_exchanges:
        print(f"User: {user}")
        print(f"Assistant: {assistant[:50]}...")
        await memory_system.on_message(user, assistant)
        print()
    
    # Test retrieval
    print("\n=== Testing Retrieval ===\n")
    
    test_queries = [
        "FastAPI install",
        "Django or FastAPI",
        "web framework"
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        context = memory_system.search_and_format(query)
        if context:
            print(context)
        else:
            print("No memories found")
        print()
    
    # Show stats
    print("\n=== Memory Stats ===")
    stats = memory_system.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_memory_system())