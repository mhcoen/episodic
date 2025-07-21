#!/usr/bin/env python3
"""
Demo script showing the Episodic memory system in action.

This script demonstrates key features of the memory system including:
- Automatic conversation memory
- Document indexing
- Context-aware responses
- Memory management
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from episodic.config import config
from episodic.conversation import conversation_manager
from episodic.rag import get_rag_system
from episodic.commands.memory import memory_command, forget_command, memory_stats_command
from episodic.llm import query_llm

# Enable demo mode - no actual LLM calls
DEMO_MODE = True

def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")

def simulate_user_input(text):
    """Simulate user input."""
    print(f"> {text}")
    time.sleep(0.5)

def simulate_assistant_response(text, with_memory=False):
    """Simulate assistant response."""
    if with_memory:
        print("💭 Using memory context from previous conversations")
    print(f"\n{text}\n")
    time.sleep(0.5)

def demo_memory_system():
    """Run the memory system demo."""
    print_section("EPISODIC MEMORY SYSTEM DEMO")
    
    # Enable RAG
    config.set('rag_enabled', True)
    config.set('rag_auto_enhance', True)
    
    # Initialize
    rag = get_rag_system()
    if not rag:
        print("❌ Failed to initialize RAG system")
        return
    
    # Demo 1: Basic Document Indexing
    print_section("Demo 1: Document Indexing and Retrieval")
    
    simulate_user_input("/index README.md")
    
    # Simulate indexing
    if not DEMO_MODE:
        doc_id, chunks = rag.add_document(
            content="""# My Python Project
            
This project implements a web scraper using BeautifulSoup and async patterns.

## Key Features
- Async HTTP requests with aiohttp
- HTML parsing with BeautifulSoup4  
- Rate limiting and retry logic
- Export to JSON and CSV formats

## Configuration
Set these environment variables:
- SCRAPER_RATE_LIMIT: Requests per second (default: 10)
- SCRAPER_TIMEOUT: Request timeout in seconds (default: 30)
- SCRAPER_MAX_RETRIES: Maximum retry attempts (default: 3)
""",
            source="file",
            metadata={'filename': 'README.md', 'type': 'documentation'}
        )
        print(f"✅ Document indexed with ID: {doc_id[:8]} ({chunks} chunks)")
    else:
        print("✅ Document indexed with ID: a1b2c3d4 (3 chunks)")
    
    time.sleep(1)
    
    # Demo 2: Context-Aware Response
    print_section("Demo 2: Automatic Context Enhancement")
    
    simulate_user_input("How do I configure the rate limiting in this project?")
    
    simulate_assistant_response("""Based on your project's README, you can configure rate limiting through environment variables:

- Set `SCRAPER_RATE_LIMIT` to control requests per second (default: 10)
- This allows you to adjust the scraping speed based on the target website's limits

For example:
```bash
export SCRAPER_RATE_LIMIT=5  # 5 requests per second
```

The scraper will automatically respect this limit when making requests.""", with_memory=True)
    
    # Demo 3: Conversation Memory
    print_section("Demo 3: Conversation Memory")
    
    simulate_user_input("I need to add proxy support to handle IP blocking")
    
    simulate_assistant_response("""To add proxy support to your web scraper, you can modify the aiohttp session:

```python
import aiohttp
from aiohttp_proxy import ProxyConnector

async def create_session_with_proxy(proxy_url):
    connector = ProxyConnector.from_url(proxy_url)
    return aiohttp.ClientSession(connector=connector)

# Usage
proxy = 'http://proxy.example.com:8080'
session = await create_session_with_proxy(proxy)
```

You might also want to add proxy rotation for better reliability:
- Maintain a pool of proxy servers
- Rotate on each request or on failure
- Add proxy validation before use""")
    
    # Save to memory
    if not DEMO_MODE:
        conversation_manager.add_exchange(
            "I need to add proxy support to handle IP blocking",
            "To add proxy support to your web scraper..."
        )
    
    time.sleep(1)
    
    # Demo 4: Memory Search
    print_section("Demo 4: Searching Memories")
    
    simulate_user_input("/memory search proxy")
    print("""
🔍 Searching memories for: proxy

Found 2 matches:
1. 💬 [b2c3d4e5] (relevance: 0.94)
   Discussion about adding proxy support to web scraper with aiohttp_proxy...
   
2. 📄 [a1b2c3d4] (relevance: 0.31)  
   Project README mentioning web scraper configuration and features...
""")
    
    # Demo 5: Contextual Continuation
    print_section("Demo 5: Contextual Continuation")
    
    simulate_user_input("What about handling proxy authentication?")
    
    simulate_assistant_response("""Building on the proxy setup we discussed, you can add authentication like this:

```python
# For basic auth proxies
proxy = 'http://username:password@proxy.example.com:8080'

# For proxies requiring headers
async def create_authenticated_session(proxy_url, username, password):
    import aiohttp
    from aiohttp_proxy import ProxyConnector
    import base64
    
    connector = ProxyConnector.from_url(proxy_url)
    
    # Create session with proxy auth headers
    auth = base64.b64encode(f'{username}:{password}'.encode()).decode()
    headers = {'Proxy-Authorization': f'Basic {auth}'}
    
    return aiohttp.ClientSession(
        connector=connector,
        headers=headers
    )
```

This integrates with your existing scraper's rate limiting and retry logic.""", with_memory=True)
    
    # Demo 6: Memory Management
    print_section("Demo 6: Memory Management")
    
    simulate_user_input("/memory-stats")
    print("""
📊 Memory System Statistics
──────────────────────────────
General:
  Total documents: 2
  Total chunks: 5
  Avg chunks/doc: 2.5
  Total retrievals: 3

Documents by Source:
  💬 conversation: 1
  📄 file: 1

Storage:
  Database size: 0.1 MB
  Embedding model: all-MiniLM-L6-v2

Configuration:
  RAG enabled: True
  Auto-enhance: True
  Chunk size: 1000
  Search results: 5
""")
    
    # Demo 7: Complex Context
    print_section("Demo 7: Multi-Source Context")
    
    simulate_user_input("Show me a complete example that uses async, rate limiting, and proxies")
    
    simulate_assistant_response("""Here's a complete example combining all the features from your project:

```python
import asyncio
import aiohttp
from aiohttp_proxy import ProxyConnector
from bs4 import BeautifulSoup
import os
from asyncio import Semaphore

class WebScraper:
    def __init__(self):
        # Load configuration from environment
        self.rate_limit = int(os.getenv('SCRAPER_RATE_LIMIT', '10'))
        self.timeout = int(os.getenv('SCRAPER_TIMEOUT', '30'))
        self.max_retries = int(os.getenv('SCRAPER_MAX_RETRIES', '3'))
        
        # Rate limiting semaphore
        self.semaphore = Semaphore(self.rate_limit)
        
    async def create_session(self, proxy_url=None):
        if proxy_url:
            connector = ProxyConnector.from_url(proxy_url)
            return aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
    
    async def fetch_with_retry(self, session, url):
        async with self.semaphore:  # Rate limiting
            for attempt in range(self.max_retries):
                try:
                    async with session.get(url) as response:
                        return await response.text()
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

# Usage example
scraper = WebScraper()
proxy = 'http://username:password@proxy.example.com:8080'
session = await scraper.create_session(proxy)
html = await scraper.fetch_with_retry(session, 'https://example.com')
soup = BeautifulSoup(html, 'html.parser')
```

This integrates all your project's features:
- ✅ Async operations with aiohttp
- ✅ Rate limiting via semaphore (SCRAPER_RATE_LIMIT)
- ✅ Configurable timeouts (SCRAPER_TIMEOUT)
- ✅ Retry logic with exponential backoff
- ✅ Proxy support with authentication
- ✅ BeautifulSoup parsing""", with_memory=True)
    
    print_section("Demo Complete!")
    print("""
The memory system has:
- Indexed documentation for context
- Stored conversation history
- Retrieved relevant context automatically
- Enhanced responses with past knowledge

Try these commands:
- /memory list          - See all stored memories
- /memory search <term> - Search for specific content  
- /forget <id>          - Remove specific memories
- /set rag-enabled false - Disable memory system
""")

if __name__ == "__main__":
    demo_memory_system()