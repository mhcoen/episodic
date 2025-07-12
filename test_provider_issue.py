#!/usr/bin/env python3
"""Debug web search provider issue"""

from episodic.config import config
from episodic.web_search import WebSearchManager, GoogleProvider

# Set Google as provider
config.set("web_search_providers", ["google", "duckduckgo"])

print("1. Config check:")
print(f"   web_search_providers: {config.get('web_search_providers')}")
print(f"   google_api_key: {'SET' if config.get('google_api_key') else 'NOT SET'}")
print(f"   google_search_engine_id: {'SET' if config.get('google_search_engine_id') else 'NOT SET'}")

print("\n2. GoogleProvider check:")
provider = GoogleProvider()
print(f"   API Key: {'SET' if provider.api_key else 'NOT SET'}")
print(f"   Search Engine ID: {'SET' if provider.search_engine_id else 'NOT SET'}")
print(f"   Is Available: {provider.is_available()}")

print("\n3. WebSearchManager providers:")
manager = WebSearchManager()
print(f"   Number of providers: {len(manager.providers)}")
for i, p in enumerate(manager.providers):
    name = p.__class__.__name__.replace('Provider', '')
    print(f"   [{i}] {name}: available={p.is_available()}")

print("\n4. Simulating search...")
# Enable debug to see skip messages
config.set('debug', True)
results = manager.search("test", num_results=1)