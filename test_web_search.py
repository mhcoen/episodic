#!/usr/bin/env python3
"""Test web search functionality"""

from episodic.web_search import WebSearchManager
from episodic.config import config

# Enable debug to see what's happening
config.set('debug', True)

# Create search manager
manager = WebSearchManager()

# Print current providers
print("Providers list:", config.get('web_search_providers'))
print("\nTesting search...")

# Try searching
results = manager.search("test query", num_results=3)

print(f"\nGot {len(results)} results")