#!/usr/bin/env python3
"""Test Google search using async (like episodic does)"""

import asyncio
import aiohttp
from episodic.config import config

async def test_google_search():
    api_key = config.get('google_api_key') or config.get('GOOGLE_API_KEY')
    search_engine_id = config.get('google_search_engine_id') or config.get('GOOGLE_SEARCH_ENGINE_ID')
    
    print(f"API Key: {'SET' if api_key else 'NOT SET'}")
    print(f"Search Engine ID: {search_engine_id}")
    
    params = {
        'key': api_key,
        'cx': search_engine_id,
        'q': 'test query',
        'num': 1
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            'https://www.googleapis.com/customsearch/v1',
            params=params
        ) as response:
            print(f"\nStatus: {response.status}")
            text = await response.text()
            
            if response.status == 200:
                print("✓ Success!")
            else:
                print("✗ Error:")
                print(text)
                
                # Check specific error conditions
                if "API_KEY_SERVICE_BLOCKED" in text:
                    print("\n⚠️  API_KEY_SERVICE_BLOCKED detected")

# Run the async test
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(test_google_search())
loop.close()