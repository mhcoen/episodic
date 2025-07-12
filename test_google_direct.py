#!/usr/bin/env python3
"""Direct test of Google Custom Search API"""

import os
import requests

api_key = os.environ.get('GOOGLE_API_KEY')
search_engine_id = os.environ.get('GOOGLE_SEARCH_ENGINE_ID')

print(f"API Key: {'SET' if api_key else 'NOT SET'} ({len(api_key) if api_key else 0} chars)")
print(f"Search Engine ID: {'SET' if search_engine_id else 'NOT SET'} ({search_engine_id if search_engine_id else 'None'})")

if api_key and search_engine_id:
    # Make a simple test query
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': api_key,
        'cx': search_engine_id,
        'q': 'test query',
        'num': 1
    }
    
    print("\nMaking test request to Google Custom Search API...")
    response = requests.get(url, params=params)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("✓ Success! API is working")
        print(f"Search Information: {data.get('searchInformation', {})}")
    else:
        print("✗ Error response:")
        print(response.text)