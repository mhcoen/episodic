# Web Search Providers

Episodic supports multiple web search providers to enhance conversations with current information from the internet. This document explains how to configure and use each provider.

## Available Providers

### 1. DuckDuckGo (Default)
**Free, no API key required**

DuckDuckGo is the default provider because it:
- Requires no configuration or API keys
- Has reasonable rate limits
- Provides good search results
- Respects user privacy

```bash
# Enable DuckDuckGo (default)
> /set web_search_provider duckduckgo
> /websearch on
```

### 2. Searx/SearxNG
**Open source metasearch engine**

Searx aggregates results from multiple search engines:
- Can be self-hosted for privacy
- No API key required for public instances
- Combines results from Google, Bing, DuckDuckGo, etc.

```bash
# Configure Searx
> /set web_search_provider searx
> /set searx_instance_url https://searx.be  # Or your own instance
> /websearch on
```

Popular public instances:
- https://searx.be
- https://searx.info
- https://searx.ninja

### 3. Google Custom Search
**Most comprehensive results, requires API setup**

Google provides the best search results but requires:
1. Google Cloud account
2. Custom Search API enabled
3. Custom Search Engine created

**Setup steps:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable "Custom Search API"
4. Create credentials (API key)
5. Go to [Programmable Search Engine](https://programmablesearchengine.google.com/)
6. Create a new search engine
7. Get your Search Engine ID

```bash
# Configure Google Search
export GOOGLE_API_KEY=your_api_key_here
export GOOGLE_SEARCH_ENGINE_ID=your_engine_id_here

# Or set in Episodic
> /set google_api_key your_api_key_here
> /set google_search_engine_id your_engine_id_here
> /set web_search_provider google
> /websearch on
```

**Pricing:** Free tier includes 100 searches/day

### 4. Bing Search
**Microsoft's search API**

Bing Search requires an Azure account:
1. Go to [Azure Portal](https://portal.azure.com/)
2. Create "Bing Search v7" resource
3. Get your API key from the resource

```bash
# Configure Bing Search
export BING_API_KEY=your_api_key_here

# Or set in Episodic
> /set bing_api_key your_api_key_here
> /set web_search_provider bing
> /websearch on
```

**Pricing:** Free tier includes 1,000 searches/month

## Provider Comparison

| Provider | API Key | Cost | Privacy | Quality | Rate Limit |
|----------|---------|------|---------|---------|------------|
| DuckDuckGo | No | Free | High | Good | ~60/hour |
| Searx | No | Free | High* | Good | Varies |
| Google | Yes | Free tier | Low | Excellent | 100/day |
| Bing | Yes | Free tier | Medium | Very Good | 1000/month |

*High privacy if self-hosted

## Configuration Options

### General Settings
```bash
# Enable/disable web search
> /set web_search_enabled true

# Choose provider
> /set web_search_provider google

# Number of results
> /set web_search_max_results 5

# Auto-search when RAG has no good results
> /set web_search_auto_enhance true

# Cache results for faster repeated searches
> /set web_search_cache_duration 3600  # 1 hour

# Rate limiting
> /set web_search_rate_limit 60  # per hour

# Index results into RAG for future use
> /set web_search_index_results true
```

### Provider-Specific Settings

**Searx:**
```bash
> /set searx_instance_url https://your-searx-instance.com
```

**Google:**
```bash
> /set google_api_key YOUR_API_KEY
> /set google_search_engine_id YOUR_ENGINE_ID
```

**Bing:**
```bash
> /set bing_api_key YOUR_API_KEY
> /set bing_endpoint https://api.bing.microsoft.com/v7.0/search
```

## Usage Examples

### Basic Search
```bash
# Search with current provider
> /websearch latest Python features

# Short form
> /ws climate change solutions
```

### Check Configuration
```bash
> /websearch config

🔍 Web Search Configuration
────────────────────────────────────
Enabled: True
Provider: google
Auto-enhance: True
Max results: 5
...

Google Provider Configuration:
API Key: Configured
Search Engine ID: Configured
```

### View Statistics
```bash
> /websearch stats

📊 Web Search Statistics
────────────────────────────────────
Provider: GoogleProvider
Rate limit: 95/100 searches remaining
Cache: 5 entries
```

### Switching Providers
```bash
# Quick switch
> /set web_search_provider bing

# With verification
> /websearch config
```

## Integration with RAG

Web search integrates seamlessly with the RAG system:

1. **Manual Indexing**: Search results can be indexed for future use
   ```bash
   > /websearch latest AI breakthroughs
   > /set web_search_index_results true
   ```

2. **Auto Enhancement**: When RAG has insufficient results
   ```bash
   > /set web_search_auto_enhance true
   > What are the latest features in Python 3.13?
   # Automatically searches web if local knowledge insufficient
   ```

3. **Combined Context**: Both local and web sources in responses
   ```bash
   📚 Using sources: python_guide.txt, web:Python 3.13 Release Notes
   ```

## Troubleshooting

### DuckDuckGo Issues
- **No results**: Check internet connection
- **Rate limited**: Wait a few minutes between searches

### Searx Issues
- **Connection failed**: Instance may be down, try another
- **No results**: Instance may have issues with upstream engines

### Google Issues
- **Not configured**: Check API key and engine ID
- **Quota exceeded**: Free tier limit reached (100/day)
- **Invalid credentials**: Verify API key in Cloud Console

### Bing Issues
- **401 Unauthorized**: Check API key
- **Quota exceeded**: Free tier limit reached (1000/month)

## Privacy Considerations

1. **DuckDuckGo**: No tracking, no logs
2. **Searx**: Depends on instance, self-host for maximum privacy
3. **Google**: Tracks searches, tied to API key
4. **Bing**: Some tracking, tied to Azure account

For maximum privacy:
- Use DuckDuckGo or self-hosted Searx
- Enable caching to reduce external queries
- Consider using VPN for additional privacy

## Testing Providers

Run the test script to verify providers:
```bash
python scripts/test_search_providers.py
```

This will test each configured provider and show results.