# Web Search Provider Fallback Implementation

## Summary

Implemented automatic fallback logic for web search providers, allowing the system to try multiple providers in order when one fails (e.g., due to quota limits, API errors, or missing credentials).

## Changes Made

### 1. Configuration Updates (`episodic/config_defaults.py`)
- Added `web_search_providers`: List of providers in order of preference
- Added `web_search_fallback_enabled`: Enable/disable automatic fallback
- Added `web_search_fallback_cache_minutes`: Cache working provider for N minutes

### 2. Parameter Mappings (`episodic/param_mappings.py`)
- Added short aliases using dot notation:
  - `web.provider` → `web_search_provider`
  - `web.providers` → `web_search_providers`
  - `web.enabled` → `web_search_enabled`
  - `web.fallback` → `web_search_fallback_enabled`
  - `web.cache` → `web_search_cache_duration`
  - etc.

### 3. Settings Handlers (`episodic/commands/settings_handlers.py`)
- Added `handle_list_param()` for comma-separated list values
- Added handlers for new parameters

### 4. Web Search Manager (`episodic/web_search.py`)
- Modified to support multiple providers instead of single provider
- Implemented fallback logic in `search()` method:
  - Try providers in configured order
  - Skip providers without credentials
  - Cache working provider for faster subsequent searches
  - Show fallback attempts to user
- Updated `get_stats()` to show provider list and current cached provider

### 5. Command Updates (`episodic/commands/web_search.py`)
- Updated `websearch_config()` to show new settings
- Updated `websearch_stats()` to show provider list
- Updated `websearch_toggle()` to show configured providers

## Usage Examples

### Set providers in order of preference:
```bash
# Try Google first, then Bing, then DuckDuckGo
/set web.providers google,bing,duckduckgo

# Only use free providers
/set web.providers duckduckgo,searx

# Use single provider (no fallback)
/set web.providers google
```

### Configure fallback behavior:
```bash
# Enable/disable fallback
/set web.fallback true

# Cache working provider for 10 minutes
/set web.fallback_cache_minutes 10
```

### View current configuration:
```bash
/websearch config
/websearch stats
```

## How It Works

1. **Provider Selection**: 
   - Uses cached working provider if available and cache is fresh
   - Otherwise tries providers in configured order

2. **Error Handling**:
   - Missing credentials: Skip to next provider
   - API errors (quota, auth): Try next provider
   - Empty results: Try next provider
   - All fail: Show error message

3. **User Feedback**:
   - Shows which provider is being tried (when fallback occurs)
   - Shows success message when fallback provider works
   - Debug mode shows detailed attempts

4. **Performance**:
   - Caches working provider to avoid unnecessary fallback attempts
   - Configurable cache duration (default: 5 minutes)

## Benefits

1. **Reliability**: Automatically handles quota limits and API failures
2. **Flexibility**: Users can configure provider preference order
3. **Cost Control**: Can prioritize free providers over paid ones
4. **Transparency**: Shows which provider was used
5. **Performance**: Caches working provider to reduce latency

## Testing

Run the test script to see fallback in action:
```bash
python scripts/test_web_fallback.py
```

This demonstrates:
- Multiple provider configuration
- Automatic fallback when providers fail
- Single provider mode (no fallback)