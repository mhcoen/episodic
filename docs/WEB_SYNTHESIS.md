# Web Search Synthesis

This document explains how the web search synthesis feature works in Episodic.

## Overview

Web search synthesis provides Perplexity-like answers by searching the web, extracting content from results, and using an LLM to synthesize a comprehensive response.

## The Flow

1. **Search Query** → `/ws what's the weather?`
2. **Web Search** → Fetches 5 results from DuckDuckGo (or configured provider)
3. **Content Extraction** → Extracts full text from top 3 result pages
4. **LLM Synthesis** → Sends extracted content to LLM with synthesis prompt
5. **Formatted Output** → Displays the synthesized answer

## Key Components

### 1. Search Phase (`websearch()` in `web_search.py`)
- Searches using configured provider (DuckDuckGo by default)
- Gets title, URL, and snippet for each result
- Results are cached for 1 hour to avoid repeated searches

### 2. Content Extraction (`fetch_page_content_sync()` in `web_extract.py`)
- Fetches actual webpage content
- Strips HTML, JavaScript, CSS
- Extracts readable text
- Limits to ~1500 characters per source for synthesis

### 3. Synthesis (`synthesize_results()` in `web_synthesis.py`)
- Builds a prompt with:
  - Original query
  - Search result titles and URLs
  - Extracted content from each page
- Sends to LLM with instructions to:
  - Provide comprehensive answer
  - Use markdown formatting
  - Combine information from multiple sources
  - Handle conflicting information
  - Be specific and factual

### 4. Display (`format_synthesized_answer()`)
- Streams the response with word wrapping
- Renders markdown headers in bold
- Uses LLM color for consistency
- Optionally shows sources (if `web-show-sources=true`)

## Configuration

### Default Settings
- **Automatic synthesis**: Enabled by default (`web_search_synthesize=true`)
- **Content extraction**: Enabled by default (`web_search_extract_content=true`)
- **Show raw results**: Disabled by default (`web_show_raw=false`)
- **Synthesis model**: Uses your main conversation model by default

### User Controls
- `/ws <query>` - Automatically synthesizes answer
- `/ws <query> --summarize` - Explicitly request synthesis
- `/set web-show-raw true` - Show raw search results instead
- `/set web-show-sources true` - Show source URLs with synthesis
- `/set web-synthesis-model <model>` - Use different model for synthesis

## Example Usage

```
> /ws what's the weather like in Peru?

Current Weather in Peru

Based on the latest information, here's the current weather situation in Peru:

**Lima, Peru**
- Current Time: 4:45 PM PET (Peru Time, UTC-5)
- Temperature: 64°F (18°C)
- Conditions: Foggy
- Today's High/Low: 68°F / 59°F

**General Weather Patterns**
Peru experiences varied weather conditions due to its diverse geography...
```

## Technical Details

### Synthesis Prompt Structure
The synthesis prompt includes:
1. User's original query
2. Search result metadata (title, URL)
3. Extracted content from each source
4. Instructions for comprehensive, well-formatted answer

### Streaming Support
- Synthesis supports streaming output
- Word wrapping preserves readability
- Markdown formatting (headers, bold, lists) rendered properly
- Same streaming infrastructure as regular LLM responses

### Error Handling
- Gracefully handles extraction failures
- Falls back to snippets if content extraction fails
- Shows warning if synthesis fails
- Continues to work even if some sources fail

## Benefits

1. **Comprehensive Answers**: Combines multiple sources for complete information
2. **Current Information**: Gets latest data from the web
3. **Clean Output**: No clutter, just the synthesized answer
4. **Markdown Formatting**: Well-structured, readable responses
5. **Source Attribution**: Optional source display for verification
6. **Cached Results**: Avoids repeated searches for same queries

The result is a clean, Perplexity-like answer that combines information from multiple web sources into a coherent response.