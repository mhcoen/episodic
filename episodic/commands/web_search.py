"""Web search commands for Episodic."""

import typer
from typing import Optional

from episodic.config import config
from episodic.configuration import get_text_color, get_system_color, get_heading_color
from episodic.web_search import get_web_search_manager, SearchResult


def websearch(query: str, limit: Optional[int] = None, index: bool = None, extract: bool = None, synthesize: bool = None):
    """Perform a web search."""
    if not config.get('web_search_enabled', False):
        typer.secho("Web search is not enabled. Use '/websearch on' to enable.", fg="yellow")
        return
    
    manager = get_web_search_manager()
    
    # Use configured defaults if not specified
    if limit is None:
        limit = config.get('web_search_max_results', 5)
    if index is None:
        index = config.get('web_search_index_results', True)
    if extract is None:
        extract = config.get('web_search_extract_content', False)
    if synthesize is None:
        synthesize = config.get('web_search_synthesize', False)
    
    # If synthesizing, we need to extract content
    if synthesize:
        extract = True
    
    # Check if confirmation required
    if config.get('web_search_require_confirmation', False):
        if not typer.confirm(f"Search the web for: {query}?"):
            typer.secho("Search cancelled.", fg=get_text_color())
            return
    
    typer.secho(f"\n🔍 Searching web for: '{query}'", fg=get_heading_color(), bold=True)
    
    results = manager.search(query, num_results=limit)
    
    if not results:
        typer.secho("No results found.", fg=get_text_color())
        return
    
    typer.secho("─" * 60, fg=get_heading_color())
    
    # Track extracted content for synthesis
    extracted_content = {}
    
    # Display results
    for i, result in enumerate(results, 1):
        typer.secho(f"\n[{i}] ", nl=False, fg=get_system_color(), bold=True)
        typer.secho(result.title, fg=get_system_color(), bold=True)
        
        if config.get('web_search_show_urls', True):
            # Clean up DuckDuckGo URLs
            url = result.url
            if url.startswith('//duckduckgo.com/l/?uddg='):
                # Extract the actual URL from DuckDuckGo redirect
                import urllib.parse
                try:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                    if 'uddg' in parsed:
                        url = urllib.parse.unquote(parsed['uddg'][0])
                except:
                    pass  # Keep original if parsing fails
            
            # Truncate very long URLs
            if len(url) > 80:
                url = url[:77] + "..."
                
            typer.secho(f"    {url}", fg="cyan")
        
        # Clean up snippet - remove excessive whitespace
        snippet = ' '.join(result.snippet.split())
        typer.secho(f"    {snippet}", fg=get_text_color())
        
        # Extract content if requested
        if extract and i <= 3:  # Only extract for first 3 results to avoid delays
            typer.secho(f"    📄 Extracting content...", fg=get_system_color(), nl=False)
            
            from episodic.web_extract import fetch_page_content_sync
            
            try:
                # Fix URL if needed
                extract_url = result.url
                if extract_url.startswith('//duckduckgo.com/l/?uddg='):
                    import urllib.parse
                    try:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(extract_url).query)
                        if 'uddg' in parsed:
                            extract_url = urllib.parse.unquote(parsed['uddg'][0])
                    except:
                        pass
                
                # Ensure URL has scheme
                if not extract_url.startswith(('http://', 'https://')):
                    extract_url = 'https://' + extract_url.lstrip('/')
                
                # Extract content using synchronous version
                content = fetch_page_content_sync(extract_url)
                
                if content and len(content) > 50:
                    typer.secho(" ✓", fg="green")
                    
                    # Store extracted content
                    clean_url = result.url
                    if clean_url.startswith('//duckduckgo.com/l/?uddg='):
                        import urllib.parse
                        try:
                            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(clean_url).query)
                            if 'uddg' in parsed:
                                clean_url = urllib.parse.unquote(parsed['uddg'][0])
                        except:
                            pass
                    extracted_content[clean_url] = content
                    
                    # Only show preview if not synthesizing
                    if not synthesize:
                        typer.secho(f"    📝 Extracted: ", fg=get_system_color(), nl=False)
                        # Limit extracted content display
                        display_content = content[:300] + "..." if len(content) > 300 else content
                        typer.secho(display_content, fg=get_text_color())
                    
                    # Update the result object for indexing
                    result.snippet = content[:1000]  # Use more content for indexing
                else:
                    typer.secho(" ✗", fg="red")
            except Exception as e:
                typer.secho(" ✗", fg="red")
                # Always show extraction errors for debugging
                typer.secho(f"    Error: {type(e).__name__}: {str(e)}", fg="red")
    
    # Synthesize results if requested
    if synthesize and extracted_content:
        from episodic.web_synthesis import WebSynthesizer, format_synthesized_answer
        
        synthesizer = WebSynthesizer()
        synthesized_answer = synthesizer.synthesize_results(query, results, extracted_content)
        
        if synthesized_answer:
            format_synthesized_answer(synthesized_answer, results[:3])  # Show top 3 sources
        else:
            typer.secho("\n⚠️  Could not synthesize results", fg="yellow")
    
    # Optionally index results into RAG
    if index and config.get('rag_enabled', False):
        typer.secho(f"\n📚 Indexing {len(results)} results into knowledge base...", 
                   fg=get_system_color())
        
        from episodic.rag import get_rag_system
        rag = get_rag_system()
        
        if rag:
            indexed_count = 0
            for result in results:
                # Create content from title and snippet
                content = f"{result.title}\n\n{result.snippet}\n\nSource: {result.url}"
                
                # Check excluded domains
                excluded = config.get('web_search_excluded_domains', [])
                if any(domain in result.url for domain in excluded):
                    continue
                
                try:
                    from episodic.rag_utils import suppress_chromadb_telemetry
                    with suppress_chromadb_telemetry():
                        doc_ids = rag.add_document(
                            content=content,
                            source=f"web:{result.url}",
                            metadata={
                                'title': result.title,
                                'url': result.url,
                                'search_query': query,
                                'search_timestamp': result.timestamp.isoformat()
                            }
                        )
                    if doc_ids:
                        indexed_count += 1
                except Exception as e:
                    if config.get('debug'):
                        typer.secho(f"Failed to index: {e}", fg="red")
            
            if indexed_count > 0:
                typer.secho(f"✅ Indexed {indexed_count} results", fg=get_system_color())


def websearch_toggle(enable: Optional[bool] = None):
    """Enable or disable web search functionality."""
    if enable is None:
        # Toggle current state
        current = config.get('web_search_enabled', False)
        enable = not current
    
    config.set('web_search_enabled', enable)
    
    status = "enabled" if enable else "disabled"
    typer.secho(f"Web search {status}", fg=get_system_color())
    
    if enable:
        manager = get_web_search_manager()
        stats = manager.get_stats()
        typer.secho(f"Provider: {stats['provider']}", fg=get_text_color())
        typer.secho(f"Rate limit: {stats['rate_limit_remaining']}/{stats['rate_limit_max']} searches/hour", 
                   fg=get_text_color())


def websearch_config():
    """Show web search configuration."""
    typer.secho("\n🔍 Web Search Configuration", fg=get_heading_color(), bold=True)
    typer.secho("─" * 40, fg=get_heading_color())
    
    settings = [
        ('Enabled', 'web_search_enabled'),
        ('Provider', 'web_search_provider'),
        ('Auto-enhance', 'web_search_auto_enhance'),
        ('Max results', 'web_search_max_results'),
        ('Rate limit', 'web_search_rate_limit'),
        ('Cache duration', 'web_search_cache_duration'),
        ('Index results', 'web_search_index_results'),
        ('Require confirmation', 'web_search_require_confirmation'),
        ('Show URLs', 'web_search_show_urls'),
    ]
    
    for label, key in settings:
        value = config.get(key)
        typer.secho(f"{label}: ", nl=False, fg=get_text_color())
        typer.secho(f"{value}", fg=get_system_color())
    
    # Show excluded domains if any
    excluded = config.get('web_search_excluded_domains', [])
    if excluded:
        typer.secho("Excluded domains: ", nl=False, fg=get_text_color())
        typer.secho(f"{', '.join(excluded)}", fg=get_system_color())
    
    # Show provider-specific configuration
    provider = config.get('web_search_provider', 'duckduckgo').lower()
    typer.secho(f"\n{provider.title()} Provider Configuration:", fg=get_heading_color())
    
    if provider == 'searx':
        typer.secho("Instance URL: ", nl=False, fg=get_text_color())
        typer.secho(config.get('searx_instance_url', 'https://searx.be'), fg=get_system_color())
    elif provider == 'google':
        api_key = config.get('google_api_key') or config.get('GOOGLE_API_KEY')
        engine_id = config.get('google_search_engine_id') or config.get('GOOGLE_SEARCH_ENGINE_ID')
        typer.secho("API Key: ", nl=False, fg=get_text_color())
        typer.secho("Configured" if api_key else "Not configured", 
                   fg="green" if api_key else "red")
        typer.secho("Search Engine ID: ", nl=False, fg=get_text_color())
        typer.secho("Configured" if engine_id else "Not configured", 
                   fg="green" if engine_id else "red")
    elif provider == 'bing':
        api_key = config.get('bing_api_key') or config.get('BING_API_KEY')
        typer.secho("API Key: ", nl=False, fg=get_text_color())
        typer.secho("Configured" if api_key else "Not configured", 
                   fg="green" if api_key else "red")
        typer.secho("Endpoint: ", nl=False, fg=get_text_color())
        typer.secho(config.get('bing_endpoint', 'Default'), fg=get_system_color())
    else:  # duckduckgo
        typer.secho("No configuration required (free, no API key)", fg=get_text_color())
    
    # Show available providers
    typer.secho("\nAvailable providers: ", nl=False, fg=get_text_color())
    typer.secho("duckduckgo, searx, google, bing", fg=get_system_color())


def websearch_stats():
    """Show web search statistics."""
    if not config.get('web_search_enabled', False):
        typer.secho("Web search is not enabled.", fg="yellow")
        return
    
    manager = get_web_search_manager()
    stats = manager.get_stats()
    
    typer.secho("\n📊 Web Search Statistics", fg=get_heading_color(), bold=True)
    typer.secho("─" * 40, fg=get_heading_color())
    
    typer.secho("Provider: ", nl=False, fg=get_text_color())
    typer.secho(f"{stats['provider']}", fg=get_system_color())
    
    typer.secho("Rate limit: ", nl=False, fg=get_text_color())
    typer.secho(f"{stats['rate_limit_remaining']}/{stats['rate_limit_max']} searches remaining", 
               fg=get_system_color())
    
    cache = stats['cache']
    typer.secho("Cache: ", nl=False, fg=get_text_color())
    typer.secho(f"{cache['entries']} entries", fg=get_system_color())
    
    if cache['queries']:
        typer.secho("\nCached queries:", fg=get_heading_color())
        for query in cache['queries'][:5]:  # Show first 5
            typer.secho(f"  • {query}", fg=get_text_color())
        if len(cache['queries']) > 5:
            typer.secho(f"  ... and {len(cache['queries']) - 5} more", fg=get_text_color())


def websearch_cache_clear():
    """Clear the web search cache."""
    manager = get_web_search_manager()
    manager.clear_cache()
    typer.secho("✅ Web search cache cleared", fg=get_system_color())


def websearch_command(action: Optional[str] = None, *args):
    """Main websearch command handler."""
    if not action:
        # Default to showing config
        websearch_config()
        return
    
    action = action.lower()
    
    if action == "on":
        websearch_toggle(True)
    elif action == "off":
        websearch_toggle(False)
    elif action == "config":
        websearch_config()
    elif action == "stats":
        websearch_stats()
    elif action == "cache":
        if args and args[0] == "clear":
            websearch_cache_clear()
        else:
            typer.secho("Usage: /websearch cache clear", fg="red")
    else:
        # Treat as search query
        # Check for flags
        extract = False
        synthesize = False
        filtered_args = []
        
        for arg in args:
            if arg.lower() in ['--extract', '-e']:
                extract = True
            elif arg.lower() in ['--synthesize', '-s', '--summarize']:
                synthesize = True
            else:
                filtered_args.append(arg)
        
        query = f"{action} {' '.join(filtered_args)}".strip()
        websearch(query, extract=extract, synthesize=synthesize)