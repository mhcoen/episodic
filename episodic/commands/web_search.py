"""Web search commands for Episodic."""

import typer
from typing import Optional

from episodic.config import config
from episodic.configuration import get_text_color, get_system_color, get_heading_color
from episodic.web_search import get_web_search_manager, SearchResult


def websearch(query: str, limit: Optional[int] = None, index: bool = None):
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
    
    # Display results
    for i, result in enumerate(results, 1):
        typer.secho(f"\n[{i}] ", nl=False, fg=get_system_color(), bold=True)
        typer.secho(result.title, fg=get_system_color(), bold=True)
        
        if config.get('web_search_show_urls', True):
            typer.secho(f"    {result.url}", fg="cyan")
        
        typer.secho(f"    {result.snippet}", fg=get_text_color())
    
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
        query = f"{action} {' '.join(args)}".strip()
        websearch(query)