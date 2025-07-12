"""Web search commands for Episodic."""

import typer
from typing import Optional

from episodic.config import config
from episodic.configuration import get_text_color, get_system_color, get_heading_color
from episodic.web_search import get_web_search_manager


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
        extract = config.get('web_search_extract_content', True)
    if synthesize is None:
        synthesize = config.get('web_search_synthesize', True)
    
    # Override synthesis if web_show_raw is enabled
    if config.get('web_show_raw', False):
        synthesize = False
    
    # If synthesizing, we need to extract content
    if synthesize:
        extract = True
    
    # Check if confirmation required
    if config.get('web_search_require_confirmation', False):
        if not typer.confirm(f"Search the web for: {query}?"):
            typer.secho("Search cancelled.", fg=get_text_color())
            return
    
    
    # Show search message only if not synthesizing or debug is on
    if not synthesize or config.get('debug', False):
        typer.secho(f"\n🔍 Searching web for: '{query}'", fg=get_heading_color(), bold=True)
    
    results = manager.search(query, num_results=limit)
    
    if not results:
        typer.secho("No results found.", fg=get_text_color())
        return
    
    # Show separator only if not synthesizing or debug is on
    if not synthesize or config.get('debug', False):
        typer.secho("─" * 60, fg=get_heading_color())
    
    # Track extracted content for synthesis
    extracted_content = {}
    
    # Display results only if not synthesizing or debug is on
    if not synthesize or config.get('debug', False):
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
    if extract:
        from episodic.web_extract import fetch_page_content_sync
        
        for i, result in enumerate(results[:3], 1):  # Only extract for first 3 results
            # Show extraction status only if displaying results or debug is on
            if not synthesize or config.get('debug', False):
                typer.secho(f"    📄 Extracting content...", fg=get_system_color(), nl=False)
            
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
                    # Show success only if displaying results or debug is on
                    if not synthesize or config.get('debug', False):
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
                    # Show failure only if displaying results or debug is on
                    if not synthesize or config.get('debug', False):
                        typer.secho(" ✗", fg="red")
            except Exception as e:
                # Show errors only if displaying results or debug is on
                if not synthesize or config.get('debug', False):
                    typer.secho(" ✗", fg="red")
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
                    from episodic.rag_utils_simple import suppress_chromadb_telemetry_simple
                    with suppress_chromadb_telemetry_simple():
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
        typer.secho(f"Providers: {', '.join(stats['providers'])}", fg=get_text_color())
        typer.secho(f"Rate limit: {stats['rate_limit_remaining']}/{stats['rate_limit_max']} searches/hour", 
                   fg=get_text_color())


def websearch_config():
    """Show web search configuration."""
    typer.secho("\n🔍 Web Search Configuration", fg=get_heading_color(), bold=True)
    typer.secho("─" * 40, fg=get_heading_color())
    
    settings = [
        ('Enabled', 'web_search_enabled'),
        ('Provider', 'web_search_provider'),
        ('Providers list', 'web_search_providers'),
        ('Fallback enabled', 'web_search_fallback_enabled'),
        ('Fallback cache (min)', 'web_search_fallback_cache_minutes'),
        ('Auto-enhance', 'web_search_auto_enhance'),
        ('Max results', 'web_search_max_results'),
        ('Rate limit', 'web_search_rate_limit'),
        ('Cache duration', 'web_search_cache_duration'),
        ('Index results', 'web_search_index_results'),
        ('Require confirmation', 'web_search_require_confirmation'),
        ('Show URLs', 'web_search_show_urls'),
        ('Extract content', 'web_search_extract_content'),
        ('Synthesize', 'web_search_synthesize'),
        ('Show raw results', 'web_show_raw'),
        ('Show sources', 'web_show_sources'),
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
    
    # Show hint about synthesis settings
    if config.get('web_search_synthesize', True):
        typer.secho("\n💡 ", nl=False)
        typer.secho("For synthesis settings: ", nl=False, fg=typer.colors.WHITE, dim=True)
        typer.secho("/websearch synthesis", fg="bright_cyan")


def websearch_stats():
    """Show web search statistics."""
    if not config.get('web_search_enabled', False):
        typer.secho("Web search is not enabled.", fg="yellow")
        return
    
    manager = get_web_search_manager()
    stats = manager.get_stats()
    
    typer.secho("\n📊 Web Search Statistics", fg=get_heading_color(), bold=True)
    typer.secho("─" * 40, fg=get_heading_color())
    
    typer.secho("Providers: ", nl=False, fg=get_text_color())
    typer.secho(f"{', '.join(stats['providers'])}", fg=get_system_color())
    
    if stats['current_provider']:
        typer.secho("Current provider: ", nl=False, fg=get_text_color())
        typer.secho(f"{stats['current_provider']} (cached)", fg="green")
    
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


def websearch_synthesis():
    """Show unified web synthesis settings with current values and options."""
    
    typer.secho("\n🎨 Web Search Synthesis Settings", fg=get_heading_color(), bold=True)
    typer.secho("═" * 50, fg=get_heading_color())
    
    # Check if synthesis is enabled
    synthesis_enabled = config.get('web_search_synthesize', True)
    typer.secho("\nSynthesis: ", nl=False, fg=get_text_color())
    if synthesis_enabled:
        typer.secho("ENABLED", fg="bright_green", bold=True)
    else:
        typer.secho("DISABLED", fg="bright_red", bold=True)
        typer.secho("  (Enable with: /set web-search-synthesize true)", fg=typer.colors.WHITE, dim=True)
        return
    
    # Style settings with visual examples
    typer.secho("\n📏 ", nl=False)
    typer.secho("Length Style", fg=get_system_color(), bold=True)
    current_style = config.get('muse_style', 'standard')
    
    styles = [
        ('concise', '~150 words', 'Brief summary with key points', current_style == 'concise'),
        ('standard', '~300 words', 'Balanced detail and brevity', current_style == 'standard'),
        ('comprehensive', '~500 words', 'Detailed analysis with examples', current_style == 'comprehensive'),
        ('exhaustive', '800+ words', 'Full exploration of all aspects', current_style == 'exhaustive')
    ]
    
    for style, length, desc, is_current in styles:
        if is_current:
            typer.secho(f"  ▶ ", nl=False, fg="bright_cyan")
            typer.secho(f"{style:15}", nl=False, fg="bright_cyan", bold=True)
        else:
            typer.secho(f"    {style:15}", nl=False, fg=get_text_color())
        typer.secho(f"{length:12}", nl=False, fg="bright_magenta")
        typer.secho(f"{desc}", fg=typer.colors.WHITE, dim=True)
    
    # Detail level settings
    typer.secho("\n🔍 ", nl=False)
    typer.secho("Detail Level", fg=get_system_color(), bold=True)
    current_detail = config.get('muse_detail', 'moderate')
    
    details = [
        ('minimal', 'Just essential facts', current_detail == 'minimal'),
        ('moderate', 'Facts with context', current_detail == 'moderate'),
        ('detailed', 'Facts, context, and explanations', current_detail == 'detailed'),
        ('maximum', 'Everything including nuances', current_detail == 'maximum')
    ]
    
    for detail, desc, is_current in details:
        if is_current:
            typer.secho(f"  ▶ ", nl=False, fg="bright_cyan")
            typer.secho(f"{detail:15}", nl=False, fg="bright_cyan", bold=True)
        else:
            typer.secho(f"    {detail:15}", nl=False, fg=get_text_color())
        typer.secho(f"{desc}", fg=typer.colors.WHITE, dim=True)
    
    # Format settings
    typer.secho("\n📝 ", nl=False)
    typer.secho("Output Format", fg=get_system_color(), bold=True)
    current_format = config.get('muse_format', 'mixed')
    
    formats = [
        ('paragraph', 'Flowing prose in paragraphs', current_format == 'paragraph'),
        ('bullet-points', 'Structured lists throughout', current_format == 'bullet-points'),
        ('mixed', 'Automatic based on content', current_format == 'mixed'),
        ('academic', 'Formal style with citations', current_format == 'academic')
    ]
    
    for fmt, desc, is_current in formats:
        if is_current:
            typer.secho(f"  ▶ ", nl=False, fg="bright_cyan")
            typer.secho(f"{fmt:15}", nl=False, fg="bright_cyan", bold=True)
        else:
            typer.secho(f"    {fmt:15}", nl=False, fg=get_text_color())
        typer.secho(f"{desc}", fg=typer.colors.WHITE, dim=True)
    
    # Source selection
    typer.secho("\n📚 ", nl=False)
    typer.secho("Source Usage", fg=get_system_color(), bold=True)
    current_sources = config.get('muse_sources', 'top-three')
    
    sources = [
        ('first-only', 'Use only the top result', current_sources == 'first-only'),
        ('top-three', 'Use top 3 results', current_sources == 'top-three'),
        ('all-relevant', 'Use all search results', current_sources == 'all-relevant'),
        ('selective', 'Smart selection (coming soon)', current_sources == 'selective')
    ]
    
    for src, desc, is_current in sources:
        if is_current:
            typer.secho(f"  ▶ ", nl=False, fg="bright_cyan")
            typer.secho(f"{src:15}", nl=False, fg="bright_cyan", bold=True)
        else:
            typer.secho(f"    {src:15}", nl=False, fg=get_text_color())
        typer.secho(f"{desc}", fg=typer.colors.WHITE, dim=True)
    
    # Advanced settings
    typer.secho("\n⚙️  ", nl=False)
    typer.secho("Advanced Settings", fg=get_system_color(), bold=True)
    
    # Max tokens
    max_tokens = config.get('muse_max_tokens')
    typer.secho("  Max tokens: ", nl=False, fg=get_text_color())
    if max_tokens:
        typer.secho(f"{max_tokens}", fg="bright_yellow")
    else:
        typer.secho("Auto (based on style)", fg=typer.colors.WHITE, dim=True)
    
    # Synthesis model
    synthesis_model = config.get('muse_model')
    typer.secho("  Model: ", nl=False, fg=get_text_color())
    if synthesis_model:
        typer.secho(f"{synthesis_model}", fg="bright_yellow")
    else:
        main_model = config.get('model', 'gpt-3.5-turbo')
        typer.secho(f"Main model ({main_model})", fg=typer.colors.WHITE, dim=True)
    
    # Display settings
    typer.secho("\n👁️  ", nl=False)
    typer.secho("Display Options", fg=get_system_color(), bold=True)
    
    show_sources = config.get('web_show_sources', False)
    typer.secho("  Show sources: ", nl=False, fg=get_text_color())
    typer.secho("Yes" if show_sources else "No", fg="bright_green" if show_sources else "bright_red")
    
    show_raw = config.get('web_show_raw', False)
    typer.secho("  Show raw results: ", nl=False, fg=get_text_color())
    typer.secho("Yes" if show_raw else "No", fg="bright_green" if show_raw else "bright_red")
    if show_raw:
        typer.secho("    (This overrides synthesis)", fg="yellow")
    
    # Usage examples
    typer.secho("\n💡 ", nl=False)
    typer.secho("Quick Settings", fg=get_heading_color(), bold=True)
    typer.secho("  Brief news:  ", nl=False, fg=typer.colors.WHITE, dim=True)
    typer.secho("/set muse-style concise", fg="bright_cyan")
    typer.secho("  Research:    ", nl=False, fg=typer.colors.WHITE, dim=True)
    typer.secho("/set muse-style comprehensive", fg="bright_cyan")
    typer.secho("  Academic:    ", nl=False, fg=typer.colors.WHITE, dim=True)
    typer.secho("/set muse-format academic", fg="bright_cyan")
    
    typer.secho("\n" + "─" * 50, fg=typer.colors.WHITE, dim=True)
    typer.secho("Customize prompt: ", nl=False, fg=typer.colors.WHITE, dim=True)
    typer.secho("prompts/web_synthesis.md", fg="bright_cyan")


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
    elif action == "synthesis":
        websearch_synthesis()
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
        extract = None  # Let websearch use defaults
        synthesize = None  # Let websearch use defaults
        filtered_args = []
        
        for arg in args:
            if arg.lower() in ['--extract', '-e']:
                extract = True
            elif arg.lower() in ['--synthesize', '-s', '--summarize']:
                synthesize = True
            else:
                filtered_args.append(arg)
        
        query = f"{action} {' '.join(filtered_args)}".strip()
        # Pass all parameters explicitly with keywords
        websearch(query, limit=None, index=None, extract=extract, synthesize=synthesize)