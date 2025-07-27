"""
Enhanced search for RAG system with conceptual understanding.

This module provides search enhancements including:
- WordNet-based query expansion
- Conceptual similarity boosting
- Hybrid search strategies
"""

from typing import Dict, List, Any, Optional, Callable
from functools import wraps

from episodic.rag_wordnet import expand_search_query, concept_expander
from episodic.config import config
from episodic.debug_utils import debug_print


def enhanced_search(
    rag_search_func: Callable,
    query: str,
    n_results: int = None,
    source_filter: Optional[str] = None,
    enable_expansion: Optional[bool] = None,
    expansion_mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhanced search with conceptual query expansion.
    
    Args:
        rag_search_func: The original RAG search function
        query: Search query
        n_results: Number of results to return
        source_filter: Filter by source type
        enable_expansion: Whether to enable query expansion (None = use config)
        expansion_mode: How aggressive to expand ("narrow", "balanced", "broad")
        
    Returns:
        Search results with enhanced relevance
    """
    # Check if expansion is enabled
    if enable_expansion is None:
        enable_expansion = config.get("search_query_expansion", True)
    
    if not enable_expansion:
        # Just use original search
        return rag_search_func(query, n_results, source_filter)
    
    # Get expansion mode from config if not specified
    if expansion_mode is None:
        expansion_mode = config.get("wordnet_expansion_mode", "balanced")
    
    # Expand the query
    expanded_query = expand_search_query(query, mode=expansion_mode)
    
    if expanded_query == query:
        # No expansion possible, use original search
        debug_print(f"No expansion found for '{query}'", category="search")
        return rag_search_func(query, n_results, source_filter)
    
    debug_print(f"Expanded search: '{query}' → '{expanded_query}'", category="search")
    
    # Perform enhanced search
    results = rag_search_func(expanded_query, n_results, source_filter)
    
    # Post-process results to boost conceptually related matches
    if results.get('results'):
        results['results'] = boost_conceptual_relevance(
            results['results'], 
            query,
            expanded_query
        )
    
    # Add expansion info to results
    results['query_expanded'] = True
    results['expanded_query'] = expanded_query
    results['expansion_mode'] = expansion_mode
    
    return results


def boost_conceptual_relevance(
    results: List[Dict[str, Any]], 
    original_query: str,
    expanded_query: str
) -> List[Dict[str, Any]]:
    """
    Boost relevance scores for results that are conceptually related to the query.
    
    This helps ensure that "physics" results rank well for "science" queries.
    """
    boosted_results = []
    query_terms = original_query.lower().split()
    
    for result in results:
        content = result.get('content', '').lower()
        original_score = result.get('relevance_score', 0)
        
        # Check for conceptual matches
        boost_factor = 1.0
        concept_match_found = False
        
        for query_term in query_terms:
            # Get hypernyms (broader terms) for words in content
            content_words = content.split()[:50]  # Check first 50 words for efficiency
            
            for content_word in content_words:
                # Clean word
                content_word = content_word.strip('.,!?;:()[]{}').lower()
                
                # Check if content word is conceptually related to query
                similarity = concept_expander.get_concept_similarity(query_term, content_word)
                
                if similarity > 0.6:
                    concept_match_found = True
                    boost_factor = max(boost_factor, 1.0 + (similarity * 0.3))
                    debug_print(
                        f"Conceptual match: '{query_term}' ~ '{content_word}' "
                        f"(similarity: {similarity:.2f})",
                        category="search"
                    )
        
        # Apply boost
        if concept_match_found:
            boosted_score = min(original_score * boost_factor, 1.0)
            result['relevance_score'] = boosted_score
            result['original_score'] = original_score
            result['conceptual_boost'] = boost_factor
            debug_print(
                f"Boosted score: {original_score:.3f} → {boosted_score:.3f}",
                category="search"
            )
        
        boosted_results.append(result)
    
    # Re-sort by boosted scores
    boosted_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    return boosted_results


def search_with_fallback(
    rag_search_func: Callable,
    query: str,
    n_results: int = None,
    source_filter: Optional[str] = None,
    relevance_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Search with automatic fallback to expanded search if no good results.
    
    Tries:
    1. Exact search
    2. If no results above threshold, try expanded search
    3. If still no good results, try broad expansion
    """
    if relevance_threshold is None:
        relevance_threshold = config.get('memory_relevance_threshold', 0.3)
    
    # Try exact search first
    results = rag_search_func(query, n_results, source_filter)
    
    # Check if we have good results
    good_results = [
        r for r in results.get('results', [])
        if r.get('relevance_score', 0) >= relevance_threshold
    ]
    
    if good_results:
        debug_print(f"Found {len(good_results)} good results with exact search", category="search")
        return results
    
    # Try balanced expansion
    debug_print("No good results with exact search, trying balanced expansion", category="search")
    expanded_results = enhanced_search(
        rag_search_func, query, n_results, source_filter,
        enable_expansion=True, expansion_mode="balanced"
    )
    
    good_expanded = [
        r for r in expanded_results.get('results', [])
        if r.get('relevance_score', 0) >= relevance_threshold
    ]
    
    if good_expanded:
        debug_print(f"Found {len(good_expanded)} good results with balanced expansion", category="search")
        return expanded_results
    
    # Try broad expansion as last resort
    debug_print("Still no good results, trying broad expansion", category="search")
    broad_results = enhanced_search(
        rag_search_func, query, n_results, source_filter,
        enable_expansion=True, expansion_mode="broad"
    )
    
    return broad_results


def make_search_conceptual(search_method: Callable) -> Callable:
    """
    Decorator to add conceptual search capabilities to a search method.
    
    Usage:
        @make_search_conceptual
        def search(self, query, n_results=5):
            # original search implementation
    """
    @wraps(search_method)
    def wrapper(self, query: str, n_results: int = None, 
                source_filter: Optional[str] = None, **kwargs):
        # Check if conceptual search is enabled
        if config.get("enable_conceptual_search", True):
            # Create a bound method for the original search
            original_search = lambda q, n, s: search_method(self, q, n, s, **kwargs)
            
            # Use enhanced search with fallback
            return search_with_fallback(
                original_search,
                query,
                n_results,
                source_filter
            )
        else:
            # Use original search
            return search_method(self, query, n_results, source_filter, **kwargs)
    
    return wrapper


# Configuration additions for config_template.json
CONCEPTUAL_SEARCH_CONFIG = {
    "_comment_conceptual": "// === CONCEPTUAL SEARCH SETTINGS ===",
    "_comment_enable_conceptual_search": "Enable WordNet-based conceptual search",
    "enable_conceptual_search": True,
    "_comment_search_query_expansion": "Automatically expand queries with related concepts",  
    "search_query_expansion": True,
    "_comment_expansion_max_depth": "Maximum hierarchy depth for concept expansion",
    "expansion_max_depth": 2,
    "_comment_expansion_max_terms": "Maximum number of expansion terms",
    "expansion_max_terms": 10,
    "_comment_conceptual_boost_factor": "How much to boost conceptually related results (0.0-1.0)",
    "conceptual_boost_factor": 0.3
}