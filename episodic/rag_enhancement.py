"""RAG web-enhancement methods for EpisodicRAG.

Mixin split out of rag.py; EpisodicRAG inherits it, so these run on the instance
(self.search and other methods resolve via inheritance).
"""

import logging
from typing import List, Dict, Any, Optional

import typer

from episodic.config import config
from episodic.configuration import get_text_color, get_system_color
from episodic.rag_document_manager import record_retrieval

logger = logging.getLogger(__name__)


class _RAGEnhancementMixin:
    """enhance_with_context and its web-search decision/preview helpers."""

    def enhance_with_context(self, message: str, n_results: int = None,
                           include_web: Optional[bool] = None) -> str:
        """
        Enhance a message with relevant context from the knowledge base.
        
        Args:
            message: The user's message
            n_results: Number of results to include (default from config)
            include_web: Whether to include web search results if local results insufficient
            
        Returns:
            Enhanced message with context prepended
        """
        if n_results is None:
            n_results = config.get("rag_context_results", 3)
            
        # Search for relevant context
        results = self.search(message, n_results=n_results)
        
        # Check if we should do web search
        if self._should_search_web(results, include_web):
            from episodic.web_search import WebSearchManager
            searcher = WebSearchManager()
            
            # Perform web search
            web_results = searcher.search(message, num_results=3)
            
            if web_results:
                typer.echo("\n🌐 Augmenting with web search results...", 
                          fg=get_system_color())
                
                # Add web results to context
                for i, result in enumerate(web_results[:2]):
                    # Create a synthetic document entry
                    results['results'].append({
                        'content': result.snippet,
                        'metadata': {
                            'source': 'web',
                            'title': result.title,
                            'url': result.url
                        },
                        'relevance_score': 0.8 - (i * 0.1)  # Slightly lower than local
                    })
        
        if not results['results']:
            # No relevant context found
            return message
        
        # Build context section
        context_parts = ["### Relevant Context ###"]
        
        # Track which documents were used
        used_doc_ids = []
        chunk_texts = []
        
        for i, result in enumerate(results['results']):
            metadata = result.get('metadata', {})
            content = result['content']
            
            # Add source attribution
            source = metadata.get('source', 'unknown')
            if source == 'file':
                source_info = f"From file: {metadata.get('filename', 'unknown')}"
            elif source == 'web':
                source_info = f"From web: {metadata.get('title', 'Web Page')}"
                if metadata.get('url'):
                    source_info += f" ({metadata['url']})"
            else:
                source_info = f"From {source}"
            
            context_parts.append(f"\n[Context {i+1} - {source_info}]")
            context_parts.append(content)
            
            # Track usage
            if doc_id := metadata.get('doc_id'):
                used_doc_ids.append(doc_id)
            chunk_texts.append(content)
        
        context_parts.append("\n### User Query ###")
        context_parts.append(message)
        
        # Record retrieval for analytics (skip for help RAG)
        if used_doc_ids and not getattr(self, '_is_help_rag', False):
            record_retrieval(message, used_doc_ids, chunk_texts)
        
        return "\n".join(context_parts)
    
    def _should_search_web(self, local_results: Dict, include_web: Optional[bool]) -> bool:
        """Determine if web search should be performed."""
        # Explicit control
        if include_web is not None:
            return include_web
            
        # Check if web search is enabled
        if not config.get("web_search_enabled", False):
            return False
            
        # Auto-detect: search web if local results are insufficient
        if not local_results['results']:
            return True
            
        # Check relevance scores
        avg_score = sum(r.get('relevance_score', 0) for r in local_results['results']) / len(local_results['results'])
        
        # If average relevance is low, search web
        return avg_score < config.get("rag_web_search_threshold", 0.7)
    
    def _generate_preview(self, content: str, max_length: int = 200) -> str:
        """Generate a preview of the content.
        
        Args:
            content: The full content
            max_length: Maximum length of preview (default: 200)
            
        Returns:
            Preview text, cleaned and truncated
        """
        # Remove excessive whitespace
        preview = ' '.join(content.split())
        
        # Truncate to max length
        if len(preview) > max_length:
            # Try to cut at a word boundary
            preview = preview[:max_length]
            last_space = preview.rfind(' ')
            if last_space > max_length * 0.8:  # Only cut at word if we keep 80% of content
                preview = preview[:last_space]
            preview += '...'
        
        return preview


