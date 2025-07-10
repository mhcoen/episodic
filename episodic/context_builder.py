"""
Context building functionality for Episodic.

This module handles building conversation context, including
RAG integration and web search enhancement.
"""

from typing import List, Dict, Any, Optional, Tuple

import typer
from episodic.config import config
from episodic.configuration import get_system_color
from episodic.color_utils import secho_color
from episodic.db import get_ancestry
from episodic.debug_utils import debug_print
from episodic.benchmark import benchmark_resource


class ContextBuilder:
    """Builds conversation context with optional enhancements."""
    
    def __init__(self):
        """Initialize the context builder."""
        self.rag_context = None
        self.web_context = None
        
    def build_conversation_context(
        self,
        user_node_id: str,
        user_input: str,
        context_depth: int,
        model: str,
        skip_rag: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str], Optional[str]]:
        """
        Build conversation context with optional RAG and web search.
        
        Returns:
            Tuple of (messages, raw_messages, rag_context, web_context)
        """
        # Build basic conversation history
        with benchmark_resource("Database", "build context"):
            messages, raw_messages = self._build_basic_context(user_node_id, context_depth)
        
        # Add RAG context if enabled
        rag_context = None
        if not skip_rag:
            rag_context = self._add_rag_context(user_input, messages, model)
            self.rag_context = rag_context
        
        # Add web search context if in muse mode
        web_context = None
        if config.get("muse_mode"):
            web_context = self._add_web_context(user_input, model)
            self.web_context = web_context
        
        return messages, raw_messages, rag_context, web_context
    
    def _build_basic_context(
        self,
        user_node_id: str,
        context_depth: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build basic conversation history context."""
        # Get the conversation history from root to current node
        conversation_chain = get_ancestry(user_node_id)
        
        # Filter conversation to get only the last N exchanges
        filtered_messages = []
        raw_messages = []  # Keep raw for topic evolution display
        
        # For topic evolution, get the raw messages
        if config.get("show_topics"):
            # Get only role and content for raw display
            raw_messages = [{"role": node.get("role"), "content": node.get("content")} 
                           for node in conversation_chain 
                           if node.get("content") and node.get("content").strip()]
        
        # conversation_chain is from oldest to newest, we want newest first for filtering
        reversed_chain = list(reversed(conversation_chain))
        
        # Count exchanges (user + assistant pairs)
        exchange_count = 0
        last_role = None
        
        for node in reversed_chain:
            # Skip empty messages
            if not node.get("content") or not node.get("content").strip():
                continue
            
            # Track when we complete an exchange
            current_role = node.get("role")
            if last_role == "assistant" and current_role == "user":
                exchange_count += 1
                
            # Stop if we've collected enough exchanges
            if exchange_count >= context_depth:
                break
                
            # Add the message to our filtered list
            filtered_messages.append({
                "role": node.get("role"),
                "content": node.get("content")
            })
            
            last_role = current_role
        
        # Reverse back to chronological order (oldest to newest)
        filtered_messages.reverse()
        
        # Always ensure we have an even number of messages (complete exchanges)
        if len(filtered_messages) % 2 != 0 and len(filtered_messages) > 1:
            # Remove the oldest message if we have an odd number
            filtered_messages.pop(0)
        
        # Ensure we start with a user message
        while filtered_messages and filtered_messages[0]["role"] != "user":
            filtered_messages.pop(0)
        
        # Build final messages list
        messages = []
        
        # Add filtered conversation history
        for msg in filtered_messages:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return messages, raw_messages
    
    def _add_rag_context(
        self,
        user_input: str,
        messages: List[Dict[str, Any]],
        model: str
    ) -> Optional[str]:
        """Add RAG context if enabled."""
        if not config.get("rag_enabled", False):
            return None
            
        try:
            from episodic.rag import rag_manager
            if rag_manager.is_available() and config.get("rag_auto_search", True):
                # Search for relevant documents
                results = rag_manager.search(user_input, k=config.get("rag_max_results", 3))
                
                if results:
                    # Build context from search results
                    context_parts = []
                    for i, result in enumerate(results, 1):
                        context_parts.append(f"[{i}] {result.get('text', '')}")
                    
                    rag_context = "\n\n".join(context_parts)
                    
                    # Track which documents were used
                    doc_ids = [r['doc_id'] for r in results if 'doc_id' in r]
                    if doc_ids:
                        # We'll track this after getting the response
                        # Store for later use
                        self._pending_rag_tracking = {
                            'doc_ids': doc_ids,
                            'query': user_input
                        }
                    
                    # Insert RAG context into messages
                    if rag_context:
                        # Add a system message with the context
                        rag_message = {
                            "role": "system",
                            "content": f"{config.get('rag_context_prefix', 'Relevant context from knowledge base:')}\n\n{rag_context}"
                        }
                        # Insert after any existing system messages but before conversation
                        insert_pos = 0
                        for i, msg in enumerate(messages):
                            if msg.get("role") != "system":
                                insert_pos = i
                                break
                        messages.insert(insert_pos, rag_message)
                        
                        if config.get("debug"):
                            debug_print(f"Added RAG context: {len(results)} results, {len(rag_context)} chars")
                        
                        return rag_context
                        
        except Exception as e:
            if config.get("debug"):
                typer.echo(f"⚠️  RAG search error: {e}")
        
        return None
    
    def _add_web_context(
        self,
        user_input: str,
        model: str
    ) -> Optional[str]:
        """Add web search context for muse mode."""
        try:
            from episodic.web_search import search_manager
            
            if search_manager.is_available() and search_manager.is_enabled():
                # Perform web search
                typer.echo("")
                secho_color("🌐 Searching the web...", fg=get_system_color())
                
                results = search_manager.search(user_input)
                
                if results:
                    secho_color(f"Found {len(results)} results", fg=get_system_color())
                    
                    # Build context from search results
                    context_parts = []
                    for result in results:
                        context_parts.append(f"Title: {result.get('title', 'No title')}")
                        context_parts.append(f"URL: {result.get('url', '')}")
                        context_parts.append(f"Content: {result.get('content', '')[:500]}...")
                        context_parts.append("")
                    
                    web_context = "\n".join(context_parts)
                    
                    if config.get("debug"):
                        debug_print(f"Added web context: {len(results)} results")
                    
                    return web_context
                else:
                    secho_color("No web results found", fg=get_system_color())
                    
        except Exception as e:
            if config.get("debug"):
                typer.echo(f"⚠️  Web search error: {e}")
        
        return None
    
    def track_rag_usage(self, assistant_node_id: str) -> None:
        """Track which RAG documents were used in the response."""
        if hasattr(self, '_pending_rag_tracking'):
            try:
                from episodic.rag import rag_manager
                tracking = self._pending_rag_tracking
                rag_manager.track_retrieval(
                    doc_ids=tracking['doc_ids'],
                    query=tracking['query'],
                    response_node_id=assistant_node_id
                )
                delattr(self, '_pending_rag_tracking')
            except Exception as e:
                if config.get("debug"):
                    typer.echo(f"⚠️  Failed to track RAG usage: {e}")
    
    def get_context_info(self) -> Dict[str, Any]:
        """Get information about the context that was built."""
        info = {}
        if self.rag_context:
            info['rag_context_length'] = len(self.rag_context)
        if self.web_context:
            info['web_context_length'] = len(self.web_context)
        return info