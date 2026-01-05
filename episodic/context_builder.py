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
        self.topic_context = None
        
    def build_conversation_context(
        self,
        user_node_id: str,
        user_input: str,
        context_depth: int,
        model: str,
        skip_rag: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]]]:
        """
        Build conversation context with optional RAG and web search.

        Returns:
            Tuple of (messages, raw_messages, rag_context, web_context)
        """
        # Build basic conversation history
        with benchmark_resource("Database", "build context"):
            messages, raw_messages = self._build_basic_context(user_node_id, context_depth)

        # Process @file references in the last user message
        messages = self._process_file_references(messages)

        # Add topic-aware context retrieval (before RAG)
        topic_context = self._add_topic_context(user_input, messages)
        self.topic_context = topic_context

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

    def _add_topic_context(
        self,
        user_input: str,
        messages: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Add context from related previous topics.

        Uses the topic strategy to detect if the current query relates
        to a previous topic and retrieves relevant messages.
        """
        if not config.get('topic_context_retrieval', False):
            return None

        try:
            from episodic.topics.topic_retrieval import (
                retrieve_topic_context,
                format_topic_context
            )

            retrieved_messages, retrieval_info = retrieve_topic_context(
                query=user_input,
                current_messages=messages,
                max_messages=config.get('topic_context_max_messages', 10),
                max_tokens=config.get('topic_context_max_tokens', 2000)
            )

            if not retrieved_messages:
                return retrieval_info

            # Format the retrieved context
            context_text = format_topic_context(
                retrieved_messages,
                thread_name=retrieval_info.get('links', [{}])[0].get('thread_name')
            )

            if context_text:
                # Insert as a system message before the conversation
                topic_message = {
                    "role": "system",
                    "content": context_text
                }

                # Insert at the beginning (before conversation history)
                messages.insert(0, topic_message)

                if config.get("debug"):
                    debug_print(
                        f"Added topic context: {retrieval_info.get('total_messages', 0)} messages, "
                        f"{retrieval_info.get('total_tokens', 0)} tokens"
                    )

            return retrieval_info

        except Exception as e:
            if config.get("debug"):
                debug_print(f"Topic context error: {e}")
            return {'error': str(e)}

    def _add_rag_context(
        self,
        user_input: str,
        messages: List[Dict[str, Any]],
        model: str
    ) -> Optional[str]:
        """Add RAG context from both system memory and user documents."""
        # Check if either system memory or user RAG is enabled
        system_memory_enabled = config.get("system_memory_auto_context", True)
        user_rag_enabled = config.get("rag_enabled", False)
        
        if not system_memory_enabled and not user_rag_enabled:
            return None
            
        try:
            from episodic.rag import get_rag_system
            rag_system = get_rag_system()
            if rag_system is not None and config.get("rag_auto_search", True):
                # Search for relevant documents
                results = rag_system.search(user_input, k=config.get("rag_max_results", 3))
                
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
    ) -> Optional[Dict[str, Any]]:
        """Add web search context for muse mode."""
        try:
            from episodic.web_search import get_web_search_manager
            search_manager = get_web_search_manager()
            from episodic.web_extract import fetch_page_content_sync
            
            # Check if web search is enabled in config
            if config.get("web_search_enabled", False):
                # Perform web search
                typer.echo("")
                secho_color("🌐 Searching the web...", fg=get_system_color())
                
                results = search_manager.search(user_input)
                
                if results:
                    secho_color(f"Found {len(results)} results", fg=get_system_color())
                    
                    # Extract content from top results
                    extracted_content = {}
                    extract_enabled = config.get('web_search_extract_content', True)
                    
                    if extract_enabled:
                        for i, result in enumerate(results[:3], 1):  # Top 3 results
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
                                
                                # Extract content
                                content = fetch_page_content_sync(extract_url)
                                
                                if content and len(content) > 50:
                                    extracted_content[result.url] = content
                                    
                            except Exception as e:
                                if config.get('debug'):
                                    debug_print(f"Extract error for {result.url}: {e}")
                                # Silently continue to next result
                    
                    # Build web context dictionary in the format expected by synthesize_web_response
                    web_context = {
                        'results': [
                            {
                                'title': r.title,
                                'url': r.url,
                                'content': r.snippet,
                                'relevance_score': getattr(r, 'relevance_score', 0.0)
                            }
                            for r in results
                        ],
                        'extracted_content': extracted_content
                    }
                    
                    if config.get("debug"):
                        debug_print(f"Added web context: {len(results)} results, {len(extracted_content)} extracted")
                    
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
                from episodic.rag_document_manager import record_retrieval
                tracking = self._pending_rag_tracking
                for doc_id in tracking['doc_ids']:
                    record_retrieval(doc_id, tracking['query'])
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
        if self.topic_context and isinstance(self.topic_context, dict):
            info['topic_context'] = {
                'messages': self.topic_context.get('total_messages', 0),
                'tokens': self.topic_context.get('total_tokens', 0),
                'threads': len(self.topic_context.get('links', []))
            }
        return info

    def _process_file_references(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process @file references in the last user message.

        Handles:
        - @file.txt - injects text content
        - @"path/with spaces.txt" - quoted paths
        - @file.pdf - extracts PDF text
        - @file.png - sends as multimodal image
        - @file.pdf:vision - renders PDF pages as images
        - @file.pdf:vision:1-5 - specific page range
        """
        if not messages:
            return messages

        # Find the last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return messages

        original_content = messages[last_user_idx].get("content", "")

        # Skip if content is already multimodal (list) or empty
        if not isinstance(original_content, str) or not original_content:
            return messages

        # Check for @ references before importing (avoid import overhead if not needed)
        if "@" not in original_content:
            return messages

        try:
            from episodic.file_reference import process_file_references

            processed_text, multimodal_blocks, errors = process_file_references(original_content)

            # Show errors to user
            for error in errors:
                typer.secho(error, fg="red")

            # Update the message content
            if multimodal_blocks:
                # Convert to multimodal message format for LiteLLM
                messages[last_user_idx]["content"] = [
                    {"type": "text", "text": processed_text},
                    *multimodal_blocks
                ]
                if config.get("debug"):
                    debug_print(f"Added {len(multimodal_blocks)} multimodal blocks from @file references")
            elif processed_text != original_content:
                # Text was modified (file contents injected)
                messages[last_user_idx]["content"] = processed_text
                if config.get("debug"):
                    debug_print("Processed @file references (text injection)")

        except ImportError as e:
            if config.get("debug"):
                debug_print(f"File reference module not available: {e}")
        except Exception as e:
            typer.secho(f"Error processing @file references: {e}", fg="red")
            if config.get("debug"):
                import traceback
                debug_print(traceback.format_exc())

        return messages