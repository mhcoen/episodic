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
        self.kg_context = None
        self.last_assembly_debug = None  # Instrumentation from last assembly
        
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

        # Add KG context if enabled (after topic, before RAG)
        kg_context = self._add_kg_context(user_input, messages)
        self.kg_context = kg_context

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

    def build_with_strategy(
        self,
        user_node_id: str,
        user_input: str,
        active_topic_start_node_id: Optional[str],
        reactivation_decision: Optional[Any] = None,
        user_embedding: Optional[Any] = None,
        token_budget: int = 4000,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Build context using the configured recovery strategy.

        This is the new strategy-based context assembly that supports:
        - ancestry: Traditional DAG traversal
        - topic_local: Topic-isolated context
        - hybrid: Switches based on reactivation

        Args:
            user_node_id: Current user node ID
            user_input: Current user message text
            active_topic_start_node_id: Active topic's start node ID
            reactivation_decision: Result of reactivation probe (for hybrid mode)
            user_embedding: Pre-computed embedding for user message
            token_budget: Maximum tokens for context

        Returns:
            Tuple of (messages, debug_info)
        """
        from episodic.context_recovery.strategy import (
            ContextRecoveryMode,
            select_strategy,
            get_mode_from_config,
        )

        # Get configured mode
        mode = get_mode_from_config()

        # Select strategy based on mode and reactivation
        strategy = select_strategy(mode, reactivation_decision)

        # Assemble context
        result = strategy.assemble(
            user_turn_text=user_input,
            user_node_id=user_node_id,
            active_topic_start_node_id=active_topic_start_node_id,
            user_embedding=user_embedding,
            token_budget=token_budget,
        )

        # Store debug info
        self.last_assembly_debug = result.debug

        # Log instrumentation
        if config.get("debug"):
            self._log_assembly_debug(result.debug)

        # Compute and persist fingerprint for determinism tracking
        if user_node_id and config.get("enable_fingerprinting", False):
            try:
                from episodic.context_recovery.determinism import (
                    compute_fingerprint,
                    persist_fingerprint
                )
                fingerprint = compute_fingerprint(user_node_id, result.debug)
                persist_fingerprint(fingerprint)
                result.debug["fingerprint_hash"] = fingerprint.hash
            except Exception as e:
                debug_print(f"Fingerprinting failed: {e}", category="context")

        return result.messages, result.debug

    def _log_assembly_debug(self, debug: Dict[str, Any]) -> None:
        """Log context assembly instrumentation."""
        mode = debug.get("mode", "unknown")
        topic_id = debug.get("topic_start_node_id", "none")
        node_count = len(debug.get("included_node_ids", []))
        tokens = debug.get("token_counts", {}).get("total_estimate", 0)
        reactivation = debug.get("reactivation_fired", False)

        debug_print(f"Context assembly: mode={mode}, topic={topic_id[:8] if topic_id else 'none'}..., "
                   f"nodes={node_count}, tokens≈{tokens}, reactivation={reactivation}",
                   category="context")

    def build_context_full(
        self,
        user_node_id: str,
        user_input: str,
        active_topic_start_node_id: Optional[str],
        model: str,
        reactivation_decision: Optional[Any] = None,
        user_embedding: Optional[Any] = None,
        token_budget: Optional[int] = None,
        skip_rag: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Build context using strategy-based assembly with RAG/web enhancements.

        This combines build_with_strategy() for core message assembly with
        the RAG and web search enhancements from build_conversation_context().

        Args:
            user_node_id: Current user node ID
            user_input: Current user message text
            active_topic_start_node_id: Active topic's start node ID (POST-reactivation)
            model: Model being used (for RAG/web compatibility)
            reactivation_decision: Result of reactivation probe (for hybrid mode)
            user_embedding: Pre-computed embedding for user message
            token_budget: Maximum tokens for context (from config if not provided)
            skip_rag: Skip RAG enhancement

        Returns:
            Tuple of (messages, raw_messages, rag_context, web_context, debug_info)
        """
        # Get token budget from config if not provided
        if token_budget is None:
            token_budget = config.get("context_token_budget", 4000)

        # Build core messages using strategy
        with benchmark_resource("Database", "build context with strategy"):
            messages, debug_info = self.build_with_strategy(
                user_node_id=user_node_id,
                user_input=user_input,
                active_topic_start_node_id=active_topic_start_node_id,
                reactivation_decision=reactivation_decision,
                user_embedding=user_embedding,
                token_budget=token_budget,
            )

        # Build raw_messages for topic evolution display if enabled
        raw_messages = []
        if config.get("show_topics"):
            conversation_chain = get_ancestry(user_node_id)
            raw_messages = [
                {"role": node.get("role"), "content": node.get("content")}
                for node in conversation_chain
                if node.get("content") and node.get("content").strip()
            ]

        # Process @file references in the last user message
        messages = self._process_file_references(messages)

        # Add topic-aware context retrieval (before RAG)
        topic_context = self._add_topic_context(user_input, messages)
        self.topic_context = topic_context

        # Add KG context if enabled (after topic, before RAG)
        kg_context = self._add_kg_context(user_input, messages)
        self.kg_context = kg_context

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

        return messages, raw_messages, rag_context, web_context, debug_info

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
        topic_retrieval_enabled = config.get('topic_context_retrieval', False)

        debug_print("_add_topic_context called", category="memory")
        debug_print(f"  topic_context_retrieval={topic_retrieval_enabled}", category="memory")

        if not topic_retrieval_enabled:
            debug_print("  -> Early exit: topic_context_retrieval disabled", category="memory")
            return None

        try:
            from episodic.topics.topic_retrieval import (
                retrieve_topic_context,
                format_topic_context
            )

            debug_print(f"  Calling retrieve_topic_context for: {user_input[:50]}...", category="memory")

            retrieved_messages, retrieval_info = retrieve_topic_context(
                query=user_input,
                current_messages=messages,
                max_messages=config.get('topic_context_max_messages', 10),
                max_tokens=config.get('topic_context_max_tokens', 2000)
            )

            debug_print(f"  Retrieved {len(retrieved_messages)} messages from topic context", category="memory")

            if not retrieved_messages:
                debug_print("  -> No topic context found", category="memory")
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

                debug_print(
                    f"  -> Injected topic context: {len(context_text)} chars, "
                    f"{retrieval_info.get('total_messages', 0)} messages",
                    category="memory"
                )

            return retrieval_info

        except Exception as e:
            debug_print(f"  -> Topic context error: {e}", category="memory")
            return {'error': str(e)}

    def _add_kg_context(
        self,
        user_input: str,
        messages: List[Dict[str, Any]],
    ) -> Optional[Any]:
        """Add knowledge graph context if enabled.

        Detects entity mentions, retrieves edges, applies closure rules,
        and inserts formatted facts as a system message. No LLM calls.
        """
        if not config.get('kg_context', False):
            return None

        try:
            from episodic.kg.context_source import get_kg_context
            from episodic.db_connection import get_connection

            with get_connection() as conn:
                result = get_kg_context(user_input, conn)

            if result is None:
                return None

            # Insert after existing system messages, before conversation
            insert_pos = 0
            for i, msg in enumerate(messages):
                if msg.get("role") != "system":
                    insert_pos = i
                    break

            kg_message = {
                "role": "system",
                "content": result.text,
            }
            messages.insert(insert_pos, kg_message)

            debug_print(
                f"KG context: {result.edge_count} edges, "
                f"{result.derived_count} derived, "
                f"{result.budget_used}/{result.budget_total} tokens, "
                f"cache={result.cache_status}",
                category='kg',
            )

            return result

        except Exception as e:
            debug_print(f"KG context error: {e}", category='kg')
            return None

    def _add_rag_context(
        self,
        user_input: str,
        messages: List[Dict[str, Any]],
        model: str
    ) -> Optional[str]:
        """Add RAG context from user documents and/or conversation memory.

        SECURITY FIX: User documents and conversation memory are now searched
        separately with explicit collection filters to prevent cross-contamination.
        """
        user_rag_enabled = config.get("rag_enabled", False)
        conversation_retrieval_enabled = config.get("conversation_retrieval_enabled", False)
        rag_auto_search = config.get("rag_auto_search", True)

        # Debug: Show all gating conditions BEFORE any early returns
        debug_print(f"_add_rag_context called:", category="memory")
        debug_print(f"  rag_enabled={user_rag_enabled}", category="memory")
        debug_print(f"  conversation_retrieval_enabled={conversation_retrieval_enabled}", category="memory")
        debug_print(f"  rag_auto_search={rag_auto_search}", category="memory")
        debug_print(f"  query: {user_input[:50]}...", category="memory")

        if not user_rag_enabled and not conversation_retrieval_enabled:
            debug_print("  -> Early exit: neither rag_enabled nor conversation_retrieval_enabled", category="memory")
            return None

        try:
            from episodic.rag import get_rag_system
            rag_system = get_rag_system()
            if rag_system is None:
                debug_print("  -> Early exit: rag_system is None", category="memory")
                return None
            if not rag_auto_search:
                debug_print("  -> Early exit: rag_auto_search is False", category="memory")
                return None

            all_results = []
            context_parts = []

            # 1. Search user documents (explicitly filtered to USER_DOCS collection)
            if user_rag_enabled:
                doc_results = rag_system.search(
                    user_input,
                    n_results=config.get("rag_max_results", 5),
                    source_filter='file'  # Forces USER_DOCS collection only
                )
                if doc_results and doc_results.get('results'):
                    for result in doc_results['results']:
                        content = result.get('content', result.get('text', ''))
                        if content:
                            all_results.append(result)
                            source = result.get('metadata', {}).get('filename', 'document')
                            context_parts.append(f"[Doc: {source}] {content}")

            # 2. Search conversation memory (explicitly filtered to CONVERSATION collection)
            if conversation_retrieval_enabled:
                debug_print(f"Conversation retrieval enabled, searching for: {user_input[:50]}...", category="memory")

                # Get IDs of messages already in context to avoid duplication
                recent_turn_ids = self._get_recent_turn_ids(messages)

                memory_results = rag_system.search(
                    user_input,
                    n_results=config.get("conversation_retrieval_k", 5),
                    source_filter='conversation'  # Forces CONVERSATION collection only
                )
                if memory_results and memory_results.get('results'):
                    debug_print(f"Found {len(memory_results['results'])} memory results", category="memory")
                    for i, result in enumerate(memory_results['results'][:3]):
                        score = result.get('relevance_score', 0)
                        preview = result.get('content', result.get('text', ''))[:50]
                        debug_print(f"  [{i+1}] score={score:.3f}: {preview}...", category="memory")

                    memory_injected = 0
                    for result in memory_results['results']:
                        # Skip if this turn is already in recent context
                        result_id = result.get('metadata', {}).get('user_id', '')
                        if result_id in recent_turn_ids:
                            debug_print(f"  Skipping {result_id[:8]} (already in context)", category="memory")
                            continue
                        content = result.get('content', result.get('text', ''))
                        if content:
                            all_results.append(result)
                            context_parts.append(f"[Memory] {content}")
                            memory_injected += 1
                    debug_print(f"Injected {memory_injected} memory chunks into context", category="memory")
                else:
                    debug_print("No memory results found", category="memory")

            if not context_parts:
                debug_print("  -> Early exit: no context_parts collected", category="memory")
                return None

            # Separate document context from memory context for different prefixes
            doc_parts = [p for p in context_parts if p.startswith("[Doc:")]
            memory_parts = [p for p in context_parts if p.startswith("[Memory]")]

            # Track which documents were used
            doc_ids = [r.get('metadata', {}).get('doc_id') for r in all_results
                       if r.get('metadata', {}).get('doc_id')]
            if doc_ids:
                self._pending_rag_tracking = {
                    'doc_ids': doc_ids,
                    'query': user_input
                }

            # Insert after any existing system messages but before conversation
            insert_pos = 0
            for i, msg in enumerate(messages):
                if msg.get("role") != "system":
                    insert_pos = i
                    break

            # Insert memory context with conversation-specific prefix
            if memory_parts:
                memory_prefix = config.get(
                    'conversation_memory_prefix',
                    "IMPORTANT: Below are excerpts from your previous conversations with this user.\n"
                    "You MUST base your answer ONLY on this information.\n"
                    "Do NOT use general knowledge. If the information isn't in these excerpts, "
                    "say \"I don't have that in our conversation history.\"\n\n"
                    "Previous conversations:"
                )
                memory_context = "\n\n".join(memory_parts)
                memory_message = {
                    "role": "system",
                    "content": f"{memory_prefix}\n\n{memory_context}"
                }
                messages.insert(insert_pos, memory_message)
                insert_pos += 1  # Adjust for next insertion

            # Insert document context with standard prefix
            if doc_parts:
                doc_prefix = config.get('rag_context_prefix', 'Relevant context from knowledge base:')
                doc_context = "\n\n".join(doc_parts)
                doc_message = {
                    "role": "system",
                    "content": f"{doc_prefix}\n\n{doc_context}"
                }
                messages.insert(insert_pos, doc_message)

            rag_context = "\n\n".join(context_parts)

            if config.get("debug"):
                debug_print(f"Added RAG context: {len(all_results)} results, {len(rag_context)} chars")
                if memory_parts:
                    debug_print(f"  Memory context: {len(memory_parts)} items with conversation-specific prefix", category="memory")
                if doc_parts:
                    debug_print(f"  Document context: {len(doc_parts)} items", category="memory")

            return rag_context

        except Exception as e:
            if config.get("debug"):
                typer.echo(f"⚠️  RAG search error: {e}")

        return None

    def _get_recent_turn_ids(self, messages: List[Dict[str, Any]]) -> set:
        """Extract turn IDs from recent messages to avoid retrieval duplication."""
        # This is a placeholder - actual implementation depends on how messages store IDs
        # For now, return empty set (no deduplication)
        return set()
    
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
                        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
                        import urllib.parse

                        def fix_search_url(url: str) -> str:
                            """Fix DuckDuckGo redirect URLs and ensure proper scheme."""
                            if url.startswith('//duckduckgo.com/l/?uddg='):
                                try:
                                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                                    if 'uddg' in parsed:
                                        url = urllib.parse.unquote(parsed['uddg'][0])
                                except:
                                    pass

                            if not url.startswith(('http://', 'https://')):
                                url = 'https://' + url.lstrip('/')

                            return url

                        # Prepare URLs for parallel fetch
                        max_pages = config.get('web_extract_max_pages', 3)
                        url_map = {}  # extract_url -> original result
                        for result in results[:max_pages]:
                            extract_url = fix_search_url(result.url)
                            url_map[extract_url] = result

                        # Fetch in parallel with global timeout
                        global_timeout = config.get('web_extract_timeout', 8)

                        with ThreadPoolExecutor(max_workers=max_pages) as executor:
                            futures = {
                                executor.submit(fetch_page_content_sync, url): url
                                for url in url_map
                            }

                            try:
                                for future in as_completed(futures, timeout=global_timeout):
                                    url = futures[future]
                                    result = url_map[url]
                                    try:
                                        content = future.result(timeout=0)  # Already complete
                                        if content and len(content) > 50:
                                            extracted_content[result.url] = content
                                    except Exception as e:
                                        if config.get('debug'):
                                            debug_print(f"Extract error for {result.url}: {e}")
                            except FuturesTimeoutError:
                                # Global timeout reached, use whatever we have
                                if config.get('debug'):
                                    debug_print(f"Web extract timeout after {global_timeout}s, got {len(extracted_content)} results")
                    
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
        if self.kg_context:
            info['kg_context'] = {
                'entities': len(self.kg_context.matched_entities),
                'edges': self.kg_context.edge_count,
                'derived': self.kg_context.derived_count,
                'budget_used': self.kg_context.budget_used,
                'budget_total': self.kg_context.budget_total,
                'cache': self.kg_context.cache_status,
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