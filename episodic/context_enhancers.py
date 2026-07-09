"""Context enhancement methods for ContextBuilder.

Mixin split out of context_builder.py to keep it under the size limit. These
methods run as part of a ContextBuilder instance (they access self.rag_context,
self.web_context, etc., which ContextBuilder.__init__ sets), so they live in a
mixin the builder inherits rather than as free functions.
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

import typer
from episodic.config import config
from episodic.configuration import get_system_color
from episodic.color_utils import secho_color
from episodic.db import get_ancestry
from episodic.debug_utils import debug_print
from episodic.debug_system import debug_enabled
from episodic.benchmark import benchmark_resource


class ContextEnhancersMixin:
    """Topic / KG / RAG / web / file-reference enhancement methods."""

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

            # INV-MUSE-9: Check whether untrusted RAG chunks should be included
            mcp_tools_active = self._has_mcp_tools()
            allow_untrusted_rag = config.get("muse_rag_in_tool_context", False)

            # 1. Search user documents (explicitly filtered to USER_DOCS collection)
            if user_rag_enabled:
                doc_results = rag_system.search(
                    user_input,
                    n_results=config.get("rag_max_results", 5),
                    source_filter='file'  # Forces USER_DOCS collection only
                )
                if doc_results and doc_results.get('results'):
                    for result in doc_results['results']:
                        metadata = result.get('metadata', {})
                        content = result.get('content', result.get('text', ''))
                        if not content:
                            continue

                        # INV-MUSE-4/9: Handle untrusted RAG chunks
                        trust_level = metadata.get('trust_level', 'trusted')
                        if trust_level == 'untrusted':
                            if mcp_tools_active and not allow_untrusted_rag:
                                debug_print(f"  Excluding untrusted RAG chunk (INV-MUSE-9)", category="memory")
                                continue
                            # Wrap untrusted chunks (INV-MUSE-4)
                            source_url = metadata.get('source_url', 'unknown')
                            content = (
                                f'<untrusted_content source="rag:web:{source_url}">\n'
                                f'{content}\n'
                                f'</untrusted_content>'
                            )
                            self.context_has_web_derived = True

                        all_results.append(result)
                        source = metadata.get('filename', 'document')
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
                        metadata = result.get('metadata', {})
                        content = result.get('content', result.get('text', ''))
                        if not content:
                            continue

                        # INV-MUSE-4/9: Handle untrusted memory chunks
                        trust_level = metadata.get('trust_level', 'trusted')
                        if trust_level == 'untrusted':
                            if mcp_tools_active and not allow_untrusted_rag:
                                debug_print(f"  Excluding untrusted memory chunk (INV-MUSE-9)", category="memory")
                                continue
                            source_url = metadata.get('source_url', 'unknown')
                            content = (
                                f'<untrusted_content source="rag:web:{source_url}">\n'
                                f'{content}\n'
                                f'</untrusted_content>'
                            )
                            self.context_has_web_derived = True

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

                results = self._run_web_search_async(search_manager, user_input)
                
                if results:
                    secho_color(f"Found {len(results)} results", fg=get_system_color())
                    debug_print(f"Web search returned {len(results)} result(s)", category="muse")
                    
                    # Extract content from top results
                    extracted_content = {}
                    extract_enabled = config.get('web_search_extract_content', True)
                    debug_print(f"Web content extraction enabled={extract_enabled}", category="muse")
                    
                    if extract_enabled:
                        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
                        import urllib.parse

                        def fix_search_url(url: str) -> str:
                            """Fix DuckDuckGo redirect URLs and ensure proper scheme."""
                            if (
                                url.startswith('//duckduckgo.com/l/?uddg=')
                                or url.startswith('https://duckduckgo.com/l/?uddg=')
                                or url.startswith('http://duckduckgo.com/l/?uddg=')
                            ):
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
                        debug_print(
                            f"Preparing extraction for {len(url_map)} page(s) (max_pages={max_pages})",
                            category="muse",
                        )

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
                                        if content:
                                            extracted_content[result.url] = content
                                            debug_print(
                                                f"Extracted {len(content)} chars from {result.url}",
                                                category="muse",
                                            )
                                        elif debug_enabled("muse"):
                                            debug_print(
                                                f"No usable content extracted from {result.url}",
                                                category="muse",
                                            )
                                    except Exception as e:
                                        debug_print(f"Extract error for {result.url}: {e}", category="muse")
                            except FuturesTimeoutError:
                                # Global timeout reached, use whatever we have
                                debug_print(
                                    f"Web extract timeout after {global_timeout}s, "
                                    f"got {len(extracted_content)} extracted page(s)",
                                    category="muse",
                                )
                    
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
                    
                    debug_print(
                        f"Added web context: results={len(results)} extracted={len(extracted_content)}",
                        category="muse",
                    )
                    
                    return web_context
                else:
                    diagnostics = {}
                    try:
                        diagnostics = search_manager.get_last_search_diagnostics()
                    except Exception:
                        diagnostics = {}

                    attempts = diagnostics.get("providers_attempted", [])
                    detail_parts = []
                    for entry in attempts:
                        provider = entry.get("provider", "unknown")
                        status = entry.get("status", "unknown")
                        reason = entry.get("reason", "")
                        if reason:
                            detail_parts.append(f"{provider}: {status} ({reason})")
                        else:
                            detail_parts.append(f"{provider}: {status}")

                    self.web_error_info = {
                        "reason": "no_results",
                        "summary": "Web search returned no results from configured providers.",
                        "details": detail_parts,
                    }
                    
        except Exception as e:
            self.web_error_info = {
                "reason": "search_exception",
                "summary": "Web search failed with an internal error.",
                "details": [str(e)],
            }
            debug_print(f"Web search error: {e}", category="muse")
        
        return None

    def _run_web_search_async(self, search_manager, query: str):
        """Run search_async safely from sync code, including active event loops."""
        try:
            asyncio.get_running_loop()
            has_running_loop = True
        except RuntimeError:
            has_running_loop = False

        if has_running_loop:
            # We are inside an active event loop (normal CLI path).
            # Run the async search in a dedicated worker thread.
            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(
                    lambda: asyncio.run(search_manager.search_async(query))
                ).result()

        return asyncio.run(search_manager.search_async(query))
    

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
