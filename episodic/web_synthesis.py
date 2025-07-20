"""
Web search result synthesis for enhanced answers.

This module synthesizes information from multiple web search results
into coherent, comprehensive answers similar to Perplexity.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

import typer
from episodic.config import config
from episodic.llm import query_llm
from episodic.web_search import SearchResult


class WebSynthesizer:
    """Synthesize web search results into coherent answers."""
    
    def __init__(self):
        self.synthesis_model = config.get('synthesis_model') or config.get('muse_model') or config.get('model', 'gpt-3.5-turbo')
        # Use global style system instead of muse-specific style
        self.style = config.get('response_style', 'standard')
        self.detail = config.get('muse_detail', 'moderate')
        self.format = config.get('response_format', 'mixed')
        self.max_tokens = config.get('muse_max_tokens')
        self.sources_config = config.get('muse_sources', 'top-three')
        
    def _get_style_instructions(self) -> Dict[str, Any]:
        """Get instructions based on synthesis style using global style system."""
        # Import style definitions from the style module
        from episodic.commands.style import STYLE_DEFINITIONS
        
        style_info = STYLE_DEFINITIONS.get(self.style)
        if not style_info:
            style_info = STYLE_DEFINITIONS['standard']
        
        # Convert global style to synthesis-specific instructions
        synthesis_map = {
            'concise': {
                'description': 'a brief, direct synthesis',
                'instructions': style_info['prompt'] + ' Focus on synthesizing web search results into concise answers.',
                'tokens': style_info['max_tokens'] or 500
            },
            'standard': {
                'description': 'a balanced, well-structured synthesis', 
                'instructions': style_info['prompt'] + ' Synthesize web search results with appropriate detail.',
                'tokens': style_info['max_tokens'] or 1000
            },
            'comprehensive': {
                'description': 'a thorough, detailed synthesis',
                'instructions': style_info['prompt'] + ' Synthesize web search results into comprehensive, detailed answers.',
                'tokens': style_info['max_tokens'] or 2000
            },
            'custom': {
                'description': 'synthesis with model-specific token limits',
                'instructions': style_info['prompt'] + ' Synthesize web search results appropriately.',
                'tokens': None  # Will use model-specific settings
            }
        }
        
        return synthesis_map.get(self.style, synthesis_map['standard'])
    
    def _get_detail_instructions(self) -> str:
        """Get instructions based on detail level."""
        # Load detail prompt from file
        from episodic.prompt_manager import get_prompt_manager
        prompt_manager = get_prompt_manager()
        
        detail_prompt = prompt_manager.get(f"detail/{self.detail}")
        if not detail_prompt:
            # Fallback if file not found
            detail_map = {
                'minimal': 'Include only essential facts without elaboration.',
                'moderate': 'Include facts with relevant context for understanding.',
                'detailed': 'Include facts, context, and clear explanations.',
                'maximum': 'Include all available information, nuances, and edge cases.'
            }
            detail_prompt = detail_map.get(self.detail, detail_map['moderate'])
        
        return detail_prompt.strip()
    
    def _get_format_instructions(self) -> str:
        """Get instructions based on format preference using global format system."""
        # Use the global format system
        from episodic.commands.style import get_format_prompt
        return get_format_prompt()
    
    def _load_prompt_template(self) -> str:
        """Load the customizable prompt template."""
        prompt_path = Path(__file__).parent.parent / 'prompts' / 'web_synthesis.md'
        if prompt_path.exists():
            return prompt_path.read_text()
        else:
            # Fallback to default prompt if template not found
            return self._get_default_prompt_template()
    
    def synthesize_results(self, query: str, results: List[SearchResult], 
                          extracted_content: Dict[str, str],
                          conversation_history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
        """
        Synthesize search results and extracted content into a comprehensive answer.
        
        Args:
            query: The original search query
            results: List of search results
            extracted_content: Dict mapping URLs to extracted content
            conversation_history: Optional conversation history for context
            
        Returns:
            Synthesized answer or None if synthesis fails
        """
        # Filter results based on sources configuration
        if self.sources_config == 'first-only':
            results = results[:1]
        elif self.sources_config == 'top-three':
            results = results[:3]
        elif self.sources_config == 'all-relevant':
            # Use all results provided
            pass
        elif self.sources_config == 'selective':
            # TODO: Implement selective filtering based on relevance
            results = results[:3]
        
        # Build context from search results and extracted content
        search_results_text = []
        extracted_content_text = []
        
        for i, result in enumerate(results, 1):
            # Build search results section
            search_results_text.append(f"[{i}] {result.title}\n    URL: {result.url}\n    Summary: {result.snippet}")
            
            # Build extracted content section
            if result.url in extracted_content:
                content = extracted_content[result.url]
                # Adjust content length based on style
                style_info = self._get_style_instructions()
                max_content_chars = style_info['tokens'] * 3  # Rough estimate
                content_preview = content[:max_content_chars] if len(content) > max_content_chars else content
                extracted_content_text.append(f"From Source [{i}]:\n{content_preview}")
            
        search_results_section = "\n\n".join(search_results_text)
        extracted_content_section = "\n\n".join(extracted_content_text) if extracted_content_text else "No detailed content extracted."
        
        # Build conversation history section
        conversation_section = ""
        if conversation_history and len(conversation_history) > 0:
            # Include conversation history for context
            if config.get("debug"):
                typer.secho(f"[DEBUG] WebSynthesizer: Including {len(conversation_history)} messages in context", fg="yellow")
            conv_parts = []
            for msg in conversation_history[-10:]:  # Last 10 messages max
                role = msg['role'].title()
                content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
                conv_parts.append(f"{role}: {content}")
            conversation_section = "Previous Conversation:\n" + "\n".join(conv_parts)
        else:
            if config.get("debug"):
                typer.secho("[DEBUG] WebSynthesizer: No conversation history provided", fg="yellow")
        
        # Get style and format instructions
        style_info = self._get_style_instructions()
        detail_instructions = self._get_detail_instructions()
        format_instructions = self._get_format_instructions()
        
        # Try to load custom prompt template
        prompt_template = self._load_prompt_template()
        
        # Build the synthesis prompt
        synthesis_prompt = prompt_template.format(
            query=query,
            search_results=search_results_section,
            extracted_content=extracted_content_section,
            conversation_history=conversation_section,
            style=self.style,
            style_instructions=style_info['instructions'],
            detail=self.detail,
            detail_instructions=detail_instructions,
            format=self.format,
            format_instructions=format_instructions,
            style_description=style_info['description'],
            additional_requirements=self._get_additional_requirements()
        )
        
        try:
            # Determine max tokens
            if self.max_tokens:
                max_tokens = self.max_tokens
            elif style_info['tokens']:
                max_tokens = style_info['tokens']
            else:
                # For 'custom' style, use model-specific parameters
                max_tokens = config.get('main_params', {}).get('max_tokens', 1000)
            
            # Use LLM to synthesize the answer
            # Check if streaming is enabled
            if config.get("stream_responses", True):
                # Return prompt info for streaming
                return {
                    'prompt': synthesis_prompt,
                    'system_message': "You are a helpful assistant that synthesizes web search results into clear, comprehensive answers.",
                    'model': self.synthesis_model,
                    'temperature': 0.3,
                    'max_tokens': max_tokens,
                    'streaming': True
                }
            else:
                response_text, cost_info = query_llm(
                    prompt=synthesis_prompt,
                    system_message="You are a helpful assistant that synthesizes web search results into clear, comprehensive answers.",
                    model=self.synthesis_model,
                    temperature=0.3,  # Lower temperature for factual accuracy
                    max_tokens=max_tokens
                )
                
                return response_text
            
        except Exception as e:
            if config.get('debug'):
                typer.secho(f"Synthesis error: {e}", fg="red")
            return None
    
    def _get_default_prompt_template(self) -> str:
        """Get the default prompt template if custom one not found."""
        return """Based on the following web search results and conversation context, provide a comprehensive answer to the user's query.

{conversation_history}

User Query: {query}

Search Results:
{search_results}

Extracted Content:
{extracted_content}

Synthesis Style: {style}
{style_instructions}

Detail Level: {detail}
{detail_instructions}

Format: {format}
{format_instructions}

Instructions:
- Synthesize information from multiple sources into {style_description}
- Be specific and include relevant details based on the detail level
- Format the answer according to the format preference
- If sources contain conflicting information, mention the discrepancy
- Use markdown formatting appropriately (headers, bold, lists)
- Take into account the conversation history to understand context and references

{additional_requirements}

Answer:"""
    
    def _get_additional_requirements(self) -> str:
        """Get additional requirements based on configuration."""
        requirements = []
        
        if self.format == 'bulleted':
            requirements.append("- Use bullet points with format: • **Label**: Information")
        elif self.format == 'academic':
            requirements.append("- Include citations in format [Source N] after claims")
        
        if self.style == 'concise':
            requirements.append("- Keep response under 150 words")
        elif self.style == 'exhaustive':
            requirements.append("- Be thorough and explore all aspects in depth")
        
        return "\n".join(requirements) if requirements else "No additional requirements."


def synthesize_web_response(query: str, search_results: Dict[str, Any], 
                           conversation_history: List[Dict[str, str]], 
                           model: str) -> str:
    """
    Synthesize a response from web search results.
    
    This is a compatibility wrapper for the refactored WebSynthesizer class.
    
    Args:
        query: The user's query
        search_results: Web search results dictionary
        conversation_history: Previous conversation messages
        model: Model to use for synthesis
        
    Returns:
        The synthesized response text
    """
    synthesizer = WebSynthesizer()
    
    # Extract search results from the dictionary
    results = []
    if 'results' in search_results:
        for r in search_results['results']:
            results.append(SearchResult(
                title=r.get('title', ''),
                url=r.get('url', ''),
                snippet=r.get('content', ''),
                relevance_score=r.get('relevance_score', 0.0)
            ))
    
    # Extract any extracted content
    extracted_content = search_results.get('extracted_content', {})
    
    # Synthesize the response
    response = synthesizer.synthesize_results(query, results, extracted_content, conversation_history)
    
    # Handle streaming case - when streaming is enabled, synthesize_results returns a dict
    if isinstance(response, dict) and response.get('streaming'):
        # Execute the actual synthesis with streaming
        from episodic.llm import _execute_llm_query
        messages = [
            {"role": "system", "content": response['system_message']},
            {"role": "user", "content": response['prompt']}
        ]
        
        stream_generator, _ = _execute_llm_query(
            messages,
            model=response['model'],
            temperature=response.get('temperature', 0.3),
            max_tokens=response.get('max_tokens', 1500),
            stream=True
        )
        
        # Collect the streamed response
        full_response = ""
        for chunk in stream_generator:
            if chunk:
                full_response += chunk
        
        return full_response
    
    return response or "I couldn't find relevant information to answer your question."


def format_synthesized_answer(answer, sources: List[SearchResult]) -> None:
    """
    Format and display a synthesized answer with sources.
    
    Args:
        answer: The synthesized answer (string or dict with streaming info)
        sources: List of source search results
    """
    from episodic.configuration import get_text_color, get_system_color, get_llm_color
    from episodic.text_formatter import format_and_display_text
    from episodic.llm import _execute_llm_query
    
    # Just add a blank line before the answer
    typer.echo()
    
    # Add sparkle emoji if in muse mode
    if config.get("muse_mode", False):
        typer.secho("✨ ", nl=False, fg=get_llm_color())
    
    # Check if we need to stream
    if isinstance(answer, dict) and answer.get('streaming'):
        # Instead of using our own streaming, let's use the conversation manager's
        # streaming to ensure consistent formatting including numbered list bolding
        pass
        
        # The conversation manager expects the response to come from an LLM query,
        # so we need to make this synthesis look like a regular LLM response
        messages = [
            {"role": "system", "content": answer['system_message']},
            {"role": "user", "content": answer['prompt']}
        ]
        
        # Let the conversation manager handle the streaming with all its formatting
        from episodic.llm import _execute_llm_query
        stream_generator, _ = _execute_llm_query(
            messages,
            model=answer['model'],
            temperature=answer.get('temperature', 0.3),
            max_tokens=answer.get('max_tokens', 1500),
            stream=True
        )
        
        # Use unified streaming for consistent formatting
        from episodic.unified_streaming import unified_stream_response
        # Don't add prefix here - the synthesis prompt already includes it
        unified_stream_response(stream_generator, answer['model'])
    else:
        # For non-streaming responses, we still need to use unified streaming
        # to get proper markdown processing and word wrapping
        from episodic.unified_streaming import unified_stream_response
        
        # Create a proper generator that works with process_stream_response
        def fake_stream():
            # Mimic the structure that process_stream_response expects
            class FakeChunk:
                def __init__(self, content):
                    self.choices = [type('obj', (object,), {'delta': {'content': content}})()]
            
            yield FakeChunk(answer)
        
        # Use the unified streamer with the fake generator
        unified_stream_response(fake_stream(), config.get("model", "gpt-3.5-turbo"))
    
    # Display sources only if configured to show them
    if config.get('web_show_sources', False):
        typer.secho("Sources:", fg=get_system_color(), bold=True)
        for i, source in enumerate(sources, 1):
            typer.secho(f"  [{i}] {source.title}", fg=get_text_color())
            typer.secho(f"      {source.url}", fg="cyan")