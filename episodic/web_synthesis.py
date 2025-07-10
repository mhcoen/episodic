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
        self.synthesis_model = config.get('web_synthesis_model') or config.get('model', 'gpt-3.5-turbo')
        self.style = config.get('web_synthesis_style', 'standard')
        self.detail = config.get('web_synthesis_detail', 'moderate')
        self.format = config.get('web_synthesis_format', 'mixed')
        self.max_tokens = config.get('web_synthesis_max_tokens')
        self.sources_config = config.get('web_synthesis_sources', 'top-three')
        
    def _get_style_instructions(self) -> Dict[str, Any]:
        """Get instructions based on synthesis style."""
        style_map = {
            'concise': {
                'description': 'a brief summary (~150 words)',
                'instructions': 'Provide a concise summary focusing only on the most essential information. Limit to 2-3 key points.',
                'tokens': 200
            },
            'standard': {
                'description': 'a balanced response (~300 words)',
                'instructions': 'Provide a well-balanced answer that covers the main points with appropriate context.',
                'tokens': 400
            },
            'comprehensive': {
                'description': 'a detailed analysis (~500 words)',
                'instructions': 'Provide a comprehensive analysis including examples, context, and thorough explanations.',
                'tokens': 800
            },
            'exhaustive': {
                'description': 'an exhaustive exploration (~800+ words)',
                'instructions': 'Provide an exhaustive exploration covering all aspects, nuances, edge cases, and implications.',
                'tokens': 1500
            }
        }
        return style_map.get(self.style, style_map['standard'])
    
    def _get_detail_instructions(self) -> str:
        """Get instructions based on detail level."""
        detail_map = {
            'minimal': 'Include only essential facts without elaboration.',
            'moderate': 'Include facts with relevant context for understanding.',
            'detailed': 'Include facts, context, and clear explanations.',
            'maximum': 'Include all available information, nuances, and edge cases.'
        }
        return detail_map.get(self.detail, detail_map['moderate'])
    
    def _get_format_instructions(self) -> str:
        """Get instructions based on format preference."""
        format_map = {
            'paragraph': 'Use flowing prose in paragraph form.',
            'bullet-points': 'Use bullet points and lists for all information.',
            'mixed': 'Use a mix of paragraphs and bullet points as appropriate.',
            'academic': 'Use formal academic style with proper citations [Source N].'
        }
        return format_map.get(self.format, format_map['mixed'])
    
    def _load_prompt_template(self) -> str:
        """Load the customizable prompt template."""
        prompt_path = Path(__file__).parent.parent / 'prompts' / 'web_synthesis.md'
        if prompt_path.exists():
            return prompt_path.read_text()
        else:
            # Fallback to default prompt if template not found
            return self._get_default_prompt_template()
    
    def synthesize_results(self, query: str, results: List[SearchResult], 
                          extracted_content: Dict[str, str]) -> Optional[str]:
        """
        Synthesize search results and extracted content into a comprehensive answer.
        
        Args:
            query: The original search query
            results: List of search results
            extracted_content: Dict mapping URLs to extracted content
            
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
            else:
                max_tokens = style_info['tokens']
            
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
        return """Based on the following web search results, provide a comprehensive answer to the user's query.

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

{additional_requirements}

Answer:"""
    
    def _get_additional_requirements(self) -> str:
        """Get additional requirements based on configuration."""
        requirements = []
        
        if self.format == 'bullet-points':
            requirements.append("- Use bullet points with format: • **Label**: Information")
        elif self.format == 'academic':
            requirements.append("- Include citations in format [Source N] after claims")
        
        if self.style == 'concise':
            requirements.append("- Keep response under 150 words")
        elif self.style == 'exhaustive':
            requirements.append("- Be thorough and explore all aspects in depth")
        
        return "\n".join(requirements) if requirements else "No additional requirements."


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
        # Use the unified formatter for consistent display with LLM color
        format_and_display_text(
            answer,
            base_color=get_llm_color(),
            value_color=get_system_color()  # Use system color for values after colons
        )
    
    # Add blank line after response
    typer.echo()
    
    # Display sources only if configured to show them
    if config.get('web_show_sources', False):
        typer.secho("Sources:", fg=get_system_color(), bold=True)
        for i, source in enumerate(sources, 1):
            typer.secho(f"  [{i}] {source.title}", fg=get_text_color())
            typer.secho(f"      {source.url}", fg="cyan")