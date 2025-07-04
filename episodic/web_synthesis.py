"""
Web search result synthesis for enhanced answers.

This module synthesizes information from multiple web search results
into coherent, comprehensive answers similar to Perplexity.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

import typer
from episodic.config import config
from episodic.llm import query_llm
from episodic.web_search import SearchResult
from episodic.web_extract import fetch_page_content_sync


class WebSynthesizer:
    """Synthesize web search results into coherent answers."""
    
    def __init__(self):
        self.synthesis_model = config.get('web_synthesis_model', config.get('model', 'gpt-3.5-turbo'))
    
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
        # Build context from search results and extracted content
        context_parts = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"Source {i}: {result.title}")
            context_parts.append(f"URL: {result.url}")
            
            # Add extracted content if available
            if result.url in extracted_content:
                content = extracted_content[result.url]
                context_parts.append(f"Content: {content[:1500]}")  # Limit content length
            else:
                context_parts.append(f"Summary: {result.snippet}")
            
            context_parts.append("")  # Blank line
        
        context = "\n".join(context_parts)
        
        # Create synthesis prompt
        synthesis_prompt = f"""Based on the following web search results, provide a comprehensive answer to the user's query.

User Query: {query}

Search Results:
{context}

Instructions:
- Synthesize information from multiple sources into a coherent answer
- Be specific and include relevant details (times, dates, numbers, facts)
- Format the answer clearly with proper structure
- When listing facts or key information, use bullet points with this format:
  • **Label**: Value or information
  • **Another Label**: Another value
- Use markdown headers (###) to organize sections if needed
- If the sources contain conflicting information, mention the discrepancy
- Keep the answer concise but complete

Answer:"""
        
        try:
            # Use LLM to synthesize the answer
            response_text, cost_info = query_llm(
                prompt=synthesis_prompt,
                system_message="You are a helpful assistant that synthesizes web search results into clear, comprehensive answers.",
                model=self.synthesis_model,
                temperature=0.3,  # Lower temperature for factual accuracy
                max_tokens=500
            )
            
            return response_text
            
        except Exception as e:
            if config.get('debug'):
                typer.secho(f"Synthesis error: {e}", fg="red")
            return None


def format_synthesized_answer(answer: str, sources: List[SearchResult]) -> None:
    """
    Format and display a synthesized answer with sources.
    
    Args:
        answer: The synthesized answer
        sources: List of source search results
    """
    from episodic.configuration import get_heading_color, get_text_color, get_system_color
    from episodic.text_formatter import format_and_display_text
    
    # Display the answer header
    typer.secho("\n📊 Synthesized Answer", fg=get_heading_color(), bold=True)
    typer.secho("─" * 60, fg=get_heading_color())
    
    # Display the formatted answer with proper formatting
    typer.echo()  # Blank line
    
    # Use the unified formatter for consistent display
    format_and_display_text(
        answer,
        base_color=get_text_color(),
        value_color="bright_cyan"  # Use bright cyan for values after colons
    )
    
    # Display sources
    typer.echo()  # Blank line
    typer.secho("Sources:", fg=get_system_color(), bold=True)
    for i, source in enumerate(sources, 1):
        typer.secho(f"  [{i}] {source.title}", fg=get_text_color())
        typer.secho(f"      {source.url}", fg="cyan")