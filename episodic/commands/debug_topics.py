"""
Debug commands for topic detection analysis.
"""

import typer
from typing import Optional
from episodic.db import get_topic_detection_scores, get_node
from episodic.configuration import get_text_color, get_system_color, get_heading_color
import json


def topic_scores(
    node_id: Optional[str] = typer.Argument(None, help="Specific node ID to get scores for"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of scores to show"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all score details")
):
    """View topic detection scores for debugging."""
    scores = get_topic_detection_scores(user_node_id=node_id, limit=limit)
    
    if not scores:
        typer.secho("No topic detection scores found.", fg=get_system_color())
        return
    
    typer.secho(f"\n📊 Topic Detection Scores ({len(scores)} records)", 
               fg=get_heading_color(), bold=True)
    typer.secho("=" * 80, fg=get_heading_color())
    
    for score in scores:
        # Get node info
        node = get_node(score['user_node_id'])
        if node:
            short_id = node.get('short_id', '??')
            content = node.get('content', '')[:60] + '...'
        else:
            short_id = '??'
            content = 'Node not found'
        
        # Basic info
        changed = "✓ CHANGED" if score['topic_changed'] else "✗ Same topic"
        color = "green" if score['topic_changed'] else "yellow"
        
        typer.secho(f"\n[{short_id}] {changed}", fg=color, bold=True)
        typer.secho(f"Message: {content}", fg=get_text_color())
        typer.secho(f"Method: {score['detection_method']}", fg=get_text_color())
        
        # Context info
        typer.secho(f"Context: {score['user_messages_in_topic']}/{score['effective_threshold']} messages in topic, {score['total_topics_count']} total topics", 
                   fg=get_text_color(), dim=True)
        
        # Show scores based on method
        if score['detection_method'] in ['hybrid', 'llm_fallback']:
            if score['final_score'] is not None:
                typer.secho(f"Final Score: {score['final_score']:.3f}", fg=get_text_color())
            
            # Show individual scores if verbose
            if verbose and score['semantic_drift_score'] is not None:
                typer.secho("\n  Component Scores:", fg=get_text_color())
                typer.secho(f"    Semantic Drift: {score['semantic_drift_score']:.3f}", fg=get_text_color())
                if score['keyword_explicit_score'] is not None:
                    typer.secho(f"    Explicit Keywords: {score['keyword_explicit_score']:.3f}", fg=get_text_color())
                if score['keyword_domain_score'] is not None:
                    typer.secho(f"    Domain Shift: {score['keyword_domain_score']:.3f}", fg=get_text_color())
                if score['message_gap_score'] is not None:
                    typer.secho(f"    Message Gap: {score['message_gap_score']:.3f}", fg=get_text_color())
                if score['conversation_flow_score'] is not None:
                    typer.secho(f"    Conversation Flow: {score['conversation_flow_score']:.3f}", fg=get_text_color())
        
        # Show transition info
        if score['transition_phrase']:
            typer.secho(f"Transition Phrase: \"{score['transition_phrase']}\"", fg=get_text_color())
        
        if score['dominant_domain']:
            domain_info = f"Domain: {score['dominant_domain']}"
            if score['previous_domain'] and score['previous_domain'] != score['dominant_domain']:
                domain_info += f" (was: {score['previous_domain']})"
            typer.secho(domain_info, fg=get_text_color())
        
        # Show detected domains if verbose
        if verbose and score['detected_domains']:
            try:
                domains = json.loads(score['detected_domains'])
                if domains:
                    typer.secho("  Detected Domains:", fg=get_text_color())
                    for domain, score_val in sorted(domains.items(), key=lambda x: x[1], reverse=True):
                        if score_val > 0:
                            typer.secho(f"    {domain}: {score_val:.2f}", fg=get_text_color())
            except:
                pass
        
        # Show LLM response if available and verbose
        if verbose and score['llm_response']:
            typer.secho(f"\n  LLM Response: {score['llm_response'][:100]}...", fg=get_text_color(), dim=True)
    
    typer.secho("\n" + "=" * 80, fg=get_heading_color())
    
    if not verbose:
        typer.secho("💡 Use --verbose to see detailed scores and domain analysis", 
                   fg=get_text_color(), dim=True)