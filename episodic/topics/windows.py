"""
Sliding window based topic detection.

This module implements the sliding window approach for detecting topic
boundaries based on semantic drift between consecutive windows of messages.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from episodic.ml.drift import ConversationalDrift
from episodic.config import config
from episodic.db import get_ancestry, get_recent_nodes
from episodic.db_wrappers import (
    store_topic_detection_score, get_topic_detection_scores
)
import typer

logger = logging.getLogger(__name__)


class SlidingWindowDetector:
    """
    Detects topic boundaries using sliding windows over conversation history.
    
    This approach compares windows of messages to detect when the semantic
    content shifts significantly, indicating a topic change.
    """
    
    def __init__(self, window_size: int = 3):
        """
        Initialize the sliding window detector.
        
        Args:
            window_size: Number of messages in each window
        """
        self.window_size = window_size
        self.drift_calculator = ConversationalDrift()
        self.threshold = config.get("manual_index_threshold", 0.75)
    
    def analyze_conversation(
        self, 
        conversation_nodes: List[Dict[str, Any]], 
        store_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze entire conversation using sliding windows.
        
        Args:
            conversation_nodes: All conversation nodes in chronological order
            store_scores: Whether to store scores in database
            
        Returns:
            List of detected boundaries with scores and metadata
        """
        # Filter for user messages only
        user_messages = [node for node in conversation_nodes if node.get('role') == 'user']
        
        if len(user_messages) < 2:
            if config.get("debug"):
                typer.echo("Not enough user messages for analysis")
            return []
        
        boundaries = []
        
        # Sliding window analysis
        for i in range(len(user_messages) - 1):
            # Window A: Previous messages (including current)
            window_a_start = max(0, i - self.window_size + 1)
            window_a_end = i + 1
            
            # Window B: Following messages
            window_b_start = i + 1
            window_b_end = min(len(user_messages), i + 1 + self.window_size)
            
            # Skip if Window B would be empty
            if window_b_start >= len(user_messages):
                break
            
            window_a_messages = user_messages[window_a_start:window_a_end]
            window_b_messages = user_messages[window_b_start:window_b_end]
            
            # Calculate drift between windows
            try:
                drift_score, keyword_score = self._calculate_window_drift(
                    window_a_messages, 
                    window_b_messages
                )
                
                # Use OR logic if configured
                if config.get("use_or_logic", True):
                    combined_score = max(drift_score, keyword_score)
                    is_boundary = drift_score >= self.threshold or keyword_score >= 0.5
                else:
                    # Average the scores
                    combined_score = (drift_score + keyword_score) / 2
                    is_boundary = combined_score >= self.threshold
                
                # Store score if requested
                if store_scores:
                    self._store_window_score(
                        user_messages[window_b_start],
                        window_a_messages,
                        window_b_messages,
                        drift_score,
                        keyword_score,
                        combined_score,
                        is_boundary
                    )
                
                # Add to boundaries if detected
                if is_boundary:
                    boundary_info = {
                        "position": i + 1,
                        "node_id": user_messages[window_b_start]['id'],
                        "short_id": user_messages[window_b_start]['short_id'],
                        "drift_score": drift_score,
                        "keyword_score": keyword_score,
                        "combined_score": combined_score,
                        "window_a_size": len(window_a_messages),
                        "window_b_size": len(window_b_messages)
                    }
                    boundaries.append(boundary_info)
                    
                    if config.get("debug"):
                        typer.echo(f"\n🎯 Topic boundary detected at position {i+1}:")
                        typer.echo(f"   Node: {boundary_info['short_id']}")
                        typer.echo(f"   Drift score: {drift_score:.3f}")
                        typer.echo(f"   Keyword score: {keyword_score:.3f}")
                
            except Exception as e:
                logger.warning(f"Error calculating drift at position {i}: {e}")
                continue
        
        return boundaries
    
    def _calculate_window_drift(
        self, 
        window_a: List[Dict[str, Any]], 
        window_b: List[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """
        Calculate drift between two windows of messages.
        
        Returns:
            Tuple of (drift_score, keyword_score)
        """
        # Combine messages in each window
        text_a = " ".join(msg.get('content', '') for msg in window_a)
        text_b = " ".join(msg.get('content', '') for msg in window_b)
        
        # Calculate semantic drift
        try:
            drift_score = self.drift_calculator.calculate_drift([text_a], text_b)
        except Exception as e:
            logger.warning(f"Drift calculation failed: {e}")
            drift_score = 0.0
        
        # Calculate keyword-based score
        keyword_score = self._calculate_keyword_score(text_b)
        
        return drift_score, keyword_score
    
    def _calculate_keyword_score(self, text: str) -> float:
        """
        Calculate keyword-based topic change score.
        
        Args:
            text: Combined text from a window
            
        Returns:
            Score between 0 and 1
        """
        # Import here to avoid circular imports
        from .keywords import TransitionDetector
        
        detector = TransitionDetector()
        results = detector.detect_transition_keywords(text)
        
        # Combine explicit and domain scores
        return max(
            results.get("explicit_transition", 0.0),
            results.get("domain_shift", 0.0) * 0.8  # Slightly lower weight for domain shifts
        )
    
    def _store_window_score(
        self,
        boundary_node: Dict[str, Any],
        window_a: List[Dict[str, Any]],
        window_b: List[Dict[str, Any]],
        drift_score: float,
        keyword_score: float,
        combined_score: float,
        is_boundary: bool
    ) -> None:
        """Store window analysis results in database."""
        try:
            # Get transition phrase if detected
            from .keywords import TransitionDetector
            detector = TransitionDetector()
            keyword_results = detector.detect_transition_keywords(
                " ".join(msg.get('content', '') for msg in window_b)
            )
            
            store_topic_detection_score(
                user_node_short_id=boundary_node['short_id'],
                window_size=self.window_size,
                window_a_start_short_id=window_a[0]['short_id'] if window_a else None,
                window_a_end_short_id=window_a[-1]['short_id'] if window_a else None,
                window_a_size=len(window_a),
                window_b_start_short_id=window_b[0]['short_id'] if window_b else None,
                window_b_end_short_id=window_b[-1]['short_id'] if window_b else None,
                window_b_size=len(window_b),
                drift_score=drift_score,
                keyword_score=keyword_score,
                combined_score=combined_score,
                is_boundary=is_boundary,
                threshold_used=self.threshold,
                transition_phrase=keyword_results.get('found_phrase'),
                detection_method='sliding_window'
            )
        except Exception as e:
            logger.warning(f"Failed to store window score: {e}")
    
    def get_stored_boundaries(self, window_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve previously detected boundaries from database.
        
        Args:
            window_size: Filter by window size (None for all)
            
        Returns:
            List of boundary detections
        """
        scores = get_topic_detection_scores(window_size=window_size, detection_method='sliding_window')
        return [score for score in scores if score['is_boundary']]