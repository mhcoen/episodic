"""
Simple drift-based topic detection.

This module provides a straightforward semantic drift detector that only
uses embedding similarity to detect topic changes.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from episodic.ml.drift import ConversationalDrift
from episodic.config import config
import typer

logger = logging.getLogger(__name__)


class SimpleDriftDetector:
    """
    Simple topic detection based purely on semantic drift between messages.
    
    This is the most straightforward approach - if the semantic similarity
    between consecutive user messages drops below a threshold, it's a topic change.
    """
    
    def __init__(self):
        """Initialize the simple drift detector."""
        self.drift_calculator = ConversationalDrift()
        self.threshold = 0.85  # High threshold to reduce false positives
    
    def detect_topic_change(
        self,
        recent_messages: List[Dict[str, Any]],
        new_message: str,
        current_topic: Optional[Tuple[str, str]] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Detect if topic has changed using pure semantic drift.
        
        Returns:
            Tuple of (changed: bool, new_topic: str|None, cost_info: dict|None)
        """
        try:
            # Get the most recent user message to compare with
            user_messages = [msg for msg in recent_messages if msg.get("role") == "user"]
            
            if len(user_messages) < 1:
                # No previous user message to compare
                return False, None, None
            
            # Calculate drift between last user message and new message
            prev_message = user_messages[-1].get("content", "")
            
            # Create node-like dictionaries for drift calculation
            prev_node = {"content": prev_message}
            new_node = {"content": new_message}
            
            drift_score = self.drift_calculator.calculate_drift(
                prev_node, new_node, text_field="content"
            )
            
            # Get threshold from config or use default
            threshold = float(config.get("drift_threshold", self.threshold))
            
            # Simple decision: high drift = topic change
            topic_changed = drift_score >= threshold
            
            if config.get("debug"):
                typer.echo(f"\n🔍 DEBUG: Simple drift detection")
                typer.echo(f"   Drift score: {drift_score:.3f}")
                typer.echo(f"   Threshold: {threshold}")
                typer.echo(f"   Decision: {'TOPIC CHANGED' if topic_changed else 'SAME TOPIC'}")
            
            return topic_changed, None, None
            
        except Exception as e:
            logger.error(f"Simple drift detection error: {e}")
            # On error, assume no topic change
            return False, None, None