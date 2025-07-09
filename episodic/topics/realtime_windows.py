"""
Real-time sliding window topic detection.

This module provides a sliding window detector optimized for real-time
topic detection during conversations.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from episodic.ml.drift import ConversationalDrift
from episodic.config import config
import typer

logger = logging.getLogger(__name__)


class RealtimeWindowDetector:
    """
    Real-time topic detection using sliding 3-3 windows.
    
    Compares the last 3 user messages (window A) with the current message
    plus next 2 messages (window B) to detect topic boundaries.
    """
    
    def __init__(self, window_size: int = 3):
        """Initialize with window size (default 3 for 3-3 windows)."""
        self.window_size = window_size
        self.drift_calculator = ConversationalDrift()
        self.threshold = 0.9  # Default threshold
    
    def detect_topic_change(
        self,
        recent_messages: List[Dict[str, Any]],
        new_message: str,
        current_topic: Optional[Tuple[str, str]] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Detect topic change using sliding windows in real-time.
        
        For real-time detection, we compare:
        - Window A: Last 3 user messages (not including new)
        - Window B: Current new message (we can't look ahead in real-time)
        
        Returns:
            Tuple of (changed: bool, new_topic: str|None, cost_info: dict|None)
        """
        try:
            # Extract user messages (recent_messages is newest-first)
            user_messages = [msg for msg in recent_messages if msg.get("role") == "user"]
            
            if config.get("debug"):
                typer.echo(f"   Found {len(user_messages)} user messages in history")
                if len(user_messages) > 0:
                    typer.echo(f"   User messages (newest first): {[m.get('short_id', '?') for m in user_messages[:5]]}")
            
            # Need at least window_size previous messages
            if len(user_messages) < self.window_size:
                if config.get("debug"):
                    typer.echo(f"   Not enough history for {self.window_size}-window detection")
                return False, None, None
            
            # Take the most recent window_size messages (these are the ones just before the current message)
            # Since recent_messages is newest-first, we take the first window_size messages
            window_a = user_messages[:self.window_size]
            # Reverse to get chronological order (oldest to newest)
            window_a.reverse()
            
            # Combine window A messages
            window_a_text = " ".join(msg.get("content", "") for msg in window_a)
            
            # Create nodes for drift calculation
            window_a_node = {"content": window_a_text}
            new_msg_node = {"content": new_message}
            
            # Calculate drift between window A and new message
            drift_score = self.drift_calculator.calculate_drift(
                window_a_node, new_msg_node, text_field="content"
            )
            
            # Get threshold from config
            threshold = float(config.get("drift_threshold", self.threshold))
            
            # Decision based on drift
            topic_changed = drift_score >= threshold
            
            # Prepare detection info to be stored later (after node creation)
            detection_info = {
                "method": "sliding_window",
                "window_size": self.window_size,
                "drift_score": drift_score,
                "threshold_used": threshold,
                "is_boundary": topic_changed,
                "window_a_messages": window_a,  # Store for later reference
                "detection_type": "realtime"
            }
            
            # Get keyword score for completeness
            try:
                from episodic.topics.keywords import TransitionDetector
                keyword_detector = TransitionDetector()
                keyword_results = keyword_detector.detect_transition_keywords(new_message)
                keyword_score = max(
                    keyword_results.get("explicit_transition", 0.0),
                    keyword_results.get("domain_shift", 0.0) * 0.8
                )
                detection_info["keyword_score"] = keyword_score
                detection_info["combined_score"] = max(drift_score, keyword_score)
                detection_info["transition_phrase"] = keyword_results.get('found_phrase')
            except Exception as e:
                if config.get("debug"):
                    typer.echo(f"   ⚠️ Failed to get keyword score: {e}")
            
            if config.get("debug"):
                typer.echo(f"\n🔍 DEBUG: Sliding {self.window_size}-window detection")
                typer.echo(f"   Window A messages: {len(window_a)}")
                typer.echo(f"   Window A short_ids: {[m.get('short_id', '?') for m in window_a]}")
                typer.echo(f"   Drift score: {drift_score:.3f}")
                typer.echo(f"   Threshold: {threshold}")
                typer.echo(f"   Decision: {'TOPIC CHANGED' if topic_changed else 'SAME TOPIC'}")
                
                # Show window content preview
                if config.get("debug_verbose"):
                    typer.echo(f"   Window A preview: {window_a_text[:100]}...")
                    typer.echo(f"   New message preview: {new_message[:100]}...")
            
            return topic_changed, None, detection_info
            
        except Exception as e:
            logger.error(f"Realtime window detection error: {e}")
            if config.get("debug"):
                typer.echo(f"   ❌ Window detection error: {e}")
            return False, None, None