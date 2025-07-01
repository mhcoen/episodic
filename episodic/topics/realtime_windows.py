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
        self.threshold = 0.75  # Default threshold
    
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
            # Extract user messages
            user_messages = [msg for msg in recent_messages if msg.get("role") == "user"]
            
            # Need at least window_size previous messages
            if len(user_messages) < self.window_size:
                if config.get("debug"):
                    typer.echo(f"   Not enough history for {self.window_size}-window detection")
                return False, None, None
            
            # Window A: Last 3 user messages
            window_a = user_messages[-self.window_size:]
            
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
            
            if config.get("debug"):
                typer.echo(f"\n🔍 DEBUG: Sliding {self.window_size}-window detection")
                typer.echo(f"   Window A messages: {len(window_a)}")
                typer.echo(f"   Drift score: {drift_score:.3f}")
                typer.echo(f"   Threshold: {threshold}")
                typer.echo(f"   Decision: {'TOPIC CHANGED' if topic_changed else 'SAME TOPIC'}")
                
                # Show window content preview
                if config.get("debug_verbose"):
                    typer.echo(f"   Window A preview: {window_a_text[:100]}...")
                    typer.echo(f"   New message preview: {new_message[:100]}...")
            
            return topic_changed, None, None
            
        except Exception as e:
            logger.error(f"Realtime window detection error: {e}")
            if config.get("debug"):
                typer.echo(f"   ❌ Window detection error: {e}")
            return False, None, None