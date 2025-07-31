"""
Dual-window topic detection system.

This module implements a two-tier detection system using both (4,1) and (4,2)
window configurations for optimal topic boundary detection.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from episodic.ml.drift import ConversationalDrift
from episodic.config import config
import typer

logger = logging.getLogger(__name__)


@dataclass
class WindowConfig:
    """Configuration for a detection window."""
    name: str
    before_window: int
    after_window: int
    threshold: float
    description: str


class DualWindowDetector:
    """
    Dual-window topic detection using both (4,1) and (4,2) configurations.
    
    This detector uses two window configurations:
    - (4,1) window: High precision for immediate detection
    - (4,2) window: Balanced detection as a safety net
    """
    
    def __init__(self):
        """Initialize the dual-window detector."""
        from episodic.config import config
        
        # Get embedding settings from config
        embedding_provider = config.get("drift_embedding_provider", "sentence-transformers")
        embedding_model = config.get("drift_embedding_model", "paraphrase-mpnet-base-v2")
        
        self.drift_calculator = ConversationalDrift(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )
        
        # Define window configurations
        self.high_precision_window = WindowConfig(
            name="High Precision (4,1)",
            before_window=-4,
            after_window=1,
            threshold=config.get("dual_window_high_precision_threshold", 0.2),  # Lower = more boundaries
            description="Detects boundaries with high precision"
        )
        
        self.safety_net_window = WindowConfig(
            name="Safety Net (4,2)",
            before_window=-4,
            after_window=2,
            threshold=config.get("dual_window_safety_net_threshold", 0.25),
            description="Catches boundaries missed by high precision"
        )
        
    def detect_topic_change(
        self,
        recent_messages: List[Dict[str, Any]],
        new_message: str,
        current_topic: Optional[Tuple[str, str]] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Detect topic change using dual-window approach.
        
        Returns:
            Tuple of (changed: bool, new_topic: str|None, detection_info: dict)
        """
        try:
            # Extract all messages (recent_messages is newest-first)
            all_messages = [msg for msg in recent_messages]
            
            if config.get("debug"):
                typer.echo(f"   Dual-window: Found {len(all_messages)} messages in history")
            
            # Need at least 5 messages for (4,1) window
            if len(all_messages) < 5:
                if config.get("debug"):
                    typer.echo(f"   Not enough history for dual-window detection")
                return False, None, None
            
            # Run (4,1) detection - high precision
            high_precision_result = self._detect_with_window(
                all_messages, new_message, self.high_precision_window
            )
            
            # Run (4,2) detection - safety net
            safety_net_result = None
            if len(all_messages) >= 6:  # Need 6 messages for (4,2)
                safety_net_result = self._detect_with_window(
                    all_messages, new_message, self.safety_net_window
                )
            
            # Decision logic
            topic_changed = False
            detection_type = None
            
            if high_precision_result and high_precision_result['is_boundary']:
                # High confidence boundary detected
                topic_changed = True
                detection_type = "high_precision"
            elif safety_net_result and safety_net_result['is_boundary']:
                # Safety net caught a boundary
                topic_changed = True 
                detection_type = "safety_net"
            
            # Prepare combined detection info
            detection_info = {
                "method": "dual_window",
                "detection_type": detection_type,
                "high_precision": high_precision_result,
                "safety_net": safety_net_result,
                "is_boundary": topic_changed
            }
            
            if config.get("debug"):
                typer.echo(f"\n🔍 DEBUG: Dual-window detection results")
                if high_precision_result:
                    typer.echo(f"   High precision (4,1): score={high_precision_result['drift_score']:.3f}, boundary={high_precision_result['is_boundary']}")
                if safety_net_result:
                    typer.echo(f"   Safety net (4,2): score={safety_net_result['drift_score']:.3f}, boundary={safety_net_result['is_boundary']}")
                typer.echo(f"   Final decision: {'TOPIC CHANGED' if topic_changed else 'SAME TOPIC'} ({detection_type or 'no boundary'})")
            
            return topic_changed, None, detection_info
            
        except Exception as e:
            logger.error(f"Dual-window detection error: {e}")
            if config.get("debug"):
                typer.echo(f"   ❌ Dual-window detection error: {e}")
            return False, None, None
    
    def _detect_with_window(
        self,
        all_messages: List[Dict[str, Any]],
        new_message: str,
        window_config: WindowConfig
    ) -> Optional[Dict[str, Any]]:
        """
        Detect using a specific window configuration.
        
        Args:
            all_messages: All messages (newest-first)
            new_message: The new user message
            window_config: Window configuration to use
            
        Returns:
            Detection result dictionary or None if not enough messages
        """
        try:
            # Calculate window indices
            # For (4,1): need messages at indices 0-3 (before) and new message
            # For (4,2): need messages at indices 0-3 (before) and indices 0-1 + new message (after)
            
            if window_config.after_window == 1:
                # (4,1) window - compare last 4 messages with new message
                if len(all_messages) < 4:
                    return None
                
                # Get last 4 messages (indices 0-3, newest first)
                before_messages = all_messages[:4]
                # Reverse to chronological order
                before_messages.reverse()
                
                # Combine before window text
                before_text = " ".join([
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                    for msg in before_messages
                ])
                
                # After window is just the new message
                after_text = f"user: {new_message}"
                
            else:  # (4,2) window
                if len(all_messages) < 4:
                    return None
                    
                # Get messages at indices 2-5 (4 messages before the last 2)
                before_messages = all_messages[2:6] if len(all_messages) >= 6 else all_messages[2:]
                # Reverse to chronological order
                before_messages.reverse()
                
                # Get last 2 messages plus new message for after window
                after_messages = all_messages[:2]
                after_messages.reverse()
                
                # Combine texts
                before_text = " ".join([
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                    for msg in before_messages
                ])
                
                after_text = " ".join([
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                    for msg in after_messages
                ] + [f"user: {new_message}"])
            
            # Calculate drift score
            before_node = {"content": before_text}
            after_node = {"content": after_text}
            
            drift_score = self.drift_calculator.calculate_drift(
                before_node, after_node, text_field="content"
            )
            
            # Similarity is 1 - drift
            similarity = 1 - drift_score
            
            # Determine if boundary (low similarity = topic change)
            is_boundary = similarity < window_config.threshold
            
            return {
                "window_config": window_config.name,
                "drift_score": drift_score,
                "similarity": similarity,
                "threshold": window_config.threshold,
                "is_boundary": is_boundary,
                "before_window_size": len(before_messages) if 'before_messages' in locals() else 4,
                "after_window_size": window_config.after_window
            }
            
        except Exception as e:
            logger.error(f"Window detection error for {window_config.name}: {e}")
            return None