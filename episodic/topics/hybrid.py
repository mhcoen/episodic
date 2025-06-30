"""
Hybrid topic detection system for Episodic.

Combines multiple approaches:
1. Embedding-based semantic drift (primary)
2. Keyword-based detection (secondary)
3. LLM-based detection (fallback)
4. User commands (override)
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from episodic.ml.drift import ConversationalDrift
from episodic.config import config
from episodic.llm import query_llm
from episodic.db import get_recent_nodes
from .keywords import TransitionDetector, TopicChangeSignals
from .detector import TopicManager
import typer

logger = logging.getLogger(__name__)


class HybridScorer:
    """Combines multiple signals into a final topic change score."""
    
    def __init__(self):
        # Default weights - can be configured
        self.weights = config.get("hybrid_topic_weights", {
            "semantic_drift": 0.6,  # Increased weight for semantic drift
            "keyword_explicit": 0.25,
            "keyword_domain": 0.1,
            "message_gap": 0.025,
            "conversation_flow": 0.025
        })
        
        # Threshold for topic change
        self.topic_change_threshold = config.get("hybrid_topic_threshold", 0.55)  # Lowered for better sensitivity
        self.llm_fallback_threshold = config.get("hybrid_llm_threshold", 0.3)  # Lowered to reduce LLM calls
    
    def calculate_topic_change_score(self, signals: TopicChangeSignals) -> Tuple[float, str]:
        """
        Combine all signals into a final score with explanation.
        
        Returns:
            Tuple of (score, explanation)
        """
        signal_dict = signals.to_dict()
        
        # Calculate weighted score
        weighted_score = sum(
            self.weights.get(signal, 0) * value 
            for signal, value in signal_dict.items()
        )
        
        # Generate explanation
        explanation = self._generate_explanation(signal_dict, weighted_score)
        
        return weighted_score, explanation
    
    def _generate_explanation(self, signals: Dict[str, float], score: float) -> str:
        """Create human-readable explanation of the decision."""
        reasons = []
        
        # Semantic drift explanation
        if signals.get("semantic_drift", 0) > 0.7:
            reasons.append("high semantic drift")
        elif signals.get("semantic_drift", 0) > 0.4:
            reasons.append("moderate semantic drift")
        elif signals.get("semantic_drift", 0) > 0.2:
            reasons.append("low semantic drift")
            
        # Keyword-based explanations
        if signals.get("keyword_explicit", 0) > 0.5:
            reasons.append("explicit transition phrase")
            
        if signals.get("keyword_domain", 0) > 0.4:
            reasons.append("domain shift detected")
            
        if signals.get("message_gap", 0) > 0.5:
            reasons.append("significant time/length gap")
            
        if not reasons:
            reasons.append("minimal change signals")
        
        return f"Score: {score:.2f} - {'; '.join(reasons)}"


class HybridTopicDetector:
    """
    Main hybrid topic detection system.
    
    This combines multiple signals and makes the final decision about
    whether a topic change has occurred.
    """
    
    def __init__(self):
        # Initialize components
        self.drift_detector = ConversationalDrift()
        self.transition_detector = TransitionDetector()
        self.scorer = HybridScorer()
        self.topic_manager = TopicManager()  # For LLM fallback
        
        # Configuration
        self.use_or_logic = config.get("use_or_logic", True)  # Use OR instead of weighted average
        self.drift_threshold = config.get("drift_threshold", 0.75)
        self.keyword_threshold = config.get("keyword_threshold", 0.5)
        
    def detect_topic_change(
        self,
        recent_messages: List[Dict[str, Any]],
        new_message: str,
        current_topic: Optional[Tuple[str, str]] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Detect if topic has changed using hybrid approach.
        
        Returns:
            Tuple of (changed: bool, new_topic: str|None, cost_info: dict|None, debug_info: dict)
        """
        debug_info = {"method": "hybrid", "signals": {}}
        
        try:
            # Extract messages for analysis
            recent_texts = [msg.get("content", "") for msg in recent_messages if msg.get("content")]
            
            # 1. Calculate semantic drift
            try:
                drift_score = self.drift_detector.calculate_drift(recent_texts, new_message)
                signals = TopicChangeSignals(semantic_drift=drift_score)
                debug_info["signals"]["semantic_drift"] = drift_score
            except Exception as e:
                logger.warning(f"Drift calculation failed: {e}")
                signals = TopicChangeSignals()
                debug_info["errors"] = {"drift": str(e)}
            
            # 2. Detect transition keywords
            keyword_results = self.transition_detector.detect_transition_keywords(new_message)
            signals.keyword_explicit = keyword_results["explicit_transition"]
            signals.keyword_domain = keyword_results["domain_shift"]
            debug_info["signals"]["keywords"] = keyword_results
            
            # 3. Calculate message gaps and flow (simple heuristics for now)
            if len(recent_messages) > 1:
                # Check for significant length differences
                prev_len = len(recent_messages[-1].get("content", ""))
                curr_len = len(new_message)
                if prev_len > 0:
                    length_ratio = abs(curr_len - prev_len) / prev_len
                    signals.message_gap = min(length_ratio / 2, 1.0)  # Cap at 1.0
            
            # 4. Decision logic - use OR logic if configured
            if self.use_or_logic:
                # Semantic drift OR keywords indicate topic change
                drift_change = signals.semantic_drift >= self.drift_threshold
                keyword_change = (signals.keyword_explicit >= self.keyword_threshold or 
                                signals.keyword_domain >= self.keyword_threshold)
                
                topic_changed = drift_change or keyword_change
                score = max(signals.semantic_drift, signals.keyword_explicit, signals.keyword_domain)
                explanation = f"OR logic: drift={drift_change}, keywords={keyword_change}"
                
            else:
                # Calculate combined score using weighted average
                score, explanation = self.scorer.calculate_topic_change_score(signals)
                topic_changed = score >= self.scorer.topic_change_threshold
            
            debug_info["final_score"] = score
            debug_info["explanation"] = explanation
            debug_info["decision"] = "topic_changed" if topic_changed else "same_topic"
            
            # 5. LLM fallback for uncertain cases (if not using OR logic)
            if not self.use_or_logic and not topic_changed and score >= self.scorer.llm_fallback_threshold:
                if config.get("debug"):
                    typer.echo(f"   Hybrid uncertain (score={score:.2f}), using LLM fallback")
                
                # Use topic manager's LLM detection
                llm_changed, _, cost_info = self.topic_manager.detect_topic_change_separately(
                    recent_messages, new_message, current_topic
                )
                
                if llm_changed:
                    topic_changed = True
                    debug_info["llm_override"] = True
                    debug_info["decision"] = "topic_changed_llm"
                
                return topic_changed, None, cost_info, debug_info
            
            # Log decision if debugging
            if config.get("debug"):
                typer.echo(f"\n🔍 DEBUG: Hybrid detection complete")
                typer.echo(f"   Semantic drift: {signals.semantic_drift:.2f}")
                typer.echo(f"   Keyword signals: explicit={signals.keyword_explicit:.2f}, domain={signals.keyword_domain:.2f}")
                typer.echo(f"   Final score: {score:.2f}")
                typer.echo(f"   Decision: {'TOPIC CHANGED' if topic_changed else 'SAME TOPIC'}")
                typer.echo(f"   Explanation: {explanation}")
            
            return topic_changed, None, None, debug_info
            
        except Exception as e:
            logger.error(f"Hybrid detection error: {e}")
            debug_info["error"] = str(e)
            # Fall back to LLM detection on error
            return self.topic_manager.detect_topic_change_separately(
                recent_messages, new_message, current_topic
            ) + (debug_info,)