"""
Hybrid topic detection system for Episodic.

Combines multiple approaches:
1. Embedding-based semantic drift (primary)
2. Keyword-based detection (secondary)
3. LLM-based detection (fallback)
4. User commands (override)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from episodic.ml.drift import ConversationalDrift
from episodic.config import config
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
        # Get embedding settings from config
        embedding_provider = config.get("drift_embedding_provider", "sentence-transformers")
        embedding_model = config.get("drift_embedding_model", "paraphrase-mpnet-base-v2")
        
        self.drift_detector = ConversationalDrift(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )
        self.transition_detector = TransitionDetector()
        self.scorer = HybridScorer()
        self.topic_manager = TopicManager()  # For LLM fallback
        
        # Default configuration (can be overridden per detection)
        self.default_use_or_logic = True
        self.default_drift_threshold = 0.85
        self.default_keyword_threshold = 0.5
        
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
            [msg.get("content", "") for msg in recent_messages if msg.get("content")]
            
            # 1. Calculate semantic drift
            try:
                # Get the most recent user message to compare with the new one
                user_messages = [msg for msg in recent_messages if msg.get("role") == "user"]
                if len(user_messages) >= 1:
                    # Create node-like dictionaries for drift calculation
                    prev_node = {"content": user_messages[-1].get("content", "")}
                    new_node = {"content": new_message}
                    drift_score = self.drift_detector.calculate_drift(prev_node, new_node, text_field="content")
                else:
                    # No previous user message to compare
                    drift_score = 0.0
                    
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
            # Read configuration dynamically
            use_or_logic = config.get("use_or_logic", self.default_use_or_logic)
            drift_threshold = float(config.get("drift_threshold", self.default_drift_threshold))
            keyword_threshold = float(config.get("keyword_threshold", self.default_keyword_threshold))
            
            if use_or_logic:
                # Semantic drift OR keywords indicate topic change
                drift_change = signals.semantic_drift >= drift_threshold
                keyword_change = (signals.keyword_explicit >= keyword_threshold or 
                                signals.keyword_domain >= keyword_threshold)
                
                # Check if messages are in same domain (reduces false positives)
                same_domain_penalty = 0.0
                if "dominant_domain" in keyword_results and "previous_domain" in keyword_results:
                    if keyword_results["dominant_domain"] == keyword_results["previous_domain"] and keyword_results["dominant_domain"] is not None:
                        # Both messages in same domain - reduce drift score impact
                        same_domain_penalty = 0.1
                        if config.get("debug"):
                            typer.echo(f"   Same domain detected: {keyword_results['dominant_domain']} (reducing drift impact)")
                
                # Apply domain penalty to drift threshold
                effective_drift_threshold = drift_threshold + same_domain_penalty
                drift_change = signals.semantic_drift >= effective_drift_threshold
                
                topic_changed = drift_change or keyword_change
                score = max(signals.semantic_drift, signals.keyword_explicit, signals.keyword_domain)
                explanation = f"OR logic: drift={drift_change} (threshold={effective_drift_threshold:.2f}), keywords={keyword_change}"
                
            else:
                # Calculate combined score using weighted average
                score, explanation = self.scorer.calculate_topic_change_score(signals)
                topic_changed = score >= self.scorer.topic_change_threshold
            
            debug_info["final_score"] = score
            debug_info["explanation"] = explanation
            debug_info["decision"] = "topic_changed" if topic_changed else "same_topic"
            
            # 5. LLM fallback for uncertain cases (if not using OR logic)
            if not use_or_logic and not topic_changed and score >= self.scorer.llm_fallback_threshold:
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