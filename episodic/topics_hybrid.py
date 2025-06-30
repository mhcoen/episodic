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
import typer

logger = logging.getLogger(__name__)


@dataclass
class TopicChangeSignals:
    """Container for all topic change detection signals."""
    semantic_drift: float = 0.0
    keyword_explicit: float = 0.0
    keyword_domain: float = 0.0
    message_gap: float = 0.0
    conversation_flow: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for scoring."""
        return {
            "semantic_drift": self.semantic_drift,
            "keyword_explicit": self.keyword_explicit,
            "keyword_domain": self.keyword_domain,
            "message_gap": self.message_gap,
            "conversation_flow": self.conversation_flow
        }


class TransitionDetector:
    """Detects explicit transition indicators in messages."""
    
    # Explicit transition phrases
    STRONG_TRANSITIONS = [
        "changing topics", "change topic", "different topic",
        "let's talk about", "let me ask about", "moving on to",
        "different subject", "another subject", "new subject",
        "by the way", "btw", "anyway", "speaking of",
        "on a different note", "switching gears", "switch gears", "pivot to",
        "let's switch", "can we talk about", "quick question about"
    ]
    
    # Softer transitions
    SOFT_TRANSITIONS = [
        "also", "another question", "one more thing",
        "additionally", "furthermore", "moreover"
    ]
    
    # Domain indicators
    DOMAIN_KEYWORDS = {
        "technical": ["code", "algorithm", "function", "database", "api", "debug", "error", "bug", "feature", "implement"],
        "cooking": ["recipe", "ingredient", "cook", "bake", "taste", "food", "meal", "dish", "kitchen", "chef"],
        "travel": ["visit", "trip", "flight", "hotel", "destination", "vacation", "journey", "tourist", "passport"],
        "health": ["doctor", "medicine", "symptom", "treatment", "diagnosis", "health", "medical", "patient"],
        "finance": ["money", "invest", "stock", "budget", "expense", "income", "tax", "profit", "loss"],
        "education": ["learn", "study", "course", "teacher", "student", "school", "university", "degree"],
        "entertainment": ["movie", "film", "music", "game", "show", "series", "actor", "director", "album"],
        "science": ["research", "experiment", "theory", "hypothesis", "data", "analysis", "study", "evidence"],
        "philosophy": ["think", "believe", "meaning", "purpose", "existence", "reality", "truth", "ethics"],
        "personal": ["feel", "emotion", "relationship", "family", "friend", "life", "experience", "story"]
    }
    
    def __init__(self):
        self.current_domain: Optional[str] = None
        self.domain_history: List[str] = []
    
    def detect_transition_keywords(self, message: str) -> Dict[str, Any]:
        """Detect transition indicators in a message."""
        message_lower = message.lower()
        
        # Check explicit transitions
        explicit_score = 0.0
        found_phrase = None
        for phrase in self.STRONG_TRANSITIONS:
            if phrase in message_lower:
                explicit_score = 0.9  # High confidence
                found_phrase = phrase
                break
        
        # Check soft transitions if no strong ones found
        if explicit_score < 0.5:
            for phrase in self.SOFT_TRANSITIONS:
                if phrase in message_lower:
                    explicit_score = max(explicit_score, 0.3)
                    if not found_phrase:
                        found_phrase = phrase
        
        # Check domain shifts
        domain_scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in message_lower)
            if matches > 0:
                domain_scores[domain] = matches / len(keywords)
        
        # Find dominant domain
        dominant_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else None
        
        # Calculate domain shift score
        domain_shift_score = 0.0
        if dominant_domain and self.current_domain:
            if dominant_domain != self.current_domain:
                domain_shift_score = 0.6  # Domain changed
        
        # Update current domain
        if dominant_domain:
            self.current_domain = dominant_domain
            self.domain_history.append(dominant_domain)
        
        return {
            "explicit_transition": explicit_score,
            "domain_shift": domain_shift_score,
            "detected_domains": domain_scores,
            "dominant_domain": dominant_domain,
            "found_phrase": found_phrase
        }


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
    
    Combines embedding-based drift, keyword detection, and LLM fallback.
    """
    
    def __init__(self):
        # Initialize components
        self.drift_calculator = ConversationalDrift(
            embedding_provider=config.get("embedding_provider", "sentence-transformers"),
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
            distance_algorithm=config.get("distance_algorithm", "cosine"),
            peak_strategy="threshold",
            threshold=0.35
        )
        
        self.transition_detector = TransitionDetector()
        self.scorer = HybridScorer()
        
        # Track state
        self.message_count_in_topic = 0
        self.topic_start_embedding = None
        self._last_transition_phrase = None
        
    def detect_topic_change(
        self,
        recent_messages: List[Dict],
        new_message: str,
        current_topic: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Main entry point for hybrid topic detection.
        
        Args:
            recent_messages: List of recent message dictionaries
            new_message: The new user message
            current_topic: Current topic name (if any)
            
        Returns:
            Tuple of (topic_changed, new_topic_name, metadata)
        """
        signals = TopicChangeSignals()
        
        # 1. Check minimum messages threshold
        user_messages = [msg for msg in recent_messages if msg.get('role') == 'user']
        if len(user_messages) < 1:  # Need at least 1 previous user message
            return False, None, {"method": "too_few_messages"}
        
        # 2. Calculate semantic drift
        try:
            # Get user message contents only
            user_contents = [msg['content'] for msg in user_messages[-5:]]  # Last 5 user messages
            
            # Calculate drift from recent messages
            if len(user_contents) >= 1:
                # Create node dictionaries for drift calculation
                # Compare last user message to new message
                node1 = {"message": user_contents[-1]}  # Last user message
                node2 = {"message": new_message}  # New message
                drift_score = self.drift_calculator.calculate_drift(node1, node2)
                signals.semantic_drift = drift_score
                
                # Keep raw drift score - it's already normalized 0-1
                signals.semantic_drift = drift_score
                    
        except Exception as e:
            logger.warning(f"Error calculating semantic drift: {e}")
        
        # 3. Check keyword indicators
        keyword_results = self.transition_detector.detect_transition_keywords(new_message)
        signals.keyword_explicit = keyword_results["explicit_transition"]
        signals.keyword_domain = keyword_results["domain_shift"]
        
        # Store found phrase for explanation
        self._last_transition_phrase = keyword_results.get("found_phrase")
        
        # 4. Check message gaps (simplified for now)
        signals.message_gap = 0.0  # TODO: Implement time-based gaps
        signals.conversation_flow = 0.0  # TODO: Implement question/answer pattern detection
        
        # 5. Calculate final score
        final_score, explanation = self.scorer.calculate_topic_change_score(signals)
        
        # 6. Make decision
        metadata = {
            "method": "hybrid",
            "score": final_score,
            "explanation": explanation,
            "signals": signals.to_dict(),
            "transition_phrase": self._last_transition_phrase
        }
        
        if final_score >= self.scorer.topic_change_threshold:
            # High confidence - topic change detected
            if config.get("debug", False):
                typer.echo(f"🔄 Topic change detected: {explanation}")
            return True, None, metadata
            
        elif final_score >= self.scorer.llm_fallback_threshold:
            # Uncertain - use LLM fallback
            if config.get("debug", False):
                typer.echo(f"🤔 Uncertain ({final_score:.2f}), using LLM fallback")
            
            # Call existing LLM detection
            from episodic.topics import detect_topic_change_separately
            llm_result = detect_topic_change_separately(recent_messages, new_message, current_topic)
            
            metadata["method"] = "llm_fallback"
            metadata["llm_result"] = llm_result[0]
            
            return llm_result[0], llm_result[1], metadata
        
        else:
            # Low score - no topic change
            if config.get("debug", False):
                typer.echo(f"✅ Same topic: {explanation}")
            return False, None, metadata


# Global instance
hybrid_detector = HybridTopicDetector()


def detect_topic_change_hybrid(
    recent_messages: List[Dict],
    new_message: str,
    current_topic: Optional[str] = None
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Convenience function for hybrid topic detection.
    
    This is the main entry point that other modules should use.
    """
    return hybrid_detector.detect_topic_change(recent_messages, new_message, current_topic)