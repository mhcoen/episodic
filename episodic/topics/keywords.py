"""
Keyword-based topic detection functionality.

This module contains classes for detecting topic transitions based on
keywords and domain-specific terminology.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


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
            # Count keyword matches (with word boundaries for accuracy)
            matches = 0
            for kw in keywords:
                # Check for word boundaries to avoid partial matches
                import re
                if re.search(r'\b' + re.escape(kw) + r'\b', message_lower):
                    matches += 1
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