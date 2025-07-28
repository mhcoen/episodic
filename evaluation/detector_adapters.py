#!/usr/bin/env python3
"""
Adapters to run Episodic's topic detectors on evaluation datasets.

This module provides a unified interface to test different topic detection
strategies on the SuperDialseg dataset.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from episodic.topics.realtime_windows import RealtimeWindowDetector
from episodic.topics.hybrid import HybridTopicDetector
from episodic.topics.detector import TopicManager
from episodic.topics.keywords import TransitionDetector
from episodic.config import config


class BaseDetectorAdapter:
    """Base class for topic detection adapters."""
    
    def __init__(self, name: str):
        self.name = name
        self.detection_history = []
    
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """
        Detect topic boundaries in a conversation.
        
        Args:
            messages: List of message dicts with 'role', 'content', and 'index'
            
        Returns:
            List of boundary indices (positions after which topics change)
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset detector state between conversations."""
        self.detection_history = []


class SlidingWindowAdapter(BaseDetectorAdapter):
    """Adapter for sliding window detection."""
    
    def __init__(self, window_size: int = 3, threshold: float = 0.9):
        super().__init__(f"sliding_window_w{window_size}")
        self.detector = RealtimeWindowDetector(window_size=window_size)
        self.threshold = threshold
        
        # Store threshold for later use
        self.original_threshold = config.get('drift_threshold', 0.9)
    
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect boundaries using sliding window approach."""
        boundaries = []
        recent_messages = []
        
        # Temporarily override config threshold and enable debug
        old_threshold = config.get('drift_threshold', 0.9)
        old_debug = config.get('debug', False)
        config.set('drift_threshold', self.threshold)
        # Disable debug for now
        config.set('debug', False)
        
        try:
            for i, msg in enumerate(messages):
                if msg['role'] == 'user':
                    # Build recent message history (newest first)
                    recent_reversed = list(reversed(recent_messages))
                    
                    # Detect topic change
                    changed, _, detection_info = self.detector.detect_topic_change(
                        recent_reversed,
                        msg['content'],
                        current_topic=None  # We don't track topics in evaluation
                    )
                    
                    if changed and i > 0:
                        # Add boundary at previous position
                        boundaries.append(i - 1)
                
                # Add to history
                recent_messages.append(msg)
                
                # Keep only last N messages for efficiency
                if len(recent_messages) > 20:
                    recent_messages = recent_messages[-20:]
        
        finally:
            # Restore original settings
            config.set('drift_threshold', old_threshold)
            config.set('debug', old_debug)
        
        return boundaries


class HybridDetectorAdapter(BaseDetectorAdapter):
    """Adapter for hybrid detection."""
    
    def __init__(self, threshold: float = 0.6):
        super().__init__("hybrid")
        self.detector = HybridTopicDetector()
        self.threshold = threshold
    
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect boundaries using hybrid approach."""
        boundaries = []
        recent_messages = []
        
        for i, msg in enumerate(messages):
            if msg['role'] == 'user':
                # Build recent message history (newest first)
                recent_reversed = list(reversed(recent_messages))
                
                # Detect topic change (hybrid returns 4 values)
                result = self.detector.detect_topic_change(
                    recent_reversed,
                    msg['content'],
                    current_topic=None
                )
                # Handle both 3 and 4 value returns
                if len(result) == 4:
                    changed, _, detection_info, _ = result
                else:
                    changed, _, detection_info = result
                
                if changed and i > 0:
                    boundaries.append(i - 1)
            
            recent_messages.append(msg)
            
            # Keep only last N messages
            if len(recent_messages) > 20:
                recent_messages = recent_messages[-20:]
        
        return boundaries


class KeywordDetectorAdapter(BaseDetectorAdapter):
    """Adapter for keyword-based detection."""
    
    def __init__(self, threshold: float = 0.5):
        super().__init__("keywords")
        self.detector = TransitionDetector()
        self.threshold = threshold
    
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect boundaries using keyword signals."""
        boundaries = []
        
        for i, msg in enumerate(messages):
            if msg['role'] == 'user' and i > 0:
                results = self.detector.detect_transition_keywords(msg['content'])
                
                # Combine explicit and domain shift scores
                score = max(
                    results.get('explicit_transition', 0.0),
                    results.get('domain_shift', 0.0) * 0.8
                )
                
                if score >= self.threshold:
                    boundaries.append(i - 1)
        
        return boundaries


class LLMDetectorAdapter(BaseDetectorAdapter):
    """Adapter for LLM-based detection."""
    
    def __init__(self, model: str = None):
        super().__init__(f"llm_{model or 'default'}")
        self.topic_manager = TopicManager()
        self.model = model
    
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect boundaries using LLM."""
        boundaries = []
        recent_messages = []
        
        # Configure to use specified model if provided
        if self.model:
            old_model = config.get('topic_detection_model')
            config._overrides['topic_detection_model'] = self.model
        
        try:
            for i, msg in enumerate(messages):
                if msg['role'] == 'user' and len(recent_messages) >= 4:
                    # Need at least 2 exchanges for detection
                    recent_reversed = list(reversed(recent_messages))
                    
                    # Detect topic change
                    changed, _, _ = self.topic_manager.detect_topic_change_separately(
                        recent_reversed,
                        msg['content'],
                        current_topic=None
                    )
                    
                    if changed and i > 0:
                        boundaries.append(i - 1)
                
                recent_messages.append(msg)
                
                # Keep only last N messages
                if len(recent_messages) > 20:
                    recent_messages = recent_messages[-20:]
        
        finally:
            # Restore original model
            if self.model and 'old_model' in locals():
                config._overrides['topic_detection_model'] = old_model
        
        return boundaries


class CombinedDetectorAdapter(BaseDetectorAdapter):
    """Adapter that combines multiple detection strategies."""
    
    def __init__(self, detectors: List[BaseDetectorAdapter], voting: str = 'majority'):
        names = [d.name for d in detectors]
        super().__init__(f"combined_{voting}_" + "_".join(names))
        self.detectors = detectors
        self.voting = voting
    
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect boundaries by combining multiple detectors."""
        # Get boundaries from each detector
        all_boundaries = []
        for detector in self.detectors:
            boundaries = detector.detect_boundaries(messages)
            all_boundaries.append(set(boundaries))
            detector.reset()
        
        # Combine based on voting strategy
        if self.voting == 'majority':
            # Boundary must be detected by majority of detectors
            threshold = len(self.detectors) // 2 + 1
            combined = []
            
            # Check each possible boundary position
            max_pos = len(messages) - 1
            for pos in range(max_pos):
                votes = sum(1 for boundary_set in all_boundaries if pos in boundary_set)
                if votes >= threshold:
                    combined.append(pos)
            
            return combined
        
        elif self.voting == 'any':
            # Boundary detected by any detector
            combined = set()
            for boundary_set in all_boundaries:
                combined.update(boundary_set)
            return sorted(list(combined))
        
        elif self.voting == 'unanimous':
            # Boundary must be detected by all detectors
            if not all_boundaries:
                return []
            combined = all_boundaries[0]
            for boundary_set in all_boundaries[1:]:
                combined = combined.intersection(boundary_set)
            return sorted(list(combined))
        
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting}")
    
    def reset(self):
        """Reset all sub-detectors."""
        super().reset()
        for detector in self.detectors:
            detector.reset()


def create_detector(detector_type: str, **kwargs) -> BaseDetectorAdapter:
    """
    Factory function to create detector adapters.
    
    Args:
        detector_type: Type of detector ('sliding_window', 'hybrid', 'keywords', 'llm', 'combined')
        **kwargs: Arguments specific to each detector type
        
    Returns:
        Configured detector adapter
    """
    if detector_type == 'sliding_window':
        return SlidingWindowAdapter(
            window_size=kwargs.get('window_size', 3),
            threshold=kwargs.get('threshold', 0.9)
        )
    
    elif detector_type == 'hybrid':
        return HybridDetectorAdapter(
            threshold=kwargs.get('threshold', 0.6)
        )
    
    elif detector_type == 'keywords':
        return KeywordDetectorAdapter(
            threshold=kwargs.get('threshold', 0.5)
        )
    
    elif detector_type == 'llm':
        return LLMDetectorAdapter(
            model=kwargs.get('model', None)
        )
    
    elif detector_type == 'combined':
        # Create sub-detectors
        sub_types = kwargs.get('detectors', ['sliding_window', 'keywords'])
        sub_detectors = []
        for sub_type in sub_types:
            sub_detectors.append(create_detector(sub_type))
        
        return CombinedDetectorAdapter(
            detectors=sub_detectors,
            voting=kwargs.get('voting', 'majority')
        )
    
    elif detector_type == 'supervised':
        # Import here to avoid loading models unnecessarily
        from evaluation.supervised_detector import (
            SupervisedWindowDetector, HybridSupervisedDetector
        )
        
        variant = kwargs.get('variant', 'window')
        if variant == 'hybrid':
            return HybridSupervisedDetector(
                model_name=kwargs.get('model_name', 'sentence-transformers/all-mpnet-base-v2'),
                window_size=kwargs.get('window_size', 3),
                drift_threshold=kwargs.get('threshold', 0.4)
            )
        else:
            return SupervisedWindowDetector(
                model_name=kwargs.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
                window_size=kwargs.get('window_size', 3),
                threshold=kwargs.get('threshold', 0.5)
            )
    
    elif detector_type == 'bayesian':
        # Import here to avoid dependency issues
        from evaluation.bayesian_detector import (
            ImprovedBayesianAdapter, WindowedBayesianAdapter
        )
        
        variant = kwargs.get('variant', 'windowed')
        if variant == 'windowed':
            return WindowedBayesianAdapter(
                window_size=kwargs.get('window_size', 3),
                hazard_lambda=kwargs.get('hazard_lambda', 50),
                threshold=kwargs.get('threshold', 0.3)
            )
        else:
            return ImprovedBayesianAdapter(
                hazard_lambda=kwargs.get('hazard_lambda', 100),
                threshold=kwargs.get('threshold', 0.5)
            )
    
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


if __name__ == "__main__":
    # Test the adapters
    print("Testing detector adapters...")
    
    # Create test messages
    test_messages = [
        {'role': 'user', 'content': 'Tell me about quantum physics', 'index': 0},
        {'role': 'assistant', 'content': 'Quantum physics is...', 'index': 1},
        {'role': 'user', 'content': 'What are the key principles?', 'index': 2},
        {'role': 'assistant', 'content': 'The key principles include...', 'index': 3},
        {'role': 'user', 'content': "Now let's talk about machine learning", 'index': 4},
        {'role': 'assistant', 'content': 'Machine learning is...', 'index': 5},
    ]
    
    # Test each detector type
    for detector_type in ['sliding_window', 'keywords']:
        print(f"\nTesting {detector_type}:")
        detector = create_detector(detector_type)
        boundaries = detector.detect_boundaries(test_messages)
        print(f"  Detected boundaries: {boundaries}")