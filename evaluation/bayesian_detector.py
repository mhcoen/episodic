#!/usr/bin/env python3
"""
Bayesian Online Changepoint Detection for topic segmentation.

This implements BOCPD on embedding-based features for real-time
topic boundary detection.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import bayesian_changepoint_detection.online_changepoint_detection as oncd
from functools import partial

# Import from parent directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from episodic.ml.drift import ConversationalDrift
from evaluation.detector_adapters import BaseDetectorAdapter


class BayesianChangePointAdapter(BaseDetectorAdapter):
    """Adapter for Bayesian changepoint detection."""
    
    def __init__(self, hazard_lambda: float = 100, threshold: float = 0.5):
        """
        Initialize Bayesian detector.
        
        Args:
            hazard_lambda: Expected run length (higher = fewer changepoints)
            threshold: Probability threshold for declaring changepoint
        """
        super().__init__(f"bayesian_h{hazard_lambda}_t{threshold}")
        self.hazard_lambda = hazard_lambda
        self.threshold = threshold
        self.drift_calculator = ConversationalDrift()
        
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect boundaries using Bayesian changepoint detection."""
        if len(messages) < 2:
            return []
            
        # Extract features (embeddings) for each message
        features = []
        for msg in messages:
            # Get embedding for the message
            embedding = self.drift_calculator.embedder.embed(msg['content'])
            features.append(embedding)
        
        # Convert to numpy array
        features = np.array(features)
        
        # Set up the changepoint detector
        # Using Gaussian observation model
        hazard_func = partial(oncd.constant_hazard, self.hazard_lambda)
        
        # Initialize R (run length probabilities)
        R = np.zeros((len(messages) + 1, len(messages) + 1))
        R[0, 0] = 1
        
        boundaries = []
        
        # Process each observation
        for t in range(1, len(messages)):
            # Get the feature vector for time t
            x_t = features[t]
            
            # Update run length probabilities
            # This is a simplified version - full implementation would need
            # proper likelihood computation for embeddings
            
            # Compute growth probability (no changepoint)
            growth_probs = R[t-1, :t] * (1 - 1/self.hazard_lambda)
            
            # Compute changepoint probability
            cp_prob = np.sum(R[t-1, :t]) / self.hazard_lambda
            
            # Update R
            R[t, 1:t+1] = growth_probs
            R[t, 0] = cp_prob
            
            # Normalize
            R[t, :] = R[t, :] / np.sum(R[t, :])
            
            # Check if changepoint probability exceeds threshold
            if cp_prob > self.threshold and messages[t]['role'] == 'user':
                # Mark boundary at previous position
                boundaries.append(t - 1)
        
        return boundaries


class ImprovedBayesianAdapter(BaseDetectorAdapter):
    """
    Improved Bayesian changepoint detection using drift scores as observations.
    """
    
    def __init__(self, hazard_lambda: float = 100, threshold: float = 0.5):
        super().__init__(f"bayesian_improved_h{hazard_lambda}_t{threshold}")
        self.hazard_lambda = hazard_lambda
        self.threshold = threshold
        self.drift_calculator = ConversationalDrift()
        
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect boundaries using drift scores with BOCPD."""
        if len(messages) < 4:
            return []
        
        # Calculate drift scores between consecutive user messages
        drift_scores = []
        user_messages = [(i, msg) for i, msg in enumerate(messages) if msg['role'] == 'user']
        
        if len(user_messages) < 2:
            return []
        
        # Calculate drift between each pair of consecutive user messages
        for i in range(1, len(user_messages)):
            prev_idx, prev_msg = user_messages[i-1]
            curr_idx, curr_msg = user_messages[i]
            
            # Calculate semantic drift
            drift = self.drift_calculator.calculate_drift(
                {'content': prev_msg['content']},
                {'content': curr_msg['content']},
                text_field='content'
            )
            drift_scores.append((curr_idx, drift))
        
        if not drift_scores:
            return []
        
        # Convert drift scores to observations for BOCPD
        observations = np.array([score for _, score in drift_scores]).reshape(-1, 1)
        
        # Run BOCPD on drift scores
        hazard_func = partial(oncd.constant_hazard, self.hazard_lambda)
        
        # Use Gaussian observation model
        R, maxes = oncd.online_changepoint_detection(
            observations,
            hazard_func,
            oncd.StudentT(0.1, 0.01, 1, 0)  # Student-t distribution
        )
        
        # Find changepoints where probability exceeds threshold
        boundaries = []
        for t in range(1, len(drift_scores)):
            # Changepoint probability is R[t, 0]
            cp_prob = R[t, 0]
            if cp_prob > self.threshold:
                # Get the message index where boundary should be placed
                msg_idx = drift_scores[t-1][0]
                if msg_idx > 0:
                    boundaries.append(msg_idx - 1)
        
        return boundaries


class WindowedBayesianAdapter(BaseDetectorAdapter):
    """
    Bayesian changepoint detection on windowed features.
    Similar to our sliding window but with Bayesian inference.
    """
    
    def __init__(self, window_size: int = 3, hazard_lambda: float = 50, threshold: float = 0.3):
        super().__init__(f"bayesian_windowed_w{window_size}_h{hazard_lambda}")
        self.window_size = window_size
        self.hazard_lambda = hazard_lambda
        self.threshold = threshold
        self.drift_calculator = ConversationalDrift()
        
    def detect_boundaries(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Detect boundaries using windowed BOCPD."""
        boundaries = []
        user_messages = []
        
        # Process messages sequentially
        for i, msg in enumerate(messages):
            if msg['role'] == 'user':
                user_messages.append((i, msg))
                
                # Need at least window_size + 1 user messages
                if len(user_messages) > self.window_size:
                    # Calculate features for the window
                    window_features = []
                    
                    # Get drift scores within window
                    for j in range(len(user_messages) - self.window_size, len(user_messages)):
                        if j > 0:
                            prev_msg = user_messages[j-1][1]
                            curr_msg = user_messages[j][1]
                            drift = self.drift_calculator.calculate_drift(
                                {'content': prev_msg['content']},
                                {'content': curr_msg['content']},
                                text_field='content'
                            )
                            window_features.append(drift)
                    
                    if window_features:
                        # Simple changepoint detection: high drift indicates boundary
                        avg_drift = np.mean(window_features)
                        max_drift = np.max(window_features)
                        
                        # Bayesian update (simplified)
                        # Prior: no changepoint
                        prior_no_cp = 1 - 1/self.hazard_lambda
                        prior_cp = 1/self.hazard_lambda
                        
                        # Likelihood based on drift magnitude
                        # High drift more likely under changepoint hypothesis
                        likelihood_cp = 1 / (1 + np.exp(-10 * (max_drift - 0.5)))
                        likelihood_no_cp = 1 - likelihood_cp
                        
                        # Posterior
                        posterior_cp = (likelihood_cp * prior_cp) / (
                            likelihood_cp * prior_cp + likelihood_no_cp * prior_no_cp
                        )
                        
                        if posterior_cp > self.threshold:
                            msg_idx = user_messages[-1][0]
                            if msg_idx > 0:
                                boundaries.append(msg_idx - 1)
        
        return boundaries


if __name__ == "__main__":
    # Test the Bayesian detectors
    from evaluation.superdialseg_loader import SuperDialsegLoader
    from evaluation.metrics import SegmentationMetrics
    
    print("Testing Bayesian changepoint detectors...")
    
    # Load sample data
    loader = SuperDialsegLoader()
    dataset_path = Path("/Users/mhcoen/proj/episodic/datasets/superseg")
    conversations = loader.load_conversations(dataset_path, 'test')
    
    # Test on first conversation
    conv = conversations[0]
    messages, gold_boundaries = loader.parse_conversation(conv)
    
    print(f"\nConversation: {len(messages)} messages")
    print(f"Gold boundaries: {gold_boundaries}")
    
    # Test different Bayesian detectors
    detectors = [
        ImprovedBayesianAdapter(hazard_lambda=100, threshold=0.5),
        ImprovedBayesianAdapter(hazard_lambda=50, threshold=0.3),
        WindowedBayesianAdapter(window_size=3, hazard_lambda=50, threshold=0.3),
    ]
    
    metrics_calc = SegmentationMetrics()
    
    for detector in detectors:
        predicted = detector.detect_boundaries(messages)
        print(f"\n{detector.name}:")
        print(f"  Predicted: {predicted}")
        
        if gold_boundaries:
            metrics = metrics_calc.calculate_exact_metrics(
                predicted, gold_boundaries, len(messages)
            )
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1: {metrics['f1']:.3f}")