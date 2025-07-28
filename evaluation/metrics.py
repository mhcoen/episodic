#!/usr/bin/env python3
"""
Evaluation metrics for topic segmentation.

This module implements standard metrics for evaluating dialogue segmentation:
- Precision, Recall, F1 for exact boundary matching
- WindowDiff for near-boundary evaluation
- Pk (Beeferman's metric) for segmentation quality
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict


class SegmentationMetrics:
    """Calculate segmentation evaluation metrics."""
    
    def __init__(self, tolerance_window: int = 3):
        """
        Initialize metrics calculator.
        
        Args:
            tolerance_window: Window size for WindowDiff and boundary tolerance
        """
        self.tolerance_window = tolerance_window
    
    def calculate_exact_metrics(
        self, 
        predicted_boundaries: List[int], 
        gold_boundaries: List[int],
        num_utterances: int
    ) -> Dict[str, float]:
        """
        Calculate exact boundary matching metrics.
        
        Args:
            predicted_boundaries: List of predicted boundary indices
            gold_boundaries: List of gold standard boundary indices
            num_utterances: Total number of utterances in the dialogue
            
        Returns:
            Dict with precision, recall, and F1 scores
        """
        predicted_set = set(predicted_boundaries)
        gold_set = set(gold_boundaries)
        
        # True positives: boundaries that are in both sets
        true_positives = len(predicted_set & gold_set)
        
        # False positives: predicted boundaries not in gold
        false_positives = len(predicted_set - gold_set)
        
        # False negatives: gold boundaries not predicted
        false_negatives = len(gold_set - predicted_set)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if predicted_set else 0.0
        recall = true_positives / (true_positives + false_negatives) if gold_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives)
        }
    
    def calculate_windowed_metrics(
        self,
        predicted_boundaries: List[int],
        gold_boundaries: List[int],
        num_utterances: int,
        window: int = None
    ) -> Dict[str, float]:
        """
        Calculate metrics with tolerance window.
        
        A predicted boundary is considered correct if it's within
        'window' positions of a gold boundary.
        """
        if window is None:
            window = self.tolerance_window
            
        # For each gold boundary, check if there's a predicted boundary nearby
        matched_gold = set()
        matched_pred = set()
        
        for gold_idx in gold_boundaries:
            for pred_idx in predicted_boundaries:
                if abs(gold_idx - pred_idx) <= window and pred_idx not in matched_pred:
                    matched_gold.add(gold_idx)
                    matched_pred.add(pred_idx)
                    break
        
        # Calculate metrics based on matches
        true_positives = len(matched_gold)
        false_positives = len(predicted_boundaries) - len(matched_pred)
        false_negatives = len(gold_boundaries) - len(matched_gold)
        
        precision = true_positives / (true_positives + false_positives) if predicted_boundaries else 0.0
        recall = true_positives / (true_positives + false_negatives) if gold_boundaries else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            f'precision_w{window}': float(precision),
            f'recall_w{window}': float(recall),
            f'f1_w{window}': float(f1),
            f'matched_boundaries': int(len(matched_gold))
        }
    
    def calculate_window_diff(
        self,
        predicted_boundaries: List[int],
        gold_boundaries: List[int],
        num_utterances: int,
        k: int = None
    ) -> float:
        """
        Calculate WindowDiff metric.
        
        WindowDiff slides a window of size k across the sequence and counts
        disagreements in the number of boundaries within each window.
        
        Lower is better (0 is perfect).
        """
        if k is None:
            k = self.tolerance_window
            
        # Convert boundaries to segment representation
        pred_segments = self._boundaries_to_segments(predicted_boundaries, num_utterances)
        gold_segments = self._boundaries_to_segments(gold_boundaries, num_utterances)
        
        if len(pred_segments) != len(gold_segments):
            raise ValueError("Segment representations must have same length")
        
        # Calculate WindowDiff
        errors = 0
        comparisons = 0
        
        for i in range(len(pred_segments) - k + 1):
            # Count boundaries in window for predicted
            pred_boundaries_in_window = sum(
                1 for j in range(i, i + k - 1)
                if pred_segments[j] != pred_segments[j + 1]
            )
            
            # Count boundaries in window for gold
            gold_boundaries_in_window = sum(
                1 for j in range(i, i + k - 1)
                if gold_segments[j] != gold_segments[j + 1]
            )
            
            if pred_boundaries_in_window != gold_boundaries_in_window:
                errors += 1
            comparisons += 1
        
        return errors / comparisons if comparisons > 0 else 0.0
    
    def calculate_pk(
        self,
        predicted_boundaries: List[int],
        gold_boundaries: List[int],
        num_utterances: int,
        k: int = None
    ) -> float:
        """
        Calculate Pk (Beeferman's metric).
        
        Pk measures the probability that two utterances k positions apart
        are incorrectly classified as being in the same/different segments.
        
        Lower is better (0 is perfect).
        """
        if k is None:
            # Pk typically uses k = half the average segment length
            avg_segment_length = num_utterances / (len(gold_boundaries) + 1)
            k = max(2, int(avg_segment_length / 2))
        
        # Convert boundaries to segment representation
        pred_segments = self._boundaries_to_segments(predicted_boundaries, num_utterances)
        gold_segments = self._boundaries_to_segments(gold_boundaries, num_utterances)
        
        # Calculate Pk
        errors = 0
        comparisons = 0
        
        for i in range(len(pred_segments) - k):
            # Check if utterances i and i+k are in same segment
            pred_same = pred_segments[i] == pred_segments[i + k]
            gold_same = gold_segments[i] == gold_segments[i + k]
            
            if pred_same != gold_same:
                errors += 1
            comparisons += 1
        
        return errors / comparisons if comparisons > 0 else 0.0
    
    def _boundaries_to_segments(self, boundaries: List[int], num_utterances: int) -> List[int]:
        """
        Convert boundary indices to segment labels.
        
        Returns a list where each position contains the segment ID for that utterance.
        """
        segments = []
        current_segment = 0
        boundaries_set = set(boundaries)
        
        for i in range(num_utterances):
            segments.append(current_segment)
            # If this is a boundary position, next utterance starts new segment
            if i in boundaries_set:
                current_segment += 1
        
        return segments
    
    def evaluate_all(
        self,
        predicted_boundaries: List[int],
        gold_boundaries: List[int],
        num_utterances: int
    ) -> Dict[str, float]:
        """
        Calculate all metrics at once.
        
        Returns dict with all metric values.
        """
        results = {}
        
        # Exact metrics
        exact = self.calculate_exact_metrics(predicted_boundaries, gold_boundaries, num_utterances)
        results.update(exact)
        
        # Windowed metrics (multiple window sizes)
        for window in [1, 3, 5]:
            windowed = self.calculate_windowed_metrics(
                predicted_boundaries, gold_boundaries, num_utterances, window
            )
            results.update(windowed)
        
        # WindowDiff
        results['window_diff'] = self.calculate_window_diff(
            predicted_boundaries, gold_boundaries, num_utterances
        )
        
        # Pk
        results['pk'] = self.calculate_pk(
            predicted_boundaries, gold_boundaries, num_utterances
        )
        
        return results


class EvaluationResults:
    """Aggregate evaluation results across multiple dialogues."""
    
    def __init__(self):
        self.dialogue_results = []
        self.metric_totals = defaultdict(list)
    
    def add_dialogue(self, dialogue_id: str, metrics: Dict[str, float]):
        """Add results for a single dialogue."""
        self.dialogue_results.append({
            'dialogue_id': dialogue_id,
            'metrics': metrics
        })
        
        # Accumulate metrics
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metric_totals[metric].append(value)
    
    def get_aggregate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate aggregate statistics across all dialogues."""
        aggregates = {}
        
        for metric, values in self.metric_totals.items():
            if values:
                aggregates[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        return aggregates
    
    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary."""
        aggregates = self.get_aggregate_metrics()
        
        return {
            'num_dialogues': len(self.dialogue_results),
            'metrics': {
                'precision': aggregates.get('precision', {}).get('mean', 0.0),
                'recall': aggregates.get('recall', {}).get('mean', 0.0),
                'f1': aggregates.get('f1', {}).get('mean', 0.0),
                'window_diff': aggregates.get('window_diff', {}).get('mean', 0.0),
                'pk': aggregates.get('pk', {}).get('mean', 0.0),
            },
            'detailed_aggregates': aggregates
        }
    
    def print_summary(self):
        """Print a formatted summary of results."""
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Dialogues evaluated: {summary['num_dialogues']}")
        print("\nMain Metrics:")
        print(f"  Precision: {summary['metrics']['precision']:.3f}")
        print(f"  Recall:    {summary['metrics']['recall']:.3f}")
        print(f"  F1:        {summary['metrics']['f1']:.3f}")
        print(f"  WindowDiff: {summary['metrics']['window_diff']:.3f}")
        print(f"  Pk:         {summary['metrics']['pk']:.3f}")
        
        # Print windowed metrics
        print("\nWindowed Metrics:")
        for window in [1, 3, 5]:
            f1_key = f'f1_w{window}'
            if f1_key in summary['detailed_aggregates']:
                f1_mean = summary['detailed_aggregates'][f1_key]['mean']
                print(f"  F1 (w={window}): {f1_mean:.3f}")


if __name__ == "__main__":
    # Test the metrics
    metrics_calc = SegmentationMetrics()
    
    # Example: 10 utterances with boundaries after positions 2 and 6
    gold_boundaries = [2, 6]
    predicted_boundaries = [2, 7]  # One exact match, one off by 1
    num_utterances = 10
    
    results = metrics_calc.evaluate_all(predicted_boundaries, gold_boundaries, num_utterances)
    
    print("Test evaluation:")
    for metric, value in sorted(results.items()):
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")