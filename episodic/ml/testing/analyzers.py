"""
Analysis strategies for drift testing results.

This module provides various approaches for analyzing and reporting
drift detection results from conversation sequences.
"""

from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from ..drift import ConversationalDrift
import statistics


class AnalysisStrategy:
    """Configurable analysis strategy supporting multiple approaches."""
    
    def __init__(self, strategy: str = "basic_stats", **kwargs):
        """
        Initialize analysis strategy.
        
        Args:
            strategy: Analysis approach ("basic_stats", "detailed_transitions", "peak_analysis", "comparative")
            **kwargs: Strategy-specific parameters
        """
        self.strategy = strategy
        self.kwargs = kwargs
        self._backend = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the appropriate backend based on strategy."""
        if self.strategy == "basic_stats":
            self._backend = BasicStatsAnalyzer()
        elif self.strategy == "detailed_transitions":
            content_limit = self.kwargs.get("content_limit", 100)
            self._backend = DetailedTransitionsAnalyzer(content_limit)
        elif self.strategy == "peak_analysis":
            highlight_threshold = self.kwargs.get("highlight_threshold", 0.7)
            self._backend = PeakAnalysisAnalyzer(highlight_threshold)
        elif self.strategy == "comparative":
            baseline_drift = self.kwargs.get("baseline_drift", 0.5)
            self._backend = ComparativeAnalyzer(baseline_drift)
        else:
            raise ValueError(f"Unknown analysis strategy: {self.strategy}")
    
    def analyze_drift_results(
        self,
        drift_calculator: ConversationalDrift,
        conversation_sequences: List[List[Dict[str, Any]]],
        text_field: str = "content"
    ) -> Dict[str, Any]:
        """
        Analyze drift results from conversation sequences.
        
        Args:
            drift_calculator: Configured drift calculator
            conversation_sequences: List of conversation sequences to analyze
            text_field: Field containing text content
            
        Returns:
            Analysis results dictionary
        """
        return self._backend.analyze_drift_results(drift_calculator, conversation_sequences, text_field)


class AnalysisBackend(ABC):
    """Abstract base class for analysis backends."""
    
    @abstractmethod
    def analyze_drift_results(
        self,
        drift_calculator: ConversationalDrift,
        conversation_sequences: List[List[Dict[str, Any]]],
        text_field: str = "content"
    ) -> Dict[str, Any]:
        """Analyze drift results from conversation sequences."""
        pass


class BasicStatsAnalyzer(AnalysisBackend):
    """Basic statistical analysis of drift patterns."""
    
    def analyze_drift_results(
        self,
        drift_calculator: ConversationalDrift,
        conversation_sequences: List[List[Dict[str, Any]]],
        text_field: str = "content"
    ) -> Dict[str, Any]:
        """Generate basic statistical summary."""
        if not conversation_sequences:
            return {"error": "No conversation sequences to analyze"}
        
        all_drift_scores = []
        sequence_summaries = []
        total_nodes = 0
        
        try:
            for i, sequence in enumerate(conversation_sequences):
                if len(sequence) < 2:
                    continue
                
                drift_scores = drift_calculator.calculate_drift_sequence(sequence, text_field)
                drift_peaks = drift_calculator.find_drift_peaks(sequence, text_field)
                
                all_drift_scores.extend(drift_scores)
                total_nodes += len(sequence)
                
                sequence_summaries.append({
                    "sequence_id": i,
                    "node_count": len(sequence),
                    "avg_drift": statistics.mean(drift_scores),
                    "max_drift": max(drift_scores),
                    "min_drift": min(drift_scores),
                    "peak_count": len(drift_peaks)
                })
            
            if not all_drift_scores:
                return {"error": "No drift scores computed"}
            
            # Overall statistics
            overall_stats = {
                "total_sequences": len(conversation_sequences),
                "total_nodes": total_nodes,
                "total_transitions": len(all_drift_scores),
                "overall_avg_drift": statistics.mean(all_drift_scores),
                "overall_max_drift": max(all_drift_scores),
                "overall_min_drift": min(all_drift_scores),
                "drift_std_dev": statistics.stdev(all_drift_scores) if len(all_drift_scores) > 1 else 0.0
            }
            
            return {
                "analysis_type": "basic_stats",
                "overall_stats": overall_stats,
                "sequence_summaries": sequence_summaries,
                "embedding_cache_size": drift_calculator.get_cache_size()
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}


class DetailedTransitionsAnalyzer(AnalysisBackend):
    """Detailed analysis of individual drift transitions."""
    
    def __init__(self, content_limit: int = 100):
        """
        Initialize detailed transitions analyzer.
        
        Args:
            content_limit: Maximum characters to show from each message
        """
        self.content_limit = content_limit
    
    def analyze_drift_results(
        self,
        drift_calculator: ConversationalDrift,
        conversation_sequences: List[List[Dict[str, Any]]],
        text_field: str = "content"
    ) -> Dict[str, Any]:
        """Generate detailed transition analysis."""
        if not conversation_sequences:
            return {"error": "No conversation sequences to analyze"}
        
        all_transitions = []
        
        try:
            for seq_id, sequence in enumerate(conversation_sequences):
                if len(sequence) < 2:
                    continue
                
                drift_scores = drift_calculator.calculate_drift_sequence(sequence, text_field)
                
                for i, drift_score in enumerate(drift_scores):
                    from_node = sequence[i]
                    to_node = sequence[i + 1]
                    
                    transition = {
                        "sequence_id": seq_id,
                        "transition_id": i,
                        "drift_score": drift_score,
                        "from_node_id": from_node.get("short_id", "unknown"),
                        "to_node_id": to_node.get("short_id", "unknown"),
                        "from_content": from_node.get(text_field, "")[:self.content_limit] + "...",
                        "to_content": to_node.get(text_field, "")[:self.content_limit] + "...",
                        "from_role": from_node.get("role", "unknown"),
                        "to_role": to_node.get("role", "unknown")
                    }
                    all_transitions.append(transition)
            
            # Sort transitions by drift score (highest first)
            all_transitions.sort(key=lambda x: x["drift_score"], reverse=True)
            
            return {
                "analysis_type": "detailed_transitions",
                "transition_count": len(all_transitions),
                "transitions": all_transitions[:20],  # Top 20 highest drift transitions
                "embedding_cache_size": drift_calculator.get_cache_size()
            }
            
        except Exception as e:
            return {"error": f"Detailed analysis failed: {e}"}


class PeakAnalysisAnalyzer(AnalysisBackend):
    """Analysis focused on drift peaks and patterns."""
    
    def __init__(self, highlight_threshold: float = 0.7):
        """
        Initialize peak analysis analyzer.
        
        Args:
            highlight_threshold: Drift threshold for highlighting significant peaks
        """
        self.highlight_threshold = highlight_threshold
    
    def analyze_drift_results(
        self,
        drift_calculator: ConversationalDrift,
        conversation_sequences: List[List[Dict[str, Any]]],
        text_field: str = "content"
    ) -> Dict[str, Any]:
        """Generate peak-focused analysis."""
        if not conversation_sequences:
            return {"error": "No conversation sequences to analyze"}
        
        all_peaks = []
        peak_statistics = {"total_peaks": 0, "significant_peaks": 0, "peak_density": 0.0}
        
        try:
            total_transitions = 0
            
            for seq_id, sequence in enumerate(conversation_sequences):
                if len(sequence) < 2:
                    continue
                
                drift_peaks = drift_calculator.find_drift_peaks(sequence, text_field)
                drift_scores = drift_calculator.calculate_drift_sequence(sequence, text_field)
                
                total_transitions += len(drift_scores)
                peak_statistics["total_peaks"] += len(drift_peaks)
                
                for peak_idx, peak_score in drift_peaks:
                    is_significant = peak_score >= self.highlight_threshold
                    if is_significant:
                        peak_statistics["significant_peaks"] += 1
                    
                    # Get the actual nodes involved in this peak
                    from_node = sequence[peak_idx - 1] if peak_idx > 0 else None
                    to_node = sequence[peak_idx] if peak_idx < len(sequence) else None
                    
                    peak_info = {
                        "sequence_id": seq_id,
                        "peak_node_id": to_node.get("short_id", "unknown") if to_node else "unknown",
                        "drift_score": peak_score,
                        "is_significant": is_significant,
                        "transition_from": from_node.get("short_id", "unknown") if from_node else "start",
                        "peak_content": to_node.get(text_field, "")[:100] + "..." if to_node else ""
                    }
                    all_peaks.append(peak_info)
            
            # Calculate peak density
            peak_statistics["peak_density"] = (
                peak_statistics["total_peaks"] / total_transitions if total_transitions > 0 else 0.0
            )
            
            # Sort peaks by significance and drift score
            all_peaks.sort(key=lambda x: (x["is_significant"], x["drift_score"]), reverse=True)
            
            return {
                "analysis_type": "peak_analysis",
                "peak_statistics": peak_statistics,
                "highlight_threshold": self.highlight_threshold,
                "peaks": all_peaks[:15],  # Top 15 most significant peaks
                "embedding_cache_size": drift_calculator.get_cache_size()
            }
            
        except Exception as e:
            return {"error": f"Peak analysis failed: {e}"}


class ComparativeAnalyzer(AnalysisBackend):
    """Comparative analysis against baseline drift expectations."""
    
    def __init__(self, baseline_drift: float = 0.5):
        """
        Initialize comparative analyzer.
        
        Args:
            baseline_drift: Expected baseline drift for comparison
        """
        self.baseline_drift = baseline_drift
    
    def analyze_drift_results(
        self,
        drift_calculator: ConversationalDrift,
        conversation_sequences: List[List[Dict[str, Any]]],
        text_field: str = "content"
    ) -> Dict[str, Any]:
        """Generate comparative analysis against baseline."""
        if not conversation_sequences:
            return {"error": "No conversation sequences to analyze"}
        
        try:
            all_drift_scores = []
            
            for sequence in conversation_sequences:
                if len(sequence) >= 2:
                    drift_scores = drift_calculator.calculate_drift_sequence(sequence, text_field)
                    all_drift_scores.extend(drift_scores)
            
            if not all_drift_scores:
                return {"error": "No drift scores computed"}
            
            # Comparative statistics
            above_baseline = [d for d in all_drift_scores if d > self.baseline_drift]
            below_baseline = [d for d in all_drift_scores if d <= self.baseline_drift]
            
            comparison_stats = {
                "baseline_drift": self.baseline_drift,
                "total_transitions": len(all_drift_scores),
                "above_baseline_count": len(above_baseline),
                "below_baseline_count": len(below_baseline),
                "above_baseline_pct": len(above_baseline) / len(all_drift_scores) * 100,
                "avg_above_baseline": statistics.mean(above_baseline) if above_baseline else 0.0,
                "avg_below_baseline": statistics.mean(below_baseline) if below_baseline else 0.0,
                "overall_avg": statistics.mean(all_drift_scores),
                "deviation_from_baseline": statistics.mean(all_drift_scores) - self.baseline_drift
            }
            
            # Classification of drift patterns
            classification = "unknown"
            if comparison_stats["above_baseline_pct"] > 70:
                classification = "high_drift_conversations"
            elif comparison_stats["above_baseline_pct"] < 30:
                classification = "low_drift_conversations"
            else:
                classification = "mixed_drift_conversations"
            
            return {
                "analysis_type": "comparative",
                "comparison_stats": comparison_stats,
                "classification": classification,
                "embedding_cache_size": drift_calculator.get_cache_size()
            }
            
        except Exception as e:
            return {"error": f"Comparative analysis failed: {e}"}