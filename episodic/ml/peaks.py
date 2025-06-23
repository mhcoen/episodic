"""
Peak detection strategies for conversational drift analysis.

This module provides various approaches for identifying significant
drift points in conversation sequences.
"""

from typing import List, Tuple
from abc import ABC, abstractmethod
import statistics


class PeakDetector:
    """Configurable peak detector supporting multiple strategies."""
    
    def __init__(self, strategy: str = "threshold", **kwargs):
        """
        Initialize peak detector.
        
        Args:
            strategy: Detection strategy ("threshold", "relative", "rolling_average", "statistical")
            **kwargs: Strategy-specific parameters
        """
        self.strategy = strategy
        self.kwargs = kwargs
        self._backend = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the appropriate backend based on strategy."""
        if self.strategy == "threshold":
            threshold = self.kwargs.get("threshold", 0.5)
            self._backend = ThresholdPeakDetector(threshold)
        elif self.strategy == "relative":
            min_prominence = self.kwargs.get("min_prominence", 0.1)
            self._backend = RelativePeakDetector(min_prominence)
        elif self.strategy == "rolling_average":
            window_size = self.kwargs.get("window_size", 3)
            threshold_multiplier = self.kwargs.get("threshold_multiplier", 1.5)
            self._backend = RollingAveragePeakDetector(window_size, threshold_multiplier)
        elif self.strategy == "statistical":
            std_multiplier = self.kwargs.get("std_multiplier", 2.0)
            self._backend = StatisticalPeakDetector(std_multiplier)
        else:
            raise ValueError(f"Unknown peak detection strategy: {self.strategy}")
    
    def find_peaks(self, drift_scores: List[float]) -> List[Tuple[int, float]]:
        """
        Find peaks in drift scores.
        
        Args:
            drift_scores: Sequential drift scores between nodes
            
        Returns:
            List of (node_index, drift_score) tuples for detected peaks
            Note: node_index is i+1 where i is the drift_scores index
        """
        return self._backend.find_peaks(drift_scores)


class PeakDetectionBackend(ABC):
    """Abstract base class for peak detection backends."""
    
    @abstractmethod
    def find_peaks(self, drift_scores: List[float]) -> List[Tuple[int, float]]:
        """Find peaks in drift scores."""
        pass


class ThresholdPeakDetector(PeakDetectionBackend):
    """Simple threshold-based peak detection."""
    
    def __init__(self, threshold: float):
        """
        Initialize threshold detector.
        
        Args:
            threshold: Minimum drift score to consider a peak
        """
        self.threshold = threshold
    
    def find_peaks(self, drift_scores: List[float]) -> List[Tuple[int, float]]:
        """Find all drift scores above threshold."""
        peaks = []
        for i, drift in enumerate(drift_scores):
            if drift >= self.threshold:
                peaks.append((i + 1, drift))
        return peaks


class RelativePeakDetector(PeakDetectionBackend):
    """Relative peak detection using local maxima."""
    
    def __init__(self, min_prominence: float = 0.1):
        """
        Initialize relative peak detector.
        
        Args:
            min_prominence: Minimum prominence for a peak to be considered significant
        """
        self.min_prominence = min_prominence
    
    def find_peaks(self, drift_scores: List[float]) -> List[Tuple[int, float]]:
        """Find local maxima with sufficient prominence."""
        if len(drift_scores) < 3:
            # Need at least 3 points to find local maxima
            return []
        
        peaks = []
        
        for i in range(1, len(drift_scores) - 1):
            current = drift_scores[i]
            left = drift_scores[i - 1]
            right = drift_scores[i + 1]
            
            # Check if it's a local maximum
            if current > left and current > right:
                # Calculate prominence (how much it stands out)
                prominence = min(current - left, current - right)
                
                if prominence >= self.min_prominence:
                    peaks.append((i + 1, current))
        
        # Also check endpoints if they're higher than their neighbor
        if len(drift_scores) >= 2:
            # Check first point
            if drift_scores[0] > drift_scores[1]:
                prominence = drift_scores[0] - drift_scores[1]
                if prominence >= self.min_prominence:
                    peaks.insert(0, (1, drift_scores[0]))
            
            # Check last point
            last_idx = len(drift_scores) - 1
            if drift_scores[last_idx] > drift_scores[last_idx - 1]:
                prominence = drift_scores[last_idx] - drift_scores[last_idx - 1]
                if prominence >= self.min_prominence:
                    peaks.append((last_idx + 1, drift_scores[last_idx]))
        
        return peaks


class RollingAveragePeakDetector(PeakDetectionBackend):
    """Peak detection using rolling average smoothing."""
    
    def __init__(self, window_size: int = 3, threshold_multiplier: float = 1.5):
        """
        Initialize rolling average detector.
        
        Args:
            window_size: Size of rolling window for smoothing
            threshold_multiplier: Multiplier for rolling average to set threshold
        """
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
    
    def find_peaks(self, drift_scores: List[float]) -> List[Tuple[int, float]]:
        """Find peaks that exceed rolling average threshold."""
        if len(drift_scores) < self.window_size:
            return []
        
        peaks = []
        
        for i in range(len(drift_scores)):
            # Calculate rolling average centered on current point
            start = max(0, i - self.window_size // 2)
            end = min(len(drift_scores), start + self.window_size)
            window = drift_scores[start:end]
            
            rolling_avg = sum(window) / len(window)
            threshold = rolling_avg * self.threshold_multiplier
            
            if drift_scores[i] >= threshold:
                peaks.append((i + 1, drift_scores[i]))
        
        return peaks


class StatisticalPeakDetector(PeakDetectionBackend):
    """Statistical outlier-based peak detection."""
    
    def __init__(self, std_multiplier: float = 2.0):
        """
        Initialize statistical detector.
        
        Args:
            std_multiplier: Number of standard deviations above mean to consider a peak
        """
        self.std_multiplier = std_multiplier
    
    def find_peaks(self, drift_scores: List[float]) -> List[Tuple[int, float]]:
        """Find statistical outliers as peaks."""
        if len(drift_scores) < 2:
            return []
        
        # Calculate statistical measures
        mean_drift = statistics.mean(drift_scores)
        
        # Use population standard deviation for small samples
        if len(drift_scores) >= 3:
            std_drift = statistics.stdev(drift_scores)
        else:
            # For very small samples, use simple range-based estimate
            std_drift = (max(drift_scores) - min(drift_scores)) / 2
        
        threshold = mean_drift + (self.std_multiplier * std_drift)
        
        peaks = []
        for i, drift in enumerate(drift_scores):
            if drift >= threshold:
                peaks.append((i + 1, drift))
        
        return peaks