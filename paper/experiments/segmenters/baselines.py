"""
Non-semantic baseline segmenters.

These baselines don't use any content information and serve as
lower bounds for comparison.
"""

import random
from typing import List, Dict, Any, Optional

from .base import Segmenter, SegmenterResult


class RandomSegmenter(Segmenter):
    """
    Random boundary placement baseline.

    Places boundaries randomly with a specified probability or target count.
    Useful for establishing chance-level performance.
    """

    def __init__(
        self,
        boundary_prob: Optional[float] = None,
        target_boundaries: Optional[int] = None,
        target_ratio: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize random segmenter.

        Exactly one of boundary_prob, target_boundaries, or target_ratio must be specified.

        Args:
            boundary_prob: Probability of placing a boundary at each position
            target_boundaries: Fixed number of boundaries to place
            target_ratio: Target ratio of boundaries to possible positions
            seed: Random seed for reproducibility
        """
        self.boundary_prob = boundary_prob
        self.target_boundaries = target_boundaries
        self.target_ratio = target_ratio
        self.seed = seed

        # Count how many parameters were specified
        specified = sum(x is not None for x in [boundary_prob, target_boundaries, target_ratio])
        if specified != 1:
            raise ValueError(
                "Exactly one of boundary_prob, target_boundaries, or target_ratio must be specified"
            )

        if seed is not None:
            random.seed(seed)

    @property
    def name(self) -> str:
        return "Random"

    @property
    def short_name(self) -> str:
        if self.boundary_prob:
            return f"random_p{self.boundary_prob:.2f}"
        elif self.target_boundaries:
            return f"random_n{self.target_boundaries}"
        else:
            return f"random_r{self.target_ratio:.2f}"

    @property
    def description(self) -> str:
        return "Random boundary placement baseline"

    def predict_boundaries(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> SegmenterResult:
        """Place boundaries randomly."""
        num_messages = len(messages)
        if num_messages <= 1:
            return SegmenterResult(boundaries=[])

        # Possible boundary positions: [1, T-1]
        possible_positions = list(range(1, num_messages))

        boundaries = []

        if self.boundary_prob is not None:
            # Bernoulli at each position
            for pos in possible_positions:
                if random.random() < self.boundary_prob:
                    boundaries.append(pos)

        elif self.target_boundaries is not None:
            # Fixed number of boundaries
            k = min(self.target_boundaries, len(possible_positions))
            boundaries = sorted(random.sample(possible_positions, k))

        elif self.target_ratio is not None:
            # Target ratio of positions
            k = int(len(possible_positions) * self.target_ratio)
            k = min(k, len(possible_positions))
            if k > 0:
                boundaries = sorted(random.sample(possible_positions, k))

        return SegmenterResult(
            boundaries=boundaries,
            metadata={"method": "random"}
        )


class EvenSegmenter(Segmenter):
    """
    Evenly spaced boundaries baseline.

    Places boundaries at regular intervals to create segments of roughly
    equal size. Useful for understanding the impact of uniform segmentation.
    """

    def __init__(
        self,
        num_segments: Optional[int] = None,
        segment_size: Optional[int] = None,
        match_gold: bool = False,
    ):
        """
        Initialize even segmenter.

        Args:
            num_segments: Fixed number of segments to create
            segment_size: Target size for each segment
            match_gold: If True, use the gold boundary count (requires passing gold)
        """
        self.num_segments = num_segments
        self.segment_size = segment_size
        self.match_gold = match_gold

        if not match_gold and num_segments is None and segment_size is None:
            raise ValueError(
                "Must specify num_segments, segment_size, or match_gold=True"
            )

    @property
    def name(self) -> str:
        return "Even"

    @property
    def short_name(self) -> str:
        if self.match_gold:
            return "even_match"
        elif self.num_segments:
            return f"even_k{self.num_segments}"
        else:
            return f"even_s{self.segment_size}"

    @property
    def description(self) -> str:
        return "Evenly spaced boundaries baseline"

    def predict_boundaries(
        self,
        messages: List[Dict[str, str]],
        num_gold_boundaries: Optional[int] = None,
        **kwargs
    ) -> SegmenterResult:
        """Place boundaries at even intervals."""
        num_messages = len(messages)
        if num_messages <= 1:
            return SegmenterResult(boundaries=[])

        # Determine number of boundaries to place
        if self.match_gold and num_gold_boundaries is not None:
            num_boundaries = num_gold_boundaries
        elif self.num_segments is not None:
            num_boundaries = self.num_segments - 1
        elif self.segment_size is not None:
            num_boundaries = max(0, (num_messages - 1) // self.segment_size)
        else:
            # Default: no boundaries
            num_boundaries = 0

        if num_boundaries <= 0:
            return SegmenterResult(boundaries=[])

        # Place boundaries evenly
        # With K boundaries, we have K+1 segments
        # Ideal segment size = num_messages / (num_boundaries + 1)
        segment_size = num_messages / (num_boundaries + 1)

        boundaries = []
        for i in range(1, num_boundaries + 1):
            # Place boundary at end of segment i
            pos = int(i * segment_size)
            if 1 <= pos < num_messages:
                boundaries.append(pos)

        # Remove duplicates and sort
        boundaries = sorted(set(boundaries))

        return SegmenterResult(
            boundaries=boundaries,
            metadata={
                "method": "even",
                "target_segment_size": segment_size,
            }
        )


class OracleSegmenter(Segmenter):
    """
    Oracle segmenter that returns gold boundaries.

    Useful for debugging and establishing upper bounds.
    """

    @property
    def name(self) -> str:
        return "Oracle"

    @property
    def description(self) -> str:
        return "Oracle returning gold boundaries"

    def predict_boundaries(
        self,
        messages: List[Dict[str, str]],
        gold_boundaries: List[int] = None,
        **kwargs
    ) -> SegmenterResult:
        """Return gold boundaries."""
        if gold_boundaries is None:
            return SegmenterResult(boundaries=[])
        return SegmenterResult(
            boundaries=list(gold_boundaries),
            metadata={"method": "oracle"}
        )
