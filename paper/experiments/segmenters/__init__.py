"""
External dialogue segmentation baselines.

This module provides implementations of published dialogue segmentation methods
that can be evaluated under a common framework.

Methods included:
- TextTiling: Classic lexical cohesion baseline (Hearst, 1997)
- CSM (NSP): Coherence Scoring Model with Next Sentence Prediction (SIGDIAL 2021)
- InstructGPT: text-davinci-003 instruction-following baseline (SuperDialseg)
- Random: Random boundary placement baseline
- Even-N: Evenly spaced boundaries baseline
"""

from .base import Segmenter, SegmenterResult
from .texttiling import TextTilingSegmenter
from .csm_nsp import CSMSegmenter
from .baselines import RandomSegmenter, EvenSegmenter
from .instructgpt import InstructGPTSegmenter

__all__ = [
    "Segmenter",
    "SegmenterResult",
    "TextTilingSegmenter",
    "CSMSegmenter",
    "InstructGPTSegmenter",
    "RandomSegmenter",
    "EvenSegmenter",
]
