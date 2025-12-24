"""
TextTiling-based dialogue segmenter.

Based on the Neural TextTiling approach from:
"CSM: A Coherence Scoring Model for Dialogue Topic Segmentation" (SIGDIAL 2021)

Uses sentence embeddings to compute similarity between adjacent utterances,
then applies TextTiling-style depth scoring to find topic boundaries.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .base import Segmenter, SegmenterResult, messages_to_utterances


class TextTilingSegmenter(Segmenter):
    """
    TextTiling segmenter using sentence embeddings.

    Computes similarity between adjacent utterances using a sentence encoder,
    converts to depth scores, and places boundaries where depth exceeds a threshold.

    The threshold is computed as: mean + alpha * std
    where alpha is a tunable hyperparameter (can be negative).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        alpha: float = 0.0,
        device: Optional[str] = None,
    ):
        """
        Initialize TextTiling segmenter.

        Args:
            model_name: Sentence transformer model to use
            alpha: Threshold multiplier (threshold = mean + alpha * std)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.alpha = alpha
        self._device = device
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "TextTiling"

    @property
    def short_name(self) -> str:
        return f"texttiling_a{self.alpha:.1f}"

    @property
    def description(self) -> str:
        return f"TextTiling with {self.model_name}, alpha={self.alpha}"

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                if self._device is None:
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"

                self._model = SentenceTransformer(self.model_name, device=self._device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for TextTiling. "
                    "Install with: pip install sentence-transformers"
                )

    def _compute_similarity_scores(self, utterances: List[str]) -> np.ndarray:
        """
        Compute cosine similarity between adjacent utterances.

        Args:
            utterances: List of utterance strings

        Returns:
            Array of similarity scores (length = len(utterances) - 1)
        """
        self._load_model()

        if len(utterances) < 2:
            return np.array([])

        # Encode all utterances
        embeddings = self._model.encode(utterances, convert_to_numpy=True)

        # Compute cosine similarity between adjacent pairs
        similarities = []
        for i in range(len(embeddings) - 1):
            e1 = embeddings[i]
            e2 = embeddings[i + 1]
            # Cosine similarity
            sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
            similarities.append(sim)

        return np.array(similarities)

    def _compute_depth_scores(self, similarity_scores: np.ndarray) -> np.ndarray:
        """
        Convert similarity scores to depth scores.

        Depth at position i measures how much of a "valley" exists at i,
        computed as the average height of peaks to the left and right
        minus the current value.

        Args:
            similarity_scores: Array of similarity scores

        Returns:
            Array of depth scores (same length as input)
        """
        if len(similarity_scores) == 0:
            return np.array([])

        num_scores = len(similarity_scores)
        depth_scores = []

        for i in range(num_scores):
            # Initialize left and right flags with current score
            left_flag = similarity_scores[i]
            right_flag = similarity_scores[i]

            # Search to the left for peak
            for left_idx in range(i - 1, -1, -1):
                if similarity_scores[left_idx] >= left_flag:
                    left_flag = similarity_scores[left_idx]
                else:
                    break

            # Search to the right for peak
            for right_idx in range(i + 1, num_scores):
                if similarity_scores[right_idx] >= right_flag:
                    right_flag = similarity_scores[right_idx]
                else:
                    break

            # Depth score: average height of surrounding peaks minus current
            depth = 0.5 * (left_flag + right_flag - 2 * similarity_scores[i])
            depth_scores.append(depth)

        return np.array(depth_scores)

    def predict_boundaries(
        self,
        messages: List[Dict[str, str]],
        alpha: Optional[float] = None,
        **kwargs
    ) -> SegmenterResult:
        """
        Predict topic boundaries using TextTiling.

        Args:
            messages: List of message dicts with 'role' and 'content'
            alpha: Override alpha parameter for this prediction

        Returns:
            SegmenterResult with boundary indices
        """
        utterances = messages_to_utterances(messages)
        num_messages = len(utterances)

        if num_messages <= 2:
            return SegmenterResult(boundaries=[], metadata={"method": "texttiling"})

        # Compute similarity and depth scores
        similarity_scores = self._compute_similarity_scores(utterances)
        depth_scores = self._compute_depth_scores(similarity_scores)

        if len(depth_scores) == 0:
            return SegmenterResult(boundaries=[], metadata={"method": "texttiling"})

        # Compute threshold
        use_alpha = alpha if alpha is not None else self.alpha
        threshold = depth_scores.mean() + use_alpha * depth_scores.std()

        # Find boundaries where depth exceeds threshold
        # depth_scores[i] corresponds to position between utterance i and i+1
        # So if depth_scores[i] > threshold, boundary is at i+1 (canonical format)
        boundaries = []
        scores_dict = {}
        for i, depth in enumerate(depth_scores):
            if depth > threshold:
                # Boundary position in canonical format
                boundary_pos = i + 1
                if 1 <= boundary_pos < num_messages:
                    boundaries.append(boundary_pos)
                    scores_dict[boundary_pos] = float(depth)

        return SegmenterResult(
            boundaries=boundaries,
            scores=scores_dict,
            metadata={
                "method": "texttiling",
                "alpha": use_alpha,
                "threshold": float(threshold),
                "mean_depth": float(depth_scores.mean()),
                "std_depth": float(depth_scores.std()),
            }
        )

    def find_best_alpha(
        self,
        dialogues: List[List[Dict[str, str]]],
        gold_boundaries: List[List[int]],
        alpha_range: Tuple[float, float] = (-2.0, 2.0),
        alpha_step: float = 0.1,
    ) -> Tuple[float, float]:
        """
        Find the best alpha parameter on a dev set.

        Args:
            dialogues: List of dialogues (each a list of messages)
            gold_boundaries: List of gold boundary lists
            alpha_range: (min_alpha, max_alpha) to search
            alpha_step: Step size for alpha search

        Returns:
            (best_alpha, best_score) where score is negative Pk (higher is better)
        """
        best_alpha = 0.0
        best_score = float("-inf")

        alphas = np.arange(alpha_range[0], alpha_range[1], alpha_step)

        for alpha in alphas:
            total_score = 0.0
            for messages, gold in zip(dialogues, gold_boundaries):
                result = self.predict_boundaries(messages, alpha=alpha)
                # Simple evaluation: count exact matches
                pred_set = set(result.boundaries)
                gold_set = set(gold)
                # F1-like score
                if pred_set or gold_set:
                    tp = len(pred_set & gold_set)
                    prec = tp / len(pred_set) if pred_set else 0
                    rec = tp / len(gold_set) if gold_set else 0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    total_score += f1
                else:
                    total_score += 1.0  # Both empty = perfect

            avg_score = total_score / len(dialogues)
            if avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha

        return best_alpha, best_score
