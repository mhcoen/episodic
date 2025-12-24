"""
CSM (Coherence Scoring Model) segmenter using Next Sentence Prediction.

Based on:
"CSM: A Coherence Scoring Model for Dialogue Topic Segmentation" (SIGDIAL 2021)

Uses BERT's Next Sentence Prediction task to score coherence between adjacent
utterances, then applies TextTiling-style depth scoring to find boundaries.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .base import Segmenter, SegmenterResult, messages_to_utterances


class CSMSegmenter(Segmenter):
    """
    CSM segmenter using BERT Next Sentence Prediction.

    For each pair of adjacent utterances, computes the NSP probability
    (probability that the second sentence follows the first). Low NSP
    probability suggests a topic boundary.

    Uses the same TextTiling-style depth scoring as TextTilingSegmenter
    but with NSP-based coherence scores instead of embedding similarity.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        alpha: float = 0.0,
        device: Optional[str] = None,
        max_length: int = 128,
    ):
        """
        Initialize CSM segmenter.

        Args:
            model_name: BERT model to use for NSP
            alpha: Threshold multiplier (threshold = mean + alpha * std)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.alpha = alpha
        self._device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "CSM (NSP)"

    @property
    def short_name(self) -> str:
        return f"csm_nsp_a{self.alpha:.1f}"

    @property
    def description(self) -> str:
        return f"CSM with {self.model_name}, alpha={self.alpha}"

    def _load_model(self):
        """Lazy load the BERT NSP model."""
        if self._model is None:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForNextSentencePrediction

                if self._device is None:
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForNextSentencePrediction.from_pretrained(
                    self.model_name
                ).to(self._device)
                self._model.eval()

            except ImportError:
                raise ImportError(
                    "transformers and torch are required for CSM. "
                    "Install with: pip install transformers torch"
                )

    def _compute_nsp_scores(self, utterances: List[str]) -> np.ndarray:
        """
        Compute NSP probabilities between adjacent utterances.

        Args:
            utterances: List of utterance strings

        Returns:
            Array of NSP scores (length = len(utterances) - 1)
            Higher score = more likely to be continuation (not a boundary)
        """
        import torch

        self._load_model()

        if len(utterances) < 2:
            return np.array([])

        scores = []
        with torch.no_grad():
            for i in range(len(utterances) - 1):
                sent1 = utterances[i]
                sent2 = utterances[i + 1]

                # Tokenize as sentence pair
                tokenized = self._tokenizer(
                    sent1,
                    sent2,
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                # Move to device
                tokenized = {k: v.to(self._device) for k, v in tokenized.items()}

                # Get NSP logits
                outputs = self._model(**tokenized)
                logits = outputs.logits

                # Softmax to get probabilities
                # Index 0 = IsNextSentence (continuation)
                # Index 1 = NotNextSentence (boundary)
                probs = torch.softmax(logits, dim=1)
                continuation_prob = probs[0, 0].item()

                scores.append(continuation_prob)

        return np.array(scores)

    def _compute_depth_scores(self, nsp_scores: np.ndarray) -> np.ndarray:
        """
        Convert NSP scores to depth scores.

        Uses the same TextTiling depth calculation:
        - Find peaks to left and right
        - Depth = average peak height - current value

        Args:
            nsp_scores: Array of NSP continuation probabilities

        Returns:
            Array of depth scores (same length as input)
        """
        if len(nsp_scores) == 0:
            return np.array([])

        num_scores = len(nsp_scores)
        depth_scores = []

        for i in range(num_scores):
            left_flag = nsp_scores[i]
            right_flag = nsp_scores[i]

            # Search left for peak
            for left_idx in range(i - 1, -1, -1):
                if nsp_scores[left_idx] >= left_flag:
                    left_flag = nsp_scores[left_idx]
                else:
                    break

            # Search right for peak
            for right_idx in range(i + 1, num_scores):
                if nsp_scores[right_idx] >= right_flag:
                    right_flag = nsp_scores[right_idx]
                else:
                    break

            # Depth score
            depth = 0.5 * (left_flag + right_flag - 2 * nsp_scores[i])
            depth_scores.append(depth)

        return np.array(depth_scores)

    def predict_boundaries(
        self,
        messages: List[Dict[str, str]],
        alpha: Optional[float] = None,
        **kwargs
    ) -> SegmenterResult:
        """
        Predict topic boundaries using CSM with NSP.

        Args:
            messages: List of message dicts with 'role' and 'content'
            alpha: Override alpha parameter for this prediction

        Returns:
            SegmenterResult with boundary indices
        """
        utterances = messages_to_utterances(messages)
        num_messages = len(utterances)

        if num_messages <= 2:
            return SegmenterResult(boundaries=[], metadata={"method": "csm_nsp"})

        # Compute NSP scores and depth
        nsp_scores = self._compute_nsp_scores(utterances)
        depth_scores = self._compute_depth_scores(nsp_scores)

        if len(depth_scores) == 0:
            return SegmenterResult(boundaries=[], metadata={"method": "csm_nsp"})

        # Compute threshold
        use_alpha = alpha if alpha is not None else self.alpha
        threshold = depth_scores.mean() + use_alpha * depth_scores.std()

        # Find boundaries
        boundaries = []
        scores_dict = {}
        for i, depth in enumerate(depth_scores):
            if depth > threshold:
                boundary_pos = i + 1
                if 1 <= boundary_pos < num_messages:
                    boundaries.append(boundary_pos)
                    scores_dict[boundary_pos] = float(depth)

        return SegmenterResult(
            boundaries=boundaries,
            scores=scores_dict,
            metadata={
                "method": "csm_nsp",
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

        Note: This precomputes all NSP/depth scores first for efficiency.

        Args:
            dialogues: List of dialogues (each a list of messages)
            gold_boundaries: List of gold boundary lists
            alpha_range: (min_alpha, max_alpha) to search
            alpha_step: Step size for alpha search

        Returns:
            (best_alpha, best_score)
        """
        # Precompute depth scores for all dialogues
        all_depth_scores = []
        for messages in dialogues:
            utterances = messages_to_utterances(messages)
            if len(utterances) > 2:
                nsp_scores = self._compute_nsp_scores(utterances)
                depth_scores = self._compute_depth_scores(nsp_scores)
            else:
                depth_scores = np.array([])
            all_depth_scores.append(depth_scores)

        best_alpha = 0.0
        best_score = float("-inf")

        alphas = np.arange(alpha_range[0], alpha_range[1], alpha_step)

        for alpha in alphas:
            total_score = 0.0
            for i, (messages, gold) in enumerate(zip(dialogues, gold_boundaries)):
                depth_scores = all_depth_scores[i]
                num_messages = len(messages)

                if len(depth_scores) == 0:
                    if not gold:
                        total_score += 1.0
                    continue

                threshold = depth_scores.mean() + alpha * depth_scores.std()
                boundaries = []
                for j, depth in enumerate(depth_scores):
                    if depth > threshold:
                        pos = j + 1
                        if 1 <= pos < num_messages:
                            boundaries.append(pos)

                # F1 score
                pred_set = set(boundaries)
                gold_set = set(gold)
                if pred_set or gold_set:
                    tp = len(pred_set & gold_set)
                    prec = tp / len(pred_set) if pred_set else 0
                    rec = tp / len(gold_set) if gold_set else 0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    total_score += f1
                else:
                    total_score += 1.0

            avg_score = total_score / len(dialogues)
            if avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha

        return best_alpha, best_score
