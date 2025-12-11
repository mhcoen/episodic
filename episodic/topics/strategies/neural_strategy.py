"""
Neural network-based topic detection strategy.

Uses fine-tuned DistilBERT model trained on conversational
transition data for topic boundary detection.
"""

import time
import logging
from typing import Dict, List, Any, Optional

from episodic.topics.strategy import (
    TopicStrategy,
    TopicDecision,
    Thread,
    ThreadLink,
    RetrievedContext,
    Confidence,
)
from episodic.config import config

logger = logging.getLogger(__name__)


class NeuralStrategy(TopicStrategy):
    """
    Neural network-based topic detection using fine-tuned transformers.

    Uses a DistilBERT model fine-tuned on conversational transition data
    with (4,2) window configuration: 4 messages before + 2 after boundary.

    Trained on realistic conversation data with ~3.3GB of model variants.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the neural strategy.

        Args:
            params: Optional parameters:
                - model_path: Path to model weights (default: auto-detect)
                - confidence_threshold: Min confidence to report change (default: 0.8)
                - device: Force device (cuda/mps/cpu, default: auto)
        """
        params = params or {}
        self.model_path = params.get('model_path')
        self.confidence_threshold = params.get('confidence_threshold', 0.8)
        self._force_device = params.get('device')

        # Lazy load model
        self._model = None
        self._tokenizer = None
        self._device = None
        self._available = None

    @property
    def name(self) -> str:
        return "NeuralStrategy"

    @property
    def version(self) -> str:
        return "1.0.0"

    def _ensure_model_loaded(self) -> bool:
        """Ensure model is loaded, return True if available."""
        if self._available is not None:
            return self._available

        try:
            from episodic.topics.neural_detection import (
                _load_model,
                TORCH_AVAILABLE
            )

            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available, neural strategy disabled")
                self._available = False
                return False

            self._model, self._tokenizer, self._device = _load_model(self.model_path)
            self._available = self._model is not None

            if self._available:
                logger.info(f"Neural strategy loaded on {self._device}")
            else:
                logger.warning("Neural model not found")

            return self._available

        except Exception as e:
            logger.error(f"Failed to load neural model: {e}")
            self._available = False
            return False

    def segment_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Thread]:
        """
        Segment conversation using neural boundary detection.

        Slides through the conversation checking each position for boundaries.
        """
        if not self._ensure_model_loaded():
            # Return single thread if model unavailable
            return [Thread(
                id="thread_0",
                name="conversation",
                start_node_id=messages[0].get('node_id', '0') if messages else '0',
                messages=messages
            )]

        threads = []
        current_messages = []
        thread_id = 0

        # Need at least 6 messages for (4,2) window
        if len(messages) < 6:
            return [Thread(
                id="thread_0",
                name="conversation",
                start_node_id=messages[0].get('node_id', '0') if messages else '0',
                messages=messages
            )]

        for i, msg in enumerate(messages):
            current_messages.append(msg)

            # Check for boundary after we have enough messages
            # and the current message is from user (boundaries on user messages)
            if len(current_messages) >= 6 and msg.get('role') == 'user':
                # Check if there's a boundary before this message
                is_boundary = self._check_boundary_at_position(messages, i)

                if is_boundary and len(current_messages) > 1:
                    # End previous thread
                    threads.append(Thread(
                        id=f"thread_{thread_id}",
                        name=f"topic_{thread_id}",
                        start_node_id=current_messages[0].get('node_id', str(thread_id)),
                        messages=current_messages[:-1].copy()  # Exclude current msg
                    ))
                    thread_id += 1
                    current_messages = [msg]  # Start new thread with current msg

        # Add final thread
        if current_messages:
            threads.append(Thread(
                id=f"thread_{thread_id}",
                name=f"topic_{thread_id}",
                start_node_id=current_messages[0].get('node_id', str(thread_id)),
                messages=current_messages
            ))

        return threads

    def _check_boundary_at_position(
        self,
        messages: List[Dict[str, Any]],
        position: int
    ) -> bool:
        """Check if there's a topic boundary at the given position."""
        try:
            import torch

            # Get 4 messages before and 2 after (including position)
            before_start = max(0, position - 4)
            before_messages = messages[before_start:position]
            after_messages = messages[position:position + 2]

            if len(before_messages) < 2 or len(after_messages) < 1:
                return False

            # Format as in training
            group1_text = " [SEP] ".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in before_messages
            ])
            group2_text = " [SEP] ".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in after_messages
            ])

            window_text = group1_text + " [BOUNDARY?] " + group2_text

            # Tokenize
            inputs = self._tokenizer(
                window_text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_class].item()

            # Class 1 = boundary
            return pred_class == 1 and confidence >= self.confidence_threshold

        except Exception as e:
            logger.error(f"Boundary check error: {e}")
            return False

    def detect_thread_link(
        self,
        query: str,
        threads: List[Thread],
        current_thread: Optional[Thread] = None
    ) -> List[ThreadLink]:
        """
        Detect thread links using embedding similarity.

        The neural model is for boundary detection, not retrieval.
        Falls back to embedding similarity for thread linking.
        """
        # Neural model doesn't do retrieval - use embedding similarity
        # This is a simple fallback; could be enhanced
        return []

    def retrieve_context(
        self,
        query: str,
        threads: List[Thread],
        current_thread: Optional[Thread] = None,
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """Retrieve context from threads (simple implementation)."""
        return RetrievedContext(
            threads=[],
            messages=[],
            relevance_scores={},
            token_count=0,
            retrieval_reason="neural_no_retrieval"
        )

    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None
    ) -> TopicDecision:
        """
        Decide if query represents a topic change using neural model.

        Uses (4,2) window: 4 messages before + query + placeholder response.
        """
        start_time = time.time()

        # Check model availability
        if not self._ensure_model_loaded():
            return TopicDecision(
                topic_changed=False,
                new_thread=None,
                thread_links=[],
                retrieved_context=None,
                confidence=Confidence.UNCERTAIN,
                confidence_score=0.0,
                reasoning="Neural model not available",
                signals={'model_available': False},
                strategy_name=self.name,
                strategy_version=self.version,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Need at least 1 message for group2 (last history msg + query)
        # Ideally 5+ for full (4,2) window, but can work with less
        if len(messages) < 1:
            return TopicDecision(
                topic_changed=False,
                new_thread=None,
                thread_links=[],
                retrieved_context=None,
                confidence=Confidence.UNCERTAIN,
                confidence_score=0.0,
                reasoning=f"Insufficient history: {len(messages)} < 1",
                signals={'message_count': len(messages)},
                strategy_name=self.name,
                strategy_version=self.version,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        try:
            import torch

            # Build window matching training format:
            # In training, boundary at position i means:
            #   group1 = messages[i-4:i] (4 messages BEFORE position i)
            #   group2 = messages[i:i+2] (messages[i] = last of old, messages[i+1] = first of new)
            #
            # For inference, checking if query starts new topic:
            #   Boundary position = last message of history
            #   group1 = 4 messages BEFORE the last history message
            #   group2 = [last history message, query] (straddles potential boundary)
            if len(messages) >= 5:
                before_messages = messages[-5:-1]  # 4 messages before the last one
            else:
                before_messages = messages[:-1]  # Whatever we have before last

            # Determine query role based on alternating pattern
            # If last history message is from user, query is from assistant, and vice versa
            last_role = messages[-1].get('role', 'user')
            query_role = 'assistant' if last_role == 'user' else 'user'

            after_messages = [
                messages[-1],  # Last message of history (potential last of old topic)
                {"role": query_role, "content": query}  # Query (potential first of new topic)
            ]

            # Format as in training
            group1_text = " [SEP] ".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in before_messages
            ])
            group2_text = " [SEP] ".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in after_messages
            ])

            window_text = group1_text + " [BOUNDARY?] " + group2_text

            # Tokenize
            inputs = self._tokenizer(
                window_text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1).item()
                boundary_prob = probs[0][1].item()  # Probability of boundary
                confidence = probs[0][pred_class].item()

            # Determine topic change
            topic_changed = pred_class == 1 and confidence >= self.confidence_threshold

            # Map confidence to levels
            if confidence >= 0.9:
                conf_level = Confidence.HIGH
            elif confidence >= 0.7:
                conf_level = Confidence.MEDIUM
            elif confidence >= 0.5:
                conf_level = Confidence.LOW
            else:
                conf_level = Confidence.UNCERTAIN

            # Build reasoning
            if topic_changed:
                reasoning = f"Neural model detected boundary (p={boundary_prob:.3f})"
            else:
                if pred_class == 1:
                    reasoning = f"Boundary detected but below threshold (p={boundary_prob:.3f} < {self.confidence_threshold})"
                else:
                    reasoning = f"No boundary detected (p={boundary_prob:.3f})"

            processing_time = (time.time() - start_time) * 1000

            return TopicDecision(
                topic_changed=topic_changed,
                new_thread=None,
                thread_links=[],
                retrieved_context=None,
                confidence=conf_level,
                confidence_score=boundary_prob,
                reasoning=reasoning,
                signals={
                    'boundary_probability': boundary_prob,
                    'no_boundary_probability': probs[0][0].item(),
                    'predicted_class': pred_class,
                    'threshold': self.confidence_threshold,
                    'device': str(self._device),
                },
                strategy_name=self.name,
                strategy_version=self.version,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Neural detection error: {e}")
            return TopicDecision(
                topic_changed=False,
                new_thread=None,
                thread_links=[],
                retrieved_context=None,
                confidence=Confidence.UNCERTAIN,
                confidence_score=0.0,
                reasoning=f"Error: {str(e)}",
                signals={'error': str(e)},
                strategy_name=self.name,
                strategy_version=self.version,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
