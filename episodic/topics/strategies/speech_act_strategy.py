"""
Speech-act based topic detection strategy.

Uses dialogue act (speech act) patterns to detect topic boundaries.
Certain speech acts strongly signal topic transitions:
- Topic management acts ("anyway", "by the way", "new question")
- Question-after-answer patterns (Q→A→Q often signals new topic)
- Meta-comments about conversation structure

This can be combined with semantic features for multi-view detection.
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import re
import uuid

from episodic.topics.strategy import (
    TopicStrategy,
    Thread,
    ThreadLink,
    RetrievedContext,
    TopicDecision,
    Confidence
)


# Speech act categories that often signal topic boundaries
BOUNDARY_SPEECH_ACTS = {
    # Direct topic management
    'topic_switch', 'topic_change', 'new_topic',
    # Questions after resolved exchanges
    'query_condition', 'query_fact', 'query_opinion',
    # Opening/greeting after content (restart)
    'greeting', 'opening',
}

# Speech acts that typically continue topics
CONTINUATION_SPEECH_ACTS = {
    'answer', 'inform', 'confirm', 'deny',
    'clarify', 'elaborate', 'acknowledge',
}

# Explicit transition phrases (regex patterns)
TRANSITION_PATTERNS = [
    r'\b(by the way|btw)\b',
    r'\b(anyway|anyhow)\b',
    r'\b(on (a |an )?other (note|topic|subject))\b',
    r'\b(changing (topic|subject|gears))\b',
    r'\b(speaking of which|that reminds me)\b',
    r'\b(new question|different question|another question)\b',
    r'\b(moving on|let\'?s move on)\b',
    r'\b(back to|getting back to|returning to)\b',
    r'^(so|ok|okay|alright|well)\s*[,.]?\s*(i |can |what |how |why |when |where )',
]


class SpeechActStrategy(TopicStrategy):
    """
    Speech-act based topic boundary detection.

    Uses dialogue act labels (if available) or pattern-based detection
    to identify functional signals of topic transitions.

    Works best when combined with semantic strategies in an ensemble.
    """

    def __init__(self, strategy_config: Dict[str, Any] = None):
        """
        Initialize speech-act strategy.

        Args:
            strategy_config: Optional parameters:
                - use_patterns: Use regex patterns for detection (default: True)
                - use_act_labels: Use dialogue act labels if available (default: True)
                - transition_boost: Confidence boost for explicit transitions (default: 0.3)
                - qa_pattern_weight: Weight for Q→A→Q pattern (default: 0.2)
        """
        super().__init__(strategy_config)
        strategy_config = strategy_config or {}

        self.name = "SpeechActStrategy"
        self.version = "1.0.0"

        self.use_patterns = strategy_config.get('use_patterns', True)
        self.use_act_labels = strategy_config.get('use_act_labels', True)
        self.transition_boost = strategy_config.get('transition_boost', 0.3)
        self.qa_pattern_weight = strategy_config.get('qa_pattern_weight', 0.2)

        # Compile patterns
        self._patterns = [re.compile(p, re.IGNORECASE) for p in TRANSITION_PATTERNS]

        # Track conversation structure
        self._recent_acts: List[str] = []

    def _detect_transition_phrase(self, text: str) -> Optional[str]:
        """Check for explicit transition phrases."""
        text_lower = text.lower()
        for pattern in self._patterns:
            match = pattern.search(text_lower)
            if match:
                return match.group(0)
        return None

    def _classify_speech_act(self, message: Dict[str, Any]) -> str:
        """
        Classify the speech act of a message.

        Uses 'da' field if available, otherwise heuristics.
        """
        # Check for provided dialogue act label
        if self.use_act_labels and 'da' in message:
            return message['da']

        content = message.get('content', '').strip().lower()
        role = message.get('role', 'user')

        # Simple heuristics when no label available
        if not content:
            return 'unknown'

        # Questions
        if content.endswith('?') or content.startswith(('what', 'how', 'why', 'when', 'where', 'who', 'can', 'could', 'would', 'is', 'are', 'do', 'does')):
            return 'query_fact'

        # Greetings
        if any(g in content for g in ['hello', 'hi ', 'hey ', 'good morning', 'good afternoon']):
            return 'greeting'

        # Thanks/closing
        if any(t in content for t in ['thank', 'thanks', 'goodbye', 'bye']):
            return 'closing'

        # Default based on role
        return 'inform' if role == 'assistant' else 'request'

    def _detect_qa_pattern(self, messages: List[Dict[str, Any]], query: str) -> bool:
        """
        Detect Q→A→Q pattern (question, answer, new question).

        This often signals a topic boundary after resolution.
        """
        if len(messages) < 2:
            return False

        # Check if query is a question
        query_lower = query.lower().strip()
        is_question = (
            query_lower.endswith('?') or
            query_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who', 'can', 'could', 'would'))
        )

        if not is_question:
            return False

        # Check if last message was an answer (assistant)
        last_msg = messages[-1]
        if last_msg.get('role') != 'assistant':
            return False

        # Check if message before that was a question (user)
        if len(messages) >= 2:
            prev_user_msg = messages[-2]
            if prev_user_msg.get('role') == 'user':
                prev_content = prev_user_msg.get('content', '').lower().strip()
                was_question = prev_content.endswith('?')
                if was_question:
                    return True

        return False

    def get_decision(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        current_thread: Optional[Thread] = None
    ) -> TopicDecision:
        """
        Make topic decision based on speech acts.
        """
        import time
        start_time = time.time()

        signals = {
            'transition_phrase': 0.0,
            'qa_pattern': 0.0,
            'boundary_act': 0.0,
            'phrase_detected': '',
        }

        confidence_score = 0.0
        triggers = []

        # Check for explicit transition phrases
        if self.use_patterns:
            phrase = self._detect_transition_phrase(query)
            if phrase:
                signals['transition_phrase'] = 1.0
                signals['phrase_detected'] = phrase
                confidence_score += self.transition_boost
                triggers.append(f"transition phrase: '{phrase}'")

        # Check Q→A→Q pattern
        if self._detect_qa_pattern(messages, query):
            signals['qa_pattern'] = 1.0
            confidence_score += self.qa_pattern_weight
            triggers.append("Q→A→Q pattern")

        # Check speech act of query
        query_msg = {'content': query, 'role': 'user'}
        query_act = self._classify_speech_act(query_msg)
        signals['query_act'] = query_act

        if query_act in BOUNDARY_SPEECH_ACTS:
            signals['boundary_act'] = 1.0
            confidence_score += 0.15
            triggers.append(f"boundary speech act: {query_act}")

        # Decision
        topic_changed = confidence_score >= 0.3  # Threshold for speech-act only

        if topic_changed:
            if confidence_score >= 0.5:
                confidence = Confidence.HIGH
            elif confidence_score >= 0.35:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            reasoning = f"Speech-act triggers: {', '.join(triggers)}"

            new_thread = Thread(
                id=str(uuid.uuid4()),
                name=None,
                start_node_id="",
                end_node_id=None,
                message_count=1,
                created_at=datetime.now(),
                metadata={'triggers': triggers, 'query_act': query_act}
            )
        else:
            confidence = Confidence.LOW
            confidence_score = max(confidence_score, 0.1)
            reasoning = "No strong speech-act signals for topic change"
            new_thread = None

        processing_time = (time.time() - start_time) * 1000

        return TopicDecision(
            topic_changed=topic_changed,
            new_thread=new_thread,
            thread_links=[],
            retrieved_context=None,
            confidence=confidence,
            confidence_score=confidence_score,
            strategy_name=self.name,
            strategy_version=self.version,
            reasoning=reasoning,
            signals=signals,
            processing_time_ms=processing_time,
            metadata={'query_act': query_act}
        )

    def reset(self) -> None:
        """Reset state for new conversation."""
        self._recent_acts = []

    def segment_conversation(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Thread]:
        """Not implemented - Speech-act is incremental, not batch."""
        return []

    def detect_thread_link(
        self,
        query: str,
        recent_context: List[Dict[str, Any]],
        past_threads: List[Thread]
    ) -> List[ThreadLink]:
        """Not implemented for Speech-act strategy."""
        return []

    def retrieve_context(
        self,
        query: str,
        thread_links: List[ThreadLink],
        threads: List[Thread],
        max_tokens: int = 2000
    ) -> RetrievedContext:
        """Not implemented for Speech-act strategy."""
        return RetrievedContext(
            threads=[],
            messages=[],
            relevance_scores={},
            token_count=0,
            retrieval_reason="Speech-act strategy does not support context retrieval"
        )
