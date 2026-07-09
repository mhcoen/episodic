"""Topic-reactivation methods for ConversationManager.

Mixin split out of conversation.py; ConversationManager inherits it. Heavy
dependencies (numpy, RAG collections, the recall probe) are imported locally
inside the methods, so this module only needs config/debug_print.
"""

from typing import Optional, Tuple

from episodic.config import config
from episodic.debug_utils import debug_print


class _ConversationReactivationMixin:
    """Implicit topic reactivation probe and application."""

    def probe_topic_reactivation(
        self,
        user_input: str,
        recent_nodes: list,
        is_meta_query: bool = False,
        is_recall_intent: bool = False
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Probe for implicit topic reactivation.

        Args:
            user_input: The user's message text
            recent_nodes: Recent conversation nodes
            is_meta_query: Whether this is a meta/command query
            is_recall_intent: Whether this is a memory recall query

        Returns:
            Tuple of (should_reactivate, topic_name, topic_start_node_id)
        """
        from datetime import datetime
        import numpy as np

        # Skip probe for meta queries, recall intents, or very short inputs
        if is_meta_query or is_recall_intent or len(user_input.split()) < 4:
            return False, None, None

        # Decrement cooldown
        if self.reactivation_cooldown_turns > 0:
            self.reactivation_cooldown_turns -= 1

        # Get active topic
        active_topic = self.get_current_topic()
        active_start_node_id = active_topic[1] if active_topic else None

        try:
            from episodic.recall.reactivation import probe_reactivation
            from episodic.rag_collections import get_multi_collection_rag, CollectionType

            # Get embedding for user input
            rag = get_multi_collection_rag()
            collection = rag.get_collection(CollectionType.CONVERSATION)
            embeddings = collection._embedding_function([user_input])
            user_embedding = np.array(embeddings[0])

            # Probe for reactivation
            decision = probe_reactivation(
                user_input=user_input,
                user_embedding=user_embedding,
                active_topic_start_node_id=active_start_node_id,
                cooldown_turns=self.reactivation_cooldown_turns,
                now=datetime.now(),
                recent_nodes=recent_nodes
            )

            # Store the decision on self for later persistence
            # (will be persisted once we have the user_node_id)
            self._last_reactivation_decision = decision

            # Gate reactivation to self
            if (decision.action == "REACTIVATE" and
                decision.topic_start_node_id == active_start_node_id):
                return False, None, None

            # Handle DISAMBIGUATE with best-guess-then-correction approach
            if decision.action == "DISAMBIGUATE" and decision.options:
                from episodic.recall.correction import CorrectionState

                # Store runner-ups for potential correction on next turn
                best_option = decision.options[0]
                runner_ups = decision.options[1:] if len(decision.options) > 1 else []

                if runner_ups:
                    self.pending_correction = CorrectionState(
                        query=user_input,
                        chosen_option=best_option,
                        runner_ups=runner_ups,
                        turn_created=self._get_turn_idx(),
                    )

                # Proceed with best option (conversational flow)
                decision.action = "REACTIVATE"
                decision.topic_name = best_option.topic_name
                decision.topic_start_node_id = best_option.topic_start_node_id
                decision.debug["disambiguation_choice"] = "auto_best"
                decision.debug["correction_state_stored"] = bool(runner_ups)
                return True, best_option.topic_name, best_option.topic_start_node_id

            if decision.action == "REACTIVATE":
                return True, decision.topic_name, decision.topic_start_node_id

            return False, None, None

        except Exception as e:
            debug_print(f"Reactivation probe error: {e}", category="memory")
            return False, None, None

    def apply_topic_reactivation(
        self,
        topic_name: str,
        topic_start_node_id: str,
        user_input: str
    ) -> Optional[str]:
        """
        Apply topic reactivation: switch topic and assemble context packet.

        Args:
            topic_name: Name of topic to reactivate
            topic_start_node_id: Start node ID of topic
            user_input: User's input (for anchor selection)

        Returns:
            Context packet string to inject, or None on error
        """
        import numpy as np

        try:
            from episodic.recall.reactivation import assemble_reactivation_packet
            from episodic.rag_collections import get_multi_collection_rag, CollectionType

            # Switch to reactivated topic
            self.set_current_topic(topic_name, topic_start_node_id)

            # Get embedding for anchor selection
            rag = get_multi_collection_rag()
            collection = rag.get_collection(CollectionType.CONVERSATION)
            embeddings = collection._embedding_function([user_input])
            user_embedding = np.array(embeddings[0])

            # Assemble context packet
            packet, debug_info = assemble_reactivation_packet(
                topic_start_node_id=topic_start_node_id,
                user_embedding=user_embedding,
                token_budget=150
            )

            # Set cooldown
            self.reactivation_cooldown_turns = 3
            self.last_reactivation_topic_start_node_id = topic_start_node_id

            if config.get("debug"):
                debug_print(f"Reactivated topic: {topic_name}")
                debug_print(f"Packet length: {len(packet)} chars")

            return packet if packet else None

        except Exception as e:
            debug_print(f"Reactivation apply error: {e}", category="memory")
            return None

