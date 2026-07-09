"""Semantic-drift methods for ConversationManager.

Mixin split out of conversation.py; ConversationManager inherits it, so these
run on the instance (self.drift_calculator, set in __init__).
"""

from typing import Optional

import typer

from episodic.config import config
from episodic.db import get_ancestry
from episodic.color_utils import secho_color
from episodic.debug_utils import debug_print
from episodic.ml import ConversationalDrift


class _ConversationDriftMixin:
    """Drift calculator management and semantic-drift computation/display."""

    def get_drift_calculator(self) -> Optional[ConversationalDrift]:
        """Get or create the drift calculator instance."""
        # Check if drift detection is disabled in config (check every time for runtime changes)
        if not config.get("show_drift"):
            return None

        # Get current embedding settings from config
        embedding_provider = config.get("drift_embedding_provider", "sentence-transformers")
        embedding_model = config.get("drift_embedding_model", "paraphrase-mpnet-base-v2")

        # Check if settings changed - recreate if so
        if (self.drift_calculator is not None and
            self.drift_calculator is not False and
            (getattr(self.drift_calculator, '_embedding_provider', None) != embedding_provider or
             getattr(self.drift_calculator, '_embedding_model', None) != embedding_model)):
            self.drift_calculator = None  # Force recreation

        if self.drift_calculator is None or self.drift_calculator is False:
            try:
                self.drift_calculator = ConversationalDrift(
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model
                )
                # Store settings for change detection
                self.drift_calculator._embedding_provider = embedding_provider
                self.drift_calculator._embedding_model = embedding_model
                if config.get("debug"):
                    typer.echo(f"✅ Initialized drift calculator with {embedding_provider}/{embedding_model}")
            except Exception as e:
                # If drift calculator fails to initialize (e.g., missing dependencies),
                # disable drift detection for this session
                if config.get("debug"):
                    typer.echo(f"⚠️  Drift detection disabled: {e}")
                self.drift_calculator = False  # Mark as disabled
        return self.drift_calculator if self.drift_calculator is not False else None

    def compute_semantic_drift(self, current_user_node_id: str) -> Optional[float]:
        """
        Compute semantic drift between current and previous user message.

        Returns:
            Drift score (0.0-1.0) or None if not computable (e.g., < 2 user messages)
        """
        calc = self.get_drift_calculator()
        if not calc:
            return None

        try:
            # Get conversation history from root to current node
            conversation_chain = get_ancestry(current_user_node_id)

            # Filter to user messages only
            user_messages = [node for node in conversation_chain
                            if node.get("role") == "user" and node.get("content", "").strip()]

            # Need at least 2 user messages for comparison
            if len(user_messages) < 2:
                return None

            previous_user = user_messages[-2]
            current_user = user_messages[-1]

            return calc.calculate_drift(previous_user, current_user, text_field="content")
        except Exception as e:
            if config.get("debug"):
                typer.echo(f"⚠️  Drift computation error: {e}")
            return None

    def display_semantic_drift(
        self,
        current_user_node_id: str,
        cached_drift: Optional[float] = None
    ) -> None:
        """
        Display semantic drift between consecutive user messages.

        Args:
            current_user_node_id: The current user node ID
            cached_drift: Pre-computed drift score (avoids recomputation)
        """
        try:
            # Use cached drift if provided, otherwise compute
            if cached_drift is not None:
                drift_score = cached_drift
            else:
                drift_score = self.compute_semantic_drift(current_user_node_id)
                if drift_score is None:
                    return  # Not enough data for drift

            # Get previous user info for display (need ancestry for short_id)
            conversation_chain = get_ancestry(current_user_node_id)
            user_messages = [node for node in conversation_chain
                            if node.get("role") == "user" and node.get("content", "").strip()]
            if len(user_messages) < 2:
                return
            previous_user = user_messages[-2]
            current_user = user_messages[-1]
            
            # Format drift display based on score level
            if drift_score >= 0.8:
                drift_emoji = "🔄"
                drift_desc = "High topic shift"
            elif drift_score >= 0.6:
                drift_emoji = "📈"
                drift_desc = "Moderate drift"
            elif drift_score >= 0.3:
                drift_emoji = "➡️"
                drift_desc = "Low drift"
            else:
                drift_emoji = "🎯"
                drift_desc = "Minimal drift"
            
            # Display drift information (subtle diagnostic)
            from episodic.configuration import get_drift_color
            prev_short_id = previous_user.get("short_id", "??")
            secho_color(f"\n{drift_emoji}  Semantic drift: {drift_score:.3f} ({drift_desc}) from user message {prev_short_id}", fg=get_drift_color(), dim=True)
            
            # Show additional context if debug mode is enabled
            if config.get("debug"):
                prev_content = previous_user.get("content", "")[:80]
                curr_content = current_user.get("content", "")[:80]
                debug_print(f"Previous: {prev_content}{'...' if len(previous_user.get('content', '')) > 80 else ''}", indent=True)
                debug_print(f"Current:  {curr_content}{'...' if len(current_user.get('content', '')) > 80 else ''}", indent=True)
                
                # Show embedding cache efficiency
                calc = self.get_drift_calculator()
                if calc:
                    cache_size = calc.get_cache_size()
                    debug_print(f"Embedding cache: {cache_size} entries", indent=True)
            
        except Exception as e:
            # If drift calculation fails, silently continue (don't disrupt conversation flow)
            if config.get("debug"):
                typer.echo(f"⚠️  Drift calculation error: {e}")
    
