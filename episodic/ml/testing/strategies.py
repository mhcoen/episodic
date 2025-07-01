"""
Testing strategies for conversational drift analysis.

This module provides various approaches for loading conversation data
and analyzing drift patterns with different testing methodologies.
"""

from typing import List, Dict, Any
from abc import ABC, abstractmethod
from ...db import get_all_nodes, get_ancestry, get_recent_nodes


class TestingStrategy:
    """Configurable testing strategy supporting multiple approaches."""
    
    def __init__(self, strategy: str = "recent_conversations", **kwargs):
        """
        Initialize testing strategy.
        
        Args:
            strategy: Testing approach ("recent_conversations", "conversation_chains", "full_database", "custom")
            **kwargs: Strategy-specific parameters
        """
        self.strategy = strategy
        self.kwargs = kwargs
        self._backend = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the appropriate backend based on strategy."""
        if self.strategy == "recent_conversations":
            limit = self.kwargs.get("limit", 10)
            self._backend = RecentConversationsStrategy(limit)
        elif self.strategy == "conversation_chains":
            max_chains = self.kwargs.get("max_chains", 3)
            min_length = self.kwargs.get("min_length", 3)
            self._backend = ConversationChainsStrategy(max_chains, min_length)
        elif self.strategy == "full_database":
            self._backend = FullDatabaseStrategy()
        elif self.strategy == "custom":
            node_ids = self.kwargs.get("node_ids", [])
            self._backend = CustomNodesStrategy(node_ids)
        else:
            raise ValueError(f"Unknown testing strategy: {self.strategy}")
    
    def load_test_data(self) -> List[List[Dict[str, Any]]]:
        """
        Load test conversation sequences.
        
        Returns:
            List of conversation sequences (each sequence is a list of nodes)
        """
        return self._backend.load_test_data()
    
    def get_description(self) -> str:
        """Get a description of what this strategy tests."""
        return self._backend.get_description()


class TestingBackend(ABC):
    """Abstract base class for testing backends."""
    
    @abstractmethod
    def load_test_data(self) -> List[List[Dict[str, Any]]]:
        """Load test conversation sequences."""
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of what this strategy tests."""


class RecentConversationsStrategy(TestingBackend):
    """Load recent conversations for basic functionality testing."""
    
    def __init__(self, limit: int = 10):
        """
        Initialize recent conversations strategy.
        
        Args:
            limit: Maximum number of recent nodes to load
        """
        self.limit = limit
    
    def load_test_data(self) -> List[List[Dict[str, Any]]]:
        """Load recent conversations as a single sequence."""
        try:
            nodes = get_recent_nodes(limit=self.limit)
            # Filter out empty content nodes
            content_nodes = [node for node in nodes if node.get("content", "").strip()]
            return [content_nodes] if len(content_nodes) >= 2 else []
        except Exception:
            return []
    
    def get_description(self) -> str:
        """Get strategy description."""
        return f"Recent {self.limit} conversations for basic drift testing"


class ConversationChainsStrategy(TestingBackend):
    """Load complete conversation chains (ancestry paths)."""
    
    def __init__(self, max_chains: int = 3, min_length: int = 3):
        """
        Initialize conversation chains strategy.
        
        Args:
            max_chains: Maximum number of chains to load
            min_length: Minimum chain length to consider
        """
        self.max_chains = max_chains
        self.min_length = min_length
    
    def load_test_data(self) -> List[List[Dict[str, Any]]]:
        """Load multiple conversation chains."""
        try:
            # Get recent nodes to find chain endpoints
            recent_nodes = get_recent_nodes(limit=10)
            chains = []
            
            for node in recent_nodes:
                if len(chains) >= self.max_chains:
                    break
                
                node_id = node.get("short_id") or node.get("id")
                if not node_id:
                    continue
                
                # Load full conversation chain
                chain = get_ancestry(node_id)
                content_chain = [n for n in chain if n.get("content", "").strip()]
                
                if len(content_chain) >= self.min_length:
                    chains.append(content_chain)
            
            return chains
        except Exception:
            return []
    
    def get_description(self) -> str:
        """Get strategy description."""
        return f"Up to {self.max_chains} conversation chains (min length {self.min_length})"


class FullDatabaseStrategy(TestingBackend):
    """Load all conversations from the database."""
    
    def load_test_data(self) -> List[List[Dict[str, Any]]]:
        """Load all conversations as a single large sequence."""
        try:
            all_nodes = get_all_nodes()
            content_nodes = [node for node in all_nodes if node.get("content", "").strip()]
            return [content_nodes] if len(content_nodes) >= 2 else []
        except Exception:
            return []
    
    def get_description(self) -> str:
        """Get strategy description."""
        return "All conversations in database for comprehensive analysis"


class CustomNodesStrategy(TestingBackend):
    """Load specific nodes by ID."""
    
    def __init__(self, node_ids: List[str]):
        """
        Initialize custom nodes strategy.
        
        Args:
            node_ids: List of node IDs or short_ids to load chains for
        """
        self.node_ids = node_ids
    
    def load_test_data(self) -> List[List[Dict[str, Any]]]:
        """Load conversation chains for specified nodes."""
        chains = []
        
        try:
            for node_id in self.node_ids:
                chain = get_ancestry(node_id)
                content_chain = [n for n in chain if n.get("content", "").strip()]
                if len(content_chain) >= 2:
                    chains.append(content_chain)
            return chains
        except Exception:
            return []
    
    def get_description(self) -> str:
        """Get strategy description."""
        return f"Custom node chains for {len(self.node_ids)} specified nodes"