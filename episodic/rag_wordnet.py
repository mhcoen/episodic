"""
WordNet integration for conceptual hierarchy in memory search.

This module provides query expansion using WordNet to understand
that "physics" is a type of "science", enabling better search results.
"""

import os
import nltk
from typing import List, Set, Dict, Optional
from functools import lru_cache

from episodic.debug_utils import debug_print


# Download WordNet data if not present
def ensure_wordnet_data():
    """Ensure WordNet data is downloaded."""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        debug_print("Downloading WordNet data...", category="wordnet")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)  # Open Multilingual WordNet


# Initialize WordNet
ensure_wordnet_data()
from nltk.corpus import wordnet


class ConceptExpander:
    """Expands queries using WordNet conceptual hierarchies."""
    
    def __init__(self, max_depth: int = 2, max_expansions: int = 10):
        """
        Initialize the concept expander.
        
        Args:
            max_depth: Maximum depth to traverse up/down hierarchies
            max_expansions: Maximum number of expanded terms to return
        """
        self.max_depth = max_depth
        self.max_expansions = max_expansions
        
        # Cache for common expansions
        self._cache = {}
        
        # Common scientific and technical terms that might not be in WordNet
        self.custom_hierarchy = {
            'quantum_mechanics': ['physics', 'mechanics', 'quantum_physics'],
            'machine_learning': ['artificial_intelligence', 'computer_science', 'technology'],
            'artificial_intelligence': ['computer_science', 'technology', 'science'],
            'quantum_computing': ['computing', 'quantum_mechanics', 'computer_science'],
            'programming': ['computer_science', 'software_development', 'technology'],
        }
    
    @lru_cache(maxsize=1000)
    def get_hypernyms(self, word: str, depth: int = None) -> Set[str]:
        """
        Get broader terms (parents) for a word.
        
        Example: "physics" -> {"science", "natural_science", "discipline"}
        """
        if depth is None:
            depth = self.max_depth
            
        hypernyms = set()
        
        # Check custom hierarchy first
        word_key = word.lower().replace(' ', '_')
        if word_key in self.custom_hierarchy:
            hypernyms.update(self.custom_hierarchy[word_key])
        
        # Get WordNet synsets
        for synset in wordnet.synsets(word):
            # Direct hypernyms
            for hypernym in synset.hypernyms():
                hypernyms.update(lemma.name().replace('_', ' ') 
                               for lemma in hypernym.lemmas())
                
                # Recursive hypernyms up to depth
                if depth > 1:
                    for lemma in hypernym.lemmas():
                        parent_hypernyms = self.get_hypernyms(lemma.name(), depth - 1)
                        hypernyms.update(parent_hypernyms)
        
        return hypernyms
    
    @lru_cache(maxsize=1000)
    def get_hyponyms(self, word: str, depth: int = 1) -> Set[str]:
        """
        Get more specific terms (children) for a word.
        
        Example: "science" -> {"physics", "chemistry", "biology", ...}
        """
        hyponyms = set()
        
        # Check reverse custom hierarchy
        word_lower = word.lower()
        for child, parents in self.custom_hierarchy.items():
            if word_lower in [p.lower() for p in parents]:
                hyponyms.add(child.replace('_', ' '))
        
        # Get WordNet synsets
        for synset in wordnet.synsets(word):
            # Direct hyponyms
            for hyponym in synset.hyponyms():
                hyponyms.update(lemma.name().replace('_', ' ') 
                              for lemma in hyponym.lemmas())
                
                # Recursive hyponyms (limited depth to avoid explosion)
                if depth > 1 and len(hyponyms) < self.max_expansions:
                    for lemma in hyponym.lemmas():
                        child_hyponyms = self.get_hyponyms(lemma.name(), depth - 1)
                        hyponyms.update(child_hyponyms)
                        
                        # Stop if we have too many
                        if len(hyponyms) >= self.max_expansions:
                            break
        
        return hyponyms
    
    @lru_cache(maxsize=1000)
    def get_synonyms(self, word: str) -> Set[str]:
        """Get synonyms for a word."""
        synonyms = set()
        
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        
        return synonyms
    
    def expand_query(self, query: str, 
                    include_hypernyms: bool = True,
                    include_hyponyms: bool = True,
                    include_synonyms: bool = True) -> List[str]:
        """
        Expand a query with conceptually related terms.
        
        Args:
            query: The search query to expand
            include_hypernyms: Include broader terms
            include_hyponyms: Include more specific terms
            include_synonyms: Include synonyms
            
        Returns:
            List of expanded terms including the original
        """
        # Check cache first
        cache_key = (query, include_hypernyms, include_hyponyms, include_synonyms)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        expanded = {query}  # Always include original
        
        # Split multi-word queries
        words = query.lower().split()
        
        for word in words:
            if include_hypernyms:
                hypernyms = self.get_hypernyms(word)
                expanded.update(hypernyms)
                debug_print(f"Hypernyms for '{word}': {hypernyms}", category="wordnet")
            
            if include_hyponyms:
                hyponyms = self.get_hyponyms(word)
                # Limit hyponyms to avoid explosion
                if len(hyponyms) > self.max_expansions // 2:
                    hyponyms = set(list(hyponyms)[:self.max_expansions // 2])
                expanded.update(hyponyms)
                debug_print(f"Hyponyms for '{word}': {hyponyms}", category="wordnet")
            
            if include_synonyms:
                synonyms = self.get_synonyms(word)
                expanded.update(synonyms)
                debug_print(f"Synonyms for '{word}': {synonyms}", category="wordnet")
        
        # Convert to list and limit size
        result = list(expanded)[:self.max_expansions]
        
        # Cache the result
        self._cache[cache_key] = result
        
        debug_print(f"Expanded '{query}' to: {result}", category="wordnet")
        return result
    
    def get_concept_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate conceptual similarity between two words.
        
        Returns a score between 0 and 1, where 1 means identical.
        """
        if word1.lower() == word2.lower():
            return 1.0
        
        # Check if one is a hypernym/hyponym of the other
        word1_hypernyms = self.get_hypernyms(word1, depth=3)
        word2_hypernyms = self.get_hypernyms(word2, depth=3)
        
        if word2 in word1_hypernyms or word1 in word2_hypernyms:
            return 0.8  # Direct parent/child relationship
        
        # Check for common ancestors
        common_ancestors = word1_hypernyms.intersection(word2_hypernyms)
        if common_ancestors:
            return 0.6  # Related through common ancestor
        
        # Check synonyms
        if word2 in self.get_synonyms(word1) or word1 in self.get_synonyms(word2):
            return 0.9  # Synonyms
        
        return 0.0  # No clear relationship


# Global instance for convenience
concept_expander = ConceptExpander()


def expand_search_query(query: str, mode: str = "balanced") -> str:
    """
    Expand a search query with conceptually related terms.
    
    Args:
        query: The original search query
        mode: Expansion mode:
            - "narrow": Only include hypernyms (broader terms)
            - "broad": Include hypernyms and hyponyms
            - "balanced": Include hypernyms and limited hyponyms
            - "children_only": Only include hyponyms (more specific terms)
            
    Returns:
        Expanded query string
    """
    if mode == "narrow":
        terms = concept_expander.expand_query(
            query, 
            include_hypernyms=True,
            include_hyponyms=False,
            include_synonyms=True
        )
    elif mode == "children_only":
        terms = concept_expander.expand_query(
            query,
            include_hypernyms=False,
            include_hyponyms=True,
            include_synonyms=True
        )
    elif mode == "broad":
        terms = concept_expander.expand_query(
            query,
            include_hypernyms=True,
            include_hyponyms=True,
            include_synonyms=True
        )
    else:  # balanced
        # Get expansions but limit hyponyms
        expander = ConceptExpander(max_depth=2, max_expansions=15)
        terms = expander.expand_query(
            query,
            include_hypernyms=True,
            include_hyponyms=True,
            include_synonyms=True
        )
    
    # Join terms, putting original first for higher weight
    expanded_query = query + " " + " ".join(term for term in terms if term != query)
    return expanded_query.strip()


if __name__ == "__main__":
    # Test the concept expander
    expander = ConceptExpander()
    
    print("Testing WordNet concept expansion:")
    print("-" * 50)
    
    test_queries = ["science", "physics", "quantum mechanics", "programming", "gravity"]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        expanded = expand_search_query(query, mode="balanced")
        print(f"Expanded: '{expanded}'")
        
        # Show relationships
        hypernyms = expander.get_hypernyms(query)
        hyponyms = expander.get_hyponyms(query)
        print(f"  Broader terms: {list(hypernyms)[:5]}")
        print(f"  Narrower terms: {list(hyponyms)[:5]}")