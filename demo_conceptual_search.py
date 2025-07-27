#!/usr/bin/env python3
"""
Demo of conceptual search with WordNet integration.

Shows how searching for general terms finds specific instances.
"""

from episodic.config import config
from episodic.rag_wordnet import expand_search_query, concept_expander


def main():
    print("EPISODIC CONCEPTUAL SEARCH DEMO")
    print("=" * 50)
    
    # Show query expansion examples
    print("\n1. Query Expansion Examples:")
    print("-" * 30)
    
    queries = ["science", "physics", "programming", "machine learning"]
    
    for query in queries:
        expanded = expand_search_query(query, mode="balanced")
        print(f"\n'{query}' expands to:")
        terms = expanded.split()[:10]  # Show first 10 terms
        print(f"  {', '.join(terms)}...")
    
    # Show concept relationships
    print("\n\n2. Concept Relationships:")
    print("-" * 30)
    
    pairs = [
        ("physics", "science"),
        ("quantum mechanics", "physics"),
        ("python", "programming"),
        ("chemistry", "biology"),
    ]
    
    for word1, word2 in pairs:
        similarity = concept_expander.get_concept_similarity(word1, word2)
        print(f"\n'{word1}' ~ '{word2}': {similarity:.2f}")
        
        # Show why they're related
        if similarity > 0:
            w1_hyper = concept_expander.get_hypernyms(word1, depth=2)
            w2_hyper = concept_expander.get_hypernyms(word2, depth=2)
            common = w1_hyper.intersection(w2_hyper)
            if common:
                print(f"  Common concepts: {', '.join(list(common)[:3])}")
    
    # Show how to enable/disable
    print("\n\n3. Configuration:")
    print("-" * 30)
    print("\nTo enable conceptual search:")
    print("  episodic config set enable_conceptual_search true")
    print("\nTo adjust settings:")
    print("  episodic config set expansion_max_depth 3")
    print("  episodic config set conceptual_boost_factor 0.5")
    
    print("\n\n4. How It Works:")
    print("-" * 30)
    print("\nWhen enabled, searching for 'science' will also find conversations about:")
    print("  - Physics, chemistry, biology (hyponyms)")
    print("  - Natural science, discipline (hypernyms)")
    print("  - Scientific discipline, skill (synonyms)")
    print("\nThis solves the problem where vector search alone wouldn't understand")
    print("that 'physics is science'.")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()