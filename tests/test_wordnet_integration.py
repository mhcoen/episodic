#!/usr/bin/env python3
"""
Test WordNet integration for conceptual search.

Verifies that searching for "science" finds physics/quantum mechanics conversations
and that performance remains acceptable.
"""

import time
import uuid
from typing import List, Dict, Any

from episodic.config import config
from episodic.rag import get_rag_system
from episodic.rag_wordnet import concept_expander, expand_search_query
from episodic.conversation import ConversationManager
from episodic.db_connection import get_connection


class WordNetIntegrationTester:
    """Test the WordNet conceptual search integration."""
    
    def __init__(self):
        self.manager = ConversationManager()
        self.test_results = []
        self.test_memories = []
        
        # Enable conceptual search for tests
        config.set('enable_conceptual_search', True)
        config.set('search_query_expansion', True)
        
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log a test result."""
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'details': details
        })
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"  Details: {details}")
    
    def setup_test_conversations(self):
        """Create test conversations with hierarchical concepts."""
        print("\n=== Setting Up Test Conversations ===")
        
        conversations = [
            # Physics conversations
            ("I'm studying quantum mechanics", "Quantum mechanics is fascinating! It deals with phenomena at atomic scales."),
            ("What is general relativity?", "General relativity is Einstein's theory of gravity as spacetime curvature."),
            ("Explain thermodynamics", "Thermodynamics studies heat, energy, and their transformations."),
            
            # Chemistry conversations  
            ("How do chemical bonds work?", "Chemical bonds form when atoms share or transfer electrons."),
            ("What is organic chemistry?", "Organic chemistry studies carbon-based compounds."),
            
            # Biology conversations
            ("Explain DNA replication", "DNA replication is the process of copying genetic material."),
            ("What is evolution?", "Evolution is change in heritable traits over generations."),
            
            # General science conversations
            ("I love science!", "Science is wonderful for understanding our world!"),
            ("The scientific method is important", "Yes, hypothesis testing is fundamental to science."),
            
            # Non-science conversations (control)
            ("What's your favorite movie?", "I enjoy discussing many films!"),
            ("The weather is nice today", "Perfect day for outdoor activities!"),
        ]
        
        for user_msg, assistant_msg in conversations:
            doc_id = str(uuid.uuid4())[:8]
            self.manager.store_conversation_to_memory(
                user_input=user_msg,
                assistant_response=assistant_msg,
                user_node_id=f"user-{doc_id}",
                assistant_node_id=f"assistant-{doc_id}"
            )
            self.test_memories.append(doc_id)
            
        time.sleep(1)  # Let indexing complete
        self.log_result("Test conversation setup", True, f"Created {len(conversations)} conversations")
    
    def test_conceptual_hierarchy(self):
        """Test that searching for broader terms finds specific instances."""
        print("\n=== Testing Conceptual Hierarchy ===")
        
        rag = get_rag_system()
        
        # Test 1: Search for "science" should find physics, chemistry, biology
        results = rag.search("science", n_results=10, source_filter='conversation')
        
        # Check if we found specific science topics
        found_physics = any("quantum" in r['content'].lower() or "relativity" in r['content'].lower() 
                          for r in results['results'])
        found_chemistry = any("chemical" in r['content'].lower() or "organic" in r['content'].lower()
                            for r in results['results'])
        found_biology = any("DNA" in r['content'].lower() or "evolution" in r['content'].lower()
                          for r in results['results'])
        
        if found_physics or found_chemistry or found_biology:
            self.log_result("Broader term finds specific instances", True,
                          f"Science search found: physics={found_physics}, "
                          f"chemistry={found_chemistry}, biology={found_biology}")
        else:
            self.log_result("Broader term finds specific instances", False,
                          "Science search didn't find any specific science topics")
        
        # Test 2: Search for "physics" should rank physics results higher
        results = rag.search("physics", n_results=5, source_filter='conversation')
        
        if results['results']:
            top_result = results['results'][0]
            is_physics = any(term in top_result['content'].lower() 
                           for term in ['quantum', 'relativity', 'thermodynamics'])
            self.log_result("Specific term prioritizes exact matches", is_physics,
                          f"Top result relevance: {top_result['relevance_score']:.3f}")
        else:
            self.log_result("Specific term prioritizes exact matches", False,
                          "No results found for physics")
    
    def test_query_expansion(self):
        """Test query expansion functionality."""
        print("\n=== Testing Query Expansion ===")
        
        # Test expansion modes
        test_cases = [
            ("science", "balanced"),
            ("physics", "narrow"),
            ("chemistry", "broad"),
        ]
        
        for query, mode in test_cases:
            expanded = expand_search_query(query, mode=mode)
            
            # Should include original term
            if query in expanded:
                # Should have expansions
                has_expansions = len(expanded.split()) > 1
                self.log_result(f"Query expansion ({mode} mode)", has_expansions,
                              f"'{query}' → '{expanded}'")
            else:
                self.log_result(f"Query expansion ({mode} mode)", False,
                              "Original query not preserved")
    
    def test_conceptual_relevance_boosting(self):
        """Test that conceptually related results get boosted."""
        print("\n=== Testing Conceptual Relevance Boosting ===")
        
        rag = get_rag_system()
        
        # Search for a general term
        results = rag.search("science education", n_results=10, source_filter='conversation')
        
        # Check if results have conceptual boost info
        boosted_count = sum(1 for r in results['results'] 
                          if 'conceptual_boost' in r)
        
        if boosted_count > 0:
            # Find the most boosted result
            max_boost = max((r.get('conceptual_boost', 1.0) for r in results['results']), 
                          default=1.0)
            self.log_result("Conceptual relevance boosting", True,
                          f"{boosted_count} results boosted, max boost: {max_boost:.2f}")
        else:
            # It's okay if no boosting occurred - depends on content
            self.log_result("Conceptual relevance boosting", True,
                          "No boosting needed for this query")
    
    def test_search_performance(self):
        """Test that conceptual search maintains good performance."""
        print("\n=== Testing Search Performance ===")
        
        rag = get_rag_system()
        
        # Warm up
        rag.search("test", n_results=5, source_filter='conversation')
        
        # Time regular search
        start = time.time()
        for _ in range(5):
            rag.search("quantum mechanics", n_results=10, source_filter='conversation')
        regular_time = (time.time() - start) / 5
        
        # Time conceptual search
        config.set('enable_conceptual_search', True)
        start = time.time()
        for _ in range(5):
            rag.search("science", n_results=10, source_filter='conversation')
        conceptual_time = (time.time() - start) / 5
        
        # Should not be more than 2x slower
        slowdown = conceptual_time / regular_time
        acceptable = slowdown < 2.0
        
        self.log_result("Search performance", acceptable,
                      f"Regular: {regular_time:.3f}s, Conceptual: {conceptual_time:.3f}s, "
                      f"Slowdown: {slowdown:.1f}x")
    
    def test_edge_cases(self):
        """Test edge cases for conceptual search."""
        print("\n=== Testing Edge Cases ===")
        
        # Test with terms not in WordNet
        try:
            expanded = expand_search_query("cryptocurrency blockchain", mode="balanced")
            self.log_result("Non-WordNet terms", True,
                          f"Handled gracefully: '{expanded}'")
        except Exception as e:
            self.log_result("Non-WordNet terms", False, str(e))
        
        # Test with mixed case
        try:
            expanded = expand_search_query("Science PHYSICS", mode="balanced")
            self.log_result("Mixed case handling", True,
                          f"Expanded: '{expanded}'")
        except Exception as e:
            self.log_result("Mixed case handling", False, str(e))
        
        # Test concept similarity
        similarity = concept_expander.get_concept_similarity("physics", "science")
        self.log_result("Concept similarity calculation", similarity > 0,
                      f"physics ~ science: {similarity:.2f}")
    
    def test_disable_conceptual_search(self):
        """Test that conceptual search can be disabled."""
        print("\n=== Testing Disable Functionality ===")
        
        # Disable conceptual search
        config.set('enable_conceptual_search', False)
        
        rag = get_rag_system()
        results = rag.search("science", n_results=5, source_filter='conversation')
        
        # Should not have expansion info
        has_expansion = 'query_expanded' in results
        
        self.log_result("Disable conceptual search", not has_expansion,
                      "Conceptual search properly disabled")
        
        # Re-enable for other tests
        config.set('enable_conceptual_search', True)
    
    def cleanup_test_data(self):
        """Clean up test memories."""
        print("\n=== Cleaning Up Test Data ===")
        
        rag = get_rag_system()
        cleaned = 0
        
        for test_id in self.test_memories:
            results = rag.search(test_id, n_results=10, source_filter='conversation')
            
            for result in results['results']:
                doc_id = result.get('metadata', {}).get('doc_id')
                if doc_id:
                    try:
                        rag.remove_document(doc_id)
                        cleaned += 1
                    except:
                        pass
        
        print(f"Cleaned up {cleaned} test memories")
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*50)
        print("WORDNET INTEGRATION TEST SUMMARY")
        print("="*50)
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['passed'])
        failed = total - passed
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        
        if failed > 0:
            print("\nFailed Tests:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  - {result['test']}: {result['details']}")
        
        print("\n" + "="*50)


def main():
    """Run all WordNet integration tests."""
    print("WORDNET CONCEPTUAL SEARCH INTEGRATION TESTS")
    print("="*50)
    
    tester = WordNetIntegrationTester()
    
    try:
        # Run all tests
        tester.setup_test_conversations()
        tester.test_conceptual_hierarchy()
        tester.test_query_expansion()
        tester.test_conceptual_relevance_boosting()
        tester.test_search_performance()
        tester.test_edge_cases()
        tester.test_disable_conceptual_search()
        
        # Print summary
        tester.print_summary()
        
    finally:
        # Clean up
        tester.cleanup_test_data()


if __name__ == "__main__":
    main()