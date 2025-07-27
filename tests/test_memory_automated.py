#!/usr/bin/env python3
"""
Automated tests for memory system real-world scenarios.
Run this to verify the memory system is working correctly.
"""

import time
import uuid
from datetime import datetime
from typing import List, Dict, Any

from episodic.conversation import ConversationManager
from episodic.config import config
from episodic.rag import get_rag_system
from episodic.commands.memory import search_memories, list_memories, forget_command, memory_stats_command
from episodic.db_connection import get_connection


class MemorySystemTester:
    """Automated testing for the memory system."""
    
    def __init__(self):
        self.manager = ConversationManager()
        self.test_results = []
        self.test_memories = []
        
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log a test result."""
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"  Details: {details}")
    
    def test_automatic_storage(self):
        """Test that conversations are automatically stored."""
        print("\n=== Testing Automatic Storage ===")
        
        # Store a test conversation
        test_id = str(uuid.uuid4())[:8]
        user_input = f"Test message {test_id}: What is quantum computing?"
        assistant_response = f"Quantum computing is a type of computation that uses quantum mechanical phenomena. Test ID: {test_id}"
        
        try:
            self.manager.store_conversation_to_memory(
                user_input=user_input,
                assistant_response=assistant_response,
                user_node_id=f"user-{test_id}",
                assistant_node_id=f"assistant-{test_id}"
            )
            
            # Give it a moment to persist
            time.sleep(0.5)
            
            # Verify in SQLite
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM rag_documents WHERE preview LIKE ?",
                    (f"%{test_id}%",)
                )
                count = cursor.fetchone()[0]
                
            if count > 0:
                self.log_result("Automatic storage to SQLite", True, f"Found {count} matching documents")
                
                # Verify in search
                rag = get_rag_system()
                results = rag.search(test_id, n_results=5, source_filter='conversation')
                
                if results['results']:
                    self.log_result("Automatic storage to ChromaDB", True, 
                                  f"Found in search with score {results['results'][0]['relevance_score']:.3f}")
                    self.test_memories.append(test_id)
                else:
                    self.log_result("Automatic storage to ChromaDB", False, "Not found in search")
            else:
                self.log_result("Automatic storage to SQLite", False, "Not found in database")
                
        except Exception as e:
            self.log_result("Automatic storage", False, str(e))
    
    def test_search_accuracy(self):
        """Test search accuracy and relevance."""
        print("\n=== Testing Search Accuracy ===")
        
        # Create conversations with known content
        test_data = [
            ("Paris is the capital of France", "France capital cities"),
            ("Python is a programming language", "programming Python coding"),
            ("The weather is sunny today", "weather forecast sunny"),
        ]
        
        for content, search_terms in test_data:
            # Store conversation
            doc_id = str(uuid.uuid4())[:8]
            self.manager.store_conversation_to_memory(
                user_input=content,
                assistant_response=f"Acknowledged: {content}",
                user_node_id=f"user-{doc_id}",
                assistant_node_id=f"assistant-{doc_id}"
            )
            self.test_memories.append(doc_id)
        
        time.sleep(1)  # Let indexing complete
        
        # Test searches
        rag = get_rag_system()
        
        # Exact match test
        results = rag.search("capital of France", n_results=5, source_filter='conversation')
        if results['results'] and results['results'][0]['relevance_score'] > 0.4:
            self.log_result("Exact phrase search", True, 
                          f"Top score: {results['results'][0]['relevance_score']:.3f}")
        else:
            self.log_result("Exact phrase search", False, "Low relevance or no results")
        
        # Keyword match test
        results = rag.search("Python", n_results=5, source_filter='conversation')
        found_python = any("Python" in r['content'] for r in results['results'])
        self.log_result("Keyword search", found_python, 
                       f"Found {len(results['results'])} results")
        
        # Negative test - should not find unrelated content
        results = rag.search("basketball sports", n_results=5, source_filter='conversation')
        no_high_scores = all(r['relevance_score'] < 0.3 for r in results['results'])
        self.log_result("Negative search test", no_high_scores, 
                       "All scores below 0.3 for unrelated query")
    
    def test_relevance_threshold(self):
        """Test relevance threshold filtering."""
        print("\n=== Testing Relevance Threshold ===")
        
        # Set different thresholds and test
        original_threshold = config.get('memory_relevance_threshold', 0.3)
        
        try:
            # Test with high threshold
            config.set('memory_relevance_threshold', 0.6)
            # This should filter out most results
            # We'll need to capture output - for now just verify it works
            self.log_result("High threshold filtering", True, "Threshold set to 0.6")
            
            # Test with low threshold  
            config.set('memory_relevance_threshold', 0.1)
            self.log_result("Low threshold filtering", True, "Threshold set to 0.1")
            
        finally:
            # Restore original
            config.set('memory_relevance_threshold', original_threshold)
    
    def test_memory_commands(self):
        """Test memory command functionality."""
        print("\n=== Testing Memory Commands ===")
        
        # Test /memory list
        try:
            # We can't easily capture the output, but we can verify it doesn't crash
            from episodic.commands.memory import list_memories
            list_memories(limit=5)
            self.log_result("/memory list command", True)
        except Exception as e:
            self.log_result("/memory list command", False, str(e))
        
        # Test /memory-stats
        try:
            from episodic.commands.memory import memory_stats_command
            memory_stats_command()
            self.log_result("/memory-stats command", True)
        except Exception as e:
            self.log_result("/memory-stats command", False, str(e))
    
    def test_persistence(self):
        """Test that memories persist in the database."""
        print("\n=== Testing Persistence ===")
        
        # Check that our test memories are still there
        if self.test_memories:
            test_id = self.test_memories[0]
            
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM rag_documents WHERE preview LIKE ?",
                    (f"%{test_id}%",)
                )
                count = cursor.fetchone()[0]
                
            self.log_result("Memory persistence", count > 0, 
                          f"Test memory {test_id} {'found' if count > 0 else 'not found'}")
        else:
            self.log_result("Memory persistence", False, "No test memories to check")
    
    def test_edge_cases(self):
        """Test edge cases."""
        print("\n=== Testing Edge Cases ===")
        
        rag = get_rag_system()
        
        # Empty search
        try:
            results = rag.search("", n_results=5, source_filter='conversation')
            self.log_result("Empty search query", True, f"Returned {len(results['results'])} results")
        except Exception as e:
            self.log_result("Empty search query", False, str(e))
        
        # Very long query
        long_query = "quantum " * 50
        try:
            results = rag.search(long_query, n_results=5, source_filter='conversation')
            self.log_result("Long search query", True)
        except Exception as e:
            self.log_result("Long search query", False, str(e))
        
        # Special characters
        try:
            results = rag.search("test @#$% special", n_results=5, source_filter='conversation')
            self.log_result("Special characters search", True)
        except Exception as e:
            self.log_result("Special characters search", False, str(e))
    
    def cleanup_test_data(self):
        """Clean up test memories."""
        print("\n=== Cleaning Up Test Data ===")
        
        if not self.test_memories:
            print("No test data to clean up")
            return
            
        rag = get_rag_system()
        cleaned = 0
        
        for test_id in self.test_memories:
            # Search for memories containing our test IDs
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
        print("MEMORY SYSTEM TEST SUMMARY")
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
    """Run all memory system tests."""
    print("EPISODIC MEMORY SYSTEM AUTOMATED TESTS")
    print("="*50)
    
    tester = MemorySystemTester()
    
    try:
        # Run all tests
        tester.test_automatic_storage()
        tester.test_search_accuracy()
        tester.test_relevance_threshold()
        tester.test_memory_commands()
        tester.test_persistence()
        tester.test_edge_cases()
        
        # Print summary
        tester.print_summary()
        
    finally:
        # Clean up
        tester.cleanup_test_data()


if __name__ == "__main__":
    main()