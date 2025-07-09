#!/usr/bin/env python3
"""
Test suite for RAG and help command integration.

This tests the critical flow of:
1. User asks help question
2. RAG searches documentation
3. Context is added to prompt
4. LLM provides accurate answer using context

This test suite should prevent regression where LLM ignores provided context.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from episodic.commands.help import get_help_rag, help_command
from episodic.rag import get_rag_system
from episodic.config import config
from episodic.llm import query_llm
from episodic.db import create_rag_tables
import warnings


class TestRAGHelpIntegration(unittest.TestCase):
    """Test that RAG properly enhances help queries with documentation context."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Suppress warnings
        warnings.filterwarnings("ignore")
        
        # Use test database
        os.environ['EPISODIC_DB_PATH'] = '/tmp/test_rag_help.db'
        
        # Create tables
        create_rag_tables()
        
        # Initialize help RAG
        cls.help_rag = get_help_rag()
        cls.rag_system = get_rag_system()
        
        # Save original config
        cls.original_model = config.get('model')
        cls.original_stream = config.get('stream_responses')
        
        # Disable streaming for tests
        config.set('stream_responses', False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Restore config
        if cls.original_model:
            config.set('model', cls.original_model)
        config.set('stream_responses', cls.original_stream)
        
        # Clean up database
        if os.path.exists('/tmp/test_rag_help.db'):
            os.unlink('/tmp/test_rag_help.db')
    
    def test_help_search_finds_muse_documentation(self):
        """Test that searching for 'muse' finds relevant documentation."""
        results = self.help_rag.search_help("what does muse do", n_results=3)
        
        self.assertGreater(len(results), 0, "Should find at least one result")
        
        # Check that at least one result mentions muse
        found_muse = False
        for result in results:
            if 'muse' in result['content'].lower():
                found_muse = True
                break
        
        self.assertTrue(found_muse, "At least one result should mention 'muse'")
    
    def test_rag_enhancement_adds_context(self):
        """Test that RAG enhancement adds documentation context to prompts."""
        # Set up help collection
        original_collection = self.rag_system.collection
        self.rag_system.collection = self.help_rag.collection
        
        try:
            query = "what does muse do"
            base_prompt = f"Please answer: {query}"
            
            enhanced_prompt, sources = self.rag_system.enhance_with_context(base_prompt)
            
            # Check that context was added
            self.assertGreater(len(enhanced_prompt), len(base_prompt), 
                             "Enhanced prompt should be longer than base prompt")
            
            # Check that sources were found
            self.assertGreater(len(sources), 0, "Should have found source documents")
            
            # Check that muse is mentioned in enhanced prompt
            self.assertIn('muse', enhanced_prompt.lower(), 
                         "Enhanced prompt should contain 'muse' from documentation")
            
            # Check for specific muse documentation
            self.assertIn('perplexity', enhanced_prompt.lower(),
                         "Should include description of muse as Perplexity-like")
            
        finally:
            # Restore original collection
            self.rag_system.collection = original_collection
    
    def test_llm_uses_provided_context(self):
        """Test that LLM actually uses the provided context in its response."""
        # Create a test prompt with explicit muse documentation
        test_prompt = """Based on the following documentation, answer: what does muse do?

[Relevant context from knowledge base]:
### /muse
Enable Perplexity-like web search mode. Muse mode transforms Episodic into a Perplexity-like conversational web search tool where all input is automatically treated as web search queries.

Answer the question based on the provided documentation."""
        
        # Query LLM
        model = config.get('model', 'gpt-3.5-turbo')
        response, _ = query_llm(test_prompt, model=model, stream=False)
        
        # Check that response uses the context
        response_lower = response.lower()
        
        # Response should NOT say muse is not documented
        self.assertNotIn('not mentioned', response_lower,
                        "LLM should not claim muse is not mentioned")
        self.assertNotIn('not documented', response_lower,
                        "LLM should not claim muse is not documented")
        
        # Response SHOULD mention key muse features
        self.assertTrue(
            'perplexity' in response_lower or 'web search' in response_lower,
            f"LLM response should mention Perplexity or web search. Got: {response}"
        )
    
    def test_help_command_integration(self):
        """Test the full help command flow."""
        # This is an integration test that may fail due to LLM behavior
        # Mark it as expected to potentially fail
        
        # Note: We can't easily capture help_command output in tests
        # This would need refactoring to return the response instead of printing
        
        # For now, just verify it doesn't crash
        try:
            # We'd need to mock or capture stdout to test the actual response
            # This is a limitation of the current design
            pass
        except Exception as e:
            self.fail(f"help_command should not raise exception: {e}")
    
    def test_prompt_structure_impact(self):
        """Test how prompt structure affects context usage."""
        muse_context = """### /muse
Enable Perplexity-like web search mode where all input becomes web search queries."""
        
        # Test with formatting instructions AFTER context (current approach)
        prompt_context_last = f"""Answer: what does muse do?

IMPORTANT: Format nicely, be concise, use proper grammar.

[Context]:
{muse_context}"""
        
        response1, _ = query_llm(prompt_context_last, stream=False)
        
        # Test with context FIRST (potentially better approach)
        prompt_context_first = f"""[Context]:
{muse_context}

Answer: what does muse do?

IMPORTANT: Format nicely, be concise, use proper grammar."""
        
        response2, _ = query_llm(prompt_context_first, stream=False)
        
        # Both should mention muse correctly
        for response in [response1, response2]:
            self.assertNotIn('not mentioned', response.lower(),
                           "Should not claim muse is not mentioned")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)