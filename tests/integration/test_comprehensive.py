#!/usr/bin/env python3
"""
Comprehensive system test for Episodic.
Tests all major functionality in a realistic workflow.
"""

import subprocess
import sys
import os
import tempfile
import time
from pathlib import Path

def run_episodic_script(script_content):
    """Run Episodic with a script and return output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # Run episodic with the script
        result = subprocess.run(
            [sys.executable, "-m", "episodic", "--execute", script_path],
            capture_output=True,
            text=True,
            timeout=300  # Give it more time for actual LLM calls
        )
        return result.stdout, result.stderr, result.returncode
    finally:
        os.unlink(script_path)

def test_full_conversation_flow():
    """Test a complete conversation workflow."""
    print("=" * 80)
    print("COMPREHENSIVE EPISODIC SYSTEM TEST")
    print("=" * 80)
    
    test_script = """
# Initialize database
/init

# Configure system
/set automatic_topic_detection true
/set min_messages_before_topic_change 4
/set show_topics true
/set debug false
# Remove skip_llm_response to use defaults

# Configure for brief responses
/mset chat.max_tokens 50
/mset chat.temperature 0.1

# Verify configuration
/verify

# Check models
/model list
/model chat gpt-3.5-turbo
/model detection gpt-3.5-turbo

# Test conversation with topic changes
Mars in one sentence
Main obstacle?
Journey time?
Radiation?

# Topic change - should detect
Italian pasta basics
Essential ingredients?
Carbonara steps?
Romano vs Parmesan?

# Another topic change  
Neural networks definition
Activation functions?
Backpropagation?
CNN vs RNN?

# Check topics created
/topics list

# Test HuggingFace model
/model chat huggingface/tiiuae/falcon-40b-instruct
Quantum computing in 10 words

# Check costs
/cost

# Test markdown export
/topics list
## Note: Would test export here but requires interactive filename input

# Test compression
/compression stats
/topics compress

# Test RAG if available
/rag
/index --text "Episodic is a DAG-based conversation system"
/search DAG

# Test web search
/web
/muse on
Latest AI in 5 words
/muse off

# Final status
/topics list
/cost
/benchmark

# Save session
/save test_session.txt

/exit
"""
    
    print("\nRunning comprehensive test script...")
    print("This tests:")
    print("- Database initialization")
    print("- Configuration management") 
    print("- Multi-model support (OpenAI + HuggingFace)")
    print("- Topic detection across conversation")
    print("- Cost tracking")
    print("- Compression")
    print("- RAG functionality")
    print("- Web search")
    print("- Session saving")
    print("-" * 80)
    
    stdout, stderr, returncode = run_episodic_script(test_script)
    
    # Analyze results
    print("\nTEST RESULTS:")
    print("=" * 80)
    
    tests_passed = []
    tests_failed = []
    
    # Check initialization
    if "Database initialized" in stdout or "Database already exists" in stdout:
        tests_passed.append("Database initialization")
    else:
        tests_failed.append("Database initialization")
    
    # Check configuration
    if "Set automatic_topic_detection = True" in stdout:
        tests_passed.append("Configuration setting")
    else:
        tests_failed.append("Configuration setting")
    
    # Check model listing
    if "huggingface" in stdout.lower() and "falcon" in stdout.lower():
        tests_passed.append("HuggingFace models available")
    else:
        tests_failed.append("HuggingFace models available")
    
    # Check conversation responses
    if "Mars" in stdout or "colonization" in stdout:
        tests_passed.append("Basic conversation")
    else:
        tests_failed.append("Basic conversation")
    
    # Check topic detection
    if "Topic change detected" in stdout or "Conversation Topics" in stdout:
        tests_passed.append("Topic detection")
    else:
        tests_failed.append("Topic detection")
    
    # Check cost tracking
    if "Session Costs" in stdout or "Total cost:" in stdout:
        tests_passed.append("Cost tracking")
    else:
        tests_failed.append("Cost tracking")
    
    # Check HuggingFace usage
    if "Free tier:" in stdout or "Pro tier:" in stdout:
        tests_passed.append("HuggingFace tier display")
    else:
        tests_failed.append("HuggingFace tier display")
    
    # Check compression
    if "compression" in stdout.lower() and ("stats" in stdout or "compressed" in stdout):
        tests_passed.append("Compression functionality")
    else:
        tests_failed.append("Compression functionality")
    
    # Check RAG
    if "RAG" in stdout or "knowledge base" in stdout.lower():
        tests_passed.append("RAG commands")
    else:
        tests_failed.append("RAG commands")
    
    # Check web search
    if "web" in stdout and "provider" in stdout:
        tests_passed.append("Web search commands")
    else:
        tests_failed.append("Web search commands")
    
    # Print summary
    print(f"\n✅ PASSED: {len(tests_passed)} tests")
    for test in tests_passed:
        print(f"   ✅ {test}")
    
    if tests_failed:
        print(f"\n❌ FAILED: {len(tests_failed)} tests")
        for test in tests_failed:
            print(f"   ❌ {test}")
    
    # Show concerning errors
    if stderr:
        print("\n⚠️  ERRORS DETECTED:")
        print(stderr[:500])  # First 500 chars of errors
    
    # Performance metrics
    if "Average response time:" in stdout:
        print("\n📊 PERFORMANCE METRICS FOUND")
        for line in stdout.split('\n'):
            if "Average" in line or "Total" in line:
                print(f"   {line.strip()}")
    
    # Final verdict
    print("\n" + "=" * 80)
    success_rate = len(tests_passed) / (len(tests_passed) + len(tests_failed))
    if success_rate >= 0.8:
        print(f"✅ SYSTEM TEST PASSED ({success_rate*100:.0f}% success rate)")
    else:
        print(f"❌ SYSTEM TEST FAILED ({success_rate*100:.0f}% success rate)")
    
    print("\nDETAILED OUTPUT SAVED TO: episodic_test_output.txt")
    with open("episodic_test_output.txt", "w") as f:
        f.write("STDOUT:\n" + stdout + "\n\nSTDERR:\n" + stderr)
    
    return success_rate >= 0.8

if __name__ == "__main__":
    success = test_full_conversation_flow()
    sys.exit(0 if success else 1)