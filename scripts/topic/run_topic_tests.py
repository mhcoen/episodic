#!/usr/bin/env python3
"""
Run all topic detection tests and generate a comprehensive report.

This script runs each test file in scripts/topic/ with different models
and compares the results against expected topic counts.
"""

import sys
import os
import json
import time
from typing import Dict, List, Tuple
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from episodic.db import initialize_db, get_connection
from episodic.config import config
from episodic.cli import execute_script
from episodic.conversation import conversation_manager


# Test configurations with expected topic counts
TEST_CONFIGS = {
    'test-conversation-flow.txt': {
        'expected': 3,
        'description': 'Natural conversation flow (travel → cooking → health)'
    },
    'test-explicit-transitions.txt': {
        'expected': 4,
        'description': 'Explicit topic transitions with clear markers'
    },
    'test-depth-exploration.txt': {
        'expected': 1,
        'description': 'Deep dive into single topic (machine learning)'
    },
    'test-gradual-progression.txt': {
        'expected': 1,
        'description': 'Gradual progression within Python programming'
    },
    'test-ambiguous-transitions.txt': {
        'expected': 2,  # Could be 2 or 3 depending on interpretation
        'description': 'Ambiguous transitions (tech news → programming)'
    },
    'test-related-domains.txt': {
        'expected': 3,
        'description': 'Related but distinct domains (web dev → security → cloud)'
    },
    'test-mixed-patterns.txt': {
        'expected': 5,
        'description': 'Mixed conversation patterns with various transitions'
    }
}

# Models to test
MODELS_TO_TEST = [
    'gpt-3.5-turbo',
    'gpt-4',  # Optional, more expensive
    'ollama/llama3',
    'ollama/mistral'
]


def count_topics() -> int:
    """Count the number of topics in the current conversation."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM topics')
        return cursor.fetchone()[0]


def get_topic_details() -> List[Dict]:
    """Get details of all topics."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                t.name,
                COUNT(DISTINCT n.id) as message_count,
                t.created_at
            FROM topics t
            LEFT JOIN nodes n ON (
                n.id >= t.start_node_id AND 
                (t.end_node_id IS NULL OR n.id <= t.end_node_id)
            )
            GROUP BY t.id
            ORDER BY t.created_at
        ''')
        
        topics = []
        for row in cursor.fetchall():
            topics.append({
                'name': row[0],
                'message_count': row[1],
                'created_at': row[2]
            })
        return topics


def run_single_test(test_file: str, model: str) -> Dict:
    """Run a single test with a specific model."""
    print(f"\n  Running {test_file} with {model}...")
    
    # Reset database
    db_path = os.path.expanduser("~/.episodic/episodic.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    initialize_db(migrate=True)
    
    # Configure model
    config.set('model', model)
    config.set('topic_detection_model', model)
    config.set('main.max_tokens', 30)  # Short responses for testing
    config.set('main.temperature', 0)   # Deterministic
    config.set('debug', False)          # Less verbose
    config.set('automatic_topic_detection', True)
    
    # Run the test script
    script_path = os.path.join('scripts/topic', test_file)
    start_time = time.time()
    
    try:
        # Redirect output to reduce noise
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            execute_script(script_path)
            # Ensure topics are finalized
            conversation_manager.finalize_current_topic()
        
        execution_time = time.time() - start_time
        
        # Get results
        topic_count = count_topics()
        topics = get_topic_details()
        
        result = {
            'success': True,
            'topic_count': topic_count,
            'topics': topics,
            'execution_time': execution_time,
            'output': f.getvalue()
        }
        
    except Exception as e:
        result = {
            'success': False,
            'error': str(e),
            'execution_time': time.time() - start_time
        }
    
    return result


def generate_report(results: Dict) -> str:
    """Generate a markdown report of test results."""
    report = []
    report.append("# Topic Detection Test Results")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary table
    report.append("## Summary\n")
    report.append("| Test File | Expected | " + " | ".join(MODELS_TO_TEST) + " |")
    report.append("|-----------|----------|" + "|".join(["----------"] * len(MODELS_TO_TEST)) + "|")
    
    for test_file, test_config in TEST_CONFIGS.items():
        row = [test_file, str(test_config['expected'])]
        
        for model in MODELS_TO_TEST:
            if test_file in results and model in results[test_file]:
                result = results[test_file][model]
                if result['success']:
                    count = result['topic_count']
                    if count == test_config['expected']:
                        row.append(f"✅ {count}")
                    else:
                        row.append(f"❌ {count}")
                else:
                    row.append("❌ Error")
            else:
                row.append("⏭️ Skipped")
        
        report.append("| " + " | ".join(row) + " |")
    
    # Detailed results
    report.append("\n## Detailed Results\n")
    
    for test_file, test_config in TEST_CONFIGS.items():
        report.append(f"### {test_file}")
        report.append(f"\n**Description:** {test_config['description']}")
        report.append(f"**Expected Topics:** {test_config['expected']}\n")
        
        for model in MODELS_TO_TEST:
            if test_file in results and model in results[test_file]:
                result = results[test_file][model]
                report.append(f"#### {model}")
                
                if result['success']:
                    report.append(f"- **Topic Count:** {result['topic_count']}")
                    report.append(f"- **Execution Time:** {result['execution_time']:.2f}s")
                    report.append("- **Topics Created:**")
                    
                    for topic in result['topics']:
                        report.append(f"  - {topic['name']} ({topic['message_count']} messages)")
                else:
                    report.append(f"- **Error:** {result['error']}")
                
                report.append("")
    
    # Analysis
    report.append("\n## Analysis\n")
    
    # Calculate accuracy for each model
    model_stats = {model: {'correct': 0, 'total': 0} for model in MODELS_TO_TEST}
    
    for test_file, test_config in TEST_CONFIGS.items():
        for model in MODELS_TO_TEST:
            if test_file in results and model in results[test_file]:
                result = results[test_file][model]
                if result['success']:
                    model_stats[model]['total'] += 1
                    if result['topic_count'] == test_config['expected']:
                        model_stats[model]['correct'] += 1
    
    report.append("### Model Accuracy\n")
    for model, stats in model_stats.items():
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            report.append(f"- **{model}:** {stats['correct']}/{stats['total']} ({accuracy:.1f}% accurate)")
    
    return "\n".join(report)


def main():
    """Run all tests and generate report."""
    print("🧪 Topic Detection Test Suite")
    print("=" * 50)
    
    # Check which models are available
    available_models = []
    for model in MODELS_TO_TEST:
        if model.startswith('ollama/'):
            # Check if Ollama is running
            try:
                import requests
                response = requests.get('http://localhost:11434/api/tags', timeout=1)
                if response.status_code == 200:
                    available_models.append(model)
                else:
                    print(f"⚠️  Skipping {model} - Ollama not running")
            except:
                print(f"⚠️  Skipping {model} - Ollama not available")
        elif model.startswith('gpt'):
            # Check for OpenAI API key
            if os.getenv('OPENAI_API_KEY'):
                available_models.append(model)
            else:
                print(f"⚠️  Skipping {model} - No OpenAI API key")
        else:
            available_models.append(model)
    
    if not available_models:
        print("❌ No models available for testing!")
        return
    
    print(f"\n✅ Testing with models: {', '.join(available_models)}")
    
    # Run tests
    results = {}
    
    for test_file in TEST_CONFIGS.keys():
        results[test_file] = {}
        print(f"\n📝 Testing: {test_file}")
        
        for model in available_models:
            result = run_single_test(test_file, model)
            results[test_file][model] = result
            
            if result['success']:
                expected = TEST_CONFIGS[test_file]['expected']
                actual = result['topic_count']
                status = "✅" if actual == expected else "❌"
                print(f"    {model}: {status} {actual} topics (expected {expected})")
            else:
                print(f"    {model}: ❌ Error - {result['error']}")
    
    # Generate and save report
    report = generate_report(results)
    report_path = 'scripts/topic/test_results.md'
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Also save raw results as JSON
    json_path = 'scripts/topic/test_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_path}")
    print(f"📊 Raw data saved to: {json_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    for model in available_models:
        correct = sum(1 for test_file in TEST_CONFIGS.keys() 
                     if test_file in results 
                     and model in results[test_file]
                     and results[test_file][model]['success']
                     and results[test_file][model]['topic_count'] == TEST_CONFIGS[test_file]['expected'])
        total = sum(1 for test_file in TEST_CONFIGS.keys()
                   if test_file in results
                   and model in results[test_file]
                   and results[test_file][model]['success'])
        
        if total > 0:
            print(f"{model}: {correct}/{total} tests passed ({(correct/total)*100:.1f}%)")


if __name__ == '__main__':
    main()