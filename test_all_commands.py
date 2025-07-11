#!/usr/bin/env python3
"""
Comprehensive CLI command testing script.
Tests every command in Episodic and documents all errors.
"""

import subprocess
import sys
import os
from datetime import datetime

# Test results storage
test_results = {
    "passed": [],
    "failed": [],
    "errors": []
}

def run_command(command, description=""):
    """Run a CLI command and capture output."""
    print(f"\n{'='*60}")
    print(f"Testing: {command}")
    if description:
        print(f"Description: {description}")
    print("-" * 60)
    
    try:
        # Use echo to pipe input for interactive commands
        if command.startswith("/"):
            full_command = f'echo "{command}" | python -m episodic'
        else:
            full_command = f'python -m episodic {command}'
            
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        print(f"Exit code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
        # Check for errors
        if result.returncode != 0:
            test_results["failed"].append({
                "command": command,
                "description": description,
                "exit_code": result.returncode,
                "error": result.stderr
            })
        elif "Error" in result.stdout or "Error" in result.stderr:
            test_results["errors"].append({
                "command": command,
                "description": description,
                "stdout": result.stdout,
                "stderr": result.stderr
            })
        else:
            test_results["passed"].append(command)
            
        return result
        
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Command took too long")
        test_results["failed"].append({
            "command": command,
            "description": description,
            "error": "Timeout after 10 seconds"
        })
        return None
    except Exception as e:
        print(f"EXCEPTION: {e}")
        test_results["failed"].append({
            "command": command,
            "description": description,
            "error": str(e)
        })
        return None

# Initialize database first
print("Initializing database...")
run_command("--init", "Initialize database")

# Test all commands
commands_to_test = [
    # Basic commands
    ("/help", "Show help menu"),
    ("/h", "Help shortcut"),
    ("/welcome", "Show welcome message"),
    ("/about", "Show about information"),
    
    # Model commands
    ("/model", "Show current model"),
    ("/model list", "List all models"),
    ("/model chat gpt-3.5-turbo", "Set chat model"),
    ("/model detection gpt-3.5-turbo", "Set detection model"),
    ("/model compression gpt-3.5-turbo", "Set compression model"),
    ("/model synthesis gpt-3.5-turbo", "Set synthesis model"),
    
    # Model parameter commands
    ("/mset", "Show all model parameters"),
    ("/mset chat", "Show chat model parameters"),
    ("/mset chat.temperature 0.7", "Set chat temperature"),
    ("/mset detection.temperature 0", "Set detection temperature"),
    
    # Configuration commands
    ("/config", "Show configuration"),
    ("/config-docs", "Show configuration documentation"),
    ("/set debug on", "Enable debug mode"),
    ("/set debug off", "Disable debug mode"),
    ("/set text_wrap on", "Enable text wrapping"),
    ("/set show_costs on", "Enable cost display"),
    ("/verify", "Verify configuration"),
    ("/reset", "Reset configuration"),
    
    # Topic commands
    ("/topics", "List topics"),
    ("/topics list", "List topics explicitly"),
    ("/topics rename", "Rename ongoing topics"),
    ("/topics stats", "Show topic statistics"),
    ("/topics scores", "Show topic detection scores"),
    ("/topics index 5", "Manual topic detection"),
    ("/topics compress", "Compress current topic"),
    
    # Compression commands
    ("/compression", "Show compression stats"),
    ("/compression stats", "Show compression stats explicitly"),
    ("/compression queue", "Show compression queue"),
    ("/compression api-stats", "Show API usage stats"),
    ("/compression reset-api", "Reset API stats"),
    
    # RAG commands
    ("/rag", "Show RAG status"),
    ("/rag on", "Enable RAG"),
    ("/rag off", "Disable RAG"),
    ("/search test", "Search knowledge base"),
    ("/s test", "Search shortcut"),
    ("/index README.md", "Index a file"),
    ("/i README.md", "Index shortcut"),
    ("/docs", "List documents"),
    ("/docs list", "List documents explicitly"),
    
    # Web search commands
    ("/websearch test query", "Web search test"),
    ("/ws test query", "Web search shortcut"),
    ("/websearch on", "Enable web search"),
    ("/websearch off", "Disable web search"),
    ("/websearch config", "Show web search config"),
    ("/websearch stats", "Show web search stats"),
    ("/websearch cache clear", "Clear web search cache"),
    
    # Muse mode
    ("/muse", "Toggle muse mode"),
    ("/muse on", "Enable muse mode"),
    ("/muse off", "Disable muse mode"),
    
    # History and session commands
    ("/history", "Show conversation history"),
    ("/history 5", "Show last 5 messages"),
    ("/history all", "Show all history"),
    ("/tree", "Show conversation tree"),
    ("/graph", "Show conversation graph"),
    ("/cost", "Show session costs"),
    ("/save test_session.txt", "Save session to file"),
    
    # Debug commands
    ("/debug", "Toggle debug mode"),
    ("/debug on", "Enable debug explicitly"),
    ("/debug off", "Disable debug explicitly"),
    ("/api-stats", "Show API statistics"),
    ("/reset-api-stats", "Reset API statistics"),
    
    # Other commands
    ("/init", "Re-initialize database"),
    ("/clear", "Clear screen"),
    ("/cls", "Clear screen shortcut"),
    ("/drift", "Show semantic drift"),
    ("/export", "Export conversation"),
    ("/summary", "Generate conversation summary"),
    ("/rename-topics", "Rename topics (deprecated)"),
    ("/compress-current-topic", "Compress current topic (deprecated)"),
    
    # Exit commands
    ("/exit", "Exit application"),
    ("/quit", "Quit application"),
    ("/bye", "Bye shortcut"),
]

# Run all tests
for command, description in commands_to_test:
    run_command(command, description)

# Print summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print(f"Total tests: {len(commands_to_test)}")
print(f"Passed: {len(test_results['passed'])}")
print(f"Failed: {len(test_results['failed'])}")
print(f"Errors: {len(test_results['errors'])}")

# Print failed tests
if test_results['failed']:
    print("\n" + "-"*60)
    print("FAILED TESTS:")
    print("-"*60)
    for failure in test_results['failed']:
        print(f"\nCommand: {failure['command']}")
        print(f"Description: {failure['description']}")
        print(f"Error: {failure.get('error', 'Unknown error')}")

# Print error tests
if test_results['errors']:
    print("\n" + "-"*60)
    print("TESTS WITH ERRORS:")
    print("-"*60)
    for error in test_results['errors']:
        print(f"\nCommand: {error['command']}")
        print(f"Description: {error['description']}")
        if "Error" in error.get('stdout', ''):
            print(f"Error in stdout: {error['stdout']}")
        if "Error" in error.get('stderr', ''):
            print(f"Error in stderr: {error['stderr']}")

# Save detailed results
with open("test_results.txt", "w") as f:
    f.write(f"Episodic CLI Test Results - {datetime.now()}\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Summary:\n")
    f.write(f"- Total tests: {len(commands_to_test)}\n")
    f.write(f"- Passed: {len(test_results['passed'])}\n")
    f.write(f"- Failed: {len(test_results['failed'])}\n")
    f.write(f"- Errors: {len(test_results['errors'])}\n\n")
    
    if test_results['failed']:
        f.write("FAILED TESTS:\n")
        f.write("-"*60 + "\n")
        for failure in test_results['failed']:
            f.write(f"\nCommand: {failure['command']}\n")
            f.write(f"Description: {failure['description']}\n")
            f.write(f"Exit code: {failure.get('exit_code', 'N/A')}\n")
            f.write(f"Error: {failure.get('error', 'Unknown error')}\n")
    
    if test_results['errors']:
        f.write("\n\nTESTS WITH ERRORS:\n")
        f.write("-"*60 + "\n")
        for error in test_results['errors']:
            f.write(f"\nCommand: {error['command']}\n")
            f.write(f"Description: {error['description']}\n")
            f.write(f"Stdout: {error.get('stdout', 'N/A')}\n")
            f.write(f"Stderr: {error.get('stderr', 'N/A')}\n")

print(f"\nDetailed results saved to test_results.txt")