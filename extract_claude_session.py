#!/usr/bin/env python3
"""Extract and display Claude Code session history in readable format."""

import json
import sys
from datetime import datetime
from pathlib import Path

def extract_session(session_file: Path):
    """Extract conversation from Claude Code session file."""

    messages = []

    with open(session_file) as f:
        for line in f:
            entry = json.loads(line)

            # Skip sidechains and system messages
            if entry.get('isSidechain'):
                continue

            msg_type = entry.get('type')
            timestamp = entry.get('timestamp', '')

            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    time_str = timestamp
            else:
                time_str = 'unknown'

            if msg_type == 'user':
                content = entry.get('message', {}).get('content', '')

                # Handle string content
                if isinstance(content, str):
                    messages.append({
                        'role': 'user',
                        'time': time_str,
                        'content': content
                    })
                # Handle array content (tool results)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            messages.append({
                                'role': 'user',
                                'time': time_str,
                                'content': item.get('text', '')
                            })

            elif msg_type == 'assistant':
                msg = entry.get('message', {})
                content = msg.get('content', [])

                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                        elif item.get('type') == 'tool_use':
                            tool_name = item.get('name', 'unknown')
                            text_parts.append(f'[Tool: {tool_name}]')

                if text_parts:
                    messages.append({
                        'role': 'assistant',
                        'time': time_str,
                        'content': ' '.join(text_parts)
                    })

    return messages

def print_conversation(messages, context_lines=5):
    """Print conversation with optional context around last messages."""

    print(f"\n{'='*80}")
    print(f"Total messages: {len(messages)}")
    print(f"{'='*80}\n")

    # Show all messages or just recent context
    if context_lines and len(messages) > context_lines * 2:
        print(f"[Showing first {context_lines} and last {context_lines} messages]\n")
        display_msgs = messages[:context_lines] + [{'role': 'system', 'time': '', 'content': f'... [{len(messages) - context_lines*2} messages omitted] ...'}] + messages[-context_lines:]
    else:
        display_msgs = messages

    for msg in display_msgs:
        role = msg['role'].upper()
        time = msg['time']
        content = msg['content'][:500]  # Truncate long messages

        if msg['role'] == 'system':
            print(f"\n{content}\n")
        else:
            print(f"\n[{time}] {role}:")
            print(f"{content}")
            if len(msg['content']) > 500:
                print(f"... [truncated, {len(msg['content'])} chars total]")
        print("-" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_claude_session.py <session_file.jsonl> [--all]")
        sys.exit(1)

    session_file = Path(sys.argv[1])
    show_all = '--all' in sys.argv

    if not session_file.exists():
        print(f"Error: {session_file} not found")
        sys.exit(1)

    messages = extract_session(session_file)
    print_conversation(messages, context_lines=None if show_all else 10)
