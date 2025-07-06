# Episodic Quick Reference

## Starting Episodic
```bash
python -m episodic              # Interactive mode
python -m episodic -m gpt-4     # Start with specific model
python -m episodic -e script.txt # Execute script
```

## Essential Commands

### Conversation
Just type to chat! No prefix needed.

| Command | Description |
|---------|-------------|
| `/model` | Select AI model |
| `/cost` | Show token usage |
| `/topics` | List conversation topics |
| `/help` | Show all commands |

### Knowledge & Search
| Command | Short | Description |
|---------|-------|-------------|
| `/rag on` | | Enable knowledge base |
| `/index file.txt` | `/i` | Index a document |
| `/search query` | `/s` | Search knowledge base |
| `/websearch query` | `/ws` | Search the web |

### Configuration
| Command | Description |
|---------|-------------|
| `/set` | Show all settings |
| `/set param value` | Change setting |
| `/reset all` | Reset to defaults |

### Common Settings
```bash
/set stream true              # Streaming responses
/set debug true               # Debug mode
/set show_cost true           # Show costs
/set auto_compress_topics true # Auto-compress
```

### Navigation
| Command | Description |
|---------|-------------|
| `/list` | Recent messages |
| `/show id` | Show specific node |
| `/ancestry id` | Show conversation thread |

## Quick Setup Examples

### Research Assistant
```bash
/rag on
/websearch on
/set web_search_auto_enhance true
```

### Offline Mode
```bash
/model ollama/llama3
/rag off
/websearch off
```

### Long Conversations
```bash
/set automatic_topic_detection true
/set auto_compress_topics true
/set show_topics true
```

## Tips
- Use Tab for command completion
- Use ↑/↓ for command history
- Node IDs are 2 characters (e.g., `a1`, `b2`)
- Settings are session-only unless saved