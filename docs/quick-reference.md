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
| `/model` | Show current models |
| `/model list` | View available models |
| `/cost` | Show token usage |
| `/topics` | List conversation topics |
| `/muse` | Enable web search mode |
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
| `/set` | Show common settings |
| `/set --all` | Show all settings |
| `/set param value` | Change setting |
| `/mset` | Show model parameters |
| `/mset chat.temp 0.7` | Set model parameter |
| `/reset all` | Reset to defaults |

### Common Settings
```bash
/set stream true              # Streaming responses
/set debug true               # Debug mode
/set cost true                # Show costs
/set topics true              # Show topic info
/set color-mode light         # Switch color mode
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
/set web-auto true
```

### Offline Mode
```bash
/model chat ollama/llama3
/model detection ollama/llama3
/rag off
/websearch off
```

### Long Conversations
```bash
/set topic-auto true
/set comp-auto true
/set topics true
```

## Tips
- Use Tab for command completion
- Use ↑/↓ for command history
- Node IDs are 2 characters (e.g., `a1`, `b2`)
- Settings persist in the database
- Use `/muse` for Perplexity-like web search mode