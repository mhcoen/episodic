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
| `/topics reanalyze` | Re-detect topics using full context |
| `/muse` | Enable web search mode |
| `/reflect` | Enable multi-step reflection |
| `/memory` | Search conversation memories |
| `/new` | Start new conversation branch |
| `/clear` | Clear conversation context |
| `/help` | Show all commands |

### Knowledge & Search
| Command | Short | Description |
|---------|-------|-------------|
| `/rag on` | | Enable knowledge base |
| `/rag off` | | Disable knowledge base |
| `/rag stats` | | Show RAG statistics |
| `/index file.txt` | `/i` | Index a document |
| `/search query` | `/s` | Search knowledge base |
| `/muse` | | Switch to web search mode |
| `/chat` | | Switch to chat mode |

### Voice Mode
| Command | Description |
|---------|-------------|
| `/voice` | Toggle voice mode |
| `/voice on` | Enable voice mode |
| `/voice off` | Disable voice mode |
| `/voice status` | Show voice status and providers |
| `/voice stt` | Show/configure STT provider |
| `/voice tts` | Show/configure TTS provider |
| `/voice info` | Show audio devices, test microphone |

### Configuration
| Command | Description |
|---------|-------------|
| `/set` | Show common settings |
| `/set all` | Show all settings |
| `/set param value` | Change setting |
| `/mset` | Show model parameters |
| `/mset chat.temp 0.7` | Set model parameter |
| `/style standard` | Set response style (concise/standard/comprehensive/custom) |
| `/format mixed` | Set response format (paragraph/bulleted/mixed/academic) |
| `/theme` | Change color theme |
| `/detail moderate` | Set detail level |
| `/reset all` | Reset to defaults |
| `/migrate` | Database migration |

### Common Settings
```bash
/set stream true              # Streaming responses
/set debug true               # Debug mode
/set cost true                # Show costs
/set topics true              # Show topic info
/set color-mode basic         # Switch color mode
/debug on topic               # Debug topic detection
/debug on memory              # Debug memory system
```

### Web Search Settings
```bash
# Provider configuration
/set web.providers duckduckgo         # Single provider
/set web.providers google,duckduckgo  # Fallback order (first is primary)
/set web.fallback true                # Enable fallback
/set web.cache 3600                   # Cache 1 hour
/set web.max_results 5                # Results to fetch
```

### Navigation
| Command | Description |
|---------|-------------|
| `/list` | Recent messages |
| `/show id` | Show specific node |
| `/copy` | Copy last response to clipboard |
| `/ancestry id` | Show conversation thread |

### Scripts & Automation
| Command | Description |
|---------|-------------|
| `/script file.txt` | Execute commands from file |
| `/save session-name` | Save commands to script |

## Quick Setup Examples

### Research Assistant
```bash
/rag on
/muse
/set web-auto true
```

### Offline Mode
```bash
/model chat ollama/llama3
/model detection ollama/llama3
/rag off
/chat
```

### Long Conversations
```bash
/set topic-auto true
/set comp-auto true
/set topics true
```

### Voice Mode
```bash
/voice on                 # Enable voice mode
/voice info               # Check audio devices
/set voice-stt-provider local_whisper  # Free local STT
/set voice-tts-provider local_piper    # Free local TTS
```

### Parameter Profiles via Scripts
```bash
# Save current settings
/save dev-profile

# Load different profiles
/script scripts/prod-profile.txt
/script scripts/debug-profile.txt
```

## Tips
- Use Tab for command completion
- Use ↑/↓ for command history
- Node IDs are 2 characters (e.g., `a1`, `b2`)
- Settings persist in the database
- Use `/muse` for Perplexity-like web search mode
- Use `/memory` to search past conversations
- Use `/theme` to customize the interface appearance
- Use `/dev` for advanced debugging and inspection