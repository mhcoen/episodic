# Simple Mode Ideas for Episodic

This document captures all the ideas discussed for making Episodic more approachable through a "simple mode" interface.

## Core Concept

Create two distinct modes:
- **Simple Mode**: ChatGPT/Perplexity-like experience with just essential commands
- **Advanced Mode**: Full power user interface with 50+ commands

## Simple Mode Command Set (7 Commands)

```
/chat            - Normal conversation mode
/muse            - Web search mode (like Perplexity)
/save [name]     - Save conversation (optional custom name)
/load <file>     - Load conversation
/files           - List saved conversations
/help            - Show these commands
/exit            - Leave
```

## Key Design Decisions

### 1. Intuitive Commands
- Use `/save` and `/load` instead of `/out` and `/in` (more intuitive)
- Rename current `/save scripts` command to free up `/save` for conversations

### 2. Subtle Topic Notifications
Show topic changes to hint at sophistication without overwhelming:
```
> What's quantum entanglement?
[Response about quantum physics...]

> How do I make carbonara?
📌 New topic: Italian Cooking
[Response about pasta...]
```

### 3. Save Behavior
- Default: Save entire conversation (all topics)
- Optional: `/save current` to save just current topic
- Auto-generate filenames from topic + timestamp if no name provided

### 4. Hidden Complexity
In simple mode, hide:
- Model selection (auto-select best available)
- Topic management commands
- Cost tracking
- Configuration options
- RAG commands
- Provider selection
- Style/format options

## User Experience Enhancements

### First-Run Experience
```
Welcome to Episodic! 🧠

I'm like ChatGPT but I remember everything.
Want web search like Perplexity? Just type /muse

Type anything to start chatting...
```

### Tab Completion (Essential!)
```bash
> /sa[TAB]
/save    Save conversation

> /load [TAB]
project-notes.md    quantum-discussion.md    recipes.md
```

### Friendly Error Messages
```
> /load quantum
❌ Can't find "quantum". Did you mean "quantum-discussion.md"?

> /save very/long/path/name.md  
❌ Can't save there. Try just: /save quantum-notes
```

### Mode Switching
```
> /advanced
🔓 Advanced mode unlocked! Type /simple to go back.
Now you have 50+ commands available...
```

### Visual Mode Indicators
```
[Simple] > What's quantum computing?     # Mode indicator
[Muse] > Latest AI news?                 # Shows current mode
```

### Progressive Discovery
```
> /help
Essential commands:
  /chat  - Conversation mode
  /muse  - Web search mode  
  /save  - Save conversation
  /load  - Load conversation
  /files - List saved chats
  
💡 Did you know? Type /advanced for power features
```

### Smart Suggestions
```
> What's the weather in Paris?
💡 Tip: Try /muse for current information

> [After 50 messages]
💡 Getting long? Use /save to keep this conversation
```

## Auto-Behaviors in Simple Mode

- Auto-detect questions → suggest /muse
- Auto-compress old topics (invisible)
- Auto-name saves from topic
- Auto-format responses (concise style)
- Auto-fallback for web search providers
- Remember mode preference for next session

## GUI/App Window Ideas

### Option 1: Terminal-Based GUI
- Use `textual` or `rich` for beautiful terminal UI
- Pros: Works everywhere, keeps terminal benefits
- Cons: Still requires terminal to launch

### Option 2: Web-Based Interface (Recommended)
Local web server with modern UI:
- Opens in browser: http://localhost:7860
- Familiar chat interface like ChatGPT/Perplexity
- Works on any device (even tablets/phones)
- Easy drag-drop, formatting, images

### Option 3: Native Desktop App
- Electron + React/Vue (like VS Code)
- Tauri (lighter than Electron)
- PyWebView (Python native)

### Option 4: Progressive Enhancement
```bash
# Terminal mode (current)
python -m episodic

# GUI mode (new)
python -m episodic --gui

# Or dedicated launcher
episodic-app
```

## Web Interface Implementation Options

### 1. Use Existing Templates
- **Chatbot UI**: https://github.com/mckaywrigley/chatbot-ui (MIT licensed ChatGPT clone)
- **Open WebUI**: https://github.com/open-webui/open-webui (Beautiful multi-model interface)

### 2. Gradio (Easiest Python Option)
```python
import gradio as gr

demo = gr.ChatInterface(
    episodic_chat,
    title="Episodic",
    description="Chat with persistent memory",
    examples=["What's quantum computing?", "Latest AI news?"],
    theme=gr.themes.Soft()
)
demo.launch()
```

### 3. Streamlit
```python
import streamlit as st

st.title("Episodic")

# Sidebar for saved conversations
with st.sidebar:
    st.header("Conversations")
    selected = st.selectbox("Load:", ["New", "quantum-chat.md", "recipes.md"])

# Chat interface
if prompt := st.chat_input("Type your message"):
    st.chat_message("user").write(prompt)
    response = get_episodic_response(prompt)
    st.chat_message("assistant").write(response)
```

### 4. Custom Development Options
- Hire freelancer on Fiverr/Upwork (~$500-2000)
- Use Claude/GPT-4 to generate React/Vue code
- Use v0.dev to generate UI from descriptions

## Configuration for Simple Mode

```json
{
  "interface_mode": "simple",
  "show_tips": true,
  "auto_save_on_exit": true,
  "default_style": "concise",
  "hide_technical_messages": true
}
```

## Philosophy

Simple mode should feel like:
- **Gmail** not Thunderbird  
- **iPhone** not Android with all settings exposed
- **Notion** not Emacs

Everything works perfectly out of the box, with gentle hints that more is available when needed.

## Implementation Priority

1. Start with command-line simple mode (7 commands)
2. Add Gradio web interface for better UX
3. Consider native app once product-market fit confirmed

## Next Steps

1. Implement simple mode command filtering
2. Rename `/save scripts` to free up `/save`
3. Create mode switching logic
4. Add smart suggestions system
5. Build Gradio prototype
6. Test with non-technical users