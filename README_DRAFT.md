# Episodic 💬

**Your conversations, remembered.** A friendly AI chat interface that organizes your discussions into topics and lets you pick up where you left off.

## What is Episodic?

Think of Episodic as a smart notebook for your AI conversations. Unlike regular chat interfaces that forget everything when you close them, Episodic remembers your discussions, organizes them by topic, and lets you save and share them as simple text files.

**Perfect for:**
- 📝 Writers brainstorming ideas
- 🎓 Students researching topics
- 💻 Developers getting coding help
- 🤔 Anyone who wants their AI conversations organized

## Getting Started in 2 Minutes

### 1. Install Episodic
```bash
git clone https://github.com/mhcoen/episodic.git
cd episodic
pip install -e .
```

### 2. Add Your AI Key (Pick One)

**Option A: Free Start** 🆓
```bash
# Get a free key at https://huggingface.co/settings/tokens
export HUGGINGFACE_API_KEY="hf_..."
```

**Option B: Best Experience** ⭐
```bash
# Get a key at https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-..."
```

### 3. Start Chatting!
```bash
python -m episodic

# First time? Try simple mode!
> /simple
```

That's it! Just type to chat. Your conversations are automatically saved and organized.

## Simple Mode - Just the Essentials 🌟

New to Episodic? Start with simple mode for a cleaner, friendlier experience:

```text
> /simple
✨ Simple Mode - Everything you need, nothing you don't.

> Hello! Can you help me plan a trip to Paris?
🤖 I'd be happy to help you plan your Paris trip! Let me ask a few questions...

> /save paris-trip
💾 Saved current topic to paris-trip.md

> /files
📁 Your saved conversations:
   - paris-trip.md (today)
   - recipe-ideas.md (yesterday)
   - python-help.md (last week)
```

**Simple Mode Commands:**
- `/chat` - Normal conversation
- `/muse` - Web-enhanced answers (like Perplexity)
- `/new` - Start a fresh topic
- `/save` - Save current topic
- `/load` - Resume a conversation
- `/files` - See your saved chats
- `/style` - Change response length
- `/help` - See these commands
- `/exit` - Leave Episodic

Want more features? Type `/advanced` anytime.

## Real-World Examples

### Example 1: Planning & Research
```text
> I'm writing a sci-fi novel about Mars colonization. Help me brainstorm.
🤖 Exciting project! Let's explore some unique angles for your Mars colony...

[... long brainstorming session ...]

> /save mars-novel-ideas
💾 Saved to mars-novel-ideas.md

# Next day:
> /load mars-novel-ideas
📂 Resumed conversation from mars-novel-ideas.md

> Let's develop the hydroponics subplot we discussed
🤖 Great choice! Building on our earlier discussion about the greenhouse accident...
```

### Example 2: Learning & Problem Solving
```text
> /muse
✨ Muse mode - I'll search the web for current information

> What's the latest on the James Webb telescope discoveries?
🔍 Searching for recent JWST discoveries...
✨ Here are the latest exciting findings from JWST this month:

1. **Ancient Galaxy Discovery** (November 2024): JWST spotted galaxies 
   that formed just 300 million years after the Big Bang...

2. **Exoplanet Atmosphere** (Last week): Detailed analysis of K2-18b 
   shows potential signs of biological activity...

> Can you explain how the infrared sensors work?
🤖 [Continues with context from previous search results...]
```

### Example 3: Coding Assistant
```text
> I need help debugging a Python async function
🤖 I'll help you debug that async function. Can you share the code?

> [pastes code]
🤖 I see the issue. The problem is in your event loop handling...

> /style concise
✅ Responses will be shorter and more direct

> How do I fix the await statement?
🤖 Add `async` before the function definition: `async def fetch_data():`

> /save async-debug-session
💾 Saved debugging session for future reference
```

## Power Features (When You're Ready)

### 📚 Build Your Knowledge Base
```bash
# Index your documents
/index my-notes.md
/index research-paper.pdf

# Your AI automatically searches them during conversations
/set rag-auto true
```

### 🌐 Smart Web Search
```bash
# Choose your search engine
/web provider duckduckgo  # Privacy-focused (default)
/web provider google      # Most comprehensive
/web provider brave       # Good balance

# Adjust search detail
/set muse-detail moderate  # Or: minimal, detailed, maximum
```

### 🎯 Fine-Tune Responses
```bash
# Response style
/style concise          # Short and direct
/style comprehensive    # Detailed explanations
/format bullet-points   # Easy to scan
/format academic        # Formal writing

# Model selection
/model                  # See current models
/model chat gpt-4o      # Use GPT-4 for chat
/cost                   # Track your usage
```

## Frequently Asked Questions

**Q: How is this different from ChatGPT/Claude?**
A: Episodic saves everything locally, organizes conversations by topic, works with any AI provider, can search the web, and lets you export/import conversations as simple markdown files.

**Q: Does it work offline?**
A: Yes! Use Ollama for completely local operation:
```bash
ollama pull llama3
python -m episodic
# Episodic auto-detects Ollama
```

**Q: Where are my conversations stored?**
A: In `~/.episodic/` on your computer. You own your data.

**Q: Can I use multiple AI providers?**
A: Yes! Episodic supports 20+ providers. Mix and match for best results:
```bash
/model chat openai/gpt-4o              # Best reasoning
/model detection ollama/llama3         # Free, local
/model synthesis anthropic/claude-3    # Great writing
```

**Q: How do I share conversations?**
A: Save them as markdown files:
```bash
/save conversation-name    # Creates conversation-name.md
# Share the .md file - it's just formatted text!
```

## Tips for New Users

1. **Start Simple**: Use `/simple` mode until you're comfortable
2. **Save Often**: Use `/save` to create checkpoints you can return to
3. **Try Muse Mode**: Use `/muse` for questions needing current information
4. **Experiment**: Try different styles (`/style`) and formats (`/format`)
5. **Use Tab**: Press Tab after commands to see options

## Installation Options

### Quick Install (Recommended)
```bash
git clone https://github.com/mhcoen/episodic.git
cd episodic
pip install -e .
```

### System-Wide Install
```bash
pip install episodic
```

### Development Install
```bash
git clone https://github.com/mhcoen/episodic.git
cd episodic
pip install -e ".[dev]"
```

## Getting Help

- **In-App Help**: Type `/help` or `/help <topic>`
- **Quick Start Guide**: See [QUICK_START.md](QUICK_START.md)
- **Full Documentation**: Check the [docs/](docs/) folder
- **Issues**: [GitHub Issues](https://github.com/mhcoen/episodic/issues)

## Contributing

We welcome contributions! Episodic is designed to be hackable:
- Clean, modular codebase
- Each file under 600 lines
- Extensive inline documentation
- Easy to add new commands and features

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file.

---

**Remember**: Episodic is your personal AI conversation manager. Make it work the way you want! 🚀