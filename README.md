# Episodic 🧠

A conversational memory system that creates persistent, navigable conversations with Large Language Models (LLMs). Episodic automatically organizes conversations into topics, manages context windows, and provides tools for searching both local knowledge and the web.

- Episodic is unique in offering a straightforward *simple* mode, which lets users chat and intelligently search the web. This mode hides Episodic's complexity and configuration but lets users take advantage of its advanced capabilities. Simple mode handles all details of enabling chatting and searching the web. It automatically organizes conversations by topic, detects subject changes, and keeps conversational records and summaries. Conversations are easily accessible in common *markdown files*, which Episodic can both read and write.

- Episodic has also has an *advanced* mode, which is well suited to developers, academics, researchers, and anyone interested in experimenting with LLM-based applications. This unlocks a comprehensive suite of commands for multi-model orchestration, RAG, semantic detection models, prompt engineering, performance benchmarking, cost analyses, and fine-grained system control.

Users can happily remain entirely within the *simple* mode, which makes use of free systems to provide advanced capabilities. Researchers can use *advanced* mode to quickly gain access to rich computational tools and models of conversation.

## Motivation
I originally wrote this to fill a gap I couldn't find addressed elsewhere. It has since become my preferred daily interface and framework for both routine LLM use and developing new capabilities based on them.

## ✨ Features

- **🤖 Universal LLM Interface** - Works with OpenAI, Anthropic, Google, Ollama, 20+ providers, and custom local models
- **🔗 Knowledge Graph** - Extracts structured knowledge from conversations in real-time and injects relevant facts into context at zero read-side LLM cost
- **🧠 Intelligent Topic Detection** - Neural segmentation validated on academic benchmarks, with configurable granularity
- **🔄 Topic Reactivation** - Seamlessly resume previous topics with full context restoration
- **🎭 Muse Mode** - Perplexity-like web search with many providers (e.g., DuckDuckGo, Google, Brave, Searx)
- **📚 Knowledge Base (RAG)** - Index documents and search them during chats
- **🎙️ Voice Mode** - Hands-free speech input and text-to-speech output
- **🔄 Local & Cloud Flexibility** - Easily switch between local (free, private) and cloud-based operation
- **📓 Markdown Import/Export** - Save and resume conversations anytime
- **📎 File References (@file)** - Attach local files directly in chat messages
- **💰 Cost Tracking** - Real-time token usage and costs across all providers
- **🔌 MCP Server & Client** - Model Context Protocol server with token auth, traces, and cost limits; client mode for consuming external MCP tools
- **🎨 Rich CLI** - Streaming responses, theme-based colors, tab completion

If you are here regarding the paper *When F1 Fails: Granularity-Aware Evaluation for Dialogue Topic Segmentation* ([arXiv:2512.17083](https://arxiv.org/abs/2512.17083)), see the [`paper/`](paper/) directory.

## 🎬 Example Session

A single conversation demonstrating topic detection, knowledge graph context, web search, and save/resume:

```text
> My daughter Emma just got a MacBook Pro M3 Max with 64GB of RAM for MIT
🤖 That's a great setup for computer science at MIT! The M3 Max with 64GB
   will handle everything from compiling large projects to running local ML models.

> What keyboard should she get?
📌 Topic changed → peripherals-recommendation
🤖 For a Mac-focused setup at MIT, I'd recommend...

> Can she run local LLMs on it?
🤖 Yes — with 64GB of unified memory, Emma's MacBook Pro can comfortably run
   models up to ~30B parameters via Ollama or llama.cpp...

# The model answered using KG facts: Emma → has → MacBook Pro M3 Max → has → 64GB RAM
# No embedding search needed — entity mention detection + edge traversal

> /muse
✨ Muse mode activated!

> What local models run well on Apple Silicon with 64GB?
🔍 Searching web for: local LLM models Apple Silicon 64GB
📚 Found 8 relevant sources
✨ Based on current benchmarks, the best options for 64GB unified memory are...

> /topics
📚 Conversation Topics
  [1] ✓ emmas-macbook-setup
  [2] ○ peripherals-recommendation (ongoing)

> /save
✅ Saved to: exports/peripherals-recommendation-2026-02-09.md
```

## 🚀 Quick Start

📖 **New users: See [QUICK_START.md](QUICK_START.md) for a complete 5-minute setup guide using free services!**

```bash
git clone https://github.com/mhcoen/episodic.git
cd episodic
pip install -e .

# Set up at least one provider:
export OPENAI_API_KEY="sk-..."          # or
export HUGGINGFACE_API_KEY="hf_..."     # or
ollama pull phi4                         # fully local

python -m episodic
```

## 📖 Documentation

- **[Quick Start](QUICK_START.md)** - 5-minute setup guide
- **[User Guide](USER_GUIDE.md)** - Comprehensive guide
- **[Installation](docs/installation.md)** - Detailed setup instructions
- **[Features](docs/features.md)** - Feature documentation
- **[CLI Reference](docs/cli-reference.md)** - All commands
- **[Configuration](docs/configuration.md)** - Settings and options
- **[MCP Guide](docs/mcp-guide.md)** - MCP server and client setup

## 🧪 Testing

```bash
pytest
```

## 🤝 Contributing

Contributions are welcome! Fork, branch, commit, and submit a pull request.

Areas of interest: non-linear conversation trees, additional LLM/embedding/search providers, UI/UX improvements.

## 📄 License

Apache License 2.0 — see [LICENSE](LICENSE).

## 👤 Author

**Michael H. Coen**
Email: mhcoen@gmail.com | mhcoen@alum.mit.edu
GitHub: [@mhcoen](https://github.com/mhcoen)

---

*Episodic: AI that remembers the conversation.*
