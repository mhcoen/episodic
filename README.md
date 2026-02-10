# Episodic 🧠

Episodic is a conversational memory system that makes talking to an LLM feel like talking to someone who remembers what you said and can reason about it. Two mechanisms run continuously during conversation to make this possible. Neural topic segmentation organizes the conversation into a navigable graph, compresses inactive segments, and restores them when subjects recur. A knowledge graph extracts structured facts from every message in real time, performs multi-hop inference over them, and injects relevant results back into the context window with no additional LLM calls.

- **Simple mode** lets users chat and search the web without touching any configuration. It handles topic detection, subject-change boundaries, conversation records, and summaries automatically. Conversations are stored as plain markdown files that Episodic can both read and write.

- **Advanced mode** exposes the full system: multi-model orchestration, RAG, semantic detection models, prompt engineering, performance benchmarking, cost analysis, and fine-grained control over every pipeline stage.

Simple mode uses free services by default and is self-contained. Advanced mode is available whenever you want it.

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

This example shows what Episodic does that a vanilla LLM cannot. The user mentions facts in separate turns, talks about other things long enough for the original topic to be compressed out of context, and then asks a question that requires connecting those earlier facts.

```text
# Early in the conversation, two facts are stated separately:
> My daughter Emma is starting at MIT this fall.
🤖 That's exciting! MIT is a great choice...

> We got Emma a MacBook Pro M3 Max with 64GB of RAM.
🤖 The M3 Max is a powerful machine...

# ... hours of conversation on other topics ...
# The original messages have been compressed out of the context window.
# A standard LLM has lost both facts entirely.

> Can my daughter run local models on her laptop?
🤖 Yes. With 64GB of unified memory, Emma's MacBook Pro M3 Max can run
   models up to ~30B parameters using Ollama or llama.cpp.

# The original messages were compressed long ago.
# Episodic reconstructed the answer from its knowledge graph:

> /kg explain
🔗 Injected 4 edges (83 tokens):
   user:self → related_to → Emma (seed: user:self)
   Emma → has → MacBook Pro M3 Max (seed: Emma)
   MacBook Pro M3 Max → has → 64GB RAM (derived: DEVICE_SPEC)
   Emma → located_at → MIT (derived: KINSHIP_LOCATION)
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
