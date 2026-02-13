# Episodic 🧠

Episodic is a conversational memory system that helps avoid the usual
“I already told you that” problem with chatbots.

Instead of stuffing the prompt with lots of old messages, Episodic
tracks topics, ignores irrelevant content, and injects only the most
relevant prior facts for what you’re talking about right now. It can
also link facts through relationships (multi-hop traversal) to include
context you did not explicitly repeat. In short, Episodic keeps long
chats coherent.

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
- **⏱️ Assistant Mode** - Timers, alarms, reminders, weather, news, calculator, notes, media playback, calendar, and email — all available via slash commands or natural voice
- **🔄 Local & Cloud Flexibility** - Easily switch between local (free, private) and cloud-based operation
- **📓 Markdown Import/Export** - Save and resume conversations anytime
- **📎 File References (@file)** - Attach local files directly in chat messages
- **💰 Cost Tracking** - Real-time token usage and costs across all providers
- **🔌 MCP Server & Client** - Hardened Model Context Protocol with 12-layer security pipeline: token auth with rotation, input sanitization, DLP, provenance tracking, canary detection, rate limiting, replay-attack prevention, and structured audit logging
- **🎨 Rich CLI** - Streaming responses, theme-based colors, tab completion

If you are here regarding the paper *When F1 Fails: Granularity-Aware Evaluation for Dialogue Topic Segmentation* ([arXiv:2512.17083](https://arxiv.org/abs/2512.17083)), see the [`paper/`](paper/) directory.

## 🎬 Basic Example

This is one example of what Episodic does that a vanilla LLM cannot. The user
has a working conversation, moves on to unrelated topics for hours, then
comes back. The system restores the prior discussion thread and fills in
entity relationships from the knowledge graph — the user never repeats
anything.

```text
# A working conversation about a paper:
> I'm working with Dr. Reyes on a paper about coastal erosion modeling.
> We're submitting to AAAI. Deadline is March 15.
> Should we use the NOAA tidal dataset or the EU Copernicus data?
🤖 NOAA has better temporal resolution for the Atlantic coast sites
   you're modeling. Copernicus has broader coverage but coarser
   granularity. For site-specific erosion rates, NOAA is stronger.
> Let's go with NOAA.

# ... hours of conversation on other topics ...
# The original messages have left the context window.

# The user returns to the topic:
> Where did we land on the dataset for the paper?

📎 Pulled 6 earlier messages about: coastal-erosion-paper

🤖 You decided on the NOAA tidal dataset over EU Copernicus. The
   reasoning was that NOAA has better temporal resolution for the
   Atlantic coast sites. I had suggested downloading the hourly
   tide gauge records for your three stations first.

# That answer came from the restored conversation thread — the original
# messages were retrieved from the database and placed back into context.

# Now the user asks for a task:
> Send Dr. Reyes a reminder about the submission deadline.
🤖 Drafting a message to Dr. Reyes:

   "Hi Dr. Reyes — just a heads-up that the AAAI submission deadline
    for our coastal erosion modeling paper is March 15. Let me know
    if you need anything from my end before then."

   [Send via MCP → email tool]

# The user said eight words. Episodic filled in who, what, and when:

> /kg explain
🔗 Injected 4 edges (71 tokens):
   user:self → works_on → coastal erosion modeling paper (seed: user:self)
   user:self → related_to → Dr. Reyes (seed: user:self)
   Dr. Reyes → works_on → coastal erosion modeling paper (seed: Dr. Reyes)
   coastal erosion modeling paper → deadline → AAAI March 15 (seed: paper)

# A vanilla LLM that lost the earlier conversation would not know who
# Dr. Reyes is, what paper you're working on, or when it's due.
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
- **[Assistant](docs/assistant.md)** - Timers, alarms, weather, calendar, email
- **[MCP Guide](docs/mcp-guide.md)** - MCP server and client setup

## 🧪 Testing

The test suite covers the CLI, topic segmentation, knowledge graph (extraction, read-side, closure, ablation evaluation), RAG, persistence, determinism, and integration.

```bash
pytest                        # full suite (~3200 tests)
pytest tests/kg/               # knowledge graph tests only
pytest tests/unit/             # unit tests only
pytest -x --tb=short          # stop on first failure
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
