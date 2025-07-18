# Episodic Quick Start Guide

Get up and running with Episodic in 5 minutes using free services.

## Choose Your Setup

1. **Hugging Face** (Recommended) - Free API, no credit card, works everywhere
2. **Ollama** - Local models, unlimited use, requires 8GB+ RAM
3. **Both** - Use Hugging Face for quality, Ollama for unlimited usage

This guide covers all options!

## Prerequisites

- Python 3.8+ installed
- Git installed
- (For Ollama) 8GB+ RAM

## 1. Install Episodic

```bash
# Clone the repository
git clone https://github.com/mhcoen/episodic.git
cd episodic

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## 2. Set Up Free LLM Access with Hugging Face

Hugging Face offers a generous free tier perfect for getting started.

### Get Your Free API Key

1. Go to https://huggingface.co/settings/tokens
2. Create a free account (no credit card required)
3. Click "New token"
4. Name it "Episodic" and click "Generate"
5. Copy the token (starts with `hf_...`)

### Configure Episodic

```bash
# Set your API key
export HUGGINGFACE_API_KEY="hf_your_token_here"

# Start Episodic
python -m episodic
```

## 3. Your First Conversation

```bash
# Just start chatting!
> Hello! What can you help me with?

# Episodic automatically uses Hugging Face models for:
# - Chat: gpt-4o-mini (or your configured model)
# - Background tasks: Falcon-7B-Instruct (free tier compatible)
```

## 4. Enable Web Search (No API Key Required!)

Episodic includes DuckDuckGo search by default - completely free, no setup needed.

```bash
# Enable muse mode (web-enhanced responses)
> /muse

# Now all your questions search the web automatically
> What are the latest AI breakthroughs in 2025?
🔍 Searching with DuckDuckGo...
✨ [Synthesized response with current information]

# Return to normal chat
> /chat
```

## 5. Optimize for Better Performance

### Make Responses Concise

```bash
# Set concise style (recommended for faster, focused responses)
> /style concise

# This makes all responses brief and to-the-point
```

### Save Your Settings

```bash
# Save current configuration for next time
> /save config
✓ Configuration saved to: ~/.episodic/config.json
```

## 6. (Alternative) Use Ollama for Unlimited Free Local Models

If you want completely free, unlimited usage, Ollama runs models locally on your computer.

### Install Ollama (2 minutes)

**Mac/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from https://ollama.com/download/windows

### Download Models

```bash
# Download recommended models (runs in terminal, not Episodic)
ollama pull llama3        # Best all-around (8B parameters, 4.7GB)
ollama pull phi3          # Fast and efficient (3.8B parameters, 2.3GB)
ollama pull mistral       # Good for reasoning (7B parameters, 4.1GB)
```

### Configure Episodic for Ollama

```bash
# In Episodic, set all contexts to use local models
> /model chat ollama/llama3
> /model detection ollama/phi3
> /model compression ollama/mistral
> /model synthesis ollama/phi3

# Now everything runs locally - no API limits!
```

### Pros and Cons

**Pros:**
- ✅ Completely free, unlimited usage
- ✅ Private - nothing leaves your computer
- ✅ No internet required after download
- ✅ Fast responses (no network latency)

**Cons:**
- ⚠️ Requires 8GB+ RAM
- ⚠️ Models are 2-8GB downloads
- ⚠️ Lower quality than GPT-4 or Claude
- ⚠️ Can be slower on older computers

**Learn more:** 
- [Ollama Model Library](https://ollama.com/library) - Browse available models
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/README.md)

## 7. (Optional) Upgrade to Brave Search

While DuckDuckGo works great, Brave Search offers better results with a generous free tier:

### Get Brave API Key (Free)

1. Go to https://api.search.brave.com/
2. Sign up for free account
3. Free tier includes 2,000 queries/month
4. Copy your API key

### Configure Brave

```bash
# Set your Brave API key
export BRAVE_API_KEY="your_brave_key_here"

# Tell Episodic to use Brave
> /web provider brave
✓ Web search provider set to: Brave

# Brave will now be your primary search, with DuckDuckGo as fallback
```

## Key Commands Reference

### Essential Commands
- `/help` - Show available commands
- `/muse` - Enable web search mode
- `/chat` - Return to normal chat
- `/style concise` - Make responses brief
- `/model` - See current models
- `/exit` - Leave Episodic

### Conversation Management
- `/list` - Show recent messages
- `/topics` - View conversation topics
- `/out` - Export conversation to markdown
- `/in <file>` - Import previous conversation

### Cost Tracking
- `/set show_cost true` - Display token usage
- Hugging Face free tier: ~30,000 tokens/month
- Brave free tier: 2,000 searches/month

## Tips for Best Experience

1. **Start with `/style concise`** - Makes responses faster and more focused
2. **Use `/muse` for current events** - Gets real-time information from the web
3. **Export important conversations** - Use `/out` to save as markdown
4. **Monitor usage** - Enable `/set show_cost true` to track token usage

## Troubleshooting

### "API key not found"
Make sure you exported the environment variable:
```bash
export HUGGINGFACE_API_KEY="hf_..."  # Linux/Mac
set HUGGINGFACE_API_KEY=hf_...       # Windows
```

### "Rate limit exceeded"
- Hugging Face free tier has limits (~30,000 tokens/month)
- Consider adding OpenAI key for higher limits
- Or use local models with Ollama (see Advanced Setup)

### Web search not working
- DuckDuckGo should work immediately (no setup needed)
- If using Brave, check your API key is set correctly
- Try `/web provider duckduckgo` to switch back

## Next Steps

- Read the [User Guide](USER_GUIDE.md) for advanced features
- Try more [Ollama models](https://ollama.com/library) - Llama 3.1, Gemma 2, Qwen, and more
- Configure [multiple LLM providers](docs/multi-provider.md) for flexibility
- Enable [RAG](docs/rag-setup.md) to chat with your documents

## Getting Help

- `/help <question>` - Search built-in documentation
- GitHub Issues: https://github.com/mhcoen/episodic/issues
- `/help all` - See all available commands

---

**Welcome to Episodic!** You're now ready to have persistent, intelligent conversations with memory that lasts across sessions. Enjoy exploring!