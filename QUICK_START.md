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

# Episodic automatically detects available providers and uses them
# With Hugging Face: Uses compatible models from their inference API
# With OpenAI: Uses gpt-4o-mini by default
# With Ollama: Uses your local models
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

To make settings persist, edit `~/.episodic/config.json` directly, or copy settings from `/config` output.

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
ollama pull phi4          # Best all-around (14B parameters, ~9GB)
ollama pull qwen2.5:7b    # Fast and efficient (7B parameters, ~5GB)
```

### Configure Episodic for Ollama

```bash
# In Episodic, switch to local mode
> /mode local

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
- `/doctor` - Verify installation health
- `/muse` - Enable web search mode
- `/chat` - Return to normal chat
- `/style concise` - Make responses brief
- `/model` - See current models
- `/exit` - Leave Episodic

### Conversation Management
- `/list` - Show recent messages
- `/topics` - View conversation topics
- `/save` - Export conversation to markdown
- `/load <file>` - Import previous conversation

### Cost Tracking
- `/set show-cost true` - Display token usage
- Hugging Face free tier: ~30,000 tokens/month
- Brave free tier: 2,000 searches/month

## Tips for Best Experience

1. **Start with `/style concise`** - Makes responses faster and more focused
2. **Use `/muse` for current events** - Gets real-time information from the web
3. **Export important conversations** - Use `/save` to export as markdown
4. **Monitor usage** - Enable `/set show-cost true` to track token usage

## Troubleshooting

### Verify Your Installation

Run the health check to verify everything is working:
```bash
> /doctor
```
This checks Python version, dependencies, database, API keys, and optional features.

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

## 8. (Optional) Enable Voice Mode

Voice mode allows hands-free conversation with speech input and text-to-speech output.

### Basic Voice Mode (No Setup Required)

```bash
# Enable voice mode
> /voice on

# Speak into your microphone - Episodic will transcribe and respond
# Say "exit voice" to disable
```

### Enable Wake Word Detection

Voice mode can listen for a wake word (like "computer") so you can leave it running hands-free. Wake word detection can run locally, but for faster and lower-latency detection, you can use a free Picovoice account.

**Get Your Free Access Key (2 minutes):**
1. Go to https://console.picovoice.ai/
2. Create a free account (no credit card required)
3. Copy your Access Key from the dashboard

**Configure in Episodic:**
```bash
# Set your access key
> /set porcupine-access-key your_access_key_here

# Enable voice mode
> /voice on

# After 60 seconds of silence, voice mode enters idle state
💤 Idle - say "computer" to wake

# Say "computer" to wake it up
🎤 Listening...
```

**Available Wake Words:**
`computer`, `jarvis`, `alexa`, `hey google`, `ok google`, `hey siri`, `picovoice`, `porcupine`, `bumblebee`, `terminator`

**Change the Wake Word:**
```bash
> /set voice-wake-word jarvis
```

**Voice Commands to Enter Idle:**
- "Go to sleep"
- "Stop listening"
- "Standby"

The wake word detection runs 100% locally with minimal CPU (~1-2%) and works offline.

## Next Steps

- Read the [User Guide](USER_GUIDE.md) for advanced features
- Try more [Ollama models](https://ollama.com/library) - Llama 3.1, Gemma 2, Qwen, and more
- Configure [multiple LLM providers](docs/models-configuration.md) for flexibility
- Enable RAG to chat with your documents: `/rag on` then `/index <file>`

## Getting Help

- `/help <question>` - Search built-in documentation
- GitHub Issues: https://github.com/mhcoen/episodic/issues
- `/help all` - See all available commands

---

**Welcome to Episodic!** You're now ready to have persistent, intelligent conversations with memory that lasts across sessions. Enjoy exploring!