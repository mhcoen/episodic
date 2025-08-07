# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## Basic Installation

```bash
# Clone the repo and navigate to the directory
git clone https://github.com/mhcoen/episodic.git
cd episodic

# Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install typer rich prompt-toolkit litellm chromadb sentence-transformers

# Install Episodic in development mode
pip install -e .
```

This will install all required dependencies, including:
- **typer**: Command-line interface framework
- **rich**: Beautiful terminal formatting
- **prompt_toolkit**: Interactive interface with tab completion
- **litellm**: Unified interface for 20+ LLM providers
- **chromadb**: Vector database for RAG functionality
- **sentence-transformers**: Text embeddings for topic detection
- Other dependencies for visualization and database functionality

## Running Episodic

After installation, you can start Episodic with:

```bash
# Using the Python module syntax (recommended)
python -m episodic

# The database is automatically initialized at ~/.episodic/episodic.db on first run
```

This will start Episodic in chat mode, where you can:
- Chat directly with the LLM (no prefix needed)
- Use commands with the "/" prefix
- Enable tab completion for commands and parameters

```bash
> Hello, world!        # Chat with the LLM
🤖 Hello! How can I help you today?

> /help                # Show available commands
> /model list          # Show available models with pricing
> /muse                # Switch to web search mode
```

## LLM Provider Setup

Choose one or more providers based on your needs:

### Option 1: OpenAI (Recommended for quality)
```bash
export OPENAI_API_KEY="sk-..."
# Get your key at: https://platform.openai.com/api-keys
```

### Option 2: Anthropic (Advanced reasoning)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Get your key at: https://console.anthropic.com/settings/keys
```

### Option 3: Google Gemini (Good balance)
```bash
export GOOGLE_API_KEY="..."
# Get your key at: https://makersuite.google.com/app/apikey
```

### Option 4: Hugging Face (Free tier available)
```bash
export HUGGINGFACE_API_KEY="hf_..."
# Get your token at: https://huggingface.co/settings/tokens
```

### Option 5: Ollama (Fully local)
```bash
# Install Ollama from https://ollama.com
ollama pull llama3
ollama pull phi3  # For instruct tasks
```

For more providers and configuration options, see the [models configuration guide](./models-configuration.md).

## Optional Dependencies

### For PDF Support
```bash
pip install pypdf2
```

### For Web Interface
```bash
pip install flask plotly
```

### For Development
```bash
pip install pytest pytest-cov black flake8
```

## Database Location

Episodic stores all data in `~/.episodic/` by default:
- `~/.episodic/episodic.db` - Main conversation database
- `~/.episodic/rag/chroma/` - Vector database for RAG
- `~/.episodic/config.json` - User configuration
- `~/.episodic/models.json` - Model definitions

## Troubleshooting

### Missing typer module
```bash
pip install typer
```

### ChromaDB telemetry warnings
These are automatically suppressed but harmless if they appear.

### Database migration
The database schema is automatically migrated on startup using the migration system in `episodic/migrations/`.
