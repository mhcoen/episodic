# Installation Guide

## Basic Installation

```bash
# Clone the repo and navigate to the directory
git clone https://github.com/yourusername/episodic.git
cd episodic
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running Episodic Commands

There are two ways to run Episodic commands:

### Before Installation (or without installation)

Use the Python module syntax:

```bash
python -m episodic <command> [arguments]
```

### After Installation

After installing the package with pip (as shown above), you can use the direct command:

```bash
episodic <command> [arguments]
```

This works because pip automatically creates executable wrapper scripts for the entry points defined in setup.py.

## LLM Integration Setup

To use LLM features with OpenAI (the default provider), you need to set your API key as an environment variable:

```bash
export OPENAI_API_KEY=your_api_key_here
```

For other providers, see the [LLM Providers](./LLMProviders.md) documentation.

## Installation for LiteLLM Support

If you've previously installed Episodic and are getting a "No module named 'litellm'" error, you need to reinstall the package to include the new dependencies:

```bash
pip install -e .
```

For Ollama support, install with:

```bash
pip install "litellm[ollama]"
```

## Migrating Existing Databases

If you have an existing database created with a previous version of Episodic, you can migrate it to use short IDs by running the following Python code:

```python
from episodic.db import migrate_to_short_ids

# Migrate existing nodes to use short IDs
count = migrate_to_short_ids()
print(f"Migrated {count} nodes to use short IDs")
```

This will add short IDs to all existing nodes in your database.
