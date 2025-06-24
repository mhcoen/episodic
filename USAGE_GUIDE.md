# Episodic Usage Guide

A comprehensive guide to using Episodic, the conversational DAG-based memory agent.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Conversation Flow](#basic-conversation-flow)
3. [Navigation and History](#navigation-and-history)
4. [Topic Detection and Management](#topic-detection-and-management)
5. [Compression System](#compression-system)
6. [Model Configuration](#model-configuration)
7. [Advanced Features](#advanced-features)
8. [Scripting and Automation](#scripting-and-automation)
9. [Configuration Options](#configuration-options)
10. [Tips and Best Practices](#tips-and-best-practices)

## Getting Started

### Installation

```bash
# Install in development mode
pip install -e .

# Install required dependencies
pip install typer litellm
```

### First Run

```bash
# Start the CLI
python -m episodic

# Initialize the database (first time only)
> /init

# Start chatting
> Hello, how are you today?
```

## Basic Conversation Flow

### Talk Mode

When you start Episodic, you're in "talk mode" - simply type messages to chat with the LLM:

```
> Tell me about quantum computing
🤖 Quantum computing is a revolutionary approach...

> How do qubits work?
🤖 Qubits, or quantum bits, are the fundamental units...
```

### Commands

Commands start with `/`. Type `/help` to see all available commands:

- `/help` - Show available commands
- `/exit` or `/quit` - Exit the application
- `/last [N]` - Show recent messages
- `/topics` - Show conversation topics
- `/compress` - Compress conversation branches

## Navigation and History

### Viewing Conversation History

```bash
# Show last 5 messages (default)
> /last

# Show last 10 messages
> /last 10

# Show specific node details
> /show 0a

# View current position
> /head
```

### Navigating the Conversation DAG

Episodic stores conversations as a directed acyclic graph (DAG), allowing branching:

```bash
# Change to a different node
> /head 0f

# See the ancestry of current position
> /ancestry

# Visualize the entire conversation graph
> /visualize
```

## Topic Detection and Management

### Automatic Topic Detection

Episodic automatically detects when conversation topics change:

```
> Tell me about dogs
🤖 Dogs are domesticated mammals...

> What's the weather like?
🔄 Topic change detected (high confidence)
🤖 I don't have access to current weather...
```

### Viewing Topics

```bash
# Show recent topics
> /topics

# Show last 20 topics
> /topics 20

# Show all topics ever created
> /topics --all
```

### Topic Settings

```bash
# Enable topic detection
> /set topics true

# Set minimum messages before allowing topic change
> /set min_messages_before_topic_change 8

# Enable auto-compression of topics
> /set auto_compress_topics true
```

## Compression System

### Manual Compression

Compress conversation branches to save space and create summaries:

```bash
# Compress from root to current node
> /compress

# Compress specific branch
> /compress 0a

# Dry run to see what would be compressed
> /compress --dry-run

# Use different compression strategy
> /compress --strategy detailed
```

### Automatic Compression

When topic changes are detected, previous topics can be automatically compressed:

```bash
# Enable auto-compression
> /set auto_compress_topics true

# Set minimum nodes for compression
> /set compression_min_nodes 5

# Choose compression model
> /set compression_model ollama/llama3

# View compression queue
> /compression-queue

# View compression statistics
> /compression-stats
```

### Compression Strategies

- `simple` - Basic summary (default)
- `detailed` - More comprehensive summary
- `bullets` - Bullet-point format
- `structured` - Structured summary with sections
- `auto-topic` - Used by automatic compression

## Model Configuration

### Viewing Available Models

```bash
# Show current model
> /model

# List all available models
> /model
1. gpt-4
2. gpt-3.5-turbo
3. claude-3-opus
4. ollama/llama3
...
```

### Changing Models

```bash
# Select by number
> /model
Select a model: 3

# Or specify directly
> /model gpt-4
```

### Model Verification

```bash
# Test current model
> /verify
```

## Advanced Features

### Semantic Drift Detection

Track how conversations drift between topics:

```bash
# Enable drift display
> /set drift true

# Set semantic depth for context
> /set semdepth 5
```

### System Prompts

Customize the LLM's behavior with different prompts:

```bash
# List available prompts
> /prompts

# Use a specific prompt
> /prompts use creative

# Show current prompt
> /prompts show

# Create custom prompts
> /prompts create my-prompt
```

### Context Management

Control how much conversation history is included:

```bash
# Set context depth (number of messages)
> /set depth 10

# Enable prompt caching for performance
> /set cache true
```

## Scripting and Automation

### Running Scripts

Create text files with commands and messages to automate conversations:

```bash
# Run a script file
> /script tests/example-conversation.txt
```

### Script Format

```text
# example-script.txt
# Comments start with #

# Set configuration
/set debug true
/set topics true

# Have a conversation
Tell me about Python
What are decorators?
How do I handle exceptions?

# Change topic
Now let's talk about cooking
What's a good pasta recipe?

# Check results
/topics
/compression-stats
```

## Configuration Options

### Display Settings

```bash
# Show costs for each message
> /set cost true

# Enable debug mode
> /set debug true

# Control text wrapping
> /set wrap true

# Set color mode (auto/on/off)
> /set color auto
```

### Compression Settings

```bash
# Show compression notifications
> /set show_compression_notifications true

# Set compression model
> /set compression_model gpt-3.5-turbo

# Minimum nodes before compression
> /set compression_min_nodes 5
```

### Topic Detection Settings

```bash
# Enable topic detection
> /set topics true

# Minimum messages before topic change
> /set min_messages_before_topic_change 8
```

### View All Settings

```bash
# See current configuration
> /set
```

## Tips and Best Practices

### 1. **Use Clear Topic Transitions**
When changing subjects, be explicit:
- ❌ "What about cars?"
- ✅ "Let's switch topics to cars - what are electric vehicles?"

### 2. **Leverage Branching**
Create alternate conversation paths:
```bash
# Save current position
> /head

# Explore different response
> /head 0c
> What if we approached this differently?

# Return to saved position
> /head 0f
```

### 3. **Compress Regularly**
Keep your database manageable:
- Compress completed topics
- Use auto-compression for long conversations
- Review compression stats periodically

### 4. **Customize Prompts**
Create prompts for different use cases:
- Technical discussions
- Creative writing
- Code reviews
- Learning sessions

### 5. **Use Scripts for Testing**
Create reproducible conversations:
- Test topic detection
- Benchmark different models
- Demo specific features

### 6. **Monitor Costs**
Track token usage and costs:
```bash
> /set cost true
> /cost  # View session costs
```

### 7. **Optimize Performance**
- Use `/set cache true` for faster responses
- Choose appropriate context depth
- Use local models (Ollama) for privacy/speed

## Troubleshooting

### Common Issues

**Database locked:**
- Ensure only one instance is running
- Check file permissions

**Model errors:**
- Verify API keys are set
- Check model availability with `/model`
- Test with `/verify`

**Topic detection not working:**
- Ensure `/set topics true`
- Check minimum message threshold
- Try explicit topic changes

**Compression failing:**
- Verify sufficient nodes in topic
- Check compression model availability
- Review debug output

### Debug Mode

Enable debug mode for detailed information:
```bash
> /set debug true
```

This shows:
- Topic detection process
- Compression attempts
- Model queries
- Configuration changes

## Environment Variables

- `EPISODIC_DB_PATH` - Custom database location
- `OPENAI_API_KEY` - OpenAI API access
- `ANTHROPIC_API_KEY` - Anthropic API access
- Standard LiteLLM environment variables

## Support and Documentation

- **GitHub Issues**: Report bugs and request features
- **CLAUDE.md**: Development guidelines
- **docs/structure.md**: Codebase architecture
- **README.md**: Quick start guide