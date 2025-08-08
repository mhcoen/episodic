# GPT-5 Configuration Guide

This guide explains how to configure and use GPT-5's advanced parameters in Episodic.

## Overview

GPT-5 introduces two new configurable parameters that allow you to control the model's output characteristics:

1. **Verbosity Control** - Adjusts the length and detail of responses
2. **Reasoning Effort** - Controls the depth of reasoning applied

## Setting GPT-5 as Your Model

First, ensure GPT-5 is set as your chat model:

```bash
python -m episodic
# Then use the /model command:
/model chat gpt-5
```

## Verbosity Control

GPT-5's verbosity parameter has three levels:

### Low Verbosity
Best for concise answers, simple code generation, and SQL queries.

```bash
# In an Episodic session:
/set main.verbosity low
```

### Medium Verbosity (Default)
The standard setting used by previous models. Balanced responses.

```bash
# In an Episodic session:
/set main.verbosity medium
```

### High Verbosity
Ideal for thorough explanations, extensive code refactoring, and detailed analysis.

```bash
# In an Episodic session:
/set main.verbosity high
```

## Reasoning Effort Levels

GPT-5 offers four reasoning effort levels:

### Minimal
Produces very few reasoning tokens for fastest time-to-first-token.

```bash
# In an Episodic session:
/set main.reasoning_effort minimal
```

### Low
Favors speed and fewer tokens while maintaining quality.

```bash
# In an Episodic session:
/set main.reasoning_effort low
```

### Medium (Default)
The standard balance between speed and reasoning depth.

```bash
# In an Episodic session:
/set main.reasoning_effort medium
```

### High
Favors more thorough reasoning for complex problems.

```bash
# In an Episodic session:
/set main.reasoning_effort high
```

## Usage Examples

### Quick Code Generation
For rapid code generation with minimal explanation:

```bash
python -m episodic
# In the session:
/set main.verbosity low
/set main.reasoning_effort minimal
Write a Python function to sort a list
```

### Detailed Analysis
For comprehensive code reviews or architectural discussions:

```bash
python -m episodic
# In the session:
/set main.verbosity high
/set main.reasoning_effort high
Analyze the architecture of this codebase and suggest improvements
```

### Balanced Performance (Default)
For general conversation and standard tasks:

```bash
python -m episodic
# In the session:
/set main.verbosity medium
/set main.reasoning_effort medium
```

## Checking Current Settings

View your current GPT-5 parameter settings:

```bash
python -m episodic
# In the session:
/config show main_params

# Or check individual settings:
/config get main.verbosity
/config get main.reasoning_effort
```

## Important Notes

1. **Parameter Limitations**: GPT-5 has restricted parameter support:
   - Temperature: Only supports default value (1.0)
   - Stop sequences: Not supported
   - Top_p, presence_penalty, frequency_penalty: Not supported
   - Only `verbosity` and `reasoning_effort` are configurable

2. **Token Usage**: Lower verbosity settings reduce output tokens and overall latency while maintaining the same reasoning approach.

3. **Reasoning Tokens**: With GPT-5, invisible reasoning tokens count as output tokens. Higher reasoning effort levels will use more tokens.

4. **Model Compatibility**: Verbosity and reasoning_effort parameters only work with GPT-5. They are automatically filtered out when using other models.

4. **Cost Implications**: 
   - Input: $1.25 per 1M tokens
   - Output: $10.00 per 1M tokens (includes reasoning tokens)
   - Higher reasoning effort = more output tokens = higher cost

## Resetting to Defaults

To reset all main parameters to defaults:

```bash
python -m episodic
# In the session:
/set main.reset
```

This will restore:
- `verbosity`: medium
- `reasoning_effort`: medium
- All other main parameters to their defaults