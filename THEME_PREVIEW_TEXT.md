# Episodic Theme Preview Text

```python
def get_theme_preview_text() -> List[Tuple[str, str]]:
    """Return text that showcases all theme elements using actual Episodic UI."""
    return [
        # Header
        ("═" * 40, "separator"),
        ("          THEME PREVIEW", "heading"),
        ("═" * 40, "separator"),
        ("", ""),
        
        # Conversation with actual prompts
        ("> What's the capital of France?", "prompt"),
        ("", ""),
        ("The capital of France is Paris. It has", "assistant"),
        ("been the capital since 987 AD.", "assistant"),
        ("", ""),
        ("Key facts about Paris:", "assistant_heading"),
        ("• Population: 2.2 million", "assistant"),
        ("• Founded: 3rd century BC", "assistant"),
        ("• Known as the 'City of Light'", "assistant"),
        ("", ""),
        ("[Input: 8 tokens • Output: 47 tokens • 55 total • Cost: $0.0021]", "dim"),
        ("", ""),
        
        # Muse mode
        ("» What are today's AI news?", "prompt"),  # Muse prompt
        ("", ""),
        ("🔍 Searching web for: latest AI news today", "system"),
        ("📊 Found 8 relevant sources", "system"),
        ("✨ Based on current information:", "system_emphasis"),
        ("", ""),
        ("1. OpenAI announces GPT-5 preview", "assistant"),
        ("2. Google's Gemini 2.0 released", "assistant"),
        ("", ""),
        
        # System status messages (actual from Episodic)
        ("System Messages:", "heading"),
        ("  ✅ Current topic saved to: notes.md", "success"),
        ("  ⚠️  Rate limit: 45/50 requests", "warning"),
        ("  ❌ Error: File not found", "error"),
        ("  🤔 Thinking...", "system"),
        ("  🎭 Muse mode ENABLED", "system_emphasis"),
        ("  💡 Type /advanced for all commands", "dim"),
        ("", ""),
        
        # Topic detection (actual format)
        ("📌 New topic: python-debugging", "system_emphasis"),
        ("🔄 Semantic drift: 0.847 (High shift)", "system"),
        ("", ""),
        
        # Data display (actual format)
        ("Configuration:", "heading"),
        ("  Model:        gpt-4o-mini", "label/value"),
        ("  Provider:     OpenAI", "label/accent"),
        ("  Temperature:  0.7", "label/value"),
        ("  Cost:         $0.0234", "label/price"),
        ("  Tokens:       1,234", "label/metric"),
        ("", ""),
        
        # Commands (actual help format)
        ("Commands:", "help_heading"),
        ("  /save <name>     Save current topic", "command/argument/help_description"),
        ("  /muse            Enable web search", "command/help_description"),
        ("  /theme gemini    Switch to Gemini", "help_example"),
        ("", ""),
        
        # Mode indicators
        ("🔮 Muse mode active", "system_emphasis"),
        ("💰 Cost tracking enabled", "system"),
        ("🚀 Streaming responses", "system"),
        ("", ""),
        
        # Simple mode interface
        ("✨ Simple Mode", "heading"),
        ("Everything you need, nothing you don't.", "dim"),
        ("", ""),
        
        ("─" * 40, "separator"),
        ("Use ←/→ to navigate themes", "dim"),
    ]
```

## Compact Version with Real Episodic Elements

```python
def get_compact_theme_preview() -> str:
    """Minimal but authentic Episodic preview."""
    return """
╭─ Episodic Theme Preview ──────╮
│                               │
│ > Hello, Episodic!            │  # Normal prompt
│ Welcome! How can I help?      │  # Assistant
│                               │
│ » Search latest AI news       │  # Muse prompt
│ 🔍 Searching web...           │  # Search
│ ✨ Found 8 sources            │  # Synthesis
│                               │
│ ✅ Saved to: output.md        │  # Success
│ ⚠️  Rate limit warning        │  # Warning
│ ❌ Connection error           │  # Error
│ 🎭 Muse mode ENABLED          │  # Mode
│ 📌 New topic: ai-news         │  # Topic
│                               │
│ Model: gpt-4o • Cost: $0.03   │  # Data
│ /save • /muse • /theme        │  # Commands
│                               │
│ 💡 Type /help for commands    │  # Tip
╰───────────────────────────────╯
"""
```

## Interactive Theme Selector with Real UI

```python
def theme_selector_with_real_ui():
    """Theme selector using actual Episodic UI elements."""
    
    # Sample conversation that would actually appear
    sample = [
        ("Welcome to Episodic!", "heading"),
        ("Just start typing to chat or /help for commands.", "system"),
        ("💡 New to Episodic? Type /simple for a streamlined experience.", "dim"),
        ("", ""),
        ("> What are Python decorators?", "prompt"),
        ("🤔 Thinking...", "system"),  # This would disappear
        ("", ""),
        ("Python decorators are a powerful feature that allow you to", "assistant"),
        ("modify or enhance functions and classes. Here's how they work:", "assistant"),
        ("", ""),
        ("Basic Syntax:", "assistant_heading"),
        ("• @decorator_name above a function", "assistant"),
        ("• Wraps the original function", "assistant"),
        ("• Returns modified behavior", "assistant"),
        ("", ""),
        ("[Input: 5 tokens • Output: 89 tokens • 94 total • Cost: $0.0028]", "dim"),
        ("", ""),
        ("✅ Response complete (1.2s, 127 tokens)", "success"),
        ("", ""),
        ("> /save python-decorators", "prompt"),
        ("📁 Generated filename: python-decorators-2024-01-15.md", "system"),
        ("✅ Current topic saved to: python-decorators-2024-01-15.md", "success"),
        ("   Topic: Python Programming Fundamentals", "system"),
        ("   Size: 2.3 KB", "system"),
        ("   Load with: /load python-decorators-2024-01-15.md", "dim"),
        ("", ""),
        ("» explain async decorators", "prompt"),  # Muse mode prompt
        ("🔍 Searching for information about async decorators...", "system"),
        ("📊 Synthesizing from 5 sources...", "system"),
        ("", ""),
        ("─" * 40, "separator"),
        ("🎭 Muse mode • 🔮 RAG enabled • 💰 Cost: $0.0145", "system"),
    ]
    
    return sample
```

This preview text uses:
1. **Actual prompts**: `>` for normal, `»` for muse mode
2. **Real status messages**: Like "Current topic saved to:" and "Generated filename:"
3. **Actual icons**: 🤔, ✅, ⚠️, ❌, 🔍, 📊, ✨, 🎭, 🔮, 💡, 📌, 📁
4. **Real cost format**: `[Input: X tokens • Output: Y tokens • Z total • Cost: $N]`
5. **Authentic system messages**: Topic detection, mode indicators, etc.
6. **Actual command examples**: From the real help system

This will give users an accurate preview of how their actual Episodic sessions will look with each theme!