"""
Voice persona for Episodic voice mode.

Modifies LLM behavior for natural spoken responses.
"""

import os
import re

# Cache for loaded prompt
_voice_persona_prompt = None


def get_voice_system_prompt_addition() -> str:
    """Get the system prompt addition for voice mode from prompts/voice_persona.md."""
    global _voice_persona_prompt

    if _voice_persona_prompt is not None:
        return _voice_persona_prompt

    # Load from prompts directory
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
    prompt_file = os.path.join(prompts_dir, "voice_persona.md")

    try:
        with open(prompt_file, "r") as f:
            _voice_persona_prompt = f.read().strip()
    except OSError:
        # Fallback if file doesn't exist
        _voice_persona_prompt = (
            "You are in voice mode. Keep responses brief and conversational. "
            "Do not use markdown formatting, bullet points, or numbered lists."
        )

    return _voice_persona_prompt


def reload_voice_persona():
    """Force reload of the voice persona prompt from file."""
    global _voice_persona_prompt
    _voice_persona_prompt = None
    return get_voice_system_prompt_addition()


def clean_text_for_tts(text: str) -> str:
    """
    Clean text for TTS by removing markdown and formatting.

    Args:
        text: Raw text that may contain markdown

    Returns:
        Clean text suitable for speech synthesis
    """
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)

    # Remove headers (# ## ### etc)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_

    # Remove bullet points and list markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Remove links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove images ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)

    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Remove blockquotes
    text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)

    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Clean up multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()
