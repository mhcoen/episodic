"""
Voice provider pricing loader.

Loads pricing information from voice_pricing.json for STT and TTS providers.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

# Cache for loaded pricing data
_pricing_data: Optional[Dict] = None


def _load_pricing() -> Dict:
    """Load pricing data from JSON file."""
    global _pricing_data
    if _pricing_data is not None:
        return _pricing_data

    pricing_file = Path(__file__).parent / "voice_pricing.json"
    if not pricing_file.exists():
        # Return defaults if file doesn't exist
        _pricing_data = {
            "stt": {
                "openai_whisper": {"cost_per_minute": 0.006},
                "deepgram": {"cost_per_minute": 0.0043},
            },
            "tts": {
                "openai_tts": {"models": {"tts-1": 0.015, "tts-1-hd": 0.030}},
                "elevenlabs": {"cost_per_1k_chars": 0.11},
            },
        }
        return _pricing_data

    with open(pricing_file) as f:
        _pricing_data = json.load(f)
    return _pricing_data


def get_stt_cost_per_minute(provider: str) -> float:
    """Get STT cost per minute for a provider."""
    pricing = _load_pricing()
    provider_data = pricing.get("stt", {}).get(provider, {})
    return provider_data.get("cost_per_minute", 0.0)


def get_tts_cost_per_1k_chars(provider: str, model: Optional[str] = None) -> float:
    """Get TTS cost per 1000 characters for a provider."""
    pricing = _load_pricing()
    provider_data = pricing.get("tts", {}).get(provider, {})

    # OpenAI TTS has model-specific pricing
    if "models" in provider_data:
        models = provider_data["models"]
        if model and model in models:
            return models[model]
        # Return default model price
        default_model = provider_data.get("default_model", "tts-1")
        return models.get(default_model, 0.0)

    return provider_data.get("cost_per_1k_chars", 0.0)


def reload_pricing() -> None:
    """Force reload of pricing data from file."""
    global _pricing_data
    _pricing_data = None
    _load_pricing()
