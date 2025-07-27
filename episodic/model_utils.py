"""
Utilities for model information and type detection.
"""
import subprocess
import re
from typing import Dict, Tuple, Optional, Any

from episodic.model_config import get_model_config


def get_models_config() -> Dict[str, Any]:
    """
    Get the raw models configuration data.
    
    Returns the complete models.json data structure.
    """
    model_config = get_model_config()
    return model_config._models_data


def detect_model_type(model_name: str) -> str:
    """
    Detect if a model is a chat or instruct model based on its name.
    
    Returns: 'instruct', 'chat', 'base', 'both', or 'unknown'
    """
    # Use model config for detection
    model_config = get_model_config()
    return model_config.detect_model_type(model_name)


def get_ollama_model_info(model_name: str) -> Dict[str, Optional[str]]:
    """
    Get size and parameter information for Ollama models.
    
    Returns dict with 'size', 'parameters', and 'quantization' keys.
    """
    info = {
        'size': None,
        'parameters': None,
        'quantization': None
    }
    
    # Extract base name without ollama/ prefix
    if model_name.startswith('ollama/'):
        model_name = model_name[7:]
    
    try:
        # Run ollama list to get size info
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if model_name in line:
                    # Parse the line format: NAME ID SIZE MODIFIED
                    parts = line.split()
                    if len(parts) >= 3:
                        # Extract size (e.g., "4.7 GB")
                        for i, part in enumerate(parts):
                            if part in ['GB', 'MB', 'KB']:
                                if i > 0:
                                    info['size'] = f"{parts[i-1]} {part}"
                                break
                    break
        
        # Try to extract parameter count from model name
        param_match = re.search(r'(\d+)b(?:-|_|\s|$)', model_name, re.IGNORECASE)
        if param_match:
            param_count = param_match.group(1)
            info['parameters'] = f"{param_count}B"
        
        # Check for quantization info
        quant_patterns = {
            'q4_0': 'Q4_0 (4-bit)',
            'q4_1': 'Q4_1 (4-bit)',
            'q5_0': 'Q5_0 (5-bit)',
            'q5_1': 'Q5_1 (5-bit)',
            'q8_0': 'Q8_0 (8-bit)',
            'f16': 'F16 (16-bit)',
            'f32': 'F32 (32-bit)',
            'gguf': 'GGUF',
            'ggml': 'GGML'
        }
        
        model_lower = model_name.lower()
        for pattern, desc in quant_patterns.items():
            if pattern in model_lower:
                info['quantization'] = desc
                break
                
    except Exception:
        pass
    
    return info


def get_model_info_string(model_name: str, provider: str) -> Tuple[str, str]:
    """
    Get a formatted string with model type and technical info.
    
    Returns: (type_indicator, tech_info)
    """
    model_config = get_model_config()
    
    # Get model type
    model_type = detect_model_type(model_name)
    type_indicator = model_config.get_type_indicator(model_type)
    
    # Get technical info (parameters)
    tech_info = ""
    
    # First try to get from model config
    params = model_config.get_model_parameters(provider, model_name)
    if params:
        tech_info = f"({params})"
    
    # If Ollama and no params found, try to get from runtime
    elif provider == "ollama":
        info = get_ollama_model_info(model_name)
        if info['parameters']:
            tech_info = f"({info['parameters']})"
    
    return type_indicator, tech_info


def format_model_display_name(display_name: str, max_length: int = 30) -> str:
    """
    Format display name to fit within max_length characters.
    """
    if len(display_name) <= max_length:
        return display_name.ljust(max_length)
    else:
        return display_name[:max_length-3] + "..."