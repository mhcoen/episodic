"""
Utilities for model information and type detection.
"""
import subprocess
import re
from typing import Dict, Tuple, Optional


def detect_model_type(model_name: str) -> str:
    """
    Detect if a model is a chat or instruct model based on its name.
    
    Returns: 'instruct', 'chat', 'base', 'both', or 'unknown'
    """
    model_lower = model_name.lower()
    
    # Special cases first (most specific)
    if 'claude' in model_lower or 'anthropic' in model_lower:
        return 'both'  # Anthropic models work as both chat and instruct
    if 'gpt-3.5-turbo-instruct' in model_lower:
        return 'instruct'
    
    # Instruct model patterns (expanded)
    instruct_patterns = [
        'instruct', 'instruction', '-inst', 
        'alpaca', 'vicuna', 'wizardlm', 'wizard',
        'phi3', 'phi-3', 'phi2', 'phi-2',
        'zephyr', 'openhermes', 'hermes',
        'solar', 'starling', 'openchat',
        'mistral:7b-instruct', 'mistral:instruct',
        'yi-', 'qwen', 'deepseek',
        'falcon3-', 'falcon-7b-instruct',
        'gemma:2b-instruct', 'gemma:instruct',
        'minichat', 'dolly', 'stablelm'
    ]
    
    # Chat model patterns (expanded)
    chat_patterns = [
        'chat', 'conversation', 'dialogue',
        'assistant', 'turbo', 'claude',
        'gpt-4', 'gpt-3.5', 'gpt4', 'gpt3',
        'llama-2-.*-chat', 'llama-3-.*-chat',
        'llama2:.*chat', 'llama3:.*chat',
        'mixtral', 'command', 'coral',
        'bard', 'gemini', 'palm'
    ]
    
    # Base/completion model patterns
    base_patterns = [
        ':base', '-base', 'base-',
        'completion', 'davinci', 'curie', 'babbage', 'ada',
        'bloom', 'gpt-neox', 'opt-',
        'pythia', 'galactica', 'flan-ul2'
    ]
    
    # Known model mappings (for common models)
    known_models = {
        # OpenAI models
        'gpt-4o': 'chat',
        'gpt-4o-mini': 'chat',
        'gpt-o3': 'chat',
        'gpt-4': 'chat',
        'gpt-4.5': 'chat',
        'gpt-3.5-turbo': 'chat',
        
        # Anthropic models
        'claude-opus-4': 'chat',
        'claude-3-opus': 'chat',
        'claude-3-sonnet': 'chat',
        'claude-3-haiku': 'chat',
        
        # Ollama models
        'llama3:latest': 'chat',
        'llama3:instruct': 'instruct',
        'llama3.1:latest': 'chat',
        'llama2:latest': 'chat',
        'mistral:latest': 'chat',
        'mistral:instruct': 'instruct',
        'codellama': 'base',
        'deepseek-r1': 'instruct',
        
        # Google models
        'gemini-pro': 'chat',
        'gemini-ultra': 'chat',
        
        # Groq/Together models
        'llama3-8b': 'chat',
        'llama3-70b': 'chat',
        'mixtral-8x7b': 'chat',
        'gemma-7b-it': 'instruct',
        
        # HuggingFace specific models
        'meta-llama-3-8b': 'chat',
        'meta-llama-3-70b': 'chat',
        'llama-2-7b-chat': 'chat',
        'qwen-3': 'chat',
        'mistral-small-3.1': 'chat',
        'deepseek-r1-0528': 'chat',
        'gemma-7b': 'chat'
    }
    
    # Check known models first
    for known, mtype in known_models.items():
        if known in model_lower:
            return mtype
    
    # Check patterns with regex support
    import re
    
    # Check instruct patterns
    for pattern in instruct_patterns:
        if pattern in model_lower:
            return 'instruct'
    
    # Check chat patterns with regex
    for pattern in chat_patterns:
        if '*' in pattern:
            # Convert glob to regex
            regex_pattern = pattern.replace('*', '.*')
            if re.search(regex_pattern, model_lower):
                return 'chat'
        elif pattern in model_lower:
            return 'chat'
            
    # Check base patterns
    for pattern in base_patterns:
        if pattern in model_lower:
            return 'base'
    
    # Default assumptions based on provider
    if model_lower.startswith('openai/') or model_lower.startswith('anthropic/'):
        return 'chat'  # Most OpenAI/Anthropic models are chat models
    elif model_lower.startswith('huggingface/'):
        # Most HF models in our list are instruct or base
        if any(x in model_lower for x in ['llama', 'mistral', 'falcon', 'yi-', 'qwen']):
            return 'instruct'
        else:
            return 'base'
    
    # Final fallback - if it's a small model, likely instruct
    param_match = re.search(r'(\d+)b(?:-|_|\s|$)', model_lower)
    if param_match:
        params = int(param_match.group(1))
        if params <= 13:  # Small models are often instruct-tuned
            return 'instruct'
    
    return 'chat'  # Default to chat instead of unknown


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
    model_type = detect_model_type(model_name)
    
    # Type indicator with color
    type_indicators = {
        'instruct': '[I]',  # Instruct
        'chat': '[C]',      # Chat
        'base': '[B]',      # Base/Completion
        'both': '[CI]',     # Chat & Instruct
        'unknown': '[?]'    # Unknown
    }
    
    type_indicator = type_indicators.get(model_type, '[?]')
    
    # Get technical info based on known models
    tech_info = ""
    model_lower = model_name.lower()
    
    # Known model parameters
    known_params = {
        # OpenAI models
        'gpt-4o': '200B+',
        'gpt-4o-mini': '8B',
        'gpt-3.5-turbo': '175B',
        'gpt-4': '175B+',
        
        # Anthropic models
        'claude-opus-4': '~1T',
        'claude-3-opus': '~1T',
        'claude-3-sonnet': '~300B',
        'claude-3-haiku': '~70B',
        
        # Llama models
        'llama-3-8b': '8B',
        'llama-3-70b': '70B',
        'llama-3.3-70b': '70B',
        'llama-2-7b': '7B',
        'llama-2-70b': '70B',
        
        # Mistral models
        'mistral-7b': '7B',
        'mixtral-8x7b': '8x7B',
        'mistral-small': '7B',
        
        # Others
        'phi-3': '3.8B',
        'phi-3.5': '3.8B',
        'gemma-7b': '7B',
        'qwen-2-72b': '72B',
        'yi-34b': '34B',
        'falcon-40b': '40B',
        'falcon-180b': '180B',
        'bloom': '176B',
        'gpt-neox': '20B',
        'deepseek-v3': '67B',
        'dbrx': '132B',
        'hermes-3-405b': '405B'
    }
    
    # Check for parameter match
    for model_key, params in known_params.items():
        if model_key in model_lower:
            tech_info = f"({params})"
            break
    
    # If Ollama and no params found, try to get from model name
    if provider == "ollama" and not tech_info:
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