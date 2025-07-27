"""
OpenRouter pricing integration.

Fetches and caches pricing information from OpenRouter API.
"""

import json
import os
import time
from typing import Dict, Optional, Tuple
import requests

from episodic.config import config


class OpenRouterPricing:
    """Manages OpenRouter pricing information."""
    
    def __init__(self):
        self.cache_file = os.path.expanduser("~/.episodic/openrouter_pricing_cache.json")
        self.cache_duration = 3600  # 1 hour
        self._pricing_data = None
        self._last_fetch = 0
    
    def get_pricing(self, model_id: str) -> Optional[Tuple[float, float]]:
        """
        Get pricing for a specific OpenRouter model.
        
        Args:
            model_id: The model ID (e.g., "anthropic/claude-3-sonnet")
            
        Returns:
            Tuple of (input_cost_per_1k, output_cost_per_1k) or None if not found
        """
        # Ensure we have fresh data
        self._ensure_fresh_data()
        
        if not self._pricing_data:
            return None
        
        # Look up the model
        model_info = self._pricing_data.get(model_id)
        if not model_info:
            return None
        
        return (
            model_info.get('prompt', 0),
            model_info.get('completion', 0)
        )
    
    def get_all_pricing(self) -> Dict[str, Dict]:
        """Get all pricing data."""
        self._ensure_fresh_data()
        return self._pricing_data or {}
    
    def _ensure_fresh_data(self):
        """Ensure we have fresh pricing data."""
        current_time = time.time()
        
        # Check if we need to refresh
        if (self._pricing_data is None or 
            current_time - self._last_fetch > self.cache_duration):
            
            # Try to load from cache first
            if self._load_cache():
                return
            
            # Fetch fresh data
            self._fetch_pricing()
    
    def _load_cache(self) -> bool:
        """Load pricing from cache file."""
        if not os.path.exists(self.cache_file):
            return False
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is still valid
            if time.time() - cache_data.get('timestamp', 0) < self.cache_duration:
                self._pricing_data = cache_data.get('pricing', {})
                self._last_fetch = cache_data.get('timestamp', 0)
                return True
        except:
            pass
        
        return False
    
    def _save_cache(self):
        """Save pricing to cache file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            cache_data = {
                'timestamp': self._last_fetch,
                'pricing': self._pricing_data
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except:
            pass  # Caching is optional
    
    def _fetch_pricing(self):
        """Fetch pricing from OpenRouter API."""
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Process the data
                self._pricing_data = {}
                
                for model in data.get('data', []):
                    model_id = model.get('id', '')
                    pricing = model.get('pricing', {})
                    
                    # Convert to cost per 1K tokens
                    prompt_cost = float(pricing.get('prompt', 0)) * 1000
                    completion_cost = float(pricing.get('completion', 0)) * 1000
                    
                    self._pricing_data[model_id] = {
                        'prompt': prompt_cost,
                        'completion': completion_cost,
                        'name': model.get('name', model_id),
                        'context_length': model.get('context_length', 0)
                    }
                
                self._last_fetch = time.time()
                self._save_cache()
                
        except Exception as e:
            # Silently fail and use cached data if available
            pass


# Global instance
_openrouter_pricing = None


def get_openrouter_pricing() -> OpenRouterPricing:
    """Get the global OpenRouter pricing instance."""
    global _openrouter_pricing
    if _openrouter_pricing is None:
        _openrouter_pricing = OpenRouterPricing()
    return _openrouter_pricing