#!/usr/bin/env python3
"""
Update model pricing information from various sources.

This script fetches current pricing from:
- OpenRouter API (automatic)
- Anthropic website (manual/scraped)
- Other providers (from LiteLLM if available)

Updates the models.json file in the user's .episodic directory.
"""

import json
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import requests


def load_models_config(path: str) -> Dict[str, Any]:
    """Load the models.json configuration file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_models_config(path: str, data: Dict[str, Any]):
    """Save the models.json configuration file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Updated models configuration saved to: {path}")


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float]]:
    """
    Fetch OpenRouter pricing from their API.
    
    Returns:
        Dict mapping model IDs to (input_price, output_price) tuples
    """
    print("🔍 Fetching OpenRouter pricing from API...")
    
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        if response.status_code != 200:
            print(f"  ❌ Error: API returned status {response.status_code}")
            return {}
        
        data = response.json()
        pricing = {}
        
        for model in data.get('data', []):
            model_id = model.get('id', '')
            model_pricing = model.get('pricing', {})
            
            # Convert to cost per 1K tokens
            prompt_cost = float(model_pricing.get('prompt', 0)) * 1000
            completion_cost = float(model_pricing.get('completion', 0)) * 1000
            
            if prompt_cost > 0 or completion_cost > 0:
                pricing[model_id] = (prompt_cost, completion_cost)
        
        print(f"  ✅ Found pricing for {len(pricing)} OpenRouter models")
        return pricing
        
    except Exception as e:
        print(f"  ❌ Error fetching OpenRouter pricing: {e}")
        return {}


def fetch_anthropic_pricing() -> Dict[str, Dict[str, float]]:
    """
    Fetch Anthropic pricing from their website.
    
    Note: Since Anthropic doesn't provide a pricing API, this uses
    hardcoded values that should be updated manually when prices change.
    
    Returns:
        Dict mapping model names to pricing info
    """
    print("🔍 Checking Anthropic pricing...")
    
    # Known pricing as of 2025-07-27
    # TODO: Implement web scraping from https://www.anthropic.com/pricing
    pricing = {
        "claude-opus-4": {"input": 0.015, "output": 0.075},
        "claude-sonnet-4": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3.7-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3.5-haiku": {"input": 0.00025, "output": 0.00125},
    }
    
    print(f"  ℹ️  Using hardcoded pricing for {len(pricing)} Anthropic models")
    print(f"  💡 Check https://www.anthropic.com/pricing for updates")
    return pricing


def fetch_litellm_pricing(provider: str, model_name: str) -> Optional[Tuple[float, float]]:
    """
    Try to get pricing from LiteLLM for a specific model.
    
    Returns:
        Tuple of (input_cost, output_cost) per 1K tokens, or None if not found
    """
    try:
        from litellm import cost_per_token
        
        # Calculate cost for 1000 tokens
        input_cost = cost_per_token(model=model_name, prompt_tokens=1000, completion_tokens=0)
        output_cost = cost_per_token(model=model_name, prompt_tokens=0, completion_tokens=1000)
        
        # Handle tuple results
        if isinstance(input_cost, tuple):
            input_cost = sum(input_cost)
        if isinstance(output_cost, tuple):
            output_cost = sum(output_cost)
            
        if input_cost > 0 or output_cost > 0:
            return (input_cost, output_cost)
            
    except:
        pass
    
    return None


def update_openrouter_pricing(models_data: Dict[str, Any], dry_run: bool = False) -> int:
    """
    Update OpenRouter model pricing from their API.
    
    Returns:
        Number of models updated
    """
    print("\n📦 Updating OpenRouter models...")
    
    # Fetch current pricing from OpenRouter API
    or_pricing = fetch_openrouter_pricing()
    if not or_pricing:
        return 0
    
    updated = 0
    today = datetime.now().strftime("%Y-%m-%d")
    
    provider_data = models_data.get('providers', {}).get('openrouter')
    if not provider_data:
        print("  ⚠️  OpenRouter provider not found in models.json")
        return 0
    
    for model in provider_data.get('models', []):
        model_name = model.get('name', '')
        
        # Strip the openrouter/ prefix to match API
        model_id = model_name.replace('openrouter/', '')
        
        if model_id in or_pricing:
            input_price, output_price = or_pricing[model_id]
            current_pricing = model.get('pricing', {})
            
            new_pricing = {
                "input": input_price,
                "output": output_price,
                "unit": "per_1k_tokens",
                "last_updated": today
            }
            
            # Check if pricing changed
            if (current_pricing.get('input') != new_pricing['input'] or
                current_pricing.get('output') != new_pricing['output']):
                
                if dry_run:
                    print(f"  Would update {model['display_name']}:")
                    if current_pricing:
                        print(f"    Current: ${current_pricing.get('input', 0):.6f}/1K in, "
                              f"${current_pricing.get('output', 0):.6f}/1K out")
                    else:
                        print(f"    Current: No pricing")
                    print(f"    New:     ${new_pricing['input']:.6f}/1K in, "
                          f"${new_pricing['output']:.6f}/1K out")
                else:
                    model['pricing'] = new_pricing
                    print(f"  ✅ Updated {model['display_name']}")
                
                updated += 1
    
    return updated


def update_anthropic_pricing(models_data: Dict[str, Any], dry_run: bool = False) -> int:
    """
    Update Anthropic model pricing from hardcoded/scraped data.
    
    Returns:
        Number of models updated
    """
    print("\n🤖 Updating Anthropic models...")
    
    anthropic_pricing = fetch_anthropic_pricing()
    updated = 0
    today = datetime.now().strftime("%Y-%m-%d")
    
    provider_data = models_data.get('providers', {}).get('anthropic')
    if not provider_data:
        print("  ⚠️  Anthropic provider not found in models.json")
        return 0
    
    for model in provider_data.get('models', []):
        model_name = model.get('name', '')
        
        # Try to match the model with pricing data
        for pricing_key, pricing_info in anthropic_pricing.items():
            if pricing_key in model_name:
                current_pricing = model.get('pricing', {})
                new_pricing = {
                    "input": pricing_info['input'],
                    "output": pricing_info['output'],
                    "unit": "per_1k_tokens",
                    "last_updated": today
                }
                
                # Check if pricing changed
                if (current_pricing.get('input') != new_pricing['input'] or
                    current_pricing.get('output') != new_pricing['output']):
                    
                    if dry_run:
                        print(f"  Would update {model['display_name']}:")
                        if current_pricing:
                            print(f"    Current: ${current_pricing.get('input', 0):.6f}/1K in, "
                                  f"${current_pricing.get('output', 0):.6f}/1K out")
                        else:
                            print(f"    Current: No pricing")
                        print(f"    New:     ${new_pricing['input']:.6f}/1K in, "
                              f"${new_pricing['output']:.6f}/1K out")
                    else:
                        model['pricing'] = new_pricing
                        print(f"  ✅ Updated {model['display_name']}")
                    
                    updated += 1
                break
    
    return updated


def update_other_providers(models_data: Dict[str, Any], provider: str, dry_run: bool = False) -> int:
    """
    Update pricing for other providers using LiteLLM.
    
    Returns:
        Number of models updated
    """
    print(f"\n🔍 Updating {provider} models using LiteLLM...")
    
    updated = 0
    today = datetime.now().strftime("%Y-%m-%d")
    
    provider_data = models_data.get('providers', {}).get(provider)
    if not provider_data:
        print(f"  ⚠️  Provider '{provider}' not found in models.json")
        return 0
    
    for model in provider_data.get('models', []):
        model_name = model.get('name', '')
        current_pricing = model.get('pricing', {})
        
        # Skip if already has pricing
        if current_pricing and not dry_run:
            continue
        
        # Try to get pricing from LiteLLM
        pricing = fetch_litellm_pricing(provider, model_name)
        if pricing:
            input_price, output_price = pricing
            new_pricing = {
                "input": input_price,
                "output": output_price,
                "unit": "per_1k_tokens",
                "last_updated": today
            }
            
            # Check if pricing changed
            if (not current_pricing or
                current_pricing.get('input') != new_pricing['input'] or
                current_pricing.get('output') != new_pricing['output']):
                
                if dry_run:
                    print(f"  Would update {model['display_name']}:")
                    if current_pricing:
                        print(f"    Current: ${current_pricing.get('input', 0):.6f}/1K in, "
                              f"${current_pricing.get('output', 0):.6f}/1K out")
                    else:
                        print(f"    Current: No pricing")
                    print(f"    New:     ${new_pricing['input']:.6f}/1K in, "
                          f"${new_pricing['output']:.6f}/1K out")
                else:
                    model['pricing'] = new_pricing
                    print(f"  ✅ Updated {model['display_name']}")
                
                updated += 1
    
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Update model pricing information from various sources"
    )
    parser.add_argument(
        '--models-file',
        default=os.path.expanduser("~/.episodic/models.json"),
        help='Path to models.json file (default: ~/.episodic/models.json)'
    )
    parser.add_argument(
        '--provider',
        choices=['anthropic', 'openrouter', 'openai', 'google', 'all'],
        default='all',
        help='Which provider to update (default: all)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without making changes'
    )
    
    args = parser.parse_args()
    
    # Check if models file exists
    if not os.path.exists(args.models_file):
        print(f"❌ Models file not found: {args.models_file}")
        print(f"   Run 'episodic' first to create the configuration")
        sys.exit(1)
    
    # Load current models configuration
    try:
        models_data = load_models_config(args.models_file)
    except Exception as e:
        print(f"❌ Error loading models.json: {e}")
        sys.exit(1)
    
    print(f"📋 Updating model pricing...")
    if args.dry_run:
        print("   (DRY RUN - no changes will be made)")
    
    total_updated = 0
    
    # Update OpenRouter pricing (from API)
    if args.provider in ['openrouter', 'all']:
        updated = update_openrouter_pricing(models_data, args.dry_run)
        total_updated += updated
    
    # Update Anthropic pricing (hardcoded/scraped)
    if args.provider in ['anthropic', 'all']:
        updated = update_anthropic_pricing(models_data, args.dry_run)
        total_updated += updated
    
    # Update other providers using LiteLLM
    if args.provider in ['openai', 'all']:
        updated = update_other_providers(models_data, 'openai', args.dry_run)
        total_updated += updated
    
    if args.provider in ['google', 'all']:
        updated = update_other_providers(models_data, 'google', args.dry_run)
        total_updated += updated
    
    # Save changes
    if not args.dry_run and total_updated > 0:
        save_models_config(args.models_file, models_data)
    
    print(f"\n📊 Summary: {total_updated} models {'would be' if args.dry_run else ''} updated")
    
    if args.dry_run and total_updated > 0:
        print("\n💡 Run without --dry-run to apply these changes")


if __name__ == "__main__":
    main()