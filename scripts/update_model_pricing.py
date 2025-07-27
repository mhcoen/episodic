#!/usr/bin/env python3
"""
Update model pricing information from various sources.

This script fetches current pricing from:
- OpenRouter API (automatic)
- Anthropic website (manual/scraped)
- Other providers (from LiteLLM if available)

Updates the models.json file in the user's .episodic directory.

IMPORTANT: Never blindly copy pricing from LiteLLM as it's often outdated!
Always verify prices against official provider websites.
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
            
            # Convert to cost per 1M tokens
            prompt_cost = float(model_pricing.get('prompt', 0)) * 1000000
            completion_cost = float(model_pricing.get('completion', 0)) * 1000000
            
            if prompt_cost > 0 or completion_cost > 0:
                pricing[model_id] = (prompt_cost, completion_cost)
        
        print(f"  ✅ Found pricing for {len(pricing)} OpenRouter models")
        return pricing
        
    except Exception as e:
        print(f"  ❌ Error fetching OpenRouter pricing: {e}")
        return {}


def fetch_aimultiple_pricing() -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Fetch pricing data from AIMultiple's LLM pricing comparison.
    
    Returns:
        Dict mapping provider -> model -> pricing info
    """
    print("🔍 Fetching pricing from AIMultiple...")
    
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print(f"  ⚠️  BeautifulSoup not installed, skipping AIMultiple")
        return {}
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get("https://research.aimultiple.com/llm-pricing/", headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"  ❌ Error: Site returned status {response.status_code}")
            return {}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        pricing_data = {}
        
        # Find the pricing table (first table with Model, InputPrice, OutputPrice headers)
        tables = soup.find_all('table')
        pricing_table = None
        
        for table in tables:
            header_row = table.find('thead')
            if header_row:
                headers = [th.text.strip() for th in header_row.find_all('th')]
                if 'Model' in headers and 'InputPrice' in headers and 'OutputPrice' in headers:
                    pricing_table = table
                    break
        
        if not pricing_table:
            print(f"  ⚠️  Could not find pricing table on AIMultiple")
            return {}
        
        # Parse the pricing table
        tbody = pricing_table.find('tbody')
        if tbody:
            for row in tbody.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 3:
                    model_full = cells[0].text.strip()
                    input_price = cells[1].text.strip()
                    output_price = cells[2].text.strip()
                    
                    # Parse provider and model name
                    # Format is usually "Provider ModelName" e.g., "OpenAI gpt-4o"
                    parts = model_full.split(' ', 1)
                    if len(parts) == 2:
                        provider = parts[0].lower()
                        model = parts[1]
                    else:
                        continue
                    
                    # Clean price strings (remove $, commas, etc)
                    try:
                        input_cost = float(input_price.replace('$', '').replace(',', ''))
                        output_cost = float(output_price.replace('$', '').replace(',', ''))
                        
                        # AIMultiple already shows per 1M tokens
                        # input_cost and output_cost are already in per 1M format
                        
                        if provider not in pricing_data:
                            pricing_data[provider] = {}
                        
                        pricing_data[provider][model] = {
                            "input": input_cost,
                            "output": output_cost
                        }
                    except ValueError:
                        continue
        
        total_models = sum(len(models) for models in pricing_data.values())
        print(f"  ✅ Found pricing for {total_models} models from {len(pricing_data)} providers")
        return pricing_data
        
    except Exception as e:
        print(f"  ❌ Error fetching AIMultiple pricing: {e}")
        return {}


def fetch_pricepertoken_pricing() -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Fetch pricing data from pricepertoken.com.
    
    Returns:
        Dict mapping provider -> model -> pricing info
    """
    print("🔍 Fetching pricing from pricepertoken.com...")
    
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print(f"  ⚠️  BeautifulSoup not installed, skipping pricepertoken.com")
        return {}
    
    try:
        # Note: pricepertoken.com is a client-side rendered JavaScript app
        # Direct HTML scraping won't work without a headless browser
        # TODO: Implement Selenium or Playwright for dynamic content scraping
        
        print(f"  ⚠️  pricepertoken.com requires JavaScript rendering")
        print(f"  💡 Consider using Selenium or manual updates from https://pricepertoken.com/")
        
        # For now, return empty and fall back to other sources
        return {}
        
    except Exception as e:
        print(f"  ❌ Error fetching pricepertoken.com pricing: {e}")
        return {}


def fetch_openai_pricing() -> Dict[str, Dict[str, float]]:
    """
    Fetch OpenAI pricing from pricepertoken.com or their website.
    
    Returns:
        Dict mapping model names to pricing info
    """
    print("🔍 Fetching OpenAI pricing...")
    
    # First try AIMultiple
    aimultiple_data = fetch_aimultiple_pricing()
    if aimultiple_data and 'openai' in aimultiple_data:
        pricing = aimultiple_data['openai']
        print(f"  ✅ Found pricing for {len(pricing)} OpenAI models from AIMultiple")
        return pricing
    
    # Then try pricepertoken.com
    pricepertoken_data = fetch_pricepertoken_pricing()
    if pricepertoken_data and 'openai' in pricepertoken_data:
        pricing = pricepertoken_data['openai']
        print(f"  ✅ Found pricing for {len(pricing)} OpenAI models from pricepertoken.com")
        return pricing
    
    # Then try direct scraping from OpenAI
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print(f"  ⚠️  BeautifulSoup not installed, cannot scrape pricing")
        return {}
    
    try:
        response = requests.get("https://openai.com/api/pricing/", timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Actual scraping implementation would go here
            print(f"  ⚠️  OpenAI page structure needs analysis for scraping")
        else:
            print(f"  ⚠️  OpenAI pricing page returned status {response.status_code}")
    except Exception as e:
        print(f"  ⚠️  Could not scrape OpenAI pricing: {e}")
    
    # No hardcoded fallback - return empty if scraping fails
    print(f"  ℹ️  No pricing data available - will use LiteLLM at runtime")
    return {}


def fetch_anthropic_pricing() -> Dict[str, Dict[str, float]]:
    """
    Fetch Anthropic pricing from pricepertoken.com or their website.
    
    Returns:
        Dict mapping model names to pricing info
    """
    print("🔍 Fetching Anthropic pricing...")
    
    # First try AIMultiple
    aimultiple_data = fetch_aimultiple_pricing()
    if aimultiple_data and 'anthropic' in aimultiple_data:
        pricing = aimultiple_data['anthropic']
        print(f"  ✅ Found pricing for {len(pricing)} Anthropic models from AIMultiple")
        return pricing
    
    # Then try pricepertoken.com
    pricepertoken_data = fetch_pricepertoken_pricing()
    if pricepertoken_data and 'anthropic' in pricepertoken_data:
        pricing = pricepertoken_data['anthropic']
        print(f"  ✅ Found pricing for {len(pricing)} Anthropic models from pricepertoken.com")
        return pricing
    
    # Then try direct scraping from Anthropic
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print(f"  ⚠️  BeautifulSoup not installed, cannot scrape pricing")
        return {}
    
    try:
        response = requests.get("https://www.anthropic.com/pricing", timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Actual scraping implementation would go here
            print(f"  ⚠️  Anthropic page structure needs analysis for scraping")
        else:
            print(f"  ⚠️  Anthropic pricing page returned status {response.status_code}")
    except Exception as e:
        print(f"  ⚠️  Could not scrape Anthropic pricing: {e}")
    
    # No hardcoded fallback - return empty if scraping fails
    print(f"  ℹ️  No pricing data available - will use LiteLLM at runtime")
    return {}


def fetch_litellm_pricing(provider: str, model_name: str) -> Optional[Tuple[float, float]]:
    """
    Try to get pricing from LiteLLM for a specific model.
    
    Returns:
        Tuple of (input_cost, output_cost) per 1K tokens, or None if not found
    """
    try:
        from litellm import cost_per_token
        
        # Calculate cost for 1M tokens
        input_cost = cost_per_token(model=model_name, prompt_tokens=1000000, completion_tokens=0)
        output_cost = cost_per_token(model=model_name, prompt_tokens=0, completion_tokens=1000000)
        
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
                "unit": "per_1m_tokens",
                "last_updated": today
            }
            
            # Check if pricing changed
            if (current_pricing.get('input') != new_pricing['input'] or
                current_pricing.get('output') != new_pricing['output']):
                
                if dry_run:
                    print(f"  Would update {model['display_name']}:")
                    if current_pricing:
                        print(f"    Current: ${current_pricing.get('input', 0):.2f}/1M in, "
                              f"${current_pricing.get('output', 0):.2f}/1M out")
                    else:
                        print(f"    Current: No pricing")
                    print(f"    New:     ${new_pricing['input']:.2f}/1M in, "
                          f"${new_pricing['output']:.2f}/1M out")
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
                    "unit": "per_1m_tokens",
                    "last_updated": today
                }
                
                # Check if pricing changed
                if (current_pricing.get('input') != new_pricing['input'] or
                    current_pricing.get('output') != new_pricing['output']):
                    
                    if dry_run:
                        print(f"  Would update {model['display_name']}:")
                        if current_pricing:
                            print(f"    Current: ${current_pricing.get('input', 0):.2f}/1M in, "
                                  f"${current_pricing.get('output', 0):.2f}/1M out")
                        else:
                            print(f"    Current: No pricing")
                        print(f"    New:     ${new_pricing['input']:.2f}/1M in, "
                              f"${new_pricing['output']:.2f}/1M out")
                    else:
                        model['pricing'] = new_pricing
                        print(f"  ✅ Updated {model['display_name']}")
                    
                    updated += 1
                break
    
    return updated


def update_openai_pricing(models_data: Dict[str, Any], dry_run: bool = False) -> int:
    """
    Update OpenAI model pricing from scraped/known data.
    
    Returns:
        Number of models updated
    """
    print("\n🤖 Updating OpenAI models...")
    
    openai_pricing = fetch_openai_pricing()
    updated = 0
    today = datetime.now().strftime("%Y-%m-%d")
    
    provider_data = models_data.get('providers', {}).get('openai')
    if not provider_data:
        print("  ⚠️  OpenAI provider not found in models.json")
        return 0
    
    for model in provider_data.get('models', []):
        model_name = model.get('name', '')
        
        # Try to match the model with pricing data (exact match first)
        pricing_info = None
        if model_name in openai_pricing:
            pricing_info = openai_pricing[model_name]
        else:
            # Try partial match for variations
            for pricing_key, price_data in openai_pricing.items():
                if pricing_key == model_name.replace('-', '') or model_name.replace('-', '') == pricing_key:
                    pricing_info = price_data
                    break
        
        if pricing_info:
            current_pricing = model.get('pricing', {})
            new_pricing = {
                "input": pricing_info['input'],
                "output": pricing_info['output'],
                "unit": "per_1m_tokens",
                "last_updated": today
            }
            
            # Check if pricing changed
            if (current_pricing.get('input') != new_pricing['input'] or
                current_pricing.get('output') != new_pricing['output']):
                
                if dry_run:
                    print(f"  Would update {model['display_name']}:")
                    if current_pricing:
                        print(f"    Current: ${current_pricing.get('input', 0):.2f}/1M in, "
                              f"${current_pricing.get('output', 0):.2f}/1M out")
                    else:
                        print(f"    Current: No pricing")
                    print(f"    New:     ${new_pricing['input']:.2f}/1M in, "
                          f"${new_pricing['output']:.2f}/1M out")
                else:
                    model['pricing'] = new_pricing
                    print(f"  ✅ Updated {model['display_name']}")
                
                updated += 1
    
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
                "unit": "per_1m_tokens",
                "last_updated": today
            }
            
            # Check if pricing changed
            if (not current_pricing or
                current_pricing.get('input') != new_pricing['input'] or
                current_pricing.get('output') != new_pricing['output']):
                
                if dry_run:
                    print(f"  Would update {model['display_name']}:")
                    if current_pricing:
                        print(f"    Current: ${current_pricing.get('input', 0):.2f}/1M in, "
                              f"${current_pricing.get('output', 0):.2f}/1M out")
                    else:
                        print(f"    Current: No pricing")
                    print(f"    New:     ${new_pricing['input']:.2f}/1M in, "
                          f"${new_pricing['output']:.2f}/1M out")
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
    
    # Update OpenAI pricing (hardcoded/scraped)
    if args.provider in ['openai', 'all']:
        updated = update_openai_pricing(models_data, args.dry_run)
        total_updated += updated
    
    # Update other providers using LiteLLM
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