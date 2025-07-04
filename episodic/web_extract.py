"""
Web content extraction for enhanced search results.

This module provides functionality to fetch and extract meaningful content
from web pages, going beyond just search result snippets.
"""

import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
import re

import typer
from episodic.config import config


class WebContentExtractor:
    """Extract meaningful content from web pages."""
    
    def __init__(self):
        self.timeout = config.get('web_search_timeout', 10)
    
    async def extract_content(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch and extract content from a URL.
        
        Returns:
            Dictionary with 'title', 'content', and 'metadata'
        """
        try:
            import aiohttp
            from bs4 import BeautifulSoup
        except ImportError:
            return None
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                    if response.status != 200:
                        if config.get('debug'):
                            typer.secho(f"\nHTTP {response.status} for {url}", fg="red")
                        return None
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title
                    title = soup.find('title')
                    title_text = title.get_text().strip() if title else 'Untitled'
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Try to extract main content
                    content = self._extract_main_content(soup, url)
                    
                    return {
                        'title': title_text,
                        'content': content,
                        'url': url,
                        'extracted_at': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            if config.get('debug'):
                typer.secho(f"\nFailed to extract from {url}: {type(e).__name__}: {e}", fg="red")
            return None
    
    def _extract_main_content(self, soup, url: str) -> str:
        """Extract the main content based on URL patterns and page structure."""
        
        # Weather-specific extraction
        if any(domain in url for domain in ['weather.com', 'accuweather.com', 'weather.gov']):
            return self._extract_weather_content(soup)
        
        # News article extraction
        if any(domain in url for domain in ['medium.com', 'techcrunch.com', 'reuters.com']):
            return self._extract_article_content(soup)
        
        # Generic extraction - look for main content areas
        for selector in ['main', 'article', '[role="main"]', '#content', '.content']:
            main_content = soup.select_one(selector)
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
                # Clean up excessive whitespace
                text = ' '.join(text.split())
                if len(text) > 100:  # Ensure we got meaningful content
                    return text[:2000]  # Limit length
        
        # Fallback to body text
        body = soup.find('body')
        if body:
            text = body.get_text(separator=' ', strip=True)
            text = ' '.join(text.split())
            return text[:1000]
        
        return "Could not extract content from this page."
    
    def _extract_weather_content(self, soup) -> str:
        """Extract weather-specific information."""
        content_parts = []
        
        # Look for current temperature
        temp_selectors = [
            '.CurrentConditions--tempValue--*',  # weather.com
            '.temp', '.temperature',  # generic
            '[data-testid="TemperatureValue"]',  # modern sites
            '.current-temp', '.now-temp'
        ]
        
        for selector in temp_selectors:
            temp = soup.select_one(selector)
            if temp:
                content_parts.append(f"Current temperature: {temp.get_text().strip()}")
                break
        
        # Look for conditions
        condition_selectors = [
            '.CurrentConditions--phraseValue--*',  # weather.com
            '.condition', '.weather-condition',
            '.phrase', '.weather-phrase'
        ]
        
        for selector in condition_selectors:
            condition = soup.select_one(selector)
            if condition:
                content_parts.append(f"Conditions: {condition.get_text().strip()}")
                break
        
        # Look for forecast summary
        forecast_selectors = [
            '.DetailsSummary--summaryText--*',
            '.forecast-text', '.summary',
            '.today-forecast'
        ]
        
        for selector in forecast_selectors:
            forecast = soup.select_one(selector)
            if forecast:
                content_parts.append(f"Forecast: {forecast.get_text().strip()}")
                break
        
        # Try to find any text containing temperature patterns
        if not content_parts:
            text = soup.get_text()
            # Look for temperature patterns like "72°F" or "22°C"
            temp_match = re.search(r'(\d{1,3}°[FC])', text)
            if temp_match:
                content_parts.append(f"Temperature: {temp_match.group(1)}")
            
            # Look for weather conditions
            conditions = ['sunny', 'cloudy', 'rainy', 'snowy', 'clear', 'overcast', 'partly cloudy']
            text_lower = text.lower()
            for condition in conditions:
                if condition in text_lower:
                    # Find the sentence containing the condition
                    sentences = text.split('.')
                    for sentence in sentences:
                        if condition in sentence.lower() and len(sentence) < 200:
                            content_parts.append(sentence.strip())
                            break
                    break
        
        return ' | '.join(content_parts) if content_parts else "Weather information not found on page."
    
    def _extract_article_content(self, soup) -> str:
        """Extract article content."""
        # Look for article body
        article_selectors = [
            'article .post-content',
            'article .entry-content', 
            '.article-body',
            '[itemprop="articleBody"]',
            '.story-body'
        ]
        
        for selector in article_selectors:
            content = soup.select_one(selector)
            if content:
                # Get first few paragraphs
                paragraphs = content.find_all('p', limit=3)
                if paragraphs:
                    text = ' '.join(p.get_text().strip() for p in paragraphs)
                    return text[:1000]
        
        # Fallback to first few paragraphs anywhere
        paragraphs = soup.find_all('p', limit=5)
        if paragraphs:
            text = ' '.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50)
            return text[:1000]
        
        return "Could not extract article content."


async def fetch_page_content(url: str) -> Optional[str]:
    """
    Fetch and extract meaningful content from a web page.
    
    Args:
        url: URL to fetch content from
        
    Returns:
        Extracted content as string, or None if failed
    """
    extractor = WebContentExtractor()
    result = await extractor.extract_content(url)
    
    if result:
        return result.get('content', '')
    return None