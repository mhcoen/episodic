"""
ID generation utilities for Episodic.

This module handles generation of unique IDs and short IDs for nodes.
"""

import uuid
import logging

from .configuration import (
    FALLBACK_ID_LENGTH, SHORT_ID_MAX_LENGTH,
    ID_CHARSET
)
from .db_connection import get_connection

# Set up logging
logger = logging.getLogger(__name__)


def base36_encode(number):
    """
    Convert a positive integer to a base-36 string.
    """
    if not isinstance(number, int) or number < 0:
        raise ValueError("Number must be a positive integer")

    chars = '0123456789abcdefghijklmnopqrstuvwxyz'

    if number == 0:
        return '0'

    result = ''
    while number > 0:
        result = chars[number % 36] + result
        number //= 36

    return result


def generate_short_id(fallback=False):
    """
    Generate a 2-character short ID.
    """
    with get_connection() as conn:
        c = conn.cursor()

        # Check if the short_id column exists in the nodes table
        c.execute("PRAGMA table_info(nodes)")
        columns = [column[1] for column in c.fetchall()]
        if 'short_id' not in columns:
            # If the column doesn't exist, return None
            return None

        # Step 1: Get existing short IDs
        c.execute("SELECT short_id FROM nodes WHERE short_id IS NOT NULL")
        existing_ids = set(row[0] for row in c.fetchall() if row[0])

        # Define available characters for the first and second positions
        vowels = 'aeiou'
        consonants = 'bcdfghjklmnpqrstvwxyz'
        all_letters = vowels + consonants
        
        # For two-character IDs, try alternating patterns first
        def generate_candidates():
            # First pass: consonant-vowel pattern for readability
            for first in consonants:
                for second in vowels:
                    yield first + second
            # Second pass: vowel-consonant pattern
            for first in vowels:
                for second in consonants:
                    yield first + second
            # Third pass: all two-letter combinations
            for first in all_letters:
                for second in all_letters:
                    yield first + second
                    
        # Try two-character IDs first
        for candidate in generate_candidates():
            if candidate not in existing_ids:
                return candidate

        # If no 2-character IDs are available (very unlikely), fall back to longer IDs
        logger.warning("No 2-character short IDs available, falling back to longer IDs")
        
        # Use standard characters for longer IDs
        for length in range(3, SHORT_ID_MAX_LENGTH + 1):
            # For each length, try random generation a reasonable number of times
            for _ in range(1000):
                # Use a more varied character set for longer IDs
                candidate = ''.join(
                    ID_CHARSET[int(uuid.uuid4().int % len(ID_CHARSET))] 
                    for _ in range(length)
                )
                if candidate not in existing_ids:
                    return candidate
        
        # Final fallback: generate a unique ID with timestamp
        if fallback:
            import time
            timestamp = base36_encode(int(time.time() * 1000000))
            random_suffix = ''.join(
                ID_CHARSET[int(uuid.uuid4().int % len(ID_CHARSET))] 
                for _ in range(FALLBACK_ID_LENGTH - len(timestamp))
            )
            return f"{timestamp}_{random_suffix}"
        
        # Should never reach here in practice
        raise RuntimeError("Unable to generate a unique short ID")