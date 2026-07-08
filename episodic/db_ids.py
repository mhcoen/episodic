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


# Readable-ish letters for the shortest IDs; broader charset for longer ones.
_SHORT_ID_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Bounded random attempts per length. Independent of table size — the point
# is to avoid the old O(N) "load every short_id" scan on the per-insert hot
# path. A candidate is probed against the unique index (O(log N)); collisions
# just move on. Once a length's space fills up we escalate to the next length.
_ATTEMPTS_PER_LENGTH = {2: 40, 3: 400}
_DEFAULT_ATTEMPTS = 400

# sqlite error fragments that mean the nodes.short_id column isn't there yet.
_MISSING_COLUMN_MARKERS = ("no such column", "no such table")


def _random_short_id(length):
    """Build a random short_id of the given length."""
    charset = _SHORT_ID_LETTERS if length <= 2 else ID_CHARSET
    return ''.join(
        charset[uuid.uuid4().int % len(charset)]
        for _ in range(length)
    )


def generate_short_id(fallback=False):
    """
    Generate a short ID that is free at generation time.

    Probes candidate IDs against the unique index instead of loading every
    existing short_id, so this stays O(log N) per probe on the per-insert hot
    path rather than O(N) per call. Uniqueness is ultimately guaranteed by the
    UNIQUE constraint on nodes.short_id plus insert_node's collision retry;
    randomized candidates keep concurrent writers from colliding in lockstep.
    """
    with get_connection() as conn:
        c = conn.cursor()

        for length in range(2, SHORT_ID_MAX_LENGTH + 1):
            attempts = _ATTEMPTS_PER_LENGTH.get(length, _DEFAULT_ATTEMPTS)
            for _ in range(attempts):
                candidate = _random_short_id(length)
                try:
                    c.execute(
                        "SELECT 1 FROM nodes WHERE short_id = ? LIMIT 1",
                        (candidate,),
                    )
                except Exception as e:
                    # Legacy contract: no short_id column (or no nodes table)
                    # means "can't assign a short id" → None. Any other error
                    # is real and must propagate.
                    msg = str(e).lower()
                    if any(marker in msg for marker in _MISSING_COLUMN_MARKERS):
                        return None
                    raise
                if c.fetchone() is None:
                    return candidate

        # Space at every length is saturated (very large DB). Fall back to a
        # timestamped ID that is effectively collision-proof.
        if fallback:
            import time
            timestamp = base36_encode(int(time.time() * 1000000))
            random_suffix = ''.join(
                ID_CHARSET[uuid.uuid4().int % len(ID_CHARSET)]
                for _ in range(FALLBACK_ID_LENGTH - len(timestamp))
            )
            return f"{timestamp}_{random_suffix}"

        # Non-fallback caller and no free candidate found: return a best-effort
        # random candidate at max length. The UNIQUE constraint + insert retry
        # remain the correctness backstop.
        return _random_short_id(SHORT_ID_MAX_LENGTH)