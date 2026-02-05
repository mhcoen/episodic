"""
Time Normalizer for Voice Grammar.

Converts time expressions to absolute datetime candidates.
Uses zoneinfo (Python 3.9+ stdlib) for timezone handling.

DST policy:
- Ambiguous times (fall back): use fold=1 (second occurrence)
- Non-existent times (spring forward): shift forward by 1 hour
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Optional, Tuple
import re


class TimeNormalizer:
    """
    Converts time expressions to absolute datetime candidates.
    """

    GRACE_WINDOW_SECONDS = 30

    def normalize_time_of_day(
        self,
        expr: str,
        ref_time: datetime,
        tz_name: str = "America/Chicago"
    ) -> List[datetime]:
        """Generate datetime candidates, shifted to future if past."""
        tz = ZoneInfo(tz_name)

        parsed = self._parse_time_expr(expr)
        if not parsed:
            return []

        hour, minute, ampm = parsed

        if not self._validate_time(hour, minute, ampm):
            return []

        # Handle "quarter to X" wrap
        if minute is not None and minute < 0:
            hour = (hour - 1) % 12 or 12
            minute = 60 + minute

        minute = minute or 0
        candidates: List[datetime] = []

        if ampm is None:
            for suffix in ['am', 'pm']:
                dt = self._make_datetime(ref_time, hour, minute, suffix, tz)
                if dt:
                    candidates.append(dt)
        else:
            dt = self._make_datetime(ref_time, hour, minute, ampm, tz)
            if dt:
                candidates.append(dt)

        candidates = [c for c in (self._shift_if_past(c, ref_time) for c in candidates) if c]
        return candidates

    def normalize_duration(self, expr: str) -> Optional[int]:
        """
        Parse duration expression to seconds.

        Supports:
        - "10 minutes", "10 min", "10m"
        - "1 hour 30 minutes", "1h30m"
        - "a couple minutes", "a few hours"
        - "half hour", "quarter hour"
        """
        expr = expr.lower().strip()

        # Handle "a couple/few" patterns
        couple_match = re.match(r"a?\s*couple\s+(?:of\s+)?(\w+)", expr)
        if couple_match:
            unit = couple_match.group(1)
            return self._unit_to_seconds(unit, 2)

        few_match = re.match(r"a?\s*few\s+(\w+)", expr)
        if few_match:
            unit = few_match.group(1)
            return self._unit_to_seconds(unit, 3)

        # Handle "half/quarter hour"
        if "half" in expr and "hour" in expr:
            return 1800
        if "quarter" in expr and "hour" in expr:
            return 900

        # Standard pattern: extract (number, unit) pairs
        total_seconds = 0
        pattern = r'(\d+)\s*(h|hr|hours?|m|min|mins|minutes?|s|sec|secs|seconds?)?'
        matches = re.findall(pattern, expr)

        for value_str, unit in matches:
            value = int(value_str)
            if not unit or unit.startswith('s'):
                total_seconds += value
            elif unit.startswith('m'):
                total_seconds += value * 60
            elif unit.startswith('h'):
                total_seconds += value * 3600

        return total_seconds if total_seconds > 0 else None

    def _unit_to_seconds(self, unit: str, count: int) -> Optional[int]:
        """Convert a time unit with count to seconds."""
        unit = unit.lower()
        if unit.startswith('s'):
            return count
        elif unit.startswith('m'):
            return count * 60
        elif unit.startswith('h'):
            return count * 3600
        return None

    def _validate_time(self, hour: int, minute: Optional[int], ampm: Optional[str]) -> bool:
        """Validate time components."""
        if ampm:
            if hour < 1 or hour > 12:
                return False
        else:
            if hour < 0 or hour > 23:
                return False

        if minute is not None and (minute < -15 or minute >= 60):
            return False

        return True

    def _make_datetime(
        self,
        ref: datetime,
        hour: int,
        minute: int,
        ampm: str,
        tz: ZoneInfo
    ) -> Optional[datetime]:
        """Create datetime from components with DST handling."""
        # Convert 12-hour to 24-hour
        if ampm == 'am':
            hour_24 = 0 if hour == 12 else hour
        else:
            hour_24 = 12 if hour == 12 else hour + 12

        if ref.tzinfo is None:
            ref_date = ref.date()
        else:
            ref_local = ref.astimezone(tz)
            ref_date = ref_local.date()

        try:
            naive = datetime(ref_date.year, ref_date.month, ref_date.day,
                             hour_24, minute, 0, 0)

            # Handle DST with fold
            dt_fold0 = naive.replace(tzinfo=tz, fold=0)
            dt_fold1 = naive.replace(tzinfo=tz, fold=1)

            # Check for non-existent time (spring forward)
            roundtrip = dt_fold0.astimezone(tz)
            if roundtrip.hour != hour_24 or roundtrip.minute != minute:
                return naive.replace(tzinfo=tz) + timedelta(hours=1)

            # For ambiguous times, prefer fold=1
            return dt_fold1

        except Exception:
            return None

    def _shift_if_past(self, dt: datetime, ref: datetime) -> Optional[datetime]:
        """Shift datetime to next day if it's in the past."""
        if dt is None:
            return None

        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=dt.tzinfo)

        diff = (dt - ref).total_seconds()
        if diff < -self.GRACE_WINDOW_SECONDS:
            return dt + timedelta(days=1)
        return dt

    def _parse_time_expr(self, expr: str) -> Optional[Tuple[int, Optional[int], Optional[str]]]:
        """Parse time expression. Returns (hour, minute, ampm) or None."""
        expr = expr.lower().strip()

        # Quarter/half patterns
        quarter_to = re.match(r"quarter\s+(to|til|till|before)\s+(\d+)\s*(am|pm)?", expr)
        if quarter_to:
            return (int(quarter_to.group(2)), -15, quarter_to.group(3))

        quarter_past = re.match(r"quarter\s+(past|after)\s+(\d+)\s*(am|pm)?", expr)
        if quarter_past:
            return (int(quarter_past.group(2)), 15, quarter_past.group(3))

        half_past = re.match(r"half\s+past\s+(\d+)\s*(am|pm)?", expr)
        if half_past:
            return (int(half_past.group(1)), 30, half_past.group(2))

        # Standard: "7:30am"
        colon_time = re.match(r"(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?", expr)
        if colon_time:
            ampm = colon_time.group(3)
            if ampm:
                ampm = ampm.replace(".", "").strip()
            return (int(colon_time.group(1)), int(colon_time.group(2)), ampm)

        # ASR: "7 30"
        space_time = re.match(r"(\d{1,2})\s+(\d{2})(?:\s*(am|pm))?$", expr)
        if space_time:
            return (int(space_time.group(1)), int(space_time.group(2)), space_time.group(3))

        # Simple: "7am", "7 pm", "7"
        simple_time = re.match(r"(\d{1,2})\s*(am|pm|a\.m\.|p\.m\.|o'clock)?$", expr)
        if simple_time:
            ampm = simple_time.group(2)
            if ampm:
                ampm = ampm.replace(".", "").replace("o'clock", "").strip() or None
            return (int(simple_time.group(1)), 0, ampm)

        # Named
        if expr == "noon":
            return (12, 0, "pm")
        if expr == "midnight":
            return (12, 0, "am")

        return None
