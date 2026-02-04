"""
Tests for Utility Commands module.

Tests cover:
1. UtilityQuery and UtilityResult types
2. Time/Date handlers
3. Calculator handlers (safe evaluation)
4. Dispatcher routing
5. Safety gate (confidence checking)
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from episodic.utility.types import (
    UtilityQuery,
    UtilityResult,
    ResultStatus,
    MUTATING_COMMANDS,
)
from episodic.utility.handlers.time_date import (
    handle_time,
    handle_date,
    handle_day_of_week,
    dispatch_time_command,
)
from episodic.utility.handlers.calculator import (
    handle_calc,
    handle_convert,
    safe_eval_expr,
    dispatch_calc_command,
)
from episodic.utility.dispatcher import (
    dispatch_utility,
    should_execute,
    create_utility_query,
)


class TestUtilityQueryTypes:
    """Tests for UtilityQuery dataclass."""

    def test_create_utility_query(self):
        """Create a basic UtilityQuery."""
        query = UtilityQuery(
            category="time",
            command="time_now",
            args={},
            confidence=1.0,
            source="cli",
            raw_input="what time is it",
        )
        assert query.category == "time"
        assert query.command == "time_now"
        assert query.confidence == 1.0

    def test_utility_query_to_dict(self):
        """UtilityQuery serializes correctly."""
        query = UtilityQuery(
            category="timer",
            command="timer_set",
            args={"duration_s": 300, "label": "pasta"},
            confidence=0.95,
            source="voice",
            raw_input="set a 5 minute pasta timer",
        )
        d = query.to_dict()
        assert d["ast_kind"] == "UtilityQuery"
        assert d["category"] == "timer"
        assert d["command"] == "timer_set"

    def test_is_mutating_true(self):
        """Mutating commands are correctly identified."""
        query = UtilityQuery(
            category="timer",
            command="timer_set",
            args={},
            confidence=1.0,
            source="cli",
            raw_input="",
        )
        assert query.is_mutating()

    def test_is_mutating_false(self):
        """Read-only commands are not mutating."""
        query = UtilityQuery(
            category="time",
            command="time_now",
            args={},
            confidence=1.0,
            source="cli",
            raw_input="",
        )
        assert not query.is_mutating()


class TestUtilityResultTypes:
    """Tests for UtilityResult dataclass."""

    def test_result_ok(self):
        """Create success result."""
        result = UtilityResult.ok("It's 3:45 PM", speech="It's three forty five PM")
        assert result.status == ResultStatus.OK
        assert result.display_text == "It's 3:45 PM"
        assert result.speech_text == "It's three forty five PM"

    def test_result_error(self):
        """Create error result."""
        result = UtilityResult.error("division_by_zero", "Cannot divide by zero")
        assert result.status == ResultStatus.ERROR
        assert result.error_type == "division_by_zero"
        assert "divide by zero" in result.error_message

    def test_result_confirm(self):
        """Create confirmation request."""
        result = UtilityResult.confirm("Delete all timers?")
        assert result.status == ResultStatus.CONFIRM
        assert "Delete" in result.display_text

    def test_result_fallback(self):
        """Create fallback to LLM."""
        result = UtilityResult.fallback()
        assert result.status == ResultStatus.FALLBACK


class TestTimeHandlers:
    """Tests for time/date handlers."""

    def test_handle_time_returns_time(self):
        """handle_time returns current time."""
        query = create_utility_query("time", "time_now")
        result = handle_time(query, user_tz="America/Chicago")

        assert result.status == ResultStatus.OK
        assert "time" in result.data
        assert result.data["timezone"] == "America/Chicago"
        # Display should contain AM or PM
        assert "AM" in result.display_text or "PM" in result.display_text

    def test_handle_time_different_timezone(self):
        """handle_time respects timezone."""
        query = create_utility_query("time", "time_now")
        result = handle_time(query, user_tz="Europe/London")

        assert result.status == ResultStatus.OK
        assert result.data["timezone"] == "Europe/London"

    def test_handle_date_returns_date(self):
        """handle_date returns current date."""
        query = create_utility_query("time", "date_today")
        result = handle_date(query)

        assert result.status == ResultStatus.OK
        assert "date" in result.data
        assert "day_of_week" in result.data
        # Display should contain day name
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        assert any(day in result.display_text for day in days)

    def test_handle_day_of_week_today(self):
        """handle_day_of_week for today."""
        query = create_utility_query("time", "day_of_week", args={"day_offset": 0})
        result = handle_day_of_week(query)

        assert result.status == ResultStatus.OK
        assert "Today" in result.display_text

    def test_handle_day_of_week_tomorrow(self):
        """handle_day_of_week for tomorrow."""
        query = create_utility_query("time", "day_of_week", args={"day_offset": 1})
        result = handle_day_of_week(query)

        assert result.status == ResultStatus.OK
        assert "Tomorrow" in result.display_text

    def test_dispatch_time_command_routing(self):
        """dispatch_time_command routes correctly."""
        query = create_utility_query("time", "time_now")
        result = dispatch_time_command(query)
        assert result.status == ResultStatus.OK

        query2 = create_utility_query("time", "date_today")
        result2 = dispatch_time_command(query2)
        assert result2.status == ResultStatus.OK

    def test_dispatch_time_unknown_command(self):
        """dispatch_time_command handles unknown commands."""
        query = create_utility_query("time", "unknown_command")
        result = dispatch_time_command(query)
        assert result.status == ResultStatus.ERROR


class TestCalculatorHandlers:
    """Tests for calculator handlers."""

    def test_safe_eval_basic_addition(self):
        """Safe eval handles basic addition."""
        assert safe_eval_expr("2 + 3") == 5

    def test_safe_eval_basic_subtraction(self):
        """Safe eval handles basic subtraction."""
        assert safe_eval_expr("10 - 4") == 6

    def test_safe_eval_multiplication(self):
        """Safe eval handles multiplication."""
        assert safe_eval_expr("3 * 4") == 12

    def test_safe_eval_division(self):
        """Safe eval handles division."""
        assert safe_eval_expr("15 / 3") == 5

    def test_safe_eval_integer_division(self):
        """Safe eval handles integer division."""
        assert safe_eval_expr("17 // 5") == 3

    def test_safe_eval_modulo(self):
        """Safe eval handles modulo."""
        assert safe_eval_expr("17 % 5") == 2

    def test_safe_eval_exponentiation(self):
        """Safe eval handles exponentiation."""
        assert safe_eval_expr("2 ^ 3") == 8
        assert safe_eval_expr("2 ** 3") == 8

    def test_safe_eval_parentheses(self):
        """Safe eval handles parentheses."""
        assert safe_eval_expr("(2 + 3) * 4") == 20

    def test_safe_eval_unary_minus(self):
        """Safe eval handles unary minus."""
        assert safe_eval_expr("-5 + 10") == 5

    def test_safe_eval_decimals(self):
        """Safe eval handles decimal numbers."""
        result = safe_eval_expr("3.14 * 2")
        assert abs(result - 6.28) < 0.001

    def test_safe_eval_sqrt_function(self):
        """Safe eval handles sqrt function."""
        assert safe_eval_expr("sqrt(16)") == 4

    def test_safe_eval_pi_constant(self):
        """Safe eval handles pi constant."""
        import math
        assert safe_eval_expr("pi") == math.pi

    def test_safe_eval_complex_expression(self):
        """Safe eval handles complex expressions."""
        result = safe_eval_expr("(10 + 5) * 2 - 8 / 4")
        assert result == 28

    def test_safe_eval_division_by_zero(self):
        """Safe eval raises on division by zero."""
        with pytest.raises(ValueError, match="Division by zero"):
            safe_eval_expr("5 / 0")

    def test_safe_eval_invalid_character(self):
        """Safe eval raises on invalid characters."""
        with pytest.raises(ValueError):
            safe_eval_expr("5 + $")

    def test_safe_eval_no_code_injection(self):
        """Safe eval rejects code injection attempts."""
        with pytest.raises(ValueError):
            safe_eval_expr("__import__('os').system('ls')")

    def test_handle_calc_success(self):
        """handle_calc evaluates expressions."""
        query = create_utility_query("calc", "calc_expr", args={"expr": "25 * 4"})
        result = handle_calc(query)

        assert result.status == ResultStatus.OK
        assert result.data["result"] == 100
        assert "100" in result.display_text

    def test_handle_calc_missing_expression(self):
        """handle_calc requires expression."""
        query = create_utility_query("calc", "calc_expr", args={})
        result = handle_calc(query)

        assert result.status == ResultStatus.ERROR
        assert result.error_type == "missing_expression"

    def test_handle_convert_celsius_fahrenheit(self):
        """handle_convert converts C to F."""
        query = create_utility_query("calc", "convert", args={
            "value": 100,
            "from_unit": "c",
            "to_unit": "f",
        })
        result = handle_convert(query)

        assert result.status == ResultStatus.OK
        assert result.data["result"] == 212

    def test_handle_convert_fahrenheit_celsius(self):
        """handle_convert converts F to C."""
        query = create_utility_query("calc", "convert", args={
            "value": 32,
            "from_unit": "f",
            "to_unit": "c",
        })
        result = handle_convert(query)

        assert result.status == ResultStatus.OK
        assert result.data["result"] == 0

    def test_handle_convert_km_miles(self):
        """handle_convert converts km to miles."""
        query = create_utility_query("calc", "convert", args={
            "value": 10,
            "from_unit": "km",
            "to_unit": "mi",
        })
        result = handle_convert(query)

        assert result.status == ResultStatus.OK
        assert abs(result.data["result"] - 6.21371) < 0.001

    def test_handle_convert_unsupported(self):
        """handle_convert rejects unsupported conversions."""
        query = create_utility_query("calc", "convert", args={
            "value": 10,
            "from_unit": "apples",
            "to_unit": "oranges",
        })
        result = handle_convert(query)

        assert result.status == ResultStatus.ERROR
        assert result.error_type == "unsupported_conversion"


class TestSafetyGate:
    """Tests for confidence-based safety gate."""

    def test_high_confidence_read_only_executes(self):
        """High confidence read-only commands execute."""
        query = create_utility_query("time", "time_now", confidence=1.0)
        should_exec, reason = should_execute(query)
        assert should_exec
        assert "high confidence" in reason

    def test_high_confidence_mutating_executes(self):
        """High confidence mutating commands execute."""
        query = create_utility_query("timer", "timer_set", confidence=0.95)
        should_exec, reason = should_execute(query)
        assert should_exec

    def test_medium_confidence_read_only_executes(self):
        """Medium confidence read-only commands execute."""
        query = create_utility_query("time", "time_now", confidence=0.8)
        should_exec, reason = should_execute(query)
        assert should_exec
        assert "read-only" in reason

    def test_medium_confidence_mutating_with_confirm(self):
        """Medium confidence mutating commands need confirmation when enabled."""
        query = create_utility_query("timer", "timer_set", args={"duration_s": 300}, confidence=0.75)
        should_exec, reason = should_execute(query, confirm_mutations=True)
        assert not should_exec
        assert "Confirm" in reason

    def test_medium_confidence_mutating_without_confirm(self):
        """Medium confidence mutating commands execute without confirmation setting."""
        query = create_utility_query("timer", "timer_set", confidence=0.75)
        should_exec, reason = should_execute(query, confirm_mutations=False)
        assert should_exec

    def test_low_confidence_fallback(self):
        """Low confidence commands fall back to LLM."""
        query = create_utility_query("time", "time_now", confidence=0.5)
        should_exec, reason = should_execute(query)
        assert not should_exec
        assert "low confidence" in reason


class TestDispatcher:
    """Tests for utility command dispatcher."""

    def test_dispatch_time_command(self):
        """Dispatcher handles time commands."""
        query = create_utility_query("time", "time_now")
        result = dispatch_utility(query)

        assert result.status == ResultStatus.OK

    def test_dispatch_calc_command(self):
        """Dispatcher handles calc commands."""
        query = create_utility_query("calc", "calc_expr", args={"expr": "2 + 2"})
        result = dispatch_utility(query)

        assert result.status == ResultStatus.OK
        assert result.data["result"] == 4

    def test_dispatch_unknown_category(self):
        """Dispatcher handles unknown categories."""
        query = create_utility_query("unknown", "unknown_command")
        result = dispatch_utility(query)

        assert result.status == ResultStatus.ERROR
        assert "unknown_category" in result.error_type

    def test_dispatch_low_confidence_fallback(self):
        """Dispatcher respects safety gate."""
        query = create_utility_query("time", "time_now", confidence=0.3)
        result = dispatch_utility(query)

        assert result.status == ResultStatus.FALLBACK

    def test_create_utility_query_helper(self):
        """create_utility_query helper works correctly."""
        query = create_utility_query(
            category="timer",
            command="timer_set",
            args={"duration_s": 600},
            source="voice",
            confidence=0.9,
        )
        assert query.category == "timer"
        assert query.command == "timer_set"
        assert query.args["duration_s"] == 600
        assert query.source == "voice"
        assert query.confidence == 0.9


class TestMutatingCommands:
    """Tests for mutating command detection."""

    def test_timer_commands_are_mutating(self):
        """Timer commands are correctly marked as mutating."""
        assert "timer_set" in MUTATING_COMMANDS
        assert "timer_cancel" in MUTATING_COMMANDS

    def test_alarm_commands_are_mutating(self):
        """Alarm commands are correctly marked as mutating."""
        assert "alarm_set" in MUTATING_COMMANDS
        assert "alarm_cancel" in MUTATING_COMMANDS

    def test_time_commands_not_mutating(self):
        """Time query commands are not mutating."""
        assert "time_now" not in MUTATING_COMMANDS
        assert "date_today" not in MUTATING_COMMANDS

    def test_calc_commands_not_mutating(self):
        """Calc commands are not mutating."""
        assert "calc_expr" not in MUTATING_COMMANDS
        assert "convert" not in MUTATING_COMMANDS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
