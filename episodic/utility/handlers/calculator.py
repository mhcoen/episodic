"""
Calculator Handler.

Zero-dependency handler for arithmetic expressions.
Uses safe evaluation (no exec/eval of arbitrary code).
"""

import re
import operator
import math
from typing import Union

from ..types import UtilityQuery, UtilityResult


# Safe operators for expression evaluation
OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '//': operator.floordiv,
    '%': operator.mod,
    '^': operator.pow,
    '**': operator.pow,
}

# Safe math functions
MATH_FUNCTIONS = {
    'abs': abs,
    'round': round,
    'floor': math.floor,
    'ceil': math.ceil,
    'sqrt': math.sqrt,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'log': math.log,
    'log10': math.log10,
    'exp': math.exp,
    'pi': math.pi,
    'e': math.e,
}

# Token pattern for safe expression parsing
TOKEN_PATTERN = re.compile(r'''
    (\d+\.?\d*)|          # Numbers (including decimals)
    (\+|\-|\*\*?|\/\/?|%|\^)|  # Operators
    (\(|\))|              # Parentheses
    ([a-zA-Z_][a-zA-Z0-9_]*)  # Function names or constants
''', re.VERBOSE)


def tokenize_expr(expr: str) -> list:
    """Tokenize a math expression into safe tokens."""
    tokens = []
    pos = 0
    expr = expr.strip()

    while pos < len(expr):
        # Skip whitespace
        if expr[pos].isspace():
            pos += 1
            continue

        match = TOKEN_PATTERN.match(expr, pos)
        if not match:
            raise ValueError(f"Invalid character at position {pos}: {expr[pos]}")

        token = match.group(0)
        tokens.append(token)
        pos = match.end()

    return tokens


def safe_eval_expr(expr: str) -> Union[int, float]:
    """
    Safely evaluate a mathematical expression.

    Uses recursive descent parsing instead of eval().
    Supports: +, -, *, /, //, %, ^, **, parentheses, and basic math functions.
    """
    tokens = tokenize_expr(expr)
    pos = [0]  # Mutable position for recursive calls

    def parse_expression() -> float:
        """Parse addition and subtraction."""
        left = parse_term()

        while pos[0] < len(tokens) and tokens[pos[0]] in ('+', '-'):
            op = tokens[pos[0]]
            pos[0] += 1
            right = parse_term()
            if op == '+':
                left = left + right
            else:
                left = left - right

        return left

    def parse_term() -> float:
        """Parse multiplication, division, modulo."""
        left = parse_power()

        while pos[0] < len(tokens) and tokens[pos[0]] in ('*', '/', '//', '%'):
            op = tokens[pos[0]]
            pos[0] += 1
            right = parse_power()
            if op == '*':
                left = left * right
            elif op == '/':
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right
            elif op == '//':
                if right == 0:
                    raise ValueError("Division by zero")
                left = left // right
            else:  # %
                left = left % right

        return left

    def parse_power() -> float:
        """Parse exponentiation (right-associative)."""
        left = parse_unary()

        if pos[0] < len(tokens) and tokens[pos[0]] in ('^', '**'):
            pos[0] += 1
            right = parse_power()  # Right-associative
            left = left ** right

        return left

    def parse_unary() -> float:
        """Parse unary minus."""
        if pos[0] < len(tokens) and tokens[pos[0]] == '-':
            pos[0] += 1
            return -parse_unary()
        return parse_primary()

    def parse_primary() -> float:
        """Parse numbers, parentheses, and functions."""
        if pos[0] >= len(tokens):
            raise ValueError("Unexpected end of expression")

        token = tokens[pos[0]]

        # Number
        if re.match(r'^\d+\.?\d*$', token):
            pos[0] += 1
            return float(token) if '.' in token else int(token)

        # Parentheses
        if token == '(':
            pos[0] += 1
            result = parse_expression()
            if pos[0] >= len(tokens) or tokens[pos[0]] != ')':
                raise ValueError("Missing closing parenthesis")
            pos[0] += 1
            return result

        # Function or constant
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
            name = token.lower()
            pos[0] += 1

            # Check for constant
            if name in ('pi', 'e'):
                return MATH_FUNCTIONS[name]

            # Function call
            if pos[0] < len(tokens) and tokens[pos[0]] == '(':
                pos[0] += 1
                arg = parse_expression()
                if pos[0] >= len(tokens) or tokens[pos[0]] != ')':
                    raise ValueError("Missing closing parenthesis for function")
                pos[0] += 1

                if name not in MATH_FUNCTIONS:
                    raise ValueError(f"Unknown function: {name}")
                return MATH_FUNCTIONS[name](arg)

            raise ValueError(f"Unknown identifier: {name}")

        raise ValueError(f"Unexpected token: {token}")

    result = parse_expression()

    if pos[0] < len(tokens):
        raise ValueError(f"Unexpected token: {tokens[pos[0]]}")

    return result


def format_number(n: Union[int, float]) -> str:
    """Format a number for display."""
    if isinstance(n, float):
        # Check if it's effectively an integer
        if n == int(n):
            return str(int(n))
        # Limit decimal places
        return f"{n:.6g}"
    return str(n)


def handle_calc(query: UtilityQuery) -> UtilityResult:
    """
    Handle calc_expr command.

    Evaluates a mathematical expression safely.
    """
    expr = query.args.get("expr", "")
    if not expr:
        return UtilityResult.error("missing_expression", "No expression provided")

    try:
        result = safe_eval_expr(expr)
        formatted = format_number(result)

        display = f"{expr} = {formatted}"
        speech = f"The answer is {formatted}"

        return UtilityResult.ok(
            display=display,
            speech=speech,
            expression=expr,
            result=result,
        )
    except ValueError as e:
        return UtilityResult.error("evaluation_error", str(e))
    except Exception as e:
        return UtilityResult.error("calculation_error", f"Could not evaluate: {e}")


def handle_convert(query: UtilityQuery) -> UtilityResult:
    """
    Handle unit conversion.

    Args in query:
        value: float
        from_unit: str
        to_unit: str
    """
    value = query.args.get("value")
    from_unit = query.args.get("from_unit", "").lower()
    to_unit = query.args.get("to_unit", "").lower()

    if value is None:
        return UtilityResult.error("missing_value", "No value to convert")

    # Temperature conversions
    if from_unit in ("c", "celsius") and to_unit in ("f", "fahrenheit"):
        result = (value * 9/5) + 32
        display = f"{value}°C = {format_number(result)}°F"
        speech = f"{value} degrees Celsius is {format_number(result)} degrees Fahrenheit"
    elif from_unit in ("f", "fahrenheit") and to_unit in ("c", "celsius"):
        result = (value - 32) * 5/9
        display = f"{value}°F = {format_number(result)}°C"
        speech = f"{value} degrees Fahrenheit is {format_number(result)} degrees Celsius"

    # Length conversions
    elif from_unit in ("km", "kilometers") and to_unit in ("mi", "miles"):
        result = value * 0.621371
        display = f"{value} km = {format_number(result)} miles"
        speech = f"{value} kilometers is {format_number(result)} miles"
    elif from_unit in ("mi", "miles") and to_unit in ("km", "kilometers"):
        result = value * 1.60934
        display = f"{value} miles = {format_number(result)} km"
        speech = f"{value} miles is {format_number(result)} kilometers"
    elif from_unit in ("m", "meters") and to_unit in ("ft", "feet"):
        result = value * 3.28084
        display = f"{value} m = {format_number(result)} ft"
        speech = f"{value} meters is {format_number(result)} feet"
    elif from_unit in ("ft", "feet") and to_unit in ("m", "meters"):
        result = value * 0.3048
        display = f"{value} ft = {format_number(result)} m"
        speech = f"{value} feet is {format_number(result)} meters"

    # Weight conversions
    elif from_unit in ("kg", "kilograms") and to_unit in ("lb", "lbs", "pounds"):
        result = value * 2.20462
        display = f"{value} kg = {format_number(result)} lbs"
        speech = f"{value} kilograms is {format_number(result)} pounds"
    elif from_unit in ("lb", "lbs", "pounds") and to_unit in ("kg", "kilograms"):
        result = value * 0.453592
        display = f"{value} lbs = {format_number(result)} kg"
        speech = f"{value} pounds is {format_number(result)} kilograms"

    else:
        return UtilityResult.error(
            "unsupported_conversion",
            f"Cannot convert from {from_unit} to {to_unit}"
        )

    return UtilityResult.ok(
        display=display,
        speech=speech,
        value=value,
        from_unit=from_unit,
        to_unit=to_unit,
        result=result,
    )


# Command routing for calc category
CALC_HANDLERS = {
    "calc_expr": handle_calc,
    "convert": handle_convert,
}


def dispatch_calc_command(query: UtilityQuery) -> UtilityResult:
    """Dispatch a calc category command to the appropriate handler."""
    handler = CALC_HANDLERS.get(query.command)
    if handler:
        return handler(query)
    return UtilityResult.error("unknown_command", f"Unknown calc command: {query.command}")
