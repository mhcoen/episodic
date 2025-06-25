"""
Helper functions for CLI operations.

Contains utility functions used across multiple command modules.
"""

from typing import List, Optional, Tuple


def _parse_flag_value(args: List[str], flag_names: List[str]) -> Optional[str]:
    """
    Parse a flag value from a list of arguments.
    
    Args:
        args: List of arguments to parse
        flag_names: List of flag names to look for (e.g., ['--model', '-m'])
        
    Returns:
        The value following the flag, or None if not found
    """
    for i, arg in enumerate(args):
        if arg in flag_names and i + 1 < len(args):
            return args[i + 1]
    return None


def _remove_flag_and_value(args: List[str], flag_names: List[str]) -> List[str]:
    """
    Remove a flag and its value from the argument list.
    
    Args:
        args: List of arguments
        flag_names: List of flag names to remove
        
    Returns:
        New list with flag and value removed
    """
    result = []
    i = 0
    while i < len(args):
        if args[i] in flag_names and i + 1 < len(args):
            # Skip both flag and its value
            i += 2
        else:
            result.append(args[i])
            i += 1
    return result


def _has_flag(args: List[str], flag_names: List[str]) -> bool:
    """
    Check if any of the given flags are present in the arguments.
    
    Args:
        args: List of arguments to check
        flag_names: List of flag names to look for
        
    Returns:
        True if any flag is found, False otherwise
    """
    return any(arg in flag_names for arg in args)