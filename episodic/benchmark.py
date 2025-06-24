"""
Simple benchmarking system for Episodic.

Tracks performance of conceptual operations and resource usage with minimal overhead.
"""

import time
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import typer

from episodic.config import config
from episodic.configuration import get_system_color, get_llm_color


class BenchmarkStats:
    """Stores benchmark statistics for a single operation."""
    
    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.last_time = 0.0
        
    def add_measurement(self, elapsed: float):
        """Add a new measurement."""
        self.count += 1
        self.total_time += elapsed
        self.last_time = elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        
    @property
    def avg_time(self) -> float:
        """Calculate average time."""
        return self.total_time / self.count if self.count > 0 else 0.0


class BenchmarkManager:
    """Manages benchmark collection and reporting."""
    
    def __init__(self):
        self.stats: Dict[str, BenchmarkStats] = defaultdict(lambda: BenchmarkStats(""))
        self.resource_stats: Dict[str, Dict[str, BenchmarkStats]] = defaultdict(
            lambda: defaultdict(lambda: BenchmarkStats(""))
        )
        self.current_operation: Optional[str] = None
        self.operation_start_time: Optional[float] = None
        self.operation_resources: Dict[str, float] = defaultdict(float)
        
    def is_enabled(self) -> bool:
        """Check if benchmarking is enabled."""
        return config.get("benchmark", False)
    
    def should_display(self) -> bool:
        """Check if per-operation display is enabled."""
        return self.is_enabled() and config.get("benchmark_display", False)
        
    def start_operation(self, name: str):
        """Start tracking a conceptual operation."""
        if not self.is_enabled():
            return
            
        self.current_operation = name
        self.operation_start_time = time.perf_counter()
        self.operation_resources.clear()
        
    def end_operation(self):
        """End tracking the current operation and optionally display results."""
        if not self.is_enabled() or not self.current_operation:
            return
            
        elapsed = time.perf_counter() - self.operation_start_time
        self.stats[self.current_operation].add_measurement(elapsed)
        
        # Display if enabled
        if self.should_display():
            self._display_operation_benchmark(self.current_operation, elapsed)
            
        self.current_operation = None
        self.operation_start_time = None
        
    def record_resource(self, resource_type: str, resource_name: str, elapsed: float):
        """Record resource usage (e.g., LLM call, DB query)."""
        if not self.is_enabled():
            return
            
        self.resource_stats[resource_type][resource_name].add_measurement(elapsed)
        
        # Track for current operation if one is active
        if self.current_operation:
            self.operation_resources[resource_type] += elapsed
            
    def _display_operation_benchmark(self, operation: str, elapsed: float):
        """Display benchmark results for a single operation."""
        typer.echo("")  # Blank line before benchmark
        typer.secho(f"[Benchmark] {operation}: ", nl=False, fg=typer.colors.YELLOW)
        typer.secho(f"{elapsed:.2f}s", fg=typer.colors.BRIGHT_YELLOW, bold=True)
        
        # Show resource breakdown if any
        if self.operation_resources:
            for resource_type, resource_time in sorted(self.operation_resources.items()):
                # Count resources of this type
                count = sum(1 for k, v in self.resource_stats[resource_type].items() 
                           if v.last_time > 0)
                typer.secho(f"  - {resource_type}: ", nl=False, fg=typer.colors.CYAN)
                typer.secho(f"{resource_time:.2f}s", nl=False, fg=typer.colors.BRIGHT_CYAN)
                if count > 0:
                    typer.secho(f" ({count} calls)", fg=typer.colors.CYAN)
                else:
                    typer.echo("")
    
    def display_summary(self):
        """Display comprehensive benchmark summary."""
        if not self.stats and not self.resource_stats:
            typer.echo("No benchmark data collected.")
            return
            
        typer.echo("")
        typer.secho("Session Benchmark Summary", fg=typer.colors.BRIGHT_WHITE, bold=True)
        typer.secho("=" * 40, fg=typer.colors.WHITE)
        
        # Conceptual Operations
        if self.stats:
            typer.echo("")
            typer.secho("Conceptual Operations:", fg=typer.colors.BRIGHT_GREEN, bold=True)
            for name, stats in sorted(self.stats.items()):
                if stats.count > 0:
                    typer.secho(f"  - {name}: ", nl=False, fg=typer.colors.GREEN)
                    typer.secho(f"{stats.count} calls, ", nl=False, fg=typer.colors.WHITE)
                    typer.secho(f"avg {stats.avg_time:.2f}s, ", nl=False, fg=typer.colors.YELLOW)
                    typer.secho(f"total {stats.total_time:.2f}s", fg=typer.colors.BRIGHT_YELLOW)
                    
        # Resource Breakdown
        if self.resource_stats:
            typer.echo("")
            typer.secho("Resource Breakdown:", fg=typer.colors.BRIGHT_BLUE, bold=True)
            for resource_type, resources in sorted(self.resource_stats.items()):
                total_calls = sum(s.count for s in resources.values())
                total_time = sum(s.total_time for s in resources.values())
                
                typer.secho(f"  - {resource_type}: ", nl=False, fg=typer.colors.BLUE)
                typer.secho(f"{total_calls} calls, ", nl=False, fg=typer.colors.WHITE)
                typer.secho(f"total {total_time:.2f}s", fg=typer.colors.BRIGHT_BLUE)
                
                # Show breakdown by specific resource
                for resource_name, stats in sorted(resources.items()):
                    if stats.count > 0:
                        typer.secho(f"    • {resource_name}: ", nl=False, fg=typer.colors.CYAN)
                        typer.secho(f"{stats.count} calls, ", nl=False, fg=typer.colors.WHITE)
                        typer.secho(f"{stats.total_time:.2f}s", fg=typer.colors.BRIGHT_CYAN)
        
        # Total session time
        total_time = sum(s.total_time for s in self.stats.values())
        if total_time > 0:
            typer.echo("")
            typer.secho(f"Total session time: ", nl=False, fg=typer.colors.WHITE)
            typer.secho(f"{total_time:.2f}s", fg=typer.colors.BRIGHT_WHITE, bold=True)
            
    def reset(self):
        """Reset all benchmark data."""
        self.stats.clear()
        self.resource_stats.clear()
        self.current_operation = None
        self.operation_start_time = None
        self.operation_resources.clear()


# Global benchmark manager instance
benchmark_manager = BenchmarkManager()


@contextmanager
def benchmark_operation(name: str):
    """Context manager for benchmarking a conceptual operation."""
    benchmark_manager.start_operation(name)
    try:
        yield
    finally:
        benchmark_manager.end_operation()


@contextmanager
def benchmark_resource(resource_type: str, resource_name: str):
    """Context manager for benchmarking resource usage."""
    if not benchmark_manager.is_enabled():
        yield
        return
        
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        benchmark_manager.record_resource(resource_type, resource_name, elapsed)


def benchmark_decorator(operation_name: str):
    """Decorator for benchmarking functions as conceptual operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with benchmark_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience functions
def display_benchmark_summary():
    """Display the benchmark summary."""
    benchmark_manager.display_summary()


def reset_benchmarks():
    """Reset all benchmark data."""
    benchmark_manager.reset()


def is_benchmark_enabled() -> bool:
    """Check if benchmarking is enabled."""
    return benchmark_manager.is_enabled()