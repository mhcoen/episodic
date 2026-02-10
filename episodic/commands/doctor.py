"""
Installation health check command for Episodic.

The /doctor command verifies that Episodic is properly installed and configured.
"""

import sys
import os
from typing import List, Optional
from dataclasses import dataclass



@dataclass
class CheckResult:
    """Result of a single health check."""
    name: str
    passed: bool
    message: str
    details: Optional[str] = None


def _check_python_version() -> CheckResult:
    """Check Python version meets minimum requirements."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version >= (3, 8):
        return CheckResult(
            name="Python Version",
            passed=True,
            message=f"Python {version_str}",
        )
    else:
        return CheckResult(
            name="Python Version",
            passed=False,
            message=f"Python {version_str} (requires >= 3.8)",
        )


def _check_core_dependencies() -> List[CheckResult]:
    """Check that core dependencies can be imported."""
    results = []

    core_deps = [
        ("typer", "CLI framework"),
        ("rich", "Terminal formatting"),
        ("prompt_toolkit", "Interactive input"),
        ("litellm", "LLM interface"),
        ("networkx", "Graph structures"),
        ("numpy", "Numerical computing"),
        ("sqlite3", "Database"),
    ]

    for module, description in core_deps:
        try:
            __import__(module)
            results.append(CheckResult(
                name=f"Core: {module}",
                passed=True,
                message=description,
            ))
        except ImportError as e:
            results.append(CheckResult(
                name=f"Core: {module}",
                passed=False,
                message=f"{description} - MISSING",
                details=str(e),
            ))

    return results


def _check_optional_features() -> List[CheckResult]:
    """Check availability of optional features."""
    results = []

    # RAG feature
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
        results.append(CheckResult(
            name="RAG (Knowledge Base)",
            passed=True,
            message="chromadb + sentence-transformers installed",
        ))
    except ImportError:
        results.append(CheckResult(
            name="RAG (Knowledge Base)",
            passed=False,
            message="Not installed (pip install -e '.[rag]')",
        ))

    # Web search feature
    try:
        import beautifulsoup4  # noqa: F401
        import httpx  # noqa: F401
        results.append(CheckResult(
            name="Web Search",
            passed=True,
            message="beautifulsoup4 + httpx installed",
        ))
    except ImportError:
        # Check individual components
        web_parts = []
        try:
            import bs4  # noqa: F401
            web_parts.append("bs4")
        except ImportError:
            pass
        try:
            import httpx  # noqa: F401
            web_parts.append("httpx")
        except ImportError:
            pass
        try:
            import duckduckgo_search  # noqa: F401
            web_parts.append("duckduckgo")
        except ImportError:
            pass

        if web_parts:
            results.append(CheckResult(
                name="Web Search",
                passed=True,
                message=f"Partial: {', '.join(web_parts)}",
            ))
        else:
            results.append(CheckResult(
                name="Web Search",
                passed=False,
                message="Not installed (pip install -e '.[web]')",
            ))

    # PDF feature
    try:
        import pypdf  # noqa: F401
        results.append(CheckResult(
            name="PDF Support",
            passed=True,
            message="pypdf installed",
        ))
    except ImportError:
        results.append(CheckResult(
            name="PDF Support",
            passed=False,
            message="Not installed (pip install pypdf)",
        ))

    # Voice feature
    voice_parts = []
    try:
        import sounddevice  # noqa: F401
        voice_parts.append("sounddevice")
    except ImportError:
        pass
    try:
        import webrtcvad  # noqa: F401
        voice_parts.append("webrtcvad")
    except ImportError:
        pass
    try:
        import pvporcupine  # noqa: F401
        voice_parts.append("porcupine")
    except ImportError:
        pass

    if len(voice_parts) >= 2:
        results.append(CheckResult(
            name="Voice Mode",
            passed=True,
            message=f"Components: {', '.join(voice_parts)}",
        ))
    elif voice_parts:
        results.append(CheckResult(
            name="Voice Mode",
            passed=False,
            message=f"Partial: {', '.join(voice_parts)} (pip install -e '.[voice]')",
        ))
    else:
        results.append(CheckResult(
            name="Voice Mode",
            passed=False,
            message="Not installed (pip install -e '.[voice]')",
        ))

    # ML feature
    try:
        import sklearn  # noqa: F401
        results.append(CheckResult(
            name="ML Features",
            passed=True,
            message="scikit-learn installed",
        ))
    except ImportError:
        results.append(CheckResult(
            name="ML Features",
            passed=False,
            message="Not installed (pip install -e '.[ml]')",
        ))

    return results


def _check_database() -> CheckResult:
    """Check database connectivity and migrations."""
    try:
        from episodic.db_connection import get_connection, DB_PATH
        from episodic.migrations import MigrationRunner

        # Check if database exists
        db_exists = os.path.exists(DB_PATH)

        if not db_exists:
            return CheckResult(
                name="Database",
                passed=True,
                message=f"Will be created at {DB_PATH}",
                details="Run 'python -m episodic --init' to initialize",
            )

        # Check connection and get applied migrations
        with get_connection() as conn:
            runner = MigrationRunner(conn)
            applied = runner.get_applied_migrations()

            if applied:
                latest_applied = max(applied)
                return CheckResult(
                    name="Database",
                    passed=True,
                    message=f"OK ({len(applied)} migrations, latest v{latest_applied})",
                    details=str(DB_PATH),
                )
            else:
                return CheckResult(
                    name="Database",
                    passed=True,
                    message="OK (no migrations yet)",
                    details=str(DB_PATH),
                )

    except Exception as e:
        return CheckResult(
            name="Database",
            passed=False,
            message="Connection failed",
            details=str(e),
        )


def _check_api_keys() -> List[CheckResult]:
    """Check which API keys are configured."""
    results = []

    # List of (env_var, config_key, provider_name)
    api_keys = [
        ("OPENAI_API_KEY", "openai_api_key", "OpenAI"),
        ("ANTHROPIC_API_KEY", "anthropic_api_key", "Anthropic"),
        ("GOOGLE_API_KEY", "google_api_key", "Google"),
        ("HUGGINGFACE_API_KEY", "huggingface_api_key", "Hugging Face"),
        ("TOGETHER_API_KEY", "together_api_key", "Together AI"),
        ("OPENROUTER_API_KEY", "openrouter_api_key", "OpenRouter"),
        ("BRAVE_API_KEY", "brave_api_key", "Brave Search"),
        ("PICOVOICE_ACCESS_KEY", "porcupine_access_key", "Picovoice (wake word)"),
    ]

    # Try to load config
    config_values = {}
    try:
        from episodic.config import config
        for _, config_key, _ in api_keys:
            val = config.get(config_key)
            if val:
                config_values[config_key] = val
    except Exception:
        pass

    configured = []
    missing = []

    for env_var, config_key, provider_name in api_keys:
        # Check environment variable first, then config
        env_value = os.environ.get(env_var)
        config_value = config_values.get(config_key)

        if env_value or config_value:
            source = "env" if env_value else "config"
            configured.append(f"{provider_name} ({source})")
        else:
            missing.append(provider_name)

    if configured:
        results.append(CheckResult(
            name="API Keys (configured)",
            passed=True,
            message=", ".join(configured),
        ))

    if missing:
        # Only show first few missing to avoid clutter
        shown = missing[:4]
        more = f" +{len(missing) - 4} more" if len(missing) > 4 else ""
        results.append(CheckResult(
            name="API Keys (optional)",
            passed=True,  # Not having all keys is OK
            message=f"Not set: {', '.join(shown)}{more}",
        ))

    # Check if at least one LLM provider is configured
    llm_providers = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
                     "HUGGINGFACE_API_KEY", "TOGETHER_API_KEY", "OPENROUTER_API_KEY"]
    has_llm = any(os.environ.get(k) or config_values.get(k.lower().replace("_api_key", "_api_key"))
                  for k in llm_providers)

    # Check for Ollama as alternative
    ollama_available = False
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        ollama_available = result.returncode == 0
    except Exception:
        pass

    if not has_llm and not ollama_available:
        results.append(CheckResult(
            name="LLM Provider",
            passed=False,
            message="No LLM provider configured",
            details="Set an API key or install Ollama",
        ))
    elif ollama_available and not has_llm:
        results.append(CheckResult(
            name="LLM Provider",
            passed=True,
            message="Ollama available (local)",
        ))

    return results


def _check_chromadb() -> Optional[CheckResult]:
    """Check ChromaDB status if RAG is enabled."""
    try:
        from pathlib import Path
        from episodic.config import config  # noqa: F401

        # Default ChromaDB path
        chroma_path = Path.home() / ".episodic" / "rag" / "chroma"

        if not chroma_path.exists():
            return CheckResult(
                name="ChromaDB",
                passed=True,
                message=f"Will be created at {chroma_path}",
            )

        # Try to connect and get stats
        try:
            from episodic.rag_collections import get_multi_collection_rag, CollectionType
            rag = get_multi_collection_rag()

            # Get collection stats
            stats = []
            for ctype in [CollectionType.CONVERSATION, CollectionType.DOCUMENT, CollectionType.WEB]:
                try:
                    collection = rag.get_collection(ctype)
                    if collection:
                        count = collection.count()
                        if count > 0:
                            stats.append(f"{ctype.value}:{count}")
                except Exception:
                    pass

            if stats:
                return CheckResult(
                    name="ChromaDB",
                    passed=True,
                    message=f"OK ({', '.join(stats)} entries)",
                )
            else:
                return CheckResult(
                    name="ChromaDB",
                    passed=True,
                    message="OK (empty)",
                )
        except Exception as e:
            # ChromaDB exists but couldn't connect
            return CheckResult(
                name="ChromaDB",
                passed=True,
                message=f"Directory exists at {chroma_path}",
            )

    except ImportError:
        return None  # RAG not installed
    except Exception as e:
        return CheckResult(
            name="ChromaDB",
            passed=False,
            message="Connection failed",
            details=str(e),
        )


def _check_config_file() -> CheckResult:
    """Check if user config file exists."""
    config_path = os.path.expanduser("~/.episodic/config.json")

    if os.path.exists(config_path):
        try:
            import json
            with open(config_path) as f:
                data = json.load(f)
            key_count = len(data)
            return CheckResult(
                name="Config File",
                passed=True,
                message=f"~/.episodic/config.json ({key_count} settings)",
            )
        except Exception as e:
            return CheckResult(
                name="Config File",
                passed=False,
                message="Invalid JSON",
                details=str(e),
            )
    else:
        return CheckResult(
            name="Config File",
            passed=True,
            message="Using defaults (no ~/.episodic/config.json)",
        )


def doctor_command(verbose: bool = False):
    """
    Run installation health checks.

    Verifies that Episodic is properly installed and configured.
    """
    from rich.console import Console

    console = Console()

    console.print("\n[bold cyan]Episodic Health Check[/bold cyan]")
    console.print("=" * 50)

    all_results: List[CheckResult] = []

    # Python version
    all_results.append(_check_python_version())

    # Core dependencies
    core_results = _check_core_dependencies()
    if verbose:
        all_results.extend(core_results)
    else:
        # Summarize core deps
        passed = sum(1 for r in core_results if r.passed)
        total = len(core_results)
        if passed == total:
            all_results.append(CheckResult(
                name="Core Dependencies",
                passed=True,
                message=f"All {total} packages OK",
            ))
        else:
            all_results.append(CheckResult(
                name="Core Dependencies",
                passed=False,
                message=f"{passed}/{total} packages OK",
            ))
            all_results.extend([r for r in core_results if not r.passed])

    # Database
    all_results.append(_check_database())

    # Config file
    all_results.append(_check_config_file())

    # API keys
    all_results.extend(_check_api_keys())

    # Optional features
    console.print("\n[bold]Optional Features:[/bold]")
    all_results.extend(_check_optional_features())

    # ChromaDB (if RAG available)
    chroma_result = _check_chromadb()
    if chroma_result:
        all_results.append(chroma_result)

    # Display results
    console.print()

    passed_count = 0
    failed_count = 0

    for result in all_results:
        if result.passed:
            passed_count += 1
            icon = "[green]✓[/green]"
        else:
            failed_count += 1
            icon = "[red]✗[/red]"

        console.print(f"  {icon} [bold]{result.name}[/bold]: {result.message}")
        if result.details and (verbose or not result.passed):
            console.print(f"      [dim]{result.details}[/dim]")

    # Summary
    console.print()
    console.print("=" * 50)

    if failed_count == 0:
        console.print(f"[bold green]All checks passed![/bold green] ({passed_count} checks)")
        console.print("\n[dim]Run 'python -m episodic' to start.[/dim]")
    else:
        console.print(f"[bold yellow]{passed_count} passed, {failed_count} issues[/bold yellow]")
        console.print("\n[dim]Fix issues above, or they may be optional features you don't need.[/dim]")

    console.print()


def doctor(verbose: str = None):
    """
    Entry point for /doctor command.

    Args:
        verbose: If "verbose" or "-v", show detailed output
    """
    is_verbose = verbose in ("verbose", "-v", "v", "true", "1")
    doctor_command(verbose=is_verbose)
