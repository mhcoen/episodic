# Current State

## Architecture Overview

Episodic uses a modular architecture with clear separation of concerns:

- **Core Components**: Conversation DAG, Topic Detection, LLM Integration
- **Database**: SQLite with migration system
- **CLI**: Typer-based with command registry
- **Testing**: Comprehensive pytest suite

## Key Features Status

- ✅ **Multi-provider LLM support** via LiteLLM
- ✅ **Automatic topic detection** with sliding window
- ✅ **RAG system** with ChromaDB integration
- ✅ **Web search** with provider fallback
- ✅ **Markdown import/export**
- ✅ **Cost tracking** across all providers
- ✅ **Tab completion** support

## Known Issues

1. **Topic Boundary Detection**
   - Topics currently end at user messages (where change is detected)
   - Should end after assistant response for complete conversation pairs
   - Workaround implemented in export functionality

## Testing

- Run all tests: `python tests/run_all_tests.py all`
- Run specific categories: `unit`, `integration`, `quick`, `topics`
- CLI testing: `python tests/integration/cli/test_all_commands.py`
- Documentation: `tests/ORGANIZED_TESTS.md`

## Development Guidelines

- Maximum 500 lines per file (hard cap at 600)
- Use command registry for new commands
- Add migrations for database changes
- Follow existing patterns in codebase