# Help System Indexed Files

This file lists all documentation files that are indexed by the `/help` command for searching.

## Indexed Documentation Files

The following files are automatically indexed when you use `/help <query>`:

1. **USER_GUIDE.md** - Complete user guide with all features
2. **docs/CLIReference.md** - Detailed command reference
3. **QUICK_REFERENCE.md** - Quick command guide
4. **CONFIG_REFERENCE.md** - Configuration options guide
5. **README.md** - Project overview and setup
6. **docs/LLMProviders.md** - Language model provider setup
7. **docs/WebSearchProviders.md** - Web search provider configuration
8. **docs/WEB_SYNTHESIS.md** - Muse mode documentation

## Reindexing Documentation

To manually reindex the help documentation (useful after updates):

```
/help-reindex
```

This command will:
- Clear the existing help documentation index
- Re-read all files listed above
- Index them with format-preserving chunking
- Show progress as it indexes each file

## Adding New Documentation

To add a new file to the help system:

1. Add the file path to the `help_docs` list in `episodic/commands/help.py`
2. Run `/help-reindex` to include it in the search index

## Technical Details

- The help system uses a separate ChromaDB collection (`episodic_help`)
- Documents are indexed with format-preserving chunking to maintain code examples and command alignment
- The index is persistent across sessions
- Duplicate detection prevents re-indexing unchanged files