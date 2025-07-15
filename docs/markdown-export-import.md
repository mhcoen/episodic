# Markdown Export/Import Documentation

## Overview

Episodic provides markdown export and import functionality to save and resume conversations. This feature allows you to:
- Export conversations as human-readable markdown files
- Import markdown conversations from Episodic or other sources
- Share conversations with others
- Create conversation templates
- Backup important discussions

## Export Command (`/export`)

### Basic Syntax

```bash
/export [topic-spec] [filename]
```

### Topic Specifications

| Specification | Description | Example |
|--------------|-------------|---------|
| *(empty)* | Export current topic | `/export` |
| `current` | Export current topic (explicit) | `/export current` |
| `N` | Export topic number N | `/export 3` |
| `N-M` | Export topic range N through M | `/export 1-5` |
| `N,M,P` | Export specific topics | `/export 1,3,5` |
| `all` | Export all topics | `/export all` |

### Examples

```bash
# Export current topic with auto-generated filename
/export

# Export current topic to specific file
/export current my-research.md

# Export a single topic
/export 3
/export 3 topic-three.md

# Export a range of topics
/export 1-5
/export 1-5 week-summary.md

# Export specific topics
/export 1,3,5 selected-topics.md

# Export all topics
/export all
/export all complete-conversation.md
```

### Default Behavior

- **Filename**: Auto-generated as `{topic-name}-{date}.md` 
  - Example: `mars-exploration-2025-01-15.md`
- **Directory**: Saved to `exports/` subdirectory
- **Extension**: `.md` is added automatically if not provided

## Import Command (`/import`)

### Basic Syntax

```bash
/import <filename>
```

### Examples

```bash
# Import from exports directory
/import exports/mars-exploration-2025-01-15.md

# Import from current directory
/import conversation.md

# Import from absolute path
/import ~/Documents/research-notes.md

# Import from relative path
/import ../backups/old-chat.md
```

### Import Behavior

- **Linear Only**: Always creates new nodes, maintaining linear conversation flow
- **No Branching**: Does not create branches in the conversation DAG
- **Continues Current**: Appends imported content to current conversation position
- **Topic Creation**: Creates new topics for imported content

### Supported Formats

The import command accepts various markdown dialogue formats:

**Speaker Patterns** (all are recognized):
- `**You**:` / `**User**:` / `**Human**:` → User messages
- `**LLM**:` / `**Assistant**:` / `**AI**:` → Assistant messages
- `**System**:` → System messages
- `**Web**:` → Web search results (future)
- `**RAG**:` → RAG retrieval results (future)

**Also supports non-bold variants**:
- `You:` / `User:` / `Human:`
- `LLM:` / `Assistant:` / `AI:`

## Export Format

### Structure

```markdown
# {Title based on topics}
*{Date}*
*Model: {model} • Style: {style} • Format: {format}*

## {Topic Name}
*{N} messages*

**You**: {User message}

**LLM**: {Assistant response}

**You**: {Another user message}

**LLM**: {Another assistant response}

---

## {Next Topic}
*{M} messages*
*Model changed to {new-model}*

**You**: {User message}

**LLM**: {Assistant response}

---

<!-- 
Exported from Episodic on {timestamp}
Note: Re-importing will create new nodes, not reuse existing ones
-->
```

### Special Cases

**Interrupted Responses**:
```markdown
**LLM**: [Response interrupted by user]

The response text that was generated before interruption...
```

**Model Changes**:
```markdown
*Model changed to gpt-3.5-turbo*
```

## Configuration

### Hardcoded Values

Currently, the following values are hardcoded in the implementation:

| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| `export_dir` | `"exports"` | `markdown_export.py` | Default export directory |
| `style` | `"standard"` | Export header | Placeholder for actual style |
| `format` | `"mixed"` | Export header | Placeholder for actual format |

### Future Configuration Options

These could be made configurable in future versions:
- Export directory path
- Include/exclude metadata in export
- Include node IDs in export
- Custom filename patterns
- Style and format detection from conversation state

## Use Cases

### 1. Backup Important Conversations
```bash
# Export current research topic
/export current research-backup.md

# Export all topics for full backup
/export all full-backup-2025-01-15.md
```

### 2. Share Conversations
```bash
# Export specific topics to share
/export 1-3 meeting-notes.md

# Send the file: exports/meeting-notes.md
```

### 3. Create Templates
```markdown
# Create a markdown file with conversation structure
## Code Review Session

**You**: Please review this Python function for potential issues

**LLM**: I'll analyze the function for bugs, performance, and style...

# Then import it
/import templates/code-review.md
```

### 4. Continue Conversations
```bash
# Export before closing Episodic
/export all my-work.md

# Later, import to continue
/import exports/my-work.md
```

## Limitations

1. **No Node Reuse**: Re-importing an Episodic export creates new nodes
2. **No Branching**: Cannot create conversation branches
3. **Linear Only**: Always appends to current position
4. **No Merge**: Cannot merge multiple conversation files

## Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `No topics found for specification: current` | No current/ongoing topic | Start a conversation first |
| `Invalid topic specification: 'abc'` | Invalid topic format | Use number, range (1-3), or 'all' |
| `File not found: {path}` | Import file doesn't exist | Check file path and name |
| `No conversation content found in file` | Empty or invalid markdown | Ensure file has proper format |

## Technical Notes

- Topics are numbered starting from 1 (not 0)
- Export creates directories if they don't exist
- Import continues from current `conversation_manager.current_node_id`
- Node IDs in exports are informational only (not used on import)
- All imports create new entries in the database