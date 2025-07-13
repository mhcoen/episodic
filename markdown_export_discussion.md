# Markdown Export/Import Discussion

## Overview
Discussion about implementing markdown export/import functionality for Episodic, focusing on making conversations persistent and browsable while keeping the system simple.

## Key Design Decisions

### User's Vision
- Each markdown file should be its own self-contained thing
- No embedded metadata - just user prompts and system responses
- Readable as-is, suitable for documentation/knowledge base
- Should work for both archiving AND continuing conversations
- Address the problem of LLM conversations "disappearing or quickly becoming inaccessible"

### Proposed Approach
```markdown
# API Design Discussion

**User**: What are the key differences between REST and GraphQL?

**Assistant**: Here are the main differences...

**User**: How about rate limiting?

**Assistant**: For rate limiting...
```

### Storage Strategies Considered
1. **Daily store** - Everything from today
2. **Current topic store** - When exiting
3. **User-named store** - Explicit saves
4. **Auto-assigned topic store** - Based on detected topics

### Simplified Command Proposal
Just TWO commands for 90% of users:
```bash
/keep              # Saves current topic/conversation with smart naming
/keep project-name # Saves with explicit name
```

### File Organization
```
~/episodic/
├── kept/
│   ├── 2025-01-12-api-design.md
│   ├── auth-patterns.md
│   └── client-requirements.md
├── daily/
│   └── 2025-01-12.md
└── current.md  # Always has your active session
```

## Technical Challenges

### Topic Detection Lag
- New topics aren't detected until 3-6 messages deep (3,3 window)
- Topic boundaries are detected retrospectively, not in real-time
- This makes "save current topic" ambiguous
- Complicates clean exports when topic boundaries aren't yet established

### State Management Options
1. **Pure markdown** - No metadata, filesystem is the database
2. **Dual-layer** - Clean markdown + hidden metadata store
3. **Minimal inline metadata** - Using conventions like date headers

### Continuation Strategies
- Append to existing files with continuation markers
- Create new files with references to previous discussions
- Let the system scan files to rebuild context

## UX Concerns
- System already has many commands - don't want to discourage new users
- Need progressive disclosure - simple defaults with power user options
- Most users should just need "save" and "continue" functionality
- Auto-save on exit should be smart (discard empty/trivial conversations)

## Open Questions
1. How to handle the topic detection lag when saving?
2. Should saves be conversation-window based or topic-based?
3. How much state needs to be preserved for continuations?
4. Should loaded conversations be append-only or allow full continuation?

## Next Steps
- Design the minimal command set
- Prototype the markdown format
- Test the continuation workflow
- Consider how this integrates with future DAG branching