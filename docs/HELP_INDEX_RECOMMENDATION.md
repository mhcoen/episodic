# Documentation RAG Index Recommendation

## Best Files for Online Help RAG Index

### Primary Documentation (Must Index)

1. **USER_GUIDE.md** ⭐ HIGHEST PRIORITY
   - Complete user-facing documentation
   - All commands with examples
   - Common workflows and use cases
   - Troubleshooting section
   - ~1,000 lines of practical guidance

2. **docs/CLIReference.md** ⭐ HIGHEST PRIORITY
   - Comprehensive command reference
   - Detailed parameter explanations
   - Usage examples for every command
   - Model configuration details
   - ~800 lines of reference material

3. **QUICK_REFERENCE.md** ⭐ HIGH PRIORITY
   - Concise command overview
   - Essential workflows
   - Common patterns
   - Quick tips
   - ~200 lines of quick help

4. **CONFIG_REFERENCE.md** ⭐ HIGH PRIORITY
   - All configuration options
   - Environment variables
   - Default values
   - Configuration patterns
   - ~400 lines of configuration help

### Secondary Documentation (Recommended)

5. **README.md**
   - Project overview
   - Installation instructions
   - Basic usage
   - Feature list

6. **docs/LLMProviders.md**
   - Provider setup guides
   - API key configuration
   - Model availability

7. **docs/WebSearchProviders.md**
   - Web search setup
   - Provider configuration
   - API requirements

8. **docs/WEB_SYNTHESIS.md**
   - Muse mode details
   - Web search integration
   - Advanced features

### Specialized Documentation (Optional)

9. **docs/AdvancedUsage.md**
   - Power user features
   - Complex workflows
   - Performance optimization

10. **docs/Visualization.md**
    - Graph visualization
    - Export options
    - Customization

### Not Recommended for Help Index

- **CLAUDE.md** - Developer-focused, not user-facing
- **PROJECT_MEMORY.md** - Internal state tracking
- **TODO.md** - Development planning
- **DEPRECATED.md** - Legacy information
- **ARCHITECTURE.md** - Technical implementation
- **Development.md** - Contributor guide

## Recommended Implementation

```bash
# Index the primary documentation for help
/index docs/USER_GUIDE.md
/index docs/CLIReference.md
/index QUICK_REFERENCE.md
/index CONFIG_REFERENCE.md

# Optionally add secondary docs
/index README.md
/index docs/LLMProviders.md
/index docs/WebSearchProviders.md
/index docs/WEB_SYNTHESIS.md
```

## Benefits of This Approach

1. **Comprehensive Coverage**: The four primary docs cover 95% of user questions
2. **Structured Information**: Each doc has a specific focus (tutorial vs reference)
3. **Searchable Examples**: Hundreds of command examples and use cases
4. **Up-to-date**: Recently updated with all latest features
5. **User-focused**: Written for end users, not developers

## Usage Pattern

After indexing, users could:
- Ask "how do I change models?" → Get info from CLIReference.md
- Ask "what is muse mode?" → Get explanation from USER_GUIDE.md
- Ask "how to configure web search?" → Get details from CONFIG_REFERENCE.md
- Ask "quick command list?" → Get concise help from QUICK_REFERENCE.md

This would enable a `/help <query>` command that searches the indexed documentation to provide contextual help.