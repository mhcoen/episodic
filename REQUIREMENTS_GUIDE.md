# Requirements Files Guide

This project provides different requirements files for different use cases:

## requirements.txt
The main requirements file with all dependencies needed for full functionality.
- Use this for normal installation and development
- Includes all features: LLM integration, RAG, web search, visualization, etc.
- Total: ~120 dependencies (cleaned from original 383)

## requirements_minimal.txt  
Minimal core dependencies for basic CLI functionality only.
- Use this for lightweight installations or testing
- Includes only: CLI interface, database, basic LLM support
- No RAG (ChromaDB), web search, or visualization features
- Total: ~59 dependencies

## Installation

For full installation:
```bash
pip install -r requirements.txt
```

For minimal installation:
```bash
pip install -r requirements_minimal.txt
```

## Note on requirements_cleaned.txt
This file is identical to requirements.txt and was created during the dependency cleanup process. It has been removed to avoid confusion.

## Dependency Cleanup History
In 2025-07-13, the dependencies were audited and reduced:
- Original: 383 dependencies (including 150+ unnecessary PyObjC packages)
- Cleaned: ~120 dependencies 
- Reduction: 69%

The cleanup removed platform-specific packages that were unnecessarily included, particularly macOS-specific PyObjC packages that were not used by the application.