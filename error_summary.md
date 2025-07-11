# Episodic CLI Error Summary

## Statistics
- Total commands tested: ~60
- Passed: 23
- Failed: 37

## Issues to Fix

### Unknown Commands
These commands are not recognized and need to be implemented or removed from help:

- [ ] `/about`
- [ ] `/config`
- [ ] `/graph`
- [ ] `/h`
- [ ] `/history`
- [ ] `/history 5`
- [ ] `/history all`
- [ ] `/tree`
- [ ] `/welcome`

### Import Errors
These commands fail due to missing modules or incorrect imports:

- [ ] `/model`: 'module' object is not callable[0m
- [ ] `/compression`: cannot import name 'compression' from 'episodic.commands.unified_compression' (/Users/mhcoen/proj/episodic/episodic/commands/unified_compression.py)[0m
- [ ] `/rag`: cannot import name 'rag' from 'episodic.commands.rag' (/Users/mhcoen/proj/episodic/episodic/commands/rag.py)[0m
- [ ] `/index README.md`: cannot import name 'index' from 'episodic.commands.rag' (/Users/mhcoen/proj/episodic/episodic/commands/rag.py)[0m
- [ ] `/i README.md`: cannot import name 'index' from 'episodic.commands.rag' (/Users/mhcoen/proj/episodic/episodic/commands/rag.py)[0m
- [ ] `/docs`: cannot import name 'docs' from 'episodic.commands.rag' (/Users/mhcoen/proj/episodic/episodic/commands/rag.py)[0m
- [ ] `/muse`: No module named 'episodic.commands.muse'[0m

### Attribute Errors
These commands fail due to missing methods or properties:

- [ ] `/topics compress`: 'TopicManager' object has no attribute 'get_current_topic'[0m

### Type Errors
These commands fail due to incorrect function signatures:

- [ ] `/topics scores`: Error binding parameter 1: type 'OptionInfo' is not supported[0m
- [ ] `/websearch on`: websearch() got an unexpected keyword argument 'action'[0m
- [ ] `/websearch off`: websearch() got an unexpected keyword argument 'action'[0m
- [ ] `/websearch config`: websearch() got an unexpected keyword argument 'action'[0m
- [ ] `/websearch stats`: websearch() got an unexpected keyword argument 'action'[0m
- [ ] `/websearch cache clear`: websearch() got an unexpected keyword argument 'action'[0m

