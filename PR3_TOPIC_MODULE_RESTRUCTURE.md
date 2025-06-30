# PR #3: Topic Detection Module Restructure

## Summary
Reorganized topic detection code into a proper module structure for better maintainability and separation of concerns.

## Changes Made

### Created New Module Structure
```
episodic/topics/
├── __init__.py           # Public API and exports
├── detector.py           # Main TopicManager class
├── boundaries.py         # Boundary detection logic
├── hybrid.py            # Hybrid detector implementation
├── keywords.py          # Keyword-based detection
├── windows.py           # Sliding window implementation
└── utils.py             # Shared utility functions
```

### Module Organization

#### `detector.py`
- Moved `TopicManager` class with all its methods
- Module-level wrapper functions for backward compatibility
- Global `topic_manager` instance

#### `boundaries.py`
- `analyze_topic_boundary()` - LLM-based boundary analysis
- `find_transition_point_heuristic()` - Heuristic fallback
- Private helper functions for boundary detection

#### `hybrid.py`
- `HybridTopicDetector` class
- `HybridScorer` class
- Integration with drift and keyword detection

#### `keywords.py`
- `TransitionDetector` class
- `TopicChangeSignals` dataclass
- Domain-specific keyword mappings

#### `windows.py`
- `SlidingWindowDetector` class
- Window-based drift calculation
- Database storage integration

#### `utils.py`
- `build_conversation_segment()`
- `is_node_in_topic_range()`
- `count_nodes_in_topic()`
- `_display_topic_evolution()` wrapper

### Backward Compatibility
Updated original files to maintain imports:
- `episodic/topics.py` - Imports from new module structure
- `episodic/topics_hybrid.py` - Imports hybrid classes
- `episodic/topic_boundary_analyzer.py` - Imports boundary functions

## Benefits
1. **Better Organization**: Related functionality grouped together
2. **Easier Maintenance**: Clear separation of concerns
3. **Improved Testability**: Each module can be tested independently
4. **No Breaking Changes**: All existing imports continue to work
5. **Future Ready**: Easy to add new detection methods

## Testing
```bash
# Test backward compatibility
python -c "
from episodic.topics import TopicManager, topic_manager
from episodic.topics_hybrid import HybridTopicDetector
print('All imports successful')
"

# Test in application
python -m episodic
> /init
> Hello, how are you?
> /topics
```

## Next Steps
With the topic module properly structured, we can proceed to PR #4: Database Schema Cleanup.