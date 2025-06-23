# Organized Test Suite

All test files have been organized into a structured directory for better maintainability.

## Directory Structure

```
tests/
├── __init__.py                    # Test package
├── ORGANIZED_TESTS.md            # This file
├── README.md                     # Comprehensive test documentation
├── run_tests.py                  # Enhanced test runner
├── cleanup_tests.py              # Test organization script
│
├── Core Unit Tests:
├── test_cli.py                   # CLI interface tests (some failing)
├── test_config.py                # Configuration management tests ✅
├── test_core.py                  # Core data structures tests
├── test_db.py                    # Database operations tests
├── test_integration.py           # System integration tests
├── test_llm_integration.py       # LLM provider tests ✅
├── test_prompt_manager.py        # Prompt management tests ✅
├── test_caching.py               # Prompt caching tests ✅
├── test_server.py                # HTTP server tests
├── test_websocket.py             # WebSocket functionality tests
│
├── interactive/                  # Manual/Interactive Tests
│   ├── test_interactive_features.py    # Interactive visualization tests
│   ├── test_websocket_browser.py       # Browser WebSocket tests
│   ├── test_websocket_integration.py   # WebSocket integration tests
│   └── test_native_visualization.py    # Native visualization tests
│
└── legacy/                       # Legacy/One-off Tests
    ├── test_cache_comprehensive.py     # Comprehensive cache tests
    ├── test_prompt_caching_final.py    # Final prompt caching test
    ├── test_episodic.py                # General episodic tests
    ├── test_head_reference.py          # Head reference tests
    ├── test_roles.py                   # Role system tests
    ├── test_short_ids.py               # Short ID tests
    └── ... (20+ legacy test files)
```

## Test Status

### ✅ Passing Tests
- `test_config.py` - Configuration management
- `test_prompt_manager.py` - Prompt loading and management
- `test_llm_integration.py` - LLM provider integration
- `test_caching.py` - Prompt caching functionality

### ⚠️ Failing Tests (Need Fixes)
- `test_cli.py` - 7 failures, 6 errors
  - Issues with CLI function behavior vs expected test behavior
  - Mock setup issues
  - Function signature mismatches

### 🧪 Integration Tests
- `test_core.py` - Core data structures
- `test_db.py` - Database operations  
- `test_integration.py` - System integration
- `test_server.py` - HTTP server functionality
- `test_websocket.py` - WebSocket functionality

### 🎮 Interactive Tests (Manual)
- `test_interactive_features.py` - Requires human verification
- `test_websocket_browser.py` - Browser-based testing
- `test_websocket_integration.py` - Real-time WebSocket tests
- `test_native_visualization.py` - Visualization components

## Running Tests

### Quick Tests (Stable Only)
```bash
cd tests
python run_tests.py quick
```

### All Tests
```bash
python run_tests.py
```

### Specific Test Files
```bash
python -m unittest test_config -v
python -m unittest test_prompt_manager -v
```

### With Coverage
```bash
pip install coverage
python run_tests.py coverage
```

### Individual Categories

**Core functionality only:**
```bash
python -m unittest test_config test_prompt_manager test_llm_integration test_caching
```

**Database and integration:**
```bash
python -m unittest test_db test_core test_integration
```

**Legacy tests:**
```bash
python -m unittest discover legacy/ "test_*.py"
```

## Benefits of Organization

1. **Clear Structure**: Tests are logically organized by functionality
2. **Easy Maintenance**: Related tests are grouped together
3. **Selective Testing**: Run only the tests you need
4. **Stability**: Separate failing tests from stable ones
5. **Documentation**: Clear categorization of test types

## Test Development

### Adding New Tests
1. Create tests in the main `tests/` directory for core functionality
2. Use `interactive/` for tests requiring human verification
3. Use `legacy/` for experimental or one-off tests

### Fixing Failing Tests
1. Focus on `test_cli.py` first - it has the most failures
2. Update mocks to match actual function signatures
3. Verify expected vs actual behavior
4. Update test assertions as needed

### Before Major Changes
```bash
# Run stable tests first
python run_tests.py quick

# If those pass, run full suite
python run_tests.py

# Generate coverage report
python run_tests.py coverage
```

This organized structure provides a solid foundation for maintaining test quality while you make significant modifications to the CLI application.