# Theme Switcher Implementation Plan

## Project Overview
Implement a comprehensive theme switching system for Episodic that allows users to select from multiple color themes and have their choice persist across sessions.

## Implementation Phases

### Phase 1: Extend Color System Architecture

#### 1.1 Expand Color Categories in configuration.py
Add new semantic color categories to COLOR_SCHEMES:
- `error` - For error messages (currently hardcoded as "red")
- `warning` - For warnings (currently hardcoded as "yellow")
- `success` - For success messages (currently hardcoded as "green")
- `info` - For informational messages (often "cyan")
- `accent` - For emphasis and highlights
- `dim` - For de-emphasized text
- `model_name` - For model names in listings
- `price` - For pricing information
- `metric` - For metrics and statistics

#### 1.2 Create New Helper Functions
Add to configuration.py:
```python
def get_error_color() -> str
def get_warning_color() -> str
def get_success_color() -> str
def get_info_color() -> str
def get_accent_color() -> str
def get_dim_color() -> str
def get_model_color() -> str
def get_price_color() -> str
def get_metric_color() -> str
```

#### 1.3 Define New Color Themes
Expand COLOR_SCHEMES with:
- `gemini` - Dark background with blues, teals, and magenta/orange accents
- `solarized_dark` - Classic Solarized dark palette
- `solarized_light` - Classic Solarized light palette
- `monokai` - Popular code editor theme
- `dracula` - Modern dark theme
- `nord` - Cool, muted Nordic theme
- `gruvbox` - Retro groove theme

### Phase 2: Implement Theme Command

#### 2.1 Create episodic/commands/theme.py
```python
# Core functions:
def theme_command(theme_name: Optional[str] = None)
def list_themes()
def preview_theme(theme_name: str)
def set_theme(theme_name: str)
def get_theme_preview_text(theme_name: str) -> str
```

#### 2.2 Theme Listing with Previews
When user runs `/theme` or `/theme list`:
- Display all available themes
- For each theme, show a color preview with:
  - Theme name in heading color
  - Sample text in various colors
  - Example of each color category
  - Visual separator between themes

#### 2.3 Theme Setting Logic
When user runs `/theme <name>`:
1. Validate theme exists
2. Apply theme immediately (update runtime config)
3. Save to config.json for persistence
4. Display confirmation with sample colors

### Phase 3: Command Registration and Integration

#### 3.1 Register in cli_command_router.py
Add to handle_command():
```python
elif cmd == "theme":
    return handle_theme(args)
```

#### 3.2 Add to Command Registry
Update episodic/commands/registry.py:
```python
command_registry.register(
    "theme", 
    theme_command, 
    "Manage color themes", 
    "Configuration"
)
```

#### 3.3 Tab Completion Support
Update episodic/cli_completer.py:
- Add theme name completion
- Complete available theme names after `/theme `

### Phase 4: Migrate Hardcoded Colors

#### 4.1 High Priority Files (Core Display)
1. **episodic/cli_display.py**
   - Replace `typer.colors.BRIGHT_CYAN` → `get_accent_color()`
   - Replace `typer.colors.YELLOW` → `get_warning_color()`

2. **episodic/cli_registry.py**
   - Replace `fg="red"` → `fg=get_error_color()`
   - Replace `fg="yellow"` → `fg=get_warning_color()`
   - Replace `fg="cyan"` → `fg=get_info_color()`

3. **episodic/unified_streaming.py**
   - Replace `fg="yellow"` → `fg=get_warning_color()`

#### 4.2 Command Files
1. **episodic/commands/unified_model.py**
   - Type colors: Create semantic mappings
   - Replace price colors with `get_price_color()`
   - Replace model name colors with `get_model_color()`

2. **episodic/commands/model.py**
   - Similar replacements as unified_model.py

3. **Error/Warning/Success patterns**
   - All `fg="red"` → `fg=get_error_color()`
   - All `fg="yellow"` → `fg=get_warning_color()`
   - All `fg="green"` → `fg=get_success_color()`

#### 4.3 Benchmark Display
**episodic/benchmark.py** - Special consideration:
- Create benchmark-specific color helpers
- Or add benchmark color categories to themes

### Phase 5: Theme Definitions

#### 5.1 Gemini Theme
```python
"gemini": {
    "heading": "BRIGHT_CYAN",
    "text": "WHITE",
    "llm_response": "BRIGHT_BLUE",
    "system_info": "CYAN",
    "prompt": "BRIGHT_MAGENTA",
    "error": "BRIGHT_RED",
    "warning": "YELLOW",
    "success": "BRIGHT_GREEN",
    "info": "CYAN",
    "accent": "BRIGHT_MAGENTA",
    "dim": "BRIGHT_BLACK",
    "model_name": "BRIGHT_YELLOW",
    "price": "GREEN",
    "metric": "BRIGHT_CYAN"
}
```

#### 5.2 Other Themes
Define complete color mappings for:
- solarized_dark
- solarized_light
- monokai
- dracula
- nord
- gruvbox

### Phase 6: Testing and Polish

#### 6.1 Test Cases
1. Theme listing displays correctly
2. Theme preview shows accurate colors
3. Theme setting persists across sessions
4. All commands respect theme colors
5. Invalid theme names handled gracefully
6. Tab completion works for theme names

#### 6.2 Documentation
1. Update README with theme information
2. Add theme examples to help text
3. Document custom theme creation

#### 6.3 Edge Cases
1. Handle terminal color limitations
2. Provide fallback for unsupported colors
3. Consider accessibility (colorblind-friendly themes)

## Implementation Order

1. **Day 1**: Extend color system (Phase 1)
   - Add new color categories
   - Create helper functions
   - Define all theme color schemes

2. **Day 2**: Implement theme command (Phase 2)
   - Create theme.py
   - Implement list/preview/set functionality
   - Test basic functionality

3. **Day 3**: Integration (Phase 3)
   - Register command
   - Add tab completion
   - Test persistence

4. **Day 4-5**: Color migration (Phase 4)
   - Migrate core display files
   - Update command files
   - Handle special cases

5. **Day 6**: Testing and refinement (Phase 6)
   - Comprehensive testing
   - Documentation
   - Final polish

## Technical Considerations

1. **Color Detection**: Some terminals don't support all colors. Consider using `colorama` or terminal capability detection.

2. **Theme Validation**: Ensure all required color categories are defined in each theme.

3. **Backwards Compatibility**: Maintain support for existing "dark" and "light" themes.

4. **Performance**: Color lookups should be fast - consider caching the active theme.

5. **Customization**: Consider allowing users to create custom themes by editing config.json directly.

## Success Criteria

1. Users can list all available themes with color previews
2. Theme selection persists across sessions
3. All UI elements respect the selected theme
4. No hardcoded colors remain in the codebase
5. Tab completion works for theme names
6. Documentation is complete and helpful

## Future Enhancements

1. **Theme Editor**: `/theme edit` command to customize themes
2. **Theme Export/Import**: Share themes as JSON files
3. **Auto Theme**: Switch based on terminal background or time of day
4. **Theme Variations**: Light/dark variants of each theme
5. **Accessibility Themes**: High contrast, colorblind-safe options