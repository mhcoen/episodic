# Assistant Mode

Episodic includes a built-in assistant with utility commands for timers, alarms, reminders, weather, news, calculator, notes, media playback, calendar, and email. All commands work as slash commands in the CLI and as natural language in voice mode.

## Commands

### Time & Timers

```bash
/time                        # Show current time
/timer <duration> [label]    # Set a timer (e.g., /timer 5m coffee)
/timer                       # Show active timers
/alarm <time> [label]        # Set an alarm (e.g., /alarm 7am wake up)
/alarm                       # List active alarms
/remind <text> in/at <time>  # Set a reminder (e.g., /remind call mom in 1h)
/remind                      # List active reminders
/cancel [timer|alarm]        # Cancel a timer or alarm
```

Duration formats: `5s`, `5m`, `1h`, `1h30m`, or natural language like "five minutes".

Alarm formats: `7am`, `7:30am`, `14:00`, `3pm`, or natural language like "seven thirty".

### Weather

```bash
/weather [location]          # Current weather
/forecast [location]         # Weather forecast
```

Location defaults to your current position via IP geolocation. You can specify any city name or coordinates.

Requires an OpenWeatherMap API key:
```bash
/set weather-api-key <key>
```

### News

```bash
/news [category]             # Get news headlines
```

Categories: `general`, `tech`, `business`, `science`, `health`, `politics`, `world`.

### Calculator

```bash
/calc <expression>           # Calculate expression
```

Supports arithmetic, percentages, and common math functions:
```bash
/calc 15% of 85             # → 12.75
/calc sqrt(144) + 3^2       # → 21.0
/calc (100 - 20) * 1.08     # → 86.4
```

### Notes

```bash
/note <text>                 # Add a note
/note                        # List all notes
```

### Media

```bash
/play <station>              # Play radio station (e.g., /play npr)
/pause                       # Pause media playback
/stop                        # Stop current TTS or media
```

### System

```bash
/status                      # Show active timers, alarms, media state
/dnd [on|off|duration]       # Do not disturb mode
/undo                        # Undo last utility action
```

## Calendar & Email

Calendar and email commands are provided by the Google Workspace plugin and use natural language extraction — you describe what you want in plain English.

```bash
/cal <text>                  # Calendar query or action
/email <text>                # Email query or action
```

Aliases: `/calendar`, `/mail`, `/gmail`

### Examples

```bash
/cal what's on my calendar tomorrow
/cal schedule a meeting with Bob at 3pm
/cal am I free Friday afternoon

/email check my unread
/email from Jane about the report
/email draft to Bob about the project update
```

### Setup

Calendar and email require a connection to the Google Workspace MCP server:

```bash
/mcp connect gsuite          # Connect to Google Workspace
/mcp plugin gsuite           # Check connection status
```

## Voice Mode

All assistant commands work via natural language in voice mode. Enable voice with `/voice on`, then speak commands naturally:

- "Set a timer for five minutes"
- "What's the weather like?"
- "Any tech news today?"
- "What's on my calendar tomorrow?"
- "Check my email"

The system uses a two-stage pipeline:
1. **Keyword gate** — zero-cost domain matching identifies which utility is being invoked
2. **Grammar parse** (core commands) or **LLM extraction** (calendar/email) interprets the full command

## Configuration

```bash
# Weather
/set weather-api-key <key>          # OpenWeatherMap API key
/set weather-location <city>        # Default location
/set weather-units metric           # metric or imperial

# News
/set news-country us                # Country code for news

# Do Not Disturb
/set dnd-default-duration 30m       # Default DND duration
```
