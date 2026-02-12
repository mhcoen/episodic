"""Help text for the gsuite plugin's Calendar & Email category."""


def show_calendar_email_help() -> str:
    """Return help text for Calendar & Email commands."""
    return """\
Calendar & Email (via Google Workspace)

  /cal <text>        Calendar query or action
  /email <text>      Email query or action

  Examples:
    /cal what's on my calendar tomorrow
    /cal schedule a meeting with Bob at 3pm
    /email check my unread
    /email draft to Jane about the report

  Aliases: /calendar, /mail, /gmail

  Commands are interpreted using natural language extraction.
  Connect with /mcp connect gsuite first."""
