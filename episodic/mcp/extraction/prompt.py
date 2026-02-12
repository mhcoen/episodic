"""Extraction prompt assembler for MCP intent extraction.

Builds the system prompt for the extraction LLM call, scoped to
matched domains only. Follows the prompt structure from spec section 3.3.
"""

from typing import Dict, List, Optional, Set

from episodic.mcp.extraction.types import ArgDefinition, IntentDefinition


# --- Hardcoded gsuite intent definitions ---

GSUITE_INTENTS: Dict[str, IntentDefinition] = {
    "calendar.query": IntentDefinition(
        intent_id="cal_query",
        name="calendar.query",
        description="Check schedule, find events, check if free/busy",
        action_class="read",
        args={
            "query": ArgDefinition(
                type="string",
                description="What to search for, e.g. \"doctor appointment\"",
                examples=["doctor appointment", "team standup", "1:1 with Bob"],
            ),
            "time_range": ArgDefinition(
                type="string",
                description="When, e.g. \"tomorrow\", \"next week\", \"March 15\"",
                examples=["tomorrow", "next week", "Thursday afternoon"],
            ),
        },
        examples=[
            "Do I have a meeting tomorrow?",
            "Am I free Thursday afternoon?",
            "What's on my calendar next week?",
            "When's my next doctor appointment?",
            "Do I have anything scheduled for Friday?",
        ],
        negative_examples=[
            "I hate meetings",
            "Meetings are so boring",
            "I wish I had fewer appointments",
        ],
    ),
    "calendar.create": IntentDefinition(
        intent_id="cal_create",
        name="calendar.create",
        description="Create a new calendar event",
        action_class="write",
        args={
            "summary": ArgDefinition(
                type="string",
                description="Event title",
                required=True,
                examples=["meeting with Bob", "dentist appointment"],
            ),
            "start": ArgDefinition(
                type="string",
                description="Start time in natural language",
                examples=["3pm tomorrow", "Thursday at 2"],
            ),
            "end": ArgDefinition(
                type="string",
                description="End time or duration",
                examples=["4pm", "30 minutes"],
            ),
            "attendees": ArgDefinition(
                type="list",
                description="People to invite",
                examples=["bob", "jane, bob"],
            ),
            "location": ArgDefinition(
                type="string",
                description="Where",
                examples=["Room 201", "Zoom"],
            ),
        },
        examples=[
            "Schedule a meeting with Bob at 3pm tomorrow",
            "Book a 30 minute call with the design team on Thursday",
        ],
        negative_examples=[
            "I should schedule something eventually",
            "I need to book more meetings in general",
        ],
    ),
    "email.search": IntentDefinition(
        intent_id="eml_search",
        name="email.search",
        description="Search or filter emails",
        action_class="read",
        args={
            "query": ArgDefinition(
                type="string",
                description="Search terms",
                examples=["budget report", "Q3 numbers"],
            ),
            "from_addr": ArgDefinition(
                type="string",
                description="Sender name or address",
                examples=["jane", "bob@company.com"],
            ),
            "unread_only": ArgDefinition(
                type="boolean",
                description="Only unread messages",
            ),
        },
        examples=[
            "Check my email",
            "Anything from Jane about the budget?",
            "Any unread messages?",
            "Any unread emails from Jane?",
        ],
        negative_examples=[
            "Email is so annoying",
            "I get too much email",
        ],
    ),
    "email.read": IntentDefinition(
        intent_id="eml_read",
        name="email.read",
        description="Read a specific email (requires prior search context)",
        action_class="read",
        args={
            "ref": ArgDefinition(
                type="string",
                description="Which email, e.g. \"the first one\", \"that one from Bob\"",
                examples=["the first one", "that one from Bob", "1"],
            ),
        },
        examples=[
            "Read the first one",
            "Show me that email from Bob",
        ],
        negative_examples=[],
    ),
    "email.draft": IntentDefinition(
        intent_id="eml_draft",
        name="email.draft",
        description="Draft a new email",
        action_class="draft",
        args={
            "to": ArgDefinition(
                type="string",
                description="Recipient name or address",
                required=True,
                examples=["bob", "jane@company.com"],
            ),
            "subject": ArgDefinition(
                type="string",
                description="Subject line",
                examples=["Q3 numbers", "Meeting follow-up"],
            ),
            "body": ArgDefinition(
                type="string",
                description="Message body",
                examples=["The report is ready", "Can we reschedule?"],
            ),
        },
        examples=[
            "Draft an email to Bob about the Q3 numbers",
            "Write a message to Jane saying the report is ready",
        ],
        negative_examples=[
            "I should email someone about this",
        ],
    ),
    "email.reply": IntentDefinition(
        intent_id="eml_reply",
        name="email.reply",
        description="Reply to a recent email",
        action_class="write",
        args={
            "body": ArgDefinition(
                type="string",
                description="Reply content",
                examples=["That works for me", "I'll have numbers by Friday"],
            ),
            "ref": ArgDefinition(
                type="string",
                description="Which email to reply to",
                examples=["the first one", "Jane's email", "1"],
            ),
        },
        examples=[
            "Reply saying that works for me",
            "Reply to Jane's email saying I'll have numbers by Friday",
        ],
        negative_examples=[],
    ),
}

UNKNOWN_COMMAND_INTENT = IntentDefinition(
    intent_id="rtr_unknown",
    name="router.unknown_command",
    description=(
        "The input appears to be a command but does not match any "
        "available intent. Return this with a hint."
    ),
    action_class="",
    args={
        "hint": ArgDefinition(
            type="string",
            description="What the user seems to want",
        ),
    },
    examples=[],
    negative_examples=[],
)

# Map domain names to their intent names
DOMAIN_INTENTS: Dict[str, List[str]] = {
    "calendar": ["calendar.query", "calendar.create"],
    "email": ["email.search", "email.read", "email.draft", "email.reply"],
}


def _format_intent_block(intent: IntentDefinition) -> str:
    """Format a single intent definition for the extraction prompt."""
    lines = [f"{intent.name}"]
    lines.append(f"  Description: {intent.description}")

    if intent.args:
        lines.append("  Arguments:")
        for arg_name, arg_def in intent.args.items():
            req = ", REQUIRED" if arg_def.required else ", optional"
            lines.append(f"    - {arg_name} ({arg_def.type}{req}): {arg_def.description}")

    if intent.examples:
        examples_str = ", ".join(f'"{e}"' for e in intent.examples)
        lines.append(f"  Examples: {examples_str}")

    if intent.negative_examples:
        negs_str = ", ".join(f'"{e}"' for e in intent.negative_examples)
        lines.append(f"  NOT a command: {negs_str}")

    return "\n".join(lines)


def get_intents_for_domains(
    domains: Set[str],
    intent_registry: Optional[Dict[str, IntentDefinition]] = None,
) -> Dict[str, IntentDefinition]:
    """Get intent definitions filtered to matched domains."""
    registry = intent_registry or GSUITE_INTENTS
    result: Dict[str, IntentDefinition] = {}
    for domain in domains:
        intent_names = DOMAIN_INTENTS.get(domain, [])
        for name in intent_names:
            if name in registry:
                result[name] = registry[name]
    return result


def build_extraction_prompt(
    domains: Set[str],
    contacts: Dict[str, str],
    recent_context: Optional[str] = None,
    intent_registry: Optional[Dict[str, IntentDefinition]] = None,
) -> str:
    """Assemble the complete extraction prompt scoped to matched domains.

    Args:
        domains: Set of matched domain names from the keyword gate.
        contacts: Name-to-address mapping for contact resolution.
        recent_context: Optional recent MCP result context string.
        intent_registry: Optional override for intent definitions (for testing).

    Returns:
        The complete system prompt string for the extraction LLM call.
    """
    intents = get_intents_for_domains(domains, intent_registry)

    sections: List[str] = []

    # Preamble
    sections.append(
        "You are an intent classifier for a voice assistant. Given a user "
        "utterance, determine if it is a command directed at an external "
        "service or ordinary conversation."
    )

    # Critical rules
    sections.append(
        "CRITICAL RULES:\n"
        '- Most utterances are ordinary conversation. Return {"intent": null} by default.\n'
        "- Only return a non-null intent if the user is clearly requesting an action or "
        "information from one of the available services.\n"
        "- Expressions of opinion, complaints, hypotheticals, and rhetorical questions "
        "about service-related topics are NOT commands.\n"
        "- If you believe the input is a command but cannot map it to an available intent, "
        'return {"intent": "router.unknown_command"} with a hint describing what the '
        "user seems to want.\n"
        "- If a required argument cannot be determined from the utterance, omit it from "
        "args. Do NOT fabricate values. The system will ask the user.\n"
        "- Return ONLY valid JSON. No explanation, no markdown, no preamble."
    )

    # Available intents
    intent_blocks = [_format_intent_block(i) for i in intents.values()]
    intent_blocks.append(_format_intent_block(UNKNOWN_COMMAND_INTENT))
    sections.append("AVAILABLE INTENTS:\n\n" + "\n\n".join(intent_blocks))

    # Contacts
    if contacts:
        contact_lines = [f"  {name} \u2192 {addr}" for name, addr in contacts.items()]
        sections.append(
            "CONTACTS (for resolving names to addresses):\n" + "\n".join(contact_lines)
        )

    # Recent context
    if recent_context:
        sections.append(f"RECENT CONTEXT:\n{recent_context}")

    # Response schema
    sections.append(
        "RESPONSE SCHEMA:\n"
        "{\n"
        '  "intent": "<intent_name or null>",\n'
        '  "confidence": <0.0 to 1.0>,\n'
        '  "args": { ... },\n'
        '  "followup_suggestion": "<string or null>"\n'
        "}\n\n"
        "If the utterance contains a secondary request beyond the primary intent "
        "(e.g., \"check my email and see if I'm free tomorrow\"), set the primary "
        "intent for the most prominent request and put the secondary request in "
        "followup_suggestion as a natural language description. Do NOT return "
        "multiple intents."
    )

    return "\n\n".join(sections)
