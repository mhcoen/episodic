"""Tests for prompt assembly (episodic.mcp.extraction.prompt)."""

import pytest

from episodic.mcp.extraction.prompt import (
    GSUITE_INTENTS,
    UNKNOWN_COMMAND_INTENT,
    build_extraction_prompt,
    get_intents_for_domains,
)


class TestGetIntentsForDomains:
    def test_calendar_only(self):
        intents = get_intents_for_domains({"calendar"})
        assert "calendar.query" in intents
        assert "calendar.create" in intents
        assert "email.search" not in intents

    def test_email_only(self):
        intents = get_intents_for_domains({"email"})
        assert "email.search" in intents
        assert "email.read" in intents
        assert "email.draft" in intents
        assert "email.reply" in intents
        assert "calendar.query" not in intents

    def test_both_domains(self):
        intents = get_intents_for_domains({"calendar", "email"})
        assert len(intents) == 6  # 2 calendar + 4 email

    def test_unknown_domain(self):
        intents = get_intents_for_domains({"slack"})
        assert len(intents) == 0


class TestBuildExtractionPrompt:
    def test_single_domain_calendar(self):
        prompt = build_extraction_prompt(
            domains={"calendar"},
            contacts={},
        )
        assert "calendar.query" in prompt
        assert "calendar.create" in prompt
        assert "email.search" not in prompt
        assert "email.draft" not in prompt

    def test_single_domain_email(self):
        prompt = build_extraction_prompt(
            domains={"email"},
            contacts={},
        )
        assert "email.search" in prompt
        assert "email.draft" in prompt
        assert "calendar.query" not in prompt

    def test_multi_domain(self):
        prompt = build_extraction_prompt(
            domains={"calendar", "email"},
            contacts={},
        )
        assert "calendar.query" in prompt
        assert "email.search" in prompt

    def test_unknown_command_always_included(self):
        prompt = build_extraction_prompt(
            domains={"calendar"},
            contacts={},
        )
        assert "router.unknown_command" in prompt

    def test_contacts_populated(self):
        prompt = build_extraction_prompt(
            domains={"email"},
            contacts={"bob": "bob@company.com", "jane": "jane@company.com"},
        )
        assert "bob" in prompt
        assert "bob@company.com" in prompt
        assert "jane@company.com" in prompt

    def test_contacts_empty(self):
        prompt = build_extraction_prompt(
            domains={"email"},
            contacts={},
        )
        assert "CONTACTS" not in prompt

    def test_recent_context_included(self):
        context = 'Last email search:\n  1. From: jane@co.com, Subject: "Budget"'
        prompt = build_extraction_prompt(
            domains={"email"},
            contacts={},
            recent_context=context,
        )
        assert "RECENT CONTEXT" in prompt
        assert "jane@co.com" in prompt

    def test_recent_context_omitted(self):
        prompt = build_extraction_prompt(
            domains={"email"},
            contacts={},
            recent_context=None,
        )
        assert "RECENT CONTEXT" not in prompt

    def test_required_args_marked(self):
        prompt = build_extraction_prompt(
            domains={"email"},
            contacts={},
        )
        # email.draft has "to" as REQUIRED
        assert "REQUIRED" in prompt

    def test_critical_rules_present(self):
        prompt = build_extraction_prompt(
            domains={"calendar"},
            contacts={},
        )
        assert "CRITICAL RULES" in prompt
        assert '{"intent": null}' in prompt

    def test_response_schema_present(self):
        prompt = build_extraction_prompt(
            domains={"calendar"},
            contacts={},
        )
        assert "RESPONSE SCHEMA" in prompt
        assert "followup_suggestion" in prompt
