"""Stub integration tests for MCP server-mode input pipeline (spec tests 84-92).

These tests require the full security pipeline to be assembled and are
marked as skipped until integration infrastructure is available.
"""

import pytest


@pytest.mark.skip(reason="Integration test - requires full pipeline")
def test_84_server_receives_html_content():
    """Spec test 84: Server receives HTML content from MCP client.

    Pipeline: validate args -> sanitize HTML -> isolate -> tag provenance.
    Verify that HTML is sanitized (scripts removed, hidden elements stripped),
    wrapped in untrusted_content tags, and tagged with source provenance.
    """


@pytest.mark.skip(reason="Integration test - requires full pipeline")
def test_85_server_receives_oversized_message():
    """Spec test 85: Server receives oversized message from MCP client.

    Verify that validation rejects the request before any security layer
    processes it. Response should be a sanitized error with invalid_params
    code.
    """


@pytest.mark.skip(reason="Integration test - requires full pipeline")
def test_86_server_receives_message_with_path_traversal():
    """Spec test 86: Server receives source_name with path traversal.

    Verify that validation catches ../../etc/passwd in source_name and
    rejects the request with a sanitized error. No file system access
    should occur.
    """


@pytest.mark.skip(reason="Integration test - requires full pipeline")
def test_87_server_tool_execution_within_rate_limits():
    """Spec test 87: Server tool execution within rate limits.

    Verify that a tool call from an MCP client proceeds through the full
    pipeline: validate -> sanitize -> policy check -> rate limit check ->
    binding verification -> execute -> audit log.
    """


@pytest.mark.skip(reason="Integration test - requires full pipeline")
def test_88_server_tool_execution_exceeds_rate_limit():
    """Spec test 88: Server tool execution exceeds rate limit.

    After exhausting the hourly limit, verify that the next tool call
    is blocked with a rate_limited error including retry_after.
    """


@pytest.mark.skip(reason="Integration test - requires full pipeline")
def test_89_server_outbound_response_with_canary():
    """Spec test 89: Server outbound response contains canary token.

    Inject a canary into the session's system prompt. If a tool response
    contains the canary, verify that the outbound check flags it and the
    response is blocked or the canary is stripped.
    """


@pytest.mark.skip(reason="Integration test - requires full pipeline")
def test_90_server_outbound_response_dlp_overlap():
    """Spec test 90: Server outbound response triggers DLP overlap.

    Feed untrusted content through inbound pipeline, then attempt to
    exfiltrate it through a tool response. Verify DLP detects the
    verbatim or n-gram overlap.
    """


@pytest.mark.skip(reason="Integration test - requires full pipeline")
def test_91_server_invalid_capability_token():
    """Spec test 91: Server receives request with invalid capability token.

    Verify the request is rejected with unauthorized error. No tool
    execution occurs. Audit log records the rejection.
    """


@pytest.mark.skip(reason="Integration test - requires full pipeline")
def test_92_server_full_pipeline_audit_trail():
    """Spec test 92: Full pipeline produces complete audit trail.

    Execute a successful tool call through all layers. Verify the audit
    log contains entries for: validation, sanitization, policy check,
    rate limit, binding verification, execution, and any warnings.
    """
