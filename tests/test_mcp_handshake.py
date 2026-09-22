"""The pre-deploy MCP handshake check.

A GatewayTarget is created by handshaking the server, so a server that refuses
the handshake fails CloudFormation and rolls the whole stack back. ScreenWeave's
MCP server answered `initialize` with a hardcoded '2024-11-05' whatever was
asked, and that is what the first deploy reaching it produced:

    GatewayTarget ... failed to stabilize, status: FAILED, reason: Failed to
    connect and fetch tools from the provided MCP target server.
    Error - Unsupported protocol version

The fakes here answer with the two real wire shapes -- plain JSON and an SSE
`data:` frame -- because an MCP server over HTTP may use either and a check
that reads only one reports every streaming server as unverified.
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
import urllib.error

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import mcp_handshake  # noqa: E402

REQUESTED = mcp_handshake.REQUESTED_VERSION


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def opener_returning(body: str, captured: list | None = None):
    def _open(request, timeout=None):
        if captured is not None:
            captured.append(request)
        return FakeResponse(body.encode())
    return _open


def initialize_result(version: str) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": 1, "result": {
        "protocolVersion": version,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "sibling", "version": "1.0.0"},
    }})


# ── the three outcomes ──────────────────────────────────────────────────────

def test_a_server_that_echoes_the_requested_revision_is_ok():
    outcome, _detail = mcp_handshake.check(
        "https://x/mcp", opener=opener_returning(initialize_result(REQUESTED))
    )
    assert outcome == mcp_handshake.OK


def test_the_production_failure_is_refused():
    """ScreenWeave's exact behaviour: answer 2024-11-05 to any request."""
    outcome, detail = mcp_handshake.check(
        "https://x/mcp", opener=opener_returning(initialize_result("2024-11-05"))
    )
    assert outcome == mcp_handshake.REFUSED
    assert "2024-11-05" in detail and REQUESTED in detail
    assert "rollback" in detail or "stack" in detail


def test_an_sse_framed_answer_is_read():
    """MCP over streamable HTTP answers `event: message\\ndata: {...}`. Reading
    only plain JSON would report every such server as unverified and wire it
    blind."""
    body = f"event: message\ndata: {initialize_result(REQUESTED)}\n\n"
    outcome, _detail = mcp_handshake.check("https://x/mcp", opener=opener_returning(body))
    assert outcome == mcp_handshake.OK


def test_an_sse_framed_refusal_is_refused():
    body = f"event: message\ndata: {initialize_result('2024-11-05')}\n\n"
    outcome, _d = mcp_handshake.check("https://x/mcp", opener=opener_returning(body))
    assert outcome == mcp_handshake.REFUSED


@pytest.mark.parametrize("body", ["", "not json at all", "{}", '{"result": {}}'])
def test_an_unreadable_answer_is_unverified_not_refused(body):
    """Unverified wires the target; refused drops it. Reading an unparseable
    answer as a refusal would silently lose working tools."""
    outcome, _d = mcp_handshake.check("https://x/mcp", opener=opener_returning(body))
    assert outcome == mcp_handshake.UNVERIFIED


def test_a_jsonrpc_error_is_unverified():
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "nope"}})
    outcome, detail = mcp_handshake.check("https://x/mcp", opener=opener_returning(body))
    assert outcome == mcp_handshake.UNVERIFIED
    assert "nope" in detail


def test_an_http_error_is_unverified():
    def _open(request, timeout=None):
        raise urllib.error.HTTPError("https://x/mcp", 403, "Forbidden", {}, None)
    outcome, detail = mcp_handshake.check("https://x/mcp", opener=_open)
    assert outcome == mcp_handshake.UNVERIFIED
    assert "403" in detail


def test_a_network_failure_is_unverified_and_never_raises():
    def _open(request, timeout=None):
        raise OSError("connection reset")
    outcome, detail = mcp_handshake.check("https://x/mcp", opener=_open)
    assert outcome == mcp_handshake.UNVERIFIED
    assert "connection reset" in detail


# ── the request it sends ────────────────────────────────────────────────────

def test_the_probe_sends_a_real_initialize():
    captured: list = []
    mcp_handshake.check("https://x/mcp", opener=opener_returning(initialize_result(REQUESTED), captured))
    request = captured[0]
    body = json.loads(request.data)
    assert request.get_method() == "POST"
    assert body["method"] == "initialize"
    assert body["params"]["protocolVersion"] == REQUESTED
    # Servers speaking the streamable-HTTP transport answer SSE unless told
    # JSON is acceptable; asking for only one of the two loses half of them.
    accept = request.get_header("Accept")
    assert "application/json" in accept and "text/event-stream" in accept


def test_the_requested_revision_is_not_the_one_that_failed():
    assert REQUESTED != "2024-11-05"


# ── exit codes, which the deploy branches on ────────────────────────────────

@pytest.mark.parametrize(
    "body,expected",
    [
        (initialize_result(REQUESTED), 0),
        (initialize_result("2024-11-05"), 1),
        ("garbage", 2),
    ],
)
def test_the_exit_code_carries_the_outcome(body, expected, monkeypatch, capsys):
    monkeypatch.setattr(mcp_handshake.urllib.request, "urlopen", opener_returning(body))
    assert mcp_handshake.main(["https://x/mcp", "--name", "Sibling"]) == expected
    out = capsys.readouterr().out
    if expected == 1:
        assert "::warning::" in out and "Sibling" in out
    if expected == 2:
        assert "NOT VERIFIED" in out
