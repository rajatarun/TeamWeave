#!/usr/bin/env python3
"""Ask an MCP endpoint whether it will survive becoming a GatewayTarget.

AgentCore Gateway handshakes the target server while *creating* the
`AWS::BedrockAgentCore::GatewayTarget`. A server that refuses the handshake
therefore does not produce a missing tool -- it produces

    GatewayTarget ... failed to stabilize, status: FAILED, reason: Failed to
    connect and fetch tools from the provided MCP target server.
    Error - Unsupported protocol version

which fails CloudFormation and rolls the whole TeamWeave stack back. That is
the opposite of the degradation the gateway wiring claims: one sibling's bug
taking the platform's deploy with it. ScreenWeave's MCP server answered
`initialize` with a hardcoded '2024-11-05' whatever was asked, and did exactly
that on the first deploy that reached it.

The check is the MCP handshake rule itself, so it needs no knowledge of what
AgentCore requires: a server that supports the requested revision echoes it.
A server that answers with a *different* version is saying it does not speak
ours, which is the refusal the Gateway acts on.

Three outcomes, kept distinct because they call for different actions:

    ok          the server echoed the requested revision -- wire the target
    refused     it answered with another revision -- skip, and say which
    unverified  no answer, or an answer this script could not read -- wire it
                anyway

`unverified` deliberately does not skip. The probe runs unauthenticated from a
CI runner while the Gateway calls with its own identity and from its own
network, so a probe failure is weak evidence about the Gateway. Dropping a
working tool on it would be the silent-missing-tool failure this repository
keeps being bitten by; letting it through means CloudFormation says so, loudly,
in a message that now has a known cause.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# The newest revision this platform expects a sibling to speak. A server that
# does not echo it is the case that failed the deploy.
REQUESTED_VERSION = "2025-06-18"

OK, REFUSED, UNVERIFIED = "ok", "refused", "unverified"


def initialize_body() -> bytes:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": REQUESTED_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "teamweave-deploy-probe", "version": "1.0.0"},
            },
        }
    ).encode()


def parse_response(raw: str) -> dict | None:
    """MCP over HTTP answers as JSON or as an SSE `data:` frame."""
    raw = (raw or "").strip()
    if not raw:
        return None
    for candidate in (raw, *(
        line[len("data:"):].strip()
        for line in raw.splitlines()
        if line.startswith("data:")
    )):
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def check(url: str, timeout: float = 15.0, opener=None) -> tuple[str, str]:
    """Return (outcome, detail).

    `opener` is resolved here rather than bound as a default argument. A
    default is captured when the function is defined, so patching
    `urllib.request.urlopen` -- the obvious way to keep a test off the
    network -- would leave the default pointing at the real one and the test
    would quietly make live calls.
    """
    if opener is None:
        opener = urllib.request.urlopen
    request = urllib.request.Request(
        url,
        data=initialize_body(),
        headers={
            "Content-Type": "application/json",
            # Servers that speak the streamable-HTTP transport answer as SSE
            # unless told JSON is acceptable; accept both rather than assume.
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": REQUESTED_VERSION,
        },
        method="POST",
    )
    try:
        with opener(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return UNVERIFIED, f"HTTP {exc.code} from the endpoint"
    except Exception as exc:  # network, TLS, timeout -- all weak evidence
        return UNVERIFIED, f"{type(exc).__name__}: {str(exc)[:120]}"

    payload = parse_response(raw)
    if payload is None:
        return UNVERIFIED, "response was neither JSON nor an SSE data frame"
    if "error" in payload:
        return UNVERIFIED, f"server returned a JSON-RPC error: {str(payload['error'])[:120]}"

    served = (payload.get("result") or {}).get("protocolVersion")
    if not served:
        return UNVERIFIED, "initialize result carried no protocolVersion"
    if served != REQUESTED_VERSION:
        return REFUSED, (
            f"asked for {REQUESTED_VERSION} and the server answered {served}, so it "
            f"does not speak this revision; AgentCore refuses such a target and the "
            f"failure is a stack rollback, not a missing tool"
        )
    return OK, f"handshake ok at {served}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--name", default="", help="sibling name, for the message")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)

    outcome, detail = check(args.url, timeout=args.timeout)
    label = args.name or args.url
    if outcome == OK:
        print(f"{label}: {detail}")
    elif outcome == REFUSED:
        print(f"::warning::{label} will not be wired as a gateway target: {detail}")
    else:
        print(f"::warning::{label} handshake NOT VERIFIED ({detail}). Wiring it anyway -- "
              f"a probe from CI is weak evidence about a call the Gateway makes with "
              f"its own identity.")
    # The outcome is the exit code so the shell can branch: 0 ok, 1 refused,
    # 2 unverified. Never a failure of the deploy itself.
    return {OK: 0, REFUSED: 1, UNVERIFIED: 2}[outcome]


if __name__ == "__main__":
    sys.exit(main())
