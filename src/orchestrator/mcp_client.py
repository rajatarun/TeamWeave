"""Call a weave sibling's MCP server over HTTP, from inside a pipeline step.

**Why this and not the AgentCore Gateway.** The gateway has four targets, and
nothing on this platform invokes them: `src/agentcore/agent.py` builds a
Converse request with no `toolConfig`, so an agent turn cannot call a tool at
all. The gateway is preparation for when it can. A team config declaring
gateway tools today would be exactly the "config says X, deployment does Y"
failure this repository keeps hitting -- dead config that reads like a feature.

What *does* execute is `tool_registry`: deterministic Python running before or
after a turn. So a sibling's tool reaches a step the same way `parse_document`
does, and the agent sees the result in `STEP_INPUTS_JSON` rather than choosing
to call it. That is a real constraint on team design, not a detail: a step can
be *given* a lookup, it cannot *decide* to make one.

Everything here degrades. A sibling that is down, slow or unreachable returns a
result carrying `error` rather than raising, because a step that loses its
lookup should say so in its inputs -- an empty block the agent cannot tell from
"nothing was found" is how the RAG layer used to lie.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .logger import get_logger

log = get_logger("mcp_client")

# The revision this platform speaks. A server that answers with a different one
# is telling us it does not speak ours -- the same rule scripts/mcp_handshake.py
# applies before a gateway target is wired.
PROTOCOL_VERSION = "2025-06-18"

# Short by construction. One worker invocation runs every step of a pipeline
# inside a 900 s Lambda, so a lookup that hangs spends budget the later agents
# need. A sibling that cannot answer in this long is a sibling to degrade past.
DEFAULT_TIMEOUT_S = 20.0

# Env var per sibling, wired from the same stack parameters the gateway targets
# are gated on, so a URL is resolved in exactly one place.
SIBLING_ENV = {
    "screenweave": "SCREENWEAVE_MCP_URL",
    "cipherweave": "CIPHERWEAVE_MCP_URL",
    "datadictionary": "DATADICTIONARY_MCP_URL",
    "toolweave": "TOOLWEAVE_MCP_URL",
}


def endpoint_for(sibling: str, env: Optional[Dict[str, str]] = None) -> str:
    env = env if env is not None else os.environ
    return (env.get(SIBLING_ENV.get(sibling, ""), "") or "").strip()


def _parse(raw: str) -> Optional[dict]:
    """MCP over HTTP answers as JSON or as an SSE `data:` frame."""
    raw = (raw or "").strip()
    if not raw:
        return None
    candidates = [raw] + [
        line[len("data:"):].strip()
        for line in raw.splitlines()
        if line.startswith("data:")
    ]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _rpc(url: str, method: str, params: dict, request_id: int, timeout: float, opener=None) -> dict:
    if opener is None:
        # Resolved here, not as a default argument: a default is captured at
        # definition time, so patching urlopen in a test would leave the
        # default pointing at the real network.
        opener = urllib.request.urlopen
    body = json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    ).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        },
        method="POST",
    )
    with opener(request, timeout=timeout) as response:
        return _parse(response.read().decode("utf-8", "replace")) or {}


def call_tool(
    sibling: str,
    tool: str,
    arguments: Dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    env: Optional[Dict[str, str]] = None,
    opener=None,
) -> Dict[str, Any]:
    """Run one MCP `tools/call` and return its result, or `{"error": ...}`.

    Never raises. The caller is a pre/post tool inside a pipeline step, and a
    sibling being unavailable is not a reason to lose an otherwise good run --
    but it *is* something the step's inputs must state, so the agent is not
    left to read an absent lookup as an empty one.
    """
    url = endpoint_for(sibling, env)
    if not url:
        return {"error": f"{sibling} has no endpoint configured ({SIBLING_ENV.get(sibling)})",
                "sibling": sibling, "tool": tool}

    try:
        # The handshake is not ceremony: a server may refuse the revision, and
        # finding that out here names the cause rather than leaving a confusing
        # error on the tools/call.
        hello = _rpc(url, "initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "teamweave", "version": "1.0.0"},
        }, 1, timeout, opener)
        served = (hello.get("result") or {}).get("protocolVersion")
        if served and served != PROTOCOL_VERSION:
            return {"error": f"{sibling} speaks MCP {served}, not {PROTOCOL_VERSION}",
                    "sibling": sibling, "tool": tool}

        answer = _rpc(url, "tools/call", {"name": tool, "arguments": arguments},
                      2, timeout, opener)
    except urllib.error.HTTPError as exc:
        log.warning("mcp_http_error sibling=%s tool=%s status=%s", sibling, tool, exc.code)
        return {"error": f"HTTP {exc.code} from {sibling}", "sibling": sibling, "tool": tool}
    except Exception as exc:
        log.warning("mcp_call_failed sibling=%s tool=%s err=%s", sibling, tool, exc)
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}",
                "sibling": sibling, "tool": tool}

    if "error" in answer:
        return {"error": str(answer["error"])[:300], "sibling": sibling, "tool": tool}

    result = answer.get("result") or {}
    # MCP returns content blocks; a tool answering JSON puts it in a text block.
    # `isError` is the protocol's own way of reporting a tool that ran and
    # failed, which is different from a transport failure and must not be
    # flattened into success.
    if result.get("isError"):
        return {"error": f"{sibling}.{tool} reported a tool error",
                "sibling": sibling, "tool": tool, "content": result.get("content")}

    if "structuredContent" in result:
        return {"sibling": sibling, "tool": tool, "data": result["structuredContent"]}

    texts = [
        block.get("text", "")
        for block in (result.get("content") or [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    joined = "\n".join(t for t in texts if t)
    for text in texts:
        try:
            return {"sibling": sibling, "tool": tool, "data": json.loads(text)}
        except (ValueError, TypeError):
            continue
    return {"sibling": sibling, "tool": tool, "data": joined}
