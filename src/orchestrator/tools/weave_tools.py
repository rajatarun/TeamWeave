"""Pipeline tools backed by the weave siblings' MCP servers.

One function per rule in `tool_rules.RULES`. Each is thin on purpose: it names
the sibling and the tool, shapes the arguments, and returns whatever came back.
Everything about *when* to call it lives in the rule, and everything about
transport lives in `mcp_client`.

Every one of these returns a dict that either carries `data` or carries
`error`. A step that lost its lookup therefore says so in its own inputs, which
is the difference between an agent knowing nothing was found and an agent
seeing a blank it will cheerfully fill in.
"""
from __future__ import annotations

from typing import Any, Dict

from ..mcp_client import call_tool
from ..tool_rules import rule_for


def _run(tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    rule = rule_for(tool)
    result = call_tool(rule.sibling, rule.mcp_tool, arguments)
    # The rule travels with the result so a step's inputs record why the call
    # was made, not just what it returned.
    result.setdefault("used_for", rule.use_when)
    return result


# ── DataDictionary ──────────────────────────────────────────────────────────

def lookup_data_element(dataElement: str = "", **_ignored: Any) -> Dict[str, Any]:
    return _run("lookup_data_element", {"dataElement": str(dataElement or "")})


def search_data_elements(query: str = "", **_ignored: Any) -> Dict[str, Any]:
    return _run("search_data_elements", {"query": str(query or "")})


def data_elements_for_context(context: str = "", **_ignored: Any) -> Dict[str, Any]:
    return _run("data_elements_for_context", {"context": str(context or "")})


def propose_data_element(prompt: str = "", **_ignored: Any) -> Dict[str, Any]:
    """Draft a catalogue entry and return its commit token for a person.

    The token is the point: this half is safe precisely because it changes
    nothing until someone spends it.
    """
    return _run("propose_data_element", {"prompt": str(prompt or "")})


# ── CipherWeave ─────────────────────────────────────────────────────────────

def encryption_strategy(dataElement: str = "", **_ignored: Any) -> Dict[str, Any]:
    return _run("encryption_strategy", {"dataElement": str(dataElement or "")})


# ── ScreenWeave ─────────────────────────────────────────────────────────────

def crawl_site(url: str = "", max_depth: int = 1, max_links: int = 10,
               **_ignored: Any) -> Dict[str, Any]:
    """Crawl shallowly by default.

    Depth and breadth are capped here rather than left to a team config: one
    worker invocation runs every step inside a 900 s Lambda, and a crawl that
    walks a site spends the budget the later agents need to say anything about
    it.
    """
    return _run("crawl_site", {
        "url": str(url or ""),
        "max_depth": max(1, min(int(max_depth or 1), 2)),
        "max_links": max(1, min(int(max_links or 10), 25)),
    })


def site_metrics(session_id: str = "", **_ignored: Any) -> Dict[str, Any]:
    return _run("site_metrics", {"session_id": str(session_id or "")})


# ── ToolWeave ───────────────────────────────────────────────────────────────

def plan_api_call(prompt: str = "", **_ignored: Any) -> Dict[str, Any]:
    """Turn prose into a concrete API call plan, without making the call."""
    return _run("plan_api_call", {"prompt": str(prompt or "")})


# ── ContextWeave: the person's own health record ────────────────────────────

def query_health_record(question: str = "", caller_token: str = "",
                        top_k: int = 6, **_ignored: Any) -> Dict[str, Any]:
    """Ask the health store, over HTTP rather than MCP, as the caller.

    `caller_token` is injected by `execute_tool` from the bearer token the
    person presented; it is deliberately not something a team config can
    supply, because a config is JSON in S3 and a credential named there would
    be a credential anyone with write access could point somewhere else.
    """
    from ..contextweave_client import query_health_record as _query

    rule = rule_for("query_health_record")
    result = _query(str(question or ""), token=str(caller_token or ""),
                    top_k=int(top_k or 6))
    result.setdefault("used_for", rule.use_when)
    return result
