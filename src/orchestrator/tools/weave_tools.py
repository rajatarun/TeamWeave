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

# One pre_tool result is stored under the tool name, so a second
# query_health_record on the same step replaces the first. These facets are
# one call that retrieves several ways and returns all of them. The agent
# cannot decide to search; a single query on the raw sentence is what made
# the record look empty and the answer turn into questions.
HEALTH_RECORD_FACETS = (
    ("reported", "{q}"),
    ("labs_vitals", "Recent laboratory results and vital signs, with dates and values. Context: {q}"),
    ("medications", "Medications on record and any recorded changes. Context: {q}"),
    ("sleep_activity", "Sleep and activity. Context: {q}"),
    ("trends", "Trends over time and what changed since the previous period. Context: {q}"),
)


def _facets_enabled(facets: Any) -> bool:
    if isinstance(facets, str):
        return facets.strip().lower() in {"1", "true", "yes"}
    return bool(facets)


def _excerpts_of(result: Dict[str, Any]) -> list:
    """Passages in the shape the health prompts read: content and sourceKey.

    The knowledge base already returns that. The health API returns an answer
    and a list of sources; without this, a facet that fell back to the API
    would look empty beside a facet that hit the base.
    """
    if not isinstance(result, dict):
        return []
    raw = result.get("excerpts")
    found: list = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            found.append({
                "sourceKey": str(item.get("sourceKey") or ""),
                "content": content,
            })
    if found:
        return found
    answer = str(result.get("answer") or "").strip()
    if answer and result.get("found"):
        sources = result.get("sources") if isinstance(result.get("sources"), list) else []
        source = str(sources[0]) if sources else ""
        return [{"sourceKey": source, "content": answer}]
    return []


def _lookup_one(question: str, token: str, top_k: int) -> Dict[str, Any]:
    from ..contextweave_client import query_health_record as _query
    from ..health_kb import retrieve

    if not token.strip():
        return _query(question, token=token, top_k=top_k)
    result = retrieve(question, top_k=top_k)
    if result is None:
        return _query(question, token=token, top_k=top_k)
    return result


def _query_facets(question: str, token: str, top_k: int, used_for: str) -> Dict[str, Any]:
    from ..contextweave_client import query_health_record as _query

    # One refusal, not five. Fanning out would look like five empty searches
    # and the agent would be told the record contains nothing.
    if not token.strip():
        result = _query(question, token=token, top_k=top_k)
        result.setdefault("used_for", used_for)
        return result
    if not question:
        return {"error": "a question is required", "used_for": used_for}

    queries = []
    excerpts: list = []
    seen = set()
    errors = []
    source = ""
    for facet, template in HEALTH_RECORD_FACETS:
        text = template.format(q=question)
        one = _lookup_one(text, token, top_k)
        entry: Dict[str, Any] = {"facet": facet, "question": text}
        if isinstance(one, dict) and one.get("source"):
            source = str(one["source"])
        got = _excerpts_of(one if isinstance(one, dict) else {})
        failed = isinstance(one, dict) and "error" in one and not got
        if failed:
            entry["error"] = one["error"]
            entry["excerpts"] = []
            errors.append(str(one["error"]))
        else:
            entry["found"] = bool(got)
            entry["excerpts"] = got
            for excerpt in got:
                key = (excerpt.get("sourceKey"), excerpt.get("content"))
                if key in seen:
                    continue
                seen.add(key)
                excerpts.append(excerpt)
        queries.append(entry)

    body: Dict[str, Any] = {
        "queries": queries,
        "searched": [item["question"] for item in queries],
        "excerpts": excerpts,
        "used_for": used_for,
    }
    if source:
        body["source"] = source
    if excerpts:
        body["found"] = True
        return body
    if errors and len(errors) == len(queries):
        # No `found`. An error and an empty record are different answers, and
        # a top-level found false here would say the records contain nothing.
        body["error"] = errors[0]
        return body
    body["found"] = False
    return body


def query_health_record(question: str = "", caller_token: str = "",
                        top_k: int = 6, facets: Any = None,
                        **_ignored: Any) -> Dict[str, Any]:
    """Ask the person's health records, as that person.

    `caller_token` is injected by `execute_tool` from the bearer token the
    person presented; it is deliberately not something a team config can
    supply, because a config is JSON in S3 and a credential named there would
    be a credential anyone with write access could point somewhere else.

    When ``HEALTH_KNOWLEDGE_BASE_ID`` is set, the excerpts come from the
    Bedrock knowledge base over ContextWeave's health bucket. The token is
    still required first: Retrieve is an IAM call, and without the gate a
    run with nobody behind it would read the record. When the base is unset,
    the same token is the bearer on the health API.

    ``facets`` runs the fixed set of targeted retrievals and returns every
    one of them under ``queries``, plus the union under ``excerpts``. The
    step still declares the tool once.
    """
    rule = rule_for("query_health_record")
    token = str(caller_token or "")
    text = str(question or "").strip()
    k = int(top_k or 6)
    if _facets_enabled(facets):
        return _query_facets(text, token, k, rule.use_when)
    result = _lookup_one(text, token, k)
    result.setdefault("used_for", rule.use_when)
    return result
