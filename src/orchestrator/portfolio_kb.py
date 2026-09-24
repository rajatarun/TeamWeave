"""Retrieve from the portfolio knowledge base, and ask it to re-read the bucket.

The documents are this stack's portfolio bucket: statements a person uploads.
The base is provisioned by the same custom resource as the health base
(``health_kb_provision``), with Nova at 1024 dimensions and FLOAT32. This
module does not create a bucket and does not embed.

``query_portfolio`` is the only caller of :func:`search`, and the rule
restricts it to ``financial_advisors``. There is no per-person token. The
bucket is the account's uploads, not a ContextWeave store keyed to a
bearer, so the allowlist is the barrier.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .health_kb import handle_sync_event
from .health_kb import retrieve as _retrieve
from .logger import get_logger

log = get_logger("portfolio_kb")

# One pre_tool result is stored under the tool name, so these are one call
# that retrieves several ways. The agent cannot decide to search.
FACETS = (
    ("holdings", "Holdings and positions, with tickers and quantities. Context: {q}"),
    ("allocation", "Asset allocation and portfolio weights. Context: {q}"),
    ("cost_basis", "Cost basis and tax lots. Context: {q}"),
    ("performance", "Performance, returns, and gains or losses. Context: {q}"),
    ("risk", "Concentration and risk. Context: {q}"),
)


def knowledge_base_id() -> str:
    return (os.environ.get("PORTFOLIO_KNOWLEDGE_BASE_ID") or "").strip()


def data_source_id() -> str:
    return (os.environ.get("PORTFOLIO_DATA_SOURCE_ID") or "").strip()


def retrieve(question: str, *, top_k: int = 6, client=None) -> Dict[str, Any]:
    """Excerpts from the portfolio base.

    An unset id is an error, not an empty portfolio. Empty would tell the
    agent the person holds nothing.
    """
    kb_id = knowledge_base_id()
    if not kb_id:
        return {"error": "the portfolio knowledge base is not configured"}
    result = _retrieve(
        question, top_k=top_k, client=client, kb_id=kb_id, log_label="portfolio_kb",
    )
    if result is None:
        return {"error": "the portfolio knowledge base is not configured"}
    return result


def _excerpts_of(result: Dict[str, Any]) -> List[Dict[str, str]]:
    if not isinstance(result, dict):
        return []
    found = []
    for item in result.get("excerpts") or []:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        found.append({
            "sourceKey": str(item.get("sourceKey") or ""),
            "content": content,
        })
    return found


def search(question: str, *, top_k: int = 6, facets: bool = False, client=None) -> Dict[str, Any]:
    """One query, or the fixed facet set, against the portfolio base."""
    text = str(question or "").strip()
    if not facets:
        if not text:
            return {"error": "a query is required"}
        return retrieve(text, top_k=top_k, client=client)
    if not text:
        return {"error": "a query is required"}
    if not knowledge_base_id():
        return {"error": "the portfolio knowledge base is not configured"}

    queries = []
    excerpts: List[Dict[str, str]] = []
    seen = set()
    errors = []
    source = ""
    for facet, template in FACETS:
        asked = template.format(q=text)
        one = retrieve(asked, top_k=top_k, client=client)
        entry: Dict[str, Any] = {"facet": facet, "question": asked}
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
    }
    if source:
        body["source"] = source
    if excerpts:
        body["found"] = True
        return body
    if errors and len(errors) == len(queries):
        body["error"] = errors[0]
        return body
    body["found"] = False
    return body


def sync_handler(event: Any, context: Any) -> Dict[str, Any]:
    """SQS entrypoint for the portfolio bucket. The event names the object; it is not logged.

    Explicit ids, including when they are empty. Omitting them would start
    a job on the health base, which is the other caller's default.
    """
    return handle_sync_event(
        event, context, kb_id=knowledge_base_id(), source_id=data_source_id(),
    )
