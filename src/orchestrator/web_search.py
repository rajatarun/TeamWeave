"""Live web search for the financial team.

There is no search tool elsewhere in the pipeline. Gemini's generateContent
call already sends ``google_search`` for the research brief and then throws
the citations away. This keeps the URLs and stamps the day they were retrieved.

The key is a Secrets Manager secret, never a literal. ``WEB_SEARCH_SECRET_ARN``
overrides; otherwise the secret is ``GEMINI_SECRET_ARN``, the same parameter
the research brief uses (``GeminiSecretArn``). The secret string is
``{"key": "..."}`` or the raw key. ``WEB_SEARCH_PROVIDER`` selects the
implementation. ``gemini`` is the one this module speaks. Another name is
an error, not a silent skip, so a typo does not look like an empty market.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .deadline import budget_for_call
from .logger import get_logger

log = get_logger("web_search")

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Fixed targets. The agent cannot decide to search, and one query on the
# raw sentence misses rates when the person asked about a holding.
FACETS = (
    ("request", "{q}"),
    ("markets", "Current market prices, quotes, and recent moves. Context: {q}"),
    ("news", "Latest news. Context: {q}"),
    ("rates", "Current interest rates, yields, and inflation figures. Context: {q}"),
)


def provider() -> str:
    return (os.environ.get("WEB_SEARCH_PROVIDER") or "gemini").strip().lower() or "gemini"


def _key_from_secret(arn: str) -> str:
    import boto3

    resp = boto3.client("secretsmanager").get_secret_value(SecretId=arn)
    raw = resp.get("SecretString") or ""
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except Exception:
        return raw.strip()
    if isinstance(parsed, dict):
        return str(parsed.get("key") or parsed.get("value") or "").strip()
    return raw.strip()


def api_key() -> str:
    override = (os.environ.get("WEB_SEARCH_SECRET_ARN") or "").strip()
    if override:
        return _key_from_secret(override)
    # No secret means "not configured", which is an error the agent can read.
    # Importing gemini constructs a Secrets Manager client, and doing that
    # when there is nothing to fetch fails in a region-less process and looks
    # like a broken client rather than a missing key.
    if not (os.environ.get("GEMINI_SECRET_ARN") or "").strip():
        return ""
    from .gemini import _get_gemini_key
    return _get_gemini_key()


def _retrieved_on(today: Optional[str] = None) -> str:
    if today:
        return today
    return datetime.now(timezone.utc).date().isoformat()


def _results_from(data: Dict[str, Any], retrieved_on: str) -> List[Dict[str, str]]:
    results = []
    seen = set()
    for cand in (data or {}).get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        meta = cand.get("groundingMetadata") or {}
        chunks = meta.get("groundingChunks") or []
        snippets: Dict[int, str] = {}
        for support in meta.get("groundingSupports") or []:
            if not isinstance(support, dict):
                continue
            text = str(((support.get("segment") or {}).get("text")) or "").strip()
            for index in support.get("groundingChunkIndices") or []:
                if text and index not in snippets:
                    snippets[int(index)] = text
        for index, chunk in enumerate(chunks):
            web = (chunk or {}).get("web") if isinstance(chunk, dict) else None
            if not isinstance(web, dict):
                continue
            url = str(web.get("uri") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            results.append({
                "url": url,
                "title": str(web.get("title") or ""),
                "snippet": snippets.get(index, ""),
                "retrieved_on": retrieved_on,
            })
    return results


def _post(url: str, body: Dict[str, Any], api_key_value: str) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key_value},
        method="POST",
    )
    timeout = max(1, int(budget_for_call()))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def search_one(query: str, *, retrieved_on: Optional[str] = None, post=None) -> Dict[str, Any]:
    """One grounded search. ``found`` is true only when a URL came back."""
    name = provider()
    if name != "gemini":
        return {"error": f"web search provider '{name}' is not configured"}
    key = api_key()
    if not key:
        return {
            "error": "web search is not configured: set GEMINI_SECRET_ARN "
                     "(template parameter GeminiSecretArn) or WEB_SEARCH_SECRET_ARN",
        }
    text = str(query or "").strip()
    if not text:
        return {"error": "a query is required"}
    day = _retrieved_on(retrieved_on)
    from .model_map import resolve_model
    model = resolve_model("research_web").model_id
    override = os.environ.get("GEMINI_MODEL", "").strip()
    if override and override != model:
        log.warning("model_id_override", extra={"category": "research_web", "model_id": override})
        model = override
    url = _ENDPOINT.format(model=model)
    body = {
        "contents": [{"parts": [{"text": (
            "Search the web and report the current facts for this query. "
            "Do not give financial advice and do not promise a return.\n"
            f"Query: {text}"
        )}]}],
        "tools": [{"google_search": {}}],
    }
    send = post or _post
    try:
        data = send(url, body, key)
    except urllib.error.HTTPError:
        log.warning("web_search_http_error")
        return {"error": "web search could not be reached"}
    except Exception:
        log.warning("web_search_failed")
        return {"error": "web search could not be reached"}
    results = _results_from(data if isinstance(data, dict) else {}, day)
    out: Dict[str, Any] = {
        "query": text,
        "retrieved_on": day,
        "results": results,
        "found": bool(results),
        "source": "gemini-google-search",
    }
    return out


def search(question: str, *, facets: bool = False, retrieved_on: Optional[str] = None,
           post=None) -> Dict[str, Any]:
    """One query, or markets, news, and rates plus the person's own words."""
    text = str(question or "").strip()
    if not facets:
        return search_one(text, retrieved_on=retrieved_on, post=post)
    name = provider()
    if name != "gemini":
        return {"error": f"web search provider '{name}' is not configured"}
    if not api_key():
        return {
            "error": "web search is not configured: set GEMINI_SECRET_ARN "
                     "(template parameter GeminiSecretArn) or WEB_SEARCH_SECRET_ARN",
        }
    if not text:
        return {"error": "a query is required"}

    day = _retrieved_on(retrieved_on)
    queries = []
    results: List[Dict[str, str]] = []
    seen = set()
    errors = []
    for facet, template in FACETS:
        asked = template.format(q=text)
        one = search_one(asked, retrieved_on=day, post=post)
        entry: Dict[str, Any] = {"facet": facet, "query": asked, "retrieved_on": day}
        got = one.get("results") if isinstance(one, dict) else []
        if not isinstance(got, list):
            got = []
        if isinstance(one, dict) and one.get("error") and not got:
            entry["error"] = one["error"]
            entry["results"] = []
            errors.append(str(one["error"]))
        else:
            entry["found"] = bool(got)
            entry["results"] = got
            for item in got:
                url = item.get("url")
                if not url or url in seen:
                    continue
                seen.add(url)
                results.append(item)
        queries.append(entry)

    body: Dict[str, Any] = {
        "queries": queries,
        "searched": [item["query"] for item in queries],
        "results": results,
        "retrieved_on": day,
        "source": "gemini-google-search",
    }
    if results:
        body["found"] = True
        return body
    if errors and len(errors) == len(queries):
        body["error"] = errors[0]
        return body
    body["found"] = False
    return body
