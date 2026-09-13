"""ContextWeave knowledge-layer client — backing store for RAG mode "contextweave".

ContextWeave is the platform's shared knowledge layer: it owns the expertise
graph, the chunk vector store and the semantic cache, and it answers questions
rather than returning raw chunks.  TeamWeave therefore does not need its own
copy of the corpus for teams that select this mode.

Contract (ContextWeave ``src/query_api/handler.py``):

    POST {CONTEXTWEAVE_URL}/query-expertise
        {"question": str, "topK": int}
    -> {"queryId", "answer", "sources": [{"file", "excerpt", "weight"}],
        "confidence", "questionType", "cacheHit", ...}

    POST {CONTEXTWEAVE_URL}/feedback
        {"queryId": str, "rating": "up" | "down" | "neutral"}

Every call degrades gracefully: failures are logged and reported as "no
context", exactly the contract the pgvector modes already follow
(``retrieve_from_vector_store`` returns ``[]``, ``get_rag_context`` returns
``""``).  A knowledge layer being down must never fail a pipeline run.

HTTP uses ``urllib.request`` — the same client ``gemini.py`` uses for external
JSON APIs — so no new dependency is introduced.
"""
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .logger import get_logger

log = get_logger("contextweave_client")

QUERY_PATH = "/query-expertise"
FEEDBACK_PATH = "/feedback"

# Matches the read timeout used for the other external JSON API (gemini.py);
# ContextWeave runs a full RAG pipeline on a cache miss.
_TIMEOUT_SECONDS = 40
# Same retry budget as bedrock_invoke.invoke_agent (3 attempts total).
_MAX_RETRIES = 2


def base_url() -> str:
    return os.environ.get("CONTEXTWEAVE_URL", "").strip().rstrip("/")


def is_configured() -> bool:
    return bool(base_url())


def feedback_on_valid_output_enabled() -> bool:
    """True when a valid structured run should be reported back as an up-vote."""
    return os.environ.get("CONTEXTWEAVE_FEEDBACK_ON_VALID_OUTPUT", "").strip() == "1"


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("CONTEXTWEAVE_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _post(path: str, body: Dict[str, Any], max_retries: int = _MAX_RETRIES) -> Optional[Dict[str, Any]]:
    """POST JSON to ContextWeave; return the decoded body or None on failure.

    Retries transport errors and 5xx responses; a 4xx is a contract problem that
    a retry cannot fix, so it is reported immediately.
    """
    url = base_url()
    if not url:
        log.warning("contextweave_not_configured", extra={"required": ["CONTEXTWEAVE_URL"]})
        return None

    req = urllib.request.Request(
        url + path,
        data=json.dumps(body).encode("utf-8"),
        headers=_headers(),
        method="POST",
    )

    last_err = ""
    for attempt in range(0, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")[:300]
            if e.code < 500:
                log.warning(
                    "contextweave_http_error",
                    extra={"path": path, "code": e.code, "body_prefix": detail},
                )
                return None
            last_err = f"HTTP {e.code}: {detail}"
        except Exception as e:  # transport, timeout, malformed JSON
            last_err = str(e)[:240]
        if attempt < max_retries:
            log.warning(
                "contextweave_request_retrying",
                extra={"path": path, "attempt": attempt, "err": last_err},
            )

    log.warning("contextweave_request_failed", extra={"path": path, "err": last_err})
    return None


def query_expertise(question: str, top_k: int = 8) -> Optional[Dict[str, Any]]:
    """Ask the knowledge layer a question. Returns the response body or None."""
    if not question.strip():
        log.warning("contextweave_empty_question; skipping query")
        return None

    payload = _post(QUERY_PATH, {"question": question, "topK": int(top_k)})
    if payload is None:
        return None
    if not isinstance(payload, dict) or payload.get("error"):
        log.warning("contextweave_unexpected_response", extra={"body_prefix": str(payload)[:300]})
        return None

    log.info(
        "contextweave_query_ok",
        extra={
            "query_id": payload.get("queryId", ""),
            "confidence": payload.get("confidence"),
            "question_type": payload.get("questionType", ""),
            "cache_hit": payload.get("cacheHit"),
            "source_count": len(payload.get("sources") or []),
        },
    )
    return payload


def send_feedback(query_id: str, rating: str = "up") -> bool:
    """Rate a previous answer so the knowledge layer's router learns from it."""
    if not query_id:
        return False
    applied = _post(FEEDBACK_PATH, {"queryId": query_id, "rating": rating}, max_retries=0)
    if applied is None:
        return False
    log.info("contextweave_feedback_sent", extra={"query_id": query_id, "rating": rating})
    return True


def maybe_send_valid_output_feedback(query_id: str) -> bool:
    """Up-vote the answer that grounded a run whose structured output validated.

    Off unless CONTEXTWEAVE_FEEDBACK_ON_VALID_OUTPUT=1: a schema-valid run is a
    weak signal about answer quality, so opting in is a deliberate choice.
    """
    if not query_id or not feedback_on_valid_output_enabled():
        return False
    return send_feedback(query_id, "up")


# ── Mapping into the prompt builder's RAG_CONTEXT shape ──────────────────────


def _source_label(src: Dict[str, Any]) -> str:
    # "file" is the synthesiser's own citation shape; "sourceUri" is the
    # raw-chunk fallback ContextWeave emits when the model returns no sources.
    return str(src.get("file") or src.get("sourceUri") or src.get("source") or "contextweave")


def _source_text(src: Dict[str, Any]) -> str:
    return str(src.get("excerpt") or src.get("content") or "").strip()


def _source_weight(src: Dict[str, Any]) -> Optional[float]:
    raw = src.get("weight", src.get("sourceWeight"))
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def format_rag_context(payload: Dict[str, Any]) -> str:
    """Render a ContextWeave response as the RAG_CONTEXT block list.

    Uses the same block shape explicit (pgvector) mode produces, so the prompt
    builder and the agents see one consistent RAG_CONTEXT format regardless of
    which knowledge source answered.
    """
    blocks: List[str] = []
    index = 0

    answer = str(payload.get("answer") or "").strip()
    if answer:
        confidence = payload.get("confidence")
        label = "contextweave:answer"
        if isinstance(confidence, (int, float)):
            label += f" (confidence {float(confidence):.2f})"
        index += 1
        blocks.append(f"[RAG #{index}] SOURCE: {label}")
        blocks.append(answer)
        blocks.append("---")

    for src in payload.get("sources") or []:
        if not isinstance(src, dict):
            continue
        text = _source_text(src)
        if not text:
            continue
        label = _source_label(src)
        weight = _source_weight(src)
        if weight is not None:
            label += f" (weight {weight:.2f})"
        index += 1
        blocks.append(f"[RAG #{index}] SOURCE: {label}")
        blocks.append(text)
        blocks.append("---")

    return "\n".join(blocks).strip()
