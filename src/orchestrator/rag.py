import os
from typing import Any, Dict, List

from .db import DbDao
from .logger import get_logger

log = get_logger("rag")


def retrieve_from_vector_store(collection_id: str, query: str, top_k: int) -> List[Dict[str, str]]:
    log.info("vector_store_stub", extra={"collection_id": collection_id, "top_k": top_k})
    return [{"source": "stub://vector", "text": "RAG STUB: replace retrieve_from_vector_store() with your vector DB."}]


def get_rag_context(request_obj: Dict[str, Any], team_globals: Dict[str, Any], owner: str, dao: DbDao) -> str:
    rag = (team_globals or {}).get("rag") or {}
    mode = (rag.get("mode") or "kb").lower()
    top_k = int(rag.get("top_k") or 8)
    env_key = rag.get("rag_env_key") or "VECTOR_DB_TABLE"

    if mode == "kb":
        log.info("KB mode enabled; returning empty RAG_CONTEXT")
        return ""

    if mode == "explicit":
        collection_id = os.environ.get(env_key, "") or os.environ.get("VECTOR_DB_TABLE", "")
        query = " ".join([request_obj.get("topic", ""), request_obj.get("objective", ""), request_obj.get("audience", "")]).strip()
        hits = retrieve_from_vector_store(collection_id, query, top_k) if collection_id else []
        blocks = []
        for i, h in enumerate(hits, start=1):
            blocks.append(f"[RAG #{i}] SOURCE: {h.get('source', '')}")
            blocks.append(h.get("text", ""))
            blocks.append("---")
        return "\n".join(blocks).strip()

    if mode == "history":
        completed = dao.list_completed_topic_levels(owner=owner, limit=200)
        if not completed:
            return ""
        return "COMPLETED_TASKS_HISTORY:\n" + "\n".join([f"- {c}" for c in completed])

    log.warning("Unknown RAG mode; returning empty", extra={"mode": mode})
    return ""
