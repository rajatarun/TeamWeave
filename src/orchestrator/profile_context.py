import os
from typing import Any, Dict, List

from .logger import get_logger
from .rag import retrieve_from_vector_store

log = get_logger("profile_context")


def _to_query(owner: str, team_raw: Dict[str, Any], request_obj: Dict[str, Any], perspective_template: str) -> str:
    team_meta = team_raw.get("team") or {}
    globals_obj = team_raw.get("globals") or {}
    perspective = perspective_template.format(
        owner=owner,
        team_name=team_meta.get("name", ""),
        team_version=team_meta.get("version", ""),
        north_star=globals_obj.get("north_star", ""),
    )
    query_parts: List[str] = [
        f"Profile and achievements of {owner}",
        perspective,
        request_obj.get("topic", ""),
        request_obj.get("objective", ""),
        request_obj.get("audience", ""),
    ]
    return " ".join([part.strip() for part in query_parts if part and part.strip()])


def get_owner_profile_context(
    request_obj: Dict[str, Any],
    team_raw: Dict[str, Any],
    owner: str,
) -> str:
    globals_obj = team_raw.get("globals") or {}
    profile_cfg = globals_obj.get("owner_profile") or {}
    if profile_cfg.get("enabled", True) is False:
        return ""

    rag_cfg = globals_obj.get("rag") or {}
    env_key = profile_cfg.get("rag_env_key") or rag_cfg.get("rag_env_key") or "VECTOR_DB_TABLE"
    collection_id = os.environ.get(env_key, "") or os.environ.get("VECTOR_DB_TABLE", "")
    if not collection_id:
        log.info("owner_profile_context_skipped_missing_collection", extra={"env_key": env_key})
        return ""

    top_k = int(profile_cfg.get("top_k") or 6)
    perspective_template = profile_cfg.get(
        "perspective_template",
        "do not include any pii data about {owner}",
        "Insights about {owner} relevant for team {team_name} ({team_version}) with focus: {north_star}",
    )
    query = _to_query(owner, team_raw, request_obj, perspective_template)
    if not query:
        return ""

    hits = retrieve_from_vector_store(collection_id, query, top_k)
    if not hits:
        return ""

    blocks: List[str] = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[OWNER_PROFILE #{i}] SOURCE: {hit.get('source', '')}")
        blocks.append(hit.get("text", ""))
        blocks.append("---")

    return "\n".join(blocks).strip()

