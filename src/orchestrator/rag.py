import os
import json
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse
import boto3
import psycopg
from psycopg import sql
from .logger import get_logger
from .db import DbDao

log = get_logger("rag")
bedrock_runtime = boto3.client("bedrock-runtime")


def _embed_text(text: str) -> Optional[List[float]]:
    model_id = os.environ.get("VECTOR_EMBEDDING_MODEL_ID", "titan-embed-text-v1").strip()
    if not model_id or not text.strip():
        return None

    body = {
        "inputText": text,
    }

    try:
        resp = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        raw = resp.get("body").read().decode("utf-8")
        payload = json.loads(raw)
        emb = payload.get("embedding")
        if isinstance(emb, list) and emb:
            return [float(v) for v in emb]
    except Exception:
        log.exception("vector_embedding_failed", extra={"model_id": model_id})
    return None


def _pgvector_literal(emb: List[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in emb) + "]"


def _embedding_dimension(cur: psycopg.Cursor, table_name: str) -> Optional[int]:
    stmt = sql.SQL(
        """
        SELECT vector_dims(embedding)
        FROM {table}
        WHERE embedding IS NOT NULL
        LIMIT 1
        """
    ).format(table=sql.Identifier(table_name))
    cur.execute(stmt)
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return int(row[0])

def retrieve_from_vector_store(collection_id: str, query: str, top_k: int) -> List[Dict[str, str]]:
    table_name = os.environ.get("VECTOR_DB_TABLE", "").strip()
    db_url = os.environ.get("VECTOR_DB_URL", "").strip()
    host = os.environ.get("VECTOR_DB_HOST", "").strip()
    dbname = os.environ.get("VECTOR_DB_NAME", "").strip()
    user = os.environ.get("VECTOR_DB_USER", "").strip()
    password = os.environ.get("VECTOR_DB_PASSWORD", "").strip()
    port = int(os.environ.get("VECTOR_DB_PORT", "5432"))
    ssl_mode = os.environ.get("VECTOR_DB_SSLMODE", "verify-full").strip() or "verify-full"
    ssl_root_cert = os.environ.get("VECTOR_DB_SSL_ROOT_CERT", "/var/task/certs/rds-ca-bundle.pem").strip()

    if db_url:
        parsed = urlparse(db_url)
        if parsed.scheme not in {"postgres", "postgresql"}:
            log.warning("vector_db_url_invalid_scheme", extra={"scheme": parsed.scheme})
            return []
        host = parsed.hostname or host
        dbname = (parsed.path or "").lstrip("/") or dbname
        user = unquote(parsed.username or "") or user
        password = unquote(parsed.password or "") or password
        port = parsed.port or port

    if not table_name:
        log.warning("vector_store_not_configured", extra={"required": ["VECTOR_DB_TABLE"]})
        return []

    if not all([host, dbname, user, password]):
        log.warning(
            "vector_store_not_configured",
            extra={"required": ["VECTOR_DB_HOST", "VECTOR_DB_NAME", "VECTOR_DB_USER", "VECTOR_DB_PASSWORD"]},
        )
        return []

    try:
        final_db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        log.info("vector_db_connecting", extra={"db_url": final_db_url})
        connect_kwargs = {"sslmode": ssl_mode}
        if ssl_root_cert and ssl_mode.lower() != "disable":
            if os.path.exists(ssl_root_cert):
                connect_kwargs["sslrootcert"] = ssl_root_cert
            else:
                log.warning("vector_db_ssl_root_cert_missing", extra={"path": ssl_root_cert})

        conn = psycopg.connect(final_db_url, **connect_kwargs)
        qemb = _embed_text(query)

        with conn:
            with conn.cursor() as cur:
                if qemb:
                    stored_dim = _embedding_dimension(cur, table_name)
                    query_dim = len(qemb)
                    if stored_dim is not None and stored_dim != query_dim:
                        log.warning(
                            "vector_dimension_mismatch",
                            extra={"stored_dimension": stored_dim, "query_dimension": query_dim},
                        )
                        qemb = None

                if qemb:
                    vector_literal = _pgvector_literal(qemb)
                    stmt = sql.SQL(
                        """
                        SELECT
                            doc_id,
                            chunk_id,
                            title,
                            content,
                            (1 - (embedding <=> %s::vector)) AS score
                        FROM {table} rc
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """
                    ).format(table=sql.Identifier(table_name))
                    cur.execute(stmt, (vector_literal, vector_literal, top_k))
                else:
                    stmt = sql.SQL(
                        """
                        SELECT
                            doc_id,
                            chunk_id,
                            title,
                            COALESCE(content, '') AS content,
                            NULL::double precision AS score
                        FROM {table}
                        WHERE content ILIKE ('%%' || %s || '%%')
                        LIMIT %s
                        """
                    ).format(table=sql.Identifier(table_name))
                    cur.execute(stmt, (query, top_k))
                rows = cur.fetchall()
                if qemb:
                    return [
                        {
                            "source": str(r[2] or r[0] or f"chunk:{r[1]}"),
                            "text": str(r[3] or ""),
                            "score": r[4],
                        }
                        for r in rows
                    ]
                return [
                    {
                        "source": str(r[2] or r[0] or f"chunk:{r[1]}"),
                        "text": str(r[3] or ""),
                        "score": r[4],
                    }
                    for r in rows
                ]
    except Exception:
        log.exception("vector_store_query_failed", extra={"collection_id": collection_id, "top_k": top_k})
        return []

def get_rag_context(request_obj: Dict[str, Any], team_globals: Dict[str, Any], owner: str, dao: Optional[DbDao] = None) -> str:
    rag = (team_globals or {}).get("rag") or {}
    mode = (rag.get("mode") or "kb").lower()
    top_k = int(rag.get("top_k") or 8)
    env_key = rag.get("rag_env_key") or "VECTOR_DB_TABLE"

    if mode == "kb":
        log.info("KB mode enabled; returning empty RAG_CONTEXT")
        return ""

    if mode == "explicit":
        collection_id = os.environ.get(env_key, "") or os.environ.get("VECTOR_DB_TABLE","")
        query = " ".join([request_obj.get("topic",""), request_obj.get("objective",""), request_obj.get("audience","")]).strip()
        hits = retrieve_from_vector_store(collection_id, query, top_k) if collection_id else []
        blocks = []
        for i, h in enumerate(hits, start=1):
            blocks.append(f"[RAG #{i}] SOURCE: {h.get('source','')}")
            blocks.append(h.get("text",""))
            blocks.append("---")
        return "\n".join(blocks).strip()

    if mode == "history":
        completed = dao.list_completed_topic_levels(owner=owner, limit=200) if dao else []
        if not completed:
            return ""
        return "COMPLETED_TASKS_HISTORY:\n" + "\n".join([f"- {c}" for c in completed])

    log.warning("Unknown RAG mode; returning empty", extra={"mode": mode})
    return ""
