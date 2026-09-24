"""Retrieve from the health knowledge base, and ask it to re-read the bucket.

The documents are ContextWeave's. Deploy reads that stack's
``HealthDocsBucketName`` and ``KMSKeyArn`` outputs and points a Bedrock
knowledge base at the bucket. This module does not create a bucket and does
not embed. The base is a managed knowledge base built with
``amazon.nova-2-multimodal-embeddings-v1:0``. Querying it is
``bedrock-agent-runtime:Retrieve`` with ``managedSearchConfiguration``.
``vectorSearchConfiguration`` is the self-managed shape and does not apply.

``query_health_record`` is the only caller of :func:`retrieve`, and it
refuses the call unless the run carries the person's token and the team is
``health_prep``. The token is not sent to Bedrock. It is the gate in front
of an IAM call that would otherwise be ambient authority over the record.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from .deadline import budget_for_call, set_deadline_from_context
from .logger import get_logger

log = get_logger("health_kb")

# The model the base is created with. Retrieval does not send it — Bedrock
# already embedded the bucket with it — and a second id here would be a
# second source of truth. The template is the one that is deployed.
EMBEDDING_MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"


def knowledge_base_id() -> str:
    return (os.environ.get("HEALTH_KNOWLEDGE_BASE_ID") or "").strip()


def data_source_id() -> str:
    return (os.environ.get("HEALTH_DATA_SOURCE_ID") or "").strip()


def configured() -> bool:
    return bool(knowledge_base_id())


def _client(service: str):
    """A client whose wait fits the time this invocation has left.

    Built per call, same reason as the agent client: the budget shrinks as
    earlier steps spend it. An injected client is how tests avoid the network.

    Retrieve on a managed base sends ``managedSearchConfiguration``. The
    Lambda runtime's botocore rejects that parameter before the request is
    signed. WorkerFunction is packaged with botocore>=1.43.92 so the model
    describes it; an older client fails here, naming that pin.
    """
    client = boto3.client(
        service,
        config=Config(
            read_timeout=budget_for_call(),
            connect_timeout=60,
            retries={"max_attempts": 0},
        ),
    )
    if service == "bedrock-agent-runtime":
        _assert_managed_retrieve(client)
    return client


def _assert_managed_retrieve(client) -> None:
    members = {}
    try:
        members = client.meta.service_model.shape_for(
            "KnowledgeBaseRetrievalConfiguration"
        ).members
    except Exception:
        members = {}
    if "managedSearchConfiguration" not in members:
        import botocore
        raise RuntimeError(
            f"botocore {botocore.__version__} has no managedSearchConfiguration. "
            "Retrieve would fail parameter validation. WorkerFunction must be "
            "built with src/requirements-bedrock-kb.txt (botocore>=1.43.92)."
        )


def retrieve(question: str, *, top_k: int = 6, client=None) -> Optional[Dict[str, Any]]:
    """Excerpts from the health base, or None when no base is configured.

    None is the signal to use the health API. A configured base that fails
    returns ``{"error": ...}`` and does not fall through, because an empty
    record and a base that could not be asked are different answers. The
    service's own message is not returned: it can echo an object key, and
    the key of a health document is itself identifying.
    """
    kb_id = knowledge_base_id()
    if not kb_id:
        return None
    question = str(question or "").strip()
    if not question:
        return {"error": "a question is required"}
    if client is None:
        client = _client("bedrock-agent-runtime")
    try:
        response = client.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                # Managed bases are queried through managedSearchConfiguration.
                # vectorSearchConfiguration is for a self-managed VECTOR base.
                "managedSearchConfiguration": {
                    "numberOfResults": max(1, int(top_k or 6)),
                },
            },
        )
    except (ClientError, BotoCoreError):
        log.warning("health_kb_retrieve_failed")
        return {"error": "the health knowledge base could not be reached"}

    excerpts: List[Dict[str, str]] = []
    for item in (response or {}).get("retrievalResults") or []:
        if not isinstance(item, dict):
            continue
        text = ((item.get("content") or {}).get("text") or "").strip()
        if not text:
            continue
        location = ((item.get("location") or {}).get("s3Location") or {})
        excerpts.append({
            "sourceKey": str(location.get("uri") or ""),
            "content": text,
        })
    log.info("health_kb_retrieve", extra={"count": len(excerpts)})
    return {
        "found": bool(excerpts),
        "excerpts": excerpts,
        "source": "bedrock-knowledge-base",
    }


def start_sync(*, client=None) -> Dict[str, str]:
    """Ask the base to re-read the health bucket.

    A job already running is not a failure: it reads the bucket as it is,
    including the object that just landed. Any other failure is re-raised so
    EventBridge retries it. The message is not logged; see :func:`retrieve`.
    """
    kb_id = knowledge_base_id()
    source_id = data_source_id()
    if not kb_id or not source_id:
        log.warning("health_kb_sync_unconfigured")
        return {}
    if client is None:
        client = _client("bedrock-agent")
    try:
        client.start_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=source_id,
        )
    except ClientError as exc:
        code = str((exc.response.get("Error") or {}).get("Code") or "")
        if code == "ConflictException":
            log.info("health_kb_sync_already_running")
            return {"kbSync": "already-running"}
        log.warning("health_kb_sync_failed", extra={"code": code})
        raise
    return {"kbSync": "started"}


def sync_handler(event: Any, context: Any) -> Dict[str, str]:
    """EventBridge entrypoint. The event names the object; it is not logged."""
    set_deadline_from_context(context)
    return start_sync()
