"""Retrieve from the health knowledge base, and ask it to re-read the bucket.

The documents are ContextWeave's. Deploy reads that stack's
``HealthDocsBucketName`` and ``KMSKeyArn`` outputs and points a Bedrock
knowledge base at the bucket. This module does not create a bucket and does
not embed. The base is a managed knowledge base built with
``amazon.nova-2-multimodal-embeddings-v1:0``. Querying it is
``bedrock-agent-runtime:Retrieve`` with ``managedSearchConfiguration``.
``vectorSearchConfiguration`` is the self-managed shape and does not apply.

``query_health_record`` calls :func:`retrieve` for the health base and
refuses the call unless the run carries the person's token and the team is
``health_prep``. The token is not sent to Bedrock. It is the gate in front
of an IAM call that would otherwise be ambient authority over the record.
``portfolio_kb`` calls the same function with an explicit knowledge base id.
"""
from __future__ import annotations

import os
import random
import time
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


# StartIngestionJob is rate limited per account. A burst of uploads used to
# call it once per object with retries disabled, and Bedrock answered
# ThrottlingException on the first extra call. Sync uses adaptive retries.
# Retrieve stays at zero: a hung query must surface inside this invocation,
# not sit in a retry loop until Lambda kills the worker.
SYNC_MAX_ATTEMPTS = 4
SYNC_BASE_DELAY_SECONDS = 0.5
SYNC_MAX_DELAY_SECONDS = 8.0
_ACTIVE_JOBS = frozenset({"STARTING", "IN_PROGRESS"})
_THROTTLE_CODES = frozenset({
    "ThrottlingException",
    "TooManyRequestsException",
    "Throttling",
    "RequestLimitExceeded",
})


class SyncDeferred(RuntimeError):
    """The job was not started, and the event has to be tried again.

    A running ingestion reads the bucket as it was when that job started.
    An upload that arrives afterwards is not in it. Reporting success here
    deletes the event, and that object is never indexed.
    """


def _client(service: str, *, sync: bool = False):
    """A client whose wait fits the time this invocation has left.

    Built per call, same reason as the agent client: the budget shrinks as
    earlier steps spend it. An injected client is how tests avoid the network.

    Retrieve on a managed base sends ``managedSearchConfiguration``. The
    Lambda runtime's botocore rejects that parameter before the request is
    signed. WorkerFunction is packaged with botocore>=1.43.92 so the model
    describes it; an older client fails here, naming that pin.

    ``sync=True`` is the ingestion client. Adaptive mode retries throttling
    with its own backoff. Retrieve does not, on purpose.
    """
    client = boto3.client(
        service,
        config=Config(
            read_timeout=budget_for_call(),
            connect_timeout=60,
            retries=(
                {"max_attempts": 8, "mode": "adaptive"}
                if sync
                else {"max_attempts": 0}
            ),
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


def retrieve(question: str, *, top_k: int = 6, client=None,
             kb_id: Optional[str] = None, log_label: str = "health_kb") -> Optional[Dict[str, Any]]:
    """Excerpts from a managed knowledge base, or None when no base is configured.

    None is the signal to use the health API. A configured base that fails
    returns ``{"error": ...}`` and does not fall through, because an empty
    record and a base that could not be asked are different answers. The
    service's own message is not returned: it can echo an object key, and
    the key of a health document is itself identifying.

    ``kb_id`` defaults to the health base. The portfolio tool passes its own
    id; an explicit empty string is "not configured" and does not fall
    through to the health base.
    """
    if kb_id is None:
        kb_id = knowledge_base_id()
    else:
        kb_id = str(kb_id).strip()
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
        log.warning("%s_retrieve_failed", log_label)
        # The service message can echo an object key. Say which base failed
        # and nothing about which object was asked for.
        if log_label == "health_kb":
            return {"error": "the health knowledge base could not be reached"}
        return {"error": "the portfolio knowledge base could not be reached"}

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
    log.info("%s_retrieve", log_label, extra={"count": len(excerpts)})
    return {
        "found": bool(excerpts),
        "excerpts": excerpts,
        "source": "bedrock-knowledge-base",
    }


def _error_code(exc: BaseException) -> str:
    if not isinstance(exc, ClientError):
        return ""
    return str((exc.response.get("Error") or {}).get("Code") or "")


def _throttled(exc: BaseException) -> bool:
    if not isinstance(exc, ClientError):
        return False
    if _error_code(exc) in _THROTTLE_CODES:
        return True
    # The production failure was ThrottlingException whose message is
    # "Rate limit exceeded". A code this set does not name yet still says so.
    message = str((exc.response.get("Error") or {}).get("Message") or "")
    lowered = message.lower()
    return "rate limit" in lowered or "throttl" in lowered


def _backoff_sleep(seconds: float) -> None:
    time.sleep(seconds)


def _pause(attempt: int) -> None:
    delay = min(SYNC_MAX_DELAY_SECONDS, SYNC_BASE_DELAY_SECONDS * (2 ** attempt))
    # Full jitter: the sleep is delay plus a random slice of delay, so two
    # callers that were throttled together do not wake on the same instant.
    _backoff_sleep(delay + random.uniform(0, delay))


def job_in_progress(client, kb_id: str, source_id: str) -> bool:
    """True when this data source already has a job Bedrock has not finished.

    List is newest first. An older STARTING job still counts: Bedrock rejects
    a second start with ConflictException, and the object that provoked this
    call may have landed after that job began.
    """
    response = client.list_ingestion_jobs(
        knowledgeBaseId=kb_id,
        dataSourceId=source_id,
        maxResults=10,
        sortBy={"attribute": "STARTED_AT", "order": "DESCENDING"},
    )
    for summary in response.get("ingestionJobSummaries") or []:
        if not isinstance(summary, dict):
            continue
        if str(summary.get("status") or "").upper() in _ACTIVE_JOBS:
            return True
    return False


def start_sync(*, client=None, kb_id: Optional[str] = None,
               source_id: Optional[str] = None) -> Dict[str, str]:
    """Ask a base to re-read its bucket, or say the caller must try later.

    ``{"kbSync": "deferred"}`` means a job is already STARTING or IN_PROGRESS,
    or StartIngestionJob is still throttled after the backoff. It is not
    success. The running job does not see objects that arrive after it
    starts, so the caller has to run again once that job finishes.

    Omitted ids are the health base. An explicit empty string stays empty,
    so a portfolio sync with no id does not start a job on the health base.
    The event is not logged; see :func:`retrieve`.
    """
    if kb_id is None:
        kb_id = knowledge_base_id()
    else:
        kb_id = str(kb_id).strip()
    if source_id is None:
        source_id = data_source_id()
    else:
        source_id = str(source_id).strip()
    if not kb_id or not source_id:
        log.warning("kb_sync_unconfigured")
        return {}
    if client is None:
        client = _client("bedrock-agent", sync=True)

    for attempt in range(SYNC_MAX_ATTEMPTS):
        try:
            if job_in_progress(client, kb_id, source_id):
                log.info("kb_sync_deferred")
                return {"kbSync": "deferred"}
            client.start_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=source_id,
            )
            return {"kbSync": "started"}
        except ClientError as exc:
            if _error_code(exc) == "ConflictException":
                # Lost the race with a start this process did not make.
                log.info("kb_sync_deferred")
                return {"kbSync": "deferred"}
            if _throttled(exc) and attempt + 1 < SYNC_MAX_ATTEMPTS:
                log.warning("kb_sync_throttled", extra={"attempt": attempt + 1})
                _pause(attempt)
                continue
            if _throttled(exc):
                log.warning("kb_sync_throttled")
                return {"kbSync": "deferred"}
            log.warning("kb_sync_failed", extra={"code": _error_code(exc)})
            raise
    log.warning("kb_sync_throttled")
    return {"kbSync": "deferred"}


def _sqs_records(event: Any) -> List[Dict[str, Any]]:
    if not isinstance(event, dict):
        return []
    records = event.get("Records")
    if not isinstance(records, list):
        return []
    found = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("eventSource") != "aws:sqs":
            continue
        if not record.get("messageId"):
            continue
        found.append(record)
    return found


def handle_sync_event(event: Any, context: Any, *,
                      kb_id: Optional[str] = None,
                      source_id: Optional[str] = None) -> Dict[str, Any]:
    """One invocation starts at most one ingestion job.

    The queue gathers the object events. This function does not loop over
    them: a batch of uploads is one ``StartIngestionJob``. When that start
    is deferred, every message in the batch is returned to the queue. SQS
    makes them visible again after the visibility timeout, which is the
    delay until the running job can be followed by another.

    A direct invoke has no message to return. Defer raises so the caller
    retries instead of recording the sync as done.
    """
    set_deadline_from_context(context)
    result = start_sync(kb_id=kb_id, source_id=source_id)
    records = _sqs_records(event)
    if records:
        if result.get("kbSync") == "deferred":
            return {
                "batchItemFailures": [
                    {"itemIdentifier": record["messageId"]} for record in records
                ]
            }
        return {"batchItemFailures": []}
    if result.get("kbSync") == "deferred":
        raise SyncDeferred(
            "ingestion job was not started; the event must be retried"
        )
    return result


def sync_handler(event: Any, context: Any) -> Dict[str, Any]:
    """SQS entrypoint for the health bucket. The event names the object; it is not logged."""
    return handle_sync_event(event, context)
