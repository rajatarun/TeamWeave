"""Create the health knowledge base as a managed base.

``twelvelabs.marengo-embed-3-0-v1:0`` is rejected on a self-managed VECTOR
knowledge base ("the specified embedding model is not supported"). It is
accepted as a custom multimodal embedding on a managed base: ``type``
``MANAGED``, ``embeddingModelType`` ``CUSTOM``, ``embeddingDataType``
``FLOAT``, a ``modelConfiguration`` document, and a supplemental S3 location
for extracted media.

``AWS::Bedrock::KnowledgeBase`` does not expose that shape. Its managed
configuration has no supplemental storage and no ``modelConfiguration``
document, and its embedding data type enum is ``FLOAT32`` or ``BINARY``.
This function is the custom resource that calls ``CreateKnowledgeBase``
with the documented body. The managed base owns the vector store, so there
is no S3 Vectors index to create.

The Lambda runtime's botocore does not know that body. Parameter validation
fails in the client and Bedrock never sees the request. The function is
packaged with ``botocore>=1.43.92`` (``src/requirements-bedrock-kb.txt``),
which is the first model that describes ``modelConfiguration`` and the
supplemental storage location. ``_client`` refuses an older model before
the call, so a package that forgot the pin fails with that reason rather
than "Unknown parameter".

CloudFormation does not ship ``cfnresponse`` inside a packaged function.
The response is a PUT to ``ResponseURL`` with an empty ``Content-Type``.
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Callable, Dict, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from .logger import get_logger

log = get_logger("health_kb_provision")

DATA_SOURCE_NAME = "health-s3"
_POLL_SECONDS = 10
_RESPONSE_RESERVE_SECONDS = 20
_UNCREATED = "uncreated"
_MISSING = frozenset({"ResourceNotFoundException", "NotFoundException"})
_KB_TERMINAL_FAILURE = frozenset({"FAILED", "DELETE_UNSUCCESSFUL", "UPDATE_UNSUCCESSFUL"})
_DS_TERMINAL_FAILURE = frozenset({"FAILED", "DELETE_UNSUCCESSFUL"})


class _Failed(Exception):
    """A provision step failed after a knowledge base id may already exist.

    The id has to travel with the error. CloudFormation deletes whatever
    physical id a failed Create reports, and a response that still says
    ``uncreated`` would leave the base behind.
    """

    def __init__(self, message: str, physical_id: str):
        super().__init__(message)
        self.physical_id = physical_id


def supplemental_uri(bucket: str) -> str:
    name = str(bucket or "").strip().removeprefix("s3://").strip("/")
    if not name or "/" in name:
        raise ValueError("multimodal bucket must be a bucket name")
    return f"s3://{name}/"


def knowledge_base_configuration(model_arn: str, multimodal_bucket: str) -> Dict[str, Any]:
    """The CreateKnowledgeBase body Marengo accepts.

    ``embeddingDataType`` is ``FLOAT``. The managed-configuration enum in
    CloudFormation lists ``FLOAT32`` and ``BINARY``; the user guide for this
    model says ``FLOAT``, and that is the value the API is called with.
    Dimensions are absent on purpose: a managed base owns its index.
    """
    return {
        "type": "MANAGED",
        "managedKnowledgeBaseConfiguration": {
            "embeddingModelType": "CUSTOM",
            "embeddingModelArn": model_arn,
            "embeddingModelConfiguration": {
                "bedrockEmbeddingModelConfiguration": {
                    "embeddingDataType": "FLOAT",
                    "modelConfiguration": {
                        "version": "1",
                        "audio": {
                            "segmentation": {
                                "method": "dynamic",
                                "dynamic": {"minDurationSec": 4},
                            }
                        },
                        "video": {
                            "segmentation": {
                                "method": "fixed",
                                "fixed": {"durationSec": 6},
                            }
                        },
                    },
                }
            },
            "supplementalDataStorageConfiguration": {
                "storageLocations": [
                    {
                        "type": "S3",
                        "s3Location": {"uri": supplemental_uri(multimodal_bucket)},
                    }
                ]
            },
        },
    }


def data_source_configuration(bucket_name: str, account_id: str) -> Dict[str, Any]:
    """S3 connector for a managed base.

    The bucket is ContextWeave's health bucket, named, not an ARN. A VECTOR
    ``S3`` data source is the shape a self-managed base takes, and it is not
    this one.
    """
    return {
        "type": "MANAGED_KNOWLEDGE_BASE_CONNECTOR",
        "managedKnowledgeBaseConnectorConfiguration": {
            "mediaExtractionConfiguration": {
                "imageExtractionConfiguration": {
                    "imageExtractionStatus": "ENABLED",
                }
            },
            "connectorParameters": {
                "type": "S3",
                "version": "1",
                "connectionConfiguration": {
                    "bucketName": bucket_name,
                    "bucketOwnerAccountId": account_id,
                },
                "deletionProtectionConfiguration": {
                    "enableDeletionProtection": False,
                },
            },
        },
    }


def data_source_fields(bucket_name: str, account_id: str) -> Dict[str, Any]:
    return {
        "name": DATA_SOURCE_NAME,
        "description": "ContextWeave's health document bucket. This stack does not create it.",
        "dataDeletionPolicy": "DELETE",
        "dataSourceConfiguration": data_source_configuration(bucket_name, account_id),
        "vectorIngestionConfiguration": {
            "parsingConfiguration": {"parsingStrategy": "SMART_PARSING"},
        },
    }


def _assert_managed_model(client) -> None:
    """Refuse a botocore that would reject the Marengo body locally.

    The runtime copy stops at ``managedKnowledgeBaseConfiguration``. 1.43.32
    knows that member and still rejects ``modelConfiguration`` and the
    supplemental storage location. Either failure is a client-side
    validation error; the service is never asked.
    """
    import botocore

    model = client.meta.service_model
    missing = []
    kb_members = _shape_members(model, "KnowledgeBaseConfiguration")
    if "managedKnowledgeBaseConfiguration" not in kb_members:
        missing.append("managedKnowledgeBaseConfiguration")
    else:
        if "supplementalDataStorageConfiguration" not in _shape_members(
            model, "ManagedKnowledgeBaseConfiguration"
        ):
            missing.append("supplementalDataStorageConfiguration")
        if "modelConfiguration" not in _shape_members(
            model, "BedrockEmbeddingModelConfiguration"
        ):
            missing.append("modelConfiguration")
    if "MANAGED_KNOWLEDGE_BASE_CONNECTOR" not in _shape_enum(model, "DataSourceType"):
        missing.append("MANAGED_KNOWLEDGE_BASE_CONNECTOR")
    if missing:
        raise RuntimeError(
            f"botocore {botocore.__version__} cannot describe {', '.join(missing)}. "
            "CreateKnowledgeBase would fail parameter validation before Bedrock "
            "sees the request. This package needs src/requirements-bedrock-kb.txt "
            "(botocore>=1.43.92)."
        )


def _shape_members(model, name: str):
    try:
        return model.shape_for(name).members
    except Exception:
        return {}


def _shape_enum(model, name: str):
    try:
        return list(model.shape_for(name).enum or [])
    except Exception:
        return []


def _client():
    # Outside the worker's deadline helper on purpose. This function's only
    # job is one control-plane call sequence, and a read timeout as long as
    # the function would turn a stall into a Lambda timeout with no response
    # to CloudFormation. The poll loop is what waits for ACTIVE.
    client = boto3.client(
        "bedrock-agent",
        config=Config(read_timeout=60, connect_timeout=10, retries={"max_attempts": 2}),
    )
    _assert_managed_model(client)
    return client


def _code(exc: BaseException) -> str:
    if isinstance(exc, ClientError):
        return str(((exc.response or {}).get("Error") or {}).get("Code") or "")
    return ""


def _reason(exc: BaseException) -> str:
    if isinstance(exc, ClientError):
        err = (exc.response or {}).get("Error") or {}
        text = f"{err.get('Code') or 'ClientError'}: {err.get('Message') or ''}".strip()
    else:
        text = str(exc) or type(exc).__name__
    return text[:1024]


def _missing(exc: BaseException) -> bool:
    return _code(exc) in _MISSING


def _deadline(context: Any) -> float:
    remaining_ms = 60_000
    getter = getattr(context, "get_remaining_time_in_millis", None)
    if callable(getter):
        remaining_ms = int(getter())
    return time.monotonic() + max(0.0, remaining_ms / 1000.0 - _RESPONSE_RESERVE_SECONDS)


def _pause(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(min(_POLL_SECONDS, remaining))


def _token(request_id: str, kind: str) -> str:
    # CreateKnowledgeBase's client token is at least 33 characters. A retried
    # CloudFormation request keeps the same RequestId, so the retry adopts the
    # base the first attempt created instead of making a second one.
    raw = "".join(ch for ch in f"{request_id}-{kind}" if ch.isalnum() or ch == "-")
    if len(raw) < 33:
        raw = f"{raw}-{'0' * 33}"
    return raw[:256]


def _require(props: Dict[str, Any]) -> None:
    missing = [
        key for key in (
            "KnowledgeBaseName",
            "RoleArn",
            "EmbeddingModelArn",
            "MultimodalBucket",
            "DocsBucketName",
            "DocsBucketOwnerAccountId",
        )
        if not str(props.get(key) or "").strip()
    ]
    if missing:
        raise ValueError("missing " + ", ".join(missing))


def _pages(fetch: Callable[..., Dict[str, Any]], items_key: str) -> list:
    collected = []
    token = None
    while True:
        kwargs: Dict[str, Any] = {}
        if token:
            kwargs["nextToken"] = token
        page = fetch(**kwargs) or {}
        collected.extend(page.get(items_key) or [])
        token = page.get("nextToken")
        if not token:
            return collected


def _find_kb(client, name: str) -> Optional[str]:
    for item in _pages(client.list_knowledge_bases, "knowledgeBaseSummaries"):
        if item.get("name") == name:
            return str(item.get("knowledgeBaseId") or "") or None
    return None


def _find_data_source(client, kb_id: str) -> Optional[str]:
    def fetch(**kwargs):
        return client.list_data_sources(knowledgeBaseId=kb_id, **kwargs)

    for item in _pages(fetch, "dataSourceSummaries"):
        if item.get("name") == DATA_SOURCE_NAME:
            return str(item.get("dataSourceId") or "") or None
    return None


def _wait_kb(client, kb_id: str, context: Any) -> Dict[str, Any]:
    deadline = _deadline(context)
    while True:
        kb = client.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]
        status = str(kb.get("status") or "")
        if status == "ACTIVE":
            if not kb.get("knowledgeBaseArn"):
                raise _Failed("knowledge base became ACTIVE without an ARN", kb_id)
            return kb
        if status in _KB_TERMINAL_FAILURE:
            reasons = kb.get("failureReasons") or []
            detail = "; ".join(str(item) for item in reasons) or status
            raise _Failed(detail, kb_id)
        if time.monotonic() >= deadline:
            raise _Failed(f"knowledge base stayed {status or 'unknown'}", kb_id)
        _pause(deadline)


def _wait_data_source(client, kb_id: str, data_source_id: str, context: Any) -> None:
    deadline = _deadline(context)
    while True:
        source = client.get_data_source(
            knowledgeBaseId=kb_id, dataSourceId=data_source_id,
        )["dataSource"]
        status = str(source.get("status") or "")
        if status == "AVAILABLE":
            return
        if status in _DS_TERMINAL_FAILURE:
            raise _Failed(f"data source {status}", kb_id)
        if time.monotonic() >= deadline:
            raise _Failed(f"data source stayed {status or 'unknown'}", kb_id)
        _pause(deadline)


def _create_kb(client, event: Dict[str, Any], props: Dict[str, Any]) -> str:
    configuration = knowledge_base_configuration(
        str(props["EmbeddingModelArn"]), str(props["MultimodalBucket"]),
    )
    kwargs: Dict[str, Any] = {
        "clientToken": _token(str(event.get("RequestId") or ""), "kb"),
        "name": str(props["KnowledgeBaseName"]),
        "roleArn": str(props["RoleArn"]),
        "knowledgeBaseConfiguration": configuration,
    }
    description = str(props.get("Description") or "").strip()
    if description:
        kwargs["description"] = description
    try:
        created = client.create_knowledge_base(**kwargs)["knowledgeBase"]
        return str(created["knowledgeBaseId"])
    except ClientError as exc:
        if _code(exc) != "ConflictException":
            raise
        found = _find_kb(client, str(props["KnowledgeBaseName"]))
        if not found:
            raise
        return found


def _update_kb(client, kb_id: str, props: Dict[str, Any]) -> None:
    kwargs: Dict[str, Any] = {
        "knowledgeBaseId": kb_id,
        "name": str(props["KnowledgeBaseName"]),
        "roleArn": str(props["RoleArn"]),
        "knowledgeBaseConfiguration": knowledge_base_configuration(
            str(props["EmbeddingModelArn"]), str(props["MultimodalBucket"]),
        ),
    }
    description = str(props.get("Description") or "").strip()
    if description:
        kwargs["description"] = description
    client.update_knowledge_base(**kwargs)


def _ensure_data_source(client, kb_id: str, props: Dict[str, Any], event: Dict[str, Any], context: Any) -> str:
    fields = data_source_fields(
        str(props["DocsBucketName"]).strip(),
        str(props["DocsBucketOwnerAccountId"]).strip(),
    )
    existing = _find_data_source(client, kb_id)
    if existing:
        client.update_data_source(knowledgeBaseId=kb_id, dataSourceId=existing, **fields)
        data_source_id = existing
    else:
        try:
            created = client.create_data_source(
                knowledgeBaseId=kb_id,
                clientToken=_token(str(event.get("RequestId") or ""), "ds"),
                **fields,
            )["dataSource"]
            data_source_id = str(created["dataSourceId"])
        except ClientError as exc:
            if _code(exc) != "ConflictException":
                raise
            data_source_id = _find_data_source(client, kb_id) or ""
            if not data_source_id:
                raise
    _wait_data_source(client, kb_id, data_source_id, context)
    return data_source_id


def _upsert(event: Dict[str, Any], context: Any):
    props = event.get("ResourceProperties") or {}
    _require(props)
    client = _client()
    kb_id = ""
    try:
        if event.get("RequestType") == "Create":
            kb_id = _create_kb(client, event, props)
        else:
            kb_id = str(event.get("PhysicalResourceId") or "")
            if not kb_id or kb_id == _UNCREATED:
                kb_id = _create_kb(client, event, props)
            else:
                # A retried update can arrive while the previous one is still
                # settling. UpdateKnowledgeBase on a base that is not ACTIVE
                # conflicts, so wait first.
                _wait_kb(client, kb_id, context)
                _update_kb(client, kb_id, props)
        kb = _wait_kb(client, kb_id, context)
        data_source_id = _ensure_data_source(client, kb_id, props, event, context)
    except _Failed:
        raise
    except (ClientError, BotoCoreError, ValueError) as exc:
        raise _Failed(_reason(exc), kb_id or _UNCREATED) from exc
    return {
        "KnowledgeBaseId": kb["knowledgeBaseId"],
        "KnowledgeBaseArn": kb["knowledgeBaseArn"],
        "DataSourceId": data_source_id,
    }, kb_id


def _list_data_sources(client, kb_id: str) -> list:
    def fetch(**kwargs):
        return client.list_data_sources(knowledgeBaseId=kb_id, **kwargs)

    return _pages(fetch, "dataSourceSummaries")


def _delete(client, kb_id: str, context: Any) -> None:
    deadline = _deadline(context)
    try:
        while True:
            summaries = _list_data_sources(client, kb_id)
            if not summaries:
                break
            for item in summaries:
                if item.get("status") == "DELETING":
                    continue
                source_id = item.get("dataSourceId")
                if not source_id:
                    continue
                try:
                    client.delete_data_source(knowledgeBaseId=kb_id, dataSourceId=source_id)
                except ClientError as exc:
                    if not _missing(exc):
                        raise
            if time.monotonic() >= deadline:
                raise _Failed("data sources were still present at delete", kb_id)
            _pause(deadline)
        client.delete_knowledge_base(knowledgeBaseId=kb_id)
    except ClientError as exc:
        if _missing(exc):
            return
        raise
    while True:
        try:
            kb = client.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]
        except ClientError as exc:
            if _missing(exc):
                return
            raise
        if str(kb.get("status") or "") == "DELETE_UNSUCCESSFUL":
            raise _Failed("DELETE_UNSUCCESSFUL", kb_id)
        if time.monotonic() >= deadline:
            raise _Failed("knowledge base delete did not finish", kb_id)
        _pause(deadline)


def _respond(event: Dict[str, Any], context: Any, status: str, data: Dict[str, Any], physical_id: str, reason: str = "") -> None:
    # PUT, empty Content-Type. A JSON content type is refused, and the
    # packaged function does not contain the cfnresponse module that hides this.
    body = {
        "Status": status,
        "Reason": (reason or f"See CloudWatch Log Stream: {getattr(context, 'log_stream_name', '')}")[:1024],
        "PhysicalResourceId": physical_id or _UNCREATED,
        "StackId": event.get("StackId"),
        "RequestId": event.get("RequestId"),
        "LogicalResourceId": event.get("LogicalResourceId"),
        "NoEcho": False,
        "Data": data or {},
    }
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        event["ResponseURL"],
        data=payload,
        method="PUT",
        headers={
            "Content-Type": "",
            "Content-Length": str(len(payload)),
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    physical = str(event.get("PhysicalResourceId") or "") or _UNCREATED
    try:
        if event.get("RequestType") == "Delete":
            if physical != _UNCREATED:
                try:
                    _delete(_client(), physical, context)
                except _Failed:
                    raise
                except (ClientError, BotoCoreError) as exc:
                    if not _missing(exc):
                        raise _Failed(_reason(exc), physical) from exc
            _respond(event, context, "SUCCESS", {}, physical)
            return {"PhysicalResourceId": physical}
        data, physical = _upsert(event, context)
        _respond(event, context, "SUCCESS", data, physical)
        return data
    except Exception as exc:
        physical = getattr(exc, "physical_id", None) or physical
        # The code is enough for the log. The API's own message goes to
        # CloudFormation, which is where the last two create failures were read.
        log.warning(
            "health_kb_provision_failed",
            extra={"code": _code(exc) or type(exc).__name__},
        )
        _respond(event, context, "FAILED", {}, str(physical), reason=_reason(exc))
        return {"Status": "FAILED", "PhysicalResourceId": str(physical)}
