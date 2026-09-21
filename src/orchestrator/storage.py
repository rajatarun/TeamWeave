import json
import os
from typing import Any

import boto3

from .logger import get_logger

log = get_logger("storage")
s3 = boto3.client("s3")


def _artifact_bucket() -> str:
    return os.environ["ARTIFACT_BUCKET"]


def save_artifact(run_id: str, step_id: str, obj: Any, content_type: str = "application/json") -> str:
    bucket = _artifact_bucket()
    key = f"runs/{run_id}/{step_id}.json"
    body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
    log.info("artifact_saved", extra={"bucket": bucket, "key": key, "bytes": len(body)})
    return f"s3://{bucket}/{key}"


def save_bytes(run_id: str, step_id: str, data: bytes, *, extension: str,
               content_type: str) -> str:
    """Store a binary artifact and return its s3:// URI.

    Separate from `save_artifact` because that one JSON-encodes its object and
    always writes `.json`. An image written through it would land as a base64
    string inside a JSON envelope -- unopenable by anything that follows the
    URI, and a third larger for no reason.
    """
    bucket = _artifact_bucket()
    key = f"runs/{run_id}/{step_id}.{extension.lstrip('.')}"
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    log.info("artifact_saved", extra={"bucket": bucket, "key": key, "bytes": len(data)})
    return f"s3://{bucket}/{key}"


# Lambda's own credentials are temporary, so a URL signed with them dies when
# the session token does -- hours, not the 7 days SigV4 allows. That is enough
# to look at a run you just started and not enough to treat as a permalink,
# which is why the durable `s3://` URI is always returned alongside it.
PRESIGN_SECONDS = 12 * 60 * 60


def presign(uri: str, expires_in: int = PRESIGN_SECONDS) -> str:
    """A browser-openable URL for an s3:// URI, or "" if one cannot be made.

    Best-effort on purpose: the artifact is already stored and the step has
    already succeeded, so failing the run because a convenience link could not
    be signed would trade the deliverable for the preview of it.
    """
    if not uri.startswith("s3://"):
        return ""
    bucket, _, key = uri[len("s3://"):].partition("/")
    if not bucket or not key:
        return ""
    try:
        return s3.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in,
        )
    except Exception as exc:  # noqa: BLE001 - see docstring
        log.warning("presign_failed", extra={"uri": uri, "err": str(exc)[:200]})
        return ""
