import json
import os
import re
from typing import Any

import boto3

from .logger import get_logger

log = get_logger("storage")
s3 = boto3.client("s3")

# Words that never help a person find a run in a bucket listing.
_STOP = frozenset(
    "a an the to of for and or in on at is it my me we our your you this that "
    "with from by as be was were are do did just please can could would should "
    "i im ive".split()
)
_WORD = re.compile(r"[A-Za-z0-9]+")
# Carried across turns. They are not the prompt the run is about, and
# previous_output is large enough to become the whole name.
_SKIP_KEYS = frozenset({"previous_output", "previous_run_id", "owner", "edit_instruction"})
_PREFERRED_KEYS = (
    "dump", "thought", "idea", "experiencing", "posting_url",
    "topic", "brief", "prompt", "summary", "text", "message", "request",
)


def request_text(request: Any) -> str:
    """The words the person typed, not the bookkeeping around them."""
    if isinstance(request, str):
        return request.strip()
    if not isinstance(request, dict):
        return ""
    for key in _PREFERRED_KEYS:
        value = request.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key, value in request.items():
        if key in _SKIP_KEYS or not isinstance(value, str):
            continue
        if value.strip():
            return value.strip()
    return ""


def summarize_prompt(text: str, *, words: int = 6) -> str:
    """A short, filesystem-safe summary of a prompt.

    Extractive on purpose. Naming a run must not depend on another model call
    succeeding, and must not put the raw prompt — or a run id — in the path.
    """
    kept = []
    for raw in _WORD.findall(text or ""):
        word = raw.lower()
        if len(word) < 2 or word in _STOP:
            continue
        kept.append(word)
        if len(kept) >= words:
            break
    return "-".join(kept)


def run_folder(run_id: str, summary: str = "") -> str:
    """Human-facing folder for one run: summary slug plus a short unique suffix.

    The suffix is taken from the run id so two runs about the same thing do
    not overwrite each other. It is not the run id: a bucket listing of raw
    uuids cannot be scanned by a person.
    """
    slug = summarize_prompt(summary)[:48].strip("-")
    suffix = re.sub(r"[^a-zA-Z0-9]", "", run_id or "")[:8].lower()
    if slug and suffix:
        name = f"{slug}-{suffix}"
    elif slug:
        name = slug
    elif suffix:
        name = f"run-{suffix}"
    else:
        name = "run"
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name or "run"


def _artifact_bucket() -> str:
    return os.environ["ARTIFACT_BUCKET"]


def save_artifact(run_id: str, step_id: str, obj: Any, content_type: str = "application/json",
                  summary: str = "") -> str:
    bucket = _artifact_bucket()
    folder = run_folder(run_id, summary)
    key = f"runs/{folder}/{step_id}.json"
    body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
    s3.put_object(
        Bucket=bucket, Key=key, Body=body, ContentType=content_type,
        Metadata={"run-id": (run_id or "")[:256], "summary": folder[:256]},
    )
    log.info("artifact_saved", extra={"bucket": bucket, "key": key, "bytes": len(body), "run_id": run_id})
    return f"s3://{bucket}/{key}"


def save_bytes(run_id: str, step_id: str, data: bytes, *, extension: str,
               content_type: str, summary: str = "") -> str:
    """Store a binary artifact and return its s3:// URI.

    Separate from `save_artifact` because that one JSON-encodes its object and
    always writes `.json`. An image written through it would land as a base64
    string inside a JSON envelope -- unopenable by anything that follows the
    URI, and a third larger for no reason.
    """
    bucket = _artifact_bucket()
    folder = run_folder(run_id, summary)
    key = f"runs/{folder}/{step_id}.{extension.lstrip('.')}"
    s3.put_object(
        Bucket=bucket, Key=key, Body=data, ContentType=content_type,
        Metadata={"run-id": (run_id or "")[:256], "summary": folder[:256]},
    )
    log.info("artifact_saved", extra={"bucket": bucket, "key": key, "bytes": len(data), "run_id": run_id})
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
