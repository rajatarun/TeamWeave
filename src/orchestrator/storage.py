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
