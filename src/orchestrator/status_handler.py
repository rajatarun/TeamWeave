import json
from urllib.parse import unquote
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

sfn = boto3.client("stepfunctions")


def _cors() -> Dict[str, str]:
    return {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
        "access-control-allow-headers": "content-type",
        "access-control-allow-methods": "POST,GET,OPTIONS",
    }


def _resp(code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {"statusCode": code, "headers": _cors(), "body": json.dumps(body, ensure_ascii=False, default=str)}


# ASSUMPTION: run_id path parameter may be URL-encoded because execution ARN contains ':'.
def handler(event, context):
    run_id = (
        event.get("pathParameters", {}).get("run_id")
        or event.get("pathParameters", {}).get("runId")
        or ""
    )
    run_id = unquote(run_id)

    if not run_id:
        return _resp(400, {"error": "run_id required"})

    try:
        desc = sfn.describe_execution(executionArn=run_id)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code == "ExecutionDoesNotExist":
            return _resp(404, {"error": "run_id not found"})
        return _resp(500, {"error": exc.response.get("Error", {}).get("Message", str(exc))})

    status = desc.get("status", "UNKNOWN")
    if status == "RUNNING":
        return _resp(200, {"status": "RUNNING"})

    if status == "SUCCEEDED":
        output = desc.get("output")
        parsed_output = json.loads(output) if output else None
        return _resp(200, {"status": "SUCCEEDED", "result": parsed_output})

    if status == "FAILED":
        return _resp(200, {"status": "FAILED", "error": desc.get("cause")})

    return _resp(200, {"status": status})
