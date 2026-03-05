import json
import os
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


def _method(event: Dict[str, Any]) -> str:
    return (event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod") or "").upper()


def _to_execution_arn(run_id: str) -> str:
    if run_id.startswith("arn:"):
        return run_id

    state_machine_arn = os.getenv("STATE_MACHINE_ARN", "")
    if ":stateMachine:" not in state_machine_arn:
        raise ValueError("STATE_MACHINE_ARN must be configured when run_id is an execution id")

    base = state_machine_arn.replace(":stateMachine:", ":execution:", 1)
    return f"{base}:{run_id}"


# ASSUMPTION: run_id path parameter may be URL-encoded because execution ARN contains ':'.
def handler(event, context):
    if _method(event) == "OPTIONS":
        return {"statusCode": 200, "headers": _cors(), "body": ""}

    run_id = (
        event.get("pathParameters", {}).get("run_id")
        or event.get("pathParameters", {}).get("runId")
        or ""
    )
    run_id = unquote(run_id)

    if not run_id:
        return _resp(400, {"error": "run_id required"})

    try:
        execution_arn = _to_execution_arn(run_id)
    except ValueError as exc:
        return _resp(400, {"error": str(exc)})

    try:
        desc = sfn.describe_execution(executionArn=execution_arn)
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
