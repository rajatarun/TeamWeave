"""Observatory metrics Lambda handler.

Serves ``GET /observability/metrics`` by executing a PromQL query against the
Amazon Managed Prometheus (AMP) workspace and returning the raw API response.

Query parameters
----------------
query   (required) -- PromQL expression
start   (optional) -- Unix epoch seconds; triggers a *range* query when paired
                       with ``end``
end     (optional) -- Unix epoch seconds
step    (optional) -- range query resolution step (default: ``"60s"``)
time    (optional) -- Unix epoch seconds for an *instant* query timestamp

Responses
---------
200  JSON body from AMP  {"status": "success", "data": {...}}
400  {"error": "query parameter required"}
5xx  AMP error propagated transparently
"""

from __future__ import annotations

import json
import os
from urllib.parse import urlencode

import boto3
import botocore.auth
import botocore.awsrequest
import requests


def handler(event: dict, context: object) -> dict:
    params: dict[str, str] = event.get("queryStringParameters") or {}
    promql = params.get("query")

    if not promql:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "query parameter required"}),
        }

    workspace_id = os.environ["AMP_WORKSPACE_ID"]
    region = os.environ.get("AMP_REGION", "us-east-1")
    base = (
        f"https://aps-workspaces.{region}.amazonaws.com"
        f"/workspaces/{workspace_id}"
    )

    is_range = "start" in params and "end" in params
    if is_range:
        endpoint = f"{base}/api/v1/query_range"
        qp: dict[str, str] = {
            "query": promql,
            "start": params["start"],
            "end": params["end"],
            "step": params.get("step", "60s"),
        }
    else:
        endpoint = f"{base}/api/v1/query"
        qp = {"query": promql}
        if "time" in params:
            qp["time"] = params["time"]

    full_url = f"{endpoint}?{urlencode(qp)}"

    session = boto3.session.Session()
    creds = session.get_credentials().get_frozen_credentials()
    aws_req = botocore.awsrequest.AWSRequest(method="GET", url=full_url)
    botocore.auth.SigV4Auth(creds, "aps", region).add_auth(aws_req)

    resp = requests.get(full_url, headers=dict(aws_req.headers), timeout=10)

    return {
        "statusCode": resp.status_code,
        "headers": {"Content-Type": "application/json"},
        "body": resp.text,
    }
