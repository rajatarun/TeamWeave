#!/usr/bin/env python3
"""Resolve every API and data-store coordinate a harness needs from the stack.

The one that is easy to miss is the pair of GSI names. ``ObservatoryMetricsTable``
tells a caller which table the spans are in; it does not tell them that the
timeline is read through ``SpanTimelineIndex`` and per-agent queries through
``AgentIdTimestampIndex``. A caller who does not name an index does not get an
error -- they get a full table scan, and on an aggregate query a truncated
answer that looks like data.

Both index names are stack Outputs now, and ``tests/test_openapi_contract.py``
asserts that infra/shared.yaml really defines an index by each name, so an
output cannot point at an index that does not exist.

Usage
-----
    python scripts/stack_env.py --stack teamweave
    eval "$(python scripts/stack_env.py --stack teamweave --format sh)"

    curl -sS "$TEAMWEAVE_API_BASE/observability"

Database credentials are not resolved here. ``TEAMWEAVE_VECTOR_DB_SECRET_ARN``
names the Secrets Manager secret holding the pgvector host, port, database name
and password; fetch it at the point of use so the password never lands in a
shell environment or a CI log.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys

REQUIRED_OUTPUTS = {
    "TEAMWEAVE_API_BASE": "HttpApiUrl",
    "TEAMWEAVE_DDB_TABLE": "DdbTable",
    "TEAMWEAVE_OBSERVATORY_TABLE": "ObservatoryMetricsTable",
    "TEAMWEAVE_OBSERVATORY_SPAN_TIMELINE_INDEX": "ObservatoryMetricsSpanTimelineIndex",
    "TEAMWEAVE_OBSERVATORY_AGENT_INDEX": "ObservatoryMetricsAgentIdTimestampIndex",
    "TEAMWEAVE_CONFIG_BUCKET": "ConfigBucket",
    "TEAMWEAVE_ARTIFACT_BUCKET": "ArtifactBucket",
    "TEAMWEAVE_STATE_MACHINE_ARN": "StateMachineArn",
}

OPTIONAL_OUTPUTS = {
    # Empty when ContextWeave is not configured -- which is exactly what a
    # harness needs in order to read GET /observability's null routingGraph as
    # "not configured" rather than "broken".
    "TEAMWEAVE_CONTEXTWEAVE_URL": "ContextWeaveUrl",
    "TEAMWEAVE_VECTOR_DB_SECRET_ARN": "VectorDbSecretArn",
    "TEAMWEAVE_AMP_WORKSPACE_ID": "AMPWorkspaceId",
    "TEAMWEAVE_AMP_WORKSPACE_ARN": "AMPWorkspaceArn",
    "TEAMWEAVE_SHARED_STACK": "SharedStackName",
    "TEAMWEAVE_API_ACCESS_LOG_GROUP_ARN": "ApiGatewayAccessLogGroupArn",
    "TEAMWEAVE_OPENAPI_SPEC": "OpenApiSpecPath",
}


class MissingOutputs(RuntimeError):
    """The stack exists but does not publish something a harness needs."""


def build_env(outputs: dict) -> dict:
    """Map stack outputs onto environment variable names (pure, so it is testable)."""
    env = {}
    missing = []
    for var, key in REQUIRED_OUTPUTS.items():
        if outputs.get(key):
            env[var] = outputs[key]
        else:
            missing.append(key)
    if missing:
        raise MissingOutputs(
            "stack publishes no value for: " + ", ".join(sorted(missing))
            + ". Deploy a template that exports them (see the Outputs section of "
            "infra/template.yaml) -- do not hardcode them in the harness."
        )
    for var, key in OPTIONAL_OUTPUTS.items():
        if outputs.get(key):
            env[var] = outputs[key]
    return env


def fetch_outputs(stack: str, region: str | None) -> dict:
    try:
        import boto3  # noqa: PLC0415 -- optional; the CLI path covers images without it
    except ImportError:
        boto3 = None

    if boto3 is not None:
        cfn = boto3.client("cloudformation", region_name=region) if region else boto3.client("cloudformation")
        stacks = cfn.describe_stacks(StackName=stack)["Stacks"]
    else:
        cmd = ["aws", "cloudformation", "describe-stacks", "--stack-name", stack, "--output", "json"]
        if region:
            cmd += ["--region", region]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}): {proc.stderr.strip()}")
        stacks = json.loads(proc.stdout)["Stacks"]

    return {o["OutputKey"]: o.get("OutputValue", "") for o in stacks[0].get("Outputs", [])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Resolve TeamWeave coordinates from the stack")
    ap.add_argument("--stack", default="teamweave", help="CloudFormation stack name")
    ap.add_argument("--region", default=None)
    ap.add_argument("--format", choices=("json", "sh"), default="json")
    args = ap.parse_args(argv)

    try:
        env = build_env(fetch_outputs(args.stack, args.region))
    except (MissingOutputs, RuntimeError) as exc:
        print(f"stack_env: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(env, indent=2, sort_keys=True))
    else:
        for var in sorted(env):
            print(f"export {var}={shlex.quote(env[var])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
