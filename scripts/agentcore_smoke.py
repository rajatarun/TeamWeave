"""Prove the deployed AgentCore runtime can actually serve a turn.

A successful `sam deploy` says CloudFormation created the resource. It does
not say the program inside it starts, that the entrypoint exposes an ASGI app,
that the zip has the SDK in it, or that the execution role can reach Bedrock.
Each of those fails at the first invocation, which -- until something invokes
it in CI -- means the first real pipeline run, hours later, with the failure
surfacing as a step that timed out rather than a runtime that never booted.

So this invokes it once, for real, and fails the deploy if it cannot answer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, UnknownServiceError

# A failure of the *caller* rather than of the runtime. The CI role not being
# allowed to invoke, or a botocore too old to know the service, both mean the
# check could not run -- which is not evidence that the runtime is broken, and
# failing every deploy on it would be wrong. It is reported loudly instead of
# passing quietly, because "could not verify" must not read like "verified".
_CANNOT_VERIFY_CODES = {"AccessDeniedException", "UnrecognizedClientException", "ExpiredTokenException"}

# Exit codes, so the workflow step and a human read the same three outcomes.
EXIT_OK = 0
EXIT_BROKEN = 1
EXIT_CANNOT_VERIFY = 0


def cannot_verify_reason(exc: Exception) -> str:
    """Why this run could not check the runtime, or "" if it could."""
    if isinstance(exc, UnknownServiceError):
        return f"botocore does not know the bedrock-agentcore service: {exc}"
    if isinstance(exc, ClientError):
        code = ((exc.response or {}).get("Error") or {}).get("Code", "")
        if code in _CANNOT_VERIFY_CODES:
            return f"the CI role cannot invoke the runtime ({code})"
    return ""

# Same rule the orchestrator applies: InvokeAgentRuntime's runtimeSessionId has
# a minimum length of 33, and a shorter one is a ValidationException at the API.
SESSION_PREFIX = "teamweave-ci-smoke-"

PROMPT = (
    "ROLE: Smoke test\n"
    "STEP_GOAL:\nReply with the JSON object {\"ok\": true} and nothing else.\n"
    "OUTPUT CONTRACT:\nReturn only valid JSON."
)


def invoke_once(client, runtime_arn: str, session_id: str) -> dict:
    payload = json.dumps({"prompt": PROMPT, "sessionId": session_id}).encode("utf-8")
    response = client.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        runtimeSessionId=session_id,
        payload=payload,
    )
    status = response.get("statusCode")
    if status is not None and int(status) >= 400:
        raise RuntimeError(f"InvokeAgentRuntime returned HTTP {status}")
    stream = response.get("response")
    if stream is None:
        raise RuntimeError("InvokeAgentRuntime returned no 'response' body")
    # A streaming blob, read once -- not an event stream to iterate.
    body = stream.read() if hasattr(stream, "read") else bytes(stream)
    text = body.decode("utf-8", errors="ignore").strip()
    if not text:
        raise RuntimeError("InvokeAgentRuntime returned an empty body")
    try:
        return json.loads(text)
    except ValueError as exc:
        raise RuntimeError(f"Response was not JSON: {text[:400]}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-arn", required=True)
    parser.add_argument("--region", default=None)
    # A runtime that has just been created can take a moment to become
    # invocable. Retrying that is not papering over a failure; invoking a
    # resource the same deploy created is the one case where "not ready yet"
    # is a real and temporary state.
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--delay", type=float, default=20.0)
    args = parser.parse_args()

    try:
        client = boto3.client(
            "bedrock-agentcore",
            region_name=args.region,
            config=Config(read_timeout=300, connect_timeout=30, retries={"max_attempts": 0}),
        )
    except UnknownServiceError as exc:
        print(f"::warning::AgentCore runtime NOT VERIFIED -- {cannot_verify_reason(exc)}", file=sys.stderr)
        return EXIT_CANNOT_VERIFY

    last_error: Exception | None = None
    for attempt in range(1, args.attempts + 1):
        session_id = f"{SESSION_PREFIX}{int(time.time())}-{attempt:02d}".ljust(33, "0")
        try:
            body = invoke_once(client, args.runtime_arn, session_id)
        except (ClientError, RuntimeError) as exc:
            reason = cannot_verify_reason(exc)
            if reason:
                # Retrying will not grant a permission. Say so once and stop.
                print(f"::warning::AgentCore runtime NOT VERIFIED -- {reason}", file=sys.stderr)
                return EXIT_CANNOT_VERIFY
            last_error = exc
            print(f"attempt {attempt}/{args.attempts} failed: {exc}", file=sys.stderr)
            if attempt < args.attempts:
                time.sleep(args.delay)
            continue

        # The entrypoint returns {"error": ..., "result": ""} for a payload it
        # could not read. That is a 200, so it has to be checked explicitly or
        # a runtime that rejects every request passes this test.
        if body.get("error"):
            print(f"runtime answered with an error: {body['error']}", file=sys.stderr)
            return EXIT_BROKEN
        result = body.get("result")
        if not isinstance(result, str) or not result.strip():
            print(f"runtime answered with no result: {json.dumps(body)[:400]}", file=sys.stderr)
            return EXIT_BROKEN

        print(f"AgentCore runtime served a turn on attempt {attempt}.")
        print(f"  modelId: {body.get('modelId', '<unset>')}")
        print(f"  result:  {result[:200]}")
        return EXIT_OK

    print(f"AgentCore runtime never served a turn: {last_error}", file=sys.stderr)
    return EXIT_BROKEN


if __name__ == "__main__":
    raise SystemExit(main())
