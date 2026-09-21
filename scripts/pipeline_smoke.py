#!/usr/bin/env python3
"""Run one real team pipeline after a deploy, and read what it produced.

The platform exists so that someone can ask a team to do something and get the
output. Nothing verified that. `agentcore_smoke.py` proves one runtime boots
and answers a turn; a green `sam deploy` proves CloudFormation accepted some
resources. Neither says whether a pipeline runs end to end -- whether the
worker resolves the team's runtime, whether each agent answers, whether the
outputs validate against their schemas, whether the last step produces
anything.

Every failure this session has had the same shape: green everywhere, broken in
the one place nobody looked. So this looks.

Three outcomes, kept distinct on purpose:

  * a run that SUCCEEDS with a non-empty final step passes;
  * a run that FAILS, times out, or succeeds with nothing in it **fails the
    deploy** -- an empty success is the failure mode that hides best;
  * a CI role that cannot start executions is the *check* failing rather than
    the pipeline, so it warns with NOT VERIFIED and does not fail the deploy.
    Failing every deploy over a permissions gap would be wrong; reporting it
    as a pass would be worse.

Started through Step Functions rather than the HTTP API because the API is
behind the SIWE authorizer and CI holds no wallet. That is the trade: this
covers the pipeline, not the authorizer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.orchestrator import run_ids  # noqa: E402

# The visibility team is the product path, so the deploy exercises what people
# actually run. Small inputs on purpose: the point is that the machinery works
# end to end, not that the writing is good, and every deploy pays for this in
# Bedrock calls and wall clock.
DEFAULT_TEAM = "tarun_visibility_team"
DEFAULT_VERSION = "v1"
DEFAULT_REQUEST = {
    "topic": "Why per-team agent runtimes beat one runtime per agent",
    "objective": "Show depth in AWS agent orchestration",
    "audience": "Senior platform and AI engineers",
}


def announce(level: str, message: str) -> None:
    """GitHub Actions reads workflow commands from stdout only.

    A warning written to stderr produces no annotation at all in a step that
    exits 0, which is exactly the case this script cares most about surfacing.
    """
    print(f"::{level}::{message}", flush=True)


def start(sfn, state_machine_arn: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    """Start the run the way the trigger does, and return (arn, run_id)."""
    run_id = run_ids.new_run_id()
    response = sfn.start_execution(
        stateMachineArn=state_machine_arn,
        name=run_id,
        input=json.dumps({**payload, "run_id": run_id}),
    )
    return response["executionArn"], run_id


def wait(sfn, execution_arn: str, timeout_s: int, poll_s: int) -> Tuple[str, Dict[str, Any]]:
    """Poll to a terminal state. Returns (status, parsed output)."""
    deadline = time.time() + timeout_s
    while True:
        described = sfn.describe_execution(executionArn=execution_arn)
        status = described.get("status", "UNKNOWN")
        if status != "RUNNING":
            raw = described.get("output")
            try:
                parsed = json.loads(raw) if raw else {}
            except (ValueError, TypeError):
                parsed = {"unparseable_output": str(raw)[:500]}
            if status != "SUCCEEDED":
                parsed.setdefault("cause", described.get("cause", ""))
            return status, parsed
        if time.time() > deadline:
            return "TIMED_OUT_LOCALLY", {"cause": f"still RUNNING after {timeout_s}s"}
        time.sleep(poll_s)


def failure_detail(sfn, execution_arn: str, limit: int = 200) -> str:
    """Why the execution actually failed, from its own history.

    DescribeExecution's `cause` for a Lambda task is "Worker task failed" --
    the state machine's Catch, not the error. The Lambda's exception, message
    and stack trace are in the execution history, on the *FailedEventDetails
    of the failing event. Without this the check is red and mute, and whoever
    reads it still has to go to CloudWatch to learn anything.
    """
    try:
        history = sfn.get_execution_history(
            executionArn=execution_arn, reverseOrder=True, maxResults=limit
        )
    except Exception as exc:  # noqa: BLE001 - diagnosis must never mask the failure
        return f"(could not read the execution history: {type(exc).__name__}: {exc})"

    detail_keys = (
        "taskFailedEventDetails",
        "lambdaFunctionFailedEventDetails",
        "executionFailedEventDetails",
        "activityFailedEventDetails",
    )
    lines = []
    for event in history.get("events", []):
        for key in detail_keys:
            details = event.get(key)
            if not details:
                continue
            error = str(details.get("error") or "").strip()
            cause = str(details.get("cause") or "").strip()
            if not error and not cause:
                continue
            lines.append(f"[{event.get('type')}] {error}: {cause}"[:2000])
            break
        if len(lines) >= 3:
            break
    return "\n".join(lines) if lines else "(the history recorded no failure detail)"


def final_step_output(result: Dict[str, Any]) -> Tuple[str, Any]:
    """The last step's output -- the team's deliverable.

    `steps` is a dict keyed by step id. Python preserves insertion order and
    the worker writes steps in workflow order, so the last key is the last
    step. Read defensively anyway: this runs against whatever the deploy
    produced, not against a fixture.
    """
    steps = result.get("steps")
    if not isinstance(steps, dict) or not steps:
        return "", None
    last_id = list(steps)[-1]
    return last_id, steps[last_id]


def is_substantive(output: Any) -> bool:
    """Whether the final step actually produced something.

    A run that returns `{}` or `""` has SUCCEEDED by every signal the platform
    emits and delivered nothing. That is the outcome this whole check exists
    to catch, so it is a failure here.
    """
    if output is None:
        return False
    if isinstance(output, str):
        return bool(output.strip())
    if isinstance(output, (dict, list)):
        return len(output) > 0
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-machine-arn", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--team", default=DEFAULT_TEAM)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--poll", type=int, default=10)
    args = parser.parse_args()

    payload = {"team": args.team, "version": args.version, "request": DEFAULT_REQUEST}

    try:
        sfn = boto3.client("stepfunctions", region_name=args.region)
        execution_arn, run_id = start(sfn, args.state_machine_arn, payload)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"AccessDeniedException", "AccessDenied", "UnrecognizedClientException"}:
            announce(
                "warning",
                "NOT VERIFIED: this role cannot start executions, so the pipeline was "
                f"not exercised. The deploy is not failed over a permissions gap. ({code})",
            )
            return 0
        announce("error", f"Could not start the pipeline: {exc}")
        return 1
    except (BotoCoreError, Exception) as exc:  # noqa: BLE001
        announce(
            "warning",
            f"NOT VERIFIED: the pipeline check could not run ({type(exc).__name__}: {exc}).",
        )
        return 0

    print(f"Started {args.team} {args.version}: {execution_arn}", flush=True)

    # The one fact no unit test can establish: that the rule the status handler
    # uses to turn a run_id back into an ARN matches an ARN Step Functions
    # really produced. A fake written from the same understanding of the shape
    # would agree with a wrong rule, and a caller polling a run_id would 404.
    rebuilt = run_ids.to_execution_arn(run_id, args.state_machine_arn)
    if rebuilt != execution_arn:
        announce(
            "error",
            "A run_id does not resolve back to its execution: the status handler "
            f"would look up {rebuilt} for a run Step Functions created as "
            f"{execution_arn}. Every poll of every run would 404.",
        )
        return 1
    status, result = wait(sfn, execution_arn, args.timeout, args.poll)

    if status != "SUCCEEDED":
        detail = failure_detail(sfn, execution_arn)
        announce(
            "error",
            f"The {args.team} pipeline ended as {status}. Asking a team to do something "
            f"is what this platform is for, so this fails the deploy. "
            f"Cause: {str(result.get('cause'))[:300]}",
        )
        # Printed rather than annotated: a stack trace does not fit in an
        # annotation, and this is the part that says what to fix.
        print(f"--- why it failed ---\n{detail}\n---------------------", flush=True)
        return 1

    step_id, output = final_step_output(result)
    if not is_substantive(output):
        announce(
            "error",
            f"The {args.team} pipeline SUCCEEDED but its final step "
            f"({step_id or 'none found'}) produced nothing usable. An empty success is "
            f"the failure that hides best, so this fails the deploy. "
            f"Steps present: {sorted(result.get('steps') or {})}",
        )
        return 1

    rendered = json.dumps(output, ensure_ascii=False)
    announce(
        "notice",
        f"{args.team} ran end to end: {len(result.get('steps') or {})} steps, final step "
        f"{step_id} produced {len(rendered)} chars. Asking a team to do something works.",
    )
    print(rendered[:1500], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
