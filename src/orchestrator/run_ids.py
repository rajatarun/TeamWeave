"""One definition of what a `run_id` is, on both sides of the round trip.

A client is given a `run_id` by `POST /team/task` (or an A2A task id by
`message:send`) and polls it back with `GET /team/task/{run_id}`. The poll
resolves it by rebuilding the Step Functions execution ARN from it -- so the
id the caller is handed has to *be* the execution's name.

It was not. Both start sites minted a uuid4, put it in the execution's
*input*, and called `start_execution` without `name=`, which makes Step
Functions generate a different uuid of its own. The returned id therefore
named an execution that never existed and every poll answered

    404 {"error": "run_id not found"}

however well the pipeline ran. Nothing upstream failed: the run really did
start, the input really did carry that `run_id`, and the worker used it for
its own records -- only the caller's handle was wrong.

`scripts/pipeline_smoke.py` could not see it, because it polls the
`executionArn` that `start_execution` returns rather than an id it was given.
That is the right thing for a harness to do and exactly why the deploy stayed
green while the UI could not report a single result.
"""
from __future__ import annotations

import uuid

# Step Functions bounds an execution name at 1..80 characters (botocore's
# service model for StartExecution) and rejects whitespace, control
# characters and the set < > { } [ ] ? * " # % \ ^ | ~ ` $ & , ; : /
# A uuid4 in canonical form is 36 characters of [0-9a-f-], so it is a legal
# name under every one of those rules with no escaping.
EXECUTION_NAME_MAX = 80


def new_run_id() -> str:
    """Mint an id that is usable as a Step Functions execution name."""
    return str(uuid.uuid4())


def is_valid_execution_name(run_id: str) -> bool:
    if not run_id or len(run_id) > EXECUTION_NAME_MAX:
        return False
    return all(c.isalnum() or c in "-_." for c in run_id)
