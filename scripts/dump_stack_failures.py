#!/usr/bin/env python3
"""Print the resource failures belonging to *this* run, and nothing else.

Two problems with dumping `describe-stack-events`, both of which cost real
time on this stack:

1. Unfiltered, it prints every event with its full ResourceProperties --
   thousands of lines of JSON, almost all of it resources that updated fine,
   with the one event carrying a ResourceStatusReason pushed out of the log
   tail entirely.

2. Filtered to failures but not to *time*, it prints failures from previous
   runs. A deploy that succeeded still shows the last three deploys' rollbacks
   in its log, which reads exactly like the current run having failed. That is
   worse than noise: it sends you to debug something already fixed.

So this takes the moment the run started and shows only what failed at or
after it. When a stack has no failures in that window it says so in one line,
which is itself the answer: the deploy was fine and the failure was somewhere
else in the job.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError

FAILED = "FAILED"
MAX_EVENTS = 400


def parse_since(value: str) -> datetime:
    """Accept the ISO-8601 forms GitHub and the AWS CLI produce."""
    text = (value or "").strip()
    if not text:
        raise ValueError("a --since timestamp is required")
    # fromisoformat has handled the "Z" suffix since 3.11, and this runs on
    # 3.12, so both the GitHub form (…18:00:00Z) and the AWS CLI form
    # (…18:00:00+00:00) parse directly. Converting one to the other first was
    # dead code -- and no test could tell the difference, which is how it got
    # noticed.
    parsed = datetime.fromisoformat(text)
    # A naive timestamp compared against an aware one raises; CloudFormation
    # event timestamps are always aware.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def failures_since(events: List[Dict[str, Any]], since: datetime) -> List[Dict[str, Any]]:
    out = []
    for event in events:
        status = str(event.get("ResourceStatus") or "")
        if FAILED not in status:
            continue
        stamp = event.get("Timestamp")
        if isinstance(stamp, datetime) and stamp < since:
            continue
        out.append(event)
    return out


def stack_status(cfn, stack: str) -> str:
    try:
        return cfn.describe_stacks(StackName=stack)["Stacks"][0]["StackStatus"]
    except ClientError:
        return "NOT_FOUND"


def collect(cfn, stack: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    paginator = cfn.get_paginator("describe_stack_events")
    for page in paginator.paginate(StackName=stack):
        events.extend(page.get("StackEvents", []))
        if len(events) >= MAX_EVENTS:
            break
    return events[:MAX_EVENTS]


def report(cfn, stack: str, since: datetime) -> None:
    status = stack_status(cfn, stack)
    print(f"::group::{stack} ({status})")
    if status == "NOT_FOUND":
        print("stack does not exist")
        print("::endgroup::")
        return
    try:
        failures = failures_since(collect(cfn, stack), since)
    except ClientError as exc:
        print(f"could not read events: {exc}")
        print("::endgroup::")
        return

    if not failures:
        # The useful negative. It says the deploy was fine and sends you to
        # look at the rest of the job instead of at three runs of history.
        print(f"No resource failed in this run (nothing at or after {since.isoformat()}).")
    else:
        for event in failures:
            print(f"  {event['Timestamp'].isoformat()}  {event.get('LogicalResourceId')}"
                  f"  {event.get('ResourceType')}  {event.get('ResourceStatus')}")
            reason = event.get("ResourceStatusReason")
            if reason:
                print(f"      {reason}")
    print("::endgroup::")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stacks", nargs="+")
    parser.add_argument("--since", required=True, help="ISO-8601; only failures at or after this")
    parser.add_argument("--region", default=None)
    args = parser.parse_args()

    try:
        since = parse_since(args.since)
    except ValueError as exc:
        print(f"bad --since: {exc}", file=sys.stderr)
        return 2

    cfn = boto3.client("cloudformation", region_name=args.region)
    for stack in args.stacks:
        report(cfn, stack, since)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
