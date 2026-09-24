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

A changeset that fails early validation never writes a stack resource event.
``describe_stack_events`` then reports that nothing failed, and the only text
in the SAM log is the hook name. ``DescribeEvents`` is the call that carries
``ValidationStatusReason`` and ``ValidationPath`` for that hook. A failed
change set is listed too, because those events hang off the change set rather
than the stack.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError

FAILED = "FAILED"
MAX_EVENTS = 400


# How far back to look when the caller could not say. Long enough to cover a
# slow deploy, short enough not to drag in the previous one.
FALLBACK_WINDOW = timedelta(hours=2)


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


def at_or_after(stamp: Any, since: datetime) -> bool:
    # A missing timestamp is kept. Dropping a failure is worse than showing
    # one extra, and a comparison against a non-datetime would raise.
    if isinstance(stamp, datetime) and stamp < since:
        return False
    return True


def failures_since(events: List[Dict[str, Any]], since: datetime) -> List[Dict[str, Any]]:
    out = []
    for event in events:
        status = str(event.get("ResourceStatus") or "")
        if FAILED not in status:
            continue
        if not at_or_after(event.get("Timestamp"), since):
            continue
        out.append(event)
    return out


def is_early_validation(event: Dict[str, Any]) -> bool:
    """A hook failure, not a resource CREATE_FAILED.

    DescribeEvents with FailedEvents also returns ordinary provisioning
    failures. Those are already on the stack event stream. The ones that
    are not are validation and hook results, and they are the reason a
    changeset can fail with no stack events at all.
    """
    if event.get("EventType") == "VALIDATION_ERROR":
        return True
    if event.get("ValidationName") or event.get("ValidationStatusReason") or event.get("ValidationPath"):
        return True
    if event.get("HookType") or event.get("HookStatusReason"):
        return True
    return False


def validation_events_since(events: List[Dict[str, Any]], since: datetime) -> List[Dict[str, Any]]:
    out = []
    for event in events:
        if not is_early_validation(event):
            continue
        stamp = event.get("Timestamp") or event.get("StartTime")
        if not at_or_after(stamp, since):
            continue
        out.append(event)
    return out


def failed_change_sets_since(summaries: List[Dict[str, Any]], since: datetime) -> List[Dict[str, Any]]:
    out = []
    for summary in summaries:
        if summary.get("Status") != FAILED:
            continue
        if not at_or_after(summary.get("CreationTime"), since):
            continue
        out.append(summary)
    return out


def format_validation_event(event: Dict[str, Any]) -> List[str]:
    logical = event.get("LogicalResourceId") or "(no logical id)"
    resource_type = event.get("ResourceType") or ""
    name = event.get("ValidationName") or event.get("HookType") or event.get("EventType") or ""
    reason = (
        event.get("ValidationStatusReason")
        or event.get("HookStatusReason")
        or event.get("ResourceStatusReason")
        or ""
    )
    path = event.get("ValidationPath") or ""
    physical = event.get("PhysicalResourceId") or ""
    lines = [f"  {logical}  {resource_type}  {name}".rstrip()]
    if physical:
        lines.append(f"      identifier: {physical}")
    if reason:
        lines.append(f"      {reason}")
    if path:
        lines.append(f"      path: {path}")
    return lines


def _error_text(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        err = response.get("Error") or {}
        code = err.get("Code") or ""
        message = err.get("Message") or str(exc)
        return f"{code}: {message}" if code else message
    return str(exc)


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


def _paginate(cfn, operation: str, **kwargs) -> List[Dict[str, Any]]:
    pages = []
    for page in cfn.get_paginator(operation).paginate(**kwargs):
        pages.append(page)
        if len(pages) >= 20:
            break
    return pages


def early_findings(cfn, stack: str, since: datetime, region: str | None = None) -> List[str]:
    """Validation and hook failures for this run.

    Never raises. The step that explains a failure must not become a second
    failure, including when the deployer role cannot call DescribeEvents.
    """
    try:
        return _early_findings(cfn, stack, since, region)
    except Exception as exc:
        where = f" --region {region}" if region else ""
        return [
            f"  could not read early-validation events: {_error_text(exc)}",
            "  A failed changeset has no stack resource events. DescribeEvents "
            "is what names the resource.",
            "  aws cloudformation describe-events "
            f"--stack-name {stack} --filters FailedEvents=true{where}",
        ]


def _early_findings(cfn, stack: str, since: datetime, region: str | None) -> List[str]:
    lines: List[str] = []
    summaries: List[Dict[str, Any]] = []
    try:
        for page in _paginate(cfn, "list_change_sets", StackName=stack):
            summaries.extend(page.get("Summaries", []))
    except ClientError as exc:
        lines.append(f"  could not list change sets: {_error_text(exc)}")

    recent = failed_change_sets_since(summaries, since)[:5]
    for summary in recent:
        name = summary.get("ChangeSetName") or summary.get("ChangeSetId") or "(unnamed)"
        lines.append(f"  change set {name}  FAILED")
        reason = summary.get("StatusReason")
        if reason:
            lines.append(f"      {reason}")

    events: List[Dict[str, Any]] = []
    describe_failed = False
    try:
        for page in _paginate(
            cfn, "describe_events", StackName=stack, Filters={"FailedEvents": True}
        ):
            events.extend(page.get("OperationEvents", []))
    except (ClientError, AttributeError) as exc:
        describe_failed = True
        where = f" --region {region}" if region else ""
        lines.append(f"  could not read DescribeEvents for {stack}: {_error_text(exc)}")
        lines.append(
            "  A failed changeset has no stack resource events. DescribeEvents "
            "is what names the resource."
        )
        lines.append(
            "  aws cloudformation describe-events "
            f"--stack-name {stack} --filters FailedEvents=true{where}"
        )

    if not describe_failed:
        for summary in recent:
            change_set = summary.get("ChangeSetId") or summary.get("ChangeSetName")
            if not change_set:
                continue
            try:
                for page in _paginate(
                    cfn,
                    "describe_events",
                    StackName=stack,
                    ChangeSetName=change_set,
                    Filters={"FailedEvents": True},
                ):
                    events.extend(page.get("OperationEvents", []))
            except (ClientError, AttributeError) as exc:
                lines.append(
                    f"  could not read DescribeEvents for change set {change_set}: {_error_text(exc)}"
                )

    seen = set()
    for event in validation_events_since(events, since):
        event_id = event.get("EventId")
        if event_id:
            if event_id in seen:
                continue
            seen.add(event_id)
        lines.extend(format_validation_event(event))
    return lines


def report(cfn, stack: str, since: datetime, region: str | None = None) -> None:
    status = stack_status(cfn, stack)
    print(f"::group::{stack} ({status})")
    if status == "NOT_FOUND":
        print("stack does not exist")
        print("::endgroup::")
        return
    try:
        failures = failures_since(collect(cfn, stack), since)
        stack_error = None
    except ClientError as exc:
        # Still read the change set. Early validation fails before a stack
        # resource event exists, and this is the only place that names it.
        failures = []
        stack_error = exc

    early = early_findings(cfn, stack, since, region)
    if stack_error and not failures and not early:
        print(f"could not read events: {stack_error}")
        print("::endgroup::")
        return
    if stack_error:
        print(f"could not read stack events: {stack_error}")
    if not failures and not early:
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
        for line in early:
            print(line)
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
        # Never exit non-zero from the step whose only job is to explain a
        # failure. `${{ github.run_started_at }}` expanded to nothing and this
        # returned 2, so the one run that most needed a diagnosis got a second
        # red step and no diagnosis at all -- reporting became the failure.
        since = datetime.now(timezone.utc) - FALLBACK_WINDOW
        print(
            f"::warning::--since was unusable ({exc}); showing failures from the "
            f"last {FALLBACK_WINDOW}, which may include an earlier run",
            flush=True,
        )

    cfn = boto3.client("cloudformation", region_name=args.region)
    for stack in args.stacks:
        report(cfn, stack, since, region=args.region)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
