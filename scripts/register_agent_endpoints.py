#!/usr/bin/env python3
"""Give every TeamWeave agent its own identity in the AgentCore registry.

AgentCore has no separate "agent" resource. Its registry is a runtime plus
named **endpoints** on that runtime, and InvokeAgentRuntime's `qualifier` is
an endpoint name: "an endpoint name that points to a specific version". So an
agent's registry identity is an endpoint, and its `qualifier` in team.json is
what addresses it.

One runtime, one endpoint per agent -- not a runtime per agent. A runtime is a
whole code artifact: twelve of them would mean twelve builds, twelve uploads
and twelve startup validations per deploy, which is precisely the shape of the
Classic provisioning step that grew past the CLI timeout and orphaned agents.
An endpoint is a name and a version pin, so the platform gets per-agent
identity, per-agent telemetry and per-agent version pinning at a fraction of
the cost. Per-agent behaviour is unaffected either way: prompt_builder already
composes ROLE, STEP_GOAL and the output contract, and the per-turn instruction
travels in the payload.

What this buys beyond tidiness: two endpoints on one runtime is also the
AgentCore shape of shadow invocation, which the DPO flywheel needs and which
AgentCoreRuntime currently warns is unimplemented.

Registration is idempotent. Existing endpoints are re-pointed at the current
runtime version rather than recreated, and nothing is ever deleted: an agent
removed from a team config may still be addressed by a run in flight.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# EndpointName: [a-zA-Z][a-zA-Z0-9_]{0,47} -- letters, digits, underscores,
# must start with a letter, 48 characters. The same alphabet that made
# AgentRuntimeName fail to create when it was built from a hyphenated stack
# name; agent ids are hyphenated far more often than stack names are.
ENDPOINT_NAME_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,47}\Z")
ENDPOINT_NAME_MAX = 48

RUNTIME_KEYS_OWNED_HERE = ("runtimeArn", "qualifier")


# AgentCore caps endpoints per runtime. The cap is an account quota, so it is
# not something a deploy can route around -- but it must be told apart from a
# real error, because hitting it leaves a working platform and a permissions
# failure does not.
QUOTA_ERROR_CODES = {"ServiceQuotaExceededException", "LimitExceededException"}


def account_of(runtime_arn: str) -> str:
    """The account id out of an ARN, for a quota request that names it."""
    parts = str(runtime_arn or "").split(":")
    return parts[4] if len(parts) > 4 and parts[4].isdigit() else ""


def is_quota_error(exc: ClientError) -> bool:
    code = ((exc.response or {}).get("Error") or {}).get("Code", "")
    return code in QUOTA_ERROR_CODES


def announce(level: str, message: str) -> None:
    """Report where GitHub will surface it.

    Workflow commands are parsed from stdout, so a reason written to stderr is
    invisible on the run page and readable only by paging the job log -- which
    is exactly how this step's first real failure cost a whole cycle to
    diagnose.
    """
    print(f"::{level}::{message}", flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as handle:
                handle.write(f"**Agent registry — {level}:** {message}\n\n")
        except OSError:
            pass


def endpoint_name(agent_id: str) -> str:
    """The registry name for an agent id.

    Deterministic, because the name *is* the address: a different name on the
    next deploy would orphan the endpoint and silently repoint the agent.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", agent_id or "").strip("_")
    if not cleaned:
        raise ValueError(f"agent id {agent_id!r} has no characters an endpoint name can use")
    if not cleaned[0].isalpha():
        # Must start with a letter. A prefix keeps it derived rather than
        # invented, so the same id always lands on the same endpoint.
        cleaned = "a_" + cleaned
    return cleaned[:ENDPOINT_NAME_MAX]


def plan_endpoints(teams: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, str], List[str]]:
    """Map every agent id to its endpoint name, and report what cannot be done.

    Collisions are a hard error, never a silent alias. Two agents sharing an
    endpoint share a registry identity, so their telemetry merges and a version
    pin meant for one moves the other -- a failure that looks like nothing at
    all until someone reads a dashboard.
    """
    names: Dict[str, str] = {}
    taken: Dict[str, str] = {}
    problems: List[str] = []

    for key in sorted(teams):
        for agent in teams[key].get("agents") or []:
            agent_id = str(agent.get("id") or agent.get("name") or "").strip()
            if not agent_id:
                problems.append(f"{key}: an agent has neither id nor name")
                continue
            if agent_id in names:
                continue
            try:
                name = endpoint_name(agent_id)
            except ValueError as exc:
                problems.append(f"{key}: {exc}")
                continue
            if not ENDPOINT_NAME_PATTERN.match(name):
                problems.append(f"{key}: {agent_id!r} -> {name!r} is not a legal endpoint name")
                continue
            if name in taken and taken[name] != agent_id:
                problems.append(
                    f"{key}: {agent_id!r} and {taken[name]!r} both become endpoint {name!r}; "
                    "they would share one registry identity"
                )
                continue
            taken[name] = agent_id
            names[agent_id] = name

    return names, problems


# ── AWS ────────────────────────────────────────────────────────────────────

def existing_endpoints(client, runtime_id: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    token = None
    while True:
        kwargs = {"agentRuntimeId": runtime_id}
        if token:
            kwargs["nextToken"] = token
        page = client.list_agent_runtime_endpoints(**kwargs)
        for item in page.get("runtimeEndpoints") or []:
            name = item.get("name") or item.get("endpointName")
            if name:
                out[name] = item
        token = page.get("nextToken")
        if not token:
            break
    return out


def ensure_endpoint(client, runtime_id: str, name: str, version: str, description: str,
                    existing: Dict[str, Dict[str, Any]]) -> str:
    """Create or re-point one endpoint. Returns what happened."""
    current = existing.get(name)
    if current is None:
        client.create_agent_runtime_endpoint(
            agentRuntimeId=runtime_id,
            name=name,
            agentRuntimeVersion=version,
            description=description[:200],
        )
        return "created"

    # An endpoint left on an older version keeps serving an older artifact
    # after the runtime is updated, so the registry silently drifts from the
    # code. Re-point it rather than leaving it.
    #
    # targetVersion is the version the endpoint is moving to, liveVersion the
    # one currently serving; a list item carries both and no
    # `agentRuntimeVersion`. Comparing against a field that does not exist
    # would read as "" every time and re-point every endpoint on every deploy.
    on_version = str(current.get("targetVersion") or current.get("liveVersion") or "")
    if on_version == str(version):
        return "current"
    client.update_agent_runtime_endpoint(
        agentRuntimeId=runtime_id,
        endpointName=name,
        agentRuntimeVersion=version,
    )
    return f"repointed {on_version or '?'} -> {version}"


# ── S3 ─────────────────────────────────────────────────────────────────────

def team_keys(s3, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": f"{prefix.strip('/')}/"}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        keys.extend(
            o["Key"] for o in page.get("Contents", []) if o["Key"].endswith("/team.json")
        )
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return sorted(keys)


def load_teams(s3, bucket: str, keys: List[str]) -> Dict[str, Dict[str, Any]]:
    teams = {}
    for key in keys:
        try:
            teams[key] = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        except (ClientError, ValueError) as exc:
            print(f"  warning: skipping s3://{bucket}/{key}: {exc}", file=sys.stderr)
    return teams


def write_back(team: Dict[str, Any], names: Dict[str, str], runtime_arn: str) -> int:
    """Record each agent's registry identity in its config. Returns changes."""
    changed = 0
    for agent in team.get("agents") or []:
        agent_id = str(agent.get("id") or agent.get("name") or "").strip()
        name = names.get(agent_id)
        if not name:
            continue
        bedrock = dict(agent.get("bedrock") or {})
        wanted = {"runtimeArn": runtime_arn, "qualifier": name}
        if all(str(bedrock.get(k) or "") == v for k, v in wanted.items()):
            continue
        bedrock.update(wanted)
        agent["bedrock"] = bedrock
        changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--runtime-arn", required=True)
    parser.add_argument("--runtime-version", required=True)
    parser.add_argument("--bucket", default=os.environ.get("CONFIG_BUCKET", ""))
    parser.add_argument("--prefix", default=os.environ.get("TEAM_CONFIG_PREFIX", "teams"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.bucket:
        print("--bucket (or CONFIG_BUCKET) is required", file=sys.stderr)
        return 2

    s3 = boto3.client("s3", region_name=args.region)
    keys = team_keys(s3, args.bucket, args.prefix)
    if not keys:
        # Registering nothing is not success. A wrong prefix looks exactly
        # like a platform with no teams, and would pass quietly.
        announce("error", f"No team.json found under s3://{args.bucket}/{args.prefix}/")
        return 1

    teams = load_teams(s3, args.bucket, keys)
    names, problems = plan_endpoints(teams)
    if problems:
        announce("error", "Cannot register agents: " + "; ".join(problems)[:600])
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    if not names:
        announce("error", "Team configs contain no agents to register.")
        return 1

    print(f"Registering {len(names)} agent(s) on runtime {args.runtime_id} v{args.runtime_version}")

    registered: Dict[str, str] = {}
    unregistered: List[str] = []
    quota_message = ""
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=args.region)
        existing = {} if args.dry_run else existing_endpoints(client, args.runtime_id)

        for agent_id in sorted(names):
            name = names[agent_id]
            if args.dry_run:
                print(f"  {agent_id} -> {name} (dry run)")
                registered[agent_id] = name
                continue
            if quota_message:
                # The quota is per runtime, so once it is reached every
                # remaining create fails the same way. Asking eleven more
                # times would only be slower.
                unregistered.append(agent_id)
                continue
            try:
                action = ensure_endpoint(
                    client, args.runtime_id, name, args.runtime_version,
                    f"TeamWeave agent {agent_id}", existing,
                )
            except ClientError as exc:
                if not is_quota_error(exc):
                    raise
                quota_message = str(exc)
                unregistered.append(agent_id)
                continue
            registered[agent_id] = name
            print(f"  {agent_id} -> {name}: {action}")
    except (ClientError, BotoCoreError) as exc:
        announce("error", f"AgentCore registry call failed: {exc}")
        return 1

    written = 0
    for key, team in teams.items():
        changed = write_back(team, registered, args.runtime_arn)
        if not changed:
            continue
        written += changed
        if not args.dry_run:
            s3.put_object(
                Bucket=args.bucket, Key=key,
                Body=json.dumps(team, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
        print(f"  {key}: recorded registry identity for {changed} agent(s)")

    if unregistered:
        # Not an error, and deliberately not a failed deploy. An agent with no
        # qualifier falls back to the runtime's default endpoint, which is
        # exactly how every agent ran before this step existed -- so the
        # platform is degraded, not broken, and failing every deploy over a
        # fixed account quota would help nobody. It must still be impossible
        # to miss.
        announce(
            "warning",
            f"Registered {len(registered)} of {len(names)} agent(s). "
            f"AgentCore's per-runtime endpoint quota is reached, so these "
            f"{len(unregistered)} fall back to the runtime's default endpoint: "
            f"{', '.join(sorted(unregistered))}. "
            f"Remedy: raise the maxEndpointsPerAgent quota for account "
            f"{account_of(args.runtime_arn) or 'this account'} in {args.region} to at "
            f"least {len(names)} via AWS Support. Until it is raised this warning "
            f"repeats every deploy and the platform runs degraded, not broken. "
            f"({quota_message[:160]})",
        )
        return 0

    announce(
        "notice",
        f"Registered {len(registered)} agent(s) on runtime {args.runtime_id} "
        f"v{args.runtime_version}; updated {written} agent record(s).",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
