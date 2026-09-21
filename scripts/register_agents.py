"""Register TeamWeave's agents, and keep AgentCore's endpoints for releases.

**This deliberately does not create one endpoint per agent**, which is what it
did first and what Bedrock Agents Classic taught. AgentCore endpoints are a
*release* mechanism, not an identity one: AWS documents them as the way to
keep production on a stable version while a staging endpoint tests a newer one,
and the DEFAULT endpoint always tracks the latest version. The default quota is
ten per runtime, which is a budget for release channels, not for tenants.

Spending that budget on agent identity is wrong in three ways, and the service
said so before the design did:

- It does not scale. Twelve agents already exceeded ten, and raising the quota
  only moves the wall.
- It is self-defeating. The justification for per-agent endpoints was that two
  endpoints on one runtime is how shadow invocation works on AgentCore -- but
  consuming every endpoint for identity is precisely what leaves no endpoint
  for a shadow. The design removed the capability it was argued for.
- Identity does not belong in infrastructure. The OpenTelemetry GenAI semantic
  conventions put it on the span: `invoke_agent` carries `gen_ai.agent.id` and
  `gen_ai.agent.name`, and an orchestrator coordinating several agents reports
  an `invoke_workflow` span around them. That is what TeamWeave's Step
  Functions pipeline is, and mcp_observatory already records every field
  needed -- under bespoke names.

So: every agent runs on the shared runtime and is identified in telemetry;
endpoints are created only for release channels. Per-agent behaviour is
unchanged either way, because prompt_builder composes ROLE, STEP_GOAL and the
output contract and the per-turn instruction travels in the payload.

Registration is idempotent and nothing is ever deleted -- an endpoint from the
earlier per-agent scheme may still be addressed by a run in flight.
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
# must start with a letter, 48 characters. Only release-channel names are
# checked against it now; agent ids never become endpoint names.
ENDPOINT_NAME_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,47}\Z")

RUNTIME_KEYS_OWNED_HERE = ("runtimeArn", "qualifier")

# Release channels, not agents. DEFAULT is created by AgentCore itself and
# always tracks the newest version; `shadow` is the second endpoint the DPO
# flywheel needs to compare two versions of the same agent on live traffic.
# Keeping this list short is the point: it is a release budget of ten.
DEFAULT_CHANNELS = ("shadow",)


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


def runtime_for_team(team: Dict[str, Any], team_runtime_arns: Dict[str, str], fallback: str) -> str:
    """The runtime that serves this team.

    Teams have their own runtimes, so stamping the shared ARN onto every agent
    is not a neutral record -- the worker resolves an agent's own runtimeArn
    *first*, so that value shadows the team's runtime and sends every turn to
    the shared one. That is exactly what happened: the pipeline's failure named
    teamweave_agent, not teamweave_doc_rewrite_team, and the per-team runtimes
    sat unused while looking deployed.
    """
    name = str((team.get("team") or {}).get("name") or "").strip()
    return team_runtime_arns.get(name) or fallback


def agent_census(teams: Dict[str, Dict[str, Any]]) -> Tuple[int, List[str]]:
    """How many agents there are, and anything that makes one unusable.

    An agent is a prompt, not a resource, so there is nothing to provision --
    but an agent with no id is still broken: the worker matches agents to
    workflow steps by id, so one without an id can never be the agent for any
    step and its turn would raise StepFailed at run time.
    """
    count = 0
    problems: List[str] = []
    for key in sorted(teams):
        for agent in teams[key].get("agents") or []:
            if not str(agent.get("id") or agent.get("name") or "").strip():
                problems.append(f"{key}: an agent has neither id nor name")
                continue
            count += 1
    return count, problems


def clear_runtime_identity(team: Dict[str, Any]) -> int:
    """Strip the per-agent runtime pins this script used to write.

    An agent does not get provisioned any more, so it must not carry a
    runtimeArn of its own. `AgentCoreRuntime.resolve_arn` answers
    most-specific-first, and a stamped value is the *most* specific -- it
    shadows the team's runtime entirely. So a pin left behind does not merely
    go unused: it pins the agent to whatever runtime the last deploy wrote,
    and a team whose runtime is later replaced keeps every turn going to the
    old one. Nothing fails; the per-team isolation just stops being real.

    The escape hatch survives on purpose: a runtimeArn a *person* puts in
    team.json still wins. What is removed is this script writing one
    automatically, which made the hatch indistinguishable from the default.
    """
    cleared = 0
    for agent in team.get("agents") or []:
        bedrock = agent.get("bedrock")
        if not isinstance(bedrock, dict):
            continue
        if any(bedrock.get(k) for k in RUNTIME_KEYS_OWNED_HERE):
            for k in RUNTIME_KEYS_OWNED_HERE:
                bedrock.pop(k, None)
            cleared += 1
    return cleared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--runtime-arn", required=True)
    parser.add_argument("--runtime-version", required=True)
    parser.add_argument(
        "--team-runtime-arns",
        default=os.environ.get("AGENTCORE_TEAM_RUNTIME_ARNS", ""),
        help=(
            "JSON object of team name -> runtime ARN, as the stack publishes it. "
            "Each agent records its own team's runtime; --runtime-arn is the "
            "fallback for a team the map does not name."
        ),
    )
    parser.add_argument("--bucket", default=os.environ.get("CONFIG_BUCKET", ""))
    parser.add_argument("--prefix", default=os.environ.get("TEAM_CONFIG_PREFIX", "teams"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument(
        "--channels", default=",".join(DEFAULT_CHANNELS),
        help="Release-channel endpoints to ensure (comma separated). Not agents.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Unparseable content falls back to the shared runtime rather than raising:
    # a malformed map should cost the per-team routing, not the whole deploy.
    team_runtime_arns: Dict[str, str] = {}
    raw_map = (args.team_runtime_arns or "").strip()
    if raw_map:
        try:
            parsed = json.loads(raw_map)
            if isinstance(parsed, dict):
                team_runtime_arns = {
                    str(k): str(v) for k, v in parsed.items() if isinstance(v, str) and v.strip()
                }
            else:
                announce("warning", "--team-runtime-arns is not a JSON object; using the shared runtime")
        except ValueError:
            announce("warning", "--team-runtime-arns is not valid JSON; using the shared runtime")

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
    agent_count, problems = agent_census(teams)
    if problems:
        announce("error", "Unusable agent definitions: " + "; ".join(problems)[:600])
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    if not agent_count:
        announce("error", "Team configs contain no agents.")
        return 1

    print(f"{agent_count} agent(s) across {len(teams)} team(s); "
          f"release channels on runtime {args.runtime_id} v{args.runtime_version}")

    # Release channels, a fixed handful, independent of how many agents there
    # are. This is what the endpoint quota is for.
    channels = [c.strip() for c in (args.channels or "").split(",") if c.strip()]
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=args.region)
        existing = {} if args.dry_run else existing_endpoints(client, args.runtime_id)
        for channel in channels:
            if not ENDPOINT_NAME_PATTERN.match(channel):
                announce("error", f"{channel!r} is not a legal endpoint name")
                return 1
            if args.dry_run:
                print(f"  channel {channel} (dry run)")
                continue
            try:
                action = ensure_endpoint(
                    client, args.runtime_id, channel, args.runtime_version,
                    f"TeamWeave release channel {channel}", existing,
                )
            except ClientError as exc:
                if not is_quota_error(exc):
                    raise
                # Now a genuine signal rather than an expected outcome: the
                # release budget is full, which means endpoints are being used
                # for something other than releases.
                announce(
                    "warning",
                    f"Could not create release channel {channel!r}: the runtime's "
                    f"endpoint quota is full. Endpoints left over from the "
                    f"per-agent scheme are the likely cause; they are safe to "
                    f"delete once no run addresses them. ({exc})",
                )
                continue
            print(f"  channel {channel}: {action}")
    except (ClientError, BotoCoreError) as exc:
        announce("error", f"AgentCore registry call failed: {exc}")
        return 1

    cleared_total = 0
    # Where each team's turns will go, resolved exactly as the worker resolves
    # them, so the summary reports the runtime actually in use rather than the
    # shared one this script was handed.
    placements: Dict[str, str] = {}
    shared_fallbacks = []
    for key, team in teams.items():
        team_name = str((team.get("team") or {}).get("name") or key)
        runtime_arn = runtime_for_team(team, team_runtime_arns, args.runtime_arn)
        placements[team_name] = runtime_arn.rsplit("/", 1)[-1] or runtime_arn
        if runtime_arn == args.runtime_arn and team_name not in team_runtime_arns:
            shared_fallbacks.append(team_name)

        cleared = clear_runtime_identity(team)
        if not cleared:
            continue
        cleared_total += cleared
        if not args.dry_run:
            s3.put_object(
                Bucket=args.bucket, Key=key,
                Body=json.dumps(team, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
        print(f"  {key}: cleared {cleared} stale per-agent runtime pin(s)")

    if shared_fallbacks:
        announce(
            "warning",
            "No runtime of their own, so these teams fall back to the shared one: "
            f"{', '.join(sorted(shared_fallbacks))}. They run, but they share a blast "
            "radius and a release channel budget with every other team there.",
        )

    announce(
        "notice",
        f"{agent_count} agent(s) across {len(placements)} team(s): "
        + "; ".join(f"{t} -> {r}" for t, r in sorted(placements.items()))
        + f". Cleared {cleared_total} stale per-agent runtime pin(s); release "
        f"channels: {', '.join(channels) or 'DEFAULT only'}. An agent is a prompt, "
        "not a resource: nothing is provisioned per agent, and identity is carried "
        "on spans (gen_ai.agent.id/name).",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
