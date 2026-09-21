#!/usr/bin/env python3
"""Push team definitions to S3 without destroying what provisioning wrote.

`aws s3 sync config/examples/teams/ s3://.../teams/` was wiping the platform's
own state on every deploy. Provisioning writes the Bedrock agentId/aliasId it
creates back into the *same* S3 key the sync overwrites, and a fresh CI
checkout gives every file a current mtime, so the sync always won. Every
deploy therefore handed `needs_provisioning` twelve agents with empty ids and
forced a full rebuild -- which is why the step grew past the CLI timeout, and
why each deploy orphaned the agents the previous one had created.

`needs_provisioning` exists precisely to skip agents that already have ids, so
clobbering them first was clearly never the intent.

This script keeps the repository authoritative for *definitions* -- prompts,
workflow, schemas, roles -- while treating the runtime identifiers as state
that belongs to S3. Anything the repository does not own is carried across.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

# Per agent, the keys provisioning owns. Everything else in `bedrock` (model
# ids, aliases) is a definition the repository sets.
#
# agentId/aliasId are Bedrock Agents Classic, which still provisions one
# Bedrock agent per TeamWeave agent, so they remain S3-owned state.
#
# runtimeArn/qualifier are deliberately NOT here. Nothing provisions an agent
# on AgentCore any more -- an agent is a prompt, and its runtime comes from
# its team via AGENTCORE_TEAM_RUNTIME_ARNS. Carrying a stamped value across
# would defeat `register_agents.clear_runtime_identity` completely: it strips
# the pins from S3, and the very next deploy's merge would read them back out
# of the previous copy and write them again. The repository would then own a
# field it never sets and cannot see.
#
# A runtimeArn a person writes in team.json still wins at resolution time and
# still survives this merge -- as a *definition* from the repository, which is
# what it now is.
RUNTIME_AGENT_KEYS = ("agentId", "aliasId")
# Model-alias maps are filled in by provisioning too: the keys are declared in
# the repo, the values are alias ids it creates.
RUNTIME_ALIAS_MAP = "model_aliases"


def agent_key(agent: Dict[str, Any]) -> str:
    """Match agents across the two copies by id, falling back to name."""
    return str(agent.get("id") or agent.get("name") or "")


def merge_agent(local: Dict[str, Any], remote: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(local)
    bedrock = dict(local.get("bedrock") or {})
    remote_bedrock = remote.get("bedrock") or {}

    for key in RUNTIME_AGENT_KEYS:
        value = str(remote_bedrock.get(key) or "").strip()
        if value:
            bedrock[key] = value

    # Keep an alias id only for a model the repo still declares; a model
    # removed from the definition should not keep a stale alias alive.
    local_aliases = dict(bedrock.get(RUNTIME_ALIAS_MAP) or {})
    remote_aliases = remote_bedrock.get(RUNTIME_ALIAS_MAP) or {}
    for model_id in list(local_aliases):
        value = str(remote_aliases.get(model_id) or "").strip()
        if value:
            local_aliases[model_id] = value
    if local_aliases:
        bedrock[RUNTIME_ALIAS_MAP] = local_aliases

    merged["bedrock"] = bedrock
    return merged


def merge_team(local: Dict[str, Any], remote: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(local)

    # team_id is generated at provisioning time and referenced elsewhere;
    # regenerating it on every deploy would repoint those references.
    team = dict(local.get("team") or {})
    remote_team = remote.get("team") or {}
    for key in ("team_id",):
        value = remote_team.get(key)
        if value and not str(team.get(key) or "").strip():
            team[key] = value
    if team:
        merged["team"] = team

    remote_agents = {agent_key(a): a for a in (remote.get("agents") or [])}
    merged["agents"] = [
        merge_agent(a, remote_agents.get(agent_key(a), {})) for a in (local.get("agents") or [])
    ]
    return merged


def load_remote(s3, bucket: str, key: str) -> Dict[str, Any]:
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
            return {}
        raise
    except ValueError:
        # Unparseable remote: treat as absent rather than aborting the deploy.
        print(f"  warning: s3://{bucket}/{key} is not valid JSON; replacing it", file=sys.stderr)
        return {}


def teams_in_s3(s3, bucket: str, prefix: str) -> Dict[str, list]:
    """team name -> its keys, from what the bucket actually holds."""
    root = f"{prefix}/" if prefix else ""
    found: Dict[str, list] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=root):
        for obj in page.get("Contents") or []:
            rel = obj["Key"][len(root):]
            parts = rel.split("/")
            if len(parts) >= 2 and parts[0]:
                found.setdefault(parts[0], []).append(obj["Key"])
    return found


def prune(s3, bucket: str, prefix: str, keep: set, dry_run: bool = False) -> int:
    """Delete teams S3 still serves that the repository no longer defines.

    Without this, deleting a team from the repository removes it from nobody's
    view: GET /teams still lists it, the UI still offers it, and running it
    starts a pipeline whose definition no longer exists here.

    This is narrower than `aws s3 sync --delete`, which is banned for good
    reason -- that overwrites *keys within* a team and erased the provisioned
    ids written back by provisioning. This only removes a team directory in
    its entirety, and only when the repository has no such team at all. The
    repository owns which teams exist; S3 owns the runtime identifiers inside
    them.
    """
    removed = 0
    for team, keys in sorted(teams_in_s3(s3, bucket, prefix).items()):
        if team in keep:
            continue
        print(f"  pruning team no longer in the repository: {team} ({len(keys)} key(s))")
        for key in keys:
            print(f"    delete s3://{bucket}/{key}")
            if not dry_run:
                s3.delete_object(Bucket=bucket, Key=key)
        removed += 1
    return removed


def sync(root: Path, bucket: str, prefix: str, region: str, dry_run: bool = False,
         do_prune: bool = True) -> int:
    s3 = boto3.client("s3", region_name=region)
    prefix = prefix.strip("/")
    count = 0

    for path in sorted(root.rglob("team.json")):
        rel = path.relative_to(root).as_posix()
        key = f"{prefix}/{rel}" if prefix else rel
        local = json.loads(path.read_text())
        remote = load_remote(s3, bucket, key)
        merged = merge_team(local, remote)

        kept = sum(
            1
            for a in merged.get("agents", [])
            if str((a.get("bedrock") or {}).get("agentId") or "").strip()
        )
        total = len(merged.get("agents", []))
        print(f"  {key}: {kept}/{total} agents keep their provisioned ids")

        if not dry_run:
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(merged, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
        count += 1

    if do_prune:
        local_teams = {p.relative_to(root).parts[0] for p in root.rglob("team.json")}
        # Never prune against an empty set: an unreadable root would otherwise
        # delete every team in the bucket.
        if local_teams:
            prune(s3, bucket, prefix, local_teams, dry_run)
        else:
            print("  refusing to prune: found no local teams to compare against", file=sys.stderr)

    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="config/examples/teams")
    parser.add_argument("--bucket", default=os.environ.get("CONFIG_BUCKET", ""))
    parser.add_argument("--prefix", default=os.environ.get("TEAM_CONFIG_PREFIX", "teams"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help=(
            "Leave teams in S3 that the repository no longer defines. They stay "
            "listed by GET /teams and runnable from the UI."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.bucket:
        print("--bucket (or CONFIG_BUCKET) is required", file=sys.stderr)
        return 2

    root = Path(args.root)
    if not root.is_dir():
        print(f"{root} is not a directory", file=sys.stderr)
        return 2

    print(f"Merging {root}/ -> s3://{args.bucket}/{args.prefix}/")
    n = sync(root, args.bucket, args.prefix, args.region, args.dry_run,
             do_prune=not args.no_prune)
    print(f"Synced {n} team config(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
