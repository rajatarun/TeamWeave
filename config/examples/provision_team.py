#!/usr/bin/env python3
"""
provision_team.py
-----------------
Fully autonomous — no interactive prompts. Designed to run as a Git
workflow step (GitHub Actions, GitLab CI, etc.).

Behaviour:
  - Scans --scan-dir for ALL files matching team*.json
  - Automatically selects only files where at least one agent is missing
    agentId or aliasId (i.e. needs provisioning)
  - Files where all agents already have Bedrock IDs are skipped silently
  - S3 key is derived from the file's own team.name + team.version:
        {prefix}/{team.name}/{team.version}/team.json
  - Exits non-zero on any validation or provisioning failure

Pipeline per team file that needs provisioning:
  1. Load     — team*.json + roles.json + departments.json
  2. Validate — role/dept membership, schema allowlist, id format
  3. Enrich   — prepend dept.description + role.primary_task to goal_template
  4. Provision — create Bedrock agents for agents missing IDs only (idempotent)
  5. Push     — s3://{bucket}/{prefix}/{team.name}/{team.version}/team.json

Usage:
    python provision_team.py \
        --roles  config/roles.json \
        --depts  config/departments.json \
        --bucket my-s3-bucket

    # Scan a specific directory
    python provision_team.py \
        --roles     config/roles.json \
        --depts     config/departments.json \
        --bucket    my-s3-bucket \
        --scan-dir  ./teams

    # Dry run — validate + enrich only, no AWS calls
    python provision_team.py ... --dry-run

Environment variables (override CLI flags):
    ARTIFACT_BUCKET   S3 bucket name
    AWS_REGION        AWS region
    BEDROCK_ROLE_ARN  IAM role ARN Bedrock agents will assume (required for creation)

GitHub Actions example:
    - name: Provision teams
      run: |
        python provision_team.py \
          --roles  config/roles.json \
          --depts  config/departments.json \
          --bucket ${{ secrets.ARTIFACT_BUCKET }} \
          --region us-east-1 \
          --bedrock-role-arn ${{ secrets.BEDROCK_ROLE_ARN }}
      env:
        AWS_ACCESS_KEY_ID:     ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_role_index(roles_data: dict) -> dict:
    """Return {role_id: role_obj}"""
    return {r["role_id"]: r for r in roles_data["roles"]}


def build_dept_index(depts_data: dict) -> dict:
    """Return {department_id: dept_obj}"""
    return {d["department_id"]: d for d in depts_data["departments"]}


# ---------------------------------------------------------------------------
# Team file scanner
# ---------------------------------------------------------------------------

def needs_provisioning(data: dict) -> tuple[bool, int, int]:
    """
    Inspect a loaded team dict and determine if any agent is missing
    agentId or aliasId.

    Returns:
        (needs_work, total_agents, agents_missing_ids)
    """
    agents  = data.get("agents", [])
    total   = len(agents)
    missing = sum(
        1 for a in agents
        if not (
            a.get("bedrock", {}).get("agentId",  "").strip() and
            a.get("bedrock", {}).get("aliasId", "").strip()
        )
    )
    return missing > 0, total, missing


def generate_team_id(team_name: str) -> str:
    """
    Derive a short team_id from team_name by taking the first letter of
    each word (split on underscores or spaces), uppercased.

    Examples:
        tarun_visibility_team  → TVT
        pr_only_team           → POT
        content_growth         → CG
    """
    words = re.split(r'[_\s]+', team_name.strip())
    return "".join(w[0].upper() for w in words if w)


def resolve_team_id(team_block: dict) -> str:
    """
    Return the team_id from the team block.
    If team_id is explicitly set in the file, use it.
    Otherwise generate it from team.name and return it.
    """
    explicit = team_block.get("team_id", "").strip()
    if explicit:
        return explicit
    return generate_team_id(team_block.get("name", ""))


def scan_team_files(scan_dir: str) -> list[dict]:
    """
    Scan scan_dir for all files matching team*.json.

    Automatically filters to only files where at least one agent is missing
    agentId or aliasId. Files that are fully provisioned are logged and skipped.

    Returns a list of dicts for files that need work:
        { "path": Path, "name": str, "version": str, "owner": str,
          "total_agents": int, "agents_needing_ids": int }

    Exits non-zero if no team*.json files exist at all.
    If all files are already fully provisioned, logs a message and exits 0.
    """
    scan_path  = Path(scan_dir).resolve()
    candidates = sorted(scan_path.glob("team*.json"))

    if not candidates:
        log.error(f"No team*.json files found in: {scan_path}")
        sys.exit(1)

    needs_work = []
    skipped    = []

    for path in candidates:
        try:
            data       = load_json(str(path))
            team_block = data.get("team", {})
            name       = team_block.get("name",    "").strip()
            version    = team_block.get("version", "").strip()
            owner      = team_block.get("owner",   "unknown")
            team_id    = resolve_team_id(team_block)

            if not name:
                log.warning(f"  SKIP {path.name} — missing team.name")
                continue
            if not version:
                log.warning(f"  SKIP {path.name} — missing team.version")
                continue

            work_needed, total, missing = needs_provisioning(data)

            if not work_needed:
                skipped.append(path.name)
                log.info(
                    f"  SKIP {path.name:<35} "
                    f"all {total} agent(s) already provisioned."
                )
                continue

            needs_work.append({
                "path":               path,
                "name":               name,
                "team_id":            team_id,
                "version":            version,
                "owner":              owner,
                "total_agents":       total,
                "agents_needing_ids": missing,
            })
            log.info(
                f"  QUEUE {path.name:<34} "
                f"name={name}  team_id={team_id}  version={version}  "
                f"needs_ids={missing}/{total}"
            )

        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"  SKIP {path.name} — could not parse: {e}")

    if not needs_work:
        log.info("All team files are fully provisioned. Nothing to do.")
        sys.exit(0)

    return needs_work


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_team(team: dict, role_index: dict, dept_index: dict) -> list[str]:
    """
    Returns list of validation error strings.
    An empty list means the team is valid.
    """
    errors = []
    for agent in team.get("agents", []):
        agent_id = agent.get("id", "<missing id>")
        role_id = agent.get("role_id")
        dept_id = agent.get("department_id")

        # role_id must exist
        if role_id not in role_index:
            errors.append(f"[{agent_id}] role_id '{role_id}' not found in roles.json")
            continue

        # department_id must exist
        if dept_id not in dept_index:
            errors.append(f"[{agent_id}] department_id '{dept_id}' not found in departments.json")
            continue

        # role must be allowed in that department
        dept = dept_index[dept_id]
        if role_id not in dept["allowed_roles"]:
            errors.append(
                f"[{agent_id}] role_id '{role_id}' is NOT allowed in department '{dept['name']}'. "
                f"Allowed: {dept['allowed_roles']}"
            )

        # schema_ref must be in department's allowed_schemas
        schema_ref = agent.get("schema_ref")
        if schema_ref and schema_ref not in dept.get("allowed_schemas", []):
            errors.append(
                f"[{agent_id}] schema_ref '{schema_ref}' is NOT in department '{dept['name']}' "
                f"allowed_schemas: {dept.get('allowed_schemas')}"
            )

        # agent id format: [TEAM-ID_]DEPT-XXX_PBM-XXX_<slug>
        # Strip optional team_id prefix before checking DEPT/ROLE segments
        id_parts = agent_id.split("_")
        # Find where DEPT- segment starts (skip team_id prefix if present)
        dept_idx = next((i for i, p in enumerate(id_parts) if p.startswith("DEPT-")), None)
        if dept_idx is None or len(id_parts) < dept_idx + 3:
            errors.append(
                f"[{agent_id}] agent id must contain DEPT-ID_ROLE-ID_slug segments "
                f"(optionally prefixed with TEAM-ID_)"
            )
        else:
            if id_parts[dept_idx] != dept_id:
                errors.append(
                    f"[{agent_id}] DEPT segment '{id_parts[dept_idx]}' does not match department_id '{dept_id}'"
                )
            if id_parts[dept_idx + 1] != role_id:
                errors.append(
                    f"[{agent_id}] ROLE segment '{id_parts[dept_idx + 1]}' does not match role_id '{role_id}'"
                )

    return errors


# ---------------------------------------------------------------------------
# Goal template enrichment
# ---------------------------------------------------------------------------

def build_goal_template(
    original_goal_template: str,
    dept: dict,
    role: dict,
) -> str:
    """
    Compose a fully enriched goal_template for one agent by combining:

      1. DEPARTMENT INTENT BOUNDARY  — dept["description"]
         Injected first so the agent knows its lane before reading its task.

      2. ROLE PRIMARY TASK           — role["agent_config"]["primary_task"]
         The single agentic task this role performs: action, description,
         input contract, output contract, and constraints.

      3. ORIGINAL GOAL TEMPLATE      — the team-level goal_template string
         Kept last so team-specific instructions layer on top of role defaults.

    Returns a single formatted string ready to be used as the Bedrock agent
    instruction or stored back into team.json["agents"][n]["goal_template"].
    """
    primary_task: dict = role["agent_config"]["primary_task"]
    constraints: list  = primary_task.get("constraints", [])
    constraints_block  = "\n".join(f"  - {c}" for c in constraints)

    enriched = (
        # ── 1. Department intent boundary ────────────────────────────────
        "## DEPARTMENT INTENT BOUNDARY\n"
        f"{dept['description']}\n\n"

        # ── 2. Role primary task ─────────────────────────────────────────
        "## ROLE PRIMARY TASK\n"
        f"Action  : {primary_task['action']}\n"
        f"Task    : {primary_task['description']}\n"
        f"Input   : {primary_task['input']}\n"
        f"Output  : {primary_task['output']}\n"
        f"Constraints:\n{constraints_block}\n\n"

        # ── 3. Team-level goal template ───────────────────────────────────
        "## TEAM GOAL\n"
        f"{original_goal_template}"
    )
    return enriched


def enrich_goal_templates(
    output_team: dict,
    role_index: dict,
    dept_index: dict,
) -> int:
    """
    Mutate output_team in-place:
      1. Stamp team_id onto each agent's id if not already prefixed.
         Format becomes: {team_id}_{DEPT-ID}_{ROLE-ID}_{slug}
      2. Replace goal_template with the enriched version built from
         dept description + role primary_task + original goal_template.
      3. Preserve the original goal_template under original_goal_template.

    Returns the number of agents enriched.
    """
    team_id = resolve_team_id(output_team.get("team", {}))
    enriched_count = 0

    for agent in output_team["agents"]:
        role = role_index[agent["role_id"]]
        dept = dept_index[agent["department_id"]]

        # ── Stamp team_id into agent id ───────────────────────────────
        current_id = agent.get("id", "")
        if not current_id.startswith(f"{team_id}_"):
            agent["id"] = f"{team_id}_{current_id}"

        # ── Enrich goal_template ──────────────────────────────────────
        original = agent.get("goal_template", "")
        agent["original_goal_template"] = original
        agent["goal_template"] = build_goal_template(
            original_goal_template=original,
            dept=dept,
            role=role,
        )
        enriched_count += 1

    return enriched_count


# ---------------------------------------------------------------------------
# Bedrock helpers
# ---------------------------------------------------------------------------

def get_bedrock_client(region: str):
    return boto3.client("bedrock-agent", region_name=region)


def find_existing_agent(client, agent_name: str) -> dict | None:
    """
    Search for a Bedrock agent by name. Returns the agent summary dict or None.
    """
    paginator = client.get_paginator("list_agents")
    for page in paginator.paginate():
        for summary in page.get("agentSummaries", []):
            if summary["agentName"] == agent_name:
                return summary
    return None


def create_bedrock_agent(
    client,
    agent_id_slug: str,
    agent_name: str,
    goal_template: str,
    schema_ref: str,
    role_obj: dict,
    bedrock_role_arn: str,
    foundation_model: str = "anthropic.claude-sonnet-4-20250514-v1:0",
) -> tuple[str, str]:
    """
    Create a Bedrock agent + default alias.
    Returns (agentId, aliasId).
    """
    primary_task = role_obj["agent_config"]["primary_task"]
    constraints  = "\n".join(f"  - {c}" for c in primary_task.get("constraints", []))

    instruction = (
        f"You are {agent_name}.\n\n"
        f"Persona: {role_obj['agent_config']['persona']}\n\n"
        f"Goal:\n{goal_template}\n\n"
        f"Action  : {primary_task['action']}\n"
        f"Input   : {primary_task['input']}\n"
        f"Output  : {primary_task['output']}\n"
        f"Constraints:\n{constraints}\n\n"
        f"Output schema : {schema_ref}\n"
        f"Escalation    : {role_obj['agent_config']['escalation_policy']}"
    )

    log.info(f"  Creating Bedrock agent: {agent_name}")
    response = client.create_agent(
        agentName=agent_name,
        agentResourceRoleArn=bedrock_role_arn,
        foundationModel=foundation_model,
        instruction=instruction,
        description=f"Agent for team role {agent_id_slug}. Schema: {schema_ref}.",
    )
    agent_id = response["agent"]["agentId"]

    # Wait for agent to be ready
    _wait_for_agent(client, agent_id)

    # Prepare agent (needed before alias creation)
    log.info(f"  Preparing agent {agent_id}")
    client.prepare_agent(agentId=agent_id)
    _wait_for_agent(client, agent_id, target_status="PREPARED")

    # Create alias
    log.info(f"  Creating alias for agent {agent_id}")
    alias_resp = client.create_agent_alias(
        agentId=agent_id,
        agentAliasName="live",
        description="Default live alias created by provision_team.py",
    )
    alias_id = alias_resp["agentAlias"]["agentAliasId"]

    return agent_id, alias_id


def _wait_for_agent(client, agent_id: str, target_status: str = "NOT_PREPARED", max_wait: int = 120):
    waited = 0
    interval = 5
    while waited < max_wait:
        resp = client.get_agent(agentId=agent_id)
        status = resp["agent"]["agentStatus"]
        if status == target_status:
            return
        if "FAILED" in status:
            raise RuntimeError(f"Agent {agent_id} entered failed state: {status}")
        log.info(f"  Agent {agent_id} status: {status} — waiting...")
        time.sleep(interval)
        waited += interval
    raise TimeoutError(f"Agent {agent_id} did not reach '{target_status}' within {max_wait}s")


# ---------------------------------------------------------------------------
# S3 versioning helpers
# ---------------------------------------------------------------------------
# S3 layout:
#   {prefix}/{team_name}/v1/team.json
#   {prefix}/{team_name}/v2/team.json
#   ...
#
# The script:
#   1. Lists all existing version folders under the team prefix.
#   2. Prompts the user to pick one (to update) or create the next version.
#   3. Writes to  {prefix}/{team_name}/{chosen_version}/team.json
# ---------------------------------------------------------------------------

def push_to_s3(
    bucket: str,
    prefix: str,
    team_name: str,
    payload: dict,
    region: str,
    version: str,
) -> str:
    """
    Write payload to s3://{bucket}/{prefix}/{team_name}/{version}/team.json.
    Returns the full S3 URI.
    """
    s3  = boto3.client("s3", region_name=region)
    key = f"{prefix.rstrip('/')}/{team_name}/{version}/team.json"
    body = json.dumps(payload, indent=2).encode("utf-8")

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        Metadata={
            "team":    team_name,
            "version": version,
            "updated": datetime.now(timezone.utc).isoformat(),
        },
    )
    log.info(f"  Pushed to s3://{bucket}/{key}")
    return f"s3://{bucket}/{key}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan for team*.json files and autonomously provision only those "
            "where one or more agents are missing Bedrock agentId/aliasId. "
            "No interactive prompts — safe to run as a CI/CD workflow step."
        )
    )
    parser.add_argument("--scan-dir", default=".", dest="scan_dir",
                        help="Directory to scan for team*.json files (default: cwd)")
    parser.add_argument("--roles",  required=True, help="Path to roles.json")
    parser.add_argument("--depts",  required=True, help="Path to departments.json")
    parser.add_argument("--bucket", default=None,  help="S3 bucket (or ARTIFACT_BUCKET env var)")
    parser.add_argument("--prefix", default="teams", help="S3 key prefix (default: teams)")
    parser.add_argument("--region", default=None,   help="AWS region (or AWS_REGION env var)")
    parser.add_argument("--bedrock-role-arn", default=None, dest="bedrock_role_arn",
                        help="IAM role ARN for Bedrock agents (or BEDROCK_ROLE_ARN env var)")
    parser.add_argument("--foundation-model", default="anthropic.claude-sonnet-4-20250514-v1:0",
                        dest="foundation_model", help="Bedrock foundation model ID")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and enrich only — skip all AWS calls")
    return parser.parse_args()


def process_team(
    team_file:       dict,
    role_index:      dict,
    dept_index:      dict,
    bucket:          str,
    prefix:          str,
    region:          str,
    bedrock_role_arn: str | None,
    foundation_model: str,
    dry_run:         bool,
) -> bool:
    """
    Run the full pipeline for a single team file.
    Returns True on success, False on failure (caller decides whether to abort or continue).
    """
    team_path  = team_file["path"]
    team_name  = team_file["name"]
    s3_version = team_file["version"]

    log.info(f"{'═'*60}")
    log.info(f"  TEAM    : {team_name}")
    log.info(f"  TEAM_ID : {team_file.get('team_id', resolve_team_id({'name': team_name}))}")
    log.info(f"  FILE    : {team_path.name}")
    log.info(f"  VERSION : {s3_version}")
    log.info(f"  NEEDS   : {team_file.get('agents_needing_ids', '?')}/{team_file.get('total_agents', '?')} agent(s) missing Bedrock IDs")
    log.info(f"{'═'*60}")

    # ── Load ──────────────────────────────────────────────────────────
    try:
        team = load_json(str(team_path))
    except (json.JSONDecodeError, OSError) as e:
        log.error(f"  Failed to load {team_path.name}: {e}")
        return False

    # ── Validate ──────────────────────────────────────────────────────
    log.info("  [1/4] Validating...")
    errors = validate_team(team, role_index, dept_index)
    if errors:
        log.error(f"  Validation failed with {len(errors)} error(s):")
        for e in errors:
            log.error(f"    ✗ {e}")
        return False
    log.info(f"  ✓ {len(team['agents'])} agent(s) validated.")

    # ── Enrich ────────────────────────────────────────────────────────
    log.info("  [2/4] Enriching goal_templates...")
    output_team = copy.deepcopy(team)
    n = enrich_goal_templates(output_team, role_index, dept_index)
    log.info(f"  ✓ {n} agent(s) enriched.")

    # ── Dry run — stop here ───────────────────────────────────────────
    if dry_run:
        log.info("  [DRY RUN] Skipping Bedrock provisioning and S3 push.\n")
        for agent in output_team["agents"]:
            has_ids = bool(agent.get("bedrock", {}).get("agentId"))
            action  = "SKIP (IDs present)" if has_ids else "CREATE"
            role    = role_index[agent["role_id"]]
            log.info(
                f"    [{action}] {agent['id']}  "
                f"role={role['title']}  schema={agent['schema_ref']}"
            )
        log.info(
            f"  → Would push to: "
            f"s3://{bucket}/{prefix}/{team_name}/{s3_version}/team.json"
        )
        return True

    # ── Provision ─────────────────────────────────────────────────────
    if not bedrock_role_arn:
        log.error("  BEDROCK_ROLE_ARN not set — cannot create agents.")
        return False

    log.info("  [3/4] Provisioning Bedrock agents...")
    bedrock = get_bedrock_client(region)

    for i, agent in enumerate(output_team["agents"]):
        agent_id_slug = agent["id"]
        agent_name    = agent["name"]
        role_obj      = role_index[agent["role_id"]]
        tag           = f"    [{i+1}/{len(output_team['agents'])}]"

        existing_id    = agent.get("bedrock", {}).get("agentId",  "").strip()
        existing_alias = agent.get("bedrock", {}).get("aliasId", "").strip()

        if existing_id and existing_alias:
            log.info(f"{tag} SKIP — {agent_name} already has Bedrock IDs.")
            continue

        log.info(f"{tag} Provisioning: {agent_name} ({agent_id_slug})")

        existing = find_existing_agent(bedrock, agent_name)
        if existing:
            b_agent_id = existing["agentId"]
            log.info(f"      Found existing agent: {b_agent_id}")
            aliases = bedrock.list_agent_aliases(agentId=b_agent_id).get("agentAliasSummaries", [])
            if aliases:
                b_alias_id = aliases[0]["agentAliasId"]
                log.info(f"      Using alias: {b_alias_id}")
            else:
                log.warning(f"      No aliases found — creating one.")
                alias_resp = bedrock.create_agent_alias(agentId=b_agent_id, agentAliasName="live")
                b_alias_id = alias_resp["agentAlias"]["agentAliasId"]
        else:
            b_agent_id, b_alias_id = create_bedrock_agent(
                client=bedrock,
                agent_id_slug=agent_id_slug,
                agent_name=agent_name,
                goal_template=agent["goal_template"],
                schema_ref=agent["schema_ref"],
                role_obj=role_obj,
                bedrock_role_arn=bedrock_role_arn,
                foundation_model=foundation_model,
            )

        output_team["agents"][i]["bedrock"] = {
            "agentId": b_agent_id,
            "aliasId": b_alias_id,
        }
        log.info(f"      ✓ agentId={b_agent_id}  aliasId={b_alias_id}")

    # ── Push ──────────────────────────────────────────────────────────
    log.info(f"  [4/4] Pushing to S3...")
    s3_uri = push_to_s3(
        bucket=bucket,
        prefix=prefix,
        team_name=team_name,
        payload=output_team,
        region=region,
        version=s3_version,
    )
    log.info(f"  ✓ {s3_uri}\n")
    return True


def main():
    args = parse_args()

    bucket           = args.bucket           or os.environ.get("ARTIFACT_BUCKET")
    region           = args.region           or os.environ.get("AWS_REGION", "us-east-1")
    bedrock_role_arn = args.bedrock_role_arn or os.environ.get("BEDROCK_ROLE_ARN")

    if not bucket:
        log.error("S3 bucket not set. Use --bucket or ARTIFACT_BUCKET env var.")
        sys.exit(1)

    # ── Load shared config ────────────────────────────────────────────
    roles = load_json(args.roles)
    depts = load_json(args.depts)
    role_index = build_role_index(roles)
    dept_index = build_dept_index(depts)

    # ── Scan — only files with agents missing Bedrock IDs ────────────────
    log.info(f"Scanning: {Path(args.scan_dir).resolve()}")
    team_files = scan_team_files(args.scan_dir)
    log.info(f"  → {len(team_files)} team file(s) need provisioning.\n")

    # ── Process each team file ────────────────────────────────────────
    results: dict[str, bool] = {}
    for tf in team_files:
        success = process_team(
            team_file=tf,
            role_index=role_index,
            dept_index=dept_index,
            bucket=bucket,
            prefix=args.prefix,
            region=region,
            bedrock_role_arn=bedrock_role_arn,
            foundation_model=args.foundation_model,
            dry_run=args.dry_run,
        )
        results[tf["path"].name] = success

    # ── Final summary ─────────────────────────────────────────────────
    print("\n========== RUN SUMMARY ==========")
    all_ok = True
    for filename, ok in results.items():
        status = "✓ OK   " if ok else "✗ FAIL "
        print(f"  {status}  {filename}")
        if not ok:
            all_ok = False
    print("=================================\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
