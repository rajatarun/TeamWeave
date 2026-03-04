"""
lambda_handler.py
-----------------
AWS Lambda entry point for team provisioning.

Filesystem layout expected in the Lambda deployment package:
    /var/task/
        lambda_handler.py       ← this file
        provision_team.py       ← core logic
        roles.json              ← or override via ROLES_PATH env var
        departments.json        ← or override via DEPTS_PATH env var
        team*.json              ← any number of team files in SCAN_DIR

Environment variables:
    ARTIFACT_BUCKET   (required) S3 bucket to push provisioned team.json
    BEDROCK_ROLE_ARN  (required) IAM role ARN Bedrock agents will assume
    SCAN_DIR          (optional) directory to scan for team*.json  (default: /var/task)
    ROLES_PATH        (optional) path to roles.json               (default: /var/task/roles.json)
    DEPTS_PATH        (optional) path to departments.json         (default: /var/task/departments.json)
    OUTPUT_PREFIX     (optional) S3 key prefix for output         (default: teams)
    FOUNDATION_MODEL  (optional) Bedrock model ID                 (default: amazon.nova-micro-v1:0)

Trigger:
    API Gateway HTTP API or REST API (Lambda proxy integration).

Request:
    POST /provision
    Content-Type: application/json

    {}                          ← provision all team*.json files that need IDs
    { "dry_run": true }         ← validate + enrich only, no Bedrock or S3 calls

Response (API Gateway proxy format):
    HTTP 200 / 500
    Content-Type: application/json

    {
      "statusCode": 200 | 500,
      "results": {
        "team_name": { "success": true, "s3_uri": "s3://...", "errors": [] },
        ...
      },
      "errors": []
    }
"""

import copy
import json
import logging
import os

import boto3

from .provision_team import (
    load_json,
    build_role_index,
    build_dept_index,
    scan_team_files,
    validate_team,
    enrich_goal_templates,
    find_existing_agent,
    create_bedrock_agent,
    push_to_s3,
    resolve_team_id,
    needs_provisioning,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

def get_config() -> dict:
    bucket = os.environ.get("ARTIFACT_BUCKET", "").strip()
    if not bucket:
        raise ValueError("ARTIFACT_BUCKET environment variable is not set.")

    bedrock_role_arn = os.environ.get("BEDROCK_ROLE_ARN", "").strip()
    if not bedrock_role_arn:
        raise ValueError("BEDROCK_ROLE_ARN environment variable is not set.")

    task_dir = "/var/task"
    return {
        "bucket":           bucket,
        "bedrock_role_arn": bedrock_role_arn,
        "scan_dir":         os.environ.get("SCAN_DIR",         task_dir),
        "roles_path":       os.environ.get("ROLES_PATH",       os.path.join(task_dir, "roles.json")),
        "depts_path":       os.environ.get("DEPTS_PATH",       os.path.join(task_dir, "departments.json")),
        "output_prefix":    os.environ.get("OUTPUT_PREFIX",    "teams"),
        "foundation_model": os.environ.get("FOUNDATION_MODEL", "amazon.nova-micro-v1:0"),
        "region":           os.environ.get("AWS_REGION",       "us-east-1"),
    }


# ---------------------------------------------------------------------------
# Per-team pipeline
# ---------------------------------------------------------------------------

def process_team_entry(
    team_file:        dict,
    role_index:       dict,
    dept_index:       dict,
    cfg:              dict,
    dry_run:          bool,
) -> dict:
    """
    Run validate → enrich → provision → push for one team file entry
    (as returned by scan_team_files).

    Returns { "success": bool, "s3_uri": str|None, "errors": list[str] }
    """
    team_path  = team_file["path"]
    team_name  = team_file["name"]
    s3_version = team_file["version"]

    log.info(f"{'═'*60}")
    log.info(f"  TEAM    : {team_name}")
    log.info(f"  TEAM_ID : {team_file.get('team_id', resolve_team_id({'name': team_name}))}")
    log.info(f"  FILE    : {team_path.name}")
    log.info(f"  VERSION : {s3_version}")
    log.info(f"  NEEDS   : {team_file['agents_needing_ids']}/{team_file['total_agents']} agent(s) missing IDs")
    log.info(f"{'═'*60}")

    # Load
    try:
        team = load_json(str(team_path))
    except Exception as e:
        return {"success": False, "s3_uri": None, "errors": [f"Load failed: {e}"]}

    # Validate
    log.info("  [1/4] Validating...")
    errors = validate_team(team, role_index, dept_index)
    if errors:
        log.error(f"  Validation failed: {errors}")
        return {"success": False, "s3_uri": None, "errors": errors}
    log.info(f"  ✓ {len(team['agents'])} agent(s) validated.")

    # Enrich
    log.info("  [2/4] Enriching goal_templates...")
    output_team = copy.deepcopy(team)
    n = enrich_goal_templates(output_team, role_index, dept_index)
    log.info(f"  ✓ {n} agent(s) enriched.")

    # Dry run — stop here
    if dry_run:
        log.info("  [DRY RUN] Skipping Bedrock provisioning and S3 push.")
        return {"success": True, "s3_uri": None, "errors": [], "dry_run": True}

    # Provision
    log.info("  [3/4] Provisioning Bedrock agents...")
    bedrock = boto3.client("bedrock-agent", region_name=cfg["region"])

    for i, agent in enumerate(output_team["agents"]):
        existing_id    = agent.get("bedrock", {}).get("agentId",  "").strip()
        existing_alias = agent.get("bedrock", {}).get("aliasId", "").strip()
        tag            = f"    [{i+1}/{len(output_team['agents'])}]"

        if existing_id and existing_alias:
            log.info(f"{tag} SKIP — {agent['name']} already provisioned.")
            continue

        log.info(f"{tag} Provisioning: {agent['name']} ({agent['id']})")
        role_obj = role_index[agent["role_id"]]

        existing = find_existing_agent(bedrock, agent["name"])
        if existing:
            b_agent_id = existing["agentId"]
            aliases = bedrock.list_agent_aliases(agentId=b_agent_id).get("agentAliasSummaries", [])
            b_alias_id = (
                aliases[0]["agentAliasId"] if aliases
                else bedrock.create_agent_alias(
                    agentId=b_agent_id, agentAliasName="live"
                )["agentAlias"]["agentAliasId"]
            )
        else:
            b_agent_id, b_alias_id = create_bedrock_agent(
                client=bedrock,
                agent_id_slug=agent["id"],
                agent_name=agent["name"],
                goal_template=agent["goal_template"],
                schema_ref=agent["schema_ref"],
                role_obj=role_obj,
                bedrock_role_arn=cfg["bedrock_role_arn"],
                foundation_model=cfg["foundation_model"],
            )

        output_team["agents"][i]["bedrock"] = {
            "agentId": b_agent_id,
            "aliasId": b_alias_id,
        }
        log.info(f"      ✓ agentId={b_agent_id}  aliasId={b_alias_id}")

    # Push
    log.info("  [4/4] Pushing to S3...")
    s3_uri = push_to_s3(
        bucket=cfg["bucket"],
        prefix=cfg["output_prefix"],
        team_name=team_name,
        payload=output_team,
        region=cfg["region"],
        version=s3_version,
    )
    log.info(f"  ✓ {s3_uri}")

    return {"success": True, "s3_uri": s3_uri, "errors": []}


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def _apigw_response(status: int, body: dict) -> dict:
    """Wrap a response body in API Gateway proxy integration format."""
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _parse_body(event: dict) -> dict:
    """
    Extract and parse the request body from an API Gateway proxy event.
    Handles both REST API (event["body"] is a string) and direct invocation
    (event already is the body dict).
    """
    if "body" in event:
        raw = event["body"]
        if not raw:
            return {}
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                raise ValueError(f"Request body is not valid JSON: {raw!r}")
        if isinstance(raw, dict):
            return raw
    # Direct Lambda invocation — event itself is the payload
    return event





# ---------------------------------------------------------------------------
# Delete pipeline
# ---------------------------------------------------------------------------

def delete_agents(agent_names: list[str], region: str) -> dict:
    """
    Delete Bedrock agents by name.

    For each name:
      1. Search for the agent by sanitised name
      2. Delete all aliases first (Bedrock requires this before agent deletion)
      3. Delete the agent

    Returns:
        {
          "deleted": ["Agent_Name", ...],
          "not_found": ["Missing_Name", ...],
          "errors": { "Agent_Name": "error message", ... }
        }
    """
    from .provision_team import sanitise_agent_name

    bedrock  = boto3.client("bedrock-agent", region_name=region)
    deleted   = []
    not_found = []
    errors    = {}

    for raw_name in agent_names:
        safe_name = sanitise_agent_name(raw_name)
        log.info(f"  Searching for agent: '{safe_name}'")

        try:
            agent = find_existing_agent(bedrock, raw_name)

            if not agent:
                log.warning(f"  NOT FOUND: '{safe_name}'")
                not_found.append(raw_name)
                continue

            agent_id = agent["agentId"]
            log.info(f"  Found: {safe_name} → agentId={agent_id}")

            # Delete all aliases first — skip gracefully if none found or call fails
            try:
                aliases = bedrock.list_agent_aliases(
                    agentId=agent_id
                ).get("agentAliasSummaries", [])
            except Exception as e:
                log.warning(f"    Could not list aliases for {agent_id} — skipping alias deletion: {e}")
                aliases = []

            if not aliases:
                log.info("    No aliases found — proceeding to delete agent.")
            else:
                for alias in aliases:
                    alias_id = alias["agentAliasId"]
                    try:
                        log.info(f"    Deleting alias: {alias_id}")
                        bedrock.delete_agent_alias(agentId=agent_id, agentAliasId=alias_id)
                    except Exception as e:
                        log.warning(f"    Could not delete alias {alias_id} — skipping: {e}")
                log.info(f"    Processed {len(aliases)} alias(es). Deleting agent...")

            # Delete the agent
            bedrock.delete_agent(agentId=agent_id, skipResourceInUseCheck=True)
            log.info(f"  ✓ Deleted: {safe_name} ({agent_id})")
            deleted.append(raw_name)

        except Exception as e:
            log.error(f"  ✗ Failed to delete '{safe_name}': {e}")
            errors[raw_name] = str(e)

    return {"deleted": deleted, "not_found": not_found, "errors": errors}


# ---------------------------------------------------------------------------
# Router — dispatch by HTTP method
# ---------------------------------------------------------------------------

def handler(event: dict, context) -> dict:
    """
    Lambda entry point — triggered via API Gateway proxy integration.

    Routes:
        POST   /provision  → provision all team*.json files needing IDs
        DELETE /provision  → delete agents by name

    POST body:
        {}                          ← provision all
        { "dry_run": true }         ← validate + enrich only

    DELETE body:
        { "agent_names": ["Technical Coach", "Learning Strategist"] }
    """
    log.info(f"Event: {json.dumps(event)}")

    # Parse body
    try:
        body = _parse_body(event)
    except ValueError as e:
        log.error(str(e))
        return _apigw_response(400, {"error": str(e), "errors": [str(e)]})

    # Config
    try:
        cfg = get_config()
    except ValueError as e:
        log.error(str(e))
        return _apigw_response(500, {"errors": [str(e)]})

    # Route by HTTP method
    method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method", "POST")
    method = method.upper()
    log.info(f"Method: {method}")

    # ── DELETE ────────────────────────────────────────────────────────
    if method == "DELETE":
        agent_names = body.get("agent_names", [])
        if not agent_names:
            return _apigw_response(400, {
                "error": "'agent_names' list is required for DELETE.",
                "errors": ["'agent_names' list is required for DELETE."]
            })
        if not isinstance(agent_names, list):
            return _apigw_response(400, {
                "error": "'agent_names' must be a list of strings.",
                "errors": ["'agent_names' must be a list of strings."]
            })

        log.info(f"Deleting {len(agent_names)} agent(s): {agent_names}")
        result = delete_agents(agent_names, cfg["region"])

        status = 200 if not result["errors"] else 500
        log.info(f"Delete summary — deleted={result['deleted']} not_found={result['not_found']} errors={result['errors']}")
        return _apigw_response(status, result)

    # ── POST (provision) ──────────────────────────────────────────────
    if method == "POST":
        dry_run = bool(body.get("dry_run", False))

        # Load shared config from filesystem
        try:
            log.info(f"Loading roles:       {cfg['roles_path']}")
            log.info(f"Loading departments: {cfg['depts_path']}")
            roles_data = load_json(cfg["roles_path"])
            depts_data = load_json(cfg["depts_path"])
            role_index = build_role_index(roles_data)
            dept_index = build_dept_index(depts_data)
            log.info(f"  ✓ {len(role_index)} roles, {len(dept_index)} departments.")
        except Exception as e:
            log.error(f"Failed to load config files: {e}")
            return _apigw_response(500, {"results": {}, "errors": [str(e)]})

        # Scan filesystem for team files needing provisioning
        log.info(f"Scanning: {cfg['scan_dir']}")
        try:
            team_files = scan_team_files(cfg["scan_dir"])
        except SystemExit:
            log.info("All team files already provisioned or none found.")
            return _apigw_response(200, {"results": {}, "errors": [], "message": "Nothing to provision."})

        log.info(f"  → {len(team_files)} team file(s) queued.\n")

        results    = {}
        all_errors = []

        for tf in team_files:
            try:
                result = process_team_entry(
                    team_file=tf,
                    role_index=role_index,
                    dept_index=dept_index,
                    cfg=cfg,
                    dry_run=dry_run,
                )
            except Exception as e:
                log.exception(f"Unexpected error processing {tf['path'].name}")
                result = {"success": False, "s3_uri": None, "errors": [str(e)]}

            results[tf["name"]] = result
            if not result["success"]:
                all_errors.extend(result.get("errors", []))

        status = 200 if not all_errors else 500
        log.info(f"\n{'='*40}")
        for name, r in results.items():
            log.info(f"  {'✓' if r['success'] else '✗'}  {name}")
        log.info(f"  Status: {status}")
        log.info(f"{'='*40}\n")

        return _apigw_response(status, {"results": results, "errors": all_errors})

    # ── Unsupported method ────────────────────────────────────────────
    return _apigw_response(405, {"error": f"Method '{method}' not allowed. Use POST or DELETE."})
