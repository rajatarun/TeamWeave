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

Event payload:
    {}                          ← provision all team*.json files that need IDs
    { "dry_run": true }         ← validate + enrich only, no Bedrock or S3 calls

Response:
    {
      "statusCode": 200 | 400 | 500,
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

from provision_team import (
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

def handler(event: dict, context) -> dict:
    """
    Lambda entry point.

    Scans SCAN_DIR for team*.json files that need provisioning,
    runs the full pipeline for each, and returns a summary.
    """
    log.info(f"Event: {json.dumps(event)}")
    dry_run = bool(event.get("dry_run", False))

    # Config
    try:
        cfg = get_config()
    except ValueError as e:
        log.error(str(e))
        return {"statusCode": 500, "results": {}, "errors": [str(e)]}

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
        return {"statusCode": 500, "results": {}, "errors": [str(e)]}

    # Scan filesystem for team files needing provisioning
    log.info(f"Scanning: {cfg['scan_dir']}")
    try:
        team_files = scan_team_files(cfg["scan_dir"])
    except SystemExit:
        # scan_team_files calls sys.exit when nothing needs work — treat as success
        log.info("All team files already provisioned or none found.")
        return {"statusCode": 200, "results": {}, "errors": [],
                "message": "Nothing to provision."}

    log.info(f"  → {len(team_files)} team file(s) queued.\n")

    # Process each
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

    # Summary
    status = 200 if not all_errors else 500
    log.info(f"\n{'='*40}")
    for name, r in results.items():
        log.info(f"  {'✓' if r['success'] else '✗'}  {name}")
    log.info(f"  Status: {status}")
    log.info(f"{'='*40}\n")

    return {
        "statusCode": status,
        "results":    results,
        "errors":     all_errors,
    }
