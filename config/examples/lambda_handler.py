"""
lambda_handler.py  —  Agent Management System
──────────────────────────────────────────────
All state lives in S3 under OUTPUT_PREFIX:

    {prefix}/roles.json
    {prefix}/departments.json
    {prefix}/{team_name}/{version}/team.json

Environment variables:
    ARTIFACT_BUCKET   (required) S3 bucket — all state lives here
    BEDROCK_ROLE_ARN  (required) IAM role ARN Bedrock agents assume
    OUTPUT_PREFIX     (default: agent-management)
    FOUNDATION_MODEL  (default: amazon.nova-micro-v1:0)
    AWS_REGION        (default: us-east-1)

ROUTE TABLE
───────────────────────────────────────────────────────────────────────────────
Agents
  GET    /agents                list all Bedrock agents (live from Bedrock)
  GET    /agents/{name}         get agent + aliases by name
  POST   /agents                create single agent from body
  PUT    /agents/{name}         update agent instruction / description / model
  DELETE /agents                delete agents by name list

Teams
  GET    /teams                 list all teams from S3
  GET    /teams/{team_name}     get latest team.json from S3
  POST   /teams                 provision all unprovisioned teams in S3
  PUT    /teams/{team_name}     update team in S3 + re-provision
  DELETE /teams/{team_name}     delete all agents of team + remove all S3 versions

Roles
  GET    /roles                 list all roles
  GET    /roles/{role_id}       get one role
  POST   /roles                 add new role
  PUT    /roles/{role_id}       update existing role

Departments
  GET    /departments           list all departments
  GET    /departments/{dept_id} get one department
  POST   /departments           add new department
  PUT    /departments/{dept_id} update existing department
───────────────────────────────────────────────────────────────────────────────
"""

import copy
import json
import logging
import os
import re
import time
from collections import defaultdict
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import ClientError

from .provision_team import (
    _attach_gemini_action_group,
    build_dept_index,
    build_role_index,
    create_bedrock_agent,
    enrich_goal_templates,
    find_existing_agent,
    generate_team_id,
    needs_provisioning,
    push_to_s3,
    resolve_team_id,
    sanitise_agent_name,
    validate_team,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Model alias slugs — maps inference-profile model IDs to short alias names
# used when creating per-model Bedrock agent aliases.
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_ALIAS_SLUG: dict[str, str] = {
    "us.amazon.nova-micro-v1:0":                    "nova-micro",
    "us.amazon.nova-lite-v1:0":                     "nova-lite",
    "us.amazon.nova-pro-v1:0":                      "nova-pro",
    "us.amazon.nova-premier-v1:0":                  "nova-premier",
    "us.anthropic.claude-3-haiku-20240307-v1:0":    "claude-3-haiku",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0":  "claude-35-haiku",
}

_RETRYABLE_AGENT_STATES = {"PREPARING", "UPDATING", "VERSIONING"}


def _wait_for_agent_stable_state(
    bedrock,
    agent_id: str,
    retry_states: set[str] | None = None,
    max_wait_seconds: int = 120,
    poll_interval_seconds: int = 5,
) -> str:
    """Wait until the Bedrock agent leaves transient states and return status."""
    states = retry_states or _RETRYABLE_AGENT_STATES
    waited = 0
    while waited <= max_wait_seconds:
        status = bedrock.get_agent(agentId=agent_id)["agent"].get("agentStatus", "")
        if status not in states:
            return status
        log.info(f"  Agent {agent_id} status={status}; waiting before retry")
        time.sleep(poll_interval_seconds)
        waited += poll_interval_seconds
    raise TimeoutError(
        f"Agent {agent_id} stayed in transient states {sorted(states)} for more than "
        f"{max_wait_seconds}s"
    )


def _update_agent_with_retry(bedrock, *, agent_id: str, max_attempts: int = 6, **kwargs):
    """Retry update_agent when Bedrock rejects updates for transient agent states."""
    for attempt in range(1, max_attempts + 1):
        try:
            return bedrock.update_agent(agentId=agent_id, **kwargs)
        except ClientError as e:
            err = e.response.get("Error", {}) if isinstance(e.response, dict) else {}
            code = err.get("Code", "")
            msg = err.get("Message", str(e))
            msg_upper = msg.upper()
            retryable = (
                code == "ValidationException"
                and "CAN'T BE PERFORMED ON AGENT" in msg_upper
                and any(state in msg_upper for state in _RETRYABLE_AGENT_STATES)
            )
            if not retryable or attempt == max_attempts:
                raise
            log.warning(
                f"  WARN update_agent retry {attempt}/{max_attempts} for {agent_id}: {msg}"
            )
            _wait_for_agent_stable_state(bedrock, agent_id)



def _prepare_agent_and_wait(bedrock, agent_id: str) -> None:
    """Prepare an agent and wait for it to leave PREPARING state."""
    bedrock.prepare_agent(agentId=agent_id)
    _wait_for_agent_stable_state(
        bedrock,
        agent_id,
        retry_states={"PREPARING"},
        max_wait_seconds=180,
        poll_interval_seconds=5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def get_config() -> dict:
    bucket = os.environ.get("ARTIFACT_BUCKET", "").strip()
    if not bucket:
        raise ValueError("ARTIFACT_BUCKET environment variable is not set.")
    bedrock_role_arn = os.environ.get("BEDROCK_ROLE_ARN", "").strip()
    if not bedrock_role_arn:
        raise ValueError("BEDROCK_ROLE_ARN environment variable is not set.")
    prefix = os.environ.get("OUTPUT_PREFIX", "").strip().strip("/")
    return {
        "bucket":           bucket,
        "bedrock_role_arn": bedrock_role_arn,
        "prefix":           prefix,
        "roles_key":        f"{prefix}/roles.json" if prefix else "roles.json",
        "depts_key":        f"{prefix}/departments.json" if prefix else "departments.json",
        "teams_prefix":     f"{prefix}/teams" if prefix else "teams",
        "foundation_model": os.environ.get("FOUNDATION_MODEL", "amazon.nova-micro-v1:0"),
        "region":           os.environ.get("AWS_REGION",       "us-east-1"),
        "gemini_lambda_arn": os.environ.get("GEMINI_LAMBDA_ARN", "").strip(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# APIGW helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resp(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers":    {"Content-Type": "application/json"},
        "body":       json.dumps(body, default=str),
    }

def _ok(body)              : return _resp(200, body)
def _created(body)         : return _resp(201, body)
def _bad_request(msg)      : return _resp(400, {"error": msg})
def _not_found(msg)        : return _resp(404, {"error": msg})
def _server_error(msg)     : return _resp(500, {"error": str(msg)})
def _method_not_allowed(m) : return _resp(405, {"error": f"Method '{m}' not allowed on this resource."})


def _parse_body(event: dict) -> dict:
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
    return event


def _extract_route(event: dict) -> tuple:
    """Returns (method, resource, path_param)."""
    method = (
        event.get("httpMethod")
        or event.get("requestContext", {}).get("http", {}).get("method", "GET")
    ).upper()
    raw_path = (
        event.get("path")
        or event.get("requestContext", {}).get("http", {}).get("path", "/")
    )
    parts      = [p for p in raw_path.strip("/").split("/") if p]
    resource   = parts[0].lower() if parts else ""
    path_param = unquote_plus(parts[1]) if len(parts) > 1 else None
    return method, resource, path_param


# ─────────────────────────────────────────────────────────────────────────────
# S3 core — all reads and writes go through here
# ─────────────────────────────────────────────────────────────────────────────

def _s3(cfg):
    return boto3.client("s3", region_name=cfg["region"])


def _s3_get(cfg, key: str) -> dict:
    """Download and parse a JSON object from S3. Raises KeyError if not found."""
    try:
        resp = _s3(cfg).get_object(Bucket=cfg["bucket"], Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            raise KeyError(f"s3://{cfg['bucket']}/{key} not found.")
        raise


def _s3_put(cfg, key: str, data: dict) -> str:
    """Serialise data and write to S3. Returns the S3 URI."""
    _s3(cfg).put_object(
        Bucket=cfg["bucket"],
        Key=key,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    uri = f"s3://{cfg['bucket']}/{key}"
    log.info(f"  Wrote {uri}")
    return uri


def _s3_delete(cfg, key: str) -> None:
    _s3(cfg).delete_object(Bucket=cfg["bucket"], Key=key)
    log.info(f"  Deleted s3://{cfg['bucket']}/{key}")


def _s3_list_prefix(cfg, prefix: str) -> list[str]:
    """Return all S3 keys under prefix."""
    paginator = _s3(cfg).get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=cfg["bucket"], Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


# ─────────────────────────────────────────────────────────────────────────────
# Roles & Departments — S3-backed
# ─────────────────────────────────────────────────────────────────────────────

def _load_roles(cfg) -> dict:
    try:
        return _s3_get(cfg, cfg["roles_key"])
    except KeyError:
        raise KeyError(f"roles.json not found at s3://{cfg['bucket']}/{cfg['roles_key']}. "
                       f"Upload it first.")

def _save_roles(cfg, data: dict) -> str:
    return _s3_put(cfg, cfg["roles_key"], data)

def _load_depts(cfg) -> dict:
    try:
        return _s3_get(cfg, cfg["depts_key"])
    except KeyError:
        raise KeyError(f"departments.json not found at s3://{cfg['bucket']}/{cfg['depts_key']}. "
                       f"Upload it first.")

def _save_depts(cfg, data: dict) -> str:
    return _s3_put(cfg, cfg["depts_key"], data)


# ─────────────────────────────────────────────────────────────────────────────
# Teams — S3-backed
# ─────────────────────────────────────────────────────────────────────────────

def _scan_s3_teams(cfg) -> dict:
    """
    Scan bucket under teams_prefix and return:
        { team_name: { version: s3_key } }

    Expected key shape:
        {teams_prefix}/{team_name}/{version}/team.json
    """
    prefix   = cfg["teams_prefix"].rstrip("/") + "/"
    all_keys = _s3_list_prefix(cfg, prefix)
    team_map = defaultdict(dict)
    for key in all_keys:
        rel   = key[len(prefix):]               # team_name/v1/team.json
        parts = rel.split("/")
        if len(parts) == 3 and parts[2].startswith("team"):
            team_name, version = parts[0], parts[1]
            team_map[team_name][version] = key
    log.info(f"S3 scan — {len(team_map)} team(s) found under s3://{cfg['bucket']}/{prefix}")
    return dict(team_map)


def _latest_version(versions: dict) -> str:
    def _vnum(v):
        m = re.match(r"^v(\d+)$", v)
        return int(m.group(1)) if m else 0
    return max(versions.keys(), key=_vnum)


def _bump_version(v: str) -> str:
    m = re.match(r"^v(\d+)$", str(v))
    return f"v{int(m.group(1)) + 1}" if m else "v2"


def _bedrock(cfg):
    return boto3.client("bedrock-agent", region_name=cfg["region"])


# ─────────────────────────────────────────────────────────────────────────────
# Provision helper
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_tool_hint(enabled: bool) -> str:
    if not enabled:
        return ""
    return (
        "\n\n## RESEARCH TOOL — gemini_research\n"
        "You have access to a research tool: gemini_research(query).\n"
        "Use it to look up facts, find resources, or ground claims you are not certain about.\n\n"
        "Call it when useful:\n"
        "  - Factual claims or statistics not in your input\n"
        "  - Technical explanations or framework details\n"
        "  - Resource URLs you need to verify or find\n"
        "  - Recent trends or current data\n\n"
        "You may call it zero or more times depending on what you need.\n"
        "If the tool is unavailable or returns an error, proceed without it — "
        "do not refuse or stall. Your primary job is to produce the output, "
        "with or without the tool."
    )


def _build_instruction(agent: dict, role_obj: dict, gemini_enabled: bool = False) -> str:
    """
    Construct the full Bedrock instruction string from role + agent config.
    Uses original_goal_template (pre-enrichment) if present to avoid
    double-wrapping an already enriched goal_template.
    """
    primary_task = role_obj["agent_config"]["primary_task"]
    constraints  = "\n".join(f"  - {c}" for c in primary_task.get("constraints", []))
    goal = agent.get("original_goal_template") or agent.get("goal_template", "")
    return (
        f"You are {agent['name']}.\n\n"
        f"Persona: {role_obj['agent_config']['persona']}\n\n"
        f"Goal:\n{goal}\n\n"
        f"Action  : {primary_task['action']}\n"
        f"Input   : {primary_task['input']}\n"
        f"Output  : {primary_task['output']}\n"
        f"Constraints:\n{constraints}\n\n"
        f"Output schema : {agent['schema_ref']}\n"
        f"Escalation    : {role_obj['agent_config']['escalation_policy']}"
        f"{_gemini_tool_hint(gemini_enabled)}"
    )



def _provision_model_aliases(
    bedrock,
    agent_id: str,
    main_model_id: str,
    main_alias_id: str,
    model_aliases: dict,
    instruction: str,
    agent_name: str,
) -> dict:
    """
    Create one Bedrock agent alias per model variant listed in *model_aliases*.

    For the primary model the existing *main_alias_id* is recorded without any
    API calls.  For every other model the agent foundation model is swapped,
    the agent is prepared (creating a new version), and an alias named after
    the model slug is created.  After all variants the agent is restored to
    *main_model_id*.

    Returns an updated ``model_aliases`` dict ``{model_id: alias_id}``.
    """
    result = dict(model_aliases)
    result[main_model_id] = main_alias_id

    try:
        existing_by_name = {
            a["agentAliasName"]: a["agentAliasId"]
            for a in bedrock.list_agent_aliases(agentId=agent_id).get("agentAliasSummaries", [])
        }
    except Exception as e:
        log.warning(f"  WARN list_aliases failed for {agent_id}: {e}")
        return result

    try:
        current_agent = bedrock.get_agent(agentId=agent_id)["agent"]
    except Exception as e:
        log.warning(f"  WARN get_agent failed for {agent_id}: {e}")
        return result

    for model_id, alias_slug in _MODEL_ALIAS_SLUG.items():
        if model_id == main_model_id or model_id not in model_aliases:
            continue

        if alias_slug in existing_by_name:
            result[model_id] = existing_by_name[alias_slug]
            log.info(f"  ALIAS EXISTS {agent_name}/{alias_slug} → {existing_by_name[alias_slug]}")
            continue

        try:
            _update_agent_with_retry(
                bedrock,
                agent_id=agent_id,
                agentName=current_agent["agentName"],
                agentResourceRoleArn=current_agent["agentResourceRoleArn"],
                foundationModel=model_id,
                instruction=instruction,
                description=current_agent.get("description", ""),
            )
            _prepare_agent_and_wait(bedrock, agent_id)
            alias_resp = bedrock.create_agent_alias(
                agentId=agent_id,
                agentAliasName=alias_slug,
                description=f"Model variant alias: {model_id}",
                tags={"model_id": alias_slug, "alias_type": "model_variant"},
            )
            alias_id = alias_resp["agentAlias"]["agentAliasId"]
            result[model_id] = alias_id
            log.info(f"  ALIAS CREATED {agent_name}/{alias_slug} → {alias_id}")
        except Exception as e:
            log.warning(f"  WARN alias creation failed {agent_name}/{alias_slug}: {e}")

    # Restore agent to its primary model
    try:
        _update_agent_with_retry(
            bedrock,
            agent_id=agent_id,
            agentName=current_agent["agentName"],
            agentResourceRoleArn=current_agent["agentResourceRoleArn"],
            foundationModel=main_model_id,
            instruction=instruction,
            description=current_agent.get("description", ""),
        )
        _prepare_agent_and_wait(bedrock, agent_id)
        log.info(f"  RESTORED {agent_name} → {main_model_id}")
    except Exception as e:
        log.warning(f"  WARN restore primary model failed for {agent_name}: {e}")

    return result


def _provision_team(team_data: dict, team_name: str, version: str,
                    role_index: dict, dept_index: dict, cfg: dict,
                    dry_run: bool = False) -> dict:
    errors = validate_team(team_data, role_index, dept_index)
    if errors:
        return {"success": False, "s3_uri": None, "errors": errors}

    output_team = copy.deepcopy(team_data)
    enrich_goal_templates(output_team, role_index, dept_index)

    if dry_run:
        return {"success": True, "s3_uri": None, "errors": [], "dry_run": True}

    bedrock = _bedrock(cfg)
    gemini_lambda_arn = cfg.get("gemini_lambda_arn", "")
    gemini_enabled    = bool(gemini_lambda_arn)

    for i, agent in enumerate(output_team["agents"]):
        role_obj         = role_index[agent["role_id"]]
        agent_fm         = (agent.get("bedrock", {}).get("model_id") or
                            agent.get("bedrock", {}).get("foundation_model", "")).strip()
        foundation_model = agent_fm or cfg["foundation_model"]
        instruction      = _build_instruction(agent, role_obj, gemini_enabled)

        existing_id  = agent.get("bedrock", {}).get("agentId",  "").strip()
        existing_ali = agent.get("bedrock", {}).get("aliasId",  "").strip()

        if existing_id and existing_ali:
            # Agent already provisioned — sync instruction + model + action group
            log.info(f"  SYNC {agent['name']} ({existing_id}) — updating instruction")
            try:
                current = bedrock.get_agent(agentId=existing_id)["agent"]
                _update_agent_with_retry(
                    bedrock,
                    agent_id=existing_id,
                    agentName=current["agentName"],
                    agentResourceRoleArn=current["agentResourceRoleArn"],
                    foundationModel=foundation_model,
                    instruction=instruction,
                    description=current.get("description", ""),
                )
                if gemini_enabled:
                    _attach_gemini_action_group(bedrock, existing_id, gemini_lambda_arn)
                _prepare_agent_and_wait(bedrock, existing_id)
                log.info(f"  ✓ SYNC {agent['name']} — instruction updated")
            except Exception as e:
                log.warning(f"  WARN could not sync {agent['name']}: {e}")

            # Ensure per-model aliases exist
            model_aliases = agent.get("bedrock", {}).get("model_aliases", {})
            if model_aliases:
                updated_aliases = _provision_model_aliases(
                    bedrock, existing_id, foundation_model, existing_ali,
                    model_aliases, instruction, agent["name"],
                )
                output_team["agents"][i].setdefault("bedrock", {})["model_aliases"] = updated_aliases
            continue

        # Not yet provisioned — create or recover
        existing = find_existing_agent(bedrock, agent["name"])
        if existing:
            aid = existing["agentId"]
            try:
                current = bedrock.get_agent(agentId=aid)["agent"]
                _update_agent_with_retry(
                    bedrock,
                    agent_id=aid,
                    agentName=current["agentName"],
                    agentResourceRoleArn=current["agentResourceRoleArn"],
                    foundationModel=foundation_model,
                    instruction=instruction,
                    description=current.get("description", ""),
                )
                if gemini_enabled:
                    _attach_gemini_action_group(bedrock, aid, gemini_lambda_arn)
                _prepare_agent_and_wait(bedrock, aid)
            except Exception as e:
                log.warning(f"  WARN could not update recovered agent {agent['name']}: {e}")
            als  = bedrock.list_agent_aliases(agentId=aid).get("agentAliasSummaries", [])
            alid = (als[0]["agentAliasId"] if als else
                    bedrock.create_agent_alias(agentId=aid, agentAliasName="live")
                           ["agentAlias"]["agentAliasId"])
        else:
            aid, alid = create_bedrock_agent(
                client=bedrock, agent_id_slug=agent["id"], agent_name=agent["name"],
                goal_template=agent["goal_template"], schema_ref=agent["schema_ref"],
                role_obj=role_obj, bedrock_role_arn=cfg["bedrock_role_arn"],
                foundation_model=foundation_model,
                gemini_lambda_arn=gemini_lambda_arn if gemini_enabled else "",
            )

        bdrock = output_team["agents"][i].get("bedrock", {})
        bdrock.update({"agentId": aid, "aliasId": alid})

        # Create per-model aliases
        model_aliases = agent.get("bedrock", {}).get("model_aliases", {})
        if model_aliases:
            updated_aliases = _provision_model_aliases(
                bedrock, aid, foundation_model, alid,
                model_aliases, instruction, agent["name"],
            )
            bdrock["model_aliases"] = updated_aliases

        output_team["agents"][i]["bedrock"] = bdrock
        log.info(f"  ✓ {agent['name']} → agentId={aid} aliasId={alid}")

    # Push back to S3 under teams_prefix
    key    = f"{cfg['teams_prefix']}/{team_name}/{version}/team.json"
    s3_uri = _s3_put(cfg, key, output_team)
    return {"success": True, "s3_uri": s3_uri, "errors": [], "team": output_team}


# ─────────────────────────────────────────────────────────────────────────────
# Agent delete helper
# ─────────────────────────────────────────────────────────────────────────────

def _delete_bedrock_agents(bedrock, agent_names: list) -> dict:
    deleted, not_found, errors = [], [], {}
    for raw in agent_names:
        try:
            summary = find_existing_agent(bedrock, raw)
            if not summary:
                not_found.append(raw)
                continue
            aid = summary["agentId"]
            try:
                aliases = bedrock.list_agent_aliases(agentId=aid).get("agentAliasSummaries", [])
            except Exception as e:
                log.warning(f"Could not list aliases for {aid}: {e}")
                aliases = []
            for alias in aliases:
                try:
                    bedrock.delete_agent_alias(agentId=aid, agentAliasId=alias["agentAliasId"])
                except Exception as e:
                    log.warning(f"Could not delete alias {alias['agentAliasId']}: {e}")
            bedrock.delete_agent(agentId=aid, skipResourceInUseCheck=True)
            deleted.append(raw)
            log.info(f"  ✓ Deleted: {sanitise_agent_name(raw)} ({aid})")
        except Exception as e:
            errors[raw] = str(e)
    return {"deleted": deleted, "not_found": not_found, "errors": errors}


# ─────────────────────────────────────────────────────────────────────────────
# Team rebuild helper — called after any role / dept / agent update
# ─────────────────────────────────────────────────────────────────────────────

def _rebuild_affected_teams(cfg: dict, match_fn) -> dict:
    """
    Scan all S3 teams. For every team where match_fn(team_data) returns True,
    re-run _provision_team (which syncs instructions on existing agents and
    creates any missing ones) and push the updated team.json back to S3.

    match_fn receives the raw team_data dict and returns True/False.

    Returns:
        {
            "rebuilt":  [team_name, ...],
            "skipped":  [team_name, ...],   # match_fn returned False
            "errors":   {team_name: error_msg, ...},
        }
    """
    try:
        role_index = build_role_index(_load_roles(cfg))
        dept_index = build_dept_index(_load_depts(cfg))
    except Exception as e:
        return {"rebuilt": [], "skipped": [], "errors": {"_load": str(e)}}

    team_map = _scan_s3_teams(cfg)
    rebuilt, skipped, errors = [], [], {}

    for team_name, versions in team_map.items():
        latest_ver = _latest_version(versions)
        try:
            team_data = _s3_get(cfg, versions[latest_ver])
        except Exception as e:
            errors[team_name] = f"S3 load failed: {e}"
            continue

        if not match_fn(team_data):
            skipped.append(team_name)
            continue

        try:
            r = _provision_team(team_data, team_name, latest_ver,
                                role_index, dept_index, cfg)
            if r["success"]:
                rebuilt.append(team_name)
            else:
                errors[team_name] = r.get("errors", ["unknown error"])
        except Exception as e:
            errors[team_name] = str(e)

    log.info(f"  Team rebuild — rebuilt:{rebuilt} skipped:{len(skipped)} errors:{list(errors.keys())}")
    return {"rebuilt": rebuilt, "skipped": skipped, "errors": errors}


# ─────────────────────────────────────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────────────────────────────────────

def handle_agents(method, path_param, body, cfg) -> dict:
    bedrock = _bedrock(cfg)

    if method == "GET" and not path_param:
        try:
            paginator = bedrock.get_paginator("list_agents")
            agents = [s for page in paginator.paginate() for s in page.get("agentSummaries", [])]
            return _ok({"agents": agents, "count": len(agents)})
        except Exception as e:
            return _server_error(e)

    if method == "GET" and path_param:
        try:
            summary = find_existing_agent(bedrock, path_param)
            if not summary:
                return _not_found(f"Agent '{path_param}' not found.")
            aid     = summary["agentId"]
            details = bedrock.get_agent(agentId=aid)["agent"]
            aliases = bedrock.list_agent_aliases(agentId=aid).get("agentAliasSummaries", [])

            # Find original_goal_template from S3 team files so callers
            # can PUT back the raw goal without re-sending the full composite instruction
            original_goal_template = None
            agent_name_sanitised   = details["agentName"]
            try:
                team_map = _scan_s3_teams(cfg)
                for versions in team_map.values():
                    latest_ver = _latest_version(versions)
                    try:
                        team_data = _s3_get(cfg, versions[latest_ver])
                        for a in team_data.get("agents", []):
                            if sanitise_agent_name(a["name"]) == agent_name_sanitised:
                                original_goal_template = (
                                    a.get("original_goal_template") or a.get("goal_template")
                                )
                                break
                    except Exception:
                        continue
                    if original_goal_template:
                        break
            except Exception as e:
                log.warning(f"Could not resolve original_goal_template for {path_param}: {e}")

            return _ok({
                "agent":                 details,
                "aliases":               aliases,
                "original_goal_template": original_goal_template,
            })
        except Exception as e:
            return _server_error(e)

    if method == "POST":
        missing = [f for f in ["name", "role_id", "goal_template", "schema_ref"] if not body.get(f)]
        if missing:
            return _bad_request(f"Missing required fields: {missing}")
        try:
            role_index = build_role_index(_load_roles(cfg))
            role_obj   = role_index.get(body["role_id"])
            if not role_obj:
                return _bad_request(f"role_id '{body['role_id']}' not found.")
            aid, alid = create_bedrock_agent(
                client=bedrock, agent_id_slug=body.get("id", body["name"]),
                agent_name=body["name"], goal_template=body["goal_template"],
                schema_ref=body["schema_ref"], role_obj=role_obj,
                bedrock_role_arn=cfg["bedrock_role_arn"],
                foundation_model=body.get("foundation_model", cfg["foundation_model"]),
            )
            return _created({"agentId": aid, "aliasId": alid,
                             "name": sanitise_agent_name(body["name"])})
        except Exception as e:
            return _server_error(e)

    if method == "PUT" and path_param:
        if not body:
            return _bad_request("Request body required.")
        try:
            summary = find_existing_agent(bedrock, path_param)
            if not summary:
                return _not_found(f"Agent '{path_param}' not found.")
            aid     = summary["agentId"]
            current = bedrock.get_agent(agentId=aid)["agent"]
            # Resolve body — APIGW proxy may deliver body as a JSON string
            resolved = body
            if isinstance(body, str):
                try:
                    resolved = json.loads(body)
                except Exception:
                    return _bad_request("Request body is not valid JSON.")
            updated_fm          = resolved.get("foundation_model") or resolved.get("foundationModel") or current["foundationModel"]
            updated_instruction = resolved.get("instruction",      current.get("instruction", ""))
            updated_description = resolved.get("description",      current.get("description", ""))
            _update_agent_with_retry(
                bedrock,
                agent_id=aid,
                agentName=current["agentName"],
                agentResourceRoleArn=current["agentResourceRoleArn"],
                foundationModel=updated_fm,
                instruction=updated_instruction,
                description=updated_description,
            )
            _prepare_agent_and_wait(bedrock, aid)
            return _ok({
                "updated":    True,
                "agentId":    aid,
                "fields_set": {k: True for k in resolved if k in
                               ("instruction", "description", "foundation_model", "foundationModel")},
            })
        except Exception as e:
            return _server_error(e)

    if method == "DELETE":
        agent_names = body.get("agent_names", [])
        if not agent_names or not isinstance(agent_names, list):
            return _bad_request("'agent_names' list is required.")
        result = _delete_bedrock_agents(bedrock, agent_names)
        return _resp(200 if not result["errors"] else 500, result)

    return _method_not_allowed(method)


# ─────────────────────────────────────────────────────────────────────────────
# TEAMS
# ─────────────────────────────────────────────────────────────────────────────

def handle_teams(method, path_param, body, cfg) -> dict:

    if method == "GET" and not path_param:
        try:
            team_map = _scan_s3_teams(cfg)
            teams = []
            for name, versions in team_map.items():
                latest_ver = _latest_version(versions)
                try:
                    data        = _s3_get(cfg, versions[latest_ver])
                    provisioned = not needs_provisioning(data)[0]
                    agent_count = len(data.get("agents", []))
                    team_id     = data.get("team", {}).get("team_id") or generate_team_id(name)
                    owner       = data.get("team", {}).get("owner", "")
                except Exception:
                    provisioned = agent_count = team_id = owner = None
                teams.append({
                    "name":           name,
                    "latest_version": latest_ver,
                    "all_versions":   sorted(versions.keys()),
                    "team_id":        team_id,
                    "owner":          owner,
                    "agent_count":    agent_count,
                    "provisioned":    provisioned,
                    "latest_s3_key":  versions[latest_ver],
                })
            return _ok({"teams": teams, "count": len(teams)})
        except Exception as e:
            return _server_error(e)

    if method == "GET" and path_param:
        try:
            team_map = _scan_s3_teams(cfg)
            if path_param not in team_map:
                return _not_found(f"Team '{path_param}' not found in S3.")
            versions   = team_map[path_param]
            latest_ver = _latest_version(versions)
            data       = _s3_get(cfg, versions[latest_ver])
            return _ok({
                "team":     data,
                "version":  latest_ver,
                "s3_key":   versions[latest_ver],
                "versions": sorted(versions.keys()),
            })
        except Exception as e:
            return _server_error(e)

    if method == "POST":
        dry_run = bool(body.get("dry_run", False))
        try:
            role_index = build_role_index(_load_roles(cfg))
            dept_index = build_dept_index(_load_depts(cfg))
        except KeyError as e:
            return _server_error(str(e))

        team_map = _scan_s3_teams(cfg)
        if not team_map:
            return _ok({"results": {}, "errors": [], "message": "No teams found in S3."})

        results, all_errors = {}, []
        for team_name, versions in team_map.items():
            latest_ver = _latest_version(versions)
            try:
                team_data = _s3_get(cfg, versions[latest_ver])
            except Exception as e:
                results[team_name] = {"success": False, "errors": [f"S3 load failed: {e}"]}
                all_errors.append(str(e))
                continue

            work_needed, total, missing = needs_provisioning(team_data)
            if not work_needed:
                log.info(f"  SKIP {team_name} — all {total} agent(s) already provisioned.")
                results[team_name] = {"success": True, "skipped": True,
                                      "reason": "all agents already provisioned"}
                continue

            log.info(f"  QUEUE {team_name} — {missing}/{total} agent(s) need IDs.")
            try:
                r = _provision_team(team_data, team_name, latest_ver,
                                    role_index, dept_index, cfg, dry_run)
            except Exception as e:
                r = {"success": False, "s3_uri": None, "errors": [str(e)]}
            results[team_name] = {k: v for k, v in r.items() if k != "team"}
            if not r["success"]:
                all_errors.extend(r.get("errors", []))

        return _resp(200 if not all_errors else 500,
                     {"results": results, "errors": all_errors})

    if method == "PUT" and path_param:
        if not body:
            return _bad_request("Request body required.")
        try:
            team_map = _scan_s3_teams(cfg)
            if path_param not in team_map:
                return _not_found(f"Team '{path_param}' not found in S3.")
            versions   = team_map[path_param]
            latest_ver = _latest_version(versions)
            team_data  = _s3_get(cfg, versions[latest_ver])

            updated = copy.deepcopy(team_data)
            for k, v in body.items():
                if k in updated and isinstance(v, dict) and isinstance(updated[k], dict):
                    updated[k].update(v)
                else:
                    updated[k] = v

            role_index = build_role_index(_load_roles(cfg))
            dept_index = build_dept_index(_load_depts(cfg))

            work_needed, total, missing = needs_provisioning(updated)
            if work_needed:
                r = _provision_team(updated, path_param, latest_ver,
                                    role_index, dept_index, cfg)
            else:
                key    = f"{cfg['teams_prefix']}/{path_param}/{latest_ver}/team.json"
                s3_uri = _s3_put(cfg, key, updated)
                r = {"success": True, "s3_uri": s3_uri, "errors": [],
                     "skipped": "all agents already provisioned"}

            return (_ok if r["success"] else _server_error)({
                "updated":  True,
                "version":  latest_ver,
                "s3_uri":   r.get("s3_uri"),
                "errors":   r.get("errors", []),
            })
        except Exception as e:
            return _server_error(e)

    if method == "DELETE":
        if not path_param:
            return _bad_request("Team name required. Use DELETE /teams/{team_name}.")
        try:
            team_map = _scan_s3_teams(cfg)
            if path_param not in team_map:
                return _not_found(f"Team '{path_param}' not found in S3.")
            versions = team_map[path_param]

            # Collect all agent names across every version (deduplicated)
            agent_names = set()
            for key in versions.values():
                try:
                    data = _s3_get(cfg, key)
                    for agent in data.get("agents", []):
                        agent_names.add(agent["name"])
                except Exception as e:
                    log.warning(f"Could not load {key} for agent names: {e}")

            # Delete Bedrock agents
            agent_result = _delete_bedrock_agents(_bedrock(cfg), list(agent_names))

            # Delete all S3 versions
            deleted_keys, s3_errors = [], []
            for key in versions.values():
                try:
                    _s3_delete(cfg, key)
                    deleted_keys.append(key)
                except Exception as e:
                    s3_errors.append(f"Failed to delete {key}: {e}")

            success = not agent_result["errors"] and not s3_errors
            return _resp(200 if success else 500, {
                "team":         path_param,
                "agents":       agent_result,
                "deleted_keys": deleted_keys,
                "s3_errors":    s3_errors,
            })
        except Exception as e:
            return _server_error(e)

    return _method_not_allowed(method)


# ─────────────────────────────────────────────────────────────────────────────
# ROLES
# ─────────────────────────────────────────────────────────────────────────────

def handle_roles(method, path_param, body, cfg) -> dict:

    if method == "GET" and not path_param:
        try:
            data = _load_roles(cfg)
            return _ok({"roles": data["roles"], "count": len(data["roles"]),
                        "s3_key": cfg["roles_key"]})
        except KeyError as e:
            return _not_found(str(e))
        except Exception as e:
            return _server_error(e)

    if method == "GET" and path_param:
        try:
            data  = _load_roles(cfg)
            match = next((r for r in data["roles"] if r["role_id"] == path_param), None)
            return _ok(match) if match else _not_found(f"Role '{path_param}' not found.")
        except KeyError as e:
            return _not_found(str(e))
        except Exception as e:
            return _server_error(e)

    if method == "POST":
        missing = [f for f in ["role_id", "title", "slug", "department_id",
                                "schema_ref", "agent_config"] if not body.get(f)]
        if missing:
            return _bad_request(f"Missing required fields: {missing}")
        try:
            data = _load_roles(cfg)
            if any(r["role_id"] == body["role_id"] for r in data["roles"]):
                return _bad_request(f"role_id '{body['role_id']}' already exists.")
            data["roles"].append(body)
            data["meta"]["version"] = _bump_version(data["meta"].get("version", "v1"))
            s3_uri = _save_roles(cfg, data)
            return _created({"created": True, "role_id": body["role_id"], "s3_uri": s3_uri})
        except KeyError as e:
            return _not_found(str(e))
        except Exception as e:
            return _server_error(e)

    if method == "PUT" and path_param:
        if not body:
            return _bad_request("Request body required.")
        try:
            data = _load_roles(cfg)
            idx  = next((i for i, r in enumerate(data["roles"]) if r["role_id"] == path_param), None)
            if idx is None:
                return _not_found(f"Role '{path_param}' not found.")
            # Deep-merge nested dicts (e.g. agent_config) instead of replacing them
            def _deep_merge(base: dict, patch: dict) -> dict:
                result = copy.deepcopy(base)
                for k, v in patch.items():
                    if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                        result[k] = _deep_merge(result[k], v)
                    else:
                        result[k] = v
                return result
            data["roles"][idx] = _deep_merge(data["roles"][idx], body)
            data["meta"]["version"] = _bump_version(data["meta"].get("version", "v1"))
            s3_uri = _save_roles(cfg, data)
            # Rebuild every team that uses this role_id
            rebuild = _rebuild_affected_teams(
                cfg,
                lambda td, rid=path_param: any(
                    a.get("role_id") == rid for a in td.get("agents", [])
                )
            )
            return _ok({"updated": True, "role_id": path_param, "s3_uri": s3_uri,
                        "role": data["roles"][idx], "teams_rebuilt": rebuild})
        except KeyError as e:
            return _not_found(str(e))
        except Exception as e:
            return _server_error(e)

    return _method_not_allowed(method)


# ─────────────────────────────────────────────────────────────────────────────
# DEPARTMENTS
# ─────────────────────────────────────────────────────────────────────────────

def handle_departments(method, path_param, body, cfg) -> dict:

    if method == "GET" and not path_param:
        try:
            data = _load_depts(cfg)
            return _ok({"departments": data["departments"], "count": len(data["departments"]),
                        "s3_key": cfg["depts_key"]})
        except KeyError as e:
            return _not_found(str(e))
        except Exception as e:
            return _server_error(e)

    if method == "GET" and path_param:
        try:
            data  = _load_depts(cfg)
            match = next((d for d in data["departments"] if d["department_id"] == path_param), None)
            return _ok(match) if match else _not_found(f"Department '{path_param}' not found.")
        except KeyError as e:
            return _not_found(str(e))
        except Exception as e:
            return _server_error(e)

    if method == "POST":
        missing = [f for f in ["department_id", "name", "slug", "description",
                                "allowed_roles", "allowed_schemas"] if not body.get(f)]
        if missing:
            return _bad_request(f"Missing required fields: {missing}")
        try:
            data = _load_depts(cfg)
            if any(d["department_id"] == body["department_id"] for d in data["departments"]):
                return _bad_request(f"department_id '{body['department_id']}' already exists.")
            data["departments"].append(body)
            data["meta"]["version"] = _bump_version(data["meta"].get("version", "v1"))
            s3_uri = _save_depts(cfg, data)
            return _created({"created": True, "department_id": body["department_id"],
                             "s3_uri": s3_uri})
        except KeyError as e:
            return _not_found(str(e))
        except Exception as e:
            return _server_error(e)

    if method == "PUT" and path_param:
        if not body:
            return _bad_request("Request body required.")
        try:
            data = _load_depts(cfg)
            idx  = next((i for i, d in enumerate(data["departments"])
                         if d["department_id"] == path_param), None)
            if idx is None:
                return _not_found(f"Department '{path_param}' not found.")
            # Merge list fields (allowed_roles, allowed_schemas) — additive, not replacing
            for list_field in ("allowed_roles", "allowed_schemas"):
                if list_field in body:
                    existing = set(data["departments"][idx].get(list_field, []))
                    existing.update(body.pop(list_field))
                    data["departments"][idx][list_field] = sorted(existing)
            # Deep-merge remaining fields
            def _deep_merge(base: dict, patch: dict) -> dict:
                result = copy.deepcopy(base)
                for k, v in patch.items():
                    if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                        result[k] = _deep_merge(result[k], v)
                    else:
                        result[k] = v
                return result
            data["departments"][idx] = _deep_merge(data["departments"][idx], body)
            data["meta"]["version"] = _bump_version(data["meta"].get("version", "v1"))
            s3_uri = _save_depts(cfg, data)
            # Rebuild every team that uses this department_id
            rebuild = _rebuild_affected_teams(
                cfg,
                lambda td, did=path_param: any(
                    a.get("department_id") == did for a in td.get("agents", [])
                )
            )
            return _ok({"updated": True, "department_id": path_param, "s3_uri": s3_uri,
                        "department": data["departments"][idx], "teams_rebuilt": rebuild})
        except KeyError as e:
            return _not_found(str(e))
        except Exception as e:
            return _server_error(e)

    return _method_not_allowed(method)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

_ROUTES = {
    "agents":      handle_agents,
    "teams":       handle_teams,
    "roles":       handle_roles,
    "departments": handle_departments,
}

def handler(event: dict, context) -> dict:
    log.info(f"Event: {json.dumps(event, default=str)}")
    try:
        body = _parse_body(event)
    except ValueError as e:
        return _bad_request(str(e))
    try:
        cfg = get_config()
    except ValueError as e:
        return _server_error(str(e))

    method, resource, path_param = _extract_route(event)
    log.info(f"Route: {method} /{resource}" + (f"/{path_param}" if path_param else ""))

    route_fn = _ROUTES.get(resource)
    if not route_fn:
        return _resp(404, {
            "error":  f"Unknown resource '/{resource}'.",
            "routes": list(_ROUTES.keys()),
        })
    return route_fn(method, path_param, body, cfg)
