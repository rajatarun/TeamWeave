import json
import os
import uuid
from typing import Any, Dict, Optional

import boto3

from .bedrock_invoke import invoke_agent
from .config_loader import load_team_config
from .db import DbDao
from .gemini import gemini_research_brief
from .json_utils import build_standard_response, extract_json_payload
from .logger import get_logger
from .models import StepFailed
from .profile_context import get_owner_profile_context
from .prompt_builder import build_prompt
from .rag import get_rag_context
from .storage import save_artifact
from .structured_transform import transform_json_to_schema

log = get_logger("worker_handler")
lambda_client = boto3.client("lambda")


# ASSUMPTION: The Step Functions execution input preserves the existing POST body contract:
# {"team": "...", "version": "...", "request": {...}}.


def _find_agent(team_cfg, agent_id: str):
    for a in team_cfg.agents:
        if a.id == agent_id:
            return a
    return None


def _load_step_schema(team_raw: Dict[str, Any], schema_ref: str) -> Optional[Dict[str, Any]]:
    schemas = team_raw.get("schemas") or {}
    schema_cfg = schemas.get(schema_ref) or {}
    if not isinstance(schema_cfg, dict):
        return None

    if isinstance(schema_cfg.get("schema"), dict):
        return schema_cfg["schema"]

    path = schema_cfg.get("path")
    if not path:
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        log.warning("unable_to_load_step_schema", extra={"schema_ref": schema_ref, "path": path})
        return None


def _build_transform_fallback(raw_text: str, error: Exception) -> Dict[str, Any]:
    return build_standard_response(raw_text, f"schema transformation failed: {error}")


def _resolve_supervisor_step_id(team_cfg, workflow: list) -> Optional[str]:
    """
    Determine which step acts as the supervisor (brief source).

    Priority:
      1. First agent in the workflow with agentRole == "supervisor"
      2. Fallback: first step in the workflow
    """
    for step_def in workflow:
        agent = _find_agent(team_cfg, step_def["step"])
        if getattr(agent, "role", None) == "supervisor":
            return step_def["step"]

    # Fallback to first step
    return workflow[0]["step"] if workflow else None


def _build_step_inputs(
    step_def: Dict[str, Any],
    request_obj: Dict[str, Any],
    owner: str,
    rag_context: str,
    owner_profile_context: str,
    gemini_brief: str,
    outputs: Dict[str, Any],
    supervisor_brief: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the full input context for a step.

    Base context keys are always injected. All prior step outputs are merged in
    so any step can reference any predecessor without requiring explicit `inputs`
    declaration in the workflow YAML.

    Supervisor brief is injected as a convenience alias once the supervisor step
    has completed. If no agentRole: supervisor is declared, the first step's
    output is used as the fallback supervisor brief.

    Explicit `inputs` entries in the step definition are still honoured; a
    warning is logged if a declared input hasn't been produced yet.
    """
    base: Dict[str, Any] = {
        "request": request_obj,
        "owner": owner,
        "rag_context": rag_context,
        "owner_profile_context": owner_profile_context,
        "gemini_brief": gemini_brief,
    }

    # Merge all completed step outputs so every downstream step can access
    # any predecessor without explicit YAML wiring.
    base.update(outputs)

    # Supervisor brief — available once the supervisor step has completed.
    if supervisor_brief:
        base["supervisor"] = supervisor_brief

    # Explicit `inputs` declarations are validated for early warning.
    for inp in step_def.get("inputs", []):
        key = inp.split(".")[0] if inp.endswith(".output") else inp
        if key not in base:
            log.warning(
                "step_input_not_yet_available inp=%s available=%s",
                inp,
                list(outputs.keys()),
            )
            base[key] = {}

    return base


def run_team_pipeline(
    team: str,
    version: str,
    request_obj: Dict[str, Any],
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    run_id = run_id or str(uuid.uuid4())
    team_cfg, team_raw = load_team_config(team, version)
    dao = DbDao.from_team_config(team_raw)

    owner = (
        (team_raw.get("team") or {}).get("owner")
        or request_obj.get("owner")
        or "Tarun Raja"
    )

    dao.put_run_meta(
        run_id,
        "RUNNING",
        {"team": team, "version": version, "owner": owner, "request": request_obj},
    )

    try:
        rag_context = get_rag_context(
            request_obj,
            {"rag": team_cfg.globals.rag, "features": team_cfg.globals.features},
            owner=owner,
            dao=dao,
        )
    except Exception:
        log.exception(
            "rag_context_unavailable_proceeding_without_rag",
            extra={"run_id": run_id, "owner": owner},
        )
        rag_context = ""

    owner_profile_context = get_owner_profile_context(request_obj, team_raw, owner)

    completed_topics = rag_context if rag_context.startswith("COMPLETED_TASKS_HISTORY") else ""
    gemini_brief = gemini_research_brief(
        {"features": team_cfg.globals.features},
        request_obj,
        completed_topics=completed_topics,
    )

    outputs: Dict[str, Any] = {}
    supervisor_brief: Dict[str, Any] = {}

    workflow = team_raw.get("workflow") or []

    supervisor_step_id = _resolve_supervisor_step_id(team_cfg, workflow)
    log.info("supervisor_step_resolved step=%s run_id=%s", supervisor_step_id, run_id)

    for step_def in workflow:
        step_id = step_def["step"]
        agent = _find_agent(team_cfg, step_id)
        if not agent:
            raise StepFailed(step_id, f"Agent not found for step '{step_id}'")

        step_inputs = _build_step_inputs(
            step_def,
            request_obj,
            owner,
            rag_context,
            owner_profile_context,
            gemini_brief,
            outputs,
            supervisor_brief,
        )

        prompt = build_prompt(
            team_cfg,
            agent,
            step_inputs,
            supervisor_brief,
            rag_context,
            owner_profile_context,
            gemini_brief,
        )

        log.info(
            "agent_prompt_built step=%s run_id=%s prompt_len=%d",
            step_id,
            run_id,
            len(prompt),
        )

        raw_text = invoke_agent(agent.bedrock.agentId, agent.bedrock.aliasId, run_id, prompt)

        try:
            out_json = extract_json_payload(raw_text)
        except Exception as e:
            log.warning(
                "json_parse_failed_coercing_to_payload step=%s run_id=%s raw_len=%d",
                step_id,
                run_id,
                len(raw_text),
            )
            out_json = build_standard_response(raw_text, str(e))

        step_schema = _load_step_schema(team_raw, agent.schema_ref)
        if step_schema:
            try:
                out_json = transform_json_to_schema(out_json, step_schema)
            except Exception as transform_error:
                log.exception(
                    "schema_transform_failed step=%s run_id=%s schema_ref=%s",
                    step_id,
                    run_id,
                    agent.schema_ref,
                )
                out_json = _build_transform_fallback(raw_text, transform_error)

        artifact_uri = save_artifact(run_id, step_id, out_json)
        dao.put_step(
            run_id, step_id, "SUCCEEDED", step_inputs, out_json, error=None, artifact_uri=artifact_uri
        )

        outputs[step_id] = out_json

        if step_id == supervisor_step_id:
            supervisor_brief = out_json
            log.info(
                "supervisor_brief_captured step=%s run_id=%s keys=%s",
                step_id,
                run_id,
                list(supervisor_brief.keys()),
            )

        if step_id == "advisor" and isinstance(out_json.get("daily_tasks"), list):
            dao.put_tasks(owner=owner, tasks=out_json["daily_tasks"], source_run_id=run_id)

    dao.put_run_meta(
        run_id,
        "SUCCEEDED",
        {"team": team, "version": version, "owner": owner, "steps": list(outputs.keys())},
    )
    return {"run_id": run_id, "status": "SUCCEEDED", "steps": outputs, "owner": owner}


def handler(event, context):
    log.info("worker_handler_received_event", extra={"event": event})

    if event.get("operation") in {"provision", "agent_management"}:
        function_name = os.environ.get("PROVISION_FUNCTION_NAME")
        if not function_name:
            raise ValueError("PROVISION_FUNCTION_NAME env var not set")

        proxy_path = event.get("path") or "/teams"
        invoke_payload = {
            "httpMethod": event.get("method", "POST"),
            "path": proxy_path,
            "rawPath": proxy_path,
            "queryStringParameters": event.get("query") or {},
            "body": json.dumps(event.get("body") or {}),
        }
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(invoke_payload).encode("utf-8"),
        )
        payload_bytes = response["Payload"].read()
        payload = json.loads(payload_bytes.decode("utf-8") or "{}")

        status_code = int(payload.get("statusCode", 500))
        raw_body = payload.get("body")
        body = json.loads(raw_body) if isinstance(raw_body, str) else raw_body

        if status_code >= 400:
            raise ValueError(f"provision request failed [{status_code}]: {body}")

        return {
            "status": "SUCCEEDED",
            "operation": "agent_management",
            "method": event.get("method", "POST"),
            "path": proxy_path,
            "result": body,
        }

    team = event.get("team")
    version = event.get("version")
    request_obj = event.get("request") or {}
    if not team or not version:
        raise ValueError("team and version are required in the event payload")

    return run_team_pipeline(team, version, request_obj, run_id=event.get("run_id"))
