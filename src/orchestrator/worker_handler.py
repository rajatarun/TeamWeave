import json
import uuid
from typing import Any, Dict, Optional

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


def run_team_pipeline(team: str, version: str, request_obj: Dict[str, Any]) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    team_cfg, team_raw = load_team_config(team, version)
    dao = DbDao.from_team_config(team_raw)

    owner = (team_raw.get("team") or {}).get("owner") or request_obj.get("owner") or "Tarun Raja"

    dao.put_run_meta(run_id, "RUNNING", {"team": team, "version": version, "owner": owner, "request": request_obj})

    try:
        rag_context = get_rag_context(
            request_obj,
            {"rag": team_cfg.globals.rag, "features": team_cfg.globals.features},
            owner=owner,
            dao=dao,
        )
    except Exception:
        log.exception("rag_context_unavailable_proceeding_without_rag", extra={"run_id": run_id, "owner": owner})
        rag_context = ""
    owner_profile_context = get_owner_profile_context(request_obj, team_raw, owner)

    completed_topics = ""
    if rag_context.startswith("COMPLETED_TASKS_HISTORY"):
        completed_topics = rag_context
    gemini_brief = gemini_research_brief(
        {"features": team_cfg.globals.features},
        request_obj,
        completed_topics=completed_topics,
    )

    outputs: Dict[str, Any] = {}
    director_brief: Dict[str, Any] = {}

    workflow = team_raw.get("workflow") or []
    step_index = 0

    while step_index < len(workflow):
        step_def = workflow[step_index]
        step_id = step_def["step"]
        agent = _find_agent(team_cfg, step_id)
        if not agent:
            raise StepFailed(step_id, f"Agent not found for step {step_id}")

        step_inputs: Dict[str, Any] = {
            "request": request_obj,
            "owner": owner,
            "rag_context": rag_context,
            "owner_profile_context": owner_profile_context,
            "gemini_brief": gemini_brief,
        }
        for inp in step_def.get("inputs", []):
            if inp == "request":
                step_inputs["request"] = request_obj
            elif inp == "rag_context":
                step_inputs["rag_context"] = rag_context
            elif inp == "owner_profile_context":
                step_inputs["owner_profile_context"] = owner_profile_context
            elif inp.endswith(".output"):
                key = inp.split(".")[0]
                step_inputs[key] = outputs.get(key, {})
            else:
                step_inputs[inp] = outputs.get(inp, {})

        prompt = build_prompt(
            team_cfg,
            agent,
            step_inputs,
            director_brief,
            rag_context,
            owner_profile_context,
            gemini_brief,
        )

        log.info("agent_prompt_built step=%s run_id=%s prompt=%s", step_id, run_id, prompt)

        raw_text = invoke_agent(agent.bedrock.agentId, agent.bedrock.aliasId, run_id, prompt)

        try:
            out_json = extract_json_payload(raw_text)
        except Exception as e:
            log.warning(
                "json_parse_failed_coercing_to_payload step=%s run_id=%s raw_response=%s",
                step_id,
                run_id,
                raw_text,
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
        dao.put_step(run_id, step_id, "SUCCEEDED", step_inputs, out_json, error=None, artifact_uri=artifact_uri)

        outputs[step_id] = out_json
        if step_id == "director":
            director_brief = out_json

        if step_id == "advisor" and isinstance(out_json.get("daily_tasks"), list):
            dao.put_tasks(owner=owner, tasks=out_json.get("daily_tasks"), source_run_id=run_id)

        step_index += 1

    dao.put_run_meta(run_id, "SUCCEEDED", {"team": team, "version": version, "owner": owner, "steps": list(outputs.keys())})
    return {"run_id": run_id, "status": "SUCCEEDED", "steps": outputs, "owner": owner}


def handler(event, context):
    team = event.get("team")
    version = event.get("version")
    request_obj = event.get("request") or {}
    if not team or not version:
        raise ValueError("team and version required")
    return run_team_pipeline(team, version, request_obj)
