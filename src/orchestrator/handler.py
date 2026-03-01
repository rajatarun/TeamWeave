import json
import uuid
from typing import Any, Dict, Optional

from .logger import get_logger
from .config_loader import load_team_config
from .prompt_builder import build_prompt
from .bedrock_invoke import invoke_agent
from .schema_validate import validate_output, format_validation_error
from .storage import save_artifact
from .rag import get_rag_context
from .gemini import gemini_research_brief
from .profile_context import get_owner_profile_context
from .models import StepFailed
from .db import DbDao

log = get_logger("handler")


def _cors() -> Dict[str, str]:
    return {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
        "access-control-allow-headers": "content-type",
        "access-control-allow-methods": "POST,GET,OPTIONS",
    }


def _resp(code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {"statusCode": code, "headers": _cors(), "body": json.dumps(body, ensure_ascii=False, default=str)}


def _method(event: Dict[str, Any]) -> str:
    return (event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod") or "").upper()


def _path(event: Dict[str, Any]) -> str:
    return event.get("rawPath") or event.get("path") or ""


def _qs(event: Dict[str, Any]) -> Dict[str, Any]:
    return event.get("queryStringParameters") or {}


def _json_body(event: Dict[str, Any]) -> Dict[str, Any]:
    b = event.get("body")
    if not b:
        return {}
    try:
        return json.loads(b) if isinstance(b, str) else b
    except Exception:
        return {}


def _load_schema_objects(team_raw: Dict[str, Any]) -> Dict[str, Any]:
    schemas = {}
    schema_map = (team_raw.get("schemas") or {})
    for ref, meta in schema_map.items():
        path = (meta.get("path") or "").lstrip("/")
        if not path:
            continue
        with open(path, "r", encoding="utf-8") as f:
            schemas[ref] = json.load(f)
    return schemas


def _find_agent(team_cfg, agent_id: str):
    for a in team_cfg.agents:
        if a.id == agent_id:
            return a
    return None


def _dao_from_optional_team(team: Optional[str], version: Optional[str]) -> DbDao:
    if team and version:
        _, team_raw = load_team_config(team, version)
        return DbDao.from_team_config(team_raw)
    return DbDao.from_team_config({})


def _run_team_pipeline(team: str, version: str, request_obj: Dict[str, Any]) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    team_cfg, team_raw = load_team_config(team, version)
    dao = DbDao.from_team_config(team_raw)
    schema_objs = _load_schema_objects(team_raw)

    owner = (team_raw.get("team") or {}).get("owner") or request_obj.get("owner") or "Tarun Raja"

    dao.put_run_meta(run_id, "RUNNING", {"team": team, "version": version, "owner": owner, "request": request_obj})

    rag_context = get_rag_context(
        request_obj,
        {"rag": team_cfg.globals.rag, "features": team_cfg.globals.features},
        owner=owner,
        dao=dao,
    )
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

        step_inputs: Dict[str, Any] = {"request": request_obj, "owner": owner, "rag_context": rag_context, "owner_profile_context": owner_profile_context, "gemini_brief": gemini_brief}
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

        raw_text = invoke_agent(agent.bedrock.agentId, agent.bedrock.aliasId, run_id, prompt)

        try:
            out_json = json.loads(raw_text)
        except Exception as e:
            artifact_uri = save_artifact(run_id, step_id, {"raw_output": raw_text})
            dao.put_step(run_id, step_id, "FAILED", step_inputs, None, error=f"JSON parse failed: {e}", artifact_uri=artifact_uri)
            dao.put_run_meta(run_id, "FAILED", {"team": team, "version": version, "owner": owner, "error": f"JSON parse failed: {e}", "failed_step": step_id})
            raise StepFailed(step_id, f"JSON parse failed: {e}", raw_output=raw_text)

        schema_ref = agent.schema_ref
        schema = schema_objs.get(schema_ref)
        if not schema:
            raise StepFailed(step_id, f"Schema not found for ref {schema_ref}")

        try:
            validate_output(out_json, schema)
        except Exception as e:
            msg = str(e)
            try:
                from jsonschema.exceptions import ValidationError
                if isinstance(e, ValidationError):
                    msg = format_validation_error(e)
            except Exception:
                pass
            artifact_uri = save_artifact(run_id, step_id, {"output": out_json, "schema_ref": schema_ref})
            dao.put_step(run_id, step_id, "FAILED", step_inputs, out_json, error=f"Schema validation failed: {msg}", artifact_uri=artifact_uri)
            dao.put_run_meta(run_id, "FAILED", {"team": team, "version": version, "owner": owner, "error": f"Schema validation failed: {msg}", "failed_step": step_id})
            raise StepFailed(step_id, f"Schema validation failed: {msg}")

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
    m = _method(event)
    p = _path(event)

    if m == "OPTIONS":
        return {"statusCode": 200, "headers": _cors(), "body": ""}

    if p == "/improve/tasks" and m == "GET":
        qs = _qs(event)
        owner = (qs.get("owner") or "Tarun Raja")
        limit = int(qs.get("limit") or 50)
        team = qs.get("team")
        version = qs.get("version")
        dao = _dao_from_optional_team(team, version)
        return _resp(200, {"items": dao.list_tasks(owner, limit=limit)})

    if p == "/improve/task/done" and m == "POST":
        body = _json_body(event)
        owner = body.get("owner") or "Tarun Raja"
        task_id = body.get("task_id") or ""
        team = body.get("team")
        version = body.get("version")
        if not task_id:
            return _resp(400, {"error": "task_id required"})
        dao = _dao_from_optional_team(team, version)
        return _resp(200, dao.mark_task_done(owner, task_id))

    if p == "/team/task" and m == "POST":
        body = _json_body(event)
        team = body.get("team")
        version = body.get("version")
        request_obj = body.get("request") or {}
        if not team or not version:
            return _resp(400, {"error": "team and version required"})
        try:
            return _resp(200, _run_team_pipeline(team, version, request_obj))
        except StepFailed as sf:
            log.warning("run_failed_step", extra={"step": sf.step_id, "err": str(sf)})
            return _resp(200, {"status": "FAILED", "error": {"step": sf.step_id, "message": str(sf)}})
        except Exception as e:
            log.exception("run_failed_unhandled")
            return _resp(500, {"status": "FAILED", "error": str(e)})

    return _resp(404, {"error": "Route not found"})
