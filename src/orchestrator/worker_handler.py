import json
import os
import uuid
from typing import Any, Dict, Optional

import boto3

from . import deadline
from . import bedrock_image
from .bedrock_invoke import invoke_agent, invoke_agent_with_metrics
from . import contextweave_client
from . import dpo_collector
from .config_loader import load_team_config
from .db import DbDao
from .enrich import enrich_step_output
from .gemini import gemini_research_brief
from .json_utils import build_standard_response, extract_json_payload
from .logger import get_logger
from .models import StepFailed
from .profile_context import get_owner_profile_context
from .prompt_builder import build_prompt
from .rag import get_rag_context_with_meta
from .storage import presign, save_artifact, save_bytes
from .structured_transform import transform_json_to_schema
from .tool_registry import execute_post_tools, execute_pre_tools

log = get_logger("worker_handler")
lambda_client = boto3.client("lambda")


# ASSUMPTION: The Step Functions execution input preserves the existing POST body contract:
# {"team": "...", "version": "...", "request": {...}}.


def _find_agent(team_cfg, agent_id: str):
    for a in team_cfg.agents:
        if a.id == agent_id:
            return a
    return None


def _image_prompt(agent, step_inputs: Dict[str, Any]) -> str:
    """The art direction, then the content it illustrates.

    The agent's `goal_template` is the art direction -- style, framing, what to
    avoid -- and the approved copy is what the image is *of*. Both matter and
    the direction goes first, because the model's prompt cap truncates the
    tail and losing the style is worse than losing the last sentence of a post
    the image only has to evoke.
    """
    parts = [str(agent.goal_template or "").strip()]
    for key, value in (step_inputs or {}).items():
        if key in {"request", "rag_context", "research_context", "owner",
                   "rag_meta", "owner_profile_context"}:
            continue
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (dict, list)):
            parts.append(json.dumps(value, ensure_ascii=False))
    return " ".join(p for p in parts if p)


def _run_image_step(agent, step_id: str, run_id: str, step_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """An image step's output: a reference, never the bytes.

    A base64 PNG is far past Step Functions' 256 KB state limit, and every
    step output travels through it -- returning the image inline would fail
    the whole run at the state transition, after paying for the image.
    """
    prompt = _image_prompt(agent, step_inputs)
    declared = getattr(agent.bedrock, "model_id", "") or ""
    try:
        result = bedrock_image.generate(prompt, declared_model_id=declared)
    except Exception as exc:  # noqa: BLE001 - see below
        # The illustration adorns the deliverable; it is not the deliverable.
        # Failing the run here threw away a finished, approved post because a
        # picture of it could not be made -- which is what happened on the
        # first real run, when Bedrock refused the image model outright.
        #
        # This is the same call the RAG layer makes and the opposite of the
        # empty-success trap: nothing claims an image exists. The step records
        # what failed, the run keeps its post, and the smoke test and the UI
        # both read `error` and say so.
        log.warning(
            "image_step_failed step=%s run_id=%s model=%s err=%s",
            step_id, run_id, bedrock_image.model_id(declared), str(exc)[:300],
        )
        return {
            "image_uri": "",
            "image_url": "",
            "model_id": bedrock_image.model_id(declared),
            "prompt": prompt[:500],
            "content_type": "image/png",
            "error": f"{type(exc).__name__}: {str(exc)[:400]}",
        }

    uri = save_bytes(
        run_id, step_id, result["bytes"], extension="png", content_type="image/png",
    )
    log.info("image_step_complete step=%s run_id=%s uri=%s", step_id, run_id, uri)
    return {
        "image_uri": uri,
        # A browser cannot open an s3:// URI, so the run would produce an
        # image nobody could look at. Best-effort and short-lived: the durable
        # reference is image_uri.
        "image_url": presign(uri),
        "model_id": result["model_id"],
        "prompt": result["prompt"],
        "width": result["width"],
        "height": result["height"],
        "content_type": "image/png",
    }


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
    rag_meta: Dict[str, Any],
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
        # Retrieval provenance (e.g. ContextWeave queryId) — persisted with the
        # step record so an answer can be rated later; kept out of the prompt.
        "rag_meta": rag_meta,
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


def _invoke_with_dpo(
    *,
    agent,
    prompt: str,
    run_id: str,
    step_id: str,
    team: str,
    step_inputs: Dict[str, Any],
    shadow_alias_id: Optional[str],
) -> str:
    """Invoke the agent twice, rank by composite_risk_score, upload DPO pair if delta > threshold.

    Uses independent session IDs so the agent treats both calls as separate
    conversations.  The better response (lower composite_risk_score) is
    returned for use in the pipeline.  If invocation B fails, falls back to
    response A gracefully.
    """
    def _invoke(session_id: str):
        return invoke_agent_with_metrics(
            agent.bedrock.agentId,
            agent.bedrock.aliasId,
            session_id,
            prompt,
            shadow_alias_id=shadow_alias_id,
            runtime_arn=agent.bedrock.runtimeArn,
            qualifier=agent.bedrock.qualifier,
            team=team,
            model_id=agent.bedrock.model_id,
        )

    return dpo_collector.collect_dpo_step(
        _invoke,
        team=team,
        step_id=step_id,
        run_id=run_id,
        prompt=prompt,
        context=step_inputs,
        session_id_a=f"{run_id}-{step_id}-dpo-a",
        session_id_b=f"{run_id}-{step_id}-dpo-b",
    )


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
        rag_context, rag_meta = get_rag_context_with_meta(
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
        rag_context, rag_meta = "", {}

    owner_profile_context = get_owner_profile_context(request_obj, team_raw, owner)

    completed_topics = rag_context if rag_context.startswith("COMPLETED_TASKS_HISTORY") else ""
    gemini_brief = gemini_research_brief(
        {"features": team_cfg.globals.features},
        request_obj,
        completed_topics=completed_topics,
    )

    outputs: Dict[str, Any] = {}
    supervisor_brief: Dict[str, Any] = {}
    schema_valid = True

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
            rag_meta,
            outputs,
            supervisor_brief,
        )

        # ── Pre-tools — run before agent, inject results into step context ──
        step_inputs = execute_pre_tools(step_def, step_inputs)

        # An image agent is a team member in team.json but not an agent *turn*:
        # Bedrock's image models do not implement Converse, which is all the
        # AgentCore runtime program speaks. It runs through bedrock_image and
        # rejoins the pipeline with an ordinary schema-shaped step output.
        if getattr(agent.bedrock, "modality", "text") == "image":
            out_json = _run_image_step(agent, step_id, run_id, step_inputs)
            out_json = execute_post_tools(step_def, out_json, step_inputs)
            artifact_uri = save_artifact(run_id, step_id, out_json)
            dao.put_step(run_id, step_id, "SUCCEEDED", step_inputs, out_json,
                         error=None, artifact_uri=artifact_uri)
            outputs[step_id] = out_json
            continue

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

        # Resolve shadow alias from team config for mcp_observatory dual_invoke
        shadow_alias_id = None
        if agent.bedrock.shadow_model_id and agent.bedrock.model_aliases:
            shadow_alias_id = agent.bedrock.model_aliases.get(agent.bedrock.shadow_model_id) or None

        # ── Agent invocation — dual when DPO collection is enabled ──────────
        if dpo_collector.dpo_bucket():
            raw_text = _invoke_with_dpo(
                agent=agent,
                prompt=prompt,
                run_id=run_id,
                step_id=step_id,
                team=team,
                step_inputs=step_inputs,
                shadow_alias_id=shadow_alias_id,
            )
        else:
            raw_text = invoke_agent(
                agent.bedrock.agentId, agent.bedrock.aliasId, run_id, prompt,
                shadow_alias_id=shadow_alias_id,
                runtime_arn=agent.bedrock.runtimeArn,
                qualifier=agent.bedrock.qualifier,
                team=team,
                model_id=agent.bedrock.model_id,
            )

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

        # ── Enrichment — voice correction + schema enforcement via Claude ──────
        step_schema = _load_step_schema(team_raw, agent.schema_ref)
        out_json = enrich_step_output(
            agent_name=agent.name,
            schema_ref=agent.schema_ref,
            raw_output=out_json,
            schema=step_schema,
            step_inputs=step_inputs,
        )
        log.info("enrichment_complete step=%s run_id=%s", step_id, run_id)
        # ──────────────────────────────────────────────────────────────────────

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
                schema_valid = False
                out_json = _build_transform_fallback(raw_text, transform_error)

        # ── Post-tools — enrich/transform agent output after schema coercion ─
        out_json = execute_post_tools(step_def, out_json, step_inputs)

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

        if step_id == "TIT_TDEPT-002_TIT-003_advisor" and isinstance(out_json.get("daily_tasks"), list):
            dao.put_tasks(owner=owner, tasks=out_json["daily_tasks"], source_run_id=run_id)

    # Close the loop with the knowledge layer: every step produced schema-valid
    # output from this grounding, so rate it up. Opt-in (see the client) because
    # a schema-valid run is only weak evidence that the answer was good.
    if schema_valid:
        contextweave_client.maybe_send_valid_output_feedback(rag_meta.get("query_id", ""))

    dao.put_run_meta(
        run_id,
        "SUCCEEDED",
        {"team": team, "version": version, "owner": owner, "steps": list(outputs.keys())},
    )
    return {"run_id": run_id, "status": "SUCCEEDED", "steps": outputs, "owner": owner}


def handler(event, context):
    # Every outbound call from here on is bounded by what is left of this
    # invocation, so a stall raises a real error with time to report which
    # step hung -- rather than Lambda killing the process and reporting
    # Sandbox.Timedout, which names nothing.
    deadline.set_deadline_from_context(context)
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
