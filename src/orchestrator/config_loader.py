import json
import os
from typing import Any, Dict, Tuple

import boto3

from .logger import get_logger
from .models import (
    CONTEXTWEAVE_MODE,
    RAG_MODES,
    AgentConfig,
    BedrockRef,
    TeamConfig,
    TeamGlobals,
)

log = get_logger("config_loader")
s3 = boto3.client("s3")

def _s3_get_json(bucket: str, key: str) -> Dict[str, Any]:
    log.info(f"Loading JSON from s3://{bucket}/{key}")
    resp = s3.get_object(Bucket=bucket, Key=key)
    raw = resp["Body"].read().decode("utf-8")
    return json.loads(raw)

def _validate_rag(rag: Dict[str, Any], team: str, version: str) -> None:
    """Fail fast on a RAG config the deployed stack cannot serve.

    An unknown mode is only warned about (rag.get_rag_context degrades to no
    context), but a team asking for ContextWeave when no URL is wired would
    silently run ungrounded — that is worth refusing at load time.
    """
    mode = str(rag.get("mode") or "kb").lower()
    if mode not in RAG_MODES:
        log.warning(f"Unknown globals.rag.mode '{mode}' in {team}/{version}; expected one of {sorted(RAG_MODES)}")
        return

    if mode == CONTEXTWEAVE_MODE and not os.environ.get("CONTEXTWEAVE_URL", "").strip():
        raise ValueError(
            f"{team}/{version}: globals.rag.mode='{CONTEXTWEAVE_MODE}' requires the CONTEXTWEAVE_URL "
            "environment variable (SAM parameter ContextWeaveUrl). It is empty, so the ContextWeave "
            "knowledge layer is unavailable for this stack."
        )

    for key in ("top_k", "min_confidence"):
        if rag.get(key) is None:
            continue
        try:
            float(rag[key])
        except (TypeError, ValueError):
            raise ValueError(f"{team}/{version}: globals.rag.{key} must be a number, got {rag[key]!r}")


def load_team_config(team: str, version: str) -> Tuple[TeamConfig, Dict[str, Any]]:
    bucket = os.environ["CONFIG_BUCKET"]
    prefix = os.environ.get("CONFIG_PREFIX", "teams").strip("/")
    team_key = f"{prefix}/{team}/{version}/team.json"
    doc = _s3_get_json(bucket, team_key)

    g = doc.get("globals") or {}
    _validate_rag(g.get("rag") or {}, team, version)
    globals_obj = TeamGlobals(
        north_star=g.get("north_star",""),
        default_channel=g.get("default_channel","linkedin"),
        hard_constraints=g.get("hard_constraints", []),
        features=g.get("features", {}),
        rag=g.get("rag", {}),
        artifact_store=g.get("artifact_store", {}),
        revision=g.get("revision", {}),
    )

    agents = []
    for a in doc.get("agents", []):
        br = a.get("bedrock") or {}
        category = str(a.get("model_category") or br.get("model_category") or "").strip()
        if not category:
            log.warning(
                "agent_missing_model_category",
                extra={"team": team, "agent": a.get("id", "")},
            )
            category = "default"
        override = str(br.get("model_id") or "").strip()
        if override:
            log.warning(
                "model_id_override",
                extra={"team": team, "agent": a.get("id", ""), "category": category, "model_id": override},
            )
        agents.append(AgentConfig(
            id=a["id"],
            name=a.get("name", a["id"]),
            bedrock=BedrockRef(
                agentId=br.get("agentId",""),
                aliasId=br.get("aliasId",""),
                model_id=override,
                shadow_model_id=br.get("shadow_model_id", ""),
                model_aliases=br.get("model_aliases", {}),
                runtimeArn=br.get("runtimeArn", ""),
                qualifier=br.get("qualifier", ""),
                modality=str(br.get("modality", "text") or "text").strip().lower(),
                image_provider=str(br.get("image_provider", "bedrock") or "bedrock").strip().lower(),
            ),
            goal_template=a.get("goal_template",""),
            schema_ref=a.get("schema_ref",""),
            model_category=category,
        ))

    tc = TeamConfig(
        team=doc.get("team", {}),
        globals=globals_obj,
        agents=agents,
        workflow=doc.get("workflow", []),
        schemas=doc.get("schemas", {}),
    )
    return tc, doc
