import json
import os
from typing import Any, Dict, Tuple

import boto3

from .logger import get_logger
from .models import AgentConfig, BedrockRef, TeamConfig, TeamGlobals

log = get_logger("config_loader")
s3 = boto3.client("s3")

def _s3_get_json(bucket: str, key: str) -> Dict[str, Any]:
    log.info(f"Loading JSON from s3://{bucket}/{key}")
    resp = s3.get_object(Bucket=bucket, Key=key)
    raw = resp["Body"].read().decode("utf-8")
    return json.loads(raw)

def load_team_config(team: str, version: str) -> Tuple[TeamConfig, Dict[str, Any]]:
    bucket = os.environ["CONFIG_BUCKET"]
    prefix = os.environ.get("CONFIG_PREFIX", "teams").strip("/")
    team_key = f"{prefix}/{team}/{version}/team.json"
    doc = _s3_get_json(bucket, team_key)

    g = doc.get("globals") or {}
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
        agents.append(AgentConfig(
            id=a["id"],
            name=a.get("name", a["id"]),
            bedrock=BedrockRef(agentId=br.get("agentId",""), aliasId=br.get("aliasId","")),
            goal_template=a.get("goal_template",""),
            schema_ref=a.get("schema_ref",""),
        ))

    tc = TeamConfig(
        team=doc.get("team", {}),
        globals=globals_obj,
        agents=agents,
        workflow=doc.get("workflow", []),
        schemas=doc.get("schemas", {}),
    )
    return tc, doc
