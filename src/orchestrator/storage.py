import json
import os
from typing import Any, Dict, Optional, List
import boto3
from boto3.dynamodb.conditions import Key

from .logger import get_logger, now_iso, today_ymd

log = get_logger("storage")
ddb_resource = boto3.resource("dynamodb")
s3 = boto3.client("s3")

def _table():
    return ddb_resource.Table(os.environ["DDB_TABLE"])

def _artifact_bucket() -> str:
    return os.environ["ARTIFACT_BUCKET"]

def save_artifact(run_id: str, step_id: str, obj: Any, content_type: str = "application/json") -> str:
    bucket = _artifact_bucket()
    key = f"runs/{run_id}/{step_id}.json"
    body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
    log.info("artifact_saved", extra={"bucket": bucket, "key": key, "bytes": len(body)})
    return f"s3://{bucket}/{key}"

def put_step(run_id: str, step_id: str, status: str, inputs: Dict[str, Any],
             output_json: Optional[Dict[str, Any]], error: Optional[str] = None,
             artifact_uri: Optional[str] = None) -> None:
    pk = f"RUN#{run_id}"
    sk = f"STEP#{step_id}"
    item = {
        "pk": pk,
        "sk": sk,
        "entityType": "RUN_STEP",
        "run_id": run_id,
        "step_id": step_id,
        "status": status,
        "ts": now_iso(),
        "error": error or "",
        "artifact_uri": artifact_uri or "",
        "inputs_json": inputs,
        "output_json": output_json or {},
    }
    _table().put_item(Item=item)

def put_tasks(owner: str, tasks: List[Dict[str, Any]], source_run_id: str) -> List[str]:
    t = _table()
    ymd = today_ymd()
    stored = []
    for task in tasks:
        tid = task.get("task_id") or task.get("id") or ""
        if not tid:
            continue
        sk = f"DATE#{ymd}#TASK#{tid}"
        item = {
            "pk": f"TASK#{owner}",
            "sk": sk,
            "entityType": "TASK",
            "owner": owner,
            "task_id": tid,
            "date": ymd,
            "status": "OPEN",
            "topic": task.get("topic",""),
            "level": task.get("level",""),
            "type": task.get("type",""),
            "estimate_minutes": int(task.get("estimate_minutes") or 30),
            "links": task.get("links", []),
            "instructions": task.get("instructions",""),
            "reflection_prompt": task.get("reflection_prompt",""),
            "source_run_id": source_run_id,
            "createdAt": now_iso(),
            "updatedAt": now_iso(),
        }
        t.put_item(Item=item)
        stored.append(tid)
    return stored

def mark_task_done(owner: str, task_id: str) -> Dict[str, Any]:
    t = _table()
    pk = f"TASK#{owner}"
    resp = t.query(KeyConditionExpression=Key("pk").eq(pk), Limit=500, ScanIndexForward=False)
    items = resp.get("Items", [])
    target = None
    for it in items:
        if it.get("task_id") == task_id:
            target = it
            break
    if not target:
        return {"ok": False, "error": "Task not found"}
    target["status"] = "DONE"
    target["updatedAt"] = now_iso()
    t.put_item(Item=target)
    return {"ok": True, "task": target}

def list_tasks(owner: str, limit: int = 50) -> List[Dict[str, Any]]:
    t = _table()
    pk = f"TASK#{owner}"
    resp = t.query(KeyConditionExpression=Key("pk").eq(pk), Limit=limit, ScanIndexForward=False)
    return resp.get("Items", [])

def list_completed_topics(owner: str, limit: int = 200) -> List[str]:
    items = list_tasks(owner, limit=limit)
    done = []
    for it in items:
        if it.get("status") == "DONE":
            topic = (it.get("topic") or "").strip().lower()
            level = (it.get("level") or "").strip().lower()
            if topic:
                done.append(f"{topic}::{level}")
    seen = set()
    out = []
    for x in done:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out
