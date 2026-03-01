import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from boto3.dynamodb.conditions import Key

from .logger import get_logger

log = get_logger("db")


class DbDao:
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.table = boto3.resource("dynamodb").Table(table_name)

    @staticmethod
    def from_team_config(team_raw: dict) -> "DbDao":
        artifact_store = ((team_raw or {}).get("globals") or {}).get("artifact_store") or {}
        table_name = artifact_store.get("dynamo_table_name") or os.environ.get("DDB_TABLE")
        if not table_name:
            raise RuntimeError("Missing DynamoDB table name; expected globals.artifact_store.dynamo_table_name or DDB_TABLE")
        return DbDao(table_name)

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    @classmethod
    def _strip_empty_strings(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                cleaned = cls._strip_empty_strings(v)
                if cleaned == "":
                    continue
                out[k] = cleaned
            return out
        if isinstance(obj, list):
            return [cls._strip_empty_strings(x) for x in obj]
        return obj

    @staticmethod
    def _ensure_required_indexes(item: Dict[str, Any]) -> None:
        status = item.get("status")
        updated_at = item.get("updatedAt")
        if not status or not isinstance(status, str) or not status.strip():
            raise RuntimeError("DynamoDB item requires non-empty status for GSI indexing")
        if not updated_at or not isinstance(updated_at, str) or not updated_at.strip():
            raise RuntimeError("DynamoDB item requires non-empty updatedAt for GSI indexing")
        if "publishedAt" in item and item.get("publishedAt") == "":
            item.pop("publishedAt", None)

    def _safe_put(self, item: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = self._strip_empty_strings(item)
        self._ensure_required_indexes(cleaned)
        try:
            log.info("ddb_put", extra={"pk": cleaned.get("pk"), "sk": cleaned.get("sk"), "table": self.table_name})
            self.table.put_item(Item=cleaned)
        except Exception as e:
            pk = cleaned.get("pk")
            sk = cleaned.get("sk")
            raise RuntimeError(f"Failed DynamoDB put_item for table={self.table_name} pk={pk} sk={sk}: {e}") from e
        return cleaned

    def get_item(self, pk: str, sk: str) -> Optional[Dict[str, Any]]:
        resp = self.table.get_item(Key={"pk": pk, "sk": sk})
        return resp.get("Item")

    def put_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self._safe_put(item)

    def put_run_meta(self, run_id: str, status: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = self._now_iso()
        data = payload or {}
        item: Dict[str, Any] = {
            "pk": f"RUN#{run_id}",
            "sk": "META",
            "status": status,
            "updatedAt": now,
            "data": data,
        }
        published_at = data.get("publishedAt") if isinstance(data, dict) else None
        if isinstance(published_at, str) and published_at.strip():
            item["publishedAt"] = published_at
        return self._safe_put(item)

    def put_step(
        self,
        run_id: str,
        step_id: str,
        status: str,
        inputs: Dict[str, Any],
        output: Optional[Dict[str, Any]],
        error: Optional[str],
        artifact_uri: Optional[str],
    ) -> Dict[str, Any]:
        item = {
            "pk": f"RUN#{run_id}",
            "sk": f"STEP#{step_id}",
            "status": status,
            "updatedAt": self._now_iso(),
            "error": error or "",
            "artifact_uri": artifact_uri or "",
            "inputs_json": inputs or {},
            "output_json": output if output is not None else {},
        }
        return self._safe_put(item)

    def list_runs_by_status(self, status: str, limit: int = 20) -> List[Dict[str, Any]]:
        log.info("ddb_query", extra={"index": "StatusUpdatedIndex", "status": status, "limit": limit, "table": self.table_name})
        resp = self.table.query(
            IndexName="StatusUpdatedIndex",
            KeyConditionExpression=Key("status").eq(status),
            Limit=limit,
            ScanIndexForward=False,
        )
        items = resp.get("Items", [])
        return [x for x in items if x.get("sk") == "META"]

    def get_run(self, run_id: str) -> Dict[str, Any]:
        pk = f"RUN#{run_id}"
        meta = self.get_item(pk, "META")
        resp = self.table.query(KeyConditionExpression=Key("pk").eq(pk), ScanIndexForward=True)
        items = resp.get("Items", [])
        steps = [it for it in items if str(it.get("sk", "")).startswith("STEP#")]
        return {"run_id": run_id, "meta": meta, "steps": steps}

    def put_tasks(self, owner: str, tasks: List[Dict[str, Any]], source_run_id: str) -> List[str]:
        ymd = datetime.utcnow().strftime("%Y-%m-%d")
        now = self._now_iso()
        stored: List[str] = []
        for task in tasks:
            tid = (task.get("task_id") or task.get("id") or "").strip()
            topic = task.get("topic")
            level = task.get("level")
            task_type = task.get("type")
            if not tid or not topic or not level or not task_type:
                continue

            item = {
                "pk": f"TASK#{owner}",
                "sk": f"DATE#{ymd}#TASK#{tid}",
                "task_id": tid,
                "topic": topic,
                "level": level,
                "type": task_type,
                "status": "TASK#OPEN",
                "updatedAt": now,
                "owner": owner,
                "date": ymd,
                "source_run_id": source_run_id,
                "estimate_minutes": int(task.get("estimate_minutes") or 30),
                "links": task.get("links", []),
                "instructions": task.get("instructions", ""),
                "reflection_prompt": task.get("reflection_prompt", ""),
                "createdAt": now,
            }
            self._safe_put(item)
            stored.append(tid)
        return stored

    def list_tasks(self, owner: str, limit: int = 50) -> List[Dict[str, Any]]:
        pk = f"TASK#{owner}"
        log.info("ddb_query", extra={"index": "PRIMARY", "pk": pk, "limit": limit, "table": self.table_name})
        resp = self.table.query(
            KeyConditionExpression=Key("pk").eq(pk),
            Limit=limit,
            ScanIndexForward=False,
        )
        return resp.get("Items", [])

    def mark_task_done(self, owner: str, task_id: str) -> Dict[str, Any]:
        pk = f"TASK#{owner}"
        resp = self.table.query(KeyConditionExpression=Key("pk").eq(pk), Limit=500, ScanIndexForward=False)
        items = resp.get("Items", [])
        target = None
        for it in items:
            if it.get("task_id") == task_id:
                target = it
                break
        if not target:
            return {"ok": False, "error": "Task not found"}
        target["status"] = "TASK#DONE"
        target["updatedAt"] = self._now_iso()
        self._safe_put(target)
        return {"ok": True, "task": target}

    def list_completed_topic_levels(self, owner: str, limit: int = 200) -> List[str]:
        items = self.list_tasks(owner, limit=limit)
        done: List[str] = []
        for it in items:
            if it.get("status") != "TASK#DONE":
                continue
            topic = (it.get("topic") or "").strip().lower()
            level = (it.get("level") or "").strip().lower()
            if topic and level:
                done.append(f"{topic}::{level}")
        seen = set()
        out = []
        for x in done:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out
