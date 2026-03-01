import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from boto3.dynamodb.conditions import Key

from .logger import get_logger

log = get_logger("db")


class DbDao:
    def __init__(self, table_name: str):
        if not table_name:
            raise RuntimeError("DynamoDB table name is required")
        self.table_name = table_name
        self.table = boto3.resource("dynamodb").Table(table_name)

    @staticmethod
    def from_team_config(team_raw: Dict[str, Any]) -> "DbDao":
        globals_cfg = (team_raw or {}).get("globals") or {}
        artifact_store = globals_cfg.get("artifact_store") or {}
        table_name = artifact_store.get("dynamo_table_name") or os.environ.get("DDB_TABLE")
        if not table_name:
            raise RuntimeError("DynamoDB table name missing (globals.artifact_store.dynamo_table_name or DDB_TABLE)")
        return DbDao(table_name)

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @classmethod
    def _strip_empty_strings(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            cleaned: Dict[str, Any] = {}
            for k, v in obj.items():
                if v == "":
                    continue
                cleaned_value = cls._strip_empty_strings(v)
                if cleaned_value == "":
                    continue
                cleaned[k] = cleaned_value
            return cleaned
        if isinstance(obj, list):
            cleaned_list = []
            for v in obj:
                cleaned_value = cls._strip_empty_strings(v)
                if cleaned_value == "":
                    continue
                cleaned_list.append(cleaned_value)
            return cleaned_list
        return obj

    @staticmethod
    def _ensure_required_indexes(item: Dict[str, Any]) -> None:
        status = item.get("status")
        updated_at = item.get("updatedAt")
        if not isinstance(status, str) or not status.strip():
            raise RuntimeError("DynamoDB item must include non-empty 'status'")
        if not isinstance(updated_at, str) or not updated_at.strip():
            raise RuntimeError("DynamoDB item must include non-empty 'updatedAt'")
        if "publishedAt" in item and item.get("publishedAt") == "":
            del item["publishedAt"]

    def _safe_put(self, item: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = self._strip_empty_strings(item)
        if "status" in cleaned or "updatedAt" in cleaned:
            self._ensure_required_indexes(cleaned)
        try:
            log.info("ddb_put", extra={"pk": cleaned.get("pk"), "sk": cleaned.get("sk"), "table": self.table_name})
            self.table.put_item(Item=cleaned)
        except Exception as e:
            raise RuntimeError(
                f"Failed DynamoDB put_item table={self.table_name} pk={cleaned.get('pk')} sk={cleaned.get('sk')}: {e}"
            ) from e
        return cleaned

    def get_item(self, pk: str, sk: str) -> Optional[Dict[str, Any]]:
        resp = self.table.get_item(Key={"pk": pk, "sk": sk})
        return resp.get("Item")

    def put_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self._safe_put(item)

    def put_run_meta(self, run_id: str, status: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = self._now_iso()
        item: Dict[str, Any] = {
            "pk": f"RUN#{run_id}",
            "sk": "META",
            "status": status,
            "updatedAt": now,
            "data": payload or {},
        }
        published_at = (payload or {}).get("publishedAt")
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
            "inputs_json": inputs or {},
            "output_json": output if output is not None else {},
            "error": error or "",
            "artifact_uri": artifact_uri or "",
        }
        return self._safe_put(item)

    def list_runs_by_status(self, status: str, limit: int = 20) -> List[Dict[str, Any]]:
        log.info("ddb_query", extra={"index": "StatusUpdatedIndex", "status": status, "limit": limit, "table": self.table_name})
        items: List[Dict[str, Any]] = []
        last_key = None
        while len(items) < limit:
            kwargs = {
                "IndexName": "StatusUpdatedIndex",
                "KeyConditionExpression": Key("status").eq(status),
                "Limit": max(1, limit - len(items)),
                "ScanIndexForward": False,
            }
            if last_key:
                kwargs["ExclusiveStartKey"] = last_key
            resp = self.table.query(**kwargs)
            for it in resp.get("Items", []):
                if it.get("sk") == "META":
                    items.append(it)
                    if len(items) >= limit:
                        break
            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break
        return items

    def get_run(self, run_id: str) -> Dict[str, Any]:
        pk = f"RUN#{run_id}"
        meta = self.get_item(pk, "META") or {}
        steps: List[Dict[str, Any]] = []
        last_key = None
        while True:
            kwargs = {"KeyConditionExpression": Key("pk").eq(pk), "ScanIndexForward": True}
            if last_key:
                kwargs["ExclusiveStartKey"] = last_key
            resp = self.table.query(**kwargs)
            items = resp.get("Items", [])
            steps.extend([it for it in items if str(it.get("sk", "")).startswith("STEP#")])
            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break
        return {"meta": meta, "steps": steps}

    def put_tasks(self, owner: str, tasks: List[Dict[str, Any]], source_run_id: str) -> List[str]:
        ymd = datetime.utcnow().strftime("%Y-%m-%d")
        now = self._now_iso()
        stored: List[str] = []
        for task in tasks:
            tid = (task.get("task_id") or task.get("id") or "").strip()
            topic = (task.get("topic") or "").strip()
            level = (task.get("level") or "").strip()
            task_type = (task.get("type") or "").strip()
            if not (tid and topic and level and task_type):
                continue
            item = {
                "pk": f"TASK#{owner}",
                "sk": f"DATE#{ymd}#TASK#{tid}",
                "owner": owner,
                "task_id": tid,
                "date": ymd,
                "status": "TASK#OPEN",
                "updatedAt": now,
                "topic": topic,
                "level": level,
                "type": task_type,
                "estimate_minutes": self._safe_int(task.get("estimate_minutes") or 30, 30),
                "links": task.get("links", []),
                "instructions": task.get("instructions", ""),
                "reflection_prompt": task.get("reflection_prompt", ""),
                "source_run_id": source_run_id,
            }
            self._safe_put(item)
            stored.append(tid)
        return stored

    def list_tasks(self, owner: str, limit: int = 50) -> List[Dict[str, Any]]:
        pk = f"TASK#{owner}"
        log.info("ddb_query", extra={"index": "PRIMARY", "status": None, "limit": limit, "table": self.table_name})
        items: List[Dict[str, Any]] = []
        last_key = None
        while len(items) < limit:
            kwargs = {
                "KeyConditionExpression": Key("pk").eq(pk),
                "Limit": max(1, limit - len(items)),
                "ScanIndexForward": False,
            }
            if last_key:
                kwargs["ExclusiveStartKey"] = last_key
            resp = self.table.query(**kwargs)
            items.extend(resp.get("Items", []))
            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break
        return items[:limit]

    def mark_task_done(self, owner: str, task_id: str) -> Dict[str, Any]:
        pk = f"TASK#{owner}"
        target = None
        last_key = None
        while True:
            kwargs = {
                "KeyConditionExpression": Key("pk").eq(pk),
                "Limit": 100,
                "ScanIndexForward": False,
            }
            if last_key:
                kwargs["ExclusiveStartKey"] = last_key
            resp = self.table.query(**kwargs)
            for it in resp.get("Items", []):
                if it.get("task_id") == task_id:
                    target = it
                    break
            if target is not None:
                break
            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break
        if not target:
            return {"ok": False, "error": "Task not found"}
        target["status"] = "TASK#DONE"
        target["updatedAt"] = self._now_iso()
        cleaned = self._safe_put(target)
        return {"ok": True, "task": cleaned}

    def list_completed_topic_levels(self, owner: str, limit: int = 200) -> List[str]:
        items = self.list_tasks(owner, limit=limit)
        out: List[str] = []
        seen = set()
        for it in items:
            if it.get("status") != "TASK#DONE":
                continue
            topic = (it.get("topic") or "").strip().lower()
            level = (it.get("level") or "").strip().lower()
            if not topic or not level:
                continue
            key = f"{topic}::{level}"
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out
