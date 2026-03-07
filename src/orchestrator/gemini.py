import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict

import boto3
from .logger import get_logger

log = get_logger("gemini")
secrets = boto3.client("secretsmanager")

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

def _get_gemini_key() -> str:
    arn = os.environ.get("GEMINI_SECRET_ARN", "")
    if not arn:
        return ""
    resp = secrets.get_secret_value(SecretId=arn)
    s = resp.get("SecretString") or ""
    if not s:
        return ""
    try:
        j = json.loads(s)
        return j.get("key") or j.get("value") or ""
    except Exception:
        return s

def gemini_research_brief(team_globals: Dict[str, Any], request_obj: Dict[str, Any], completed_topics: str = "") -> str:
    feats = (team_globals or {}).get("features") or {}
    if not feats.get("gemini_research", False):
        log.warning("gemini disabled")
        return ""

    api_key = _get_gemini_key()
    if not api_key:
        log.warning("Gemini enabled but GEMINI_SECRET_ARN missing/empty")
        return ""

    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    url = GEMINI_ENDPOINT.format(model=model)

    topic = request_obj.get("topic", "")
    objective = request_obj.get("objective","")
    channel = request_obj.get("channel", "other")

    prompt = (
        "You are a research assistant for a personal technical improvement plan. "
        "Return a concise brief with 2000-3000 words in bullets, each bullet must include a credible URL. "
        "Prefer primary sources (official docs, specs, RFCs, vendor docs). "
        "Avoid repeating resources that map to completed topics/levels. "
        "If uncertain, tag NEEDS_SOURCE. Keep under 1200 chars. Return plain text only.\n"
        f"Topic: {topic}\nObjective: {objective}\nChannel: {channel}\n"
    )
    if completed_topics:
        prompt += f"Completed topics/levels (avoid repeats):\n{completed_topics}\n"

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}]
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="ignore")
        log.error("Gemini HTTPError", extra={"code": e.code, "body_prefix": msg[:300]})
        return ""
    except Exception as e:
        log.error("Gemini request failed", extra={"err": str(e)[:240]})
        return ""

    cands = data.get("candidates") or []
    if not cands:
        return ""
    parts = (((cands[0] or {}).get("content") or {}).get("parts") or [])
    return "".join([p.get("text", "") for p in parts if isinstance(p, dict)]).strip()
