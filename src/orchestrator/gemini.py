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


def _build_writer_prompt(topic: str, objective: str) -> str:
    return f"""You are a senior technical learning advisor writing a curated research brief for an engineer's personal improvement plan.

=== ASSIGNMENT ===
Topic: {topic}
Objective: {objective}

=== YOUR TASK ===
Before writing the brief, reason through the following and embed that reasoning into your output structure:

1. Who is likely studying this topic toward this objective?
   - Infer the probable current skill level (beginner / intermediate / advanced)
   - Infer the target role or outcome implied by the objective
   - Infer the likely tech stack or ecosystem involved

2. What does mastery of this topic actually require?
   - Break the topic into 3-5 sub-skills or knowledge areas
   - Identify which sub-skills are foundational vs. advanced
   - Identify common misconceptions or gaps engineers have here

3. What is the minimum viable learning path?
   - Order sub-skills by dependency (what must come first)
   - Flag what can be skipped if objective is narrow/applied

=== OUTPUT FORMAT ===
Write a 2000-3000 word brief in plain text bullets. Structure:

INFERRED CONTEXT
- Probable level, target role, ecosystem (your reasoning made explicit)

TOPIC DECOMPOSITION
- 3-5 sub-skills with one-line descriptions
- Mark each: [FOUNDATIONAL] or [ADVANCED]

CONCEPT OVERVIEW
- 3-5 bullets: what this is, why it matters for the objective

FOUNDATIONAL RESOURCES (skip if objective implies advanced level)
- 4-6 bullets: resource name | what to focus on | URL

CORE RESOURCES
- 8-12 bullets: resource name | specific section/chapter/module to focus on | estimated time | URL
- Prefer: official docs > RFCs/specs > vendor engineering blogs > ACM/IEEE papers > curated tutorials
- Avoid: Medium, Reddit, paywalled content without free tier

APPLIED / HANDS-ON
- 4-6 bullets: resource name | what to build or do | URL

ADVANCED / CUTTING EDGE
- 3-5 bullets from last 2 years: resource name | key insight | URL

KNOWLEDGE CHECK
- 3 questions the engineer must be able to answer after this brief
- 1 concrete mini-project to validate understanding

=== RULES ===
- Every resource bullet must include a working URL
- If no credible URL exists, tag: [NEEDS_SOURCE]
- Calibrate depth to inferred level - skip basics if objective implies senior/applied work
- Plain text only, no markdown, no code fences
- 2000-3000 words total
"""


def gemini_research_brief(feats: Dict[str, Any], request_obj: Dict[str, Any], completed_topics: str = "") -> str:

    #if not feats.get("gemini_research", False):
        #log.warning("gemini disabled")
        #return ""

    api_key = _get_gemini_key()
    if not api_key:
        log.warning("Gemini enabled but GEMINI_SECRET_ARN missing/empty")
        return ""

    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    url = GEMINI_ENDPOINT.format(model=model)

    topic = request_obj.get("topic", "")
    objective = request_obj.get("objective", "")

    prompt = _build_writer_prompt(topic, objective)

    if completed_topics:
        prompt += f"\nCOMPLETED TOPICS (do NOT resurface resources for these):\n{completed_topics}\n"

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
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
