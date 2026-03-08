"""
enrich.py
─────────────────────────────────────────────────────────────────────────────
Post-processing enrichment using Claude.

Called after every agent step in worker_handler.py to:
  1. Detect and replace placeholder/refusal text from Nova
  2. Enforce voice — rewrite copy fields to Tarun's first-person LinkedIn voice
  3. Enforce schema — ensure all required fields are present and correctly typed

Uses Claude 3.5 Haiku via InvokeModel (direct, not Bedrock Agents).

Environment variables:
  ENRICH_MODEL  — Bedrock model ID (default: us.anthropic.claude-3-5-haiku-20241022-v1:0)
  AWS_REGION    — AWS region (default: us-east-1)
"""

import json
import os
import boto3
from .logger import get_logger

log = get_logger("enrich")

ENRICH_MODEL = os.environ.get(
    "ENRICH_MODEL",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
)

# Phrases that indicate Nova summarised instead of generating
_REFUSAL_PHRASES = [
    "the provided information is sufficient",
    "i cannot generate",
    "i am unable to generate",
    "i cannot produce",
    "i do not have",
    "i apologize",
    "i apologise",
    "please provide",
    "as an ai",
    "i cannot complete",
    "the information provided",
    "based on the information",
]

_VOICE_STYLE = """
Tarun Raja's LinkedIn voice rules:
- First-person throughout: "I built", "I learned", "I made this mistake"
- Short punchy sentences. Short paragraphs. Blank line between each section.
- Contrarian hook in first two lines — challenges conventional wisdom
- Concrete and specific: names tools, numbers, real outcomes
- Never: "game-changer", "leverage", "synergy", "delighted to share", "excited to announce"
- Never: passive voice, third-person self-reference, generic methodology frameworks
- Ends with a question or single sharp closing statement, then 5-6 hashtags
"""

_bedrock_runtime = None


def _client():
    global _bedrock_runtime
    if not _bedrock_runtime:
        _bedrock_runtime = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _bedrock_runtime


def _is_placeholder(value: str) -> bool:
    if not isinstance(value, str):
        return False
    lower = value.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


def _has_placeholder(obj, depth=0) -> bool:
    """Recursively check if any string value in the JSON is a placeholder."""
    if depth > 5:
        return False
    if isinstance(obj, str):
        return _is_placeholder(obj)
    if isinstance(obj, dict):
        return any(_has_placeholder(v, depth + 1) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_placeholder(item, depth + 1) for item in obj)
    return False


def _invoke_claude(prompt: str) -> str:
    response = _client().invoke_model(
        modelId=ENRICH_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    body = json.loads(response["body"].read())
    text = body["content"][0]["text"].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text


def _build_prompt(
    agent_name: str,
    schema_ref: str,
    raw_output: dict,
    schema: dict | None,
    has_placeholders: bool,
) -> str:
    schema_block = (
        f"\n\nTARGET SCHEMA:\n{json.dumps(schema, indent=2)}"
        if schema else ""
    )

    placeholder_block = """
## CRITICAL — PLACEHOLDER DETECTED
The input JSON contains placeholder or refusal text (e.g. "The provided information is sufficient",
"I cannot generate", etc.). This means the upstream agent failed to produce real content.
You MUST replace every placeholder string with actual generated content appropriate to the
agent's role and schema. Use the field names and schema to infer what content is needed.
Produce real LinkedIn post copy, real strategy output, real structured data — whatever the
schema requires. Do not carry any placeholder text through to the output.
""" if has_placeholders else ""

    return f"""You are a post-processing enrichment step for a LinkedIn content pipeline for Tarun Raja.

You receive the raw JSON output from agent "{agent_name}" (schema: {schema_ref}).
Return a corrected version of this JSON with the following fixes applied:
{placeholder_block}
## FIX 1 — VOICE CORRECTION
Rewrite any field containing LinkedIn copy (post_copy, hook, body, final_post, linkedin_post,
post, variants, content, draft, etc.) to match this voice:

{_VOICE_STYLE}

Rules:
- Only rewrite copy/text fields
- Do NOT rewrite structured fields (IDs, dates, booleans, URLs, scores, hashtag lists)
- Never carry through placeholder or refusal text

## FIX 2 — SCHEMA ENFORCEMENT
- Ensure all required fields are present
- Fill null/missing string fields with a generated default — never leave null where string is expected
- Ensure correct types: arrays are arrays, booleans are booleans, strings are strings
- Do not add fields absent from the schema{schema_block}

## INPUT JSON:
{json.dumps(raw_output, indent=2)}

## OUTPUT RULES:
- Return ONLY valid JSON — no preamble, no explanation, no markdown fences
- Preserve all fields and structure — only fix content and types"""


def enrich_step_output(
    agent_name: str,
    schema_ref: str,
    raw_output: dict,
    schema: dict | None = None,
) -> dict:
    """
    Enrich a single agent's JSON output through Claude.
    Returns the corrected dict. Falls back to raw_output on any error.
    """
    if not isinstance(raw_output, dict):
        return raw_output

    has_placeholders = _has_placeholder(raw_output)
    if has_placeholders:
        log.warning(
            "enrich_placeholder_detected agent=%s schema=%s — forcing rewrite",
            agent_name, schema_ref,
        )

    prompt = _build_prompt(agent_name, schema_ref, raw_output, schema, has_placeholders)

    try:
        text     = _invoke_claude(prompt)
        enriched = json.loads(text)
        log.info(
            "enrich_ok agent=%s schema=%s placeholders_fixed=%s",
            agent_name, schema_ref, has_placeholders,
        )
        return enriched
    except json.JSONDecodeError as e:
        log.warning("enrich_json_parse_failed agent=%s err=%s", agent_name, e)
        return raw_output
    except Exception as e:
        log.warning("enrich_failed agent=%s err=%s — returning raw output", agent_name, e)
        return raw_output
