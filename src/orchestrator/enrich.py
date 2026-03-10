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
from .json_utils import extract_json_payload
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
    "not to describe",
    "sorry",
    "task is to",
    "i apologise",
    "please provide",
    "not provided",
    "as an ai",
    "i cannot complete",
    "the information provided",
    "based on the information",
    "the task requires",
    "not supported by the available tools",
    "is not supported",
    "requires generating",
    "i don't have the ability",
    "i lack the",
    "this task requires",
    "i need more",
    "without additional",
    "cannot be completed",
    "no tools available",
    "tool is required",
    "tools are required",
    "i am not able",
    "not supported",
    "the task is to generate",
    "not to perform an action",
    "not to use a tool",
    "use a tool",
    "perform an action",
    "instead of performing",
    "rather than performing",
    "generate content, not",
]

_CRITICAL_ARRAYS = {"hooks", "outline", "drafts", "variants", "tasks", "hashtags", "claims"}

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


def _trim_for_prompt(value, max_str_len: int = 2000, max_items: int = 20):
    """Recursively trim large prompt context values without breaking JSON encoding."""
    if isinstance(value, str):
        if len(value) <= max_str_len:
            return value
        return value[:max_str_len] + "...[truncated]"

    if isinstance(value, list):
        trimmed = [_trim_for_prompt(item, max_str_len=max_str_len, max_items=max_items) for item in value[:max_items]]
        if len(value) > max_items:
            trimmed.append(f"...[{len(value) - max_items} more items truncated]")
        return trimmed

    if isinstance(value, dict):
        out = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                out["_truncated"] = f"...[{len(value) - max_items} more keys truncated]"
                break
            out[k] = _trim_for_prompt(v, max_str_len=max_str_len, max_items=max_items)
        return out

    return value


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
    """Recursively check if any string value is a placeholder,
    or if critical array fields are empty at any level."""
    if depth > 5:
        return False
    if isinstance(obj, str):
        return _is_placeholder(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _CRITICAL_ARRAYS and isinstance(v, list) and len(v) == 0:
                return True
            if _has_placeholder(v, depth + 1):
                return True
        return False
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
            "max_tokens": 8192,
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
    step_inputs: dict | None = None,
) -> str:
    schema_block = (
        f"\n\nTARGET SCHEMA:\n{json.dumps(schema, indent=2)}"
        if schema else ""
    )

    # Summarise step_inputs for context — strip embeddings/large blobs
    context_block = ""
    if step_inputs and has_placeholders:
        safe_inputs = {}
        for k, v in step_inputs.items():
            if k in ("embedding", "rag_context", "owner_profile_context"):
                continue
            safe_inputs[k] = _trim_for_prompt(v)
        context_block = f"\n\n## PIPELINE CONTEXT (use this to generate content):\n{json.dumps(safe_inputs, indent=2)}"

    if has_placeholders:
        placeholder_block = f"""
## CRITICAL — UPSTREAM AGENT FAILED TO GENERATE CONTENT
The agent "{agent_name}" produced only meta-commentary instead of actual content.
The raw output contains phrases like "The task requires generating..." or "not supported by available tools".
This is a generation failure — NOT a real constraint.

You must FULLY REGENERATE this output from scratch.
Use the PIPELINE CONTEXT below to understand what content is needed.
Produce real, specific content for every field in the schema.
Do not reference the failed output — treat it as if it never existed.{context_block}
"""
    else:
        placeholder_block = ""

    return f"""You are a post-processing enrichment step for a LinkedIn content pipeline for Tarun Raja.

You receive output from agent "{agent_name}" (schema: {schema_ref}).
Return a corrected JSON with the following fixes:
{placeholder_block}
## FIX 1 — VOICE CORRECTION
Rewrite any field containing LinkedIn copy (post_copy, hook, body, final_post, linkedin_post,
post, variants, content, draft, angle, cta, outline items, hooks) to match this voice:

{_VOICE_STYLE}

Rules:
- Only rewrite copy/text fields
- Do NOT rewrite structured fields (IDs, dates, booleans, URLs, scores, numeric values)
- Never carry through placeholder or refusal text

## FIX 2 — SCHEMA ENFORCEMENT
- Ensure all required fields are present and non-empty
- Arrays like hooks/outline/drafts must have actual items — never return empty arrays
- Fill null/missing string fields with generated content — never leave null
- Ensure correct types throughout{schema_block}

## INPUT JSON (may be a failed output — regenerate if needed):
{json.dumps(raw_output, indent=2)}

## OUTPUT RULES:
- Return ONLY valid JSON — no preamble, no explanation, no markdown fences
- Every array field must have at least one real item
- Every string field must have real content, not meta-commentary"""


def enrich_step_output(
    agent_name: str,
    schema_ref: str,
    raw_output: dict,
    schema: dict | None = None,
    step_inputs: dict | None = None,
) -> dict:
    """
    Enrich a single agent's JSON output through Claude.
    When placeholder/refusal text is detected, fully regenerates from step_inputs context.
    Returns the corrected dict. Falls back to raw_output on any error.
    """
    if not isinstance(raw_output, dict):
        return raw_output

    has_placeholders = _has_placeholder(raw_output)
    if has_placeholders:
        log.warning(
            "enrich_placeholder_detected agent=%s schema=%s — forcing full regeneration",
            agent_name, schema_ref,
        )

    prompt = _build_prompt(agent_name, schema_ref, raw_output, schema, has_placeholders, step_inputs)

    try:
        text     = _invoke_claude(prompt)
        enriched = extract_json_payload(text)
        log.info(
            "enrich_ok agent=%s schema=%s placeholders_fixed=%s",
            agent_name, schema_ref, has_placeholders,
        )
        return enriched
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("enrich_json_parse_failed agent=%s err=%s", agent_name, e)
        return raw_output
    except Exception as e:
        log.warning("enrich_failed agent=%s err=%s — returning raw output", agent_name, e)
        return raw_output
