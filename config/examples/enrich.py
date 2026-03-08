"""
enrich.py
─────────────────────────────────────────────────────────────────────────────
Post-processing enrichment using Claude.

Called after every agent step in worker_handler.py to:
  1. Enforce voice — rewrite any copy fields to Tarun's first-person LinkedIn
     voice where they deviate from the established style.
  2. Enforce schema — ensure all required fields are present and correctly
     typed. Fill missing string fields with sensible defaults rather than
     leaving nulls that break downstream agents.

Uses claude-3-5-haiku on Bedrock via InvokeModel (not Bedrock Agents) —
this is a direct generation call, not an agent invocation.

Environment variables:
  ENRICH_MODEL  — Bedrock model ID for enrichment (default: us.anthropic.claude-3-5-haiku-20241022-v1:0)
  AWS_REGION    — AWS region (default: us-east-1)
"""

import json
import os
import boto3
from .logger import get_logger

log = get_logger("enrich")

ENRICH_MODEL = os.environ.get(
    "ENRICH_MODEL",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0"
)

_bedrock_runtime = None


def _client():
    global _bedrock_runtime
    if not _bedrock_runtime:
        _bedrock_runtime = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _bedrock_runtime


_VOICE_STYLE = """
Tarun Raja's LinkedIn voice:
- First-person throughout ("I built", "I learned", "I made this mistake")
- Direct, no corporate jargon or hype words
- Short punchy sentences. Short paragraphs. Blank line between each section.
- Contrarian hook — challenges conventional wisdom in the first two lines
- Concrete and specific — names tools, numbers, outcomes
- Never: "game-changer", "leverage", "synergy", "delighted to share", "excited to announce"
- Never: passive voice, third-person references to self, generic methodology frameworks
- Ends with a question or a single sharp closing statement, then 5-6 hashtags
""".strip()


def _build_enrichment_prompt(
    agent_name: str,
    schema_ref: str,
    raw_output: dict,
    schema: dict | None,
) -> str:
    schema_block = (
        f"\n\nTARGET SCHEMA:\n{json.dumps(schema, indent=2)}"
        if schema else ""
    )
    return f"""You are a post-processing enrichment step for a LinkedIn content pipeline.

You receive the raw JSON output from an agent called "{agent_name}" (schema: {schema_ref}).
Your job is to return a corrected version of this JSON with exactly two fixes applied:

## FIX 1 — VOICE CORRECTION
Any field that contains LinkedIn post copy, hooks, headlines, body text, or written content
must be rewritten to match this voice profile:

{_voice_style_block()}

Rules:
- Only rewrite copy fields (post_copy, hook, body, final_post, variants, etc.)
- Do NOT rewrite structured fields (IDs, dates, booleans, URLs, hashtag lists, scores)
- If the voice is already correct, leave the field unchanged
- If the agent produced placeholder text like "I cannot generate...", replace it with
  a best-effort attempt based on the other fields in the JSON

## FIX 2 — SCHEMA ENFORCEMENT
- Ensure all required fields from the schema are present
- Fill any null or missing string fields with a sensible default (never leave null
  where a string is expected)
- Ensure arrays are arrays, booleans are booleans, strings are strings
- Do NOT add fields that are not in the schema{schema_block}

## INPUT JSON:
{json.dumps(raw_output, indent=2)}

## OUTPUT RULES:
- Return ONLY valid JSON — no preamble, no explanation, no markdown fences
- The output must be the corrected version of the input JSON
- Preserve all fields and structure — only fix voice and schema compliance"""


def _voice_style_block() -> str:
    return _VOICE_STYLE


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

    prompt = _build_enrichment_prompt(agent_name, schema_ref, raw_output, schema)

    try:
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
        body    = json.loads(response["body"].read())
        text    = body["content"][0]["text"].strip()
        # Strip any accidental markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        enriched = json.loads(text)
        log.info(f"enrich_ok agent={agent_name} schema={schema_ref}")
        return enriched
    except json.JSONDecodeError as e:
        log.warning(f"enrich_json_parse_failed agent={agent_name} err={e}")
        return raw_output
    except Exception as e:
        log.warning(f"enrich_failed agent={agent_name} err={e} — returning raw output")
        return raw_output
