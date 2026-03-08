"""
gemini_lambda.py
─────────────────────────────────────────────────────────────────────────────
Bedrock Action Group Lambda — exposes Gemini as a research tool.

Bedrock calls this Lambda with the standard Action Group event format.
The agent invokes the `gemini_research` function with a `query` parameter.

Environment variables:
  GEMINI_API_KEY   — Google AI Studio API key (required)
  GEMINI_MODEL     — model to use (default: gemini-1.5-flash)
  MAX_TOKENS       — max output tokens (default: 1024)
"""

import json
import logging
import os
import urllib.request
import urllib.error

log = logging.getLogger()
log.setLevel(logging.INFO)

GEMINI_MODEL  = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
MAX_TOKENS    = int(os.environ.get("MAX_TOKENS", "1024"))
API_BASE      = "https://generativelanguage.googleapis.com/v1beta/models"


def _gemini_search(query: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    url     = f"{API_BASE}/{GEMINI_MODEL}:generateContent?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": query}]}],
        "generationConfig": {"maxOutputTokens": MAX_TOKENS},
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"Gemini API error {e.code}: {body}")

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response shape: {data}") from e


def _action_group_response(event: dict, result: str, error: str | None = None) -> dict:
    """
    Build the Bedrock Action Group response envelope.
    https://docs.aws.amazon.com/bedrock/latest/userguide/agents-lambda.html
    """
    action_group  = event.get("actionGroup", "")
    function_name = event.get("function", "")
    session_attrs = event.get("sessionAttributes", {})
    prompt_attrs  = event.get("promptSessionAttributes", {})

    if error:
        body = {"error": error}
        status = "FAILURE"
    else:
        body = {"result": result}
        status = "SUCCESS"

    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup":  action_group,
            "function":     function_name,
            "functionResponse": {
                "responseState": status,
                "responseBody":  {
                    "TEXT": {"body": json.dumps(body)},
                },
            },
        },
        "sessionAttributes":       session_attrs,
        "promptSessionAttributes": prompt_attrs,
    }


def handler(event: dict, context) -> dict:
    log.info(f"Gemini action group event: {json.dumps(event, default=str)}")

    function_name = event.get("function", "")
    parameters    = {p["name"]: p["value"] for p in event.get("parameters", [])}

    if function_name != "gemini_research":
        return _action_group_response(
            event, result="",
            error=f"Unknown function '{function_name}'. Only 'gemini_research' is supported."
        )

    query = parameters.get("query", "").strip()
    if not query:
        return _action_group_response(event, result="", error="'query' parameter is required.")

    log.info(f"Gemini query: {query[:200]}")
    try:
        result = _gemini_search(query)
        log.info(f"Gemini result length: {len(result)}")
        return _action_group_response(event, result=result)
    except Exception as e:
        log.error(f"Gemini research failed: {e}")
        return _action_group_response(event, result="", error=str(e))
