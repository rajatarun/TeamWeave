"""The agent program that runs on an AgentCore runtime.

Bedrock Agents Classic and AgentCore Runtime are different shapes. Classic is
declarative: you register an agent with an instruction and Bedrock executes it.
AgentCore Runtime hosts *your* code behind ``POST /invocations``, so the
equivalent of "the agent" has to exist as a program. This is that program, and
it is deliberately thin.

It has to be thin because TeamWeave already does the work. ``prompt_builder``
composes the whole per-turn prompt -- role, step goal, output contract, RAG
context, upstream inputs -- and the worker validates the result against the
step's JSON Schema afterwards. Duplicating any of that here would give the two
substrates different behaviour, which is the one thing a migration must not do.
So this applies the agent's instruction as the system prompt, calls the model,
and returns the text.

Per-agent configuration arrives as environment variables set when the runtime
is created, mirroring how a Classic agent carries its instruction and model.

The payload contract is shared with ``agent_runtime.AgentCoreRuntime``:
it sends ``{"prompt", "sessionId"}`` and reads ``result`` back.
``tests/test_agentcore_app.py`` holds both halves to it.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

DEFAULT_MODEL_ID = "us.amazon.nova-micro-v1:0"
DEFAULT_MAX_TOKENS = 4096

_client = None


def _bedrock():
    global _client
    if _client is None:
        import boto3

        _client = boto3.client("bedrock-runtime")
    return _client


def extract_prompt(payload: Any) -> str:
    """Pull the prompt out of an invocation payload.

    The SDK passes payloads to the entrypoint unchanged, so this accepts the
    shape TeamWeave sends and tolerates a bare string or a couple of common
    aliases rather than failing on a caller that is almost right.
    """
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="ignore")
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except ValueError:
                return text
        else:
            return text
    if isinstance(payload, dict):
        for key in ("prompt", "inputText", "input", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _text_from_converse(response: Dict[str, Any]) -> str:
    blocks = (((response or {}).get("output") or {}).get("message") or {}).get("content") or []
    return "".join(b.get("text", "") for b in blocks if isinstance(b, dict)).strip()


def run_turn(payload: Any, *, client=None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """One agent turn. Pure enough to test without the runtime or AWS."""
    env = env if env is not None else os.environ
    prompt = extract_prompt(payload)
    if not prompt:
        # A blank prompt is a caller error, and returning an empty answer would
        # surface downstream as a schema validation failure pointing at the
        # model instead of at the request.
        return {"error": "payload contained no prompt", "result": ""}

    instruction = (env.get("AGENT_INSTRUCTION") or "").strip()
    model_id = (env.get("AGENT_MODEL_ID") or "").strip() or DEFAULT_MODEL_ID
    try:
        max_tokens = int(env.get("AGENT_MAX_TOKENS") or DEFAULT_MAX_TOKENS)
    except ValueError:
        max_tokens = DEFAULT_MAX_TOKENS

    request: Dict[str, Any] = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0},
    }
    if instruction:
        request["system"] = [{"text": instruction}]

    response = (client or _bedrock()).converse(**request)
    return {"result": _text_from_converse(response), "modelId": model_id}


def build_app():
    """Wrap run_turn in the AgentCore runtime app.

    Imported lazily so the module is testable without the SDK installed, and
    so a packaging mistake surfaces here rather than at import time.
    """
    from bedrock_agentcore.runtime import BedrockAgentCoreApp

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def invoke(payload):  # pragma: no cover - exercised via run_turn
        return run_turn(payload)

    return app


if __name__ == "__main__":  # pragma: no cover
    build_app().run()
