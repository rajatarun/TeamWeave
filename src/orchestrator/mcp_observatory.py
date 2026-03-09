"""MCP Observatory integration for Bedrock wrapper observability.

Wraps Bedrock agent and model invocations with the mcp-observatory
InvocationWrapperAPI, recording per-call telemetry (trace ID, token
estimates, cost, latency, policy decision) via structured log records.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from mcp_observatory.instrument import instrument_wrapper_api

from .bedrock_wrappers import invoke_agent_request, invoke_model_request
from .logger import get_logger

log = get_logger("mcp_observatory")

# One shared wrapper for all Bedrock calls made by this service.
_wrapper = instrument_wrapper_api("teamweave-bedrock")


def observe_agent_request(
    runtime_client,
    *,
    agent_id: str,
    alias_id: str,
    session_id: str,
    input_text: str,
) -> dict:
    """Invoke a Bedrock agent through the mcp-observatory wrapper.

    Delegates to ``invoke_agent_request`` while recording a telemetry span
    (trace_id, token estimates, cost_usd, latency, policy decision) and
    emitting a single structured log record per invocation.
    """
    result = asyncio.run(
        _wrapper.invoke(
            source="agent",
            model="bedrock-agent",
            prompt=input_text,
            input_payload={
                "agent_id": agent_id,
                "alias_id": alias_id,
                "session_id": session_id,
            },
            call=lambda: invoke_agent_request(
                runtime_client,
                agent_id=agent_id,
                alias_id=alias_id,
                session_id=session_id,
                input_text=input_text,
            ),
        )
    )

    log.info(
        "mcp_observatory",
        extra={
            "operation": "invoke_agent",
            "agent_id": agent_id,
            "alias_id": alias_id,
            "session_id": session_id,
            "input_len": len(input_text),
            "trace_id": result.span.trace_id,
            "prompt_tokens": result.span.prompt_tokens,
            "completion_tokens": result.span.completion_tokens,
            "cost_usd": result.span.cost_usd,
            "decision": result.decision.action,
            "decision_reason": result.decision.reason,
        },
    )
    return result.output


def observe_model_request(
    runtime_client,
    *,
    model_id: str,
    body: str,
    content_type: Optional[str] = None,
    accept: Optional[str] = None,
) -> dict:
    """Invoke a Bedrock model through the mcp-observatory wrapper.

    Delegates to ``invoke_model_request`` while recording a telemetry span
    and emitting a single structured log record per invocation.
    """
    result = asyncio.run(
        _wrapper.invoke(
            source="model",
            model=model_id,
            prompt=body,
            input_payload={
                "model_id": model_id,
                "content_type": content_type,
                "accept": accept,
            },
            call=lambda: invoke_model_request(
                runtime_client,
                model_id=model_id,
                body=body,
                content_type=content_type,
                accept=accept,
            ),
        )
    )

    log.info(
        "mcp_observatory",
        extra={
            "operation": "invoke_model",
            "model_id": model_id,
            "body_len": len(body),
            "trace_id": result.span.trace_id,
            "prompt_tokens": result.span.prompt_tokens,
            "completion_tokens": result.span.completion_tokens,
            "cost_usd": result.span.cost_usd,
            "decision": result.decision.action,
            "decision_reason": result.decision.reason,
        },
    )
    return result.output
