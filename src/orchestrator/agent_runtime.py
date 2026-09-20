"""Which substrate actually runs an agent turn.

Bedrock Agents Classic closed to new customers on 30 July 2026, takes no
further features, and its model catalogue is frozen as of that date -- a model
released after it is reachable only through AgentCore. Existing workloads keep
running, so nothing is urgent, but the substrate has to stop being welded to
the orchestrator.

This module is that seam and nothing more. What it deliberately does *not*
own is everything that is not a substrate concern: the retry policy, the
Observatory gate, guardrail logging and the StepFailed contract all stay in
``bedrock_invoke``, shared by every implementation. A runtime here is only
"how do I turn a prompt into text on this platform".

Selection is by the ``AGENT_RUNTIME`` environment variable and defaults to
``classic``, so importing this module changes nothing about how TeamWeave
behaves today.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import boto3
from botocore.config import Config

from .logger import get_logger
from .mcp_observatory import observe_agent_request

log = get_logger("agent_runtime")

# The Classic runtime client. Defined here but re-exported by
# ``bedrock_invoke`` so existing callers and tests that patch
# ``bedrock_invoke.brt`` keep working -- it is the same object.
brt = boto3.client(
    "bedrock-agent-runtime",
    config=Config(
        read_timeout=1800,
        connect_timeout=60,
        retries={"max_attempts": 0},
    ),
)


@dataclass(frozen=True)
class AgentRef:
    """Whatever identifies one agent on the selected substrate.

    Classic needs an agent id and an alias id. AgentCore needs a runtime ARN
    and optionally a qualifier. Carrying both keeps ``team.json`` and the
    agent registry able to describe either without a schema change per
    substrate.
    """

    agent_id: str = ""
    alias_id: str = ""
    runtime_arn: str = ""
    qualifier: str = ""


class AgentRuntime(Protocol):
    """One substrate's answer to "run this turn"."""

    name: str

    def missing_fields(self, ref: AgentRef) -> str:
        """Return a human-readable reason the ref is unusable, or ""."""

    def invoke(
        self,
        ref: AgentRef,
        *,
        session_id: str,
        input_text: str,
        shadow_alias_id: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """Run one turn. Returns (response_text, span_metrics)."""


def _drain_completion(resp: dict, log_guardrail) -> str:
    """Collect the text of a Classic InvokeAgent streaming response."""
    stream = resp.get("completion")
    if stream is None:
        raise RuntimeError("InvokeAgent missing 'completion' stream")

    out_chunks = []
    for event in stream:
        chunk = event.get("chunk")
        if chunk and chunk.get("bytes"):
            out_chunks.append(chunk["bytes"].decode("utf-8", errors="ignore"))
        log_guardrail(
            event.get("amazon-bedrock-guardrailAction"),
            event.get("amazon-bedrock-trace"),
            "Bedrock guardrail trace event",
        )
    return "".join(out_chunks).strip()


def _log_guardrail(action, trace, message: str) -> None:
    if action == "INTERVENED" or trace:
        log.info(
            message,
            extra={
                "amazon-bedrock-guardrailAction": action,
                "amazon-bedrock-trace": json.dumps(trace, default=str)[:4000],
            },
        )


class BedrockAgentsClassicRuntime:
    """Bedrock Agents Classic -- what TeamWeave runs on today."""

    name = "classic"

    def missing_fields(self, ref: AgentRef) -> str:
        if not ref.agent_id or not ref.alias_id:
            return "Missing agentId/aliasId in config"
        return ""

    def invoke(
        self,
        ref: AgentRef,
        *,
        session_id: str,
        input_text: str,
        shadow_alias_id: Optional[str] = None,
    ) -> Tuple[str, dict]:
        resp, span_metrics = observe_agent_request(
            brt,
            agent_id=ref.agent_id,
            alias_id=ref.alias_id,
            session_id=session_id,
            input_text=input_text,
            shadow_alias_id=shadow_alias_id,
        )
        _log_guardrail(
            resp.get("amazon-bedrock-guardrailAction"),
            resp.get("amazon-bedrock-trace"),
            "Bedrock guardrail trace",
        )
        return _drain_completion(resp, _log_guardrail), span_metrics


class AgentCoreRuntime:
    """Bedrock AgentCore Runtime -- the migration target. Not yet wired.

    What is confirmed: an agent is deployed as an HTTP service exposing
    ``POST /invocations`` and ``GET /ping`` (the ``@app.entrypoint`` decorator
    in ``bedrock-agentcore`` wraps that), it is addressed by ARN, and callers
    need ``bedrock-agentcore:InvokeAgentRuntime``.

    What is NOT confirmed, and is why this raises instead of guessing: the
    exact request and response envelope of ``InvokeAgentRuntime``, how a
    session id is carried, and how the streamed response frames text. Filling
    those in from the API reference is Phase 1. A plausible-looking
    implementation written from memory would fail in production rather than
    here, so it fails here.
    """

    name = "agentcore"

    def missing_fields(self, ref: AgentRef) -> str:
        if not ref.runtime_arn:
            return "Missing runtimeArn in config for the agentcore runtime"
        return ""

    def invoke(
        self,
        ref: AgentRef,
        *,
        session_id: str,
        input_text: str,
        shadow_alias_id: Optional[str] = None,
    ) -> Tuple[str, dict]:
        raise NotImplementedError(
            "AgentCoreRuntime is a declared seam, not an implementation. "
            "Phase 1 fills in InvokeAgentRuntime against the confirmed API "
            "reference. Set AGENT_RUNTIME=classic (the default) until then."
        )


_RUNTIMES = {
    BedrockAgentsClassicRuntime.name: BedrockAgentsClassicRuntime,
    AgentCoreRuntime.name: AgentCoreRuntime,
}

DEFAULT_RUNTIME = BedrockAgentsClassicRuntime.name


def runtime_name() -> str:
    """The configured substrate, falling back to Classic.

    An unrecognised value falls back rather than raising: a typo in an
    environment variable should not take the orchestrator down, and the
    warning says which substrate actually ran.
    """
    configured = (os.environ.get("AGENT_RUNTIME") or "").strip().lower()
    if not configured:
        return DEFAULT_RUNTIME
    if configured not in _RUNTIMES:
        log.warning(
            "Unknown AGENT_RUNTIME; falling back",
            extra={"configured": configured, "using": DEFAULT_RUNTIME, "known": sorted(_RUNTIMES)},
        )
        return DEFAULT_RUNTIME
    return configured


def get_runtime() -> AgentRuntime:
    """Build the configured runtime.

    Deliberately not cached: the environment is read per call so a test can
    switch substrate without reloading the module, and the cost is one small
    object against a network round trip.
    """
    return _RUNTIMES[runtime_name()]()
