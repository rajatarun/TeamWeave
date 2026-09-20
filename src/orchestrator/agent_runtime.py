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

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import boto3
from botocore.config import Config

from .logger import get_logger
from .mcp_observatory import observe_agent_request, observe_agentcore_request

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


# InvokeAgentRuntime's runtimeSessionId is min length 33 / max 256
# (botocore model bedrock-agentcore/2024-02-28). TeamWeave's run-scoped
# session ids are shorter than that, and a short one is a ValidationException
# at the API rather than a local error -- the kind of detail that makes a
# guessed implementation fail in production instead of in a test.
_AGENTCORE_SESSION_MIN = 33
_AGENTCORE_SESSION_MAX = 256


def agentcore_session_id(session_id: str) -> str:
    """Map a TeamWeave session id onto AgentCore's length window.

    Padding is deterministic so repeated turns of one run land on the same
    AgentCore session: that is what keeps the runtime's own conversational
    memory attached to the right conversation.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "-", session_id or "").strip("-") or "session"
    if len(cleaned) < _AGENTCORE_SESSION_MIN:
        # A hash of the original, not random padding: the same session id must
        # always produce the same AgentCore session id.
        digest = hashlib.sha256((session_id or "").encode("utf-8")).hexdigest()
        cleaned = cleaned + "-" + digest
    # One cap for both paths, so the bound is actually reachable and testable.
    return cleaned[:_AGENTCORE_SESSION_MAX]


def _extract_text(body: bytes) -> str:
    """Pull the agent's text out of whatever the entrypoint returned.

    The SDK passes invocation payloads to the entrypoint unchanged and returns
    whatever it produces, so the response is only JSON by convention. Falling
    back to the decoded body keeps a plain-text agent working instead of
    failing on a json.loads.
    """
    text = body.decode("utf-8", errors="ignore").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except ValueError:
        return text
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        for key in ("result", "output", "response", "completion", "text", "message"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(parsed)
    return text


class AgentCoreRuntime:
    """Bedrock AgentCore Runtime.

    Built against the botocore service model for bedrock-agentcore
    (2024-02-28), not from recollection: InvokeAgentRuntime takes the runtime
    ARN in the URI, an optional qualifier in the query string, the session id
    in a header, and a blob payload; the response body is a streaming blob
    rather than an event stream, so it is read once rather than iterated.

    It is still opt-in. Nothing selects it unless AGENT_RUNTIME says so, and
    no agent carries a runtimeArn until something provisions one.
    """

    name = "agentcore"

    def __init__(self, client=None):
        self._client = client

    def _runtime_client(self):
        if self._client is None:
            self._client = boto3.client(
                "bedrock-agentcore",
                config=Config(
                    read_timeout=1800,
                    connect_timeout=60,
                    retries={"max_attempts": 0},
                ),
            )
        return self._client

    def missing_fields(self, ref: AgentRef) -> str:
        if not ref.runtime_arn:
            return "Missing runtimeArn in config for the agentcore runtime"
        return ""

    def build_payload(self, session_id: str, input_text: str) -> bytes:
        # The entrypoint receives this unchanged, so the shape is TeamWeave's
        # to define. `prompt` is what the agent reads; the rest is context an
        # agent may use and can otherwise ignore.
        return json.dumps(
            {"prompt": input_text, "sessionId": session_id},
            ensure_ascii=False,
        ).encode("utf-8")

    def invoke(
        self,
        ref: AgentRef,
        *,
        session_id: str,
        input_text: str,
        shadow_alias_id: Optional[str] = None,
    ) -> Tuple[str, dict]:
        if shadow_alias_id:
            # Dual-invoke drives DPO pair collection and is alias-shaped. On
            # AgentCore the equivalent is two qualifiers, which is a separate
            # piece of work -- so say so rather than silently collecting
            # nothing and letting the flywheel look healthy.
            log.warning(
                "shadow invocation is not implemented on the agentcore runtime; "
                "running the primary only",
                extra={"runtime_arn": ref.runtime_arn, "shadow_alias_id": shadow_alias_id},
            )

        resp, span_metrics = observe_agentcore_request(
            self._runtime_client(),
            runtime_arn=ref.runtime_arn,
            qualifier=ref.qualifier,
            session_id=session_id,
            runtime_session_id=agentcore_session_id(session_id),
            input_text=input_text,
            payload=self.build_payload(session_id, input_text),
        )

        status = resp.get("statusCode")
        if status is not None and int(status) >= 400:
            raise RuntimeError(f"InvokeAgentRuntime returned HTTP {status}")

        stream = resp.get("response")
        if stream is None:
            raise RuntimeError("InvokeAgentRuntime missing 'response' body")
        body = stream.read() if hasattr(stream, "read") else bytes(stream)
        return _extract_text(body), span_metrics


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
