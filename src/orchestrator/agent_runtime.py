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
from typing import Dict, Optional, Protocol, Tuple

import boto3
from botocore.config import Config

from . import deadline
from .logger import get_logger
from .mcp_observatory import observe_agent_request, observe_agentcore_request

log = get_logger("agent_runtime")

# The Classic runtime client. Defined here but re-exported by
# ``bedrock_invoke`` so existing callers and tests that patch
# ``bedrock_invoke.brt`` keep working -- it is the same object.
# 1800 s here was six times the worker's entire Lambda budget, so a stalled
# call could never raise a read timeout -- Lambda killed the process first and
# the failure arrived as Sandbox.Timedout, naming no step. See deadline.py.
brt = boto3.client(
    "bedrock-agent-runtime",
    config=Config(
        read_timeout=deadline.DEFAULT_CALL_SECONDS,
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
    # Which team this turn belongs to. AgentCore runtimes are per team, so
    # this is how a turn finds the one that serves it.
    team: str = ""
    # The model this agent declares in team.json. On Classic the model is a
    # property of the Bedrock agent resource, so this is unused there. On
    # AgentCore one generic runtime serves every agent, so the model has to
    # travel with the turn -- exactly as the instruction does -- or every
    # agent silently runs on whatever AGENT_MODEL_ID the runtime was created
    # with, and the per-agent model_id in team.json means nothing.
    model_id: str = ""


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
        """A client whose read timeout fits the time this invocation has left.

        Built per call rather than cached, because the budget shrinks as the
        pipeline's earlier steps spend it: a client made for the first agent
        would let the last one outlive the function. An injected client (the
        tests') is always used as-is.
        """
        if self._client is not None:
            return self._client
        return boto3.client(
            "bedrock-agentcore",
            config=Config(
                read_timeout=deadline.budget_for_call(),
                connect_timeout=60,
                retries={"max_attempts": 0},
            ),
        )

    @staticmethod
    def team_runtime_arns() -> Dict[str, str]:
        """team name -> runtime ARN, as the stack publishes it.

        A JSON object in one variable rather than one variable per team:
        Lambda's environment is a flat map the template has to name
        statically, and a new team would otherwise need a new variable name
        wired through every function.

        Unparseable content yields {} rather than raising. The caller falls
        back to the stack-wide runtime, which is a worse answer than the
        right team's runtime and a far better one than failing every turn.
        """
        raw = (os.environ.get("AGENTCORE_TEAM_RUNTIME_ARNS") or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            log.warning("agentcore_team_runtime_arns_unparseable", extra={"raw": raw[:200]})
            return {}
        if not isinstance(parsed, dict):
            log.warning("agentcore_team_runtime_arns_not_an_object", extra={"raw": raw[:200]})
            return {}
        return {str(k): str(v) for k, v in parsed.items() if isinstance(v, str) and v.strip()}

    @classmethod
    def resolve_arn(cls, ref: AgentRef) -> str:
        """The runtime this turn runs on, most specific answer first.

        Each team gets its own AgentCore runtime. One runtime for the whole
        platform put every team's agents in one blast radius and one endpoint
        budget: the release channels a runtime has are ten, shared, so two
        teams could not be canaried independently and a bad version reached
        all of them at once. Per team, each has its own version history, its
        own channels and its own failure.

        Not per *agent*, which is the mistake this platform already made:
        identity is a name in a document (a skill id, gen_ai.agent.id), and
        prompt_builder composes ROLE and STEP_GOAL into the prompt, so one
        runtime serves every agent of a team without them being confusable.
        A team is a deployment unit; an agent is not.

        Order:
          1. the agent's own runtimeArn, for an agent that needs its own
          2. its team's runtime -- the normal case
          3. the stack-wide runtime, which keeps a team with no runtime of
             its own working instead of failing every turn
        """
        if ref.runtime_arn:
            return ref.runtime_arn
        by_team = cls.team_runtime_arns().get(ref.team, "")
        if by_team:
            return by_team
        return (os.environ.get("AGENTCORE_RUNTIME_ARN") or "").strip()

    def missing_fields(self, ref: AgentRef) -> str:
        if not self.resolve_arn(ref):
            known = sorted(self.team_runtime_arns())
            return (
                "No AgentCore runtime for team "
                f"{ref.team or '<unnamed>'}: the stack publishes runtimes for "
                f"{known or 'no teams'}, AGENTCORE_RUNTIME_ARN is unset, and the "
                "agent has no runtimeArn of its own in team.json"
            )
        return ""

    def build_payload(self, session_id: str, input_text: str, instruction: str = "",
                      model_id: str = "") -> bytes:
        # The entrypoint receives this unchanged, so the shape is TeamWeave's
        # to define. `prompt` is what the agent reads; `instruction` and
        # `modelId` are this turn's system prompt and model, which is what
        # lets one runtime serve every agent instead of one runtime per agent.
        # Both are omitted when empty so the runtime keeps its own default.
        body = {"prompt": input_text, "sessionId": session_id}
        if instruction:
            body["instruction"] = instruction
        if model_id:
            body["modelId"] = model_id
        return json.dumps(body, ensure_ascii=False).encode("utf-8")

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
            runtime_arn=self.resolve_arn(ref),
            qualifier=ref.qualifier,
            session_id=session_id,
            runtime_session_id=agentcore_session_id(session_id),
            input_text=input_text,
            # build_payload took an instruction from the day it was written and
            # invoke never passed one, so every turn fell back to the
            # runtime's AGENT_INSTRUCTION. The seam existed and was not
            # connected to anything.
            payload=self.build_payload(
                session_id, input_text, model_id=ref.model_id,
            ),
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

# AgentCore is the platform's substrate. Classic remains selectable --
# AGENT_RUNTIME=classic -- because it still runs the agents deployed before
# the switch, and a one-variable rollback is worth keeping until AgentCore has
# served real traffic. It takes no new work.
DEFAULT_RUNTIME = AgentCoreRuntime.name


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
