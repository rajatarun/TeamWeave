"""The substrate seam.

Bedrock Agents Classic takes no further features and its model catalogue is
frozen, so the substrate under an agent turn has to be replaceable. These
tests pin the two properties that make that safe: Classic stays the default
so nothing changes until someone opts in, and the parts that are *not*
substrate concerns -- the retry policy, the gate, the StepFailed contract --
stay shared rather than being reimplemented per runtime.

No AWS: the runtime is stubbed.
"""
from __future__ import annotations

import importlib
import os

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

agent_runtime = importlib.import_module("src.orchestrator.agent_runtime")
bedrock_invoke = importlib.import_module("src.orchestrator.bedrock_invoke")
from src.orchestrator.models import StepFailed  # noqa: E402


@pytest.fixture(autouse=True)
def clear_runtime_env(monkeypatch):
    monkeypatch.delenv("AGENT_RUNTIME", raising=False)


class StubRuntime:
    """Records calls and returns a canned answer, or raises a queued error."""

    name = "stub"

    def __init__(self, result=("ok", {"composite_risk_score": 0.1}), errors=None):
        self.result = result
        self.errors = list(errors or [])
        self.calls = []

    def missing_fields(self, ref):
        return ""

    def invoke(self, ref, *, session_id, input_text, shadow_alias_id=None):
        self.calls.append((ref, session_id, input_text, shadow_alias_id))
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def use(monkeypatch, runtime):
    monkeypatch.setattr(bedrock_invoke, "get_runtime", lambda: runtime)
    return runtime


# ── selection ────────────────────────────────────────────────────────────────


def test_classic_is_still_reachable_by_name():
    # It runs the agents deployed before the switch, and is the rollback.
    assert isinstance(
        agent_runtime._RUNTIMES["classic"](), agent_runtime.BedrockAgentsClassicRuntime
    )


def test_agentcore_is_selectable_by_name(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME", "agentcore")
    assert isinstance(agent_runtime.get_runtime(), agent_runtime.AgentCoreRuntime)


def test_selection_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME", "  AgentCore ")
    assert agent_runtime.runtime_name() == "agentcore"


def test_an_empty_runtime_var_falls_back_to_the_default(monkeypatch):
    # A typo or a blank should not take the orchestrator down.
    monkeypatch.setenv("AGENT_RUNTIME", "")
    assert agent_runtime.runtime_name() == agent_runtime.DEFAULT_RUNTIME


def test_agentcore_is_implemented_and_is_now_the_default():
    # It was a NotImplementedError placeholder until the service model gave
    # the real envelope. See tests/test_agentcore_runtime.py.
    assert hasattr(agent_runtime.AgentCoreRuntime(), "invoke")
    assert agent_runtime.DEFAULT_RUNTIME == "agentcore"


# ── what each runtime needs to be usable ─────────────────────────────────────


def test_classic_requires_agent_and_alias():
    runtime = agent_runtime.BedrockAgentsClassicRuntime()
    assert runtime.missing_fields(agent_runtime.AgentRef(agent_id="a", alias_id="b")) == ""
    assert runtime.missing_fields(agent_runtime.AgentRef(agent_id="a")) != ""
    assert runtime.missing_fields(agent_runtime.AgentRef()) != ""


def test_agentcore_requires_a_runtime_arn():
    runtime = agent_runtime.AgentCoreRuntime()
    assert runtime.missing_fields(agent_runtime.AgentRef(runtime_arn="arn:x")) == ""
    # A Classic-shaped ref is not usable on AgentCore, and saying so early is
    # the difference between a config error and a confusing runtime failure.
    assert runtime.missing_fields(agent_runtime.AgentRef(agent_id="a", alias_id="b")) != ""


# ── the shared parts stay shared ─────────────────────────────────────────────


def test_invoke_agent_returns_only_the_text(monkeypatch):
    use(monkeypatch, StubRuntime(result=("hello", {"composite_risk_score": 0.4})))
    assert bedrock_invoke.invoke_agent("a", "b", "s", "prompt") == "hello"


def test_invoke_agent_with_metrics_returns_the_span_too(monkeypatch):
    use(monkeypatch, StubRuntime(result=("hello", {"composite_risk_score": 0.4})))
    text, span = bedrock_invoke.invoke_agent_with_metrics("a", "b", "s", "prompt")
    assert text == "hello"
    # DPO ranking reads this; losing it would silently stop collecting pairs.
    assert span["composite_risk_score"] == 0.4


def test_a_missing_ref_fails_before_any_call(monkeypatch):
    stub = use(monkeypatch, StubRuntime())
    monkeypatch.setattr(stub, "missing_fields", lambda ref: "Missing agentId/aliasId in config")
    with pytest.raises(StepFailed):
        bedrock_invoke.invoke_agent("", "", "s", "prompt")
    assert stub.calls == []


def test_a_transient_error_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(bedrock_invoke.time, "sleep", lambda _s: None)
    stub = use(monkeypatch, StubRuntime(errors=[RuntimeError("boom")]))
    assert bedrock_invoke.invoke_agent("a", "b", "s", "p", max_retries=2) == "ok"
    assert len(stub.calls) == 2


def test_retries_are_bounded_and_then_raise(monkeypatch):
    monkeypatch.setattr(bedrock_invoke.time, "sleep", lambda _s: None)
    stub = use(monkeypatch, StubRuntime(errors=[RuntimeError("x")] * 5))
    with pytest.raises(StepFailed):
        bedrock_invoke.invoke_agent("a", "b", "s", "p", max_retries=2)
    assert len(stub.calls) == 3  # the first attempt plus two retries


def test_the_shadow_alias_reaches_the_runtime(monkeypatch):
    # Dual-invoke drives the DPO pairs; dropping it here would disable the
    # flywheel without failing anything.
    stub = use(monkeypatch, StubRuntime())
    bedrock_invoke.invoke_agent("a", "b", "s", "p", shadow_alias_id="shadow-1")
    assert stub.calls[0][3] == "shadow-1"


def test_an_auth_failure_is_not_retried(monkeypatch):
    from botocore.exceptions import ClientError

    monkeypatch.setattr(bedrock_invoke.time, "sleep", lambda _s: None)
    err = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "InvokeAgent")
    stub = use(monkeypatch, StubRuntime(errors=[err] * 5))
    with pytest.raises(StepFailed) as excinfo:
        bedrock_invoke.invoke_agent("a", "b", "s", "p", max_retries=3)
    # Retrying a permission error just burns the budget and delays the signal.
    assert len(stub.calls) == 1
    assert "AccessDenied" in str(excinfo.value)


def test_with_metrics_propagates_step_failed_immediately(monkeypatch):
    monkeypatch.setattr(bedrock_invoke.time, "sleep", lambda _s: None)
    stub = use(monkeypatch, StubRuntime(errors=[StepFailed("x", "inner")] * 5))
    with pytest.raises(StepFailed):
        bedrock_invoke.invoke_agent_with_metrics("a", "b", "s", "p", max_retries=3)
    assert len(stub.calls) == 1


def test_invoke_agent_retries_a_step_failed_as_it_always_did(monkeypatch):
    # Preserved difference, not endorsed: the two loops behaved differently
    # here before they were merged, and Phase 0 changes no behaviour.
    monkeypatch.setattr(bedrock_invoke.time, "sleep", lambda _s: None)
    stub = use(monkeypatch, StubRuntime(errors=[StepFailed("x", "inner")] * 5))
    with pytest.raises(StepFailed):
        bedrock_invoke.invoke_agent("a", "b", "s", "p", max_retries=3)
    assert len(stub.calls) == 4


def test_brt_is_still_reachable_from_bedrock_invoke():
    # Existing tests patch bedrock_invoke.brt; it must stay the same object
    # the Classic runtime actually calls, or patching silently does nothing.
    assert bedrock_invoke.brt is agent_runtime.brt


def test_the_give_up_message_names_the_last_error(monkeypatch):
    # "failed after retries: None" tells an operator nothing. Every retry path
    # has to leave the cause behind, including the StepFailed one.
    monkeypatch.setattr(bedrock_invoke.time, "sleep", lambda _s: None)
    use(monkeypatch, StubRuntime(errors=[StepFailed("inner_op", "the real reason")] * 5))
    with pytest.raises(StepFailed) as excinfo:
        bedrock_invoke.invoke_agent("a", "b", "s", "p", max_retries=1)
    assert "the real reason" in str(excinfo.value)
    assert "None" not in str(excinfo.value)


def test_the_give_up_message_names_a_transport_error_too(monkeypatch):
    monkeypatch.setattr(bedrock_invoke.time, "sleep", lambda _s: None)
    use(monkeypatch, StubRuntime(errors=[RuntimeError("connection reset")] * 5))
    with pytest.raises(StepFailed) as excinfo:
        bedrock_invoke.invoke_agent("a", "b", "s", "p", max_retries=1)
    assert "connection reset" in str(excinfo.value)


def test_agentcore_is_now_the_default_and_classic_the_rollback(monkeypatch):
    # Bedrock Agents Classic is in maintenance and its model catalogue is
    # frozen, so it is no longer where new work goes. It stays reachable for
    # one variable's worth of rollback until AgentCore has served real
    # traffic; it takes no new features.
    monkeypatch.delenv("AGENT_RUNTIME", raising=False)
    assert agent_runtime.runtime_name() == "agentcore"
    monkeypatch.setenv("AGENT_RUNTIME", "classic")
    assert isinstance(agent_runtime.get_runtime(), agent_runtime.BedrockAgentsClassicRuntime)


def test_an_unknown_runtime_now_falls_back_to_agentcore(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME", "agentcorre")
    assert agent_runtime.runtime_name() == "agentcore"
