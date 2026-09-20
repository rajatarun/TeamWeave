"""The AgentCore runtime, against the shape botocore actually declares.

Written from the bedrock-agentcore service model (2024-02-28), not from
recollection. The details that a guessed implementation gets wrong, and that
only fail once deployed, are pinned here: runtimeSessionId has a minimum
length of 33, the qualifier is optional, and the response body is a streaming
blob rather than an event stream.
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from src.orchestrator import agent_runtime as ar  # noqa: E402


class FakeStream:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body


class FakeClient:
    def __init__(self, body=b'{"result": "hi"}', status=200):
        self.body, self.status, self.calls = body, status, []

    def invoke_agent_runtime(self, **kwargs):
        self.calls.append(kwargs)
        return {"statusCode": self.status, "response": FakeStream(self.body),
                "contentType": "application/json"}


@pytest.fixture
def runtime(monkeypatch):
    """An AgentCoreRuntime whose gate is stubbed but still routed through."""
    seen = {}

    def fake_observe(client, **kw):
        seen.update(kw)
        payload_kwargs = {
            "agentRuntimeArn": kw["runtime_arn"],
            "runtimeSessionId": kw["runtime_session_id"],
            "payload": kw["payload"],
            "contentType": "application/json",
            "accept": "application/json",
        }
        if kw["qualifier"]:
            payload_kwargs["qualifier"] = kw["qualifier"]
        return client.invoke_agent_runtime(**payload_kwargs), {"composite_risk_score": 0.2}

    monkeypatch.setattr(ar, "observe_agentcore_request", fake_observe)
    client = FakeClient()
    return ar.AgentCoreRuntime(client=client), client, seen


REF = None


def ref(**kw):
    return ar.AgentRef(runtime_arn="arn:aws:bedrock-agentcore:us-east-1:1:runtime/r", **kw)


# ── the session id window ────────────────────────────────────────────────────


def test_a_short_session_id_is_padded_to_the_minimum():
    # min 33 in the service model. A shorter one is a ValidationException at
    # the API, which is a deployment-time failure, not a local one.
    out = ar.agentcore_session_id("run-1")
    assert 33 <= len(out) <= 256


def test_padding_is_deterministic():
    # Repeated turns of one run must land on the same AgentCore session, or
    # the runtime's conversational memory attaches to the wrong conversation.
    assert ar.agentcore_session_id("run-1") == ar.agentcore_session_id("run-1")
    assert ar.agentcore_session_id("run-1") != ar.agentcore_session_id("run-2")


def test_an_already_long_session_id_is_left_alone():
    long_id = "a" * 40
    assert ar.agentcore_session_id(long_id) == long_id


def test_an_over_long_session_id_is_capped():
    assert len(ar.agentcore_session_id("b" * 400)) == 256


def test_illegal_characters_are_replaced():
    out = ar.agentcore_session_id("run/123 456#x")
    assert "/" not in out and " " not in out and "#" not in out


def test_an_empty_session_id_still_produces_a_valid_one():
    assert 33 <= len(ar.agentcore_session_id("")) <= 256


# ── the request ──────────────────────────────────────────────────────────────


def test_invoke_sends_the_arn_session_and_payload(runtime):
    rt, client, _ = runtime
    rt.invoke(ref(), session_id="run-1", input_text="hello")
    call = client.calls[0]
    assert call["agentRuntimeArn"].endswith("runtime/r")
    assert len(call["runtimeSessionId"]) >= 33
    assert json.loads(call["payload"])["prompt"] == "hello"
    assert call["contentType"] == "application/json"


def _call_real_observe(monkeypatch, qualifier):
    """Drive the real observe_agentcore_request with only the gate stubbed."""
    from src.orchestrator import mcp_observatory as obs

    client = FakeClient()

    class _Result:
        output = {"statusCode": 200, "response": FakeStream(b'{"result":"ok"}')}

        class span:
            trace_id = "t"; prompt_tokens = 1; completion_tokens = 1; cost_usd = 0.0
            shadow_disagreement_score = None

        class decision:
            action = "ALLOW"; reason = "ok"

    async def fake_invoke(**kw):
        kw["call"]()          # run the real call lambda -> hits FakeClient
        return _Result()

    monkeypatch.setattr(obs._wrapper, "invoke", fake_invoke)
    monkeypatch.setattr(obs, "_push_metric", lambda *a, **k: None)
    monkeypatch.setattr(obs, "_get_plain_span_metrics", lambda span: {})

    obs.observe_agentcore_request(
        client,
        runtime_arn="arn:x",
        qualifier=qualifier,
        session_id="run-1",
        runtime_session_id="s" * 40,
        input_text="hello",
        payload=b"{}",
    )
    return client.calls[0]


def test_the_qualifier_is_omitted_when_unset(monkeypatch):
    # Sending qualifier="" is not the same request as omitting it, and the
    # API is entitled to reject it.
    assert "qualifier" not in _call_real_observe(monkeypatch, "")


def test_the_qualifier_is_sent_when_set(monkeypatch):
    assert _call_real_observe(monkeypatch, "PROD")["qualifier"] == "PROD"


def test_the_real_request_carries_the_model_required_fields(monkeypatch):
    call = _call_real_observe(monkeypatch, "")
    for field in ("agentRuntimeArn", "runtimeSessionId", "payload"):
        assert field in call, f"{field} is required by the service model"


def test_the_payload_is_bytes_not_a_dict(runtime):
    # The model types payload as a blob; handing boto a dict is a runtime error.
    rt, client, _ = runtime
    rt.invoke(ref(), session_id="run-1", input_text="x")
    assert isinstance(client.calls[0]["payload"], (bytes, bytearray))


def test_the_call_goes_through_the_gate(runtime):
    # A migrated agent must not stop producing spans; the gate would then
    # cover less of the platform than the dashboards claim.
    rt, _, seen = runtime
    _, span = rt.invoke(ref(), session_id="run-1", input_text="x")
    assert seen["runtime_arn"].endswith("runtime/r")
    assert span["composite_risk_score"] == 0.2


# ── the response ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "body,expected",
    [
        (b'{"result": "a"}', "a"),
        (b'{"output": "b"}', "b"),
        (b'{"response": "c"}', "c"),
        (b'"plain json string"', "plain json string"),
        (b"not json at all", "not json at all"),
        (b"", ""),
    ],
)
def test_the_text_is_pulled_out_of_whatever_came_back(body, expected):
    # The entrypoint returns whatever it likes; JSON is a convention, so a
    # plain-text agent must not fail on json.loads.
    assert ar._extract_text(body) == expected


def test_an_unrecognised_json_object_is_preserved_not_dropped():
    out = ar._extract_text(b'{"unexpected": 1}')
    assert json.loads(out) == {"unexpected": 1}


def test_the_stream_is_read_once_not_iterated(runtime):
    rt, client, _ = runtime
    client.body = b'{"result": "streamed"}'
    text, _ = rt.invoke(ref(), session_id="run-1", input_text="x")
    assert text == "streamed"


def test_an_http_error_status_raises(runtime):
    rt, client, _ = runtime
    client.status = 424  # RuntimeClientError, per the model's error list
    with pytest.raises(RuntimeError):
        rt.invoke(ref(), session_id="run-1", input_text="x")


def test_a_missing_response_body_raises(runtime, monkeypatch):
    rt, client, _ = runtime
    monkeypatch.setattr(ar, "observe_agentcore_request",
                        lambda c, **kw: ({"statusCode": 200}, {}))
    with pytest.raises(RuntimeError):
        rt.invoke(ref(), session_id="run-1", input_text="x")


# ── honesty about what is not ported ────────────────────────────────────────


def test_a_shadow_alias_warns_rather_than_silently_doing_nothing(runtime, caplog):
    # Dual-invoke drives DPO collection. Ignoring it quietly would leave the
    # flywheel looking healthy while collecting nothing.
    rt, client, _ = runtime
    with caplog.at_level("WARNING"):
        rt.invoke(ref(), session_id="run-1", input_text="x", shadow_alias_id="shadow")
    assert any("shadow" in r.getMessage().lower() for r in caplog.records)
    assert len(client.calls) == 1


def test_a_classic_shaped_ref_is_rejected_early():
    rt = ar.AgentCoreRuntime()
    assert rt.missing_fields(ar.AgentRef(agent_id="a", alias_id="b")) != ""
    assert rt.missing_fields(ref()) == ""


# ── the wiring, end to end ──────────────────────────────────────────────────
#
# Every test above stubs the runtime, which is why none of them noticed that
# nothing ever put a runtimeArn into the AgentRef. Setting
# AGENT_RUNTIME=agentcore would have failed every step with "Missing
# runtimeArn" before a call was attempted. These cross the wiring instead.


def test_a_runtime_arn_from_team_json_reaches_the_agentcore_runtime(monkeypatch):
    from src.orchestrator import bedrock_invoke

    seen = {}

    class Recorder:
        name = "agentcore"

        def missing_fields(self, ref):
            return ar.AgentCoreRuntime().missing_fields(ref)

        def invoke(self, ref, *, session_id, input_text, shadow_alias_id=None):
            seen["ref"] = ref
            return "ok", {}

    monkeypatch.setattr(bedrock_invoke, "get_runtime", lambda: Recorder())
    text = bedrock_invoke.invoke_agent(
        "", "", "run-1", "hello",
        runtime_arn="arn:aws:bedrock-agentcore:us-east-1:1:runtime/r",
        qualifier="PROD",
    )
    assert text == "ok"
    assert seen["ref"].runtime_arn.endswith("runtime/r")
    assert seen["ref"].qualifier == "PROD"


def test_without_a_runtime_arn_the_agentcore_path_is_rejected_not_attempted(monkeypatch):
    from src.orchestrator import bedrock_invoke
    from src.orchestrator.models import StepFailed

    calls = []

    class Recorder(ar.AgentCoreRuntime):
        def invoke(self, *a, **k):
            calls.append(1)

    monkeypatch.setattr(bedrock_invoke, "get_runtime", lambda: Recorder())
    with pytest.raises(StepFailed, match="runtimeArn"):
        bedrock_invoke.invoke_agent("agent-1", "alias-1", "run-1", "hello")
    assert calls == [], "a ref with no runtimeArn must not reach the API"


def _load(raw, monkeypatch):
    """Run the real loader with S3 replaced, so the mapping is exercised."""
    from src.orchestrator import config_loader

    monkeypatch.setenv("CONFIG_BUCKET", "b")
    monkeypatch.setattr(config_loader, "_s3_get_json", lambda bucket, key: raw)
    team, _ = config_loader.load_team_config("t", "v1")
    return team


def _raw(bedrock):
    return {
        "team": {"name": "t", "version": "v1"},
        "globals": {},
        "agents": [{"id": "A1", "name": "A", "goal_template": "g",
                    "schema_ref": "s", "bedrock": bedrock}],
        "workflow": [],
        "schemas": {},
    }


def test_the_config_loader_carries_both_substrates_coordinates(monkeypatch):
    # team.json is where an agent's substrate is actually declared, so the
    # loader is the first place the coordinates can be lost.
    team = _load(_raw({
        "agentId": "AID", "aliasId": "ALID",
        "runtimeArn": "arn:aws:bedrock-agentcore:us-east-1:1:runtime/r",
        "qualifier": "PROD",
    }), monkeypatch)
    bedrock = team.agents[0].bedrock
    assert bedrock.agentId == "AID"
    assert bedrock.runtimeArn.endswith("runtime/r")
    assert bedrock.qualifier == "PROD"


def test_an_agent_without_agentcore_fields_still_loads(monkeypatch):
    # Every team.json in the repo predates AgentCore; none may break.
    bedrock = _load(_raw({"agentId": "AID", "aliasId": "ALID"}), monkeypatch).agents[0].bedrock
    assert bedrock.runtimeArn == "" and bedrock.qualifier == ""


def test_one_stack_runtime_serves_agents_with_no_runtime_arn(monkeypatch):
    # Classic needed one Bedrock agent per TeamWeave agent. On AgentCore the
    # identity is in the prompt, so a stack-level ARN is enough and no
    # team.json needs editing to move a team across.
    monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-east-1:1:runtime/shared")
    rt = ar.AgentCoreRuntime()
    assert rt.missing_fields(ar.AgentRef()) == ""
    assert rt.resolve_arn(ar.AgentRef()).endswith("runtime/shared")


def test_a_per_agent_runtime_arn_still_wins(monkeypatch):
    monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-east-1:1:runtime/shared")
    own = "arn:aws:bedrock-agentcore:us-east-1:1:runtime/own"
    assert ar.AgentCoreRuntime().resolve_arn(ar.AgentRef(runtime_arn=own)) == own


def test_with_neither_arn_the_error_says_how_to_fix_it(monkeypatch):
    monkeypatch.delenv("AGENTCORE_RUNTIME_ARN", raising=False)
    msg = ar.AgentCoreRuntime().missing_fields(ar.AgentRef())
    assert "AGENTCORE_RUNTIME_ARN" in msg and "runtimeArn" in msg


def test_the_payload_carries_the_instruction_when_given():
    import json as _json
    rt = ar.AgentCoreRuntime()
    body = _json.loads(rt.build_payload("s", "p", instruction="be terse"))
    assert body["instruction"] == "be terse"
    # Omitted rather than sent empty, so the app falls back to its default.
    assert "instruction" not in _json.loads(rt.build_payload("s", "p"))
