"""The agent program, and the contract it shares with the caller.

The risk this file exists for: the two halves of the AgentCore path are
written in different places -- AgentCoreRuntime builds the payload and reads
the reply, the app reads the payload and builds the reply -- and nothing but a
test makes them agree. A mismatch would look like an agent that returns
nothing.
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from src.agentcore import agent as agent_app  # noqa: E402
from src.orchestrator import agent_runtime as ar  # noqa: E402


class FakeBedrock:
    def __init__(self, text="answer"):
        self.text, self.calls = text, []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {"output": {"message": {"content": [{"text": self.text}]}}}


# ── the shared payload contract ─────────────────────────────────────────────


def test_the_runtime_payload_is_what_the_app_reads():
    # The contract, asserted across both halves rather than assumed.
    payload = ar.AgentCoreRuntime().build_payload("run-1", "do the thing")
    assert agent_app.extract_prompt(json.loads(payload)) == "do the thing"


def test_the_app_reply_is_what_the_runtime_extracts():
    reply = agent_app.run_turn({"prompt": "hi"}, client=FakeBedrock("the answer"), env={})
    assert ar._extract_text(json.dumps(reply).encode()) == "the answer"


def test_the_round_trip_survives_non_ascii():
    payload = ar.AgentCoreRuntime().build_payload("run-1", "héllo — ünicode")
    assert agent_app.extract_prompt(json.loads(payload)) == "héllo — ünicode"


# ── reading the payload ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"prompt": "a"}, "a"),
        ({"inputText": "b"}, "b"),
        ({"input": "c"}, "c"),
        ("bare string", "bare string"),
        (b'{"prompt": "from bytes"}', "from bytes"),
        ('{"prompt": "from json string"}', "from json string"),
        ({}, ""),
        ({"prompt": "   "}, ""),
        (None, ""),
    ],
)
def test_prompt_extraction(payload, expected):
    assert agent_app.extract_prompt(payload) == expected


def test_a_blank_prompt_reports_the_caller_error(monkeypatch):
    # Returning an empty answer instead would surface downstream as a schema
    # failure blamed on the model rather than on the request.
    called = []
    out = agent_app.run_turn({}, client=type("C", (), {"converse": lambda s, **k: called.append(k)})(), env={})
    assert out["result"] == "" and "error" in out
    assert called == [], "a blank prompt must not reach the model"


# ── the request it builds ───────────────────────────────────────────────────


def test_the_instruction_becomes_the_system_prompt():
    client = FakeBedrock()
    agent_app.run_turn({"prompt": "p"}, client=client, env={"AGENT_INSTRUCTION": "You are terse."})
    assert client.calls[0]["system"] == [{"text": "You are terse."}]


def test_a_per_turn_instruction_beats_the_runtime_default():
    # This is what lets one runtime serve every agent: identity arrives with
    # the request instead of being baked into the deployment.
    client = FakeBedrock()
    agent_app.run_turn(
        {"prompt": "p", "instruction": "You are the editor."},
        client=client,
        env={"AGENT_INSTRUCTION": "You are the writer."},
    )
    assert client.calls[0]["system"] == [{"text": "You are the editor."}]


def test_with_no_instruction_anywhere_the_contract_still_holds():
    # Never an empty system block -- some models reject one -- and never
    # silently unconstrained: the JSON output contract is what the worker
    # validates against afterwards.
    client = FakeBedrock()
    agent_app.run_turn({"prompt": "p"}, client=client, env={})
    system = client.calls[0]["system"][0]["text"]
    assert system == agent_app.DEFAULT_INSTRUCTION
    assert "JSON" in system


def test_the_model_id_comes_from_the_environment():
    client = FakeBedrock()
    agent_app.run_turn({"prompt": "p"}, client=client, env={"AGENT_MODEL_ID": "anthropic.claude-3-haiku"})
    assert client.calls[0]["modelId"] == "anthropic.claude-3-haiku"


def test_the_model_id_falls_back_to_a_default():
    client = FakeBedrock()
    out = agent_app.run_turn({"prompt": "p"}, client=client, env={})
    assert client.calls[0]["modelId"] == agent_app.DEFAULT_MODEL_ID
    assert out["modelId"] == agent_app.DEFAULT_MODEL_ID


def test_a_nonsense_max_tokens_falls_back_rather_than_crashing():
    client = FakeBedrock()
    agent_app.run_turn({"prompt": "p"}, client=client, env={"AGENT_MAX_TOKENS": "lots"})
    assert client.calls[0]["inferenceConfig"]["maxTokens"] == agent_app.DEFAULT_MAX_TOKENS


def test_the_prompt_is_sent_verbatim():
    # prompt_builder already composed role, goal and output contract; altering
    # it here would give the two substrates different behaviour.
    client = FakeBedrock()
    prompt = "ROLE: x\nSTEP_GOAL:\ndo it\nOUTPUT CONTRACT:\nJSON only"
    agent_app.run_turn({"prompt": prompt}, client=client, env={})
    assert client.calls[0]["messages"][0]["content"][0]["text"] == prompt


# ── reading the model's reply ───────────────────────────────────────────────


def test_multiple_content_blocks_are_joined():
    client = FakeBedrock()
    client.converse = lambda **k: {"output": {"message": {"content": [{"text": "a"}, {"text": "b"}]}}}
    assert agent_app.run_turn({"prompt": "p"}, client=client, env={})["result"] == "ab"


@pytest.mark.parametrize(
    "response",
    [{}, {"output": {}}, {"output": {"message": {}}}, {"output": {"message": {"content": []}}},
     {"output": {"message": {"content": [{"noText": 1}]}}}],
)
def test_an_empty_or_odd_response_yields_an_empty_string_not_an_exception(response):
    client = FakeBedrock()
    client.converse = lambda _r=response, **k: _r
    assert agent_app.run_turn({"prompt": "p"}, client=client, env={})["result"] == ""
