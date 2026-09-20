"""POST /agent/converse, across substrates.

This endpoint is what the admin console's chat drawer calls. It used to
require agent_id and alias_id unconditionally, which was correct for exactly
as long as Classic was the substrate: the moment AGENT_RUNTIME became
agentcore, every agent's agentId/aliasId went blank and every chat request
became a 400 for missing fields that no longer exist. The console's Chat
button greyed out and nothing reported an error, because nothing had failed --
the console had simply stopped asking.

So what these pin is that the endpoint asks the *runtime* what it needs.
"""
from __future__ import annotations

import json
import os
from unittest import mock

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from src.orchestrator import conversation_handler as ch  # noqa: E402
from src.orchestrator.models import StepFailed  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("AGENT_RUNTIME", raising=False)
    monkeypatch.delenv("AGENTCORE_RUNTIME_ARN", raising=False)


def post(body: dict) -> dict:
    event = {"requestContext": {"http": {"method": "POST"}}, "body": json.dumps(body)}
    return ch.handler(event, None)


def body_of(response: dict) -> dict:
    return json.loads(response["body"])


CHAT = {"session_id": "s-1", "message": "hello"}


def test_agentcore_needs_no_agent_coordinates(monkeypatch):
    # The whole point of the substrate switch: one runtime serves every agent,
    # so a console that knows only the agent's name can still talk to it.
    monkeypatch.setenv("AGENT_RUNTIME", "agentcore")
    monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-east-1:1:runtime/r")
    with mock.patch.object(ch, "invoke_agent", return_value="hi") as invoke:
        response = post(CHAT)
    assert response["statusCode"] == 200
    assert body_of(response)["response"] == "hi"
    assert invoke.called


def test_the_answering_substrate_is_reported(monkeypatch):
    # agent_id and alias_id come back empty on AgentCore, so they cannot be
    # what a console infers the substrate from.
    monkeypatch.setenv("AGENT_RUNTIME", "agentcore")
    monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-east-1:1:runtime/r")
    with mock.patch.object(ch, "invoke_agent", return_value="hi"):
        payload = body_of(post(CHAT))
    assert payload["runtime"] == "agentcore"


def test_a_per_agent_runtime_arn_is_forwarded(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME", "agentcore")
    with mock.patch.object(ch, "invoke_agent", return_value="hi") as invoke:
        response = post({**CHAT, "runtime_arn": "arn:own", "qualifier": "DEFAULT"})
    assert response["statusCode"] == 200
    assert invoke.call_args.kwargs["runtime_arn"] == "arn:own"
    assert invoke.call_args.kwargs["qualifier"] == "DEFAULT"


def test_agentcore_with_no_runtime_anywhere_says_both_ways_to_fix_it(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME", "agentcore")
    response = post(CHAT)
    assert response["statusCode"] == 400
    payload = body_of(response)
    assert payload["runtime"] == "agentcore"
    assert "AGENTCORE_RUNTIME_ARN" in payload["error"]
    assert "runtimeArn" in payload["error"]


def test_classic_still_requires_its_pair(monkeypatch):
    # Classic is deprecated, not deleted. AGENT_RUNTIME=classic is the
    # one-variable rollback, and it must not have been loosened on the way.
    monkeypatch.setenv("AGENT_RUNTIME", "classic")
    response = post(CHAT)
    assert response["statusCode"] == 400
    assert body_of(response)["runtime"] == "classic"


def test_classic_passes_its_pair_through(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME", "classic")
    with mock.patch.object(ch, "invoke_agent", return_value="hi") as invoke:
        response = post({**CHAT, "agent_id": "A1", "alias_id": "L1"})
    assert response["statusCode"] == 200
    assert invoke.call_args.args[:2] == ("A1", "L1")
    assert body_of(response)["agent_id"] == "A1"


@pytest.mark.parametrize("field", ["session_id", "message"])
def test_the_two_fields_every_substrate_needs(monkeypatch, field):
    monkeypatch.setenv("AGENT_RUNTIME", "agentcore")
    monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-east-1:1:runtime/r")
    response = post({**CHAT, field: ""})
    assert response["statusCode"] == 400
    assert field in body_of(response)["error"]


def test_a_failed_turn_is_a_502_that_names_the_substrate(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME", "agentcore")
    monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-east-1:1:runtime/r")
    with mock.patch.object(ch, "invoke_agent", side_effect=StepFailed("invoke_agent", "boom")):
        response = post(CHAT)
    assert response["statusCode"] == 502
    payload = body_of(response)
    assert payload["runtime"] == "agentcore"
    assert "boom" in payload["error"]
