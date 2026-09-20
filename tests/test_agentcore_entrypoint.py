"""The entrypoint the runtime actually boots, over its actual HTTP surface.

Everything else about the AgentCore path is tested against stubs. This is the
one test that runs the real artifact: it imports `app.py` the way the platform
does, and drives `GET /ping` and `POST /invocations` through Starlette rather
than calling `run_turn` directly.

That matters because the failure it guards against is invisible to every other
test. `BedrockAgentCoreApp` extends Starlette, so the platform imports the
entrypoint file and serves what it finds; an entrypoint that builds its app
inside a function exposes nothing to serve, and a runtime that cannot boot is
only discovered at the first invocation -- long after the deploy reports
success.

Skipped where the SDK is absent; CI installs it, so there it runs.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# src.orchestrator.agent_runtime builds its boto3 client at import time, and a
# client with no region raises before a single assertion runs. Every sibling
# module sets this; without it this file passes in the full suite (some other
# module got there first) and fails when run on its own.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

pytest.importorskip("bedrock_agentcore", reason="AgentCore SDK not installed")
pytest.importorskip("httpx", reason="Starlette's TestClient needs httpx")

from starlette.testclient import TestClient  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
AGENTCORE_DIR = REPO / "src" / "agentcore"


@pytest.fixture(scope="module")
def deployed():
    """Import app.py as the runtime does: flat, with agent.py beside it."""
    sys.path.insert(0, str(AGENTCORE_DIR))
    for name in ("agent", "app"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location("app", AGENTCORE_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["app"] = module
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(AGENTCORE_DIR))
    for name in ("agent", "app"):
        sys.modules.pop(name, None)


@pytest.fixture
def bedrock(monkeypatch):
    import agent as agent_mod

    class FakeBedrock:
        def __init__(self):
            self.calls = []

        def converse(self, **kwargs):
            self.calls.append(kwargs)
            return {"output": {"message": {"content": [{"text": '{"ok": true}'}]}}}

    fake = FakeBedrock()
    monkeypatch.setattr(agent_mod, "_client", fake)
    return fake


def test_the_entrypoint_exposes_an_asgi_app_at_import_time(deployed):
    # Building it inside a function would leave the platform nothing to serve.
    assert hasattr(deployed, "app")
    assert callable(getattr(deployed, "invoke", None))


def test_ping_reports_healthy(deployed):
    # The runtime's health check. A failing ping takes the whole agent out.
    response = TestClient(deployed.app).get("/ping")
    assert response.status_code == 200
    assert "Healthy" in response.text


def test_invocations_returns_the_model_output(deployed, bedrock):
    response = TestClient(deployed.app).post(
        "/invocations",
        json={"prompt": "ROLE: Writer\nSTEP_GOAL:\nwrite it", "sessionId": "run-1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["result"] == '{"ok": true}'


def test_the_prompt_reaches_the_model_unaltered(deployed, bedrock):
    prompt = "ROLE: Writer\nSTEP_GOAL:\nwrite it\nOUTPUT CONTRACT:\nJSON only"
    TestClient(deployed.app).post("/invocations", json={"prompt": prompt})
    # prompt_builder already composed this; altering it here would give the
    # two substrates different behaviour.
    assert bedrock.calls[0]["messages"][0]["content"][0]["text"] == prompt


def test_a_per_turn_instruction_reaches_the_model(deployed, bedrock):
    # What lets one runtime serve every agent.
    TestClient(deployed.app).post(
        "/invocations", json={"prompt": "p", "instruction": "You are the editor."}
    )
    assert bedrock.calls[0]["system"] == [{"text": "You are the editor."}]


def test_the_default_instruction_holds_the_json_contract(deployed, bedrock):
    TestClient(deployed.app).post("/invocations", json={"prompt": "p"})
    assert "JSON" in bedrock.calls[0]["system"][0]["text"]


def test_the_response_shape_is_what_the_caller_parses(deployed, bedrock):
    from src.orchestrator import agent_runtime as ar

    response = TestClient(deployed.app).post("/invocations", json={"prompt": "p"})
    # The other half of the contract: AgentCoreRuntime reads this back.
    assert ar._extract_text(json.dumps(response.json()).encode()) == '{"ok": true}'


def test_a_blank_prompt_does_not_reach_the_model(deployed, bedrock):
    response = TestClient(deployed.app).post("/invocations", json={})
    assert response.status_code == 200
    assert "error" in response.json()
    assert bedrock.calls == []
