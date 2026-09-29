"""device_controller calls DeviceWeave's real HTTP API and holds risky commands.

DeviceWeave (the deviceweave-prod stack) serves GET /devices, GET /devices/{id},
and POST /execute. It has no authorizer and no MCP server. The tool is stubbed
here at the HTTP transport so a test never touches a device.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from src.orchestrator import deviceweave, tool_rules
from src.orchestrator.model_map import categories
from src.orchestrator.tool_registry import TOOL_REGISTRY, execute_tool

REPO = Path(__file__).resolve().parents[1]
TEAM = REPO / "config" / "examples" / "teams" / "device_controller" / "v1" / "team.json"
SCHEMA = REPO / "config" / "examples" / "schemas" / "device_action_v1.json"
TEMPLATE = (REPO / "infra" / "template.yaml").read_text()
WORKFLOW = (REPO / ".github" / "workflows" / "deploy.yml").read_text()

ENV = {"DEVICEWEAVE_URL": "https://deviceweave.test/prod"}

OFFICE = {
    "id": "office_light",
    "name": "Office Light",
    "device_type": "SmartBulb",
    "capabilities": ["turn_on", "turn_off", "set_brightness", "get_status"],
}
LOCK = {
    "id": "front_door",
    "name": "Front Door",
    "device_type": "WyzeLock",
    "capabilities": ["lock", "unlock", "get_status"],
}


class CfnLoader(yaml.SafeLoader):
    pass


def _keep(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {"__fn__": suffix, "__arg__": value}


CfnLoader.add_multi_constructor("!", _keep)


def _load_team():
    return json.loads(TEAM.read_text())


def _schema():
    return json.loads(SCHEMA.read_text())


def transport(catalog, *, execute_status=200, execute_body=None, status_body=None):
    """Records every call and answers list, execute, and the follow-up status."""
    calls = []

    def _send(method, url, body, headers):
        calls.append({"method": method, "url": url, "body": body, "headers": headers})
        if method == "GET" and url.endswith("/devices"):
            return 200, {"devices": catalog, "count": len(catalog)}
        if method == "GET" and "/devices/" in url:
            device_id = url.rsplit("/", 1)[-1]
            found = next((d for d in catalog if d["id"] == device_id), None)
            if found is None:
                return 404, {"error": f"Device '{device_id}' not found."}
            return 200, found
        if method == "POST" and url.endswith("/execute"):
            command = (body or {}).get("command") or ""
            if command.startswith("status of "):
                return 200, status_body or {
                    "type": "device",
                    "device_id": "office_light",
                    "device_name": command[len("status of "):],
                    "action": "get_status",
                    "result": {"state": "on", "changed": False},
                }
            if execute_status >= 300:
                return execute_status, execute_body or {"error": "policy blocked"}
            return execute_status, execute_body or {
                "type": "device",
                "device_id": "office_light",
                "device_name": "Office Light",
                "action": "turn_on",
                "result": {"state": "on", "changed": True},
            }
        return 404, {"error": f"unexpected {method} {url}"}

    return calls, _send


def test_the_team_is_one_agent_on_a_known_category():
    doc = _load_team()
    assert doc["team"]["name"] == "device_controller"
    assert [a["id"] for a in doc["agents"]] == ["device_operator"]
    assert [s["step"] for s in doc["workflow"]] == ["device_operator"]
    agent = doc["agents"][0]
    assert agent["model_category"] == "planning"
    assert agent["model_category"] in categories()
    assert "model_id" not in agent["bedrock"]
    assert doc["workflow"][0]["pre_tools"][0]["name"] == "deviceweave"


def test_the_output_schema_requires_the_safety_fields():
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    valid = {
        "summary": "Office Light is on.",
        "actions_taken": [{"device": "Office Light", "command": "turn_on", "result": "on"}],
        "current_state": [{"device": "Office Light", "state": "on"}],
        "assumptions": [],
        "errors": [],
        "needs_confirmation": [],
    }
    Draft202012Validator(schema).validate(valid)
    held = dict(valid)
    held["actions_taken"] = []
    held["current_state"] = []
    held["needs_confirmation"] = [{
        "device": "Front Door",
        "command": "Unlock the front door",
        "reason": "unlock",
    }]
    Draft202012Validator(schema).validate(held)
    missing = dict(valid)
    del missing["needs_confirmation"]
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(missing)


def test_a_clear_command_lists_then_executes_then_reads_status():
    calls, send = transport([OFFICE])
    out = deviceweave.act(
        "Turn on the office light", confirm=None, transport=send, env=ENV,
    )
    assert [c["method"] + " " + c["url"].split("/prod", 1)[-1] for c in calls] == [
        "GET /devices",
        "POST /execute",
        "POST /execute",
    ]
    assert calls[1]["body"] == {"command": "Turn on the office light"}
    assert calls[2]["body"] == {"command": "status of Office Light"}
    assert "Authorization" not in calls[0]["headers"]
    assert out["executed"] is True
    assert "error" not in out
    assert out["actions_taken"] == [{
        "device": "Office Light", "command": "turn_on",
        "result": '{"changed": true, "state": "on"}',
    }]
    assert out["current_state"] == [{"device": "Office Light", "state": "on"}]
    assert out["needs_confirmation"] == []


def test_unlock_is_not_sent_without_confirm():
    calls, send = transport([LOCK, OFFICE])
    out = deviceweave.act("Unlock the front door", transport=send, env=ENV)
    assert [c["url"].split("/prod", 1)[-1] for c in calls] == ["/devices"]
    assert out["executed"] is False
    assert out["actions_taken"] == []
    assert out["needs_confirmation"][0]["device"] == "Front Door"
    assert out["needs_confirmation"][0]["reason"] == "unlock"
    assert "error" not in out


def test_confirm_yes_sends_the_unlock():
    calls, send = transport(
        [LOCK],
        execute_body={
            "type": "device",
            "device_id": "front_door",
            "device_name": "Front Door",
            "action": "unlock",
            "result": {"state": "unlocked", "changed": True},
        },
        status_body={
            "type": "device",
            "device_name": "Front Door",
            "action": "get_status",
            "result": {"state": "unlocked"},
        },
    )
    out = deviceweave.act("Unlock the front door", confirm="yes", transport=send, env=ENV)
    posts = [c for c in calls if c["method"] == "POST"]
    assert posts[0]["body"] == {"command": "Unlock the front door"}
    assert out["executed"] is True
    assert out["needs_confirmation"] == []
    assert out["actions_taken"][0]["command"] == "unlock"
    assert any("confirm was set" in item for item in out["assumptions"])


@pytest.mark.parametrize("instruction,reason", [
    ("factory reset the hub", "factory reset"),
    ("disable the security system", "disable security"),
    ("turn off the camera", "power off critical device"),
    ("open the garage", "unlock"),
    ("power off the Front Door", "power off critical device"),
])
def test_the_other_risky_commands_are_held(instruction, reason):
    calls, send = transport([LOCK, OFFICE])
    out = deviceweave.send_command(
        instruction, confirm="", devices=[LOCK, OFFICE], transport=send, env=ENV,
    )
    assert calls == []
    assert out["executed"] is False
    assert reason in out["needs_confirmation"][0]["reason"]


def test_turning_off_a_light_is_not_held():
    calls, send = transport([OFFICE, LOCK])
    out = deviceweave.act("turn off the office light", transport=send, env=ENV)
    assert any(c["method"] == "POST" for c in calls)
    assert out["needs_confirmation"] == []
    assert out["executed"] is True


def test_an_unconfigured_url_is_an_error_and_does_not_call():
    calls, send = transport([OFFICE])
    out = deviceweave.act("Turn on the office light", transport=send, env={})
    assert calls == []
    assert "DEVICEWEAVE_URL" in out["error"]
    assert out["executed"] is False


def test_a_secret_is_sent_as_a_bearer_and_never_inlined():
    source = (REPO / "src" / "orchestrator" / "deviceweave.py").read_text()
    assert "Bearer {token}" in source
    assert "sk-" not in source

    def reader(arn):
        assert arn == "arn:aws:secretsmanager:us-east-1:1:secret:deviceweave"
        return "from-secrets-manager"

    built = deviceweave.headers(
        {"DEVICEWEAVE_SECRET_ARN": "arn:aws:secretsmanager:us-east-1:1:secret:deviceweave"},
        secret_reader=reader,
    )
    assert built["Authorization"] == "Bearer from-secrets-manager"
    assert "Authorization" not in deviceweave.headers({})


def test_the_tool_is_registered_and_refused_for_other_teams():
    assert "deviceweave" in TOOL_REGISTRY
    rule = tool_rules.rule_for("deviceweave")
    assert rule.only_teams == ("device_controller",)
    assert rule.transport == tool_rules.HTTP
    assert rule.effect == tool_rules.ACT
    assert "POST /execute" in rule.mcp_tool
    with pytest.raises(PermissionError) as raised:
        execute_tool("deviceweave", {"command": "turn on the light"}, team="linkedin_quick_post")
    assert "device_controller" in str(raised.value)


def test_the_registered_tool_calls_the_stub(monkeypatch):
    calls, send = transport([OFFICE])
    monkeypatch.setattr(deviceweave, "_http", send)
    monkeypatch.setenv("DEVICEWEAVE_URL", ENV["DEVICEWEAVE_URL"])
    monkeypatch.delenv("DEVICEWEAVE_SECRET_ARN", raising=False)
    out = execute_tool(
        "deviceweave",
        {"request": {"instruction": "Turn on the office light"}},
        team="device_controller",
    )
    assert out["executed"] is True
    assert out["used_for"]
    assert calls[0]["url"] == "https://deviceweave.test/prod/devices"


def test_status_reads_the_registry_record_then_live_state():
    calls, send = transport([OFFICE])
    out = deviceweave.run(
        "status", device_id="office_light", transport=send, env=ENV,
    )
    paths = [c["method"] + " " + c["url"].split("/prod", 1)[-1] for c in calls]
    assert paths == ["GET /devices/office_light", "POST /execute"]
    assert calls[1]["body"]["command"] == "status of Office Light"
    assert out["current_state"][0]["state"] == "on"


def test_the_worker_is_given_the_url_from_the_deviceweave_stack():
    assert "DeviceWeaveApiBaseUrl:" in TEMPLATE
    assert "DEVICEWEAVE_URL: !Ref DeviceWeaveApiBaseUrl" in TEMPLATE
    assert "DeviceWeaveSecretArn:" in TEMPLATE
    assert 'AgentRuntimeName: "teamweave_device_controller"' in TEMPLATE
    assert "device_controller" in TEMPLATE
    # Provenance: DeviceWeave's deploy workflow, stage default prod.
    assert "DEVICEWEAVE_STACK_NAME: deviceweave-prod" in WORKFLOW
    assert "DeviceWeaveApiBaseUrl=${DEVICEWEAVE_URL}" in WORKFLOW
    assert "ApiBaseUrl" in WORKFLOW
    doc = yaml.load(TEMPLATE, Loader=CfnLoader)
    runtime = doc["Resources"]["AgentCoreRuntimeDeviceController"]["Properties"]
    assert runtime["EnvironmentVariables"]["AGENT_TEAM"] == "device_controller"
    worker = doc["Resources"]["WorkerFunction"]["Properties"]["Environment"]["Variables"]
    assert "DEVICEWEAVE_URL" in worker
    assert "DEVICEWEAVE_SECRET_ARN" in worker
