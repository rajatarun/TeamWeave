"""Sibling MCP tools, the rules that gate them, and the teams that use them.

The gateway is not the path here. `src/agentcore/agent.py` calls Converse with
no `toolConfig`, so an agent turn cannot invoke a gateway target -- a team
declaring one would be dead config of exactly the kind this repository keeps
finding. These tools run in `tool_registry`, before or after a turn, which is
the path that executes today.
"""
from __future__ import annotations

import io
import json
import pathlib

import pytest

from src.orchestrator import mcp_client, tool_rules
from src.orchestrator.tool_registry import TOOL_REGISTRY, execute_tool
from src.orchestrator.tools import weave_tools

REPO = pathlib.Path(__file__).resolve().parents[1]
TEAMS = REPO / "config" / "examples" / "teams"


class FakeResponse(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *_e): self.close(); return False


def opener(*bodies, captured=None):
    """Answers initialize then tools/call, in order."""
    queue = list(bodies)
    def _open(request, timeout=None):
        if captured is not None:
            captured.append(json.loads(request.data))
        return FakeResponse(queue.pop(0).encode())
    return _open


HELLO = json.dumps({"jsonrpc": "2.0", "id": 1,
                    "result": {"protocolVersion": mcp_client.PROTOCOL_VERSION}})


def tool_result(payload, *, is_error=False):
    result = {"content": [{"type": "text", "text": json.dumps(payload)}]}
    if is_error:
        result["isError"] = True
    return json.dumps({"jsonrpc": "2.0", "id": 2, "result": result})


ENV = {name: "https://sibling.test/mcp" for name in mcp_client.SIBLING_ENV.values()}


# ── the client ──────────────────────────────────────────────────────────────

def test_a_tool_result_comes_back_as_data():
    out = mcp_client.call_tool("datadictionary", "get_data_element", {"dataElement": "x"},
                               env=ENV, opener=opener(HELLO, tool_result({"definition": "d"})))
    assert out["data"] == {"definition": "d"}
    assert "error" not in out


def test_the_handshake_runs_before_the_call():
    captured = []
    mcp_client.call_tool("toolweave", "pre_tool", {"prompt": "p"}, env=ENV,
                         opener=opener(HELLO, tool_result({"ok": True}), captured=captured))
    assert [c["method"] for c in captured] == ["initialize", "tools/call"]
    assert captured[1]["params"]["name"] == "pre_tool"
    assert captured[1]["params"]["arguments"] == {"prompt": "p"}


def test_a_sibling_speaking_another_revision_is_an_error_not_a_call():
    older = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}})
    out = mcp_client.call_tool("screenweave", "crawl_url", {}, env=ENV,
                               opener=opener(older))
    assert "2024-11-05" in out["error"]


def test_an_unconfigured_sibling_names_the_variable():
    out = mcp_client.call_tool("cipherweave", "get_encryption_strategy", {}, env={})
    assert "CIPHERWEAVE_MCP_URL" in out["error"]


def test_a_network_failure_degrades_rather_than_raising():
    def boom(request, timeout=None):
        raise OSError("connection reset")
    out = mcp_client.call_tool("screenweave", "crawl_url", {}, env=ENV, opener=boom)
    assert "connection reset" in out["error"]


def test_a_tool_that_ran_and_failed_is_not_reported_as_success():
    """MCP's `isError` means the tool ran and failed, which is different from a
    transport failure and must not flatten into data."""
    out = mcp_client.call_tool("toolweave", "pre_tool", {}, env=ENV,
                               opener=opener(HELLO, tool_result({"why": "no catalogue"}, is_error=True)))
    assert "error" in out and "data" not in out


def test_an_sse_framed_answer_is_read():
    body = f"event: message\ndata: {tool_result({'v': 1})}\n\n"
    out = mcp_client.call_tool("datadictionary", "get_data_element", {}, env=ENV,
                               opener=opener(HELLO, body))
    assert out["data"] == {"v": 1}


# ── the rules ───────────────────────────────────────────────────────────────

def test_every_registered_weave_tool_has_a_rule():
    """A capability with no written reason is one nobody decided to grant."""
    for name in tool_rules.RULES:
        if tool_rules.is_refused(name):
            continue
        assert name in TOOL_REGISTRY, f"{name} has a rule but is not callable"


def test_commit_tools_are_refused_at_execution():
    """DataDictionary and ToolWeave split write into propose then commit so an
    unattended caller cannot mutate state. A pipeline step is that caller."""
    for name in tool_rules.tools_by_effect(tool_rules.COMMIT):
        with pytest.raises(PermissionError) as raised:
            execute_tool(name, {})
        assert "propose" in str(raised.value)


def test_commit_tools_are_not_in_the_registry_either():
    """Two independent barriers: removing one must not silently open the path."""
    for name in tool_rules.tools_by_effect(tool_rules.COMMIT):
        assert name not in TOOL_REGISTRY


def test_the_refusal_is_raised_not_skipped():
    """execute_pre_tools swallows tool failures so one bad lookup cannot lose a
    run. A skipped commit would report success for work never done, so the
    refusal must be a hard failure at execute_tool."""
    import inspect
    from src.orchestrator import tool_registry
    source = inspect.getsource(tool_registry.execute_tool)
    assert "raise PermissionError" in source


def test_every_rule_says_when_and_when_not():
    for name, rule in tool_rules.RULES.items():
        assert rule.use_when.strip(), f"{name} has no use_when"
        assert rule.never_when.strip(), f"{name} has no never_when"
        assert rule.effect in {tool_rules.READ, tool_rules.PROPOSE, tool_rules.COMMIT}
        assert rule.sibling in mcp_client.SIBLING_ENV, f"{name} names unknown sibling"


def test_an_unknown_tool_names_the_rules_table():
    with pytest.raises(KeyError) as raised:
        tool_rules.rule_for("invented_tool")
    assert "tool_rules.RULES" in str(raised.value)


# ── the tools carry their rule ──────────────────────────────────────────────

def test_a_result_records_why_the_call_was_made(monkeypatch):
    monkeypatch.setattr(weave_tools, "call_tool",
                        lambda *a, **k: {"data": {"x": 1}})
    out = weave_tools.lookup_data_element(dataElement="wallet_address")
    assert out["used_for"] == tool_rules.RULES["lookup_data_element"].use_when


def test_a_crawl_is_capped_regardless_of_what_it_is_asked_for(monkeypatch):
    """One worker invocation runs every step inside a 900 s Lambda; a crawl
    that walks a site spends the budget the later agents need."""
    seen = {}
    monkeypatch.setattr(weave_tools, "call_tool",
                        lambda s, t, args, **k: seen.update(args) or {"data": {}})
    weave_tools.crawl_site(url="https://x.test", max_depth=99, max_links=9999)
    assert seen["max_depth"] <= 2 and seen["max_links"] <= 25


# ── the teams ───────────────────────────────────────────────────────────────

WEAVE_TEAMS = ["weave_api_caller", "weave_data_steward", "weave_site_auditor"]


def load(name):
    return json.loads((TEAMS / name / "v1" / "team.json").read_text())


@pytest.mark.parametrize("name", WEAVE_TEAMS)
def test_a_team_stays_short(name):
    """A run is one Lambda invocation for every step. Small teams were asked
    for and are also what the 900 s ceiling affords."""
    config = load(name)
    assert len(config["agents"]) <= 3, "too many turns for the work"
    assert len(config["workflow"]) == len(config["agents"])


@pytest.mark.parametrize("name", WEAVE_TEAMS)
def test_every_agent_earns_its_turn(name):
    """Two agents with the same job is one agent and a wasted model call."""
    config = load(name)
    goals = [a["goal_template"] for a in config["agents"]]
    assert len(set(goals)) == len(goals), "two agents share a goal"
    schemas = [a["schema_ref"] for a in config["agents"]]
    assert len(set(schemas)) == len(schemas), "two agents produce the same shape"


@pytest.mark.parametrize("name", WEAVE_TEAMS)
def test_every_declared_tool_has_a_rule_and_is_callable(name):
    config = load(name)
    for step in config["workflow"]:
        for tool in (step.get("pre_tools") or []) + (step.get("post_tools") or []):
            rule = tool_rules.rule_for(tool["name"])
            assert not tool_rules.is_refused(tool["name"]), (
                f"{name} declares {tool['name']}, which commits"
            )
            assert tool["name"] in TOOL_REGISTRY
            assert rule.sibling


def test_the_teams_between_them_reach_every_wired_sibling():
    """Four gateway targets were wired; a sibling no team can reach is a target
    for nothing."""
    used = set()
    for name in WEAVE_TEAMS:
        for step in load(name)["workflow"]:
            for tool in (step.get("pre_tools") or []):
                used.add(tool_rules.rule_for(tool["name"]).sibling)
    assert used == set(mcp_client.SIBLING_ENV), f"unreached siblings: {set(mcp_client.SIBLING_ENV) - used}"


@pytest.mark.parametrize("name", WEAVE_TEAMS)
def test_an_agent_is_told_what_a_failed_lookup_looks_like(name):
    """A tool result carrying `error` is a real signal. An agent that cannot
    tell it from an empty one fills the blank in -- the same failure the RAG
    layer had before NO_VERIFIED_EXPERIENCE."""
    config = load(name)
    tooled_steps = {s["step"] for s in config["workflow"] if s.get("pre_tools")}
    for agent in config["agents"]:
        if agent["id"] in tooled_steps:
            assert "error" in agent["goal_template"], (
                f"{agent['id']} consumes a tool result but is never told what a "
                f"failed lookup looks like"
            )
