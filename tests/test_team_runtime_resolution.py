"""Which runtime a turn lands on, most specific answer first.

The resolution order is the whole of the per-team design at run time. If a
team's turns quietly resolve to the shared runtime, every deploy still
succeeds, every run still answers, and the isolation the runtimes were created
for does not exist — which is indistinguishable from it working.
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from src.orchestrator.agent_runtime import AgentCoreRuntime, AgentRef  # noqa: E402

TEAM_MAP = {
    "doc_rewrite_team": "arn:aws:bedrock-agentcore:us-east-1:1:runtime/doc-1",
    "tarun_visibility_team": "arn:aws:bedrock-agentcore:us-east-1:1:runtime/vis-1",
}


@pytest.fixture
def stack(monkeypatch):
    monkeypatch.setenv("AGENTCORE_TEAM_RUNTIME_ARNS", json.dumps(TEAM_MAP))
    monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-east-1:1:runtime/shared")


def test_a_turn_runs_on_its_own_teams_runtime(stack):
    assert AgentCoreRuntime.resolve_arn(AgentRef(team="doc_rewrite_team")) == TEAM_MAP["doc_rewrite_team"]
    assert AgentCoreRuntime.resolve_arn(AgentRef(team="tarun_visibility_team")) == TEAM_MAP["tarun_visibility_team"]


def test_two_teams_do_not_land_on_the_same_runtime(stack):
    a = AgentCoreRuntime.resolve_arn(AgentRef(team="doc_rewrite_team"))
    b = AgentCoreRuntime.resolve_arn(AgentRef(team="tarun_visibility_team"))
    assert a != b, "per-team runtimes that resolve to one runtime are not per-team"


def test_an_agent_with_its_own_runtime_still_wins(stack):
    # The escape hatch for an agent that genuinely needs its own substrate.
    ref = AgentRef(team="doc_rewrite_team", runtime_arn="arn:aws:bedrock-agentcore:us-east-1:1:runtime/special")
    assert AgentCoreRuntime.resolve_arn(ref).endswith("/special")


def test_an_unknown_team_falls_back_rather_than_failing(stack):
    # A team added as JSON before its runtime exists should degrade, not die.
    assert AgentCoreRuntime.resolve_arn(AgentRef(team="brand_new_team")).endswith("/shared")


def test_a_team_with_no_name_falls_back(stack):
    assert AgentCoreRuntime.resolve_arn(AgentRef()).endswith("/shared")


@pytest.mark.parametrize("raw", ["", "not json", "[1,2]", '"a string"', "null"])
def test_a_broken_map_degrades_to_the_shared_runtime(monkeypatch, raw):
    """Unparseable configuration must not take every run down.

    The map arrives as a JSON string in an environment variable, so a template
    edit can malform it. Raising here would fail every turn of every team; the
    shared runtime is a worse answer and a far better failure.
    """
    monkeypatch.setenv("AGENTCORE_TEAM_RUNTIME_ARNS", raw)
    monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:shared")
    assert AgentCoreRuntime.team_runtime_arns() == {}
    assert AgentCoreRuntime.resolve_arn(AgentRef(team="doc_rewrite_team")) == "arn:shared"


def test_blank_and_non_string_entries_are_ignored(monkeypatch):
    monkeypatch.setenv("AGENTCORE_TEAM_RUNTIME_ARNS",
                       json.dumps({"a": "", "b": "   ", "c": None, "d": 7, "e": "arn:ok"}))
    assert AgentCoreRuntime.team_runtime_arns() == {"e": "arn:ok"}


def test_with_nothing_configured_the_failure_names_the_team(monkeypatch):
    # The message has to say which team, or the operator cannot tell whether
    # the map is empty or just missing one entry.
    monkeypatch.delenv("AGENTCORE_TEAM_RUNTIME_ARNS", raising=False)
    monkeypatch.delenv("AGENTCORE_RUNTIME_ARN", raising=False)
    message = AgentCoreRuntime().missing_fields(AgentRef(team="doc_rewrite_team"))
    assert message
    assert "doc_rewrite_team" in message


def test_a_usable_ref_reports_no_problem(stack):
    assert AgentCoreRuntime().missing_fields(AgentRef(team="doc_rewrite_team")) == ""


# ── The wiring, not just the logic ───────────────────────────────────────────
# resolve_arn() can be perfect and the design still absent: if the worker never
# tells bedrock_invoke which team a turn belongs to, every ref arrives with
# team="" and every team resolves to the shared runtime. Nothing fails, every
# run answers, and the isolation silently does not exist.


def test_invoke_agent_carries_the_team_down_to_the_ref(monkeypatch):
    """Drive the real call path and capture what the substrate is handed."""
    from src.orchestrator import bedrock_invoke

    seen = {}

    class CapturingRuntime:
        name = "capturing"

        def missing_fields(self, ref):
            seen["ref"] = ref
            return ""

        def invoke(self, ref, *, session_id, input_text, shadow_alias_id=None):
            seen["ref"] = ref
            return "{}", {}

    monkeypatch.setattr(bedrock_invoke, "get_runtime", lambda: CapturingRuntime())
    bedrock_invoke.invoke_agent("a", "b", "run-1", "prompt", team="doc_rewrite_team")
    assert seen["ref"].team == "doc_rewrite_team"


def test_the_metrics_variant_carries_it_too(monkeypatch):
    from src.orchestrator import bedrock_invoke

    seen = {}

    class CapturingRuntime:
        name = "capturing"

        def missing_fields(self, ref):
            return ""

        def invoke(self, ref, *, session_id, input_text, shadow_alias_id=None):
            seen["ref"] = ref
            return "{}", {}

    monkeypatch.setattr(bedrock_invoke, "get_runtime", lambda: CapturingRuntime())
    bedrock_invoke.invoke_agent_with_metrics("a", "b", "run-1", "prompt", team="tarun_visibility_team")
    assert seen["ref"].team == "tarun_visibility_team"


def test_every_worker_invocation_names_its_team():
    """Structural, not textual: parse the worker and inspect each call.

    A grep for "team=team" passes on a comment. This walks the AST for calls to
    the two invoke functions and checks each one actually passes the keyword,
    so a call added later without it fails here rather than in production as a
    team quietly sharing the wrong runtime.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src" / "orchestrator" / "worker_handler.py").read_text()
    tree = ast.parse(source)
    targets = {"invoke_agent", "invoke_agent_with_metrics"}

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in targets
    ]
    assert calls, "found no agent invocations in the worker — has it been renamed?"
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        assert "team" in keywords, (
            f"{call.func.id} at line {call.lineno} does not pass team=, so its turns "
            f"resolve to the shared runtime whatever the team map says"
        )
