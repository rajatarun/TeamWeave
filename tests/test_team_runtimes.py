"""One AgentCore runtime per team, and no team left without one.

A team is the deployment unit. One runtime for the whole platform put every
team's agents in one blast radius and one endpoint budget: release channels
are ten per runtime and shared, so two teams could not be canaried
independently and a bad version reached all of them at once.

Not one per *agent* — that was tried, and AWS's quota refused at twelve.
Endpoints are a release mechanism, not an identity axis; identity is a name in
a document (a skill id on the A2A card, `gen_ai.agent.id` on a span), and
`prompt_builder` composes ROLE and STEP_GOAL into every turn, so one runtime
serves a team's agents without them being confusable.

The cost of per-team runtimes is that adding a team now needs a template
change, where the platform's premise is that a team is just JSON in S3. That
tension is real, and this test is where it is made visible: a team in
`config/examples/teams` with no runtime fails here rather than silently
falling back to the stack-wide runtime and losing the isolation that was the
whole point.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = (REPO / "infra" / "template.yaml").read_text()
TEAMS_DIR = REPO / "config" / "examples" / "teams"

# CreateAgentRuntime's agentRuntimeName. No hyphens — the reason these are
# named from the team and not from ${AWS::StackName}, which has them.
NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,47}$")


class CfnLoader(yaml.SafeLoader):
    """CloudFormation short forms, keeping the argument."""


def _keep(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {"__fn__": suffix, "__arg__": value}


CfnLoader.add_multi_constructor("!", _keep)


@pytest.fixture(scope="module")
def resources():
    return yaml.load(TEMPLATE, Loader=CfnLoader)["Resources"]


@pytest.fixture(scope="module")
def runtimes(resources):
    """logical id -> properties, for every AgentCore runtime in the stack."""
    return {
        name: r["Properties"]
        for name, r in resources.items()
        if r.get("Type") == "AWS::BedrockAgentCore::Runtime"
    }


def configured_teams() -> list[str]:
    """Team names as the repository defines them."""
    names = []
    for path in sorted(TEAMS_DIR.glob("*/v1/team.json")):
        data = json.loads(path.read_text())
        name = str((data.get("team") or {}).get("name") or "").strip()
        if name:
            names.append(name)
    return names


def runtime_names(runtimes) -> set[str]:
    return {str(p.get("AgentRuntimeName")) for p in runtimes.values()}


def test_the_scan_finds_teams_and_runtimes(runtimes):
    # A glob or loader change could make every assertion below vacuous.
    # One team is the real floor now: tarun_visibility_team is the product
    # path and the only team the platform ships. A count of zero still means
    # the scan found nothing and every assertion below passed for free.
    teams = configured_teams()
    assert len(teams) >= 1, f"the team scan found nothing: {teams}"
    assert "tarun_visibility_team" in teams
    assert len(runtimes) >= len(teams), f"only found runtimes: {sorted(runtimes)}"


def test_every_team_has_a_runtime_of_its_own(runtimes):
    """The check that makes the template/config coupling visible.

    A team with no runtime here still runs — it falls back to the stack-wide
    one — so nothing would fail, and the isolation would quietly not exist for
    that team. Silent partial correctness is the failure mode this whole
    session has been about.
    """
    names = runtime_names(runtimes)
    missing = [t for t in configured_teams() if f"teamweave_{t}" not in names]
    assert not missing, (
        f"these teams have no AgentCore runtime: {missing} — they would fall back "
        f"to the shared runtime and lose the per-team isolation. Runtimes present: "
        f"{sorted(names)}"
    )


def test_no_two_teams_share_a_runtime(runtimes):
    names = [str(p.get("AgentRuntimeName")) for p in runtimes.values()]
    assert len(names) == len(set(names)), f"duplicate runtime names: {names}"


@pytest.mark.parametrize("team", configured_teams())
def test_each_team_runtime_is_named_legally(team, runtimes):
    # CreateAgentRuntime rejects a hyphen, and the rejection arrives at deploy
    # time after the artifact is built and uploaded.
    name = f"teamweave_{team}"
    assert NAME_PATTERN.match(name), f"{name} is not a legal AgentRuntimeName"
    assert name in runtime_names(runtimes)


def test_each_team_runtime_knows_which_team_it_is(runtimes):
    # The program's own logs and spans should say which team they served;
    # without it every runtime looks identical in the Observatory.
    for logical, props in runtimes.items():
        env = props.get("EnvironmentVariables") or {}
        if "AGENT_TEAM" not in env:
            continue
        assert env["AGENT_TEAM"] in configured_teams(), (
            f"{logical} claims team {env['AGENT_TEAM']!r}, which is not a configured team"
        )
    tagged = {(p.get("EnvironmentVariables") or {}).get("AGENT_TEAM")
              for p in runtimes.values()}
    assert set(configured_teams()) <= tagged, "a team's runtime does not identify its team"


def runtime_map():
    """The (team -> substitution name) map the worker is actually handed.

    Parsed from the `Fn::Sub` that builds AGENTCORE_TEAM_RUNTIME_ARNS rather
    than grepped out of the template. Searching the whole file for a team's
    name passes on any mention of it -- and every per-team runtime carries its
    own name twice, in `AGENT_TEAM` and in its `Description`. So a team
    dropped from the map while keeping its runtime resource read as correct,
    which is the exact shape of "the isolation silently does not exist".
    """
    doc = yaml.load(TEMPLATE, Loader=CfnLoader)
    value = doc["Globals"]["Function"]["Environment"]["Variables"]["AGENTCORE_TEAM_RUNTIME_ARNS"]
    # !If [cond, <the map>, ""] -> the map
    if isinstance(value, dict) and value.get("__fn__") == "If":
        value = value["__arg__"][1]
    assert isinstance(value, dict) and value.get("__fn__") == "Sub", value
    template_str, substitutions = value["__arg__"]
    # Substitute placeholder text so the JSON parses, keeping the names.
    rendered = template_str
    for name in substitutions:
        rendered = rendered.replace("${%s}" % name, name)
    return json.loads(rendered), substitutions


def test_the_worker_is_told_where_each_team_runs():
    # The resources exist for nothing if the map never reaches the worker.
    mapping, _ = runtime_map()
    for team in configured_teams():
        assert team in mapping, (
            f"{team} is missing from AGENTCORE_TEAM_RUNTIME_ARNS, so every one of "
            f"its turns falls back to the shared runtime. Present: {sorted(mapping)}"
        )


def test_the_map_names_no_team_that_does_not_exist():
    """A stale entry points the worker at a runtime for a deleted team."""
    mapping, _ = runtime_map()
    teams = configured_teams()
    stale = sorted(set(mapping) - set(teams))
    assert not stale, f"the runtime map names teams that no longer exist: {stale}"


def test_each_teams_entry_resolves_to_its_own_runtime(resources):
    """The value must be that team's runtime, not another team's.

    Both sides of the map are strings in a template; crossing two teams'
    substitutions would deploy, pass every other check here, and route one
    team's agents into the other's runtime.
    """
    mapping, substitutions = runtime_map()
    for team, placeholder in mapping.items():
        getatt = substitutions[placeholder]
        assert getatt.get("__fn__") == "GetAtt", (team, getatt)
        resource_name = str(getatt["__arg__"]).split(".")[0]
        agent_team = (
            (resources[resource_name].get("Properties") or {})
            .get("EnvironmentVariables") or {}
        ).get("AGENT_TEAM")
        assert agent_team == team, (
            f"{team} maps to {resource_name}, whose AGENT_TEAM is {agent_team!r}"
        )


def test_the_shared_runtime_remains_as_a_fallback():
    """A team with no runtime of its own keeps working.

    Removing the fallback would turn "this team has no runtime yet" from a
    degraded run into a failed one, on a platform whose premise is that teams
    are added as JSON.
    """
    assert "AGENTCORE_RUNTIME_ARN:" in TEMPLATE


def test_the_runtime_map_references_resources_that_exist(resources):
    """A dangling !GetAtt is a deploy failure, not a test failure.

    The map is built from !GetAtt on each team's runtime. Rename or remove a
    resource and the template still parses, the suite still passes, and
    CloudFormation rejects the stack minutes into a deploy.
    """
    import re as _re

    referenced = set(_re.findall(r"!GetAtt (AgentCoreRuntime\w*)\.AgentRuntimeArn", TEMPLATE))
    assert referenced, "the runtime map references no runtimes at all"
    missing = sorted(referenced - set(resources))
    assert not missing, f"the template references undefined resources: {missing}"


def test_every_team_runtime_resource_is_actually_used(resources, runtimes):
    """The other direction: a runtime nothing references serves no traffic.

    It would still be created, still cost, still look like the design was
    implemented — and every turn would go to the shared runtime.
    """
    import re as _re

    referenced = set(_re.findall(r"!GetAtt (AgentCoreRuntime\w*)\.AgentRuntimeArn", TEMPLATE))
    per_team = {
        name for name, props in runtimes.items()
        if (props.get("EnvironmentVariables") or {}).get("AGENT_TEAM")
    }
    orphaned = sorted(per_team - referenced)
    assert not orphaned, (
        f"these per-team runtimes are created but nothing routes to them: {orphaned}"
    )
