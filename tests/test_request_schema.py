"""Each team declares what a run needs from the person asking.

Without this a UI cannot offer "ask a team to do something" without hardcoding
one form per team — which contradicts the platform's premise that a team is
JSON in S3 and adding one costs no code. `GET /teams/{name}` already returns
the whole document, so the schema reaches the UI with no new endpoint.

The check that matters is the derived one: a config that reads
`request.document_text` somewhere in its workflow must declare
`document_text`. Otherwise the form omits a field the pipeline needs, the run
starts anyway, and the agent is handed an empty string — a bad answer rather
than an error.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TEAMS_DIR = REPO / "config" / "examples" / "teams"

# "hidden" is continuation state, not a question: on a follow-up turn the run
# page submits the previous answer and run id alongside the person's edit, and
# nobody types either. It is declared in request_schema rather than smuggled in
# as an undeclared extra so the "a config that reads request.X must declare X"
# rule below still covers it -- and the UI is required to skip rendering it,
# which tests/test_conversational_contract.py holds it to.
ALLOWED_TYPES = {"text", "textarea", "hidden"}
HIDDEN_TYPE = "hidden"


def team_files() -> list[Path]:
    return sorted(TEAMS_DIR.glob("*/v1/team.json"))


def team_ids() -> list[str]:
    return [p.parent.parent.name for p in team_files()]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_there_are_teams_to_check():
    # A guard against the parametrised tests below becoming vacuous, not a
    # statement about how many teams the platform should have.
    # One team ships now (tarun_visibility_team). Zero means the glob stopped
    # matching and every check below passes for free, which is the case worth
    # catching.
    assert team_files(), "found no team configs at all — has the layout moved?"


@pytest.mark.parametrize("path", team_files(), ids=team_ids())
def test_every_team_declares_a_request_schema(path):
    schema = load(path).get("request_schema")
    assert isinstance(schema, dict), f"{path.parent.parent.name} declares no request_schema"
    assert schema.get("summary", "").strip(), "a schema with no summary tells the user nothing"
    assert schema.get("fields"), "a schema with no fields cannot collect a request"


@pytest.mark.parametrize("path", team_files(), ids=team_ids())
def test_every_field_is_renderable(path):
    for field in load(path)["request_schema"]["fields"]:
        assert field.get("name"), f"a field with no name cannot be submitted: {field}"
        assert field.get("label", "").strip(), f"{field['name']} has no label"
        if field.get("type") == HIDDEN_TYPE:
            # A hidden field the person cannot see must not be required: the
            # form would block on a value there is no way to supply.
            assert not field.get("required"), (
                f"{field['name']} is hidden and required, so a first turn can never submit"
            )
        assert field.get("type") in ALLOWED_TYPES, (
            f"{field['name']} has type {field.get('type')!r}; the UI renders {sorted(ALLOWED_TYPES)}"
        )
        assert isinstance(field.get("required"), bool), f"{field['name']} does not say if it is required"


@pytest.mark.parametrize("path", team_files(), ids=team_ids())
def test_at_least_one_field_is_required(path):
    # A form where everything is optional lets someone start a run that asks
    # for nothing, which burns a pipeline and returns noise.
    fields = load(path)["request_schema"]["fields"]
    assert any(f["required"] for f in fields), "no required field: a run could be started empty"


@pytest.mark.parametrize("path", team_files(), ids=team_ids())
def test_the_schema_covers_every_request_field_the_config_reads(path):
    """Derived, so a workflow change cannot outrun the form.

    A config referencing `request.document_text` needs the form to collect it;
    otherwise the pipeline runs with an empty value and produces a confidently
    wrong answer instead of failing.
    """
    raw = path.read_text()
    referenced = set(re.findall(r"request\.([a-zA-Z_][a-zA-Z0-9_]*)", raw))
    declared = {f["name"] for f in load(path)["request_schema"]["fields"]}
    missing = sorted(referenced - declared)
    assert not missing, (
        f"{path.parent.parent.name} reads {missing} from the request but its "
        f"request_schema declares {sorted(declared)} — the form would leave them empty"
    )


@pytest.mark.parametrize("path", team_files(), ids=team_ids())
def test_field_names_are_unique(path):
    names = [f["name"] for f in load(path)["request_schema"]["fields"]]
    assert len(names) == len(set(names)), f"duplicate field names: {names}"


def test_the_derivation_actually_finds_references():
    # If the regex matched nothing anywhere, the coverage test above would be
    # vacuous for every team at once.
    found = {
        p.parent.parent.name: set(re.findall(r"request\.([a-zA-Z_][a-zA-Z0-9_]*)", p.read_text()))
        for p in team_files()
    }
    assert any(found.values()), f"no request.* references found in any config: {found}"
