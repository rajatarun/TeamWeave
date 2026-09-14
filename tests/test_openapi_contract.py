"""Keep openapi/teamweave.yaml honest against the template and the handlers.

TeamWeave has seventeen paths across five Lambdas, and the routing lives in
two places that can disagree: ``infra/template.yaml`` decides what API Gateway
forwards, and ``src/orchestrator/*.py`` decides what a Lambda does with it.
Writing this spec surfaced exactly that -- the trigger handler accepts DELETE
on every management path, but the template routes DELETE only on ``/agents``
and ``/teams/{team_name}``, so a documented ``DELETE /roles/{role_id}`` would
have been a route no client could ever reach.

So the spec is checked against the template method by method, not path by
path, and the Outputs a harness resolves its coordinates from are checked
against the templates that publish them. None of it needs AWS.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "openapi" / "teamweave.yaml"
TEMPLATE_PATH = REPO_ROOT / "infra" / "template.yaml"
SHARED_PATH = REPO_ROOT / "infra" / "shared.yaml"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import stack_env  # noqa: E402

HTTP_METHODS = ("get", "post", "put", "patch", "delete")


@pytest.fixture(scope="module")
def spec() -> dict:
    with open(SPEC_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _template_routes() -> set:
    """(path, method) pairs API Gateway actually forwards.

    SAM writes each Api event as a Path: line immediately followed by a
    Method: line, so this pairs them positionally rather than trying to parse
    a template full of !Sub and !GetAtt tags.
    """
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    return {
        (m.group(1), m.group(2).lower())
        for m in re.finditer(r"^\s+Path:\s*(\S+)\s*\n\s+Method:\s*(\S+)\s*$", text, re.M)
    }


def _spec_routes(spec: dict) -> set:
    return {(p, m) for p, item in spec["paths"].items()
            for m in item if m in HTTP_METHODS}


def test_template_routes_were_actually_parsed():
    """Guard the guard: an empty parse would make the comparison vacuous."""
    routes = _template_routes()
    assert len(routes) >= 20, f"only parsed {len(routes)} routes -- the Path/Method pairing has gone stale"


def test_spec_documents_exactly_the_routes_the_gateway_forwards(spec):
    """Method-level, both directions.

    A documented route the gateway does not forward is a 403 no client
    expects; a forwarded route the spec omits is one nothing will call.
    """
    template, documented = _template_routes(), _spec_routes(spec)
    only_template = sorted(template - documented)
    only_spec = sorted(documented - template)
    assert not only_template, (
        f"API Gateway forwards these, but openapi/ does not document them: {only_template}"
    )
    assert not only_spec, (
        f"openapi/ documents these, but API Gateway does not forward them "
        f"(they would 403 before reaching a handler): {only_spec}"
    )


def test_async_routes_document_202_not_200(spec):
    """Every write here starts a Step Functions execution and returns 202.

    A spec that promises 200 on POST /team/task tells a harness the work is
    done when it has not started. These are the routes whose handler path ends
    in _start_async_execution.
    """
    async_routes = [
        ("/team/task", "post"),
        ("/agents", "post"), ("/agents", "delete"), ("/agents/{name}", "put"),
        ("/teams", "post"), ("/teams/{team_name}", "put"), ("/teams/{team_name}", "delete"),
        ("/roles", "post"), ("/roles/{role_id}", "put"),
        ("/departments", "post"), ("/departments/{dept_id}", "put"),
    ]
    for path, method in async_routes:
        responses = spec["paths"][path][method]["responses"]
        assert "202" in responses, (
            f"{method.upper()} {path} starts a Step Functions execution and returns "
            f"202, but the spec documents {sorted(responses)}"
        )
        assert "200" not in responses, (
            f"{method.upper()} {path} never returns 200 -- documenting one tells a "
            f"client the work finished when it has not started"
        )


def test_run_status_documents_failed_as_a_200(spec):
    """A FAILED run arrives as a 200 with status=FAILED in the body.

    A harness that checks only the HTTP code records every failed run as a
    pass, so the spec has to say FAILED is a legal 200.
    """
    for path in ("/team/task/{run_id}", "/teams/task/{run_id}"):
        ref = spec["paths"][path]["get"]["responses"]["200"]["$ref"]
        name = ref.rsplit("/", 1)[-1]
        schema = spec["components"]["responses"][name]["content"]["application/json"]["schema"]
        enum = schema["properties"]["status"]["enum"]
        assert "FAILED" in enum, f"{path}: 200 response does not admit status=FAILED ({enum})"
        assert "SUCCEEDED" in enum


def _output_names(path: Path) -> set:
    """Top-level keys of a template's Outputs block.

    Regex rather than yaml.safe_load: these templates are full of !Sub/!Ref/!If
    tags a safe loader rejects, and a permissive loader that maps unknown tags
    to None reports every !Sub-valued field as absent -- which is how a check
    like this ends up asserting nothing.
    """
    text = path.read_text(encoding="utf-8")
    block = text.split("\nOutputs:\n", 1)
    if len(block) != 2:
        return set()
    names = set()
    for line in block[1].splitlines():
        if line and not line.startswith(" ") and not line.startswith("#"):
            break
        m = re.match(r"^  ([A-Za-z][A-Za-z0-9]*):\s*$", line)
        if m:
            names.add(m.group(1))
    return names


def test_stack_env_reads_only_outputs_the_template_publishes():
    published = _output_names(TEMPLATE_PATH)
    assert published, "parsed no outputs from infra/template.yaml -- the parser is broken"
    wanted = set(stack_env.REQUIRED_OUTPUTS.values()) | set(stack_env.OPTIONAL_OUTPUTS.values())
    missing = sorted(wanted - published)
    assert not missing, (
        f"scripts/stack_env.py reads stack outputs that infra/template.yaml does "
        f"not publish: {missing}. Either add the Output or stop reading it."
    )


def test_required_outputs_cover_the_api_and_both_span_indexes():
    """A table name does not let you query a table whose access pattern is an index."""
    required = set(stack_env.REQUIRED_OUTPUTS.values())
    for needed in ("HttpApiUrl", "DdbTable", "ObservatoryMetricsTable",
                   "ObservatoryMetricsSpanTimelineIndex",
                   "ObservatoryMetricsAgentIdTimestampIndex"):
        assert needed in required, f"{needed} is not resolved from the stack"


def test_index_outputs_name_indexes_the_shared_stack_defines():
    """An output naming a nonexistent index fails at query time, not deploy time."""
    defined = set(re.findall(r"^\s+- IndexName:\s*(\S+)\s*$",
                             SHARED_PATH.read_text(encoding="utf-8"), re.M))
    assert defined, "parsed no IndexName entries from infra/shared.yaml"
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    for output in ("ObservatoryMetricsSpanTimelineIndex",
                   "ObservatoryMetricsAgentIdTimestampIndex"):
        m = re.search(rf"^  {output}:\n\s+Value:\s*(\S+)\s*$", text, re.M)
        assert m, f"infra/template.yaml has no {output} output with a literal Value"
        assert m.group(1) in defined, (
            f"{output} names index {m.group(1)!r}, which infra/shared.yaml does "
            f"not define (defined: {sorted(defined)})"
        )


def test_agent_metrics_enums_match_the_handler(spec):
    """The query parameters' enums are the handler's own validation sets.

    The handler rejects an unlisted value with a 400 that enumerates what it
    accepts, so a spec enum that has drifted produces requests the API refuses
    for a reason the client's own generated types said was impossible.
    """
    handler = (REPO_ROOT / "src" / "orchestrator" / "agent_metrics_handler.py").read_text(encoding="utf-8")

    def literal_set(name: str) -> set:
        m = re.search(rf"^{name}\s*=\s*\{{(.*?)\}}", handler, re.M | re.S)
        assert m, f"could not find {name} in agent_metrics_handler.py"
        return set(re.findall(r'"([^"]+)"', m.group(1)))

    params = {p["name"]: p for p in spec["paths"]["/observability/agent-metrics"]["get"]["parameters"]}
    for param_name, const_name in (("operation", "_VALID_OPERATIONS"),
                                   ("aggregate", "_VALID_AGGREGATES"),
                                   ("sort_by", "_VALID_SORT_BY")):
        spec_enum = set(params[param_name]["schema"]["enum"])
        handler_enum = literal_set(const_name)
        assert spec_enum == handler_enum, (
            f"?{param_name}: spec allows {sorted(spec_enum)}, the handler accepts "
            f"{sorted(handler_enum)}"
        )


def test_openapi_spec_path_output_points_at_the_spec():
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    m = re.search(r"^  OpenApiSpecPath:\n\s+Value:\s*(\S+)\s*$", text, re.M)
    assert m, "infra/template.yaml has no OpenApiSpecPath output with a literal Value"
    assert (REPO_ROOT / m.group(1)).is_file(), (
        f"OpenApiSpecPath points at {m.group(1)}, which does not exist"
    )
