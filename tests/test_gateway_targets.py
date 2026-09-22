"""The weave systems that are MCP servers, as AgentCore Gateway targets.

Only a system speaking MCP over **HTTP** can be one: a `GatewayTarget` takes
an `Endpoint`, so a stdio server has nothing to point at. DeployWeave is stdio
(`mcp.run()` with no transport) and is absent for that reason rather than by
oversight.

The defect these exist to stop is the one the gateway shipped with: it was
created with a single target, that target was gated on
`ScreenWeaveMcpEndpoint`, and **nothing in the deploy passed that parameter**.
So the condition was false on every deploy, the gateway had nothing behind it,
and no signal said so -- the parameter's own description already called that
"infrastructure for nothing".
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = (REPO / "infra" / "template.yaml").read_text()
WORKFLOW = (REPO / ".github" / "workflows" / "deploy.yml").read_text()


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
def doc():
    return yaml.load(TEMPLATE, Loader=CfnLoader)


@pytest.fixture(scope="module")
def targets(doc):
    return {name: res for name, res in doc["Resources"].items()
            if res.get("Type") == "AWS::BedrockAgentCore::GatewayTarget"}


def mcp_parameters(doc):
    return sorted(p for p in doc["Parameters"] if p.endswith("McpEndpoint"))


# ── the gateway has something behind it ─────────────────────────────────────

def test_there_are_targets_at_all():
    """A gateway with no targets is infrastructure for nothing."""
    assert TEMPLATE.count("AWS::BedrockAgentCore::GatewayTarget") >= 1


def test_every_mcp_parameter_has_a_target(doc, targets):
    endpoints = set()
    for res in targets.values():
        ep = res["Properties"]["TargetConfiguration"]["Mcp"]["McpServer"]["Endpoint"]
        assert ep.get("__fn__") == "Ref", ep
        endpoints.add(ep["__arg__"])
    missing = sorted(set(mcp_parameters(doc)) - endpoints)
    assert not missing, f"declared but pointed at by no target: {missing}"


def test_every_target_points_at_its_own_parameter(doc, targets):
    """A target reading another's parameter deploys fine and sends every call
    for one tool to a different service. The runtime map had exactly this bug
    with two teams' substitutions crossed."""
    for name, res in targets.items():
        ep = res["Properties"]["TargetConfiguration"]["Mcp"]["McpServer"]["Endpoint"]["__arg__"]
        stem = name.replace("GatewayTarget", "")
        assert ep == f"{stem}McpEndpoint", f"{name} points at {ep}"


def test_every_target_is_gated_on_its_own_parameter(doc, targets):
    """The shipped bug, one level up: one shared condition gated every target
    on ScreenWeave's parameter, so a sibling that *was* configured still got
    no target whenever ScreenWeave was not."""
    conditions = doc["Conditions"]
    for name, res in targets.items():
        stem = name.replace("GatewayTarget", "")
        cond = conditions[res["Condition"]]
        assert f"{stem}McpEndpoint" in str(cond), (
            f"{name}'s condition {res['Condition']} does not test {stem}McpEndpoint"
        )


def test_no_two_targets_share_a_condition(doc, targets):
    used = [res["Condition"] for res in targets.values()]
    assert len(used) == len(set(used)), f"targets share a condition: {used}"


def test_every_target_names_a_distinct_tool(targets):
    names = [res["Properties"]["Name"] for res in targets.values()]
    assert len(names) == len(set(names)), names


# ── the deploy actually passes them ─────────────────────────────────────────

def test_the_deploy_passes_every_mcp_parameter(doc):
    """The original defect. The parameter existed, the target was gated on it,
    and nothing set it -- so the condition was false on every deploy and the
    gateway stayed empty while every signal read as success."""
    for param in mcp_parameters(doc):
        assert param in WORKFLOW, (
            f"{param} is declared and gates a target, but the deploy never "
            f"passes it -- the target is created on no deploy"
        )


def test_the_deploy_passes_no_parameter_the_template_does_not_declare(doc):
    """The other direction, which the check above cannot see.

    `sam deploy` refuses a `--parameter-overrides` key the template does not
    declare -- but only on a run where that sibling's stack actually resolves,
    so a spec added for an undeclared parameter deploys green until the day
    the sibling exists. Offline, both sides are readable now.
    """
    declared = set(mcp_parameters(doc))
    for param in re.findall(r"(\w+McpEndpoint)", WORKFLOW):
        assert param in declared, (
            f"the deploy passes {param}, which infra/template.yaml does not "
            f"declare -- sam deploy will reject it once the sibling resolves"
        )


def test_each_endpoint_is_read_from_a_sibling_stack_not_hardcoded(doc):
    """A pasted URL rots silently when a sibling is redeployed."""
    for param in mcp_parameters(doc):
        assert f"{param}\"" in WORKFLOW or f"{param}=" in WORKFLOW
    # No literal API hostnames among the resolution specs.
    specs = re.findall(r'"([a-z0-9-]+:[A-Za-z]+:[^:]*:\w+McpEndpoint)"', WORKFLOW)
    assert len(specs) >= len(mcp_parameters(doc)), specs
    for spec in specs:
        assert "execute-api" not in spec and "lambda-url" not in spec, spec


# ── the stack names, which nothing offline can derive ───────────────────────

# Each sibling's deploy creates a differently-shaped name, and this repository
# cannot check them against anything -- a wrong one is indistinguishable from a
# sibling that was never deployed, which is exactly how the gateway came up
# with two targets instead of four. Pinned here with provenance so that
# changing one is a deliberate act against a stated source, and so the pairing
# of sibling to stack survives a later edit to the loop.
#
# Every entry cites the sibling's **CI workflow**, because that is what creates
# the stack. A samconfig's default_name and a deploy.sh's derived name are what
# someone gets running it by hand; neither is evidence of what exists. Guessing
# from those cost three wrong names across two rounds:
#   screenweave-prod, screenweave-dev  <- deploy.sh's screenweave-${ENV}
#   data-dictionary-mcp                <- samconfig's unsuffixed default
EXPECTED_STACKS = {
    "ScreenWeaveMcpEndpoint": "screenweave",
    # .github/workflows/deploy.yaml: STACK_NAME: screenweave
    # (also README.md and docs/architecture.md, both of which spell out
    # `--stack-name screenweave`)
    "CipherWeaveMcpEndpoint": "cipherweave-prod",
    # .github/workflows: STACK_NAME: cipherweave-${{ inputs.stage || 'prod' }}
    "DataDictionaryMcpEndpoint": "data-dictionary-mcp-prod",
    # .github/workflows: --stack-name "data-dictionary-mcp-${{ env.STAGE }}"
    # with STAGE defaulting to prod.
    "ToolWeaveMcpEndpoint": "toolweave",
    # samconfig.toml, unsuffixed. Its `sam deploy` takes the name from there
    # rather than a flag, and its workflow's own describe-stacks calls say
    # `--stack-name toolweave`, so both sources agree -- which is why this one
    # was right when read from the samconfig and DataDictionary's was not.
}


def test_each_target_resolves_from_the_stack_its_sibling_actually_deploys():
    specs = re.findall(r'"([a-z0-9-]+):([A-Za-z]+):([^:]*):(\w+McpEndpoint)"', WORKFLOW)
    found = {param: stack for stack, _out, _suffix, param in specs}
    assert found == EXPECTED_STACKS, (
        "a sibling stack name changed; check it against that repository's own "
        "deploy before accepting this diff -- a wrong name warns and wires "
        "nothing, exactly like a sibling that does not exist"
    )


def test_every_spec_has_all_four_fields(doc):
    """A spec short one field silently shifts the others: the output key
    becomes the suffix and the parameter name is never passed."""
    for line in WORKFLOW.splitlines():
        match = re.search(r'"([a-z0-9-]+:[^"]*McpEndpoint)"', line)
        if not match:
            continue
        fields = match.group(1).split(":")
        assert len(fields) == 4, f"{match.group(1)} has {len(fields)} fields, expected 4"
        stack, output, _suffix, param = fields
        assert stack and output and param
        assert param in mcp_parameters(doc)


def test_a_wrong_stack_name_is_told_apart_from_a_wrong_output_key():
    """They look identical as "did not resolve" and are different bugs: one is
    fixed in this file, the other in the sibling's template. Reported together
    they cost a round trip through a human, which is what they cost."""
    block = WORKFLOW[WORKFLOW.index("MCP_WIRED=()"):]
    block = block[:block.index("sam deploy \\")] if "sam deploy \\" in block else block
    assert "no stack" in block, "a missing stack is not reported as a missing stack"
    assert "publishes no" in block, "a missing output key is not distinguished"
    assert "list-stacks" in block, (
        "nothing names the stacks that do exist, so a wrong guess is not "
        "fixable from the log"
    )


def test_a_missing_sibling_is_reported_rather_than_silent():
    assert "MCP_MISSING" in WORKFLOW
    # One warning per sibling rather than one listing them all: the reason
    # differs per sibling now, so a combined line could not carry it.
    assert "::warning::No MCP endpoint for ${missing}" in WORKFLOW
    assert 'for missing in "${MCP_MISSING[@]}"' in WORKFLOW


def test_an_empty_gateway_is_reported_rather_than_silent():
    assert "infrastructure for nothing" in WORKFLOW


def test_a_missing_sibling_does_not_fail_the_deploy():
    """One tool fewer is not a broken platform -- unlike the ContextWeave URL,
    which a declared RAG mode makes mandatory."""
    start = WORKFLOW.index("MCP_WIRED=()")
    # Search for the end marker *from* the start: `sam deploy \\` also appears in
    # the shared-stack step far earlier in the file, and anchoring on the first
    # occurrence produced a reversed slice -- an empty string, in which any
    # "not in" assertion passes. A mutation that inserted `exit 1` into the
    # loop survived this test until the search was anchored.
    end = WORKFLOW.index("sam deploy \\", start)
    block = WORKFLOW[start:end]
    assert "MCP_MISSING" in block, "the slice no longer covers the MCP resolution loop"
    assert "exit 1" not in block, "a missing MCP sibling must not fail the deploy"


# ── only HTTP MCP servers can be targets ────────────────────────────────────

def test_stdio_only_servers_are_not_targets(doc):
    """DeployWeave speaks MCP over stdio, so there is no endpoint to point a
    target at. Absent on purpose; a target for it could never resolve."""
    assert "DeployWeaveMcpEndpoint" not in doc["Parameters"]
    # And the deploy must not try to resolve one either: a spec here would
    # pass a parameter the template does not declare (see above) rather than
    # quietly doing nothing.
    assert "DeployWeaveMcpEndpoint" not in WORKFLOW


# ── the loop, actually run ──────────────────────────────────────────────────

class TestTheResolutionLoopRuns:
    """The assertions above read the workflow as text, which cannot tell a
    branch that works from one that is present and dead. This extracts the
    real loop and runs it under bash against a fake `aws`, so what is checked
    is what it does.
    """

    @staticmethod
    def _loop_source() -> str:
        start = WORKFLOW.index("          MCP_WIRED=()")
        end = WORKFLOW.index('          [ -n "${LAMBDA_SUBNET_IDS}" ]')
        body = "\n".join(
            line[10:] if line.startswith("          ") else line
            for line in WORKFLOW[start:end].splitlines()
        )
        return (
            # `set -e` matches how GitHub runs a `run:` block -- the job log
            # shows `shell: /usr/bin/bash -e {0}`. Without it these tests pass
            # on a loop that dies in CI: the handshake probe reports its
            # outcome as an exit code, and under -e a bare non-zero command
            # ends the script before the next line can read $?. The deploy
            # failed with "Process completed with exit code 2" on the very
            # case -- `unverified` -- that exists in order not to fail.
            #
            # REPO_ROOT is captured by the real step before its `cd infra`, so
            # the extracted loop needs it too. Pointing it at the temporary
            # directory is what puts the probe stub on the path the loop uses.
            "set -eu\nPARAM_OVERRIDES=()\nAWS_REGION=us-east-1\nREPO_ROOT=\"$PWD\"\n"
            + body
            + '\necho "---PARAMS---"\n'
            + 'if [ ${#PARAM_OVERRIDES[@]} -gt 0 ]; then printf "%s\\n" "${PARAM_OVERRIDES[@]}"; fi\n'
        )

    def _run(self, tmp_path, describe: dict, listing: str = "", probe_exit: int = 0):
        """describe: stack name -> describe-stacks JSON. Absent = no such stack.

        probe_exit is what the MCP handshake check reports: 0 ok, 1 refused,
        2 unverified. The real script is replaced by a stub and the loop runs
        with tmp_path as its working directory, so what is exercised here is
        the loop's branching on that outcome -- the script's own behaviour is
        covered by tests/test_mcp_handshake.py.
        """
        import json as _json
        import os
        import shutil
        import subprocess

        if shutil.which("bash") is None:  # pragma: no cover
            pytest.skip("bash not available")

        bindir = tmp_path / "bin"
        bindir.mkdir(exist_ok=True)
        (tmp_path / "stacks.json").write_text(_json.dumps(describe))
        (tmp_path / "listing.txt").write_text(listing)

        # A Python stub rather than shell: the real `aws` exits non-zero for a
        # stack that does not exist and prints JSON otherwise, and those two
        # behaviours are the whole thing under test.
        (bindir / "aws").write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys, pathlib\n"
            f"base = pathlib.Path({str(tmp_path)!r})\n"
            "argv = sys.argv[1:]\n"
            "if 'describe-stacks' in argv:\n"
            "    name = argv[argv.index('--stack-name') + 1]\n"
            "    stacks = json.loads((base / 'stacks.json').read_text())\n"
            "    if name not in stacks:\n"
            "        sys.stderr.write('ValidationError: Stack does not exist\\n')\n"
            "        sys.exit(255)\n"
            "    print(json.dumps(stacks[name]))\n"
            "elif 'list-stacks' in argv:\n"
            "    sys.stdout.write((base / 'listing.txt').read_text())\n"
        )
        (bindir / "aws").chmod(0o755)

        probe_dir = tmp_path / "scripts"
        probe_dir.mkdir(exist_ok=True)
        (probe_dir / "mcp_handshake.py").write_text(
            "import sys\n"
            f"print('probe stub ->', ' '.join(sys.argv[1:]))\n"
            f"sys.exit({probe_exit})\n"
        )

        script = tmp_path / "loop.sh"
        script.write_text(self._loop_source())
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
        proc = subprocess.run(
            ["bash", str(script)], capture_output=True, text=True, env=env,
            timeout=120, cwd=str(tmp_path),
        )
        out = proc.stdout
        params = out.split("---PARAMS---", 1)[1].split() if "---PARAMS---" in out else []
        return proc, out, params

    ALL_GOOD = {
        "screenweave": {"Stacks": [{"Outputs": [
            {"OutputKey": "McpEndpoint", "OutputValue": "https://sw.test/mcp"}]}]},
        "cipherweave-prod": {"Stacks": [{"Outputs": [
            {"OutputKey": "CipherWeaveApiEndpoint", "OutputValue": "https://cw.test/"}]}]},
        "data-dictionary-mcp-prod": {"Stacks": [{"Outputs": [
            {"OutputKey": "DataDictionaryFunctionUrl", "OutputValue": "https://dd.test"}]}]},
        "toolweave": {"Stacks": [{"Outputs": [
            {"OutputKey": "ToolWeaveApiEndpoint", "OutputValue": "https://tw.test"}]}]},
    }

    def test_all_four_siblings_wire_all_four_parameters(self, tmp_path, doc):
        """The bug the user saw was two of four. Four resolvable stacks must
        produce four overrides."""
        _proc, out, params = self._run(tmp_path, self.ALL_GOOD)
        assert len(params) == 4, params
        assert {p.split("=", 1)[0] for p in params} == set(mcp_parameters(doc))
        # What the log says has to match what was wired. A deploy that wires
        # four targets while reporting "infrastructure for nothing" sends the
        # next reader to debug a gateway that is fine.
        assert "Gateway targets wired: ScreenWeave CipherWeave DataDictionary ToolWeave" in out
        assert "infrastructure for nothing" not in out
        assert "::warning::" not in out

    def test_the_suffix_is_applied_only_where_the_sibling_omits_it(self, tmp_path):
        _proc, _out, params = self._run(tmp_path, self.ALL_GOOD)
        by_param = dict(p.split("=", 1) for p in params)
        # ScreenWeave publishes the path itself -- appending again gives /mcp/mcp.
        assert by_param["ScreenWeaveMcpEndpoint"] == "https://sw.test/mcp"
        # A trailing slash must not survive into //mcp.
        assert by_param["CipherWeaveMcpEndpoint"] == "https://cw.test/mcp"
        assert by_param["ToolWeaveMcpEndpoint"] == "https://tw.test/mcp"

    def test_a_missing_stack_is_named_with_the_stacks_that_do_exist(self, tmp_path):
        """The original failure: `screenweave-prod` does not exist and
        `screenweave-dev` does. The log has to say so, or the next person
        guesses again."""
        describe = {k: v for k, v in self.ALL_GOOD.items() if k != "screenweave"}
        _proc, out, params = self._run(tmp_path, describe, listing="screenweave-prod")
        assert "no stack 'screenweave'" in out
        assert "found: screenweave-prod" in out
        assert len(params) == 3

    def test_a_sibling_that_was_never_deployed_says_so(self, tmp_path):
        describe = {k: v for k, v in self.ALL_GOOD.items() if k != "toolweave"}
        _proc, out, _params = self._run(tmp_path, describe, listing="")
        assert "not deployed" in out

    def test_a_wrong_output_key_names_the_keys_the_stack_has(self, tmp_path):
        """A different bug from a missing stack, and fixed in a different
        repository."""
        describe = dict(self.ALL_GOOD)
        describe["toolweave"] = {"Stacks": [{"Outputs": [
            {"OutputKey": "ApiSpecsBucketName", "OutputValue": "b"}]}]}
        _proc, out, params = self._run(tmp_path, describe)
        assert "publishes no 'ToolWeaveApiEndpoint'" in out
        assert "ApiSpecsBucketName" in out
        assert len(params) == 3

    def test_a_stack_with_no_outputs_at_all_does_not_crash(self, tmp_path):
        describe = dict(self.ALL_GOOD)
        describe["toolweave"] = {"Stacks": [{}]}
        proc, out, params = self._run(tmp_path, describe)
        assert proc.returncode == 0, proc.stderr
        assert len(params) == 3
        assert "it has: none" in out

    def test_no_resolvable_sibling_is_reported_and_still_succeeds(self, tmp_path):
        proc, out, params = self._run(tmp_path, {})
        assert proc.returncode == 0, proc.stderr
        assert params == []
        assert "infrastructure for nothing" in out

    def test_a_sibling_that_refuses_the_handshake_is_not_wired(self, tmp_path):
        """The failure that rolls the stack back. AgentCore handshakes the
        server while creating the target, so a refusal is not a missing tool --
        it is a CloudFormation failure that takes every other target with it."""
        proc, out, params = self._run(tmp_path, self.ALL_GOOD, probe_exit=1)
        assert params == [], params
        assert "handshake refused" in out
        assert "not lose a tool" in out
        assert proc.returncode == 0, "refusing a target must not fail the deploy step"

    def test_the_harness_runs_the_loop_the_way_github_does(self):
        """If this drifts from the real shell flags, every test in this class
        is checking a loop CI does not run."""
        source = self._loop_source()
        assert source.startswith("set -e"), (
            "the extracted loop must run under -e, as `shell: /usr/bin/bash -e {0}` does"
        )
        workflow_shell = [
            line for line in WORKFLOW.splitlines() if line.strip().startswith("shell:")
        ]
        for line in workflow_shell:
            assert "-e" in line or "bash" not in line, (
                f"a step opts out of -e ({line.strip()!r}); these tests assume the default"
            )

    def test_an_unverified_handshake_still_wires_the_target(self, tmp_path):
        """A probe from CI is weak evidence about a call the Gateway makes with
        its own identity. Dropping a working tool on it would be the silent
        failure; letting it through means CloudFormation says so."""
        _proc, _out, params = self._run(tmp_path, self.ALL_GOOD, probe_exit=2)
        assert len(params) == 4, params

    @pytest.mark.parametrize("case", ["none", "all", "partial"])
    def test_the_loop_never_fails_the_deploy(self, tmp_path, case):
        describe = {
            "none": {},
            "all": self.ALL_GOOD,
            "partial": {k: v for k, v in self.ALL_GOOD.items() if k == "toolweave"},
        }[case]
        proc, _out, _params = self._run(tmp_path, describe)
        assert proc.returncode == 0, proc.stderr
