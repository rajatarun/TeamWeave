"""Every script the deploy invokes must exist from the directory it runs in.

The deploy died mid-run with

    python: can't open file
    '/home/runner/work/TeamWeave/TeamWeave/infra/scripts/mcp_handshake.py':
    [Errno 2] No such file or directory

because the SAM Deploy step does `cd infra` and the call used a repo-root
relative path. Nothing offline saw it: the template was valid, the suite green,
and the loop's own tests run bash in a temporary directory with a stub script,
so they exercise the branching and not the path.

This resolves each `python <path>.py` against the working directory its step
actually has, tracking `cd` within the step's script the way the shell does.
It is the one check that could have caught it without deploying.
"""
from __future__ import annotations

import pathlib
import re

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO / ".github" / "workflows"

# Shell variables that name the checkout root. Both resolve to the repository
# root, and a path built on either is cwd-independent -- which is the point.
ROOT_VARS = ("${GITHUB_WORKSPACE}", "$GITHUB_WORKSPACE", "${REPO_ROOT}", "$REPO_ROOT")

INVOCATION = re.compile(r"\bpython3?\s+(?:-m\s+\S+|\"([^\"]+\.py)\"|'([^']+\.py)'|(\S+\.py))")
CD = re.compile(r"^\s*cd\s+([^\s;&|]+)\s*$")


def steps():
    """(workflow, job, step name, default working-directory, script) tuples."""
    out = []
    for path in sorted(WORKFLOW_DIR.glob("*.y*ml")):
        doc = yaml.safe_load(path.read_text()) or {}
        for job_name, job in (doc.get("jobs") or {}).items():
            job_wd = ((job.get("defaults") or {}).get("run") or {}).get("working-directory", "")
            for index, step in enumerate(job.get("steps") or []):
                script = step.get("run")
                if not script:
                    continue
                wd = ((step.get("defaults") or {}).get("run") or {}).get(
                    "working-directory", step.get("working-directory", job_wd)
                )
                out.append((path.name, job_name, step.get("name", f"step {index}"), wd, script))
    return out


def invocations(script: str, start_dir: pathlib.Path):
    """Yield (line, resolved path) for each python script call in `script`.

    `cd` is followed line by line, which is how the shell reads it. A `cd` in a
    subshell or a branch would over- or under-shoot; that is deliberate, since
    the check should be simple enough to trust and this workflow's `cd`s are
    all top level.
    """
    cwd = start_dir
    for line in script.splitlines():
        moved = CD.match(line)
        if moved:
            target = moved.group(1)
            if not target.startswith(("$", "/", "~")):
                cwd = (cwd / target).resolve()
            continue
        match = INVOCATION.search(line)
        if not match:
            continue
        raw = next((g for g in match.groups() if g), None)
        if raw is None:
            continue
        resolved = raw
        for var in ROOT_VARS:
            if resolved.startswith(var + "/"):
                resolved = resolved[len(var) + 1:]
                yield line.strip(), (REPO / resolved)
                break
        else:
            if resolved.startswith(("$", "/")):
                continue  # some other variable or an absolute path -- not ours to judge
            yield line.strip(), (cwd / resolved)


ALL = [
    (workflow, job, name, wd, script)
    for workflow, job, name, wd, script in steps()
]


@pytest.mark.parametrize("workflow,job,name,wd,script", ALL,
                         ids=[f"{w}:{n}" for w, _j, n, _d, _s in ALL])
def test_every_script_the_deploy_runs_exists_from_where_it_runs(workflow, job, name, wd, script):
    start = (REPO / wd).resolve() if wd else REPO
    missing = [
        (line, path) for line, path in invocations(script, start) if not path.is_file()
    ]
    assert not missing, (
        f"{workflow} / {name}: these resolve to nothing from "
        f"{start.relative_to(REPO) if start != REPO else '.'} — "
        + "; ".join(f"{line!r} -> {path}" for line, path in missing)
    )


def test_the_check_actually_finds_invocations():
    """A regex that matches nothing passes every step above in silence -- the
    same shape as the IPv6 JMESPath that filtered to an empty list."""
    found = [
        path
        for _w, _j, _n, wd, script in ALL
        for _line, path in invocations(script, (REPO / wd).resolve() if wd else REPO)
    ]
    assert len(found) >= 8, f"only found {len(found)} script invocations; the regex has stopped matching"


@pytest.mark.parametrize("workflow,job,name,wd,script", ALL,
                         ids=[f"{w}:{n}" for w, _j, n, _d, _s in ALL])
def test_a_step_that_uses_a_root_variable_defines_it(workflow, job, name, wd, script):
    """`${REPO_ROOT}` is this repository's own convention, not something the
    runner sets. Unset, it expands to nothing and the path becomes
    /scripts/... -- the same "can't open file" as the relative path, so the
    check above must not accept it on faith."""
    if "REPO_ROOT" not in script:
        return
    assert re.search(r"^\s*REPO_ROOT=", script, re.MULTILINE), (
        f"{workflow} / {name} uses ${{REPO_ROOT}} without assigning it; it "
        f"expands to nothing and every path built on it starts at /"
    )


def test_the_probe_is_called_by_an_absolute_path():
    """It is invoked from inside the step that does `cd infra`, so a
    repo-root-relative path is wrong there however right it looks."""
    workflow = (WORKFLOW_DIR / "deploy.yml").read_text()
    line = next(l for l in workflow.splitlines() if "mcp_handshake.py" in l and "python" in l)
    assert any(var in line for var in ROOT_VARS), (
        "mcp_handshake.py is called by a relative path from a step that changes "
        f"directory: {line.strip()!r}"
    )
