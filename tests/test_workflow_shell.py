"""Shell constructs that end a GitHub step early, silently.

Every `run:` block runs as `/usr/bin/bash -e {0}`, so a **bare** command
returning non-zero ends the step there:

    cmd                       # reports its outcome as an exit code
    status=$?                 # never reached: the step already ended

That is what produced "Process completed with exit code 2" on the handshake
probe's `unverified` result -- the outcome that exists in order *not* to fail
the deploy. The fix is `cmd || status=$?`, which puts the call in a `&&`/`||`
list where `-e` does not apply.

`[ -n "${X}" ] && push` looks like the same trap and is **not** one: a failing
command in a `&&` list is exempt, and bash does not exit on the list's status
either. Verified rather than assumed --

    set -e; X=value; [ "$X" = "None" ] && X=""; echo reached   # prints

-- because an earlier pass through this file "fixed" those lines on the belief
that they were landmines, which added noise and a check that failed correct
code.

This is static because the behavioural harness extracts only the MCP
resolution loop; a harness running the whole step would need stubs for every
AWS call in it.
"""
from __future__ import annotations

import re

import pytest
import yaml

from tests.test_workflow_script_paths import REPO, WORKFLOW_DIR, steps  # noqa: F401

# `x=$?` reading the status of a command that was not guarded on the line above.
READS_STATUS = re.compile(r"^\s*\w+=\$\?\s*$")

GUARDED = ("||", "if ", "while ", "until ", "&& \\")


def run_blocks():
    out = []
    for path in sorted(WORKFLOW_DIR.glob("*.y*ml")):
        doc = yaml.safe_load(path.read_text()) or {}
        for job_name, job in (doc.get("jobs") or {}).items():
            for index, step in enumerate(job.get("steps") or []):
                script = step.get("run")
                if script:
                    out.append((path.name, step.get("name", f"step {index}"), script))
    return out


BLOCKS = run_blocks()
IDS = [f"{w}:{n}" for w, n, _s in BLOCKS]


@pytest.mark.parametrize("workflow,name,script", BLOCKS, ids=IDS)
def test_exit_status_is_captured_on_the_same_line(workflow, name, script):
    """`cmd` then `status=$?` never reaches the second line under -e."""
    lines = script.splitlines()
    offenders = []
    for index, line in enumerate(lines):
        if not READS_STATUS.match(line):
            continue
        previous = next(
            (lines[j].strip() for j in range(index - 1, -1, -1)
             if lines[j].strip() and not lines[j].strip().startswith("#")),
            "",
        )
        if not any(token in previous for token in GUARDED):
            offenders.append(f"{previous!r} -> {line.strip()!r}")
    assert not offenders, (
        f"{workflow} / {name}: the command exits the step before its status is "
        f"read — capture it as `cmd || status=$?`: " + "; ".join(offenders)
    )


def test_the_patterns_still_match_something():
    """A regex that matches nothing passes every block in silence."""
    assert READS_STATUS.match("              mcp_probe=$?")
    assert BLOCKS, "no run: blocks found -- the parser has stopped working"
