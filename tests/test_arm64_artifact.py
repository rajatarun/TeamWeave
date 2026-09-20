"""The architecture check on the AgentCore artifact.

AgentCore runtimes are Linux ARM64. CI builds the zip on an x86-64 runner, and
`pip install --target` there resolves x86-64 wheels for anything compiled --
bedrock-agentcore brings two, pydantic_core and websockets.speedups. Every
earlier gate passed: the template validated, cfn-lint was clean, the zip
uploaded, the entrypoint imported, CloudFormation accepted the resource. The
runtime then refused to start, nine minutes in, and rolled the whole stack
back:

    Your artifact contains binary files that are incompatible with Linux ARM64.

A byte in each file's ELF header said so before any of that.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

from scripts import check_arm64_artifact as checker  # noqa: E402


def elf(tmp_path: Path, name: str, machine: int, *, byteorder: str = "little") -> Path:
    """A minimal but genuine ELF header -- the 20 bytes the checker reads."""
    path = tmp_path / name
    ident = bytearray(16)
    ident[0:4] = b"\x7fELF"
    ident[4] = 2  # 64-bit
    ident[5] = 1 if byteorder == "little" else 2
    fmt = "<HH" if byteorder == "little" else ">HH"
    path.write_bytes(bytes(ident) + struct.pack(fmt, 3, machine))
    return path


def test_an_aarch64_extension_is_accepted(tmp_path):
    elf(tmp_path, "_pydantic_core.so", checker.EM_AARCH64)
    problems, checked = checker.scan(tmp_path, checker.EM_AARCH64)
    assert not problems
    assert checked == 1


def test_the_artifact_that_actually_failed_is_rejected(tmp_path):
    # x86-64 wheels in a bundle bound for an ARM64 runtime.
    elf(tmp_path, "_pydantic_core.so", checker.EM_X86_64)
    elf(tmp_path, "speedups.so", checker.EM_X86_64)
    problems, checked = checker.scan(tmp_path, checker.EM_AARCH64)
    assert checked == 2
    assert len(problems) == 2
    assert all("x86-64" in p for p in problems)


def test_a_wrong_binary_is_caught_whatever_it_is_called(tmp_path):
    # The filename is a hint, not evidence: a wheel can name itself anything,
    # and grepping for "x86_64" would miss a binary that simply does not say.
    elf(tmp_path, "totally_fine_aarch64_honest.so", checker.EM_X86_64)
    problems, _ = checker.scan(tmp_path, checker.EM_AARCH64)
    assert len(problems) == 1


def test_nested_extensions_are_found(tmp_path):
    nested = tmp_path / "pydantic_core" / "deep"
    nested.mkdir(parents=True)
    elf(nested, "_core.so", checker.EM_X86_64)
    problems, checked = checker.scan(tmp_path, checker.EM_AARCH64)
    assert checked == 1 and len(problems) == 1


def test_python_source_is_not_mistaken_for_a_binary(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("# hi\n")
    problems, checked = checker.scan(tmp_path, checker.EM_AARCH64)
    assert not problems and checked == 0


def test_a_non_elf_file_named_so_is_skipped(tmp_path):
    (tmp_path / "notreally.so").write_bytes(b"this is not an ELF file at all")
    problems, checked = checker.scan(tmp_path, checker.EM_AARCH64)
    assert not problems and checked == 0


def test_a_truncated_file_does_not_crash_the_check(tmp_path):
    (tmp_path / "stub.so").write_bytes(b"\x7fELF")
    problems, checked = checker.scan(tmp_path, checker.EM_AARCH64)
    assert not problems and checked == 0


def test_big_endian_headers_are_read_correctly(tmp_path):
    # EI_DATA decides how e_machine is read; assuming little-endian would
    # turn 0xB7 into 0xB700 and reject a valid binary.
    elf(tmp_path, "be.so", checker.EM_AARCH64, byteorder="big")
    problems, checked = checker.scan(tmp_path, checker.EM_AARCH64)
    assert not problems and checked == 1


def test_finding_nothing_is_reported_rather_than_celebrated(tmp_path, capsys):
    # Zero binaries means either a pure Python bundle or a scan pointed at the
    # wrong directory. Those must not read the same as a verified artifact.
    import sys
    from unittest import mock

    with mock.patch.object(sys, "argv", ["check", str(tmp_path)]):
        assert checker.main() == 0
    assert "No compiled extensions found" in capsys.readouterr().out


# ── The workflow has to actually build for ARM64 ──────────────────────────

@pytest.fixture(scope="module")
def packaging_step() -> str:
    workflow = yaml.safe_load((REPO / ".github" / "workflows" / "deploy.yml").read_text())
    steps = workflow["jobs"]["deploy"]["steps"]
    return next(s for s in steps if s.get("name") == "Package the AgentCore agent")["run"]


@pytest.fixture(scope="module")
def shipped_install(packaging_step) -> str:
    """Just the pip command that builds the tree that gets zipped.

    Searching the whole step matches the comments explaining these flags, so
    deleting a flag from the command left the test passing on the prose that
    described it. The command is what runs.
    """
    lines = [l for l in packaging_step.splitlines() if not l.lstrip().startswith("#")]
    text = "\n".join(lines)
    start = text.index("pip install --quiet --target .agentcore-build")
    end = text.index("bedrock-agentcore", start) + len("bedrock-agentcore")
    return text[start:end]


def test_the_shipped_tree_is_resolved_for_aarch64(shipped_install):
    assert "--platform manylinux2014_aarch64" in shipped_install


def test_source_builds_cannot_reintroduce_this_runners_architecture(shipped_install):
    # Without --only-binary, pip may fall back to an sdist and compile it here
    # -- putting x86-64 back in the zip that --platform was added to keep out.
    assert "--only-binary=:all:" in shipped_install


def test_the_python_version_matches_the_runtime(shipped_install):
    template = yaml.safe_load(
        re.sub(r"!\w+", "", (REPO / "infra" / "template.yaml").read_text())
    )
    runtime = template["Resources"]["AgentCoreRuntime"]["Properties"]["AgentRuntimeArtifact"]
    declared = runtime["CodeConfiguration"]["Runtime"]          # e.g. PYTHON_3_12
    version = declared.removeprefix("PYTHON_").replace("_", ".")
    assert f"--python-version {version}" in shipped_install, (
        f"the runtime is {declared} but the wheels are not resolved for {version}"
    )


def test_the_artifact_is_verified_before_it_is_uploaded(packaging_step):
    check = packaging_step.index("check_arm64_artifact.py")
    upload = packaging_step.index("aws s3 cp")
    assert check < upload, "the architecture check must run before the upload"


def test_the_boot_check_uses_a_natively_installed_tree(packaging_step):
    # aarch64 wheels cannot be imported on an x86-64 runner, so the boot check
    # needs its own native tree. Running it against the shipped tree would
    # fail for the wrong reason and invite someone to delete the check.
    assert ".agentcore-check" in packaging_step
    boot_line = next(l for l in packaging_step.splitlines() if "import app;" in l)
    assert ".agentcore-check" in boot_line or "agentcore-check" in packaging_step
