"""Refuse to ship an AgentCore artifact with the wrong CPU architecture.

AgentCore runtimes run on Linux ARM64. CI builds the zip on an x86-64 runner,
so a plain `pip install --target` resolves x86-64 wheels for anything with a
compiled extension -- bedrock-agentcore pulls in two, pydantic_core and
websockets.speedups. The template is valid, the zip uploads, CloudFormation
accepts the resource, and the runtime then fails to start with:

    Your artifact contains binary files that are incompatible with Linux ARM64.

That is a nine-minute deploy and a full stack rollback to learn something a
byte in each file's ELF header already said.

So this reads that byte. Filenames are a hint, not evidence: a wheel can name
itself anything, and `find -name '*x86_64*'` would miss a binary that simply
does not say. e_machine at offset 18 is what the loader itself reads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ELF e_machine values (offset 18, 2 bytes, endianness per EI_DATA at offset 5).
EM_X86_64 = 0x3E
EM_AARCH64 = 0xB7
EM_NAMES = {EM_X86_64: "x86-64", EM_AARCH64: "AArch64", 0x03: "x86", 0x28: "ARM"}

ELF_MAGIC = b"\x7fELF"
# Extensions worth reading. A compiled extension is a .so; the rest of a wheel
# is Python source and carries no architecture.
BINARY_SUFFIXES = (".so",)


def elf_machine(path: Path) -> int | None:
    """The ELF e_machine of this file, or None if it is not an ELF object."""
    with path.open("rb") as handle:
        header = handle.read(20)
    if len(header) < 20 or header[:4] != ELF_MAGIC:
        return None
    # EI_DATA: 1 = little endian, 2 = big endian.
    byteorder = "little" if header[5] == 1 else "big"
    return int.from_bytes(header[18:20], byteorder)


def scan(root: Path, expected: int) -> tuple[list[str], int]:
    problems: list[str] = []
    checked = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in BINARY_SUFFIXES:
            continue
        machine = elf_machine(path)
        if machine is None:
            continue
        checked += 1
        if machine != expected:
            found = EM_NAMES.get(machine, hex(machine))
            problems.append(f"{path.relative_to(root)}: built for {found}")
    return problems, checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="the built artifact directory")
    parser.add_argument("--expect", default="aarch64", choices=["aarch64", "x86_64"])
    args = parser.parse_args()

    root = Path(args.directory)
    if not root.is_dir():
        print(f"{root} is not a directory", file=sys.stderr)
        return 1

    expected = EM_AARCH64 if args.expect == "aarch64" else EM_X86_64
    problems, checked = scan(root, expected)

    if problems:
        print(f"Artifact is not {args.expect}. The runtime would fail to start:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    # Zero binaries is reported, never celebrated: it means either a pure
    # Python bundle or a scan that looked in the wrong place, and those must
    # not read the same.
    if checked == 0:
        print(f"No compiled extensions found under {root} (pure Python, or nothing was installed).")
    else:
        print(f"All {checked} compiled extension(s) under {root} are {args.expect}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
