#!/usr/bin/env python3
"""Build the submission archive reproducibly.

The archive root is the contents of code/, so that evaluation/usage_report.md sits at the
archive root as the submission requires. Secrets, caches, virtualenvs and bytecode are
excluded by construction rather than by remembering to delete them.

    python3 code/evaluation/package.py
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent

EXCLUDED_DIRECTORIES = {"__pycache__", ".cache", ".venv", "venv", ".git", ".pytest_cache", ".ruff_cache"}
EXCLUDED_NAMES = {".env", ".DS_Store"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log"}


def included_files(root: Path) -> list[Path]:
    keep: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if EXCLUDED_DIRECTORIES.intersection(relative.parts):
            continue
        if path.name in EXCLUDED_NAMES or path.suffix in EXCLUDED_SUFFIXES:
            continue
        keep.append(path)
    return keep


def build(destination: Path) -> tuple[int, list[str]]:
    files = included_files(CODE_ROOT)
    names = [str(path.relative_to(CODE_ROOT)) for path in files]
    if "evaluation/usage_report.md" not in names:
        raise SystemExit("refusing to package: evaluation/usage_report.md is missing from the archive root")
    leaked = [name for name in names if Path(name).name in EXCLUDED_NAMES]
    if leaked:
        raise SystemExit(f"refusing to package: secrets would be included: {leaked}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in zip(files, names):
            archive.write(path, name)
    return len(files), names


def main() -> int:
    parser = argparse.ArgumentParser(description="Package the submission archive")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "code.zip")
    args = parser.parse_args()
    count, names = build(args.output.resolve())
    print(f"packaged {count} files into {args.output.resolve()}")
    for name in names:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
