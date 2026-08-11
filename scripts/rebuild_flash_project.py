#!/usr/bin/env python3
"""Re-run a portable FLASH project's recorded setup command and build it."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
FLASH_ROOT = BUNDLE_ROOT / "FLASH4.8"
PROJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not PROJECT_RE.fullmatch(args.project):
        raise SystemExit("invalid project name")
    project = FLASH_ROOT / args.project
    setup_call = project / "setup_call"
    if not setup_call.is_file():
        raise SystemExit(f"missing setup_call: {setup_call}")

    first_line = next((line.strip() for line in setup_call.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()), "")
    tokens = shlex.split(first_line)
    if not tokens:
        raise SystemExit(f"empty setup_call: {setup_call}")
    setup_args = tokens[1:]
    normalized: list[str] = []
    for token in setup_args:
        if token.startswith("-objdir="):
            token = f"-objdir={args.project}"
        for old in ("/home/user2/Haigerloch/FLASH4.8/", "/home/user/FLASH4.8/", "/home/Von_Hohenschaws/Haigerloch/FLASH4.8/"):
            token = token.replace(old, "")
        normalized.append(token)

    setup_cmd = [sys.executable, str(FLASH_ROOT / "bin" / "setup.py"), *normalized]
    build_cmd = ["make", "-C", str(project), f"-j{max(1, args.jobs)}"]
    print(json.dumps({"setup": setup_cmd, "build": None if args.setup_only else build_cmd}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    subprocess.run(setup_cmd, cwd=FLASH_ROOT, check=True)
    if not args.setup_only:
        subprocess.run(build_cmd, cwd=FLASH_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
