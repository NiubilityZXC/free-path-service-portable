#!/usr/bin/env python3
"""Audit or repair FLASH project links that still point at the source machine.

The copied FLASH object/project directories contain many relative links whose
text ultimately resolves to ``/home/user2/Haigerloch/FLASH4.8``.  Those links
work on the source host but break after moving the portable bundle.  This tool
rewrites only links containing that exact legacy marker and makes each target
relative to the bundled ``FLASH4.8`` directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
FLASH_ROOT = BUNDLE_ROOT / "FLASH4.8"
LEGACY_MARKER = "/user2/Haigerloch/FLASH4.8/"


def inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def scan(*, apply: bool) -> dict[str, object]:
    legacy: list[tuple[Path, str, Path]] = []
    missing: list[str] = []

    for link in FLASH_ROOT.rglob("*"):
        if not link.is_symlink():
            continue
        old_target = os.readlink(link)
        if LEGACY_MARKER not in old_target:
            continue
        suffix = old_target.split(LEGACY_MARKER, 1)[1]
        candidate = Path(suffix)
        if candidate.is_absolute() or ".." in candidate.parts:
            missing.append(f"{link.relative_to(BUNDLE_ROOT)} -> unsafe suffix {suffix!r}")
            continue
        destination = FLASH_ROOT / candidate
        if not os.path.lexists(destination):
            missing.append(f"{link.relative_to(BUNDLE_ROOT)} -> missing {candidate}")
            continue
        legacy.append((link, old_target, destination))

    if apply and missing:
        raise RuntimeError(f"refusing to modify links: {len(missing)} destinations are invalid")

    changed = 0
    if apply:
        for link, _old_target, destination in legacy:
            new_target = os.path.relpath(destination, start=link.parent)
            temporary = link.with_name(f".{link.name}.portable-link-tmp")
            if os.path.lexists(temporary):
                temporary.unlink()
            os.symlink(new_target, temporary)
            os.replace(temporary, link)
            changed += 1

    total_links = 0
    broken: list[str] = []
    outside_bundle: list[str] = []
    remaining_legacy: list[str] = []
    for link in BUNDLE_ROOT.rglob("*"):
        if not link.is_symlink():
            continue
        total_links += 1
        target = os.readlink(link)
        if LEGACY_MARKER in target:
            remaining_legacy.append(str(link.relative_to(BUNDLE_ROOT)))
        if not link.exists():
            broken.append(str(link.relative_to(BUNDLE_ROOT)))
            continue
        resolved = link.resolve()
        if not inside(resolved, BUNDLE_ROOT):
            outside_bundle.append(
                f"{link.relative_to(BUNDLE_ROOT)} -> {resolved}"
            )

    return {
        "ok": not missing and not broken and not outside_bundle and (apply or not remaining_legacy),
        "mode": "apply" if apply else "audit",
        "bundle_root": str(BUNDLE_ROOT),
        "total_links": total_links,
        "legacy_links_found": len(legacy),
        "changed": changed,
        "invalid_destinations": len(missing),
        "remaining_legacy": len(remaining_legacy),
        "broken": len(broken),
        "outside_bundle": len(outside_bundle),
        "samples": {
            "invalid_destinations": missing[:10],
            "remaining_legacy": remaining_legacy[:10],
            "broken": broken[:10],
            "outside_bundle": outside_bundle[:10],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="atomically rewrite validated legacy links; default is audit only",
    )
    args = parser.parse_args()
    result = scan(apply=args.apply)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
