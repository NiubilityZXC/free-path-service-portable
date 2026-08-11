#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait until an HTTP URL responds successfully.")
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(args.url, timeout=min(3.0, max(args.interval, 0.2))) as response:
                if 200 <= response.status < 400:
                    if not args.quiet:
                        print(json.dumps({"ok": True, "url": args.url, "status": response.status}, ensure_ascii=False))
                    return 0
                last_error = f"HTTP {response.status}"
        except (OSError, URLError) as exc:
            last_error = str(exc)
        time.sleep(args.interval)
    if not args.quiet:
        print(json.dumps({"ok": False, "url": args.url, "error": last_error}, ensure_ascii=False), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
