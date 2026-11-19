from __future__ import annotations

import re
import sys

QUERY = re.compile(r"^Query .+ is (true|false)\.$", re.MULTILINE)


def main() -> int:
    output = sys.stdin.read()
    sys.stdout.write(output)
    if "Verification summary:" not in output:
        print("ProVerif verification summary is missing", file=sys.stderr)
        return 1
    queries = list(QUERY.finditer(output))
    if not queries:
        print("ProVerif verification summary contains no query results", file=sys.stderr)
        return 1
    failed = [match.group(0) for match in queries if match.group(1) == "false"]
    if failed:
        print("ProVerif query failed: " + "; ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
