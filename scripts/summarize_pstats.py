#!/usr/bin/env python3
"""
Summarize a cProfile .pstats file into plain text.
"""

from __future__ import annotations

import argparse
import io
import pstats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pstats_path", help="Path to .pstats file")
    parser.add_argument(
        "--out",
        required=True,
        help="Output text file path",
    )
    parser.add_argument(
        "--sort",
        default="cumtime",
        choices=["cumtime", "tottime", "calls", "time"],
        help="Sort key for pstats output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=120,
        help="Number of entries to print",
    )
    args = parser.parse_args()

    stream = io.StringIO()
    stats = pstats.Stats(args.pstats_path, stream=stream).sort_stats(args.sort)
    stats.print_stats(args.limit)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(stream.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
