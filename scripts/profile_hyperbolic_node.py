#!/usr/bin/env python3
"""
Run a pytest node under cProfile and capture Plum pending-cache diagnostics.
"""

from __future__ import annotations

import argparse
import cProfile
import json
from dataclasses import asdict, dataclass
from typing import Any

import pytest
from plum.function import Function


@dataclass
class FunctionState:
    obj_id: int
    name: str
    pending: int
    resolved: int
    cache: int
    resolver_methods: int
    resolver_faithful: bool | None


def function_name(f: Function) -> str:
    qualname = getattr(f, "__qualname__", None)
    if qualname:
        return str(qualname)
    name = getattr(f, "__name__", None)
    if name:
        return str(name)
    resolver = getattr(f, "_resolver", None)
    resolver_name = getattr(resolver, "function_name", None)
    if resolver_name:
        return str(resolver_name)
    return repr(f)


def snapshot_states() -> dict[int, FunctionState]:
    states: dict[int, FunctionState] = {}
    for f in Function._instances:
        resolver = getattr(f, "_resolver", None)
        try:
            resolver_methods = len(resolver) if resolver is not None else 0
        except Exception:
            resolver_methods = -1
        resolver_faithful = getattr(resolver, "is_faithful", None)
        states[id(f)] = FunctionState(
            obj_id=id(f),
            name=function_name(f),
            pending=len(getattr(f, "_pending", [])),
            resolved=len(getattr(f, "_resolved", [])),
            cache=len(getattr(f, "_cache", {})),
            resolver_methods=resolver_methods,
            resolver_faithful=resolver_faithful,
        )
    return states


def sorted_pending(states: dict[int, FunctionState]) -> list[FunctionState]:
    return sorted(
        (s for s in states.values() if s.pending > 0),
        key=lambda s: (s.pending, s.resolved, s.cache, s.name),
        reverse=True,
    )


def render_report(
    before: dict[int, FunctionState],
    after: dict[int, FunctionState],
    pytest_exit: int,
    limit: int = 200,
) -> str:
    before_pending = sorted_pending(before)
    after_pending = sorted_pending(after)

    persistent_pending = sorted(
        (
            after[obj_id]
            for obj_id in set(before).intersection(after)
            if before[obj_id].pending > 0 and after[obj_id].pending > 0
        ),
        key=lambda s: (s.pending, s.resolved, s.cache, s.name),
        reverse=True,
    )
    new_pending = sorted(
        (
            s
            for obj_id, s in after.items()
            if s.pending > 0 and (obj_id not in before or before[obj_id].pending == 0)
        ),
        key=lambda s: (s.pending, s.resolved, s.cache, s.name),
        reverse=True,
    )

    lines: list[str] = []
    lines.append(f"pytest_exit={pytest_exit}")
    lines.append("")
    lines.append("Snapshot Summary")
    lines.append(
        f"before: total_functions={len(before)} pending_functions={len(before_pending)} "
        f"total_pending={sum(s.pending for s in before_pending)}"
    )
    lines.append(
        f"after: total_functions={len(after)} pending_functions={len(after_pending)} "
        f"total_pending={sum(s.pending for s in after_pending)}"
    )
    lines.append("")

    def emit_table(title: str, rows: list[FunctionState]) -> None:
        lines.append(title)
        lines.append("pending resolved cache methods faithful name")
        for s in rows[:limit]:
            lines.append(
                f"{s.pending:7d} {s.resolved:8d} {s.cache:5d} "
                f"{s.resolver_methods:7d} {str(s.resolver_faithful):8s} {s.name}"
            )
        if len(rows) > limit:
            lines.append(f"... truncated {len(rows) - limit} rows")
        lines.append("")

    emit_table("After: Functions With Non-Empty _pending", after_pending)
    emit_table("Persistent Pending (Before and After)", persistent_pending)
    emit_table("Newly Pending After Pytest", new_pending)

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pstats-out", required=True)
    parser.add_argument("--pending-out", required=True)
    parser.add_argument("--pending-json-out")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Pass pytest args after '--'",
    )
    args = parser.parse_args()

    pytest_args = args.pytest_args
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    if not pytest_args:
        pytest_args = [
            "-q",
            "-s",
            "--maxfail=1",
            "--durations=20",
            "tests/spaces/test_hyperbolic.py::test_equivalence_kernel[numpy-2.0-2]",
        ]

    before = snapshot_states()

    profiler = cProfile.Profile()
    profiler.enable()
    pytest_exit = pytest.main(pytest_args)
    profiler.disable()
    profiler.dump_stats(args.pstats_out)

    after = snapshot_states()
    report = render_report(before, after, pytest_exit=pytest_exit, limit=args.limit)
    with open(args.pending_out, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    if args.pending_json_out:
        payload: dict[str, Any] = {
            "pytest_exit": pytest_exit,
            "before": [asdict(s) for s in before.values()],
            "after": [asdict(s) for s in after.values()],
        }
        with open(args.pending_json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return pytest_exit


if __name__ == "__main__":
    raise SystemExit(main())
