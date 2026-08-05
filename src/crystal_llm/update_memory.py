"""Update the natural-language memory store from completed evaluator rounds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from crystal_llm.memory import (
    infer_completed_rounds,
    load_jsonl,
    read_json,
    round_label,
    write_json,
    write_round_memory,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or update the project memory from completed rounds.")
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument("--memory-dir", default="memory", help="Output memory directory.")
    parser.add_argument("--round", type=int, action="append", dest="rounds", help="Round number to import.")
    parser.add_argument("--start", type=int, default=None, help="First round to import when using a range.")
    parser.add_argument("--end", type=int, default=None, help="Last round to import when using a range.")
    parser.add_argument("--all", action="store_true", help="Import all completed rounds found on disk.")
    parser.add_argument("--reset", action="store_true", help="Clear JSONL memory files before importing.")
    return parser.parse_args(argv)


def selected_rounds(args: argparse.Namespace, root: Path) -> list[int]:
    if args.all:
        rounds = infer_completed_rounds(root)
    elif args.rounds:
        rounds = list(args.rounds)
    elif args.start is not None or args.end is not None:
        if args.start is None or args.end is None:
            raise ValueError("provide both --start and --end for range import")
        rounds = list(range(args.start, args.end + 1))
    else:
        completed = infer_completed_rounds(root)
        if not completed:
            raise ValueError("no completed rounds found")
        rounds = [completed[-1]]
    return sorted(set(rounds))


def reset_memory(memory_dir: Path) -> None:
    for name in ("observations.jsonl", "hypotheses.jsonl", "counterexamples.jsonl"):
        path = memory_dir / name
        if path.exists():
            path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    memory_dir = (root / args.memory_dir).resolve()
    rounds = selected_rounds(args, root)

    if args.reset:
        reset_memory(memory_dir)

    imported = []
    for round_number in rounds:
        summary = write_round_memory(root, memory_dir, round_number)
        imported.append(round_number)
        print(
            f"imported {round_label(round_number)}: "
            f"strict_sun={summary['metrics']['strict_sun']} "
            f"probe_sun={summary['source_split']['probe_strict_sun']} "
            f"elite_sun={summary['source_split']['elite_strict_sun']}"
        )

    state = read_json(memory_dir / "state.json", {})
    if not isinstance(state, dict):
        state = {}
    state.update(
        {
            "schema_version": "memory_state.v1",
            "latest_imported_round": max(imported) if imported else state.get("latest_imported_round"),
            "imported_rounds": sorted(set(state.get("imported_rounds", [])) | set(imported)),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": {
                "observations": "observations.jsonl",
                "hypotheses": "hypotheses.jsonl",
                "counterexamples": "counterexamples.jsonl",
                "round_summaries": "round_summaries/",
            },
            "counts": {
                "observations": len(load_jsonl(memory_dir / "observations.jsonl")),
                "hypotheses": len(load_jsonl(memory_dir / "hypotheses.jsonl")),
                "counterexamples": len(load_jsonl(memory_dir / "counterexamples.jsonl")),
            },
        }
    )
    write_json(memory_dir / "state.json", state)
    print(f"updated memory state: {memory_dir / 'state.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
