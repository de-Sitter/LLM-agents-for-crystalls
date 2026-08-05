"""Build full-history context files for LLM reflection."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from crystal_llm.experience import compact_experience, load_experiences
from crystal_llm.memory import (
    STRICT_SUN_NOTE,
    build_round_summary,
    infer_completed_rounds,
    load_round_records,
    read_json,
    round_label,
    write_json,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an LLM-readable full-history round context.")
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument("--memory-dir", default="memory", help="Memory directory.")
    parser.add_argument("--round", type=int, default=None, help="Reflection round. Defaults to latest completed round.")
    parser.add_argument("--latest-round", type=int, default=None, help="Latest completed round to include.")
    parser.add_argument("--start-round", type=int, default=1, help="First historical round to include.")
    parser.add_argument("--output", default=None, help="Output context JSON path.")
    parser.add_argument("--markdown-output", default=None, help="Optional compact context markdown path.")
    parser.add_argument("--include-structures", action="store_true", help="Include full pymatgen structure dictionaries.")
    return parser.parse_args(argv)


def latest_completed_round(root: Path, memory_dir: Path, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    state = read_json(memory_dir / "state.json", {})
    if isinstance(state, Mapping) and isinstance(state.get("latest_imported_round"), int):
        return int(state["latest_imported_round"])
    completed = infer_completed_rounds(root)
    if not completed:
        raise ValueError("no completed rounds found")
    return completed[-1]


def compact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": record.get("index"),
        "formula": record.get("formula"),
        "template": record.get("template"),
        "e_hull": record.get("e_hull"),
        "strict_sun": float(record.get("e_hull", 1.0)) < 0.0,
        "nsites": record.get("nsites"),
        "volume_per_atom": record.get("volume_per_atom"),
        "source": {
            "elite_replay": bool(record.get("elite_replay")),
            "elite_source_round": record.get("elite_source_round"),
            "formula_probe": bool(record.get("formula_probe")),
            "formula_probe_id": record.get("formula_probe_id"),
            "formula_probe_family": record.get("formula_probe_family"),
        },
    }


def round_context(root: Path, round_number: int, *, include_structures: bool = False) -> dict[str, Any]:
    summary = build_round_summary(root, round_number)
    records = load_round_records(root, round_number)
    compact_records = [compact_record(record) for record in records]
    template_counts = Counter(str(record.get("template")) for record in compact_records)
    source_counts = {
        "elite_replay": sum(1 for record in compact_records if record["source"]["elite_replay"]),
        "formula_probe": sum(1 for record in compact_records if record["source"]["formula_probe"]),
        "ordinary": sum(
            1
            for record in compact_records
            if not record["source"]["elite_replay"] and not record["source"]["formula_probe"]
        ),
    }
    payload = {
        "round": round_number,
        "round_label": round_label(round_number),
        "summary": summary,
        "template_counts_from_records": dict(template_counts),
        "source_counts_from_records": source_counts,
        "records": compact_records,
    }
    if include_structures:
        input_path = root / summary["source_files"]["input"]
        structures = read_json(input_path, [])
        payload["structures"] = structures if isinstance(structures, list) else []
    return payload


def build_context_payload(
    root: Path,
    memory_dir: Path,
    *,
    reflection_round: int,
    latest_round: int,
    start_round: int = 1,
    include_structures: bool = False,
) -> dict[str, Any]:
    experiences = load_experiences(memory_dir, include_inactive=True)
    active_experiences = load_experiences(memory_dir, include_inactive=False)
    rounds = [
        round_context(root, round_number, include_structures=include_structures)
        for round_number in range(start_round, latest_round + 1)
    ]
    strict_sun_curve = [
        {
            "round": item["round"],
            "strict_sun": item["summary"]["metrics"]["strict_sun"],
            "elite_sun": item["summary"]["source_split"]["elite_strict_sun"],
            "probe_sun": item["summary"]["source_split"]["probe_strict_sun"],
            "ordinary_non_probe_sun": item["summary"]["source_split"]["ordinary_non_probe_strict_sun"],
        }
        for item in rounds
    ]
    return {
        "schema_version": "round_context.v1",
        "reflection_round": reflection_round,
        "latest_completed_round": latest_round,
        "strict_sun_definition": STRICT_SUN_NOTE,
        "history_scope": {
            "start_round": start_round,
            "end_round": latest_round,
            "includes_all_round_records": True,
            "includes_full_structure_dicts": include_structures,
        },
        "strict_sun_curve": strict_sun_curve,
        "existing_experience": {
            "active_positive": [compact_experience(record) for record in active_experiences["positive"]],
            "active_negative": [compact_experience(record) for record in active_experiences["negative"]],
            "all_positive": experiences["positive"],
            "all_negative": experiences["negative"],
        },
        "rounds": rounds,
    }


def context_markdown(context: Mapping[str, Any]) -> str:
    lines = [
        "# LLM Reflection Context",
        "",
        f"- Reflection round: `{context['reflection_round']}`",
        f"- Latest completed round: `{context['latest_completed_round']}`",
        f"- Strict SUN: `{context['strict_sun_definition']}`",
        f"- Includes all compact per-structure records: `{context['history_scope']['includes_all_round_records']}`",
        f"- Includes full structure dicts: `{context['history_scope']['includes_full_structure_dicts']}`",
        "",
        "## SUN Curve",
        "",
        "| Round | Strict SUN | Elite SUN | Probe SUN | Ordinary SUN |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in context.get("strict_sun_curve", []):
        lines.append(
            f"| {item['round']} | {item['strict_sun']} | {item['elite_sun']} | "
            f"{item['probe_sun']} | {item['ordinary_non_probe_sun']} |"
        )
    lines.extend(["", "## Round Summaries", ""])
    for item in context.get("rounds", []):
        summary = item["summary"]
        metrics = summary["metrics"]
        split = summary["source_split"]
        lines.extend(
            [
                f"### {item['round_label']}",
                "",
                f"- strict SUN: `{metrics['strict_sun']}` / `{metrics['total_structures']}`",
                f"- e_hull < 0.03: `{metrics['e_hull_lt_0_03']}`",
                f"- e_hull < 0.10: `{metrics['e_hull_lt_0_10']}`",
                f"- elite replay SUN: `{split['elite_strict_sun']}` / `{split['elite_total']}`",
                f"- probe SUN: `{split['probe_strict_sun']}` / `{split['probe_total']}`",
                f"- ordinary non-probe SUN: `{split['ordinary_non_probe_strict_sun']}` / `{split['ordinary_non_probe_total']}`",
                f"- records included in JSON context: `{len(item.get('records', []))}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    memory_dir = (root / args.memory_dir).resolve()
    latest_round = latest_completed_round(root, memory_dir, args.latest_round)
    reflection_round = args.round if args.round is not None else latest_round
    output = Path(args.output) if args.output else memory_dir / "round_contexts" / f"{round_label(reflection_round)}.json"
    markdown_output = (
        Path(args.markdown_output)
        if args.markdown_output
        else memory_dir / "round_contexts" / f"{round_label(reflection_round)}.md"
    )

    payload = build_context_payload(
        root,
        memory_dir,
        reflection_round=reflection_round,
        latest_round=latest_round,
        start_round=args.start_round,
        include_structures=args.include_structures,
    )
    write_json(output, payload)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(context_markdown(payload), encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {markdown_output}")
    print(f"included_rounds={len(payload['rounds'])}")
    print(f"included_records={sum(len(item.get('records', [])) for item in payload['rounds'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
