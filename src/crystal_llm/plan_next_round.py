"""Create an LLM-facing round plan from natural-language memory.

The output is intentionally richer than the executable generator strategy.  It
is a draft plan that can be reviewed or edited by an LLM/human before it is
compiled into the compact strategy consumed by ``crystal_llm.generate``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from crystal_llm.memory import (
    STRICT_SUN_NOTE,
    infer_completed_rounds,
    load_jsonl,
    read_json,
    round_label,
    write_json,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draft the next-round LLM plan from memory records.")
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument("--memory-dir", default="memory", help="Memory directory.")
    parser.add_argument("--strategy", default="strategy.json", help="Current executable strategy JSON.")
    parser.add_argument("--round", type=int, default=None, help="Planned round number. Defaults to latest+1.")
    parser.add_argument("--latest-round", type=int, default=None, help="Latest completed round. Defaults to memory state.")
    parser.add_argument("--output", default=None, help="Output round plan JSON.")
    parser.add_argument("--prompt-output", default=None, help="Optional LLM prompt markdown output.")
    return parser.parse_args(argv)


def latest_round(root: Path, memory_dir: Path, args: argparse.Namespace) -> int:
    if args.latest_round is not None:
        return args.latest_round
    state = read_json(memory_dir / "state.json", {})
    if isinstance(state, dict) and isinstance(state.get("latest_imported_round"), int):
        return int(state["latest_imported_round"])
    completed = infer_completed_rounds(root)
    if not completed:
        raise ValueError("cannot infer latest completed round")
    return completed[-1]


def sort_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(record) for record in records),
        key=lambda item: (int(item.get("round", -1)), str(item.get("id", ""))),
    )


def tail_records(records: Sequence[Mapping[str, Any]], count: int = 12) -> list[dict[str, Any]]:
    return sort_records(records)[-count:]


def collect_candidate_probes(strategy: Mapping[str, Any], planned_round: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    resume = strategy.get("resume_guidance")
    if isinstance(resume, Mapping):
        raw = resume.get("planned_formula_probe_candidates_not_active")
        if isinstance(raw, list):
            candidates.extend(item for item in raw if isinstance(item, dict))

    planned_key = f"round_{planned_round:03d}_planning_consensus"
    planned = strategy.get(planned_key)
    if isinstance(planned, Mapping):
        raw = planned.get("candidate_probes_not_active")
        if isinstance(raw, list):
            candidates.extend(item for item in raw if isinstance(item, dict))

    # If the current strategy is paused after the previous round, the not-active
    # candidates are often stored under that already-created planning key.
    for key, value in strategy.items():
        if not (isinstance(key, str) and key.endswith("_planning_consensus") and isinstance(value, Mapping)):
            continue
        raw = value.get("candidate_probes_not_active")
        if isinstance(raw, list):
            candidates.extend(item for item in raw if isinstance(item, dict))

    deduped: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("id") or f"candidate_{index:03d}")
        deduped[candidate_id] = dict(candidate)
    return list(deduped.values())


def build_prompt(plan: Mapping[str, Any], recent_summaries: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# LLM Round Planning Prompt",
        "",
        "You are planning the next crystal-generation round. Preserve natural-language reasoning; do not over-compress observations into broad boosts.",
        "",
        f"Strict metric: `{STRICT_SUN_NOTE}`",
        "",
        "## Draft Plan JSON",
        "",
        "```json",
        json.dumps(plan, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Recent Round Summaries",
        "",
    ]
    for summary in recent_summaries:
        metrics = summary.get("metrics", {})
        split = summary.get("source_split", {})
        lines.extend(
            [
                f"### {summary.get('round_label')}",
                "",
                f"- strict SUN: `{metrics.get('strict_sun')}`",
                f"- replay SUN: `{split.get('elite_strict_sun')}` / `{split.get('elite_total')}`",
                f"- probe SUN: `{split.get('probe_strict_sun')}` / `{split.get('probe_total')}`",
                f"- ordinary SUN: `{split.get('ordinary_non_probe_strict_sun')}` / `{split.get('ordinary_non_probe_total')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Editing Instructions",
            "",
            "- Keep successful examples, counterexamples, uncertainty, and do-not-generalize clauses in natural language.",
            "- Only mark probes active if their scope is exact and they have been locally validated by the compiler.",
            "- Never treat e_hull < 0.03 or e_hull < 0.10 as SUN.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    memory_dir = (root / args.memory_dir).resolve()
    strategy_path = (root / args.strategy).resolve()
    strategy = read_json(strategy_path, {})
    if not isinstance(strategy, dict):
        raise ValueError("strategy JSON must be an object")

    last_round = latest_round(root, memory_dir, args)
    planned_round = args.round if args.round is not None else last_round + 1
    output = Path(args.output) if args.output else root / "plans" / f"{round_label(planned_round)}_plan.json"
    prompt_output = (
        Path(args.prompt_output)
        if args.prompt_output
        else root / "plans" / f"{round_label(planned_round)}_planner_prompt.md"
    )

    observations = tail_records(load_jsonl(memory_dir / "observations.jsonl"), 16)
    hypotheses = tail_records(load_jsonl(memory_dir / "hypotheses.jsonl"), 12)
    counterexamples = tail_records(load_jsonl(memory_dir / "counterexamples.jsonl"), 16)
    recent_summaries: list[dict[str, Any]] = []
    for round_number in range(max(1, last_round - 2), last_round + 1):
        summary = read_json(memory_dir / "round_summaries" / f"{round_label(round_number)}.json", {})
        if isinstance(summary, dict) and summary:
            recent_summaries.append(summary)

    elite = strategy.get("elite_replay", {})
    if not isinstance(elite, Mapping):
        elite = {}
    exclude_formulas = list(dict.fromkeys(str(item) for item in elite.get("exclude_formulas", []) if item))
    quarantined = list(dict.fromkeys(str(item) for item in elite.get("quarantined_formulas", []) if item))
    for item in quarantined:
        if item not in exclude_formulas:
            exclude_formulas.append(item)

    candidate_probes = collect_candidate_probes(strategy, planned_round)
    plan = {
        "schema_version": "round_plan.v1",
        "round": round_label(planned_round),
        "planned_round_number": planned_round,
        "status": "draft_from_memory",
        "strict_sun_definition": STRICT_SUN_NOTE,
        "memory_sources": {
            "memory_dir": str(memory_dir.relative_to(root) if memory_dir.is_relative_to(root) else memory_dir),
            "latest_completed_round": round_label(last_round),
            "observations_used": [record.get("id") for record in observations],
            "hypotheses_used": [record.get("id") for record in hypotheses],
            "counterexamples_used": [record.get("id") for record in counterexamples],
        },
        "natural_language_reasoning": (
            "Use memory as the primary knowledge store. Replay preserves strict SUN hits, while probes should be "
            "narrow executable tests of natural-language hypotheses. Current memory supports exact Li-halide Br "
            "double-perovskite probes and rejects F/V/near-zero overgeneralization."
        ),
        "actions": {
            "elite_replay": {
                "enabled": True,
                "add_sources_through_round": last_round,
                "exclude_formulas": exclude_formulas,
                "quarantined_formulas": quarantined,
                "rationale": "Stable replay should preserve only strict e_hull < 0 records and exclude volatile exact formulas.",
            },
            "formula_probe_candidates": candidate_probes,
            "activate_formula_probes_by_default": False,
            "do_not_generalize": [
                "Do not use global Li/Br/F/V/Ni/Co boosts from exact probe evidence.",
                "Do not resume F analog probes without independent strict evidence.",
                "Do not resume near-zero boundary probes as a discovery strategy.",
                "Do not count e_hull < 0.03 or e_hull < 0.10 as SUN.",
            ],
        },
        "recent_memory": {
            "observations": observations,
            "hypotheses": hypotheses,
            "counterexamples": counterexamples,
        },
        "compiler_directives": {
            "default_output_strategy": f"strategies/{round_label(planned_round)}_strategy.json",
            "activate_formula_probes": False,
            "require_local_probe_validation": True,
        },
    }

    write_json(output, plan)
    prompt_output.parent.mkdir(parents=True, exist_ok=True)
    prompt_output.write_text(build_prompt(plan, recent_summaries), encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {prompt_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
