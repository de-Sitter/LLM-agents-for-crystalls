"""Compile an LLM round plan into the executable generator strategy JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from crystal_llm.filters import load_known_formulas
from crystal_llm.generate import load_formula_probes
from crystal_llm.memory import read_json, round_label, sources_through_round, write_json


EXECUTABLE_KEYS = (
    "focus_templates",
    "preferred_anions",
    "boost_elements",
    "avoid_elements",
    "template_target_counts",
    "template_max_counts",
    "fallback_template_order",
    "template_rules",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile round_plan JSON to executable strategy JSON.")
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument("--plan", required=True, help="Round plan JSON.")
    parser.add_argument("--base-strategy", default="strategy.json", help="Base strategy to inherit executable defaults from.")
    parser.add_argument("--output", default=None, help="Output strategy JSON.")
    parser.add_argument("--activate-probes", action="store_true", help="Compile candidate probes as active formula_probes.")
    parser.add_argument("--validate-probes", action="store_true", help="Drop active probes that do not load under generator filters.")
    parser.add_argument("--seed", type=int, default=20260525, help="Seed used for local formula probe validation.")
    parser.add_argument("--training-data", default="archive/matllmsearch_evaluator/data/a_training.json")
    parser.add_argument("--max-sites", type=int, default=80)
    return parser.parse_args(argv)


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def unique_strings(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def candidate_probes(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = plan.get("actions", {})
    if not isinstance(actions, Mapping):
        return []
    raw = actions.get("formula_probe_candidates", [])
    if not isinstance(raw, list):
        return []
    probes: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            probe = dict(item)
            probe.pop("local_validation", None)
            probes.append(probe)
    return probes


def validate_active_probes(
    probes: list[dict[str, Any]],
    *,
    seed: int,
    training_data: str | None,
    max_sites: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = {"formula_probes": probes}
    known = load_known_formulas(training_data)
    loaded = load_formula_probes(raw, max_sites=max_sites, base_seed=seed, known_formulas=known)
    loaded_ids = {candidate.metadata.get("formula_probe_id") for candidate in loaded}
    active: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for probe in probes:
        probe_id = str(probe.get("id", ""))
        if probe_id in loaded_ids:
            active.append(probe)
        else:
            skipped.append({"id": probe_id, "reason": "not_loaded_by_generator_filters", "probe": probe})
    return active, skipped


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    plan_path = (root / args.plan).resolve() if not Path(args.plan).is_absolute() else Path(args.plan)
    base_path = (root / args.base_strategy).resolve() if not Path(args.base_strategy).is_absolute() else Path(args.base_strategy)
    plan = read_json(plan_path, {})
    base = read_json(base_path, {})
    if not isinstance(plan, dict):
        raise ValueError("round plan JSON must be an object")
    if not isinstance(base, dict):
        raise ValueError("base strategy JSON must be an object")

    planned_round_number = int(plan.get("planned_round_number"))
    output = (
        (root / "strategies" / f"{round_label(planned_round_number)}_strategy.json")
        if args.output is None
        else ((root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output))
    )

    compiled: dict[str, Any] = {}
    for key in EXECUTABLE_KEYS:
        if key in base:
            compiled[key] = base[key]

    actions = plan.get("actions", {})
    if not isinstance(actions, Mapping):
        actions = {}
    replay = actions.get("elite_replay", {})
    if not isinstance(replay, Mapping):
        replay = {}

    existing_elite = base.get("elite_replay", {})
    if not isinstance(existing_elite, Mapping):
        existing_elite = {}
    exclude_formulas = unique_strings(
        as_list(existing_elite.get("exclude_formulas"))
        + as_list(replay.get("exclude_formulas"))
        + as_list(replay.get("quarantined_formulas"))
    )
    add_sources_through = int(replay.get("add_sources_through_round", plan.get("planned_round_number", 0)) or 0)
    compiled["elite_replay"] = {
        "enabled": bool(replay.get("enabled", True)),
        "e_hull_max": 0.0,
        "max_count": int(existing_elite.get("max_count", 140)),
        "deduplicate_by": "reduced_formula",
        "sources": sources_through_round(add_sources_through),
        "exclude_formulas": exclude_formulas,
        "quarantined_formulas": unique_strings(as_list(replay.get("quarantined_formulas"))),
        "note": "Compiled from natural-language memory round plan. Replay remains strict e_hull < 0 only.",
    }

    probes = candidate_probes(plan) if args.activate_probes else []
    skipped: list[dict[str, Any]] = []
    if probes and args.validate_probes:
        probes, skipped = validate_active_probes(
            probes,
            seed=args.seed,
            training_data=str(root / args.training_data) if args.training_data else None,
            max_sites=args.max_sites,
        )
    compiled["formula_probes"] = probes

    compiled.update(
        {
            "schema_version": "strategy.v4_memory_compiled",
            "iteration_id": f"{round_label(planned_round_number)}_compiled_from_memory_plan",
            "architecture_version": "memory_plan_compile.v1",
            "plan_source": str(plan_path.relative_to(root) if plan_path.is_relative_to(root) else plan_path),
            "strict_sun_definition": plan.get("strict_sun_definition"),
            "memory_architecture": {
                "memory_is_natural_language": True,
                "round_plan_is_llm_facing": True,
                "strategy_is_executable_projection": True,
                "note": "Do not treat this executable JSON as the complete project memory.",
            },
            "resume_guidance": {
                "do_not_auto_submit_next_round": True,
                "recommended_if_resuming": plan.get("natural_language_reasoning"),
                "planned_formula_probe_candidates_not_active": [] if args.activate_probes else candidate_probes(plan),
                "do_not_generalize": actions.get("do_not_generalize", []),
                "probe_validation": {
                    "active_probe_count": len(probes),
                    "skipped": skipped,
                    "seed": args.seed if args.validate_probes else None,
                },
            },
            f"{round_label(planned_round_number)}_planning_consensus": {
                "status": "compiled_not_submitted",
                "round_plan": str(plan_path.relative_to(root) if plan_path.is_relative_to(root) else plan_path),
                "natural_language_reasoning": plan.get("natural_language_reasoning"),
                "actions": actions,
                "compiler_options": {
                    "activate_probes": args.activate_probes,
                    "validate_probes": args.validate_probes,
                    "seed": args.seed,
                },
            },
        }
    )

    direct = dict(base.get("direct_strategy", {})) if isinstance(base.get("direct_strategy"), Mapping) else {}
    for key in EXECUTABLE_KEYS + ("elite_replay", "formula_probes"):
        if key in compiled:
            direct[key] = compiled[key]
    direct.update(
        {
            "usable_by_current_generator": True,
            "note": "Compiled executable projection from round plan; natural-language memory remains in memory/.",
        }
    )
    compiled["direct_strategy"] = direct

    write_json(output, compiled)
    print(f"wrote {output}")
    print(f"active_formula_probes={len(probes)}")
    if skipped:
        print(f"skipped_formula_probes={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
