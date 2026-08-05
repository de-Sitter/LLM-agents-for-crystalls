"""Deterministic compact memory for the hypothesis-first MVP."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from crystal_llm.experience import load_experiences
from crystal_llm.memory import STRICT_SUN_NOTE, read_json, write_json


SCHEMA_VERSION = "compact_hypothesis_memory.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def truncate_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def slugify(value: str, limit: int = 96) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return (text[:limit] or "hypothesis").rstrip("_")


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def top_counter(counter: Counter[str], limit: int = 8) -> list[str]:
    return [key for key, _ in counter.most_common(limit)]


def sorted_numeric(values: Iterable[int], limit: int = 20) -> list[int]:
    unique = sorted(set(values))
    if len(unique) <= limit:
        return unique
    head = unique[: limit // 2]
    tail = unique[-(limit - len(head)) :]
    return head + tail


def compact_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    roles = probe.get("roles")
    compact_roles: dict[str, Any] = {}
    if isinstance(roles, Mapping):
        for role, value in roles.items():
            if not isinstance(value, Mapping):
                continue
            compact_roles[str(role)] = {
                "element": value.get("element"),
                "oxidation_state": value.get("oxidation_state"),
            }
    return {
        "template": probe.get("template"),
        "family": probe.get("family"),
        "roles": compact_roles,
        "hypothesis_ids": string_list(probe.get("hypothesis_ids")),
        "rationale_summary": truncate_text(probe.get("rationale_summary"), 220),
    }


def round_formula_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    analysis = record.get("analysis_summary")
    if not isinstance(analysis, Mapping):
        analysis = {}
    top = analysis.get("top_25")
    ranked: list[Mapping[str, Any]] = [item for item in top if isinstance(item, Mapping)] if isinstance(top, list) else []
    probes = record.get("materialized_formula_probes")
    probe_list: list[Mapping[str, Any]] = [item for item in probes if isinstance(item, Mapping)] if isinstance(probes, list) else []
    validation = record.get("materializer_validation")
    formulas = validation.get("formulas") if isinstance(validation, Mapping) else None
    formula_by_index = list(formulas) if isinstance(formulas, list) else []
    probe_by_formula: dict[str, Mapping[str, Any]] = {}
    for index, probe in enumerate(probe_list):
        if index < len(formula_by_index):
            probe_by_formula[str(formula_by_index[index])] = probe

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ranked:
        formula = str(item.get("formula") or "").strip()
        if not formula:
            continue
        seen.add(formula)
        probe = probe_by_formula.get(formula, {})
        rows.append(
            {
                "formula": formula,
                "e_hull": safe_float(item.get("e_hull")),
                "template": item.get("template_guess") or item.get("template") or probe.get("template"),
                "probe": probe,
            }
        )
    for formula, probe in probe_by_formula.items():
        if formula in seen:
            continue
        rows.append({"formula": formula, "e_hull": None, "template": probe.get("template"), "probe": probe})
    return rows


def formula_status(best_e_hull: float | None, strict_count: int, evaluated_count: int) -> str:
    if strict_count > 0:
        return "confirmed_sun"
    if best_e_hull is not None and best_e_hull < 0.03:
        return "near_miss_non_sun"
    if evaluated_count >= 2:
        return "repeated_non_sun"
    if best_e_hull is not None and best_e_hull >= 0.10:
        return "bad_non_sun"
    return "explored_non_sun"


def formula_recommendation(status: str) -> str:
    if status == "confirmed_sun":
        return "allow_one_anchor_replay_or_mutate_around"
    if status == "near_miss_non_sun":
        return "mutate_around_but_do_not_exactly_repeat"
    if status in {"repeated_non_sun", "bad_non_sun"}:
        return "avoid_exact_repeat"
    return "avoid_exact_repeat_unless_new_hypothesis_requires_it"


def build_formula_memory(history: Sequence[Mapping[str, Any]], max_formulas: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "evaluated_count": 0,
            "strict_sun_count": 0,
            "near_miss_count": 0,
            "lt_0_10_count": 0,
            "best_e_hull": None,
            "latest_e_hull": None,
            "sum_e_hull": 0.0,
            "e_hull_count": 0,
            "first_round": None,
            "latest_round": None,
            "rounds": [],
            "templates": Counter(),
            "families": Counter(),
            "hypothesis_ids": Counter(),
            "sample_probe": None,
        }
    )
    for record in history:
        round_number = safe_int(record.get("round"))
        for row in round_formula_rows(record):
            formula = row["formula"]
            e_hull = row["e_hull"]
            probe = row["probe"] if isinstance(row.get("probe"), Mapping) else {}
            item = stats[formula]
            item["evaluated_count"] += 1
            item["latest_round"] = max(round_number, item["latest_round"] or round_number)
            item["first_round"] = min(round_number, item["first_round"] or round_number)
            item["rounds"].append(round_number)
            template = row.get("template") or probe.get("template")
            if template:
                item["templates"][str(template)] += 1
            family = probe.get("family")
            if family:
                item["families"][str(family)] += 1
            for hyp_id in string_list(probe.get("hypothesis_ids")):
                item["hypothesis_ids"][hyp_id] += 1
            if item["sample_probe"] is None and isinstance(probe, Mapping):
                item["sample_probe"] = compact_probe(probe)
            if e_hull is None:
                continue
            item["latest_e_hull"] = e_hull
            item["sum_e_hull"] += e_hull
            item["e_hull_count"] += 1
            if e_hull < 0:
                item["strict_sun_count"] += 1
            if e_hull < 0.03:
                item["near_miss_count"] += 1
            if e_hull < 0.10:
                item["lt_0_10_count"] += 1
            best = item["best_e_hull"]
            if best is None or e_hull < best:
                item["best_e_hull"] = e_hull

    rows: list[dict[str, Any]] = []
    for formula, item in stats.items():
        best_e_hull = item["best_e_hull"]
        status = formula_status(best_e_hull, item["strict_sun_count"], item["evaluated_count"])
        rows.append(
            {
                "formula": formula,
                "status": status,
                "recommendation": formula_recommendation(status),
                "evaluated_count": item["evaluated_count"],
                "strict_sun_count": item["strict_sun_count"],
                "near_miss_count": item["near_miss_count"],
                "lt_0_10_count": item["lt_0_10_count"],
                "best_e_hull": best_e_hull,
                "mean_e_hull": (
                    item["sum_e_hull"] / item["e_hull_count"] if item["e_hull_count"] else None
                ),
                "latest_e_hull": item["latest_e_hull"],
                "first_round": item["first_round"],
                "latest_round": item["latest_round"],
                "source_rounds": sorted_numeric(item["rounds"]),
                "templates": top_counter(item["templates"]),
                "families": top_counter(item["families"]),
                "hypothesis_ids": top_counter(item["hypothesis_ids"]),
                "sample_probe": item["sample_probe"],
            }
        )

    rows.sort(
        key=lambda item: (
            0 if item["status"] == "confirmed_sun" else 1,
            item["best_e_hull"] if item["best_e_hull"] is not None else 999.0,
            -int(item["evaluated_count"]),
            str(item["formula"]),
        )
    )
    forbidden = sorted(item["formula"] for item in rows if item["strict_sun_count"] == 0)
    confirmed = sorted(item["formula"] for item in rows if item["strict_sun_count"] > 0)
    repeated = sorted(
        item["formula"] for item in rows if item["strict_sun_count"] == 0 and item["evaluated_count"] >= 2
    )
    near_miss = sorted(
        item["formula"]
        for item in rows
        if item["strict_sun_count"] == 0 and item["best_e_hull"] is not None and item["best_e_hull"] < 0.03
    )
    constraints = {
        "confirmed_sun_anchors": confirmed,
        "forbidden_exact_repeat_formulas": forbidden,
        "repeated_non_sun_formulas": repeated,
        "near_miss_non_sun_formulas": near_miss,
        "policy": (
            "Agent C may include at most one exact confirmed_sun anchor replay per round. "
            "All evaluated non-SUN formulas are forbidden as exact repeats; use mutations or new formulas instead."
        ),
    }
    return rows[:max_formulas], constraints


def build_hypothesis_memory(history: Sequence[Mapping[str, Any]], max_hypotheses: int) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for record in history:
        round_number = safe_int(record.get("round"))
        accepted = record.get("accepted_hypotheses")
        if not isinstance(accepted, list):
            continue
        formula_rows = round_formula_rows(record)
        formula_by_hypothesis: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in formula_rows:
            probe = row.get("probe")
            if not isinstance(probe, Mapping):
                continue
            for hyp_id in string_list(probe.get("hypothesis_ids")):
                formula_by_hypothesis[hyp_id].append(row)
        for hypothesis in accepted:
            if not isinstance(hypothesis, Mapping):
                continue
            claim = str(hypothesis.get("claim") or "").strip()
            if not claim:
                continue
            key = slugify(claim)
            hyp_id = str(hypothesis.get("id") or key)
            item = stats.setdefault(
                key,
                {
                    "hypothesis_key": key,
                    "latest_id": hyp_id,
                    "claim": truncate_text(claim, 360),
                    "rationale_summary": truncate_text(hypothesis.get("rationale_summary"), 300),
                    "target_templates": Counter(),
                    "times_seen": 0,
                    "first_round": round_number,
                    "latest_round": round_number,
                    "source_rounds": [],
                    "best_e_hull": None,
                    "strict_sun_hits": 0,
                    "near_miss_non_sun_hits": 0,
                    "formulas": Counter(),
                    "supporting_formulas": set(),
                    "counterexample_formulas": set(),
                },
            )
            item["latest_id"] = hyp_id
            item["times_seen"] += 1
            item["latest_round"] = max(round_number, item["latest_round"])
            item["first_round"] = min(round_number, item["first_round"])
            item["source_rounds"].append(round_number)
            for template in string_list(hypothesis.get("target_templates")):
                item["target_templates"][template] += 1
            for row in formula_by_hypothesis.get(hyp_id, []):
                formula = row["formula"]
                e_hull = row["e_hull"]
                item["formulas"][formula] += 1
                if e_hull is None:
                    continue
                best = item["best_e_hull"]
                if best is None or e_hull < best:
                    item["best_e_hull"] = e_hull
                if e_hull < 0:
                    item["strict_sun_hits"] += 1
                    item["supporting_formulas"].add(formula)
                elif e_hull < 0.03:
                    item["near_miss_non_sun_hits"] += 1
                    item["counterexample_formulas"].add(formula)
                else:
                    item["counterexample_formulas"].add(formula)

    rows: list[dict[str, Any]] = []
    for item in stats.values():
        if item["strict_sun_hits"] > 0:
            status = "supported_by_strict_sun"
        elif item["near_miss_non_sun_hits"] > 0:
            status = "tentative_near_miss_only"
        elif item["times_seen"] >= 3:
            status = "weakened_no_strict_sun"
        else:
            status = "tentative"
        rows.append(
            {
                "hypothesis_key": item["hypothesis_key"],
                "latest_id": item["latest_id"],
                "status": status,
                "claim": item["claim"],
                "rationale_summary": item["rationale_summary"],
                "target_templates": top_counter(item["target_templates"]),
                "times_seen": item["times_seen"],
                "first_round": item["first_round"],
                "latest_round": item["latest_round"],
                "source_rounds": sorted_numeric(item["source_rounds"]),
                "best_e_hull": item["best_e_hull"],
                "strict_sun_hits": item["strict_sun_hits"],
                "near_miss_non_sun_hits": item["near_miss_non_sun_hits"],
                "representative_formulas": top_counter(item["formulas"], limit=10),
                "supporting_formulas": sorted(item["supporting_formulas"])[:10],
                "counterexample_formulas": sorted(item["counterexample_formulas"])[:12],
            }
        )
    rows.sort(
        key=lambda item: (
            0 if item["status"] == "supported_by_strict_sun" else 1,
            item["best_e_hull"] if item["best_e_hull"] is not None else 999.0,
            -int(item["times_seen"]),
            -int(item["latest_round"]),
        )
    )
    return rows[:max_hypotheses]


def compact_round(record: Mapping[str, Any]) -> dict[str, Any]:
    analysis = record.get("analysis_summary")
    if not isinstance(analysis, Mapping):
        analysis = {}
    rows = []
    for row in round_formula_rows(record):
        rows.append(
            {
                "formula": row["formula"],
                "e_hull": row["e_hull"],
                "template": row.get("template"),
                "status": "strict_sun" if row["e_hull"] is not None and row["e_hull"] < 0 else "non_sun",
            }
        )
    rows.sort(key=lambda item: item["e_hull"] if item["e_hull"] is not None else 999.0)
    accepted = []
    for hypothesis in record.get("accepted_hypotheses", []) if isinstance(record.get("accepted_hypotheses"), list) else []:
        if isinstance(hypothesis, Mapping):
            accepted.append(
                {
                    "id": hypothesis.get("id"),
                    "claim": truncate_text(hypothesis.get("claim"), 220),
                    "target_templates": string_list(hypothesis.get("target_templates")),
                }
            )
    return {
        "round": record.get("round"),
        "success": record.get("success"),
        "strict_sun_count": record.get("strict_sun_count"),
        "total_count": record.get("total_count"),
        "strict_sun_rate": record.get("strict_sun_rate"),
        "consensus_summary": truncate_text(record.get("consensus_summary"), 520),
        "accepted_hypotheses": accepted[:8],
        "top_formulas": rows[:10],
        "summary_metrics": {
            "min_e_hull": analysis.get("min_e_hull"),
            "mean_e_hull": analysis.get("mean_e_hull"),
            "e_hull_lt_0_03": analysis.get("e_hull_lt_0_03"),
            "e_hull_lt_0_10": analysis.get("e_hull_lt_0_10"),
            "template_stats": analysis.get("template_stats"),
        },
    }


def build_experience_digest(memory_dir: Path, max_per_polarity: int) -> dict[str, Any]:
    experiences = load_experiences(memory_dir, include_inactive=False)
    digest: dict[str, Any] = {}
    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    for polarity, records in experiences.items():
        sorted_records = sorted(
            records,
            key=lambda record: (
                confidence_rank.get(str(record.get("confidence", "low")).lower(), 3),
                -safe_int(record.get("source_round")),
                str(record.get("id") or ""),
            ),
        )
        digest[polarity] = {
            "active_count": len(records),
            "included_count": min(len(sorted_records), max_per_polarity),
            "records": [
                {
                    "id": record.get("id"),
                    "claim": truncate_text(record.get("claim"), 260),
                    "scope": truncate_text(record.get("scope"), 180),
                    "confidence": record.get("confidence"),
                    "recommended_action": truncate_text(record.get("recommended_action"), 180),
                    "do_not_generalize_to": string_list(record.get("do_not_generalize_to"))[:6],
                    "tags": string_list(record.get("tags"))[:8],
                }
                for record in sorted_records[:max_per_polarity]
            ],
        }
    return digest


def build_compact_memory(
    state: Mapping[str, Any],
    *,
    memory_dir: Path | None = None,
    recent_rounds: int = 3,
    max_formulas: int = 140,
    max_hypotheses: int = 40,
    max_experiences_per_polarity: int = 24,
) -> dict[str, Any]:
    history_raw = state.get("history")
    history = [item for item in history_raw if isinstance(item, Mapping)] if isinstance(history_raw, list) else []
    formula_memory, negative_constraints = build_formula_memory(history, max_formulas=max_formulas)
    hypothesis_memory = build_hypothesis_memory(history, max_hypotheses=max_hypotheses)
    total_count = sum(safe_int(record.get("total_count")) for record in history)
    strict_total = sum(safe_int(record.get("strict_sun_count")) for record in history)
    latest = history[-1] if history else {}
    best_round = state.get("best_round")
    current_round = safe_int(state.get("current_round"))
    rounds_since_best = None
    if best_round is not None:
        rounds_since_best = max(0, current_round - safe_int(best_round))
    recent = [compact_round(record) for record in history[-max(0, recent_rounds) :]]
    unresolved = state.get("unresolved_debates")
    unresolved_list = unresolved if isinstance(unresolved, list) else []
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_state_updated_at_utc": state.get("updated_at_utc"),
        "strict_sun_definition": STRICT_SUN_NOTE,
        "global_summary": {
            "current_round": state.get("current_round"),
            "completed_successful_round_records": len(history),
            "evaluated_structure_count": total_count,
            "strict_sun_total_count": strict_total,
            "overall_strict_sun_rate": strict_total / total_count if total_count else 0.0,
            "latest_round": latest.get("round"),
            "latest_round_strict_sun_count": latest.get("strict_sun_count"),
            "latest_round_total_count": latest.get("total_count"),
            "latest_round_strict_sun_rate": latest.get("strict_sun_rate"),
            "best_round": state.get("best_round"),
            "best_formula": state.get("best_formula"),
            "best_template": state.get("best_template"),
            "best_e_hull": state.get("best_e_hull"),
            "rounds_since_best": rounds_since_best,
            "unique_formula_count_in_memory": len({item["formula"] for item in formula_memory}),
            "recent_unresolved_debate_count": len(unresolved_list),
        },
        "negative_constraints": negative_constraints,
        "formula_memory": formula_memory,
        "hypothesis_memory": hypothesis_memory,
        "recent_rounds": recent,
        "unresolved_debates": unresolved_list[-5:],
        "instructions_for_agents": [
            "Treat formula_memory and negative_constraints as deterministic facts from evaluator results.",
            "Do not reinterpret e_hull >= 0 as SUN.",
            "Agent C must not exactly repeat formulas listed in forbidden_exact_repeat_formulas.",
            "Use near_miss_non_sun formulas only as mutation neighborhoods, not as exact replays.",
        ],
    }
    if memory_dir is not None:
        payload["experience_digest"] = build_experience_digest(
            memory_dir,
            max_per_polarity=max_experiences_per_polarity,
        )
    return payload


def write_compact_memory(
    state_path: Path,
    output_path: Path,
    *,
    memory_dir: Path | None = None,
    recent_rounds: int = 3,
    max_formulas: int = 140,
    max_hypotheses: int = 40,
    max_experiences_per_polarity: int = 24,
) -> dict[str, Any]:
    state = read_json(state_path, {})
    if not isinstance(state, Mapping):
        raise ValueError(f"{state_path} must contain a JSON object")
    payload = build_compact_memory(
        state,
        memory_dir=memory_dir,
        recent_rounds=recent_rounds,
        max_formulas=max_formulas,
        max_hypotheses=max_hypotheses,
        max_experiences_per_polarity=max_experiences_per_polarity,
    )
    write_json(output_path, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact deterministic memory for hypothesis-first MVP.")
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-dir", default=None)
    parser.add_argument("--recent-rounds", type=int, default=3)
    parser.add_argument("--max-formulas", type=int, default=140)
    parser.add_argument("--max-hypotheses", type=int, default=40)
    parser.add_argument("--max-experiences-per-polarity", type=int, default=24)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    state_path = Path(args.state)
    output_path = Path(args.output)
    memory_dir = Path(args.memory_dir) if args.memory_dir else None
    payload = write_compact_memory(
        state_path,
        output_path,
        memory_dir=memory_dir,
        recent_rounds=args.recent_rounds,
        max_formulas=args.max_formulas,
        max_hypotheses=args.max_hypotheses,
        max_experiences_per_polarity=args.max_experiences_per_polarity,
    )
    print(f"wrote {output_path}")
    print(
        "summary="
        f"rounds={payload['global_summary']['completed_successful_round_records']} "
        f"formulas={len(payload['formula_memory'])} "
        f"forbidden={len(payload['negative_constraints']['forbidden_exact_repeat_formulas'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
