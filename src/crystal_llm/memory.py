"""Natural-language memory and round-summary utilities.

This module is the bridge between evaluator outputs and the higher-level
planning layer.  It deliberately stores richer observations than the executable
generator can consume.  The generator still receives a compact strategy JSON,
but the planner can read these memory records before deciding what to compile
into that strategy.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from pymatgen.core import Structure


STRICT_SUN_NOTE = "SUN = e_hull < 0 only; e_hull < 0.03 and < 0.10 are diagnostics."


def round_label(round_number: int) -> str:
    return f"round_{round_number:03d}"


def round_dir_path(root: Path, round_number: int) -> Path:
    return root / "rounds" / round_label(round_number)


def round_input_path(root: Path, round_number: int) -> Path:
    candidate = round_dir_path(root, round_number) / "input.json"
    if round_number == 1 and not candidate.exists():
        return root / "input.json"
    return candidate


def round_analysis_dir(root: Path, round_number: int) -> Path:
    candidate = round_dir_path(root, round_number) / "analysis"
    if round_number == 1 and not candidate.exists():
        return root / "analysis"
    return candidate


def round_ranked_path(root: Path, round_number: int) -> Path:
    return round_analysis_dir(root, round_number) / "e_hull_ranked.csv"


def round_report_path(root: Path, round_number: int) -> Path:
    round_dir = round_dir_path(root, round_number)
    for name in ("input.report.json", "llm_review.report.json", "candidate_pool.report.json"):
        candidate = round_dir / name
        if candidate.exists():
            return candidate
    if round_number == 1:
        return root / "input.report.json"
    return round_dir / "input.report.json"


def source_for_round(round_number: int) -> dict[str, str]:
    label = round_label(round_number)
    return {
        "round": label,
        "input": f"rounds/{label}/input.json",
        "ranked_e_hull": f"rounds/{label}/analysis/e_hull_ranked.csv",
    }


def sources_through_round(round_number: int) -> list[dict[str, str]]:
    return [source_for_round(value) for value in range(1, round_number + 1)]


def infer_completed_rounds(root: Path) -> list[int]:
    rounds: list[int] = []
    if round_ranked_path(root, 1).exists() and round_input_path(root, 1).exists():
        rounds.append(1)
    rounds_root = root / "rounds"
    if rounds_root.exists():
        for path in sorted(rounds_root.glob("round_*")):
            try:
                round_number = int(path.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if round_ranked_path(root, round_number).exists() and round_input_path(root, round_number).exists():
                rounds.append(round_number)
    return sorted(set(rounds))


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if isinstance(item, dict):
                records.append(item)
    return records


def upsert_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    merged: dict[str, dict[str, Any]] = {}
    for item in load_jsonl(path):
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            merged[item_id] = item
    for record in records:
        item = dict(record)
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            merged[item_id] = item
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key in sorted(merged):
            json.dump(merged[key], handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def read_ranked_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            rows[int(row["index"])] = row
        return rows


def load_round_records(root: Path, round_number: int) -> list[dict[str, Any]]:
    input_path = round_input_path(root, round_number)
    ranked_path = round_ranked_path(root, round_number)
    structures_raw = read_json(input_path, [])
    if not isinstance(structures_raw, list):
        raise ValueError(f"{input_path} must contain a JSON list")

    ranked = read_ranked_rows(ranked_path)
    records: list[dict[str, Any]] = []
    for index, structure_raw in enumerate(structures_raw, start=1):
        row = ranked.get(index)
        if row is None:
            raise ValueError(f"missing e_hull row for index {index} in {ranked_path}")
        structure = Structure.from_dict(structure_raw)
        metadata = structure_raw.get("properties", {}).get("crystal_llm_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        e_hull = float(row["e_hull"])
        records.append(
            {
                "index": index,
                "formula": structure.composition.reduced_formula,
                "template": row.get("template_guess") or structure_raw.get("properties", {}).get("crystal_llm_template"),
                "e_hull": e_hull,
                "nsites": len(structure),
                "volume_per_atom": float(row.get("volume_per_atom", structure.volume / len(structure))),
                "elite_replay": metadata.get("elite_replay") == "true",
                "elite_source_round": metadata.get("elite_source_round"),
                "formula_probe": metadata.get("formula_probe") == "true",
                "formula_probe_id": metadata.get("formula_probe_id"),
                "formula_probe_family": metadata.get("formula_probe_family"),
            }
        )
    return records


def slim_record(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "index",
        "formula",
        "template",
        "e_hull",
        "elite_replay",
        "elite_source_round",
        "formula_probe",
        "formula_probe_id",
        "formula_probe_family",
    )
    return {key: record.get(key) for key in keys}


def count_if(records: Sequence[Mapping[str, Any]], threshold: float) -> int:
    return sum(float(record["e_hull"]) < threshold for record in records)


def build_round_summary(root: Path, round_number: int) -> dict[str, Any]:
    records = load_round_records(root, round_number)
    values = [float(record["e_hull"]) for record in records]
    elite = [record for record in records if record["elite_replay"]]
    probes = [record for record in records if record["formula_probe"]]
    non_elite = [record for record in records if not record["elite_replay"]]
    ordinary = [record for record in records if not record["elite_replay"] and not record["formula_probe"]]
    report = read_json(round_report_path(root, round_number), {})
    if not isinstance(report, dict):
        report = {}

    summary = {
        "round": round_number,
        "round_label": round_label(round_number),
        "strict_sun_definition": STRICT_SUN_NOTE,
        "source_files": {
            "input": str(round_input_path(root, round_number).relative_to(root)),
            "ranked_e_hull": str(round_ranked_path(root, round_number).relative_to(root)),
            "generation_report": str(round_report_path(root, round_number).relative_to(root)),
        },
        "metrics": {
            "total_structures": len(records),
            "strict_sun": count_if(records, 0.0),
            "e_hull_lt_0_03": count_if(records, 0.03),
            "e_hull_lt_0_10": count_if(records, 0.10),
            "mean_e_hull": mean(values) if values else None,
            "min_e_hull": min(values) if values else None,
            "max_e_hull": max(values) if values else None,
        },
        "generation_report": {
            "generated_count": report.get("generated_count"),
            "unique_reduced_formulas": report.get("unique_reduced_formulas"),
            "template_counts": report.get("template_counts"),
            "elite_replay_selected": report.get("elite_replay_selected"),
            "formula_probe_selected": report.get("formula_probe_selected"),
            "reject_reasons": report.get("reject_reasons"),
        },
        "source_split": {
            "elite_total": len(elite),
            "elite_strict_sun": count_if(elite, 0.0),
            "probe_total": len(probes),
            "probe_strict_sun": count_if(probes, 0.0),
            "non_elite_total": len(non_elite),
            "non_elite_strict_sun": count_if(non_elite, 0.0),
            "ordinary_non_probe_total": len(ordinary),
            "ordinary_non_probe_strict_sun": count_if(ordinary, 0.0),
        },
        "elite_failed": [slim_record(record) for record in sorted(elite, key=lambda item: float(item["e_hull"])) if float(record["e_hull"]) >= 0.0],
        "new_strict_sun": [slim_record(record) for record in sorted(non_elite, key=lambda item: float(item["e_hull"])) if float(record["e_hull"]) < 0.0],
        "formula_probes": [slim_record(record) for record in sorted(probes, key=lambda item: float(item["e_hull"]))],
        "ordinary_near_zero_top20": [
            slim_record(record)
            for record in sorted(ordinary, key=lambda item: float(item["e_hull"]))[:20]
            if float(record["e_hull"]) < 0.03
        ],
    }
    return summary


def round_summary_markdown(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]
    split = summary["source_split"]
    lines = [
        f"# {summary['round_label']} Memory Summary",
        "",
        f"- Strict SUN definition: `{summary['strict_sun_definition']}`",
        f"- Total structures: `{metrics['total_structures']}`",
        f"- Strict SUN: `{metrics['strict_sun']}`",
        f"- e_hull < 0.03: `{metrics['e_hull_lt_0_03']}`",
        f"- e_hull < 0.10: `{metrics['e_hull_lt_0_10']}`",
        f"- Mean e_hull: `{metrics['mean_e_hull']}`",
        "",
        "## Source Split",
        "",
        "| Source | Total | Strict SUN |",
        "|---|---:|---:|",
        f"| elite replay | {split['elite_total']} | {split['elite_strict_sun']} |",
        f"| formula probes | {split['probe_total']} | {split['probe_strict_sun']} |",
        f"| non-elite total | {split['non_elite_total']} | {split['non_elite_strict_sun']} |",
        f"| ordinary non-probe | {split['ordinary_non_probe_total']} | {split['ordinary_non_probe_strict_sun']} |",
        "",
        "## New Strict SUN",
        "",
    ]
    new_hits = summary.get("new_strict_sun", [])
    if new_hits:
        lines.extend(["| Formula | Template | e_hull | Source |", "|---|---|---:|---|"])
        for item in new_hits:
            source = item.get("formula_probe_id") or "ordinary"
            lines.append(f"| `{item['formula']}` | `{item['template']}` | {float(item['e_hull']):.6f} | `{source}` |")
    else:
        lines.append("No non-elite strict SUN in this round.")

    lines.extend(["", "## Formula Probes", ""])
    probes = summary.get("formula_probes", [])
    if probes:
        lines.extend(["| Formula | e_hull | Verdict | Probe id |", "|---|---:|---|---|"])
        for item in probes:
            verdict = "strict_sun" if float(item["e_hull"]) < 0.0 else "positive"
            lines.append(f"| `{item['formula']}` | {float(item['e_hull']):.6f} | `{verdict}` | `{item.get('formula_probe_id')}` |")
    else:
        lines.append("No formula probes were selected.")

    lines.extend(["", "## Elite Replay Failures", ""])
    failures = summary.get("elite_failed", [])
    if failures:
        lines.extend(["| Formula | e_hull | Source round |", "|---|---:|---|"])
        for item in failures:
            lines.append(f"| `{item['formula']}` | {float(item['e_hull']):.6f} | `{item.get('elite_source_round')}` |")
    else:
        lines.append("No elite replay failures.")

    lines.append("")
    return "\n".join(lines)


def _formula_contains(formula: str, symbols: Iterable[str]) -> bool:
    return all(symbol in formula for symbol in symbols)


def memory_records_from_summary(summary: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    round_number = int(summary["round"])
    label = str(summary["round_label"])
    metrics = summary["metrics"]
    split = summary["source_split"]
    observations: list[dict[str, Any]] = [
        {
            "id": f"{label}_metrics",
            "type": "round_metrics",
            "round": round_number,
            "status": "observed",
            "natural_language": (
                f"{label} produced {metrics['strict_sun']} strict SUN out of "
                f"{metrics['total_structures']} structures. {STRICT_SUN_NOTE}"
            ),
            "metrics": metrics,
            "source_split": split,
        }
    ]
    hypotheses: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []

    new_hits = list(summary.get("new_strict_sun", []))
    if new_hits:
        observations.append(
            {
                "id": f"{label}_new_strict_sun",
                "type": "new_strict_sun",
                "round": round_number,
                "status": "observed",
                "natural_language": f"{label} found {len(new_hits)} non-elite strict SUN structures.",
                "examples": new_hits,
            }
        )

    probe_hits = [item for item in summary.get("formula_probes", []) if float(item["e_hull"]) < 0.0]
    probe_failures = [item for item in summary.get("formula_probes", []) if float(item["e_hull"]) >= 0.0]
    if probe_hits:
        observations.append(
            {
                "id": f"{label}_probe_strict_hits",
                "type": "probe_strict_hits",
                "round": round_number,
                "status": "observed",
                "natural_language": f"{label} formula probes produced {len(probe_hits)} strict SUN hits.",
                "examples": probe_hits,
            }
        )

    li_br_hits = [item for item in probe_hits if _formula_contains(str(item["formula"]), ("Li", "Br"))]
    if li_br_hits:
        br_counterexamples = [
            item
            for item in probe_failures
            if _formula_contains(str(item["formula"]), ("Li",)) and not _formula_contains(str(item["formula"]), ("Br",))
        ]
        hypotheses.append(
            {
                "id": f"{label}_li_halide_br_exact_validated",
                "type": "hypothesis",
                "round": round_number,
                "status": "supported",
                "confidence": "high",
                "natural_language": (
                    "Li-halide Br exact double-perovskite probes produced strict SUN hits. "
                    "Treat this as an exact formula-family signal, not a broad Li/Br/Ni/Co/V boost."
                ),
                "scope": "Exact A2LiMBr6 double_perovskite probes; do not generalize globally.",
                "positive_examples": li_br_hits,
                "counterexamples": br_counterexamples,
                "do_not_generalize_to": ["global Li boost", "global Br boost", "F analogs", "VCl/VF branches"],
            }
        )

    f_failures = [item for item in probe_failures if _formula_contains(str(item["formula"]), ("Li", "F"))]
    if f_failures:
        counterexamples.append(
            {
                "id": f"{label}_li_halide_f_branch_rejected",
                "type": "counterexample",
                "round": round_number,
                "status": "rejected",
                "natural_language": (
                    "Li-halide F analog probes remained positive. Near-zero F outcomes must not be treated as strict SUN."
                ),
                "examples": f_failures,
                "rejects_hypothesis": "Li-halide success generalizes from Cl/Br to F.",
            }
        )

    v_failures = [item for item in probe_failures if "V" in str(item["formula"]) and _formula_contains(str(item["formula"]), ("Li",))]
    if v_failures:
        counterexamples.append(
            {
                "id": f"{label}_li_halide_v_branch_limited",
                "type": "counterexample",
                "round": round_number,
                "status": "role_limited",
                "natural_language": (
                    "V in Li-halide probes remains exact-formula limited. Failed V probes argue against broad V expansion."
                ),
                "examples": v_failures,
                "keep_exact_replay_only": ["Cs2LiVBr6"],
            }
        )

    for failure in summary.get("elite_failed", []):
        formula = str(failure["formula"])
        counterexamples.append(
            {
                "id": f"{label}_elite_replay_failure_{formula}",
                "type": "counterexample",
                "round": round_number,
                "status": "elite_replay_failure",
                "natural_language": f"{formula} failed elite replay in {label} with e_hull={float(failure['e_hull']):.6f}.",
                "formula": formula,
                "example": failure,
                "recommended_action": "quarantine_or_exact_exclude_if_repeated",
            }
        )

    near_zero_positive_probes = [item for item in probe_failures if float(item["e_hull"]) < 0.03]
    if near_zero_positive_probes and not probe_hits:
        counterexamples.append(
            {
                "id": f"{label}_near_zero_probe_not_sun",
                "type": "counterexample",
                "round": round_number,
                "status": "metric_guard",
                "natural_language": (
                    "Formula probes can be e_hull < 0.03 while still not being strict SUN. "
                    "Do not promote near-zero positives."
                ),
                "examples": near_zero_positive_probes,
                "strict_sun_definition": STRICT_SUN_NOTE,
            }
        )

    return observations, hypotheses, counterexamples


def write_round_memory(root: Path, memory_dir: Path, round_number: int) -> dict[str, Any]:
    summary = build_round_summary(root, round_number)
    summary_dir = memory_dir / "round_summaries"
    write_json(summary_dir / f"{round_label(round_number)}.json", summary)
    markdown_path = summary_dir / f"{round_label(round_number)}.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(round_summary_markdown(summary), encoding="utf-8")

    observations, hypotheses, counterexamples = memory_records_from_summary(summary)
    upsert_jsonl(memory_dir / "observations.jsonl", observations)
    upsert_jsonl(memory_dir / "hypotheses.jsonl", hypotheses)
    upsert_jsonl(memory_dir / "counterexamples.jsonl", counterexamples)
    return summary
