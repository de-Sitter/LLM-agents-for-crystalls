"""Merge multiple MatterGen adapter runs into one evaluator input directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from monty.json import MontyEncoder
from pymatgen.core import Structure

from crystal_llm.filters import reduced_formula


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge MatterGen adapter outputs for evaluator benchmarking.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Adapter run directories.")
    parser.add_argument("--output-dir", required=True, help="Directory for merged input.json and metadata.")
    parser.add_argument(
        "--dedupe-reduced-formula",
        action="store_true",
        help="Keep only the first accepted structure for each reduced formula across all runs.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, cls=MontyEncoder) + "\n", encoding="utf-8")


def request_chemical_system(report: Mapping[str, Any]) -> str | None:
    request = report.get("request") if isinstance(report.get("request"), Mapping) else {}
    props = request.get("properties_to_condition_on") if isinstance(request.get("properties_to_condition_on"), Mapping) else {}
    value = props.get("chemical_system") if isinstance(props, Mapping) else None
    return str(value) if value is not None else None


def merge_run(
    run_dir: Path,
    *,
    seen_formulas: set[str],
    dedupe_formula: bool,
    merged_structures: list[dict[str, Any]],
    merged_records: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> None:
    input_path = run_dir / "input.json"
    records_path = run_dir / "selected_records.json"
    report_path = run_dir / "generation_report.json"
    if not input_path.exists():
        raise FileNotFoundError(f"missing {input_path}")
    structures = load_json(input_path)
    records = load_json(records_path) if records_path.exists() else []
    report = load_json(report_path) if report_path.exists() else {}
    if not isinstance(structures, list):
        raise ValueError(f"{input_path} must contain a list")
    if not isinstance(records, list):
        records = []
    if not isinstance(report, dict):
        report = {}

    chemical_system = request_chemical_system(report)
    run_summary = {
        "run_dir": str(run_dir),
        "chemical_system": chemical_system,
        "accepted_count": len(structures),
        "generation_report": str(report_path) if report_path.exists() else None,
        "accepted_formulas": [],
        "kept_count": 0,
        "dropped_duplicate_formula_count": 0,
    }

    for index, structure_dict in enumerate(structures):
        structure = Structure.from_dict(structure_dict)
        formula = reduced_formula(structure)
        run_summary["accepted_formulas"].append(formula)
        if dedupe_formula and formula in seen_formulas:
            run_summary["dropped_duplicate_formula_count"] += 1
            continue
        seen_formulas.add(formula)
        record = dict(records[index]) if index < len(records) and isinstance(records[index], Mapping) else {}
        record.setdefault("formula", formula)
        record.setdefault("material_id", f"mattergen::{run_dir.name}::{index:04d}::{formula}")
        record.setdefault("source", "ml_generator")
        record.setdefault("generator_backend", "mattergen")
        if chemical_system:
            record.setdefault("generator_chemical_system", chemical_system)

        props = dict(structure_dict.get("properties") or {})
        props.update(
            {
                "crystal_llm_source": "ml_generator",
                "crystal_llm_generator_backend": "mattergen",
                "crystal_llm_generator_request_id": record.get("generator_request_id"),
                "crystal_llm_generator_checkpoint": record.get("generator_checkpoint"),
                "crystal_llm_generator_chemical_system": chemical_system,
                "crystal_llm_material_id": record.get("material_id"),
            }
        )
        merged = dict(structure_dict)
        merged["properties"] = props
        merged_structures.append(merged)
        merged_records.append(record)
        run_summary["kept_count"] += 1

    reports.append(run_summary)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    merged_structures: list[dict[str, Any]] = []
    merged_records: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    seen_formulas: set[str] = set()

    for raw_run_dir in args.run_dirs:
        merge_run(
            Path(raw_run_dir),
            seen_formulas=seen_formulas,
            dedupe_formula=bool(args.dedupe_reduced_formula),
            merged_structures=merged_structures,
            merged_records=merged_records,
            reports=reports,
        )

    summary = {
        "run_count": len(args.run_dirs),
        "structure_count": len(merged_structures),
        "unique_reduced_formulas": len({record.get("formula") for record in merged_records}),
        "dedupe_reduced_formula": bool(args.dedupe_reduced_formula),
        "runs": reports,
    }
    write_json(output_dir / "input.json", merged_structures)
    write_json(output_dir / "selected_records.json", merged_records)
    write_json(output_dir / "benchmark_generation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if merged_structures else 1


if __name__ == "__main__":
    raise SystemExit(main())
