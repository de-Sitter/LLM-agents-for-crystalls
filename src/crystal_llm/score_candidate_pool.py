"""Score MP candidate-pool records with the restricted scorer DSL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import orjson
from pymatgen.core import Structure

from crystal_llm.scorer_dsl import (
    diagnostic_distribution,
    score_records,
    validate_scorer_behavior,
    validate_scorer_payload,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank and sample MP candidate-pool records.")
    parser.add_argument("--candidate-pool", default="data/mp_candidate_pool/mp_candidates_filtered.jsonl")
    parser.add_argument("--scorer", required=True, help="Scorer DSL JSON.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--random-k", type=int, default=50)
    parser.add_argument("--bottom-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--validate-sample-size", type=int, default=20000)
    parser.add_argument("--max-records", type=int, default=0, help="Debug limit for candidate pool records.")
    return parser.parse_args(argv)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: Path, max_records: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = orjson.loads(line)
            if isinstance(item, dict):
                records.append(item)
                if max_records > 0 and len(records) >= max_records:
                    break
    return records


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for record in records:
            handle.write(orjson.dumps(dict(record), option=orjson.OPT_APPEND_NEWLINE))


def as_evaluator_input(records: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    structures = []
    for record in records:
        cif_path = Path(str(record.get("cif_path") or ""))
        structure = Structure.from_file(str(cif_path))
        structure.properties["crystal_llm_candidate_source"] = "mp_candidate_pool"
        structure.properties["crystal_llm_material_id"] = record.get("material_id")
        structure.properties["crystal_llm_pool_formula"] = record.get("formula")
        structure.properties["crystal_llm_pool_score"] = record.get("score")
        structure.properties["crystal_llm_pool_bucket"] = record.get("bucket")
        structures.append(structure.as_dict())
    write_json(output_path, structures)


def select_records(scored: list[dict[str, Any]], *, top_k: int, random_k: int, bottom_k: int, seed: int) -> list[dict[str, Any]]:
    sorted_desc = sorted(scored, key=lambda item: (float(item["score"]), str(item.get("material_id"))), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_bucket(records: Sequence[dict[str, Any]], bucket: str, limit: int) -> None:
        count = 0
        for record in records:
            material_id = str(record.get("material_id"))
            if material_id in seen:
                continue
            item = dict(record)
            item["bucket"] = bucket
            selected.append(item)
            seen.add(material_id)
            count += 1
            if count >= limit:
                break

    add_bucket(sorted_desc, "top", max(0, top_k))
    rng = random.Random(seed)
    random_pool = [record for record in scored if str(record.get("material_id")) not in seen]
    rng.shuffle(random_pool)
    add_bucket(random_pool, "random", max(0, random_k))
    bottom_pool = list(reversed(sorted_desc))
    add_bucket(bottom_pool, "bottom", max(0, bottom_k))
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scorer = read_json(Path(args.scorer))
    if not isinstance(scorer, Mapping):
        raise ValueError("scorer JSON must be an object")

    schema_errors = validate_scorer_payload(scorer)
    if schema_errors:
        raise ValueError("invalid scorer DSL: " + "; ".join(schema_errors))

    records = read_jsonl(Path(args.candidate_pool), args.max_records)
    validation_records = records
    if args.validate_sample_size > 0 and len(records) > args.validate_sample_size:
        validation_records = random.Random(args.seed).sample(records, args.validate_sample_size)
    behavior_errors = validate_scorer_behavior(scorer, validation_records)
    if behavior_errors:
        raise ValueError("invalid scorer behavior: " + "; ".join(behavior_errors))

    scores = score_records(scorer, records)
    scored: list[dict[str, Any]] = []
    for record, score in zip(records, scores):
        item = dict(record)
        item["score"] = score
        scored.append(item)

    selected = select_records(
        scored,
        top_k=args.top_k,
        random_k=args.random_k,
        bottom_k=args.bottom_k,
        seed=args.seed,
    )

    ranked = sorted(scored, key=lambda item: (float(item["score"]), str(item.get("material_id"))), reverse=True)
    write_jsonl(output_dir / "ranked_all.jsonl", ranked)
    write_jsonl(output_dir / "selected_records.jsonl", selected)
    as_evaluator_input(selected, output_dir / "input.json")

    bucket_counts: dict[str, int] = {}
    for record in selected:
        bucket = str(record.get("bucket"))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    summary = {
        "candidate_pool": str(Path(args.candidate_pool).resolve()),
        "scorer": str(Path(args.scorer).resolve()),
        "records_scored": len(scored),
        "selected_records": len(selected),
        "bucket_counts": bucket_counts,
        "score_distribution": diagnostic_distribution(scorer, records),
        "validation_distribution": diagnostic_distribution(scorer, validation_records),
        "outputs": {
            "ranked_all": str((output_dir / "ranked_all.jsonl").resolve()),
            "selected_records": str((output_dir / "selected_records.jsonl").resolve()),
            "input_json": str((output_dir / "input.json").resolve()),
        },
    }
    write_json(output_dir / "score_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
