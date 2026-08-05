"""Build a reusable candidate-pool index from a Materials Project directory.

The MP dump used in this project is laid out as one directory per material:

    mp-123/
      structure.cif
      properties.json

Scanning that tree directly during optimization is too slow on a shared
filesystem. This command builds compact JSONL/Parquet indexes once, and writes
a filtered candidate pool that later workflows can sample without touching
154k small directories again.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import math
import os
from pathlib import Path
import pickle
import random
from typing import Any, Iterable, Sequence

import orjson
from pymatgen.core import Composition, Structure
import pyarrow as pa
import pyarrow.parquet as pq

from crystal_llm.filters import load_known_formulas, validate_structure


DEFAULT_FIELDS = (
    "material_id",
    "formula",
    "elements",
    "nelements",
    "nsites",
    "band_gap",
    "is_stable",
    "formation_energy_per_atom",
    "energy_per_atom",
    "volume",
    "density",
    "crystal_system",
    "spacegroup_symbol",
    "spacegroup_number",
    "cif_path",
    "properties_path",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MP JSONL/Parquet candidate-pool indexes.")
    parser.add_argument("--mp-dir", default=str(Path.home() / "mp"), help="Directory containing mp-* subdirectories.")
    parser.add_argument("--output-dir", default="data/mp_candidate_pool", help="Directory for index outputs.")
    parser.add_argument(
        "--training-data",
        default="archive/matllmsearch_evaluator/data/a_training.json",
        help="Evaluator training data used to exclude known reduced formulas.",
    )
    parser.add_argument(
        "--ppd-path",
        default="archive/matllmsearch_evaluator/data/2024-08-07-ppd-mp.pkl",
        help="Patched phase diagram pickle; used only to load covered elements.",
    )
    parser.add_argument("--jobs", type=int, default=16, help="Parallel workers for reading material directories.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Number of directories per worker task.")
    parser.add_argument("--seed", type=int, default=20260507, help="Random seed for sampled files.")
    parser.add_argument("--sample-size", type=int, default=10000, help="Write a random sample of filtered candidates.")
    parser.add_argument("--max-sites", type=int, default=80, help="Filtered pool maximum number of sites.")
    parser.add_argument("--max-elements", type=int, default=6, help="Filtered pool maximum number of elements.")
    parser.add_argument("--min-band-gap", type=float, default=None, help="Optional minimum MP band gap for filtered pool.")
    parser.add_argument("--max-band-gap", type=float, default=None, help="Optional maximum MP band gap for filtered pool.")
    parser.add_argument("--require-cif-parse", action="store_true", help="Parse CIFs and run local structure validation.")
    parser.add_argument("--no-parquet", action="store_true", help="Skip Parquet output.")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit on number of mp directories.")
    return parser.parse_args(argv)


def chunked(items: Sequence[Path], size: int) -> Iterable[list[Path]]:
    for index in range(0, len(items), size):
        yield list(items[index : index + size])


def reduced_formula_from_composition(value: Any) -> str | None:
    if not isinstance(value, dict) or not value:
        return None
    try:
        return Composition({str(k): float(v) for k, v in value.items()}).reduced_formula
    except Exception:
        return None


def finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def load_ppd_elements(ppd_path: Path) -> set[str]:
    if not ppd_path.exists():
        return set()
    with ppd_path.open("rb") as probe:
        magic = probe.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(ppd_path, "rb") as handle:
        ppd = pickle.load(handle)
    elements = getattr(ppd, "elements", []) or []
    return {str(element.symbol) for element in elements if getattr(element, "symbol", None)}


def read_one_material(path: Path, require_cif_parse: bool, max_sites: int) -> tuple[dict[str, Any] | None, str | None]:
    properties_path = path / "properties.json"
    cif_path = path / "structure.cif"
    if not properties_path.exists():
        return None, "missing_properties"
    if not cif_path.exists():
        return None, "missing_cif"

    try:
        raw = orjson.loads(properties_path.read_bytes())
    except Exception:
        return None, "bad_properties_json"
    if not isinstance(raw, dict):
        return None, "properties_not_object"

    formula = reduced_formula_from_composition(raw.get("composition_reduced")) or reduced_formula_from_composition(
        raw.get("composition")
    )
    if not formula:
        return None, "missing_formula"

    sym = raw.get("symmetry") if isinstance(raw.get("symmetry"), dict) else {}
    elements = raw.get("elements") if isinstance(raw.get("elements"), list) else []
    record: dict[str, Any] = {
        "material_id": str(raw.get("material_id") or path.name),
        "formula": formula,
        "elements": [str(item) for item in elements],
        "nelements": int(raw["nelements"]) if isinstance(raw.get("nelements"), int) else len(elements),
        "nsites": int(raw["nsites"]) if isinstance(raw.get("nsites"), int) else None,
        "band_gap": finite_float(raw.get("band_gap")),
        "is_stable": raw.get("is_stable") if isinstance(raw.get("is_stable"), bool) else None,
        "formation_energy_per_atom": finite_float(raw.get("formation_energy_per_atom")),
        "energy_per_atom": finite_float(raw.get("energy_per_atom")),
        "volume": finite_float(raw.get("volume")),
        "density": finite_float(raw.get("density")),
        "crystal_system": sym.get("crystal_system"),
        "spacegroup_symbol": sym.get("symbol"),
        "spacegroup_number": int(sym["number"]) if isinstance(sym.get("number"), int) else None,
        "cif_path": str(cif_path),
        "properties_path": str(properties_path),
    }

    if require_cif_parse:
        try:
            structure = Structure.from_file(str(cif_path))
        except Exception:
            return None, "bad_cif"
        validation = validate_structure(structure, max_sites=max_sites)
        record["local_structure_valid"] = validation.ok
        record["validation_reasons"] = validation.reasons
        record["volume_per_atom"] = validation.volume_per_atom
        record["min_distance"] = validation.min_distance if math.isfinite(validation.min_distance) else None

    return record, None


def read_chunk(paths: list[str], require_cif_parse: bool, max_sites: int) -> tuple[list[dict[str, Any]], Counter]:
    records: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for item in paths:
        record, reason = read_one_material(Path(item), require_cif_parse, max_sites)
        if record is None:
            skipped[reason or "unknown"] += 1
        else:
            records.append(record)
    return records, skipped


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
            count += 1
    return count


def write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, compression="zstd")


def should_keep(
    record: dict[str, Any],
    *,
    training_formulas: set[str],
    ppd_elements: set[str],
    max_sites: int,
    max_elements: int,
    min_band_gap: float | None,
    max_band_gap: float | None,
    require_cif_parse: bool,
) -> tuple[bool, str | None]:
    formula = str(record.get("formula") or "")
    if formula in training_formulas:
        return False, "known_training_formula"
    nsites = record.get("nsites")
    if not isinstance(nsites, int) or nsites <= 0:
        return False, "bad_nsites"
    if nsites > max_sites:
        return False, "too_many_sites"
    nelements = record.get("nelements")
    if not isinstance(nelements, int) or nelements <= 0:
        return False, "bad_nelements"
    if nelements > max_elements:
        return False, "too_many_elements"
    elements = set(str(item) for item in (record.get("elements") or []))
    if ppd_elements and not elements.issubset(ppd_elements):
        return False, "outside_ppd_elements"
    band_gap = record.get("band_gap")
    if min_band_gap is not None and (not isinstance(band_gap, (int, float)) or float(band_gap) < min_band_gap):
        return False, "band_gap_below_min"
    if max_band_gap is not None and (not isinstance(band_gap, (int, float)) or float(band_gap) > max_band_gap):
        return False, "band_gap_above_max"
    if require_cif_parse and record.get("local_structure_valid") is not True:
        return False, "local_structure_invalid"
    return True, None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mp_dir = Path(args.mp_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mp_dir.exists():
        raise FileNotFoundError(f"MP directory not found: {mp_dir}")

    paths = [entry.path for entry in os.scandir(mp_dir) if entry.is_dir(follow_symlinks=False)]
    if args.limit > 0:
        paths = paths[: args.limit]
    paths = sorted(paths)

    training_formulas = load_known_formulas(args.training_data)
    ppd_elements = load_ppd_elements(Path(args.ppd_path)) if args.ppd_path else set()

    all_records: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    chunks = list(chunked([Path(item) for item in paths], max(1, args.chunk_size)))

    print(f"mp_dir={mp_dir}")
    print(f"material_dirs={len(paths)} chunks={len(chunks)} jobs={args.jobs}")
    print(f"training_formulas={len(training_formulas)} ppd_elements={len(ppd_elements)}")

    if args.jobs <= 1:
        for index, chunk in enumerate(chunks, 1):
            records, chunk_skipped = read_chunk([str(item) for item in chunk], args.require_cif_parse, args.max_sites)
            all_records.extend(records)
            skipped.update(chunk_skipped)
            if index % 10 == 0 or index == len(chunks):
                print(f"indexed_chunks={index}/{len(chunks)} records={len(all_records)} skipped={sum(skipped.values())}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(read_chunk, [str(item) for item in chunk], args.require_cif_parse, args.max_sites)
                for chunk in chunks
            ]
            for index, future in enumerate(as_completed(futures), 1):
                records, chunk_skipped = future.result()
                all_records.extend(records)
                skipped.update(chunk_skipped)
                if index % 10 == 0 or index == len(futures):
                    print(f"indexed_chunks={index}/{len(futures)} records={len(all_records)} skipped={sum(skipped.values())}", flush=True)

    all_records.sort(key=lambda item: item.get("material_id") or "")

    filter_reasons: Counter = Counter()
    filtered_records: list[dict[str, Any]] = []
    for record in all_records:
        keep, reason = should_keep(
            record,
            training_formulas=training_formulas,
            ppd_elements=ppd_elements,
            max_sites=args.max_sites,
            max_elements=args.max_elements,
            min_band_gap=args.min_band_gap,
            max_band_gap=args.max_band_gap,
            require_cif_parse=args.require_cif_parse,
        )
        if keep:
            filtered_records.append(record)
        else:
            filter_reasons[reason or "unknown"] += 1

    rng = random.Random(args.seed)
    sample_records = list(filtered_records)
    rng.shuffle(sample_records)
    if args.sample_size > 0:
        sample_records = sample_records[: args.sample_size]

    all_jsonl = output_dir / "mp_index.jsonl"
    filtered_jsonl = output_dir / "mp_candidates_filtered.jsonl"
    sample_jsonl = output_dir / "mp_candidates_sample.jsonl"
    write_jsonl(all_jsonl, all_records)
    write_jsonl(filtered_jsonl, filtered_records)
    write_jsonl(sample_jsonl, sample_records)

    if not args.no_parquet:
        write_parquet(output_dir / "mp_index.parquet", all_records)
        write_parquet(output_dir / "mp_candidates_filtered.parquet", filtered_records)
        write_parquet(output_dir / "mp_candidates_sample.parquet", sample_records)

    unique_formulas = {record["formula"] for record in all_records}
    filtered_unique_formulas = {record["formula"] for record in filtered_records}
    band_gap_bins = Counter()
    stable_counter = Counter()
    crystal_systems = Counter()
    for record in filtered_records:
        band_gap = record.get("band_gap")
        if isinstance(band_gap, (int, float)):
            value = float(band_gap)
            if value == 0:
                band_gap_bins["0"] += 1
            elif value < 0.5:
                band_gap_bins["0-0.5"] += 1
            elif value <= 1.0:
                band_gap_bins["0.5-1.0"] += 1
            elif value <= 2.0:
                band_gap_bins["1.0-2.0"] += 1
            else:
                band_gap_bins[">2.0"] += 1
        stable_counter[str(record.get("is_stable"))] += 1
        crystal_systems[str(record.get("crystal_system"))] += 1

    summary = {
        "mp_dir": str(mp_dir),
        "output_dir": str(output_dir),
        "input_material_dirs": len(paths),
        "indexed_records": len(all_records),
        "unique_formulas": len(unique_formulas),
        "filtered_records": len(filtered_records),
        "filtered_unique_formulas": len(filtered_unique_formulas),
        "sample_records": len(sample_records),
        "skipped": dict(skipped),
        "filter_reasons": dict(filter_reasons),
        "training_formulas_excluded": len(training_formulas),
        "ppd_elements": sorted(ppd_elements),
        "settings": {
            "max_sites": args.max_sites,
            "max_elements": args.max_elements,
            "min_band_gap": args.min_band_gap,
            "max_band_gap": args.max_band_gap,
            "require_cif_parse": args.require_cif_parse,
            "sample_size": args.sample_size,
            "seed": args.seed,
        },
        "filtered_band_gap_bins": dict(band_gap_bins),
        "filtered_is_stable_counts": dict(stable_counter),
        "filtered_crystal_systems": dict(crystal_systems),
        "outputs": {
            "index_jsonl": str(all_jsonl),
            "filtered_jsonl": str(filtered_jsonl),
            "sample_jsonl": str(sample_jsonl),
        },
        "schema_fields": DEFAULT_FIELDS,
    }
    write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
