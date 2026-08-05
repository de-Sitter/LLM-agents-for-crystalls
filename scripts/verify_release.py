#!/usr/bin/env python3
"""Verify alignment of the frozen structures, per-material energies, and metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from pymatgen.core import Structure


ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_FILE = ROOT / "CHECKSUMS.sha256"
IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "generated",
    "runs",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def release_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != CHECKSUM_FILE
        and not any(part in IGNORED_PARTS for part in path.relative_to(ROOT).parts)
        and path.suffix not in {".pyc", ".pyo"}
    )


def write_checksums() -> None:
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in release_files()]
    CHECKSUM_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_checksums() -> int:
    if not CHECKSUM_FILE.exists():
        return 0
    recorded: dict[str, str] = {}
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        recorded[relative] = expected
    actual_paths = {path.relative_to(ROOT).as_posix(): path for path in release_files()}
    if set(recorded) != set(actual_paths):
        raise AssertionError("CHECKSUMS.sha256 file list differs from the release file set")
    mismatches = [relative for relative, path in actual_paths.items() if digest(path) != recorded[relative]]
    if mismatches:
        raise AssertionError(f"checksum mismatch: {mismatches[:5]}")
    return len(recorded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-checksums", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_checksums:
        write_checksums()

    structures_raw = json.loads((ROOT / "input.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((ROOT / "artifacts/metrics_summary.json").read_text(encoding="utf-8"))
    with (ROOT / "artifacts/final_materials.csv").open(encoding="utf-8", newline="") as handle:
        material_rows = list(csv.DictReader(handle))

    if not (len(structures_raw) == len(material_rows) == 1000):
        raise AssertionError("input.json and final_materials.csv must both contain 1000 aligned records")

    structures = [Structure.from_dict(item) for item in structures_raw]
    formulas = [structure.composition.reduced_formula for structure in structures]
    csv_formulas = [row["reduced_formula"] for row in material_rows]
    if formulas != csv_formulas:
        raise AssertionError("formula order differs between input.json and final_materials.csv")

    energies = [float(row["e_hull_eV_per_atom"]) for row in material_rows]
    sun = [row["in_sample_sun"].lower() == "true" for row in material_rows]
    if sun != [energy < 0 for energy in energies]:
        raise AssertionError("S.U.N. labels do not match the strict E_hull < 0 rule")

    bins = {
        "e_hull_lt_0": sum(energy < 0 for energy in energies),
        "e_hull_0_to_0.03": sum(0 <= energy < 0.03 for energy in energies),
        "e_hull_0.03_to_0.10": sum(0.03 <= energy < 0.10 for energy in energies),
        "e_hull_ge_0.10": sum(energy >= 0.10 for energy in energies),
    }
    if bins != metrics["stability_bins"]:
        raise AssertionError("energy bins differ from metrics_summary.json")
    if results["novelty"]["sun_score"] != bins["e_hull_lt_0"] / 1000:
        raise AssertionError("results.json S.U.N. score differs from per-material energies")

    evaluator_geometric_valid = sum(
        structure.volume / structure.composition.num_atoms < 30 for structure in structures
    )
    missing_site_properties = sum(
        1
        for item in structures_raw
        if any("properties" not in site for site in item.get("sites", []))
    )
    summary = {
        "status": "ok",
        "structures": len(structures),
        "unique_reduced_formulas": len(set(formulas)),
        "strict_in_sample_sun": bins["e_hull_lt_0"],
        "energy_bins": bins,
        "evaluator_geometric_valid": evaluator_geometric_valid,
        "structures_missing_site_properties": missing_site_properties,
        "input_sha256": digest(ROOT / "input.json"),
        "results_sha256": digest(ROOT / "results.json"),
        "checksum_files_verified": verify_checksums(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
