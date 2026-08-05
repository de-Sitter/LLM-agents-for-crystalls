"""Validation, de-duplication, and training-set novelty utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import ast
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure


FORMULA_COLUMNS = (
    "formula",
    "formula_pretty",
    "pretty_formula",
    "reduced_formula",
    "composition",
    "full_formula",
    "anonymous_formula",
)


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    min_distance: float = math.inf
    volume_per_atom: float = 0.0


def reduced_formula(structure: Structure) -> str:
    return structure.composition.reduced_formula


def min_interatomic_distance(structure: Structure) -> float:
    if len(structure) < 2:
        return math.inf
    matrix = np.array(structure.distance_matrix, dtype=float)
    np.fill_diagonal(matrix, np.inf)
    return float(np.min(matrix))


def validate_structure(
    structure: Structure,
    *,
    max_elements: int = 10,
    max_sites: int = 80,
    min_distance: float = 0.75,
    min_volume_per_atom: float = 4.0,
    max_volume_per_atom: float = 29.5,
) -> ValidationResult:
    reasons: list[str] = []

    if len(structure) == 0:
        reasons.append("empty_structure")
        return ValidationResult(False, reasons)

    if len(structure) > max_sites:
        reasons.append("too_many_sites")

    if len(structure.composition.elements) > max_elements:
        reasons.append("too_many_elements")

    if structure.lattice.volume <= 0:
        reasons.append("non_positive_volume")

    volume_per_atom = float(structure.lattice.volume / max(1, len(structure)))
    if volume_per_atom < min_volume_per_atom:
        reasons.append("volume_per_atom_too_small")
    if volume_per_atom > max_volume_per_atom:
        reasons.append("volume_per_atom_too_large")

    if not structure.is_ordered:
        reasons.append("disordered_sites")

    min_dist = min_interatomic_distance(structure)
    if min_dist < min_distance:
        reasons.append("atoms_too_close")

    for site in structure:
        if not np.all(np.isfinite(site.frac_coords)):
            reasons.append("non_finite_coords")
            break

    return ValidationResult(not reasons, reasons, min_distance=min_dist, volume_per_atom=volume_per_atom)


def structure_signature(structure: Structure) -> tuple:
    lengths = tuple(round(float(x), 2) for x in structure.lattice.abc)
    angles = tuple(round(float(x), 1) for x in structure.lattice.angles)
    return (reduced_formula(structure), len(structure), lengths, angles)


class StructureDeduplicator:
    """StructureMatcher-backed de-duplicator grouped by reduced formula."""

    def __init__(self) -> None:
        self.matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
        self.by_formula: dict[str, list[Structure]] = {}
        self.signatures: set[tuple] = set()

    def add(self, structure: Structure) -> bool:
        signature = structure_signature(structure)
        if signature in self.signatures:
            return False

        formula = reduced_formula(structure)
        for existing in self.by_formula.get(formula, []):
            if self.matcher.fit(existing, structure):
                return False

        self.signatures.add(signature)
        self.by_formula.setdefault(formula, []).append(structure)
        return True


def _formula_from_text(value: str) -> str | None:
    value = value.strip()
    if not value or len(value) > 80:
        return None
    if not re.match(r"^[A-Z][A-Za-z0-9().+\-\s]*$", value):
        return None
    try:
        return Composition(value).reduced_formula
    except Exception:
        return None


def _formula_from_structure_blob(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None

    if text.startswith("{"):
        try:
            blob = json.loads(text)
            if isinstance(blob, dict) and "lattice" in blob and "sites" in blob:
                return Structure.from_dict(blob).composition.reduced_formula
        except Exception:
            try:
                blob = ast.literal_eval(text)
                if isinstance(blob, dict) and "lattice" in blob and "sites" in blob:
                    return Structure.from_dict(blob).composition.reduced_formula
            except Exception:
                return None

    if "_cell_length_a" in text or text.startswith("data_"):
        try:
            return Structure.from_str(text, fmt="cif").composition.reduced_formula
        except Exception:
            return None

    return None


def load_known_formulas(path: str | None) -> set[str]:
    """Best-effort parser for the provided MatBench/Kaggle training CSV."""

    if not path:
        return set()

    csv_path = Path(path)
    if not csv_path.exists():
        return set()

    formulas: set[str] = set()

    if csv_path.suffix.lower() == ".json":
        with csv_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        items = data if isinstance(data, list) else [data]
        for item in items:
            try:
                if isinstance(item, dict):
                    formulas.add(Structure.from_dict(item).composition.reduced_formula)
            except Exception:
                continue
        return formulas

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for column in FORMULA_COLUMNS:
                if column in row:
                    formula = _formula_from_text(row[column])
                    if formula:
                        formulas.add(formula)

            for value in row.values():
                if not value:
                    continue
                formula = _formula_from_structure_blob(value)
                if formula:
                    formulas.add(formula)

    return formulas


def count_formulas(structures: Iterable[Structure]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for structure in structures:
        formula = reduced_formula(structure)
        counts[formula] = counts.get(formula, 0) + 1
    return counts
