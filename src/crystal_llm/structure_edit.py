"""Constrained structure-edit operations for single-structure evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
from pymatgen.core import Element, Lattice, Structure

from crystal_llm.filters import validate_structure
from crystal_llm.llm_client import extract_json_object


ALLOWED_OPERATIONS = {
    "noop",
    "scale_lattice",
    "strain_lattice",
    "substitute_species",
    "replace_site_species",
    "swap_site_species",
    "perturb_site",
    "translate_sublattice",
}


EDIT_SCHEMA_DESCRIPTION = """Executable edit JSON schema.

Return exactly one JSON object. The top-level field "op" must be one of:
- {"op":"noop","reason":"..."}
- {"op":"scale_lattice","scale_factor":1.02,"reason":"..."}
- {"op":"strain_lattice","factors":[1.01,0.99,1.00],"reason":"..."}
- {"op":"substitute_species","from_element":"K","to_element":"Rb","max_sites":"all","reason":"..."}
- {"op":"replace_site_species","site_index":1,"to_element":"Rb","reason":"..."}
- {"op":"swap_site_species","site_index_a":1,"site_index_b":2,"reason":"..."}
- {"op":"perturb_site","site_index":1,"delta_frac":[0.01,-0.01,0.0],"reason":"..."}
- {"op":"translate_sublattice","element":"F","delta_frac":[0.01,0.0,0.0],"reason":"..."}

Index fields are 1-based. Lattice scale/strain factors are limited to 0.85-1.15
per edit. Fractional-coordinate deltas are limited to [-0.12, 0.12] per axis.
The controller validates the edited structure and rejects invalid outputs.
"""


@dataclass
class EditResult:
    ok: bool
    structure: Structure | None = None
    edit: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    validation_reasons: list[str] = field(default_factory=list)


def structure_to_single_json(structure: Structure) -> dict[str, Any]:
    """Return a JSON-serializable single-structure payload."""

    return structure.as_dict()


def structure_summary(structure: Structure) -> dict[str, Any]:
    """Compact summary used in LLM prompts and state files."""

    formula = structure.composition.reduced_formula
    elements = [element.symbol for element in structure.composition.elements]
    site_rows = []
    for index, site in enumerate(structure, start=1):
        site_rows.append(
            {
                "site_index": index,
                "species": site.species_string,
                "frac_coords": [round(float(value), 6) for value in site.frac_coords],
            }
        )
    validation = validate_structure(structure)
    return {
        "formula": formula,
        "elements": elements,
        "nsites": len(structure),
        "lattice_matrix": [
            [round(float(value), 6) for value in row]
            for row in structure.lattice.matrix
        ],
        "lattice_abc": [round(float(value), 6) for value in structure.lattice.abc],
        "lattice_angles": [round(float(value), 6) for value in structure.lattice.angles],
        "volume": round(float(structure.volume), 6),
        "volume_per_atom": round(float(structure.volume / max(1, len(structure))), 6),
        "locally_valid": validation.ok,
        "local_validation_reasons": validation.reasons,
        "sites": site_rows,
    }


def extract_edit_json(text: str) -> dict[str, Any]:
    """Parse an edit object from raw LLM text or a JSON-encoded string field."""

    parsed = extract_json_object(text)
    if isinstance(parsed, str):
        parsed = extract_json_object(parsed)
    if not isinstance(parsed, Mapping):
        raise ValueError("edit output must be a JSON object")
    if isinstance(parsed.get("final_edit_json"), str):
        nested = extract_json_object(str(parsed["final_edit_json"]))
        if isinstance(nested, Mapping):
            parsed = nested
    elif isinstance(parsed.get("final_edit"), Mapping):
        parsed = parsed["final_edit"]
    return dict(parsed)


def _clean_element(value: Any, field_name: str) -> str:
    symbol = str(value or "").strip()
    if not symbol:
        raise ValueError(f"{field_name} is required")
    try:
        element = Element(symbol)
    except Exception as exc:
        raise ValueError(f"{field_name} is not a valid element: {symbol}") from exc
    return element.symbol


def _factor(value: Any, field_name: str, *, low: float = 0.85, high: float = 1.15) -> float:
    try:
        factor = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not math.isfinite(factor) or not (low <= factor <= high):
        raise ValueError(f"{field_name} must be between {low} and {high}")
    return factor


def _site_index(value: Any, nsites: int, field_name: str) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a 1-based integer site index") from exc
    if index < 1 or index > nsites:
        raise ValueError(f"{field_name} out of range: {index} for {nsites} sites")
    return index - 1


def _delta_frac(value: Any, field_name: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{field_name} must be a 3-number list")
    try:
        delta = np.array([float(item) for item in value], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain numbers") from exc
    if not np.all(np.isfinite(delta)):
        raise ValueError(f"{field_name} contains non-finite values")
    if np.max(np.abs(delta)) > 0.12:
        raise ValueError(f"{field_name} values must be in [-0.12, 0.12]")
    return delta


def _ordered_species_and_coords(structure: Structure) -> tuple[list[str], list[list[float]], dict[str, list[Any]]]:
    if not structure.is_ordered:
        raise ValueError("disordered structures are not supported by the edit MVP")
    species = [site.specie.symbol for site in structure]
    coords = [[float(value) for value in site.frac_coords] for site in structure]
    site_properties = {key: list(values) for key, values in structure.site_properties.items()}
    return species, coords, site_properties


def _rebuild(
    source: Structure,
    *,
    lattice: Lattice | None = None,
    species: list[str] | None = None,
    coords: list[list[float]] | None = None,
    site_properties: dict[str, list[Any]] | None = None,
) -> Structure:
    old_species, old_coords, old_site_properties = _ordered_species_and_coords(source)
    rebuilt = Structure(
        lattice or source.lattice,
        species or old_species,
        coords or old_coords,
        coords_are_cartesian=False,
        to_unit_cell=True,
        site_properties=site_properties if site_properties is not None else old_site_properties,
    )
    rebuilt.properties = dict(source.properties or {})
    return rebuilt


def normalize_edit(edit: Mapping[str, Any]) -> dict[str, Any]:
    op = str(edit.get("op") or edit.get("operation") or "").strip().lower()
    op = re.sub(r"[^a-z0-9_]+", "_", op)
    if op not in ALLOWED_OPERATIONS:
        raise ValueError(f"unsupported edit operation: {op or 'empty'}")
    normalized = dict(edit)
    normalized["op"] = op
    return normalized


def apply_edit(structure: Structure, raw_edit: Mapping[str, Any], *, max_sites: int = 80) -> EditResult:
    """Apply one constrained edit and validate the resulting structure."""

    try:
        edit = normalize_edit(raw_edit)
        op = edit["op"]

        if op == "noop":
            edited = structure.copy()

        elif op == "scale_lattice":
            scale_factor = _factor(edit.get("scale_factor", edit.get("factor")), "scale_factor")
            edited = structure.copy()
            edited.scale_lattice(float(edited.volume) * scale_factor**3)

        elif op == "strain_lattice":
            factors_raw = edit.get("factors")
            if not isinstance(factors_raw, Sequence) or isinstance(factors_raw, (str, bytes)) or len(factors_raw) != 3:
                raise ValueError("factors must be a 3-number list")
            factors = [_factor(value, f"factors[{index}]") for index, value in enumerate(factors_raw)]
            matrix = np.array(structure.lattice.matrix, dtype=float)
            for row_index, factor in enumerate(factors):
                matrix[row_index, :] *= factor
            edited = _rebuild(structure, lattice=Lattice(matrix))

        elif op == "substitute_species":
            from_element = _clean_element(edit.get("from_element"), "from_element")
            to_element = _clean_element(edit.get("to_element"), "to_element")
            species, coords, site_properties = _ordered_species_and_coords(structure)
            matching = [index for index, symbol in enumerate(species) if symbol == from_element]
            if not matching:
                raise ValueError(f"from_element not present in structure: {from_element}")
            max_sites_value = edit.get("max_sites", "all")
            if str(max_sites_value).strip().lower() == "all":
                chosen = matching
            else:
                try:
                    limit = int(max_sites_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError("max_sites must be 'all' or an integer") from exc
                if limit <= 0:
                    raise ValueError("max_sites must be positive")
                chosen = matching[:limit]
            for index in chosen:
                species[index] = to_element
            edited = _rebuild(structure, species=species, coords=coords, site_properties=site_properties)

        elif op == "replace_site_species":
            to_element = _clean_element(edit.get("to_element"), "to_element")
            species, coords, site_properties = _ordered_species_and_coords(structure)
            index = _site_index(edit.get("site_index"), len(species), "site_index")
            species[index] = to_element
            edited = _rebuild(structure, species=species, coords=coords, site_properties=site_properties)

        elif op == "swap_site_species":
            species, coords, site_properties = _ordered_species_and_coords(structure)
            index_a = _site_index(edit.get("site_index_a"), len(species), "site_index_a")
            index_b = _site_index(edit.get("site_index_b"), len(species), "site_index_b")
            if index_a == index_b:
                raise ValueError("site_index_a and site_index_b must be different")
            species[index_a], species[index_b] = species[index_b], species[index_a]
            edited = _rebuild(structure, species=species, coords=coords, site_properties=site_properties)

        elif op == "perturb_site":
            species, coords, site_properties = _ordered_species_and_coords(structure)
            index = _site_index(edit.get("site_index"), len(species), "site_index")
            delta = _delta_frac(edit.get("delta_frac"), "delta_frac")
            coords[index] = list((np.array(coords[index], dtype=float) + delta) % 1.0)
            edited = _rebuild(structure, species=species, coords=coords, site_properties=site_properties)

        elif op == "translate_sublattice":
            element = _clean_element(edit.get("element"), "element")
            delta = _delta_frac(edit.get("delta_frac"), "delta_frac")
            species, coords, site_properties = _ordered_species_and_coords(structure)
            matching = [index for index, symbol in enumerate(species) if symbol == element]
            if not matching:
                raise ValueError(f"element not present in structure: {element}")
            for index in matching:
                coords[index] = list((np.array(coords[index], dtype=float) + delta) % 1.0)
            edited = _rebuild(structure, species=species, coords=coords, site_properties=site_properties)

        else:
            raise ValueError(f"unhandled edit operation: {op}")

        validation = validate_structure(edited, max_sites=max_sites)
        if not validation.ok:
            return EditResult(False, edit=edit, error="invalid_after_edit", validation_reasons=validation.reasons)
        return EditResult(True, structure=edited, edit=edit)

    except Exception as exc:
        return EditResult(False, edit=dict(raw_edit), error=str(exc))


def apply_edit_text(structure: Structure, text: str, *, max_sites: int = 80) -> EditResult:
    edit = extract_edit_json(text)
    return apply_edit(structure, edit, max_sites=max_sites)


def edit_to_json_string(edit: Mapping[str, Any]) -> str:
    return json.dumps(dict(edit), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
