"""MatterGen adapter for the crystal_LLM evaluator contract.

This prototype keeps MatterGen isolated from the production X/Y/Z/W runner.
It accepts a structured request from an LLM/controller, optionally launches the
MatterGen CLI, reads MatterGen CIF/extxyz outputs, filters the generated
structures, and writes the same `input.json` shape used by this repository:
a list of `pymatgen.core.Structure.as_dict()` objects.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from monty.json import MontyEncoder
from pymatgen.core import Structure

try:
    from pymatgen.io.ase import AseAtomsAdaptor
except Exception:  # pragma: no cover - optional import for extxyz conversion
    AseAtomsAdaptor = None  # type: ignore[assignment]

try:
    from pymatgen.io.cif import CifParser
except Exception:  # pragma: no cover - pymatgen always provides this here
    CifParser = None  # type: ignore[assignment]

try:
    from crystal_llm.filters import reduced_formula, validate_structure
except Exception:  # pragma: no cover - allows running outside PYTHONPATH=src

    @dataclass
    class _ValidationResult:
        ok: bool
        reasons: list[str]

    def reduced_formula(structure: Structure) -> str:
        return structure.composition.reduced_formula

    def validate_structure(
        structure: Structure,
        *,
        max_sites: int = 80,
        min_volume_per_atom: float = 4.0,
        max_volume_per_atom: float = 45.0,
    ) -> _ValidationResult:
        reasons: list[str] = []
        if len(structure) <= 0:
            reasons.append("empty_structure")
        if len(structure) > max_sites:
            reasons.append("too_many_sites")
        if structure.volume <= 0:
            reasons.append("nonpositive_volume")
        volume_per_atom = float(structure.volume / max(1, len(structure)))
        if volume_per_atom < min_volume_per_atom:
            reasons.append("volume_per_atom_too_small")
        if volume_per_atom > max_volume_per_atom:
            reasons.append("volume_per_atom_too_large")
        return _ValidationResult(ok=not reasons, reasons=reasons)


SUPPORTED_STRUCTURE_SUFFIXES = {".cif", ".xyz", ".extxyz", ".zip"}
DEFAULT_DIFFUSION_GUIDANCE_FACTOR = 1.0
DEFAULT_MAX_VOLUME_PER_ATOM = 45.0


@dataclass
class MatterGenFilters:
    allowed_elements: set[str] = field(default_factory=set)
    forbidden_elements: set[str] = field(default_factory=set)
    chemical_system: set[str] = field(default_factory=set)
    require_chemical_system_exact: bool = False
    target_reduced_formula: str | None = None
    require_target_reduced_formula: bool = False
    exclude_reduced_formulas: set[str] = field(default_factory=set)
    max_sites: int = 20
    min_sites: int = 1
    max_volume_per_atom: float = DEFAULT_MAX_VOLUME_PER_ATOM
    min_volume_per_atom: float = 4.0
    deduplicate_reduced_formula: bool = True


@dataclass
class MatterGenRequest:
    request_id: str
    checkpoint: str
    model_path: str | None
    target_count: int
    batch_size: int
    num_batches: int
    properties_to_condition_on: dict[str, Any] = field(default_factory=dict)
    diffusion_guidance_factor: float | None = None
    filters: MatterGenFilters = field(default_factory=MatterGenFilters)
    extra_cli_args: list[str] = field(default_factory=list)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or convert a MatterGen generation request.")
    parser.add_argument("--request", required=True, help="MatterGen request JSON.")
    parser.add_argument("--work-dir", default="mattergen_work", help="Directory for command/output artifacts.")
    parser.add_argument(
        "--from-existing",
        action="append",
        default=[],
        help="Existing MatterGen result path to convert. May be a directory, CIF/ZIP, or extxyz file.",
    )
    parser.add_argument("--mattergen-bin", default="mattergen-generate", help="MatterGen CLI executable.")
    parser.add_argument("--dry-run", action="store_true", help="Write normalized request/command but do not run.")
    parser.add_argument(
        "--strict-count",
        action="store_true",
        help="Exit nonzero if fewer accepted structures than target_count are produced.",
    )
    parser.add_argument("--input-name", default="input.json", help="Evaluator input filename.")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def as_string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        if "-" in value:
            return {part.strip() for part in value.split("-") if part.strip()}
        return {value.strip()} if value.strip() else set()
    if isinstance(value, Iterable):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def normalize_request(payload: Mapping[str, Any]) -> MatterGenRequest:
    backend = str(payload.get("backend", "mattergen"))
    if backend != "mattergen":
        raise ValueError(f"backend must be 'mattergen', got {backend!r}")
    request_id = str(payload.get("request_id") or payload.get("id") or "mattergen_request")
    checkpoint = str(payload.get("checkpoint") or payload.get("pretrained_name") or "mattergen_base")
    model_path = payload.get("model_path")
    if model_path is not None:
        model_path = str(Path(str(model_path)).expanduser())
    target_count = int(payload.get("target_count", payload.get("count", 16)))
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    batch_size = int(payload.get("batch_size", min(16, max(1, target_count))))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    oversample_factor = float(payload.get("oversample_factor", 2.0))
    num_batches = payload.get("num_batches")
    if num_batches is None:
        num_batches = max(1, math.ceil(target_count * oversample_factor / batch_size))
    num_batches = int(num_batches)
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")

    properties_raw = payload.get("properties_to_condition_on") or {}
    if not isinstance(properties_raw, Mapping):
        raise ValueError("properties_to_condition_on must be an object")
    properties_to_condition_on = dict(properties_raw)

    filter_payload = payload.get("filters") or {}
    if not isinstance(filter_payload, Mapping):
        raise ValueError("filters must be an object")
    chemical_system = as_string_set(filter_payload.get("chemical_system"))
    property_chemical_system = properties_to_condition_on.get("chemical_system")
    if not chemical_system and property_chemical_system:
        chemical_system = as_string_set(property_chemical_system)
    filters = MatterGenFilters(
        allowed_elements=as_string_set(filter_payload.get("allowed_elements")),
        forbidden_elements=as_string_set(filter_payload.get("forbidden_elements")),
        chemical_system=chemical_system,
        require_chemical_system_exact=as_bool(filter_payload.get("require_chemical_system_exact"), default=False),
        target_reduced_formula=(
            str(filter_payload["target_reduced_formula"])
            if filter_payload.get("target_reduced_formula")
            else None
        ),
        require_target_reduced_formula=as_bool(
            filter_payload.get("require_target_reduced_formula", payload.get("require_target_reduced_formula")),
            default=False,
        ),
        exclude_reduced_formulas=as_string_set(filter_payload.get("exclude_reduced_formulas")),
        max_sites=int(filter_payload.get("max_sites", 20)),
        min_sites=int(filter_payload.get("min_sites", 1)),
        max_volume_per_atom=float(filter_payload.get("max_volume_per_atom", DEFAULT_MAX_VOLUME_PER_ATOM)),
        min_volume_per_atom=float(filter_payload.get("min_volume_per_atom", 4.0)),
        deduplicate_reduced_formula=as_bool(filter_payload.get("deduplicate_reduced_formula"), default=True),
    )
    if filters.max_sites < filters.min_sites:
        raise ValueError("filters.max_sites must be >= filters.min_sites")

    extra_cli_args_raw = payload.get("extra_cli_args") or []
    if not isinstance(extra_cli_args_raw, Sequence) or isinstance(extra_cli_args_raw, str):
        raise ValueError("extra_cli_args must be a list of strings")
    extra_cli_args = [str(item) for item in extra_cli_args_raw]

    guidance = payload.get("diffusion_guidance_factor")
    if guidance is None and properties_to_condition_on:
        guidance = DEFAULT_DIFFUSION_GUIDANCE_FACTOR
    return MatterGenRequest(
        request_id=request_id,
        checkpoint=checkpoint,
        model_path=model_path,
        target_count=target_count,
        batch_size=batch_size,
        num_batches=num_batches,
        properties_to_condition_on=properties_to_condition_on,
        diffusion_guidance_factor=float(guidance) if guidance is not None else None,
        filters=filters,
        extra_cli_args=extra_cli_args,
    )


def request_to_jsonable(request: MatterGenRequest) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "backend": "mattergen",
        "checkpoint": request.checkpoint,
        "model_path": request.model_path,
        "target_count": request.target_count,
        "batch_size": request.batch_size,
        "num_batches": request.num_batches,
        "properties_to_condition_on": request.properties_to_condition_on,
        "diffusion_guidance_factor": request.diffusion_guidance_factor,
        "filters": {
            "allowed_elements": sorted(request.filters.allowed_elements),
            "forbidden_elements": sorted(request.filters.forbidden_elements),
            "chemical_system": sorted(request.filters.chemical_system),
            "require_chemical_system_exact": request.filters.require_chemical_system_exact,
            "target_reduced_formula": request.filters.target_reduced_formula,
            "require_target_reduced_formula": request.filters.require_target_reduced_formula,
            "exclude_reduced_formulas": sorted(request.filters.exclude_reduced_formulas),
            "max_sites": request.filters.max_sites,
            "min_sites": request.filters.min_sites,
            "max_volume_per_atom": request.filters.max_volume_per_atom,
            "min_volume_per_atom": request.filters.min_volume_per_atom,
            "deduplicate_reduced_formula": request.filters.deduplicate_reduced_formula,
        },
        "extra_cli_args": list(request.extra_cli_args),
    }


def build_mattergen_command(request: MatterGenRequest, results_path: Path, mattergen_bin: str) -> list[str]:
    command = [
        mattergen_bin,
        str(results_path),
        f"--batch_size={request.batch_size}",
        f"--num_batches={request.num_batches}",
    ]
    if request.model_path:
        command.append(f"--model_path={request.model_path}")
    else:
        command.append(f"--pretrained-name={request.checkpoint}")
    if request.properties_to_condition_on:
        condition_json = json.dumps(request.properties_to_condition_on, ensure_ascii=False, separators=(",", ":"))
        command.append(f"--properties_to_condition_on={condition_json}")
    if request.diffusion_guidance_factor is not None:
        command.append(f"--diffusion_guidance_factor={request.diffusion_guidance_factor}")
    command.extend(request.extra_cli_args)
    return command


def read_cif_from_text(text: str) -> list[Structure]:
    if CifParser is not None and hasattr(CifParser, "from_str"):
        parser = CifParser.from_str(text)
        return list(parser.parse_structures(primitive=False))
    with tempfile.NamedTemporaryFile("w", suffix=".cif", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        return [Structure.from_file(handle.name)]


def read_extxyz(path: Path) -> list[Structure]:
    if AseAtomsAdaptor is None:
        raise RuntimeError("ase/pymatgen ASE adaptor is required to read extxyz outputs")
    import ase.io

    atoms_list = ase.io.read(str(path), index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    return [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]


def read_structures_from_path(path: Path) -> list[Structure]:
    if path.is_dir():
        preferred = [
            path / "generated_crystals.extxyz",
            path / "generated_crystals.xyz",
            path / "generated_crystals_cif.zip",
        ]
        for child in preferred:
            if child.exists():
                return read_structures_from_path(child)
        structures: list[Structure] = []
        for child in sorted(path.iterdir()):
            if child.name in {"generated_trajectories.zip", "generated_trajectories.extxyz"}:
                continue
            if child.suffix.lower() in SUPPORTED_STRUCTURE_SUFFIXES:
                structures.extend(read_structures_from_path(child))
        return structures

    suffix = path.suffix.lower()
    if suffix == ".cif":
        return [Structure.from_file(str(path))]
    if suffix in {".xyz", ".extxyz"}:
        return read_extxyz(path)
    if suffix == ".zip":
        structures = []
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if not name.lower().endswith(".cif"):
                    continue
                text = archive.read(name).decode("utf-8")
                structures.extend(read_cif_from_text(text))
        return structures
    raise ValueError(f"unsupported structure path: {path}")


def reject_reason(structure: Structure, filters: MatterGenFilters, seen_formulas: set[str]) -> str | None:
    formula = reduced_formula(structure)
    elements = {element.symbol for element in structure.composition.elements}
    if len(structure) < filters.min_sites:
        return "too_few_sites"
    if len(structure) > filters.max_sites:
        return "too_many_sites"
    if filters.allowed_elements and not elements <= filters.allowed_elements:
        return "outside_allowed_elements"
    if filters.forbidden_elements and elements & filters.forbidden_elements:
        return "forbidden_element"
    if filters.chemical_system:
        if filters.require_chemical_system_exact and elements != filters.chemical_system:
            return "chemical_system_not_exact"
        if not filters.require_chemical_system_exact and not elements <= filters.chemical_system:
            return "outside_chemical_system"
    if (
        filters.target_reduced_formula
        and filters.require_target_reduced_formula
        and formula != filters.target_reduced_formula
    ):
        return "not_target_reduced_formula"
    if formula in filters.exclude_reduced_formulas:
        return "excluded_reduced_formula"
    if filters.deduplicate_reduced_formula and formula in seen_formulas:
        return "duplicate_reduced_formula"
    validation = validate_structure(
        structure,
        max_sites=filters.max_sites,
        min_volume_per_atom=filters.min_volume_per_atom,
        max_volume_per_atom=filters.max_volume_per_atom,
    )
    if not validation.ok:
        return "structure_validation:" + ",".join(validation.reasons)
    return None


def filter_structures(
    structures: Sequence[Structure],
    request: MatterGenRequest,
) -> tuple[list[Structure], Counter[str], list[dict[str, Any]]]:
    accepted: list[Structure] = []
    rejects: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    seen_formulas: set[str] = set()
    indexed_structures = list(enumerate(structures))
    target_formula = request.filters.target_reduced_formula
    if target_formula and not request.filters.require_target_reduced_formula:
        indexed_structures.sort(key=lambda item: (reduced_formula(item[1]) != target_formula, item[0]))
    for index, structure in indexed_structures:
        formula = reduced_formula(structure)
        reason = reject_reason(structure, request.filters, seen_formulas)
        if reason:
            rejects[reason] += 1
            if len(examples) < 20:
                examples.append({"index": index, "formula": formula, "reason": reason})
            continue
        accepted.append(structure)
        seen_formulas.add(formula)
        if len(accepted) >= request.target_count:
            break
    return accepted, rejects, examples


def selected_record(structure: Structure, request: MatterGenRequest, index: int) -> dict[str, Any]:
    formula = reduced_formula(structure)
    elements = sorted(element.symbol for element in structure.composition.elements)
    return {
        "material_id": f"mattergen::{request.request_id}::{index:04d}::{formula}",
        "formula": formula,
        "elements": elements,
        "nelements": len(elements),
        "nsites": len(structure),
        "source": "ml_generator",
        "generator_backend": "mattergen",
        "generator_request_id": request.request_id,
        "generator_checkpoint": request.checkpoint,
        "generator_properties_to_condition_on": dict(request.properties_to_condition_on),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, cls=MontyEncoder) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    request_path = Path(args.request)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    request = normalize_request(load_json(request_path))
    normalized = request_to_jsonable(request)
    results_path = work_dir / "mattergen_results"
    command = build_mattergen_command(request, results_path, args.mattergen_bin)
    write_json(work_dir / "request.normalized.json", normalized)
    write_json(work_dir / "mattergen_command.json", {"command": command, "cwd": str(Path.cwd())})

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "work_dir": str(work_dir), "command": command}, indent=2))
        return 0

    source_paths = [Path(item) for item in args.from_existing]
    if not source_paths:
        if shutil.which(args.mattergen_bin) is None:
            raise RuntimeError(
                f"{args.mattergen_bin!r} was not found. Install MatterGen or pass --from-existing."
            )
        completed = subprocess.run(command, cwd=Path.cwd(), check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"MatterGen command failed with exit code {completed.returncode}")
        source_paths = [results_path]

    raw_structures: list[Structure] = []
    for source_path in source_paths:
        raw_structures.extend(read_structures_from_path(source_path))

    accepted, rejects, reject_examples = filter_structures(raw_structures, request)
    input_structures = [structure.as_dict() for structure in accepted]
    records = [selected_record(structure, request, index) for index, structure in enumerate(accepted)]

    input_path = work_dir / args.input_name
    write_json(input_path, input_structures)
    write_json(work_dir / "selected_records.json", records)
    report = {
        "status": "ok" if accepted else "no_accepted_structures",
        "request": normalized,
        "raw_structure_count": len(raw_structures),
        "accepted_count": len(accepted),
        "target_count": request.target_count,
        "accepted_formulas": [record["formula"] for record in records],
        "reject_reasons": dict(sorted(rejects.items())),
        "reject_examples": reject_examples,
        "input_json": str(input_path),
        "selected_records_json": str(work_dir / "selected_records.json"),
    }
    write_json(work_dir / "generation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict_count and len(accepted) < request.target_count:
        return 2
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
