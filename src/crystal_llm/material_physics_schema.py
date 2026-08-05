"""Schemas and utilities for the materials-physics MVP."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
import re
from typing import Any, Mapping, Sequence


MATERIAL_PHYSICS_DIRECTIVE = "根据你自己的材料学知识和你自己对于材料物理底层原理的理解来判断。"

ALLOWED_QUERY_KEYS = {
    "material_ids",
    "formula_in",
    "formula_regex",
    "elements_all",
    "elements_any",
    "elements_none",
    "nelements_min",
    "nelements_max",
    "nsites_min",
    "nsites_max",
    "band_gap_min",
    "band_gap_max",
    "formation_energy_per_atom_min",
    "formation_energy_per_atom_max",
    "density_min",
    "density_max",
    "volume_per_atom_min",
    "volume_per_atom_max",
    "crystal_system_in",
    "spacegroup_number_in",
    "spacegroup_number_min",
    "spacegroup_number_max",
    "preferred_order",
}

ALLOWED_BRANCH_SOURCES = {"mp_pool", "generator", "mattergen"}

ALLOWED_RELATIONS = {
    "primary_lower_e_hull_than_control",
    "primary_higher_e_hull_than_control",
    "primary_lower_form_e_than_control",
    "primary_higher_form_e_than_control",
}

ALLOWED_CONFIDENCE = {"low", "medium-low", "medium", "medium-high", "high"}
ALLOWED_PREFERRED_ORDER_FIELDS = {
    "material_id",
    "formula",
    "formation_energy_per_atom",
    "band_gap",
    "density",
    "volume_per_atom",
    "nelements",
    "nsites",
    "spacegroup_number",
}


@dataclass
class MaterializationSelection:
    record: dict[str, Any]
    query: dict[str, Any]
    role: str
    bundle_id: str
    prediction_ids: list[str]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _invert_text_for_desc(text: str) -> str:
    return "".join(chr(0x10FFFF - ord(char)) for char in text)


def _preferred_order_terms(preferred_order: Any) -> list[tuple[str, str]]:
    if isinstance(preferred_order, str):
        text = preferred_order.strip()
        if text in {"material_id", "random"}:
            return [(text, "asc")]
        if not text:
            return []
        preferred_order = [text]
    if not isinstance(preferred_order, Sequence) or isinstance(preferred_order, (str, bytes)):
        return []
    terms: list[tuple[str, str]] = []
    for item in preferred_order:
        text = str(item or "").strip()
        if not text:
            continue
        parts = text.split()
        field = parts[0]
        direction = parts[1].lower() if len(parts) > 1 else "asc"
        terms.append((field, direction))
    return terms


def _sort_value_for_term(record: Mapping[str, Any], field: str, direction: str) -> tuple[int, Any]:
    value = record.get(field)
    if field == "volume_per_atom" and value is None and record.get("volume") is not None and record.get("nsites"):
        try:
            value = float(record["volume"]) / max(1, int(record["nsites"]))
        except Exception:
            value = None
    if value is None:
        return (1, "")
    if isinstance(value, (int, float)):
        number = float(value)
        return (0, -number if direction == "desc" else number)
    text = str(value)
    if direction == "desc":
        text = _invert_text_for_desc(text)
    return (0, text)


def _as_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _is_number(value: Any) -> bool:
    return _as_float(value) is not None


def schema_reference_json(target_count: int = 10) -> str:
    reference = {
        "material_physics_mechanism_consensus": {
            "status": "consensus",
            "accepted_mechanisms": [
                {
                    "id": "m001",
                    "claim": "A narrow structural motif appears more stable because its local coordination and charge balance reduce lattice strain.",
                    "rationale_summary": "Explain the causal mechanism in observable materials terms.",
                    "evidence_chain": [
                        {
                            "premise": "A small set of known materials shows lower e_hull in the same family.",
                            "observation": "The family shares a recurring local coordination pattern.",
                            "implication": "The coordination motif is a plausible stability driver.",
                            "confidence": "medium",
                        }
                    ],
                    "counterevidence_considered": ["List at least one near-miss or exception."],
                    "scope": "Where the mechanism should and should not be applied.",
                    "confidence": "medium",
                    "expected_material_signs": [
                        "observable compositional or structural signals that should co-occur if the mechanism is right"
                    ],
                    "do_not_generalize_to": ["explicit exclusions"],
                    "testable_implications": ["Concrete, falsifiable consequences for new materials."],
                }
            ],
            "rejected_mechanisms": [
                {"id": "m002", "claim": "A rejected overgeneralization", "rejection_reason": "why it failed"}
            ],
            "consensus_summary": "Brief summary of the surviving mechanism claims.",
            "mechanism_requirements": {
                "must_use_observable_fields": [
                    "elements",
                    "nelements",
                    "nsites",
                    "band_gap",
                    "formation_energy_per_atom",
                    "density",
                    "volume",
                    "crystal_system",
                    "spacegroup_number",
                    "spacegroup_symbol",
                    "formula",
                ],
                "must_not_use_fields": [
                    "is_stable",
                    "e_hull",
                    "strict_sun",
                    "sun",
                    "novelty",
                    "material_id",
                    "cif_path",
                    "properties_path",
                ],
                "desired_behavior": [
                    "Explain why the materials should be stable or unstable.",
                    "Make the mechanism falsifiable on the candidate pool.",
                ],
            },
        },
        "material_physics_prediction_consensus": {
            "status": "consensus",
            "accepted_predictions": [
                {
                    "id": "p001",
                    "mechanism_ids": ["m001"],
                    "claim": "If the mechanism is correct, the proposed primary family should be more stable than a matched control family.",
                    "predicted_relation": "primary_lower_e_hull_than_control",
                    "observable_basis": [
                        "The predicted difference should follow from composition, coordination, and structural constraints."
                    ],
                    "comparison_design": {
                        "primary_query": {"elements_all": ["La", "O"], "elements_any": ["F", "S"], "nelements_min": 4},
                        "control_query": {"elements_all": ["La", "O"], "elements_any": ["F", "S"], "nelements_min": 4},
                        "matching_notes": [
                            "Use closely matched chemistry except for the causal variable under discussion."
                        ],
                    },
                    "falsification_criteria": ["The primary group does not outperform the control group on e_hull."],
                    "scope": "Specific to the tested family and not a global law.",
                    "confidence": "medium",
                }
            ],
            "rejected_predictions": [
                {"id": "p002", "claim": "Rejected prediction", "rejection_reason": "why it failed"}
            ],
            "consensus_summary": "Brief summary of the surviving predictive claims.",
            "prediction_requirements": {
                "must_use_relation": [
                    "primary_lower_e_hull_than_control",
                    "primary_higher_e_hull_than_control",
                    "primary_lower_form_e_than_control",
                    "primary_higher_form_e_than_control",
                ],
                "must_not_use_fields": [
                    "is_stable",
                    "e_hull",
                    "strict_sun",
                    "sun",
                    "novelty",
                    "material_id",
                    "cif_path",
                    "properties_path",
                ],
                "desired_behavior": [
                    "Translate mechanisms into pairwise, falsifiable predictions.",
                ],
            },
        },
        "material_physics_execution_plan": {
            "status": "consensus",
            "accepted_bundles": [
                {
                    "id": "b001",
                    "prediction_ids": ["p001"],
                    "expected_relation": "primary_lower_e_hull_than_control",
                    "primary": {
                        "count": 3,
                        "query": {
                            "elements_all": ["La", "O"],
                            "elements_any": ["F", "S"],
                            "nelements_min": 4,
                            "preferred_order": [
                                "formation_energy_per_atom asc",
                                "band_gap desc",
                                "density asc",
                                "volume_per_atom asc",
                                "formula asc",
                            ],
                        },
                    },
                    "control": {
                        "count": 3,
                        "query": {
                            "elements_all": ["La", "O"],
                            "elements_any": ["F", "S"],
                            "nelements_min": 4,
                            "preferred_order": [
                                "formation_energy_per_atom asc",
                                "band_gap desc",
                                "density asc",
                                "volume_per_atom asc",
                                "formula asc",
                            ],
                        },
                    },
                    "rationale_summary": "How this bundle makes the prediction falsifiable.",
                    "selection_notes": "Keep the two groups close enough to isolate the proposed mechanism.",
                }
            ],
            "rejected_bundles": [
                {"id": "b002", "reason": "Rejected materialization idea"}
            ],
            "consensus_summary": "Brief summary of the executable test plan.",
            "materialization_constraints": {
                "candidate_count": target_count,
                "allowed_sources": ["mp_pool", "generator", "mattergen"],
                "generator_branch_fields": [
                    "formula_probes for local template-based generator probes",
                    "structure_dicts for explicit pymatgen Structure.as_dict() objects when formula_probes cannot represent the chemistry",
                ],
                "mattergen_branch_fields": [
                    "mattergen_requests for conditional MatterGen generation from chemical_system and low energy_above_hull conditioning",
                ],
                "preferred_total_bundles": 1,
            },
        },
    }
    return json.dumps(reference, ensure_ascii=False, indent=2)


def validate_mechanism_payload(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ["payload must be a JSON object"]
    if payload.get("agent") not in {"A", "B"} and payload.get("status") != "consensus":
        errors.append('payload.agent must be "A" or "B" or payload.status must be "consensus"')

    if payload.get("status") == "consensus":
        accepted = payload.get("accepted_mechanisms")
        if not isinstance(accepted, list) or not accepted:
            errors.append("accepted_mechanisms must be a non-empty list")
        else:
            for index, item in enumerate(accepted):
                path = f"accepted_mechanisms[{index}]"
                if not isinstance(item, Mapping):
                    errors.append(f"{path} must be an object")
                    continue
                if not str(item.get("id") or "").strip():
                    errors.append(f"{path}.id is required")
                if not str(item.get("claim") or "").strip():
                    errors.append(f"{path}.claim is required")
                if not str(item.get("rationale_summary") or "").strip():
                    errors.append(f"{path}.rationale_summary is required")
                if not _as_string_list(item.get("evidence_chain")):
                    errors.append(f"{path}.evidence_chain must be a non-empty list")
                if not str(item.get("scope") or "").strip():
                    errors.append(f"{path}.scope is required")
                confidence = str(item.get("confidence") or "").strip().lower()
                if confidence not in ALLOWED_CONFIDENCE:
                    errors.append(f"{path}.confidence must be one of {sorted(ALLOWED_CONFIDENCE)}")
        if not str(payload.get("consensus_summary") or "").strip():
            errors.append("consensus_summary is required")
        return errors

    if payload.get("agent") == "A":
        if not isinstance(payload.get("mechanisms"), list):
            errors.append("mechanisms must be a list")
    if payload.get("agent") == "B":
        if not isinstance(payload.get("required_revisions"), list):
            errors.append("required_revisions must be a list")
    return errors


def validate_prediction_payload(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ["payload must be a JSON object"]
    if payload.get("status") == "consensus":
        accepted = payload.get("accepted_predictions")
        if not isinstance(accepted, list) or not accepted:
            errors.append("accepted_predictions must be a non-empty list")
        else:
            for index, item in enumerate(accepted):
                path = f"accepted_predictions[{index}]"
                if not isinstance(item, Mapping):
                    errors.append(f"{path} must be an object")
                    continue
                if not str(item.get("id") or "").strip():
                    errors.append(f"{path}.id is required")
                if not _as_string_list(item.get("mechanism_ids")):
                    errors.append(f"{path}.mechanism_ids must be a non-empty list")
                if not str(item.get("claim") or "").strip():
                    errors.append(f"{path}.claim is required")
                if str(item.get("predicted_relation") or "") not in ALLOWED_RELATIONS:
                    errors.append(f"{path}.predicted_relation must be one of {sorted(ALLOWED_RELATIONS)}")
                if not isinstance(item.get("comparison_design"), Mapping):
                    errors.append(f"{path}.comparison_design must be an object")
                if not _as_string_list(item.get("falsification_criteria")):
                    errors.append(f"{path}.falsification_criteria must be a non-empty list")
        if not str(payload.get("consensus_summary") or "").strip():
            errors.append("consensus_summary is required")
        return errors

    if payload.get("agent") == "C":
        if not isinstance(payload.get("predictions"), list):
            errors.append("predictions must be a list")
    if payload.get("agent") == "D":
        if not isinstance(payload.get("required_revisions"), list):
            errors.append("required_revisions must be a list")
    return errors


def validate_execution_payload(payload: Any, *, target_count: int = 10) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ["payload must be a JSON object"]
    if payload.get("status") == "materialization_conflict":
        if not _as_string_list(payload.get("conflicting_bundle_ids")):
            errors.append("conflicting_bundle_ids must be a non-empty list")
        if not str(payload.get("reason") or "").strip():
            errors.append("reason is required")
        if not str(payload.get("minimal_fix_needed") or "").strip():
            errors.append("minimal_fix_needed is required")
        return errors
    if payload.get("status") != "consensus":
        errors.append('payload.status must be "consensus" or "materialization_conflict"')
        return errors

    bundles = payload.get("accepted_bundles")
    if not isinstance(bundles, list) or not bundles:
        errors.append("accepted_bundles must be a non-empty list")
        return errors

    bundle_total = 0
    for index, item in enumerate(bundles):
        path = f"accepted_bundles[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{path} must be an object")
            continue
        if not str(item.get("id") or "").strip():
            errors.append(f"{path}.id is required")
        prediction_ids = _as_string_list(item.get("prediction_ids"))
        if not prediction_ids:
            errors.append(f"{path}.prediction_ids must be a non-empty list")
        elif len(prediction_ids) != 1:
            errors.append(f"{path}.prediction_ids must contain exactly one prediction id")
        if str(item.get("expected_relation") or "") not in ALLOWED_RELATIONS:
            errors.append(f"{path}.expected_relation must be one of {sorted(ALLOWED_RELATIONS)}")
        if not str(item.get("rationale_summary") or "").strip():
            errors.append(f"{path}.rationale_summary is required")
        if not str(item.get("selection_notes") or "").strip():
            errors.append(f"{path}.selection_notes is required")
        bundle_source = str(item.get("source") or "mp_pool")
        if bundle_source not in ALLOWED_BRANCH_SOURCES:
            errors.append(f"{path}.source must be one of {sorted(ALLOWED_BRANCH_SOURCES)}")
        for role in ("primary", "control"):
            branch = item.get(role)
            branch_path = f"{path}.{role}"
            if not isinstance(branch, Mapping):
                errors.append(f"{branch_path} must be an object")
                continue
            formula_probes = branch.get("formula_probes")
            structure_dicts = branch.get("structure_dicts")
            mattergen_requests = branch.get("mattergen_requests")
            has_formula_probes = isinstance(formula_probes, list) and bool(formula_probes)
            has_structure_dicts = isinstance(structure_dicts, list) and bool(structure_dicts)
            has_mattergen_requests = isinstance(mattergen_requests, list) and bool(mattergen_requests)
            source = str(branch.get("source") or bundle_source)
            if (not branch.get("source")) and (has_formula_probes or has_structure_dicts):
                source = "generator"
            if (not branch.get("source")) and has_mattergen_requests:
                source = "mattergen"
            if source not in ALLOWED_BRANCH_SOURCES:
                errors.append(f"{branch_path}.source must be one of {sorted(ALLOWED_BRANCH_SOURCES)}")
            count = branch.get("count")
            if source == "generator" and has_formula_probes and has_structure_dicts:
                errors.append(f"{branch_path} must include only one of formula_probes or structure_dicts")
            if source == "mattergen" and (has_formula_probes or has_structure_dicts):
                errors.append(f"{branch_path} must not include formula_probes or structure_dicts when source is mattergen")
            if source != "mattergen" and has_mattergen_requests:
                errors.append(f"{branch_path}.mattergen_requests is only allowed when source is mattergen")
            if source == "generator" and (has_formula_probes or has_structure_dicts):
                inferred_count = len(structure_dicts) if has_structure_dicts else len(formula_probes)
                if not isinstance(count, int) or count <= 0:
                    count = inferred_count
                elif count != inferred_count:
                    errors.append(
                        f"{branch_path}.count must match generator item length {inferred_count} when source is generator"
                    )
                bundle_total += inferred_count
            elif source == "mattergen":
                if not isinstance(count, int) or count <= 0:
                    errors.append(f"{branch_path}.count must be a positive integer")
                else:
                    bundle_total += count
                if not has_mattergen_requests:
                    errors.append(f"{branch_path} must include non-empty mattergen_requests when source is mattergen")
                elif len(mattergen_requests) != 1:
                    errors.append(f"{branch_path}.mattergen_requests must contain exactly one request object")
                else:
                    request = mattergen_requests[0]
                    if not isinstance(request, Mapping):
                        errors.append(f"{branch_path}.mattergen_requests[0] must be an object")
                    else:
                        request_errors = validate_mattergen_request(request)
                        errors.extend(f"{branch_path}.mattergen_requests[0]: {err}" for err in request_errors)
            else:
                if not isinstance(count, int) or count <= 0:
                    errors.append(f"{branch_path}.count must be a positive integer")
                else:
                    bundle_total += count
            query = branch.get("query")
            material_ids = _as_string_list(branch.get("material_ids"))
            if source == "generator":
                if not has_formula_probes and not has_structure_dicts:
                    errors.append(
                        f"{branch_path} must include non-empty formula_probes or structure_dicts when source is generator"
                    )
                if has_structure_dicts:
                    for structure_index, structure_dict in enumerate(structure_dicts):
                        if not isinstance(structure_dict, Mapping):
                            errors.append(f"{branch_path}.structure_dicts[{structure_index}] must be an object")
            elif source == "mattergen":
                if material_ids:
                    errors.append(f"{branch_path}.material_ids is not allowed when source is mattergen")
                if isinstance(query, Mapping):
                    qerrors = validate_query(query)
                    errors.extend(f"{branch_path}.query: {err}" for err in qerrors)
            else:
                if not isinstance(query, Mapping) and not material_ids:
                    errors.append(f"{branch_path} must include query or material_ids")
                if isinstance(query, Mapping):
                    qerrors = validate_query(query)
                    errors.extend(f"{branch_path}.query: {err}" for err in qerrors)
            selection_order = branch.get("selection_order")
            if selection_order is not None and not isinstance(selection_order, str):
                errors.append(f"{branch_path}.selection_order must be material_id or random")
            elif selection_order not in {None, "material_id", "random"}:
                errors.append(f"{branch_path}.selection_order must be material_id or random")

    if bundle_total != target_count:
        errors.append(f"total materialization count must equal {target_count}, got {bundle_total}")
    if not str(payload.get("consensus_summary") or "").strip():
        errors.append("consensus_summary is required")
    return errors


def validate_mattergen_request(request: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    backend = str(request.get("backend") or "mattergen")
    if backend != "mattergen":
        errors.append("backend must be mattergen")
    properties = request.get("properties_to_condition_on")
    if not isinstance(properties, Mapping):
        errors.append("properties_to_condition_on must be an object")
        properties = {}
    filters = request.get("filters")
    if filters is not None and not isinstance(filters, Mapping):
        errors.append("filters must be an object when provided")
        filters = {}
    if filters is None:
        filters = {}
    chemical_system = properties.get("chemical_system") or filters.get("chemical_system")
    if not _as_string_list(chemical_system):
        errors.append("chemical_system is required in properties_to_condition_on or filters")
    energy = properties.get("energy_above_hull")
    if energy is None:
        errors.append("properties_to_condition_on.energy_above_hull is required")
    elif not _is_number(energy):
        errors.append("properties_to_condition_on.energy_above_hull must be a finite number")
    guidance = request.get("diffusion_guidance_factor")
    if guidance is not None and not _is_number(guidance):
        errors.append("diffusion_guidance_factor must be a finite number when provided")
    max_sites = filters.get("max_sites")
    if max_sites is not None and (not isinstance(max_sites, int) or max_sites <= 0):
        errors.append("filters.max_sites must be a positive integer when provided")
    target_count = request.get("target_count")
    if target_count is not None and (not isinstance(target_count, int) or target_count <= 0):
        errors.append("target_count must be a positive integer when provided")
    return errors


def validate_query(query: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in query:
        if key not in ALLOWED_QUERY_KEYS:
            errors.append(f"unsupported query key: {key}")
    if "material_ids" in query and not _as_string_list(query.get("material_ids")):
        errors.append("material_ids must be a non-empty list when provided")
    for key in ("formula_in", "elements_all", "elements_any", "elements_none", "crystal_system_in"):
        value = query.get(key)
        if value is not None and not _as_string_list(value):
            errors.append(f"{key} must be a non-empty list when provided")
    for key in (
        "nelements_min",
        "nelements_max",
        "nsites_min",
        "nsites_max",
        "band_gap_min",
        "band_gap_max",
        "formation_energy_per_atom_min",
        "formation_energy_per_atom_max",
        "density_min",
        "density_max",
        "volume_per_atom_min",
        "volume_per_atom_max",
        "spacegroup_number_min",
        "spacegroup_number_max",
    ):
        if key in query and not _is_number(query.get(key)):
            errors.append(f"{key} must be a finite number when provided")
    sg_in = query.get("spacegroup_number_in")
    if sg_in is not None:
        if not isinstance(sg_in, Sequence) or isinstance(sg_in, (str, bytes)) or not sg_in:
            errors.append("spacegroup_number_in must be a non-empty list when provided")
        else:
            for item in sg_in:
                if not isinstance(item, int):
                    errors.append("spacegroup_number_in entries must be integers")
                    break
    preferred_order = query.get("preferred_order")
    if preferred_order is not None:
        if isinstance(preferred_order, str):
            if preferred_order not in {"material_id", "random"}:
                errors.append("preferred_order must be material_id, random, or an ordered list of sort directives")
        elif isinstance(preferred_order, Sequence) and not isinstance(preferred_order, (str, bytes)):
            if not preferred_order:
                errors.append("preferred_order list must be non-empty when provided")
            for item in preferred_order:
                text = str(item or "").strip()
                if not text:
                    errors.append("preferred_order list entries must be non-empty strings")
                    continue
                parts = text.split()
                field = parts[0]
                direction = parts[1].lower() if len(parts) > 1 else "asc"
                if field not in ALLOWED_PREFERRED_ORDER_FIELDS:
                    errors.append(f"preferred_order field {field!r} is not supported")
                if direction not in {"asc", "desc"}:
                    errors.append(f"preferred_order direction {direction!r} is not supported")
        else:
            errors.append("preferred_order must be a string or an ordered list of sort directives")
    return errors


def query_matches(record: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
    if not isinstance(record, Mapping):
        return False
    if not isinstance(query, Mapping):
        return False

    if "material_ids" in query:
        material_ids = {str(item) for item in _as_string_list(query.get("material_ids"))}
        if material_ids and str(record.get("material_id")) not in material_ids:
            return False
    if "formula_in" in query:
        formulas = {str(item) for item in _as_string_list(query.get("formula_in"))}
        if formulas and str(record.get("formula")) not in formulas:
            return False
    if "formula_regex" in query:
        pattern = str(query.get("formula_regex") or "")
        if pattern and not re.search(pattern, str(record.get("formula") or "")):
            return False

    elements = {str(item) for item in _as_string_list(record.get("elements"))}
    for key, mode in (("elements_all", "all"), ("elements_any", "any"), ("elements_none", "none")):
        items = {str(item) for item in _as_string_list(query.get(key))}
        if not items:
            continue
        if mode == "all" and not items.issubset(elements):
            return False
        if mode == "any" and not items.intersection(elements):
            return False
        if mode == "none" and items.intersection(elements):
            return False

    if "nelements_min" in query and int(record.get("nelements", -1) or -1) < int(query["nelements_min"]):
        return False
    if "nelements_max" in query and int(record.get("nelements", 10**9) or 10**9) > int(query["nelements_max"]):
        return False
    if "nsites_min" in query and int(record.get("nsites", -1) or -1) < int(query["nsites_min"]):
        return False
    if "nsites_max" in query and int(record.get("nsites", 10**9) or 10**9) > int(query["nsites_max"]):
        return False

    for field in ("band_gap", "formation_energy_per_atom", "density", "volume_per_atom"):
        min_key = f"{field}_min"
        max_key = f"{field}_max"
        value = _as_float(record.get(field))
        if "volume_per_atom" == field and value is None and record.get("volume") is not None and record.get("nsites"):
            try:
                value = float(record["volume"]) / max(1, int(record["nsites"]))
            except Exception:
                value = None
        if value is None:
            continue
        if min_key in query and value < float(query[min_key]):
            return False
        if max_key in query and value > float(query[max_key]):
            return False

    if "crystal_system_in" in query:
        allowed = {str(item) for item in _as_string_list(query.get("crystal_system_in"))}
        if allowed and str(record.get("crystal_system")) not in allowed:
            return False
    if "spacegroup_number_in" in query:
        allowed_numbers = {int(item) for item in query.get("spacegroup_number_in", []) if isinstance(item, int)}
        if allowed_numbers and int(record.get("spacegroup_number", -1) or -1) not in allowed_numbers:
            return False
    if "spacegroup_number_min" in query and int(record.get("spacegroup_number", -1) or -1) < int(query["spacegroup_number_min"]):
        return False
    if "spacegroup_number_max" in query and int(record.get("spacegroup_number", 10**9) or 10**9) > int(query["spacegroup_number_max"]):
        return False

    return True


def select_matches(
    records: Sequence[Mapping[str, Any]],
    query: Mapping[str, Any],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    matches = [dict(record) for record in records if query_matches(record, query)]
    preferred_order = query.get("preferred_order")
    if isinstance(preferred_order, Sequence) and not isinstance(preferred_order, (str, bytes)):
        terms = _preferred_order_terms(preferred_order)
        if terms:
            matches.sort(key=lambda item: tuple(_sort_value_for_term(item, field, direction) for field, direction in terms))
        else:
            matches.sort(key=lambda item: (str(item.get("material_id")), str(item.get("formula"))))
    else:
        order = str(preferred_order or "material_id")
        if order == "random":
            rng = random.Random(seed)
            rng.shuffle(matches)
        else:
            matches.sort(key=lambda item: (str(item.get("material_id")), str(item.get("formula"))))
    return matches[: max(0, count)]
