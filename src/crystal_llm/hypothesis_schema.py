"""Schemas and local validation for hypothesis-first generation MVP."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from typing import Any, Mapping, Sequence

from pymatgen.core.periodic_table import Element

from crystal_llm.chemistry import ANION_POOLS, CATION_POOLS
from crystal_llm.filters import reduced_formula
from crystal_llm.generate import load_formula_probes
from crystal_llm.templates import BUILDERS


TEMPLATE_ROLE_COUNTS: dict[str, dict[str, int]] = {
    "rocksalt": {"A": 1, "X": 1},
    "cesium_chloride": {"A": 1, "X": 1},
    "zincblende": {"A": 1, "X": 1},
    "wurtzite": {"A": 1, "X": 1},
    "fluorite": {"A": 1, "X": 2},
    "rutile": {"A": 1, "X": 2},
    "antifluorite": {"A": 2, "X": 1},
    "perovskite": {"A": 1, "B": 1, "X": 3},
    "double_perovskite": {"A": 2, "B": 1, "B2": 1, "X": 6},
    "spinel": {"A": 1, "B": 2, "X": 4},
    "corundum": {"A": 2, "X": 3},
    "delafossite": {"A": 1, "B": 1, "X": 2},
}

ALLOWED_TEMPLATES: tuple[str, ...] = tuple(BUILDERS)


@dataclass
class MaterializerValidation:
    ok: bool
    status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    strategy: dict[str, Any] = field(default_factory=dict)
    generated_count: int = 0
    formulas: list[str] = field(default_factory=list)
    template_counts: dict[str, int] = field(default_factory=dict)


def required_roles(template: str) -> tuple[str, ...]:
    return tuple(TEMPLATE_ROLE_COUNTS[template])


def template_interface() -> dict[str, Any]:
    return {
        name: {
            "required_roles": list(required_roles(name)),
            "role_stoichiometry": dict(TEMPLATE_ROLE_COUNTS[name]),
            "role_rules": {
                "X": "anion role; oxidation_state must be negative",
                "A/B/B2": "cation roles; oxidation_state must be positive",
            },
        }
        for name in ALLOWED_TEMPLATES
    }


def generator_element_pools() -> dict[str, Any]:
    return {
        "cation_pools_by_oxidation_state": {str(key): list(value) for key, value in CATION_POOLS.items()},
        "anion_pools_by_oxidation_state": {str(key): list(value) for key, value in ANION_POOLS.items()},
    }


def schema_reference_json(target_count: int = 10) -> str:
    reference = {
        "agent_c_success_shape": {
            "status": "ok",
            "hypothesis_ids": ["h001"],
            "formula_probes": [
                {
                    "id": "r0001_probe_001",
                    "template": "perovskite",
                    "family": "oxide_perovskite",
                    "roles": {
                        "A": {"element": "Ba", "oxidation_state": 2},
                        "B": {"element": "Hf", "oxidation_state": 4},
                        "X": {"element": "O", "oxidation_state": -2},
                    },
                    "hypothesis_ids": ["h001"],
                    "rationale_summary": "Why this concrete probe follows from the consensus.",
                }
            ],
            "materialization_notes": ["exactly %d probes are required" % target_count],
        },
        "agent_c_conflict_shape": {
            "status": "hypothesis_conflict",
            "conflicting_hypothesis_ids": ["h001", "h002"],
            "reason": "Why the accepted hypotheses cannot all be satisfied.",
            "minimal_fix_needed": "Smallest hypothesis change needed before materialization can continue.",
        },
        "allowed_templates": list(ALLOWED_TEMPLATES),
        "template_interface": template_interface(),
        "generator_element_pools": generator_element_pools(),
    }
    return json.dumps(reference, ensure_ascii=False, indent=2)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def validate_consensus_payload(payload: Any, *, target_count: int = 10) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ["consensus payload must be a JSON object"]
    if payload.get("status") != "consensus":
        errors.append('consensus.status must be "consensus"')
    accepted = payload.get("accepted_hypotheses")
    if not isinstance(accepted, list) or not accepted:
        errors.append("consensus.accepted_hypotheses must be a non-empty list")
    else:
        seen_ids: set[str] = set()
        for index, item in enumerate(accepted):
            path = f"accepted_hypotheses[{index}]"
            if not isinstance(item, Mapping):
                errors.append(f"{path} must be an object")
                continue
            hyp_id = str(item.get("id") or "").strip()
            if not hyp_id:
                errors.append(f"{path}.id is required")
            elif hyp_id in seen_ids:
                errors.append(f"{path}.id duplicates {hyp_id}")
            seen_ids.add(hyp_id)
            if not str(item.get("claim") or "").strip():
                errors.append(f"{path}.claim is required")
            if not str(item.get("rationale_summary") or item.get("reasoning_summary") or "").strip():
                errors.append(f"{path}.rationale_summary is required")
            templates = _string_list(item.get("target_templates"))
            if not templates:
                errors.append(f"{path}.target_templates must name at least one allowed template")
            for template in templates:
                if template not in BUILDERS:
                    errors.append(f"{path}.target_templates contains unsupported template {template!r}")
    constraints = payload.get("materialization_constraints")
    if isinstance(constraints, Mapping):
        count = constraints.get("candidate_count")
        if count is not None:
            try:
                if int(count) != target_count:
                    errors.append(
                        f"materialization_constraints.candidate_count must be {target_count}, got {count}"
                    )
            except (TypeError, ValueError):
                errors.append("materialization_constraints.candidate_count must be an integer")
        for key in ("allowed_templates", "preferred_templates"):
            templates = _string_list(constraints.get(key))
            for template in templates:
                if template not in BUILDERS:
                    errors.append(f"materialization_constraints.{key} contains unsupported template {template!r}")
    if not str(payload.get("consensus_summary") or "").strip():
        errors.append("consensus.consensus_summary is required")
    return errors


def parse_role(value: Any, path: str, errors: list[str]) -> tuple[str, int] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object with element and oxidation_state")
        return None
    element_raw = value.get("element")
    oxidation_raw = value.get("oxidation_state", value.get("oxi"))
    if not isinstance(element_raw, str) or not element_raw.strip():
        errors.append(f"{path}.element must be a non-empty string")
        return None
    element = element_raw.strip()
    try:
        symbol = Element(element).symbol
    except Exception as exc:
        errors.append(f"{path}.element {element!r} is not a valid element: {exc}")
        return None
    if symbol != element:
        errors.append(f"{path}.element must use canonical capitalization {symbol!r}, got {element!r}")
        return None
    try:
        oxidation_state = int(oxidation_raw)
    except (TypeError, ValueError):
        errors.append(f"{path}.oxidation_state must be an integer")
        return None
    return element, oxidation_state


def validate_probe(probe: Any, index: int) -> tuple[dict[str, tuple[str, int]], str | None, list[str]]:
    errors: list[str] = []
    parsed_roles: dict[str, tuple[str, int]] = {}
    if not isinstance(probe, Mapping):
        return {}, None, [f"formula_probes[{index}] must be an object"]
    path = f"formula_probes[{index}]"
    probe_id = str(probe.get("id") or "").strip()
    if not probe_id:
        errors.append(f"{path}.id is required")
    template = str(probe.get("template") or "").strip()
    if template not in BUILDERS:
        errors.append(f"{path}.template must be one of {list(ALLOWED_TEMPLATES)}, got {template!r}")
        return parsed_roles, probe_id or None, errors

    roles = probe.get("roles")
    if not isinstance(roles, Mapping):
        errors.append(f"{path}.roles must be an object")
        return parsed_roles, probe_id or None, errors

    required = set(required_roles(template))
    actual = {str(key) for key in roles}
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing:
        errors.append(f"{path}.roles missing required roles for {template}: {missing}")
    if extra:
        errors.append(f"{path}.roles has unsupported roles for {template}: {extra}")

    for role in required_roles(template):
        if role not in roles:
            continue
        parsed = parse_role(roles[role], f"{path}.roles.{role}", errors)
        if parsed is None:
            continue
        element, oxidation_state = parsed
        if role == "X":
            if oxidation_state >= 0:
                errors.append(f"{path}.roles.X.oxidation_state must be negative")
            pool = ANION_POOLS.get(oxidation_state)
            if pool is None or element not in pool:
                errors.append(
                    f"{path}.roles.X uses {element}{oxidation_state:+d}, "
                    "which is not in the generator anion pool"
                )
        else:
            if oxidation_state <= 0:
                errors.append(f"{path}.roles.{role}.oxidation_state must be positive")
            pool = CATION_POOLS.get(oxidation_state)
            if pool is None or element not in pool:
                errors.append(
                    f"{path}.roles.{role} uses {element}{oxidation_state:+d}, "
                    "which is not in the generator cation pool"
                )
        parsed_roles[role] = (element, oxidation_state)

    if len(parsed_roles) == len(required):
        elements = [item[0] for item in parsed_roles.values()]
        if len(elements) != len(set(elements)):
            errors.append(f"{path}.roles must use distinct elements across roles")
        charge = sum(
            parsed_roles[role][1] * count
            for role, count in TEMPLATE_ROLE_COUNTS[template].items()
        )
        if charge != 0:
            errors.append(f"{path} is not charge neutral for {template}: total oxidation charge {charge}")

    return parsed_roles, probe_id or None, errors


def validate_materializer_payload(
    payload: Any,
    *,
    target_count: int = 10,
    max_sites: int = 80,
    seed: int = 0,
    known_formulas: set[str] | None = None,
    min_template_diversity: int = 2,
) -> MaterializerValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, Mapping):
        return MaterializerValidation(False, "invalid", ["payload must be a JSON object"])
    status = str(payload.get("status") or "").strip()
    if status == "hypothesis_conflict":
        if not _string_list(payload.get("conflicting_hypothesis_ids")):
            errors.append("hypothesis_conflict.conflicting_hypothesis_ids must be non-empty")
        if not str(payload.get("reason") or "").strip():
            errors.append("hypothesis_conflict.reason is required")
        if not str(payload.get("minimal_fix_needed") or "").strip():
            errors.append("hypothesis_conflict.minimal_fix_needed is required")
        return MaterializerValidation(not errors, status, errors, warnings)
    if status != "ok":
        errors.append('payload.status must be "ok" or "hypothesis_conflict"')
        return MaterializerValidation(False, status or "invalid", errors, warnings)

    probes = payload.get("formula_probes")
    if not isinstance(probes, list):
        return MaterializerValidation(False, status, ["formula_probes must be a list"])
    if len(probes) != target_count:
        errors.append(f"formula_probes must contain exactly {target_count} probes, got {len(probes)}")

    probe_ids: set[str] = set()
    templates: list[str] = []
    for index, probe in enumerate(probes):
        _, probe_id, probe_errors = validate_probe(probe, index)
        errors.extend(probe_errors)
        if probe_id:
            if probe_id in probe_ids:
                errors.append(f"formula_probes[{index}].id duplicates {probe_id!r}")
            probe_ids.add(probe_id)
        if isinstance(probe, Mapping):
            template = str(probe.get("template") or "").strip()
            if template in BUILDERS:
                templates.append(template)

    if len(set(templates)) < min_template_diversity:
        errors.append(
            f"formula_probes must use at least {min_template_diversity} distinct templates; "
            f"got {sorted(set(templates))}"
        )
    if errors:
        return MaterializerValidation(False, status, errors, warnings)

    strategy = {"formula_probes": [dict(probe) for probe in probes]}
    candidates = load_formula_probes(strategy, max_sites=max_sites, base_seed=seed, known_formulas=known_formulas)
    formulas = [reduced_formula(candidate.structure) for candidate in candidates]
    template_counts = dict(Counter(candidate.template for candidate in candidates))
    if len(candidates) != target_count:
        errors.append(
            f"generator accepted {len(candidates)} formula_probes, expected {target_count}. "
            "Likely causes: duplicate reduced formula, known training composition, invalid geometry, "
            "or unsupported role chemistry."
        )
    if len(set(formulas)) != len(formulas):
        errors.append("generator produced duplicate reduced formulas")
    return MaterializerValidation(
        ok=not errors,
        status=status,
        errors=errors,
        warnings=warnings,
        strategy=strategy,
        generated_count=len(candidates),
        formulas=formulas,
        template_counts=template_counts,
    )
