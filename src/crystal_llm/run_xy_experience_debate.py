"""Run X/Y experience-driven blind material generation debates.

This runner is deliberately separate from the A/B/C/D/E/F discovery loop.
It measures whether the accumulated principle book helps an LLM generate
candidate materials before any new e_hull/SUN results are visible.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys
import time
import warnings
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from pymatgen.core import Composition, Structure
from pymatgen.core.periodic_table import Element

from crystal_llm.chemistry import make_ion
from crystal_llm.filters import load_known_formulas, reduced_formula, validate_structure
from crystal_llm.generate import load_formula_probes
from crystal_llm.hypothesis_schema import (
    ALLOWED_TEMPLATES,
    TEMPLATE_ROLE_COUNTS,
    required_roles,
    schema_reference_json as generator_schema_reference_json,
)
from crystal_llm.llm_client import LLMConfig, LLMError, extract_json_object, make_llm_client
from crystal_llm.local_agent_runtime import LocalAgentRuntime
from crystal_llm.material_physics_schema import query_matches, select_matches, validate_query
from crystal_llm.memory import STRICT_SUN_NOTE, read_json, write_json
from crystal_llm.run_material_physics_mvp import (
    DEFAULT_AGENT_MAX_STEPS,
    DEFAULT_AGENT_MAX_TOOL_CALLS,
    MATERIAL_PHYSICS_DIRECTIVE,
    command_text,
    default_ppd_path,
    default_training_data,
    json_ok,
    load_candidate_pool,
    load_e_hull_rows,
    pool_digest,
    run_analysis,
    run_evaluator,
    utc_now,
)
from crystal_llm.templates import BUILDERS


DEFAULT_CANDIDATE_COUNT = 100
DEFAULT_MAX_DEBATE_ROUNDS = 6
DEFAULT_MIN_DEBATE_ROUNDS = 3
DEFAULT_SHARDS = 20
DEFAULT_OVERSAMPLE = 1.25
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CRITIC_COUNTERPROPOSAL_AFTER = 1
DEFAULT_GENERATOR_BACKEND = "template"
DEFAULT_MATTERGEN_ROOT = os.environ.get("MATTERGEN_ROOT", "external/mattergen")
DEFAULT_MATTERGEN_VENV = os.environ.get("MATTERGEN_VENV", ".venv-mattergen")
DEFAULT_MATTERGEN_MODEL_PATH = os.environ.get("MATTERGEN_MODEL_PATH", "external/mattergen/checkpoints/chemical_system_energy_above_hull")
DEFAULT_MATTERGEN_CHECKPOINT = "chemical_system_energy_above_hull"
DEFAULT_MATTERGEN_TARGET_COUNT = 4
DEFAULT_MATTERGEN_BATCH_SIZE = 8
DEFAULT_MATTERGEN_NUM_BATCHES = 2
DEFAULT_MATTERGEN_HARD_TARGET_MIN_NUM_BATCHES = 8
DEFAULT_MATTERGEN_HARD_TARGET_MAX_EXACT_ELEMENTS = 3
DEFAULT_MATTERGEN_MAX_SITES = 20
DEFAULT_MATTERGEN_MIN_VOLUME_PER_ATOM = 4.0
DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM = 45.0
DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM = 43.0
DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR = 1.0
DEFAULT_MATTERGEN_ALLOW_HARD_TARGET_FORMULA = False
XY_CONTEXT_FAILED_OR_USED_FORMULA_LIMIT = 12
XY_CONTEXT_FAILED_OR_USED_FORMULA_HINT_LIMIT = 32
XY_CONTEXT_FAILED_VOLUME_BOUNDARY_LIMIT = 2
XY_CONTEXT_MATTERGEN_LEGACY_BASIN_LIMIT = 4
DEFAULT_MATTERGEN_PARTITION = "8-5090"
DEFAULT_MATTERGEN_GRES = "gpu:1"
DEFAULT_MATTERGEN_CPUS_PER_TASK = 16
DEFAULT_MATTERGEN_MODULE_INIT = ""
DEFAULT_MATTERGEN_MODULES = ""
DEFAULT_MATTERGEN_CUDA_HOME = ""
MATTERGEN_SLURM_FATAL_PENDING_REASONS = (
    "launch_failed",
    "held",
    "jobheld",
    "bad_constraints",
    "invalid_account",
)
XY_DENSITY_EDGE_TRIGGER_PATTERNS = (
    "high-volume",
    "high volume",
    "volume penalty",
    "volume risk",
    "volume ceiling",
    "volume_per_atom",
    "max_volume_per_atom",
    "open-framework",
    "too open",
)
DEFAULT_FORBIDDEN_SELECTION_FIELDS = {
    "e_hull",
    "energy_above_hull",
    "is_stable",
    "sun",
    "stable_count",
    "sun_score",
    "formation_energy_per_atom",
}
DEFAULT_ALLOWED_PREFERRED_ORDERS = {"random", "material_id", ""}
GENERATED_FORMULA_ERROR_RE = re.compile(r"\bgenerated\s+([A-Z][A-Za-z0-9()]+)\b")
GENERATED_FORMULA_EXCLUDED_RE = re.compile(r"\bgenerated\s+formula\s+([A-Z][A-Za-z0-9()]+)\b")
CHEMICAL_SYSTEM_ERROR_RE = re.compile(
    r"\bchemical_system(?:\s*[:=]|\s+)(\[?[A-Z][A-Za-z0-9'\",_\-\s\[\]]{1,120}\]?)"
)
RECOVERABLE_LLM_EXIT_CODE = 75
RECOVERABLE_LLM_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
RECOVERABLE_LLM_ERROR_PATTERNS = (
    "urlopen error",
    "timed out",
    "read operation timed out",
    "connection reset",
    "connection refused",
    "incompleteread",
    "remote end closed connection",
    "stream disconnected before completion",
    "error sending request for url",
    "transport channel closed",
    "temporarily unavailable",
    "server_is_overloaded",
    "currently overloaded",
    "codex cli timed out",
    "timeout waiting for child process to exit",
    "try again later",
)
XY_STRATEGY_COOLDOWN_RECENT_RECORD_LIMIT = 24
XY_STRATEGY_COOLDOWN_VISIBLE_LIMIT = 6
XY_STRATEGY_COOLDOWN_EXACT_FAILURE_THRESHOLD = 1
XY_STRATEGY_COOLDOWN_FAMILY_DISTINCT_SYSTEM_THRESHOLD = 2
XY_STRATEGY_COOLDOWN_ERROR_MARKERS = (
    "known/training formula",
    "duplicate reduced_formula",
    "duplicate_reduced_formula",
    "generator materialized 0 records",
    "mattergen materialized 0 records",
    "no_accepted_structures",
    "chemical_system_not_exact",
    "outside_chemical_system",
    "outside_allowed_elements",
    "missing_required_elements",
    "not_target_reduced_formula",
    "strategy_cooldown_block",
)
XY_MATTERGEN_SATURATION_MIN_REJECTIONS = 8
XY_MATTERGEN_SATURATION_EXCLUDED_RATIO = 0.75


class RecoverableLLMFailure(RuntimeError):
    """Provider or transport failure that should pause and resume the run."""

    def __init__(
        self,
        *,
        role: str,
        metadata: Mapping[str, Any],
        error: str,
        attempts: int,
    ) -> None:
        super().__init__(error)
        self.role = role
        self.metadata = dict(metadata)
        self.error = error
        self.attempts = attempts


class JSONOutputRepairFailure(ValueError):
    """Model returned non-JSON text after the local JSON repair budget."""

    def __init__(
        self,
        *,
        role: str,
        metadata: Mapping[str, Any],
        error: str,
        attempts: int,
        last_text: str,
    ) -> None:
        super().__init__(f"LLM output for {role} could not be repaired: {error}")
        self.role = role
        self.metadata = dict(metadata)
        self.error = error
        self.attempts = attempts
        self.last_text = last_text


def is_recoverable_llm_error_message(message: str) -> bool:
    lowered = str(message).lower()
    http_match = re.search(r"llm http\s+(\d{3})", lowered)
    if http_match:
        return int(http_match.group(1)) in RECOVERABLE_LLM_HTTP_STATUS_CODES
    return any(pattern in lowered for pattern in RECOVERABLE_LLM_ERROR_PATTERNS)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def recoverable_llm_pause_payload(
    *,
    exc: RecoverableLLMFailure,
    args: argparse.Namespace,
    work_dir: Path,
    memory_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "xy_recoverable_llm_pause.v1",
        "created_at_utc": utc_now(),
        "status": "recoverable_llm_failure",
        "exit_code": RECOVERABLE_LLM_EXIT_CODE,
        "role": exc.role,
        "stage": exc.metadata.get("stage"),
        "iteration": exc.metadata.get("iteration"),
        "description_attempt": exc.metadata.get("description_attempt"),
        "cycle": exc.metadata.get("cycle"),
        "repair_round": exc.metadata.get("repair_round"),
        "attempts": exc.attempts,
        "error": exc.error,
        "work_dir": str(work_dir),
        "memory_path": str(memory_path),
        "memory_record_count": len(read_sequential_memory(memory_path).get("records", [])) if memory_path.exists() else 0,
        "resume_seed": str(memory_path),
        "controller_args": {key: value for key, value in vars(args).items() if "key" not in key.lower()},
        "resume_note": (
            "This is a transient provider/transport failure. Resume by using memory_path as "
            "--seed-sequential-memory in a fresh work-dir after relay smoke succeeds."
        ),
    }


XY_HISTORY_RAG_REQUIREMENT = """MANDATORY_HISTORY_RAG:
- Controller-managed local RAG evidence is supplied before final JSON; do not output tool requests, `rag_required`, or requests for raw datasets.
- The required evidence page is summarize_mechanism_evidence with {"limit": 12, "offset": 0, "include_failed": true}.
- If the first page has has_more=true and the shard depends on more history, request another page with a different offset.
- Use the complete principle-book lessons: validated mechanisms, rejected mechanisms, failure boundaries, residual risks, control-derived leads, and near misses.
- Distill retrieved evidence into cited_principle_ids, principle_use, mechanism_rationale, risk_boundaries, and audit fields. Do not paste raw tool output.
"""


XY_SEQUENTIAL_RAG_REQUIREMENT = """MANDATORY_XY_SEQUENTIAL_RAG:
- Controller injects compact A/B evidence and X/Y history; do not request tools/raw data.
- Binding mode: CONTEXT_JSON.controller_constraints.search_policy.current_search_mode. `acquisition_mode` is this controller budget label, not a free-form chemistry novelty label. Avoid reduced_formula repeats unless control_candidate_requested=true.
"""


XY_BLIND_EVALUATION_POLICY = """XY_BLIND_EVALUATION_POLICY:
- This is a blind evaluation of material-design ability after reading the existing experience book.
- Do not ask for, infer, or use new e_hull/SUN results for proposed candidates before the final 100 candidates are locked.
- The controller must not preselect high-SUN principles for you. You receive the full experience book, including rejected and boundary entries.
- You may learn from SUN-active experiences only through their stated microscopic mechanism and boundaries. Do not rank candidates by historical SUN count, stable_count, sun_score, or known e_hull.
- Do not choose candidates because candidate-pool rows have low formation_energy_per_atom or because a tool example appears energy-ranked.
- Candidate selection must be mechanism-driven and executable. Use candidate-pool tools for availability and exact IDs only, not for stability cherry-picking.
- Use rejected principles and failure boundaries as negative design knowledge, not as material generators.
- Avoid duplicate formula/material replay unless it is explicitly a boundary-control candidate; repeated candidates reduce evaluation value.
- The controller can enforce reduced_formula hard de-duplication: when enabled, only the first candidate with a given reduced_formula survives materialization.
"""


XY_SEQUENTIAL_OPTIMIZATION_POLICY = """XY_SEQUENTIAL_OPTIMIZATION_POLICY:
- Goal: maximize strict SUN hit rate; avoid duplicates/evaluator-null/blocked routes.
- X/Y output one natural-language material_description, not probes, MP ids/queries, or structures.
- Include chemical_system/family, elements/valence, motif, generator_template, SUN rationale.
- Follow search_policy 70/20/10: exploit, explore_adjacent, orthogonal_jump.
- `acquisition_mode` must match current_search_mode. If a stale next_strategy formula is blocked by cooldown, a legal same-mechanism replacement still uses the current mode label after auditing the block.
- MatterGen uses chemical_system/filters; legacy templates are hints, not hard gates.
- Keep at least two legal ranked queue items unless no_valid_material_description.
- Repeats/controls require control_candidate_requested=true.
"""


XY_SEQUENTIAL_STRATEGY_BOUNDARY_POLICY = """BINDING_XY_STRATEGY_BOUNDARY_POLICY:
- latest_xy_strategy_constraints is binding: next_strategy default; failure_boundaries reject.
- search_policy may supersede stale latest strategy when current_search_mode or blocked route conflicts.
- no_valid_material_description must audit every next_strategy_candidate_formulas item.
- Reject repeats/controls/boundary probes unless control_candidate_requested=true.
- Candidate-list exhaustion is not basin-level failure; same basin needs two new legal formulas.
"""

XY_SEQUENTIAL_MACHINE_STRATEGY_POLICY = """BINDING_XY_MACHINE_STRATEGY_CONSTRAINTS:
- CONTEXT_JSON.controller_constraints.strategy_constraints is controller-generated and binding.
- Every sun_candidate_queue item must set acquisition_mode equal to strategy_constraints.required_acquisition_mode when present.
- If strategy_constraints.latest_strategy_order_enforced=true, the queue must start with strategy_constraints.first_required_formula and include the legal_ordered_candidate_formulas visible in the constraint before inventing lower-priority alternatives.
- legal_ordered_candidate_formulas already excludes formulas blocked by failed/used history, generator feedback, evaluator-null elements, or cooldowns. Do not require blocked_candidate_formula_reasons formulas in the queue; mention them only as audited blocked formulas.
- CONTEXT_JSON.controller_constraints.failed_or_used_reduced_formulas is a compact recent/system-relevant hint list, not the full set. If controller feedback names a hidden failed/used formula, remove it and revise the whole queue rather than arguing from the shorter visible list.
- If fewer than two legal_ordered_candidate_formulas remain, do not select a singleton. Return no_valid_material_description with an impossibility_certificate unless search_policy explicitly supersedes the route.
- For MatterGen, reduced_formula is a soft target for the request, but it is still the X/Y strategy target and must be copied into the selected queue item and material_description.
"""


XY_SEQUENTIAL_FINAL_POLICY_REMINDER = """XY_SEQUENTIAL_FINAL_POLICY_REMINDER:
- Lock only the queue/material already reviewed by X/Y; do not introduce a new route in finalization.
- Recheck CONTEXT_JSON.controller_constraints.search_policy.current_search_mode, cooled_templates, failed_or_used_reduced_formulas, and latest_xy_strategy_constraints before finalizing.
- If the reviewed mapping is blocked by Z/W feedback, cooled policy, duplicate memory, failed volume/template boundaries, or stale latest strategy conflict that is not superseded by search_policy, return no_valid_material_consensus rather than finalizing it.
- If search_policy superseded stale latest_xy_strategy_constraints during X/Y debate, finalize the reviewed current-search-policy queue; do not revert to no_valid only because the stale route was exhausted.
- Keep selected_candidate_id, selection_rationale, and material_description consistent with the selected queue item.
"""

XY_SEQUENTIAL_COMPACT_DEBATE_POLICY = """BINDING_XY_STRATEGY_BOUNDARY_POLICY:
- Goal: one natural-language material_description queue; no generator strings, structures, MP ids, or MP queries.
- search_policy 70/20/10 controls the route. latest_xy_strategy_constraints is default only until search_policy supersedes a stale/cooled/boundary-blocked route.
- Avoid failure_boundaries, failed_or_used formulas, and forbidden_evaluator_null_elements unless control_candidate_requested=true.
- no_valid_material_description requires an impossibility_certificate auditing every next_strategy_candidate_formulas item.
"""


XY_GENERATOR_ONLY_POLICY = """XY_GENERATOR_ONLY_POLICY:
- X/Y/Z/W must not select, cite, request, or design around MP-pool rows, MP material_ids, candidate-pool examples, or MP queries.
- Do not call query_candidate_pool. The MP pool is unavailable for final concrete candidate design in this run.
- Final candidates must be generated materials: source must be "generator".
- Prefer formula_probe_strings: compact strings parsed by the controller and sent to the local material generator.
- When CONTEXT_JSON.generator_backend is "mattergen", prefer a MatterGen request over hand-built cells: specify properties_to_condition_on.chemical_system, properties_to_condition_on.energy_above_hull=0.0, allowed elements, max_sites, and diversity filters. Do not put chemical_system or energy_above_hull at the MatterGen request top level. Do not invent lattice vectors or fractional coordinates for MatterGen.
- Use structure_dicts only when the existing formula-probe templates cannot express the intended chemistry or local motif.
- When GENERATOR_TEMPLATE_ONLY_CONSTRAINT is present, structure_dicts are forbidden; choose only formula_probe_strings or formula_probes from the allowed template library.
- The controller will reject mp_pool, material_ids, query-only candidates, formula/e_hull/stability prefilters, and MP replay.
"""


XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY = """XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY:
- Before proposing a candidate, perform a mechanism-to-template consistency audit. The audit is required for every candidate.
- The chosen generator template must physically realize the microscopic motif claimed in mechanism_rationale. Do not use a template as a generic formula container.
- If the principle requires a local motif that no formula-probe template can represent, either provide a faithful structure_dict or reject that candidate. Do not force the chemistry into perovskite, spinel, or delafossite as a proxy.
- A template is faithful only when its built-in topology/coordination is the local motif being tested. It is invalid to justify a template as a "stoichiometry vehicle", "stoichiometry proxy", "formula container", "role container", or any wording that admits the template may not capture the claimed topology.
- Examples of invalid proxy mappings: PSx/thiophosphate tetrahedral-network mechanisms into perovskite ABX3; compact Ca-Si-N nitridosilicate/Si-N network mechanisms into delafossite or perovskite ABX2/ABX3; borate/silicate framework mechanisms into binary fluoride endpoints as positive evidence.
- Boundary/control candidates are allowed only when explicitly labeled as boundary tests and when the template actually realizes the boundary motif being tested.
- Agent Y must reject candidates whose template_consistency_audit is missing, says the template does not realize the motif, or admits an unsupported motif substitution.
- The controller will reject generator candidates whose template_consistency_audit is missing or internally inconsistent before materialization.
"""


XY_DESIGN_BOOK_POLICY = """XY_DESIGN_BOOK_POLICY:
- X/Y do not propose concrete candidate materials in the first stage.
- First convert A/B's general principle book into a shard-local SUN design experience book: material features that could become SUN without violating the validated microscopic principles and known failure boundaries.
- Each design rule must be mechanistically concrete enough for a later Z/W generator to act on: specify allowed element families, oxidation-state patterns, local motifs, coordination/polyhedra, allowed generator templates, forbidden proxy templates, charge-balance constraints, diversity constraints, and expected failure modes.
- Do not rank by historical SUN count, stable_count, sun_score, e_hull, or formation energy. Historical results may be used only as evidence for microscopic mechanisms and boundaries.
- Y must reject generic restatements such as "use stable structures" or "try perovskites"; every rule must explain why the structure features follow from A/B's principle-level mechanism.
- The final X/Y output is a design_experience_book, not candidate_specs.
"""


XY_DESIGN_BOOK_SCHEMA = """Return JSON only.
Design rule schema:
{
  "design_rule_id": "xy_s001_r001",
  "source_principle_ids": ["principle_program_015"],
  "principle_interpretation": "how the A/B principle is concretized for generation",
  "sun_favorable_material_features": ["specific material features expected to improve strict SUN probability"],
  "allowed_element_families": {
    "cations": ["element families or explicit element sets"],
    "anions": ["element families or explicit element sets"],
    "substitution_axes": ["allowed substitutions and why"]
  },
  "oxidation_state_patterns": ["charge-balanced valence patterns"],
  "required_local_motifs": ["coordination/polyhedra/network motifs required by the mechanism"],
  "allowed_generator_templates": [
    {
      "template": "rocksalt|cesium_chloride|zincblende|wurtzite|fluorite|rutile|antifluorite|perovskite|double_perovskite|spinel|corundum|delafossite|structure_dict",
      "faithful_motif_reason": "why this template realizes the motif"
    }
  ],
  "forbidden_proxy_templates": [
    {"template": "perovskite", "reason": "why this would be an unfaithful proxy for the mechanism"}
  ],
  "charge_balance_rule": "specific charge-balance requirement",
  "diversity_constraints": ["how Z/W should avoid repeated formula families"],
  "negative_boundaries": ["known failure modes and rejected substitutions"],
  "zw_generation_instructions": ["direct instructions Z/W must follow"],
  "mechanism_confidence": "high|medium|low",
  "audit_questions_for_zw": ["questions Z/W must answer for every concrete candidate"]
}
Rules:
- At least one design_rule_id must be usable by Z/W for every later candidate.
- Do not include formula_probe_strings, formula_probes, material_ids, MP queries, structure_dicts, or concrete material candidates in X/Y's design_experience_book.
- If a principle is too vague to generate from, convert it into constraints and boundaries rather than a candidate.
"""


ZW_GENERATION_POLICY = """ZW_GENERATION_POLICY:
- Z/W receive A/B's principle book plus X/Y's design_experience_book. Z/W must generate concrete materials only by applying X/Y design rules.
- Every candidate must cite at least one design_rule_id and at least one source principle id. Candidates without design_rule_ids are rejected by the controller in two-stage mode.
- Z must translate each design rule into diverse generator inputs; W must reject candidates that violate the design rule, copy historical formulas without a mechanism reason, or use an unfaithful template proxy.
- The candidate's template_consistency_audit must explicitly answer the design rule's audit_questions_for_zw where applicable.
- Z/W are not allowed to revise X/Y's material principle. If a design rule is not executable, mark it as unusable for this shard and choose another rule or report underfill.
- The output remains blind: no new e_hull/SUN/stability/formation-energy labels may be requested or used before candidate lock.
"""


XY_CANDIDATE_SCHEMA = """Return JSON only.
Candidate object schema:
{
  "id": "xy_s001_c001",
  "source": "generator",
  "count": 1,
  "formula_probe_strings": ["template=perovskite;A=Ba:+2;B=Hf:+4;X=O:-2;family=oxide_perovskite"],
  "query": {},
  "material_ids": [],
  "exclude_formulas": [],
  "formula_probes": [
    {
      "id": "xy_s001_c001_probe",
      "template": "perovskite",
      "family": "oxide_perovskite",
      "roles": {
        "A": {"element": "Ba", "oxidation_state": 2},
        "B": {"element": "Hf", "oxidation_state": 4},
        "X": {"element": "O", "oxidation_state": -2}
      }
    }
  ],
  "mattergen_requests": [
    {
      "backend": "mattergen",
      "request_id": "xy_s001_c001_mattergen",
      "target_count": 4,
      "batch_size": 8,
      "num_batches": 1,
      "properties_to_condition_on": {
        "chemical_system": "Ce-H",
        "energy_above_hull": 0.0
      },
      "filters": {
        "chemical_system": ["Ce", "H"],
        "require_chemical_system_exact": false,
        "max_sites": 20,
        "deduplicate_reduced_formula": true
      }
    }
  ],
  "structure_dicts": [],
  "design_rule_ids": ["xy_s001_r001"],
  "cited_principle_ids": ["principle_program_015"],
  "principle_use": "how this candidate uses one or more experience-book lessons",
  "mechanism_rationale": "observable microscopic design rationale; no hidden chain-of-thought",
  "expected_structure_features": ["concise observable features"],
  "template_consistency_audit": {
    "mechanism_local_motif": "specific local motif required by the cited principle",
    "required_coordination_or_polyhedra": ["observable coordination/polyhedra/network requirement"],
    "chosen_template": "perovskite",
    "template_realizes_motif": true,
    "unsupported_motif_substitution": false,
    "structure_dict_required": false,
    "why_template_is_faithful": "why the chosen generator template implements the motif rather than merely matching stoichiometry",
    "generator_limitations": ["remaining limitations, or []"]
  },
  "risk_boundaries": ["which known boundaries or rejected mechanisms were considered"],
  "selection_policy": "generator_probe|generator_structure"
}
Rules:
- Each candidate must count as exactly one material after materialization.
- Source must be "generator"; mp_pool, material_ids, MP queries, and candidate-pool selection are forbidden.
- Provide exactly one generator input per candidate: one formula_probe_string, one structured formula_probe, or one parseable pymatgen Structure.as_dict().
- If CONTEXT_JSON.generator_backend is "mattergen", provide exactly one mattergen_requests object instead. Do not also include formula_probes or structure_dicts.
- MatterGen request must condition through properties_to_condition_on.chemical_system and properties_to_condition_on.energy_above_hull=0.0; include diffusion_guidance_factor=1.0 so the conditional chemical-system guidance is active. Do not put chemical_system or energy_above_hull at the request top level. filters.chemical_system must match the conditioned elements. It may generate varied stoichiometries inside that system; target_reduced_formula is only a soft preference unless require_target_reduced_formula=true.
- formula_probe_string format is semicolon-separated: template=<allowed_template>;Role=Element:+oxidation_state;Role2=Element:-oxidation_state;family=<short_family>.
- Allowed templates are rocksalt, cesium_chloride, zincblende, wurtzite, fluorite, rutile, antifluorite, perovskite, double_perovskite, spinel, corundum, and delafossite.
- Roles must match the template: e.g. rocksalt needs A and X; perovskite needs A, B, X; double_perovskite needs A, B, B2, X; spinel needs A, B, X.
- Use positive oxidation states for cation roles and negative oxidation states for X.
- To avoid canonical formula replay, put formulas in top-level exclude_formulas. Do not invent query keys such as exclude_formula_probes.
- Cite at least one principle id in experience mode unless the candidate is a no-experience baseline.
- In two-stage mode, cite at least one X/Y design_rule_id in design_rule_ids; the controller rejects candidates missing it.
- template_consistency_audit is mandatory. The controller rejects candidates when chosen_template does not match the formula_probe template, template_realizes_motif is not true, unsupported_motif_substitution is true, or the local motif/coordination explanation is empty.
"""


GENERATOR_AUDIT_REQUIRED_KEYS = (
    "chosen_template",
    "template_realizes_motif",
    "unsupported_motif_substitution",
    "mechanism_local_motif",
    "required_coordination_or_polyhedra",
    "why_template_is_faithful",
)


COMMON_ALKALI_ELEMENTS = {"Li", "Na", "K", "Rb", "Cs", "Fr"}
COMMON_ALKALINE_EARTH_ELEMENTS = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
COMMON_HALOGEN_ELEMENTS = {"F", "Cl", "Br", "I"}
COMMON_CHALCOGEN_ELEMENTS = {"O", "S", "Se", "Te"}
COMMON_PNICTOGEN_ELEMENTS = {"N", "P", "As", "Sb", "Bi"}
COMMON_GROUP_12_ELEMENTS = {"Zn", "Cd", "Hg"}
COMMON_ANION_ELEMENTS = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "Te", "I"}
EVALUATOR_NULL_E_HULL_ELEMENTS = {
    "Ra",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
}


def generator_template_only_rules() -> dict[str, Any]:
    return {
        "generator_template_only": True,
        "allowed_templates": list(ALLOWED_TEMPLATES),
        "role_requirements": {template: list(required_roles(template)) for template in ALLOWED_TEMPLATES},
        "role_stoichiometry": {template: dict(TEMPLATE_ROLE_COUNTS[template]) for template in ALLOWED_TEMPLATES},
        "template_faithfulness_rule": (
            "The template topology/coordination must be the claimed local motif. Reject any material description "
            "or Z/W candidate that uses an allowed template only as a stoichiometry vehicle/proxy/container."
        ),
        "x_y_material_description_required_fields": {
            "generator_template": "exactly one allowed template name",
            "generator_role_mapping": {
                "Role": {
                    "element": "element symbol assigned to this generator role",
                    "oxidation_state": "integer oxidation state used for charge balance",
                }
            },
            "template_formula_family": "short family label, not a generator string",
            "why_template_is_faithful": "why this template physically realizes the intended motif",
        },
        "z_w_generator_inputs_allowed": ["formula_probe_strings", "formula_probes"],
        "z_w_generator_inputs_forbidden": ["structure_dicts", "structure_dict"],
    }


def generator_template_role_summary() -> str:
    return "; ".join(f"{template}({','.join(required_roles(template))})" for template in ALLOWED_TEMPLATES)


def generator_template_only_prompt_block() -> str:
    return (
        "GENERATOR_TEMPLATE_ONLY_CONSTRAINT:\n"
        "- Custom structures are forbidden. Choose one allowed template and map every required role.\n"
        "- Required X/Y fields: generator_template, generator_role_mapping, template_formula_family, expected_local_motif, why_template_is_faithful.\n"
        "- The generator_role_mapping must be charge-neutral under the template role stoichiometry; do not merely claim charge balance.\n"
        "- Template faithfulness requires topology/coordination fidelity: reject any route where the template is only a stoichiometry vehicle, proxy, formula container, or generic role container.\n"
        "- In output JSON, state template faithfulness positively; avoid negated reject-label wording such as saying the template is not a proxy/container.\n"
        "- Z/W output only formula_probe_strings or formula_probes; structure_dicts/structure_dict are invalid.\n"
        "- If the best A/B lesson needs an unsupported topology or stoichiometry, choose a faithful in-template analog or a different principle.\n"
        f"- Allowed template roles: {generator_template_role_summary()}.\n"
    )


def mattergen_xy_prompt_block(context: Mapping[str, Any]) -> str:
    if str(context.get("generator_backend") or DEFAULT_GENERATOR_BACKEND) != "mattergen":
        return ""
    controller_constraints = context.get("controller_constraints")
    operational_status = (
        controller_constraints.get("mattergen_operational_status")
        if isinstance(controller_constraints, Mapping)
        else None
    )
    status_line = ""
    if isinstance(operational_status, Mapping) and bool(operational_status.get("verified")):
        status_line = (
            "- Stale backend/CUDA/no-kernel repair gates in historical postmortems are superseded.\n"
        )
    return (
        "\nMATTERGEN_BACKEND_XY_CONSTRAINT:\n"
        "- Backend is MatterGen, not formula-probe templates; set generator_template=\"mattergen\".\n"
        f"{status_line}"
        "- Template gates are disabled; legacy template facts are chemistry hints.\n"
        "- Specify chemical_system, required/forbidden elements, motif/valence constraints, exclusions, and SUN rationale.\n"
        "- All chemical_system elements are required unless allow_subset_fallback=true; reduced_formula is soft by default.\n"
        "- MatterGen filters.chemical_system must be a list or hyphen string of element symbols, never an object/dict; put allowed_elements and required_elements in their own list fields.\n"
        "- CONTEXT_JSON.controller_constraints.strategy_cooldowns is hard: leave blocked chemical_systems and blocked family_patterns by changing the chemistry basin, not only by adding excluded formulas.\n"
        f"- For dense/compact or high-volume-risk routes, state max_volume_per_atom <= {DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM:g}.\n"
        "- No manual cells/coordinates/structure_dicts.\n"
    )


def xy_search_policy_instruction(context: Mapping[str, Any], *, reviewer: bool = False) -> str:
    if str(context.get("generator_backend") or DEFAULT_GENERATOR_BACKEND) == "mattergen":
        if reviewer:
            return (
                "Approve only if X obeys current_search_mode as a MatterGen chemical-system acquisition budget: "
                "the JSON acquisition_mode must equal the active current_search_mode. Do not relabel a legal "
                "same-mechanism cooldown escape as explore_adjacent when the active mode is exploit; instead audit "
                "which stale next_strategy formulas are blocked. No template hard gates."
            )
        return (
            "Use current_search_mode as the MatterGen chemical-system acquisition budget; every queue item's "
            "acquisition_mode must equal that active mode. In exploit mode, choose the best legal continuation of "
            "the successful mechanism; if the named next_strategy formulas are blocked by cooldown, audit that block "
            "and keep acquisition_mode='exploit' for the legal same-mechanism replacement. In explore_adjacent mode, "
            "make a one-axis change; in orthogonal_jump mode, use a new basin. MatterGen has no template hard gates."
        )
    if reviewer:
        return (
            "Approve only if X obeys CONTEXT_JSON.controller_constraints.search_policy.current_search_mode. "
            "In exploit mode, the selected material and first two legal queue items must stay inside "
            "preferred_exploitation_basins or exploit_template_allowlist. In explore_adjacent mode, reject cooled "
            "templates and require a one-axis variant near a success/near-success basin. In orthogonal_jump mode, "
            "reject proposals that merely continue the same recent basin instead of using preferred_jump_templates "
            "or another underexplored non-cooled native template."
        )
    return (
        "Before ranking, read CONTEXT_JSON.controller_constraints.search_policy. Set each queue item's "
        "acquisition_mode to the current_search_mode unless it is a clearly justified fallback. In exploit mode, "
        "candidate 1 and candidate 2 must come from preferred_exploitation_basins or exploit_template_allowlist. "
        "In explore_adjacent mode, keep one successful/near-success basin mechanism and change one axis while "
        "avoiding cooled templates. In orthogonal_jump mode, use preferred_jump_templates or an explicitly "
        "underexplored non-cooled native template."
    )


def mattergen_backend_rules() -> dict[str, Any]:
    return {
        "generator_backend": "mattergen",
        "candidate_contract": {
            "source": "generator",
            "count": 1,
            "exactly_one_input_field": ["mattergen_requests"],
        },
        "mattergen_requests": {
            "type": "list[object]",
            "required_shape": {
                "backend": "mattergen",
                "request_id": "stable id",
                "properties_to_condition_on": {
                    "chemical_system": "Element-Element or Element-Element-Element",
                    "energy_above_hull": 0.0,
                },
                "filters": {
                    "chemical_system": ["same elements as chemical_system"],
                    "require_chemical_system_exact": True,
                    "allowed_elements": ["same elements as chemical_system"],
                    "required_elements": ["same elements as chemical_system"],
                    "max_sites": DEFAULT_MATTERGEN_MAX_SITES,
                    "min_volume_per_atom": DEFAULT_MATTERGEN_MIN_VOLUME_PER_ATOM,
                    "max_volume_per_atom": DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM,
                    "deduplicate_reduced_formula": True,
                    "require_target_reduced_formula": False,
                },
                "diffusion_guidance_factor": DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR,
            },
            "optional_fields": {
                "target_count": DEFAULT_MATTERGEN_TARGET_COUNT,
                "batch_size": DEFAULT_MATTERGEN_BATCH_SIZE,
                "num_batches": DEFAULT_MATTERGEN_NUM_BATCHES,
                "target_reduced_formula": "soft preference by default; hard filtering requires filters.require_target_reduced_formula=true and at least the hard-target sampling budget",
                "exclude_reduced_formulas": "controller fills historical duplicates; Z/W may add known bad formulas",
            },
            "hard_target_rule": (
                "Sequential X/Y MatterGen blind SUN search is controller-forced to "
                "require_target_reduced_formula=false; target_reduced_formula is a soft preference. "
                "If a hard target is unavoidable outside this blind-search controller path, "
                f"use it only for chemical systems with at most {DEFAULT_MATTERGEN_HARD_TARGET_MAX_EXACT_ELEMENTS} elements "
                f"and at least num_batches={DEFAULT_MATTERGEN_HARD_TARGET_MIN_NUM_BATCHES}; after not_target_reduced_formula "
                "or no_accepted_structures feedback, do not switch to another sibling hard target in the same basin."
            ),
        },
        "template_consistency_audit_required_keys": list(GENERATOR_AUDIT_REQUIRED_KEYS),
        "template_consistency_audit_required_shape": {
            "chosen_template": "mattergen",
            "template_realizes_motif": True,
            "unsupported_motif_substitution": False,
            "mechanism_local_motif": "short description of the local motif or generated full-periodic coordination target",
            "required_coordination_or_polyhedra": "coordination/polyhedra or bonding environment MatterGen should be able to sample",
            "why_template_is_faithful": (
                "MatterGen samples full periodic structures conditioned on the requested chemical system and low hull "
                "energy; it is not a fixed-template proxy or manual cell."
            ),
        },
        "audit_convention": (
            "Use template_consistency_audit.chosen_template='mattergen'. Explain that MatterGen samples a full "
            "periodic structure conditioned on chemical system and low hull energy, rather than using a fixed template."
        ),
        "forbidden_when_mattergen": [
            "formula_probe_strings",
            "formula_probes",
            "structure_dicts",
            "manual lattice vectors",
            "manual fractional coordinates",
            "mp_pool/material_ids/query",
        ],
            "controller_rule": (
                "If MatterGen returns no accepted structures or only duplicates, Z/W must revise the chemical_system, "
                "filters, or selected X/Y description before returning to X/Y. If no_accepted_structures is dominated "
                "by excluded_reduced_formula from the controller history, allow at most one controlled rescue that keeps "
                "hidden exclusions and materially increases sampling; repeated high-duplicate zero-accepted feedback means "
                "the chemical basin is saturated. Removing explicit exclude_reduced_formulas is not a repair because the "
                "controller re-injects hidden history."
            ),
    }


def generator_executable_schema_rules(*, template_only: bool = False, backend: str = DEFAULT_GENERATOR_BACKEND) -> dict[str, Any]:
    """Machine-readable generator contract shared with Z/W after controller failures."""

    if backend == "mattergen":
        return mattergen_backend_rules()

    input_fields = ["formula_probe_strings", "formula_probes"] if template_only else [
        "formula_probe_strings",
        "formula_probes",
        "structure_dicts",
    ]
    rules: dict[str, Any] = {
        "candidate_contract": {
            "source": "generator",
            "count": 1,
            "exactly_one_input_field": input_fields,
        },
        "formula_probe_strings": {
            "type": "list[str]",
            "example": "template=perovskite;A=Ba:+2;B=Hf:+4;X=O:-2;family=oxide_perovskite",
            "format": "semicolon-separated template=<allowed_template>;Role=Element:+/-oxidation_state;family=<short_family>",
            "allowed_templates": list(ALLOWED_TEMPLATES),
            "role_requirements": {template: list(required_roles(template)) for template in ALLOWED_TEMPLATES},
        },
        "formula_probes": {
            "type": "list[object]",
            "required_shape": {
                "template": "one allowed template",
                "roles": {"Role": {"element": "valid element symbol", "oxidation_state": "integer"}},
            },
        },
        "template_consistency_audit_required_keys": list(GENERATOR_AUDIT_REQUIRED_KEYS),
        "template_consistency_hard_rejects": [
            "template described as stoichiometry vehicle/proxy/container",
            "template described as a generic formula or role container",
            "audit admits the template may not capture the claimed local topology",
        ],
        "forbidden_common_aliases": [
            "formula_probe_string",
            "formula_probe",
            "structure_dict",
            "chosen_generator_template",
            "custom_structure_dict_A2CdBr4_tetrahedral_bromocadmate",
        ],
        "controller_rule": (
            "Schema/materialization errors must be repaired by Z/W in the next generator attempt. "
            "Return to X/Y only when the natural-language material description is unrepresentable even with a valid formula_probe or Structure.as_dict()."
        ),
    }
    if template_only:
        rules["template_only"] = generator_template_only_rules()
        rules["structure_dicts"] = {
            "accepted": False,
            "reason": "template-only run forbids custom structures; use formula_probe_strings or formula_probes with an allowed template",
        }
        rules["controller_rule"] = (
            "Schema/materialization errors must be repaired by Z/W in the next generator attempt. "
            "In template-only mode, return to X/Y only when the X/Y material_description cannot be represented by any allowed formula_probe template. "
            "Do not use structure_dicts."
        )
    else:
        rules["structure_dicts"] = {
            "type": "list[pymatgen Structure.as_dict() object]",
            "not_accepted": "Do not use constructor shorthand such as {'lattice': {'matrix': ...}, 'species': ..., 'coords': ...}; pass Structure(...).as_dict() JSON.",
            "validation_window": {
                "max_sites": "controller --max-sites",
                "volume_per_atom_ang3": "4.0 <= volume_per_atom <= 29.5",
                "min_interatomic_distance_ang": ">= 0.75",
            },
        }
    return rules


def generator_executable_schema_reminder(*, template_only: bool = False, backend: str = DEFAULT_GENERATOR_BACKEND) -> str:
    if backend == "mattergen":
        return (
            "EXECUTABLE_GENERATOR_SCHEMA_RULES_FOR_MATTERGEN:\n"
            "- Candidate object: source='generator', count=1, exactly one field \"mattergen_requests\".\n"
            "- MatterGen request: backend='mattergen', properties_to_condition_on.chemical_system='A-B[-C]', "
            "properties_to_condition_on.energy_above_hull=0.0, filters.chemical_system matching those elements, "
            "filters.require_chemical_system_exact=true unless the X/Y material_description explicitly sets allow_subset_fallback=true, "
            "filters.allowed_elements and filters.required_elements containing the same named chemical-system elements, "
            f"filters.max_sites<={DEFAULT_MATTERGEN_MAX_SITES}, deduplicate_reduced_formula=true, "
            f"diffusion_guidance_factor={DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR}, "
            f"min_volume_per_atom={DEFAULT_MATTERGEN_MIN_VOLUME_PER_ATOM}, "
            f"max_volume_per_atom={DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM}.\n"
            "- Do not put chemical_system or energy_above_hull at the MatterGen request top level; the controller rejects top-level selection fields before normalization.\n"
            "- filters.chemical_system must be a list or hyphen string of element symbols, never an object/dict with allowed/required keys; use filters.allowed_elements and filters.required_elements as separate list fields.\n"
            f"- If X/Y's selected material_description, history lessons, known_risks, or failure_boundaries mention high-volume/high volume, volume penalty/risk, volume_per_atom near the ceiling, open-framework packing, or a need for denser/compact structures, Z must set filters.max_volume_per_atom <= {DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM:g}; W must reject a default {DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM:g} cap unless Z explicitly explains that the tighter cap caused underfill in current controller feedback.\n"
            "- Put template_consistency_audit at the candidate top level with chosen_template='mattergen'; explain full-structure ML generation rather than fixed-template faithfulness.\n"
            "- The MatterGen template_consistency_audit must include all keys exactly: chosen_template, template_realizes_motif, unsupported_motif_substitution, mechanism_local_motif, required_coordination_or_polyhedra, why_template_is_faithful. Use chosen_template='mattergen', template_realizes_motif=true, unsupported_motif_substitution=false.\n"
            "- If X/Y names a desired reduced_formula or candidate target, copy that formula into the MatterGen request as filters.target_reduced_formula and set filters.require_target_reduced_formula=false; do not leave the target only at the candidate top level.\n"
            "- JSON values must be literal JSON only. Never write expressions such as \"abc\".replace(\" \", \"\"), string concatenation, comments, or function calls in request_id or any other field.\n"
            "- Forbidden with MatterGen backend: formula_probe_strings, formula_probes, structure_dicts, manual cells, manual fractional coordinates, mp_pool/material_ids/query.\n"
            "- MatterGen does not guarantee exact stoichiometry. In sequential X/Y MatterGen SUN optimization, filters.target_reduced_formula is always a soft preference: set filters.require_target_reduced_formula=false. The controller will force it back to false in this blind-search path even if Z sets true.\n"
            f"- Outside sequential X/Y blind search, hard target_reduced_formula is allowed only for chemical systems with at most {DEFAULT_MATTERGEN_HARD_TARGET_MAX_EXACT_ELEMENTS} elements. For four-element or larger systems, set filters.require_target_reduced_formula=false and treat target_reduced_formula as a soft preference even if X/Y named a desired formula.\n"
            "- After hard-target feedback with not_target_reduced_formula or no_accepted_structures, do not propose another sibling hard target formula in the same basin. Prefer require_target_reduced_formula=false with the exact chemical_system; this softening is generator-policy compliant and remains faithful when the chemical system, required elements, and target formula preference are preserved.\n"
            "- Do not repair a failed request by dropping required_elements or relaxing to element subsets. The controller will reject generated structures that omit any required chemical-system element.\n"
            "- For halide-rich soft targets, the controller rejects exact-system drift formulas that are severely halogen-deficient relative to filters.target_reduced_formula; preserve the halide/cation stoichiometry regime, not only the element set.\n"
        )
    if template_only:
        return (
            "EXECUTABLE_GENERATOR_SCHEMA_RULES:\n"
            "- Candidate object: source='generator', count=1, exactly one of \"formula_probe_strings\" or \"formula_probes\".\n"
            "- formula_probe_string format: template=<allowed>;Role=Element:+/-oxidation_state;...;family=<short_family>.\n"
            f"- Allowed template roles: {generator_template_role_summary()}.\n"
            f"- Required template_consistency_audit keys: {', '.join(GENERATOR_AUDIT_REQUIRED_KEYS)}; "
            "chosen_template must match the formula-probe template, template_realizes_motif=true, unsupported_motif_substitution=false.\n"
            "- Hard reject: any audit that calls the template a stoichiometry vehicle/proxy/container or admits the template may not capture the claimed topology.\n"
            "- Forbidden in template-only mode: structure_dicts, structure_dict, mp_pool, material_ids, query.\n"
        )
    return (
        "EXECUTABLE_GENERATOR_SCHEMA_RULES:\n```json\n"
        f"{prompt_json(generator_executable_schema_rules(template_only=template_only))}\n"
        "```\n"
    )


def sequential_mattergen_schema_reminder() -> str:
    return (
        "MATTERGEN_SEQUENTIAL_SCHEMA:\n"
        "- Candidate: source='generator', count=1, one mattergen_requests list; no formula_probes, structure_dicts, MP query, material_ids, manual cells, or coordinates.\n"
        "- Request: backend='mattergen', properties_to_condition_on.chemical_system='A-B[-C]', "
        "properties_to_condition_on.energy_above_hull=0.0, optional top-level target_count/batch_size/num_batches; "
        f"filters chemical_system=allowed=required elements, max_sites<={DEFAULT_MATTERGEN_MAX_SITES}.\n"
        "- Do not put chemical_system or energy_above_hull at the MatterGen request top level; those belong under properties_to_condition_on.\n"
        "- filters.chemical_system must be a list or hyphen string, not an object/dict; use separate filters.allowed_elements and filters.required_elements lists.\n"
        "- If the selected X/Y queue item names a reduced_formula target, copy it into filters.target_reduced_formula inside the MatterGen request and set filters.require_target_reduced_formula=false; do not leave the target only at the candidate top level.\n"
        f"- Sequential target_reduced_formula is soft: require_target_reduced_formula=false. If dense/high-volume risk appears, set filters.max_volume_per_atom<={DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM:g}.\n"
        "- template_consistency_audit: chosen_template='mattergen', template_realizes_motif=true, unsupported_motif_substitution=false, mechanism_local_motif, required_coordination_or_polyhedra, why_template_is_faithful.\n"
        "- Literal JSON only; no comments, concatenation, function calls, or Python expressions.\n"
    )


def sequential_generator_policy_block(*, backend: str, template_only: bool = False) -> str:
    if backend == "mattergen":
        return (
            "SEQUENTIAL_ZW_POLICY:\n"
            "- Z/W translate exactly the selected X/Y material description into one executable generated candidate; no MP-pool rows or stability/e_hull/SUN lookup.\n"
            "- Preserve the selected queue item unless Z/W proves it is unrepresentable or dominated by a more faithful legal queue item.\n"
            "- Audit motif faithfulness; MatterGen is a full-structure generator, not a hand-built template or formula container.\n"
            "- Treat controller generator feedback as binding: do not repeat failed chemical_system, hard target, duplicate/known formula, missing-element, or unsupported-field errors.\n"
            "- CONTEXT_JSON.controller_constraints.strategy_cooldowns is binding: do not submit blocked chemical_systems or blocked family_patterns. Repair by changing cation family, anion family, element count, or mechanism family; exclude_reduced_formulas alone is not a repair.\n"
            "- If feedback says MatterGen duplicate_pressure because accepted_count=0 and excluded_reduced_formula dominates the rejection reasons, one controlled rescue may increase sampling budget while keeping the same hidden exclusions. If feedback says saturated_mattergen_basin, do not increase sampling or drop visible exclusions; return to X/Y unless a faithful request leaves that chemical system.\n"
            "- Return to X/Y when no faithful MatterGen request remains after applying binding feedback, including saturated-basin feedback.\n"
        )
    return (
        f"{STRICT_COUNTERPROPOSAL_PROTOCOL}\n"
        f"{XY_GENERATOR_ONLY_POLICY}\n"
        f"{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}\n"
        f"{generator_executable_schema_reminder(template_only=template_only, backend=backend)}\n"
        f"{GENERATOR_VALIDATION_FEEDBACK_RULE}"
    )


def sequential_schema_reminder(*, backend: str, template_only: bool = False) -> str:
    if backend == "mattergen":
        return sequential_mattergen_schema_reminder()
    return generator_executable_schema_reminder(template_only=template_only, backend=backend)


GENERATOR_VALIDATION_FEEDBACK_RULE = """GENERATOR_VALIDATION_FEEDBACK_RULE:
- If CONTROLLER_GENERATOR_OR_W_FEEDBACK contains structure validation failures such as volume_per_atom_too_large, volume_per_atom_too_small, min_distance_too_small, known/training formula, or generator materialized 0 records, treat them as hard generator results.
- If feedback contains MatterGen failures such as no_accepted_structures, duplicate reduced_formula, known/training formula, outside_chemical_system, or Slurm/adapter failure, treat them as hard generator results and revise the MatterGen chemical_system/filters or return the description to X/Y only when the material description is genuinely unrepresentable or the requested chemical basin is saturated by repeated/high-budget controller exclusions.
- If feedback contains halogen_stoichiometry_guard, the request generated exact-system but halogen-deficient drift formulas. Do not evaluate or repeat those low-halogen formulas; repair by choosing a more halide-faithful target/filter strategy or return to X/Y when the selected target only materializes as low-halogen drift.
- If MatterGen feedback contains not_target_reduced_formula for a hard target_reduced_formula request, do not repeat that hard-target style by merely changing to a sibling exact formula in the same chemical basin. Prefer require_target_reduced_formula=false under the exact chemical_system. In MatterGen X/Y mode, preserving the same chemical_system, required elements, and target_reduced_formula as a soft preference is a faithful generator-policy repair, not an unfaithful material substitution.
- If feedback or CONTEXT_JSON lists strategy_cooldowns, treat blocked chemical_systems and blocked family_patterns as hard preflight failures. The next request must leave the basin; adding more exclude_reduced_formulas inside the same basin is invalid.
- Every listed generator error is binding, not only the most recent one. Z must not repeat any failed formula_probe_string, generated formula, template, and role mapping after such a failure.
- Change the template/roles to a materially different allowed generator input that fixes the stated numeric failure, or return candidate_specs=[] with can_generate_faithfully=false when the X/Y description is not representable inside the allowed template library or when every faithful allowed representation has been barred by binding generator feedback.
- W must reject any Z repair or W counterproposal that repeats any failed formula/template/role mapping from the feedback error history, even if the JSON schema is otherwise valid.
    - W must set return_to_xy=true only when no non-repeating faithful candidate exists after applying controller generator policy. In MatterGen mode, do not return to X/Y merely because Z converted a failed hard target_reduced_formula into the same exact chemical_system with require_target_reduced_formula=false; do return to X/Y when saturated_mattergen_basin feedback shows the same chemical_system is exhausted by repeated historical exclusions.
"""


STRICT_COUNTERPROPOSAL_PROTOCOL = """STRICT_COUNTERPROPOSAL_PROTOCOL:
- Use strict alternating proposal/review. If the reviewer rejects the current proposal, the reviewer must be able to make a constructive counterproposal rather than only object.
- The counterproposal must use the original proposer's JSON shape for parser compatibility. The original proposer then reviews that counterproposal using the reviewer's JSON shape.
- Stop only when one side accepts the other side's proposal, or when the controller exhausts the debate budget.
"""


AGENT_X_SYSTEM = (
    "You are Agent X, a peer materials-design theorist for blind post-experience evaluation. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Convert A/B's general experience book into concrete SUN-favorable material feature rules, not material candidates. "
    "You are equal to Agent Y; your proposal can be rejected or revised. "
    "Use concise, evidence-backed JSON only."
)


AGENT_Y_SYSTEM = (
    "You are Agent Y, a peer critic and scientific auditor for blind post-experience design-rule generation. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Audit whether Agent X really converted the full experience book into mechanistically concrete, executable material-feature rules while respecting failures and boundaries. "
    "Reject rules that are generic, stability-label-driven, or impossible for Z/W to instantiate faithfully. "
    "If you reject and the controller asks for a counterproposal, you must propose your own executable design book. "
    "Use concise, evidence-backed JSON only."
)


AGENT_DIRECT_X_SYSTEM = (
    "You are Agent X, a peer materials-design debater for blind post-experience evaluation. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Generate executable local-generator inputs from the available experience, not MP-pool material IDs and not by optimizing a known e_hull/SUN label. "
    "Before each candidate, audit whether the generator template really realizes the mechanism's local structure. "
    "You are equal to Agent Y; your proposal can be rejected or revised. "
    "Use concise, evidence-backed JSON only."
)


AGENT_DIRECT_Y_SYSTEM = (
    "You are Agent Y, a peer critic and scientific auditor for blind post-experience material generation. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Audit whether Agent X really used the full experience book mechanistically, respected failures and boundaries, avoided MP-pool selection and high-SUN prefiltering, and produced executable local-generator inputs. "
    "Reject candidates that use a generator template as an unfaithful proxy for the claimed microscopic mechanism. "
    "If you reject and the controller asks for a counterproposal, you must propose your own executable candidates, but you must not become a SUN hunter. "
    "Use concise, evidence-backed JSON only."
)


AGENT_Z_SYSTEM = (
    "You are Agent Z, a concrete material generator for blind post-experience evaluation. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Generate executable local-generator candidate strings only from X/Y's design_experience_book and A/B source principles. "
    "Every candidate must cite design_rule_ids and pass mechanism-to-template consistency. "
    "You are equal to Agent W; your candidates can be rejected or revised. "
    "Use concise, evidence-backed JSON only."
)


AGENT_W_SYSTEM = (
    "You are Agent W, a peer critic and scientific auditor for concrete material generation. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Audit whether Agent Z faithfully instantiated X/Y's design rules into diverse executable generator candidates without using stability labels or unfaithful template proxies. "
    "Reject candidates missing design_rule_ids, violating the design book, or copying a formula family without a rule-grounded reason. "
    "If you reject and the controller asks for a counterproposal, you must propose your own executable generator candidates. "
    "Use concise, evidence-backed JSON only."
)


AGENT_X_SEQUENTIAL_SYSTEM = (
    "You are Agent X, an equal peer in a sequential SUN optimization debate. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Rank a small queue of natural-language material descriptions and select the best legal strict-SUN bet. "
    "Z/W handle generator strings and feasibility. "
    "Use concise, evidence-backed JSON only."
)


AGENT_Y_SEQUENTIAL_SYSTEM = (
    "You are Agent Y, an equal peer and scientific critic in a sequential SUN optimization debate. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Audit X's whole ranked queue and selected natural-language material description. "
    "If rejecting on request, counterpropose your own ranked queue and selected material. "
    "Use concise, evidence-backed JSON only."
)


AGENT_Z_SEQUENTIAL_SYSTEM = (
    "You are Agent Z, a concrete generator translator and batch feasibility screener for X/Y's selected material design. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Inspect X/Y's ranked queue, then convert the selected material description into exactly one executable generator candidate when faithful generation is possible. "
    "If the description cannot be represented faithfully, say so through the review path instead of inventing a proxy. "
    "Use concise JSON only."
)


AGENT_W_SEQUENTIAL_SYSTEM = (
    "You are Agent W, a feasibility auditor and batch screener for one concrete generator candidate. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Approve exactly one candidate only when it faithfully represents X/Y's selected material description, survives comparison against the ranked queue, and is generator-executable. "
    "If the description itself is impossible or too vague for the current generator, return it to X/Y with specific repair feedback. "
    "If you reject and the controller asks for a counterproposal, you must propose your own executable generator candidate when one exists. "
    "Use concise JSON only."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run X/Y experience-driven blind candidate generation and SUN evaluation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--state", default="physics_mvp_runs/current/state.json", help="A/B discovery state containing principle_book.")
    parser.add_argument("--work-dir", default=None, help="Output directory. Defaults to xy_runs/<timestamp>.")
    parser.add_argument("--candidate-pool", default="data/mp_candidate_pool/mp_candidates_filtered.jsonl")
    parser.add_argument("--training-data", default=None)
    parser.add_argument("--ppd-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--mode", choices=("experience_xy", "no_experience_xy"), default="experience_xy")
    parser.add_argument(
        "--generation-protocol",
        choices=("two_stage", "direct", "sequential_single"),
        default="two_stage",
        help="two_stage runs X/Y design-book debate before Z/W candidate generation; direct preserves the legacy X/Y candidate debate; sequential_single runs one-material closed-loop SUN optimization.",
    )
    parser.add_argument(
        "--candidate-source",
        choices=("generator", "mixed", "mp_pool"),
        default="generator",
        help="Allowed final X/Y candidate source. Default generator forbids MP-pool candidate design.",
    )
    parser.add_argument(
        "--generator-template-only",
        action="store_true",
        help="For generator-only runs, forbid custom structure_dicts and require X/Y plus Z/W to stay inside formula_probe templates.",
    )
    parser.add_argument(
        "--generator-backend",
        choices=("template", "mattergen"),
        default=DEFAULT_GENERATOR_BACKEND,
        help="Concrete generator backend for source='generator'. Default template preserves the original formula-probe path.",
    )
    parser.add_argument("--mattergen-root", default=DEFAULT_MATTERGEN_ROOT)
    parser.add_argument(
        "--mattergen-venv",
        default=None,
        help=(
            "MatterGen Python environment to use for adapter and CLI execution. "
            "Defaults to <mattergen-root>/.venv; use a CUDA-12.8 venv for RTX 5090."
        ),
    )
    parser.add_argument("--mattergen-adapter", default="mattergen_backend_prototype/mattergen_adapter.py")
    parser.add_argument("--mattergen-model-path", default=DEFAULT_MATTERGEN_MODEL_PATH)
    parser.add_argument("--mattergen-checkpoint", default=DEFAULT_MATTERGEN_CHECKPOINT)
    parser.add_argument("--mattergen-bin", default=None)
    parser.add_argument("--mattergen-target-count", type=int, default=DEFAULT_MATTERGEN_TARGET_COUNT)
    parser.add_argument("--mattergen-batch-size", type=int, default=DEFAULT_MATTERGEN_BATCH_SIZE)
    parser.add_argument("--mattergen-num-batches", type=int, default=DEFAULT_MATTERGEN_NUM_BATCHES)
    parser.add_argument("--mattergen-max-sites", type=int, default=DEFAULT_MATTERGEN_MAX_SITES)
    parser.add_argument("--mattergen-min-volume-per-atom", type=float, default=DEFAULT_MATTERGEN_MIN_VOLUME_PER_ATOM)
    parser.add_argument("--mattergen-max-volume-per-atom", type=float, default=DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM)
    parser.add_argument(
        "--mattergen-allow-hard-target-formula",
        action="store_true",
        default=DEFAULT_MATTERGEN_ALLOW_HARD_TARGET_FORMULA,
        help=(
            "Legacy compatibility flag. MatterGen hard target filtering is now controlled directly by "
            "filters.require_target_reduced_formula in each request; omitted/false keeps target_reduced_formula soft."
        ),
    )
    parser.add_argument(
        "--mattergen-diffusion-guidance-factor",
        type=float,
        default=DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR,
        help=(
            "Classifier-free guidance scale for MatterGen conditional sampling. "
            "Keep >0 for chemical_system conditioning; 0 disables useful conditioning."
        ),
    )
    parser.add_argument("--mattergen-partition", default=DEFAULT_MATTERGEN_PARTITION)
    parser.add_argument("--mattergen-gres", default=DEFAULT_MATTERGEN_GRES)
    parser.add_argument("--mattergen-cpus-per-task", type=int, default=DEFAULT_MATTERGEN_CPUS_PER_TASK)
    parser.add_argument(
        "--mattergen-module-init",
        default=DEFAULT_MATTERGEN_MODULE_INIT,
        help="Optional shell init script for environment modules before running MatterGen Slurm jobs.",
    )
    parser.add_argument(
        "--mattergen-modules",
        default=DEFAULT_MATTERGEN_MODULES,
        help="Optional comma/space-separated module names to load before running MatterGen Slurm jobs.",
    )
    parser.add_argument(
        "--mattergen-cuda-home",
        default=DEFAULT_MATTERGEN_CUDA_HOME,
        help="Optional CUDA_HOME for MatterGen Slurm jobs; also prepends bin and lib64 paths.",
    )
    parser.add_argument(
        "--mattergen-runner",
        choices=("slurm", "local"),
        default="slurm",
        help="Run MatterGen adapter through Slurm by default. Use local only for conversion tests or explicit smoke runs.",
    )
    parser.add_argument(
        "--mattergen-job-timeout",
        type=int,
        default=0,
        help="Seconds to wait for a MatterGen Slurm job; 0 means wait indefinitely.",
    )
    parser.add_argument("--mattergen-poll-sec", type=float, default=10.0)
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=None,
        help=(
            "Target candidate/iteration count. In sequential_single, omit this option or pass <=0 to run without "
            f"a fixed iteration limit. Other protocols default to {DEFAULT_CANDIDATE_COUNT} when omitted."
        ),
    )
    parser.add_argument("--oversample", type=float, default=DEFAULT_OVERSAMPLE)
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    parser.add_argument(
        "--resume-existing-shards",
        action="store_true",
        help="Reuse existing debates/shard_*/xy_debate.json files in work-dir instead of rerunning those shards.",
    )
    parser.add_argument(
        "--skip-missing-initial-shards-on-resume",
        action="store_true",
        help="When resuming, mark missing initial shards unresolved and move on to backfill instead of rerunning them.",
    )
    parser.add_argument(
        "--max-backfill-batches",
        type=int,
        default=0,
        help="Additional X/Y shard batches to run when locked candidates remain below --candidate-count.",
    )
    parser.add_argument(
        "--backfill-shards",
        type=int,
        default=10,
        help="Shard count for each dynamic backfill batch.",
    )
    parser.add_argument(
        "--backfill-oversample",
        type=float,
        default=2.0,
        help="Oversample factor for dynamic backfill batches.",
    )
    parser.add_argument("--parallel-workers", type=int, default=None)
    parser.add_argument("--max-debate-rounds", type=int, default=DEFAULT_MAX_DEBATE_ROUNDS)
    parser.add_argument("--min-debate-rounds", type=int, default=DEFAULT_MIN_DEBATE_ROUNDS)
    parser.add_argument(
        "--critic-counterproposal-after",
        type=int,
        default=DEFAULT_CRITIC_COUNTERPROPOSAL_AFTER,
        help="After this many consecutive reviewer rejections, force the reviewer to produce a counterproposal and let the original proposer reverse-review it.",
    )
    parser.add_argument(
        "--max-materialization-repair-rounds",
        type=int,
        default=4,
        help="Per-shard controller materialization dry-run repair rounds after X/Y agreement.",
    )
    parser.add_argument(
        "--max-zw-generator-repair-rounds",
        type=int,
        default=None,
        help="Sequential-single Z/W generator/materializer repair rounds. Defaults to --max-materialization-repair-rounds.",
    )
    parser.add_argument(
        "--max-description-revision-rounds",
        type=int,
        default=3,
        help="Sequential-single attempts per material where Z/W can return an infeasible natural-language description to X/Y.",
    )
    parser.add_argument(
        "--xy-sun-candidate-queue-size",
        type=int,
        default=4,
        help=(
            "Sequential-single X/Y prompt target for a ranked SUN candidate queue. "
            "The controller still selects one X/Y/Z/W route per iteration; MatterGen may materialize a batch from that route."
        ),
    )
    parser.add_argument(
        "--sequential-materialization-target-count",
        type=int,
        default=1,
        help=(
            "Sequential-single materialization batch size. Default 1 preserves the original one-structure loop; "
            "larger values let one X/Y/Z/W MatterGen request contribute multiple filtered structures."
        ),
    )
    parser.add_argument(
        "--seed-sequential-memory",
        default=None,
        help=(
            "Optional sequential_memory.json to copy into a fresh sequential-single work-dir before iteration 1. "
            "Use this when restarting a run so X/Y keeps prior generated formulas, generator failures, and postmortems."
        ),
    )
    parser.add_argument("--json-repair-attempts", type=int, default=2)
    parser.add_argument("--model", default=None)
    parser.add_argument("--x-model", default=None)
    parser.add_argument("--y-model", default=None)
    parser.add_argument("--z-model", default=None)
    parser.add_argument("--w-model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--disable-local-agents", action="store_true")
    parser.add_argument("--agent-max-steps", type=int, default=DEFAULT_AGENT_MAX_STEPS)
    parser.add_argument("--agent-max-tool-calls", type=int, default=DEFAULT_AGENT_MAX_TOOL_CALLS)
    parser.add_argument("--agent-max-tool-result-chars", type=int, default=8000)
    parser.add_argument("--evaluator-backend", choices=("local", "slurm"), default="slurm")
    parser.add_argument("--slurm-evaluator-script", default="slurm_evaluate_round_4090.sbatch")
    parser.add_argument("--slurm-partition", default="8-4090,8-5090")
    parser.add_argument("--slurm-gres", default="gpu:1")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=16)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--seed-base", type=int, default=20260525)
    parser.add_argument("--max-sites", type=int, default=80)
    parser.add_argument(
        "--max-per-reduced-formula",
        type=int,
        default=1,
        help="Maximum locked candidates per reduced formula. Default 1 is hard reduced_formula de-duplication; 0 disables.",
    )
    parser.add_argument("--skip-evaluation", action="store_true", help="Only run X/Y debates and materialization.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def prompt_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def short_text(value: Any, max_chars: int = 360) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def compact_principle_book(book: Any, *, include_full: bool) -> list[dict[str, Any]]:
    if not include_full or not isinstance(book, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in book:
        if not isinstance(item, Mapping):
            continue
        compact.append(
            {
                key: value
                for key, value in {
                    "program_id": item.get("program_id"),
                    "status": item.get("status"),
                    "topic_key": item.get("topic_key"),
                    "principle_statement": short_text(item.get("principle_statement"), 520),
                    "micro_mechanism": short_text(item.get("micro_mechanism"), 520),
                    "reasoning_chain": [short_text(raw, 180) for raw in item.get("reasoning_chain", [])[:5]]
                    if isinstance(item.get("reasoning_chain"), list)
                    else None,
                    "evidence_rounds": item.get("evidence_rounds"),
                    "boundaries": [short_text(raw, 180) for raw in item.get("boundaries", [])[:5]]
                    if isinstance(item.get("boundaries"), list)
                    else None,
                    "residual_risks": [short_text(raw, 180) for raw in item.get("residual_risks", [])[:4]]
                    if isinstance(item.get("residual_risks"), list)
                    else None,
                }.items()
                if value not in (None, "", [])
            }
        )
    return compact


def compact_principle_index(book: Any, *, max_items: int = 20) -> list[dict[str, Any]]:
    if not isinstance(book, list) or max_items <= 0:
        return []
    compact: list[dict[str, Any]] = []
    for item in book[-max_items:]:
        if not isinstance(item, Mapping):
            continue
        compact.append(
            {
                key: value
                for key, value in {
                    "program_id": item.get("program_id"),
                    "status": item.get("status"),
                    "topic_key": short_text(item.get("topic_key"), 120),
                }.items()
                if value not in (None, "", [])
            }
        )
    return compact


def compact_active_principle_program(program: Any) -> dict[str, Any] | None:
    if not isinstance(program, Mapping):
        return None
    return {
        key: value
        for key, value in {
            "program_id": program.get("program_id"),
            "status": program.get("status"),
            "current_round": program.get("current_round"),
            "inner_iteration": program.get("inner_iteration"),
            "current_principle_statement": short_text(program.get("current_principle_statement"), 90),
            "micro_mechanism": short_text(program.get("micro_mechanism"), 70),
        }.items()
        if value not in (None, "", [])
    }


def compact_state_context(state: Mapping[str, Any], *, include_experience: bool) -> dict[str, Any]:
    history = state.get("history")
    history_len = len(history) if isinstance(history, list) else 0
    return {
        "schema_version": "xy_blind_generation_context.v2",
        "mode": "experience_xy" if include_experience else "no_experience_xy",
        "state_summary": {
            "status": state.get("status"),
            "current_round": state.get("current_round"),
            "history_len": history_len,
            "principle_book_len": len(state.get("principle_book", [])) if isinstance(state.get("principle_book"), list) else 0,
        },
        "principle_book": compact_principle_book(state.get("principle_book"), include_full=include_experience),
        "active_principle_program": state.get("current_principle_program") if include_experience else None,
        "instructions": [
            "Generate candidates before seeing any new candidate e_hull/SUN result.",
            "Use the full principle book when mode=experience_xy; no controller-side high-SUN principle prefilter is allowed.",
            "When mode=no_experience_xy, ignore principle_book even if the state file contains one.",
            "Treat rejected principles and boundaries as important negative knowledge.",
            "In two_stage protocol, X/Y first produce a design_experience_book and Z/W later produce concrete candidates from that book.",
        ],
    }


def compact_sequential_state_context(state: Mapping[str, Any], *, include_experience: bool) -> dict[str, Any]:
    history = state.get("history")
    history_len = len(history) if isinstance(history, list) else 0
    principle_book = state.get("principle_book")
    principle_book_len = len(principle_book) if isinstance(principle_book, list) else 0
    return {
        "schema_version": "xy_sequential_single_context.v2.compact",
        "mode": "experience_xy" if include_experience else "no_experience_xy",
        "strict_sun_definition": STRICT_SUN_NOTE,
        "state_summary": {
            "status": state.get("status"),
            "current_round": state.get("current_round"),
            "history_len": history_len,
            "principle_book_len": principle_book_len,
        },
        "active_principle_program": compact_active_principle_program(state.get("current_principle_program"))
        if include_experience
        else None,
        "principle_book_index_tail": [],
    }


def split_shards(candidate_count: int, oversample: float, shards: int) -> list[int]:
    target_total = max(candidate_count, int(math.ceil(candidate_count * max(1.0, oversample))))
    shard_count = max(1, min(max(1, shards), target_total))
    base, remainder = divmod(target_total, shard_count)
    return [base + (1 if index < remainder else 0) for index in range(shard_count)]


def make_client(
    *,
    role: str,
    args: argparse.Namespace,
    root: Path,
    log_dir: Path,
    state_path: Path,
    candidate_pool_path: Path,
    xy_history_path: Path | None = None,
) -> Any:
    role_model = {
        "X": args.x_model,
        "Y": args.y_model,
        "Z": getattr(args, "z_model", None),
        "W": getattr(args, "w_model", None),
    }.get(role)
    client = make_llm_client(
        LLMConfig.from_env(
            dotenv=root / args.dotenv,
            role=f"XY_{role}" if role in {"X", "Y"} else f"ZW_{role}",
            model=role_model or args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        log_dir=log_dir,
    )
    if not args.disable_local_agents:
        tool_result_chars = max(1000, int(args.agent_max_tool_result_chars))
        if getattr(args, "generation_protocol", "") == "sequential_single":
            tool_result_chars = min(tool_result_chars, 4000)
        client.local_agent_runtime = LocalAgentRuntime(  # type: ignore[attr-defined]
            root=root,
            trace_dir=log_dir / "agent_traces",
            writable_dir=root / "agent_artifacts" / "xy_experience_debate" / role.lower(),
            candidate_pool_path=candidate_pool_path,
            state_path=state_path,
            xy_history_path=xy_history_path,
            max_steps=max(0, int(args.agent_max_steps)),
            max_tool_calls_per_step=max(1, int(args.agent_max_tool_calls)),
            max_tool_result_chars=tool_result_chars,
            allow_project_writes=False,
            allow_candidate_pool_tools=str(getattr(args, "candidate_source", "generator")) != "generator",
        )
    return client


def allowed_candidate_sources(args: argparse.Namespace) -> set[str]:
    source = str(getattr(args, "candidate_source", "generator") or "generator")
    if source == "generator":
        return {"generator"}
    if source == "mp_pool":
        return {"mp_pool"}
    return {"generator", "mp_pool"}


_JSON_STRING_REPLACE_EMPTY_RE = re.compile(
    r'("(?:[^"\\]|\\.)*")\s*\.replace\(\s*(" ")\s*,\s*("")\s*\)'
)


def sanitize_json_like_model_output(text: str) -> str:
    """Repair simple code-like string expressions that models sometimes put inside JSON."""

    def replace_literal(match: re.Match[str]) -> str:
        try:
            literal = json.loads(match.group(1))
        except Exception:
            return match.group(0)
        if not isinstance(literal, str):
            return match.group(0)
        return json.dumps(literal.replace(" ", ""))

    return _JSON_STRING_REPLACE_EMPTY_RE.sub(replace_literal, text)


def compact_invalid_json_preview_for_prompt(text: str, *, max_chars: int = 1200) -> str:
    raw = str(text or "")
    try:
        parsed = extract_json_object(sanitize_json_like_model_output(raw))
    except Exception:
        preview = short_text(raw, max_chars)
    else:
        if isinstance(parsed, Mapping):
            preview = prompt_json(compact_payload_for_dialogue(parsed))
        else:
            preview = prompt_json(parsed)
        preview = short_text(preview, max_chars)
    if len(raw) > max_chars:
        preview += f"\n[invalid_output_omitted_chars={len(raw) - max_chars}]"
    return preview


def call_json_object(
    client: Any,
    *,
    system: str,
    user: str,
    role: str,
    metadata: Mapping[str, Any],
    json_repair_attempts: int,
) -> dict[str, Any]:
    prompt = user
    last_text = ""
    last_error = ""
    last_error_was_recoverable_llm = False
    role_recoverable_retries = max(0, _env_int("LLM_ROLE_RECOVERABLE_RETRIES", 0))
    role_recoverable_retry_sleep = max(0.0, _env_float("LLM_ROLE_RECOVERABLE_RETRY_SLEEP", 30.0))
    role_recoverable_attempts = 0
    max_json_attempt = max(0, json_repair_attempts)
    attempt = 0
    while attempt <= max_json_attempt:
        try:
            call_metadata = {**dict(metadata), "json_retry_attempt": attempt}
            runtime = getattr(client, "local_agent_runtime", None)
            if isinstance(runtime, LocalAgentRuntime):
                text = runtime.complete_text(client, system=system, user=prompt, metadata=call_metadata)
            else:
                text = client.complete_text(system=system, user=prompt, metadata=call_metadata)
            last_text = text
            try:
                parsed = extract_json_object(text)
            except (ValueError, TypeError):
                sanitized_text = sanitize_json_like_model_output(text)
                if sanitized_text == text:
                    raise
                last_text = sanitized_text
                parsed = extract_json_object(sanitized_text)
            if isinstance(parsed, Mapping):
                return dict(parsed)
            last_error = "top-level JSON value is not an object"
            last_error_was_recoverable_llm = False
        except LLMError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            last_error_was_recoverable_llm = is_recoverable_llm_error_message(str(exc))
            last_text = ""
            if last_error_was_recoverable_llm and role_recoverable_attempts < role_recoverable_retries:
                role_recoverable_attempts += 1
                sleep_seconds = role_recoverable_retry_sleep * min(role_recoverable_attempts, 3)
                print(
                    f"[{utc_now()}] xy_recoverable_llm_role_retry role={role} "
                    f"attempt={role_recoverable_attempts}/{role_recoverable_retries} "
                    f"json_retry_attempt={attempt} sleep_sec={sleep_seconds:.1f} "
                    f"error={last_error.replace(chr(10), ' ')[:500]}",
                    flush=True,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                continue
            if attempt < max_json_attempt:
                prompt = user
            attempt += 1
            continue
        except (ValueError, TypeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            last_error_was_recoverable_llm = False
        if attempt < max_json_attempt:
            prompt = (
                f"{user}\n\nJSON_REPAIR_REQUEST:\n"
                f"The previous {role} response was not a valid JSON object for this task.\n"
                f"Error: {last_error}\n"
                f"Previous response preview:\n{compact_invalid_json_preview_for_prompt(last_text)}\n"
                "Return one corrected JSON object only. Do not request tools in this JSON repair response, do not output "
                "status=\"tool_request\", and do not concatenate a tool_request JSON object with a final role JSON object. "
                "Do not include Python expressions, function calls, comments, trailing commas, or code such as .replace(...); "
                "every value must be literal JSON."
            )
        attempt += 1
    if last_error_was_recoverable_llm:
        raise RecoverableLLMFailure(
            role=role,
            metadata=metadata,
            error=last_error,
            attempts=max_json_attempt + 1 + role_recoverable_attempts,
        )
    raise JSONOutputRepairFailure(
        role=role,
        metadata=metadata,
        error=last_error,
        attempts=max(0, json_repair_attempts) + 1,
        last_text=last_text,
    )


def json_output_repair_feedback(
    *,
    exc: JSONOutputRepairFailure,
    iteration: int,
    description_attempt: int,
    repair_round: int,
    backend: str = DEFAULT_GENERATOR_BACKEND,
    template_only: bool = False,
) -> dict[str, Any]:
    return {
        "source": "controller_json_parse_feedback",
        "iteration": iteration,
        "description_attempt": description_attempt,
        "failed_repair_round": repair_round,
        "role": exc.role,
        "controller_error": exc.error,
        "controller_instruction": (
            "The previous response was not valid JSON and was not accepted. "
            "Return one JSON object only. Do not include Python expressions, function calls, comments, trailing commas, "
            "or code such as .replace(...). All field values must be literal JSON strings, numbers, booleans, arrays, "
            "objects, or null."
        ),
        "previous_response_preview": exc.last_text[:2000],
        "json_rules": {
            "must_be_single_json_object": True,
            "forbidden_in_json_values": [
                ".replace(...)",
                "function calls",
                "Python expressions",
                "comments",
                "trailing commas",
            ],
        },
        "executable_generator_rules": generator_executable_schema_rules(
            template_only=template_only,
            backend=backend,
        ),
    }


def shard_context_payload(
    *,
    state: Mapping[str, Any],
    pool_summary: Mapping[str, Any],
    mode: str,
    generation_protocol: str,
    candidate_source: str,
    shard_index: int,
    shard_target: int,
    total_target: int,
    seed: int,
) -> dict[str, Any]:
    include_experience = mode == "experience_xy"
    context = compact_sequential_state_context(state, include_experience=include_experience)
    context.update(
        {
            "shard": {
                "index": shard_index,
                "target_candidate_count": shard_target,
                "overall_locked_candidate_count": total_target,
                "seed": seed,
            },
            "candidate_pool_summary": (
                pool_summary
                if candidate_source != "generator"
                else {
                    "available_to_xy": False,
                    "reason": "MP-pool candidate selection is disabled. X/Y must design generator inputs directly.",
                }
            ),
            "generator_formula_probe_schema_reference": json.loads(generator_schema_reference_json(target_count=max(1, shard_target))),
            "material_source_policy": {
                "generation_protocol": generation_protocol,
                "candidate_source": candidate_source,
                "generator_backend": DEFAULT_GENERATOR_BACKEND,
                "allowed_sources": ["generator"] if candidate_source == "generator" else (["mp_pool"] if candidate_source == "mp_pool" else ["generator", "mp_pool"]),
                "mp_pool_final_candidates_allowed": candidate_source != "generator",
                "query_candidate_pool_allowed": candidate_source != "generator",
            },
            "blind_evaluation_controls": {
                "candidate_lock_before_evaluation": True,
                "no_controller_high_sun_prefilter": True,
                "no_candidate_e_hull_lookup_before_lock": True,
                "generation_protocol": generation_protocol,
                "disallowed_selection_fields": sorted(DEFAULT_FORBIDDEN_SELECTION_FIELDS),
                "allowed_preferred_order": sorted(order for order in DEFAULT_ALLOWED_PREFERRED_ORDERS if order),
            },
        }
    )
    return context


def prompt_x_design_book_proposal(context: Mapping[str, Any]) -> str:
    return f"""Agent X design-experience proposal for one parallel shard.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_DESIGN_BOOK_POLICY}

{XY_DESIGN_BOOK_SCHEMA}

CONTEXT_JSON:
```json
{prompt_json(context)}
```

Return JSON:
{{
  "status": "design_proposal",
  "agent": "X",
  "shard_id": {context["shard"]["index"]},
  "target_count": {context["shard"]["target_candidate_count"]},
  "design_experience_book": ["design rule objects"],
  "experience_coverage": {{
    "principle_ids_considered": [],
    "validated_lessons_used": [],
    "rejected_or_boundary_lessons_used": [],
    "no_high_sun_prefilter": true
  }},
  "proposal_summary": "concise summary of how general A/B principles were concretized"
}}
"""


def prompt_y_design_book_review(context: Mapping[str, Any], proposal: Mapping[str, Any], cycle: int) -> str:
    return f"""Agent Y design-experience review cycle {cycle}.

{STRICT_COUNTERPROPOSAL_PROTOCOL}

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_DESIGN_BOOK_POLICY}

Audit Agent X's design_experience_book. It must transform A/B's general material principles into concrete, executable material-feature rules for Z/W, without naming final material candidates.

Require:
- every rule cites source_principle_ids and respects negative boundaries;
- every rule specifies local motif, coordination/polyhedra, charge balance, allowed templates, forbidden proxy templates, and Z/W generation instructions;
- no rule ranks by or depends on historical SUN/e_hull/stability labels;
- no formula_probe_strings, material_ids, MP queries, or concrete candidate materials appear in the design book.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

AGENT_X_DESIGN_PROPOSAL:
```json
{prompt_json(proposal)}
```

Return JSON:
{{
  "status": "design_review",
  "agent": "Y",
  "shard_id": {context["shard"]["index"]},
  "agree": false,
  "approved_design_rule_ids": [],
  "rejected_design_rules": [
    {{"design_rule_id": "xy_s001_r001", "reason": "specific flaw", "required_revision": "specific fix"}}
  ],
  "counterproposal_design_rules": [],
  "experience_audit": {{
    "used_full_experience_book": true,
    "no_high_sun_prefilter_violation": false,
    "boundary_failures_checked": [],
    "rules_are_executable_by_zw": false,
    "comments": "concise audit"
  }},
  "overall_reasoning_summary": "concise summary"
}}
Set agree=true only when the design_experience_book is concrete enough for Z/W to generate at least the shard target count without violating A/B principles.
"""


def prompt_y_design_book_counterproposal(context: Mapping[str, Any], proposal: Mapping[str, Any], review: Mapping[str, Any], cycle: int) -> str:
    return f"""Agent Y: you rejected Agent X's design-experience book in cycle {cycle}. You must now propose your own design-experience book that satisfies your critique.

{STRICT_COUNTERPROPOSAL_PROTOCOL}

Do not only object. Either produce Agent X-compatible design rules, or produce an empty design_experience_book with a concise impossibility_certificate explaining why no faithful executable design book can be stated from the available A/B principles.
For parser compatibility, set the JSON field agent to "X" and use the same output shape as Agent X: status, agent, shard_id, target_count, design_experience_book, experience_coverage, proposal_summary.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_DESIGN_BOOK_POLICY}

{XY_DESIGN_BOOK_SCHEMA}

CONTEXT_JSON:
```json
{prompt_json(context)}
```

PREVIOUS_AGENT_X_DESIGN_PROPOSAL:
```json
{prompt_json(proposal)}
```

AGENT_Y_DESIGN_REVIEW:
```json
{prompt_json(review)}
```

Return only Agent X JSON.
"""


def prompt_x_design_book_reverse_review(
    context: Mapping[str, Any],
    original_proposal: Mapping[str, Any],
    counterproposal: Mapping[str, Any],
    cycle: int,
) -> str:
    return f"""Agent X: critique Agent Y's design-experience counterproposal for cycle {cycle}.

{STRICT_COUNTERPROPOSAL_PROTOCOL}

You are now the reviewer. If Agent Y's counterproposal is more faithful, executable, and less risky than yours, set agree=true. If not, provide exact required revisions.
For parser compatibility, set the JSON field agent to "Y" and use the same output shape as Agent Y: status, agent, shard_id, agree, approved_design_rule_ids, rejected_design_rules, counterproposal_design_rules, experience_audit, overall_reasoning_summary.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_DESIGN_BOOK_POLICY}

CONTEXT_JSON:
```json
{prompt_json(context)}
```

ORIGINAL_AGENT_X_DESIGN_PROPOSAL:
```json
{prompt_json(original_proposal)}
```

AGENT_Y_COUNTERPROPOSAL_JSON:
```json
{prompt_json(counterproposal)}
```

Return only Agent Y JSON.
"""


def prompt_x_design_book_revision(
    context: Mapping[str, Any],
    proposal: Mapping[str, Any],
    review: Mapping[str, Any],
    cycle: int,
) -> str:
    return f"""Agent X design-experience revision cycle {cycle}.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_DESIGN_BOOK_POLICY}

Revise the design_experience_book in response to Agent Y. Preserve approved rules, repair rejected rules, and keep the output as design rules only, not material candidates.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

PREVIOUS_X_DESIGN_PROPOSAL:
```json
{prompt_json(proposal)}
```

AGENT_Y_DESIGN_REVIEW:
```json
{prompt_json(review)}
```

Return the same Agent X design proposal JSON shape with updated design_experience_book.
"""


def prompt_y_design_book_final(context: Mapping[str, Any], proposal: Mapping[str, Any], review: Mapping[str, Any]) -> str:
    return f"""Agent Y final design-experience consensus.

{XY_BLIND_EVALUATION_POLICY}

{XY_DESIGN_BOOK_POLICY}

Write the final X/Y design_experience_book. Do not introduce rules that X/Y have not reviewed. Do not include concrete candidate materials.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

LATEST_X_DESIGN_PROPOSAL:
```json
{prompt_json(proposal)}
```

LATEST_Y_DESIGN_REVIEW:
```json
{prompt_json(review)}
```

Return JSON:
{{
  "status": "design_consensus",
  "agent": "Y",
  "shard_id": {context["shard"]["index"]},
  "design_experience_book": ["design rule objects"],
  "experience_coverage": {{
    "principle_ids_used": [],
    "rejected_or_boundary_lessons_used": [],
    "no_high_sun_prefilter": true
  }},
  "blind_lock_statement": "This design book is frozen before any Z/W candidate e_hull/SUN evaluation.",
  "debate_summary": "concise summary"
}}
"""


def prompt_x_proposal(context: Mapping[str, Any]) -> str:
    return f"""Agent X proposal for one parallel shard of blind material generation.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

{XY_CANDIDATE_SCHEMA}

CONTEXT_JSON:
```json
{prompt_json(context)}
```

Return JSON:
{{
  "status": "proposal",
  "agent": "X",
  "shard_id": {context["shard"]["index"]},
  "target_count": {context["shard"]["target_candidate_count"]},
  "candidate_specs": ["candidate objects"],
  "experience_coverage": {{
    "principle_ids_considered": [],
    "validated_lessons_used": [],
    "rejected_or_boundary_lessons_used": [],
    "no_high_sun_prefilter": true
  }},
  "proposal_summary": "concise summary"
}}
"""


def prompt_y_review(context: Mapping[str, Any], proposal: Mapping[str, Any], cycle: int) -> str:
    return f"""Agent Y review cycle {cycle}.

{STRICT_COUNTERPROPOSAL_PROTOCOL}

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

Audit Agent X's proposal for scientific use of the full experience book, no high-SUN/stability-label prefiltering, generator executability, mechanism-to-template consistency, duplicate risk, and respect for rejected principles and boundaries.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

AGENT_X_PROPOSAL:
```json
{prompt_json(compact_material_description_for_review_prompt(proposal))}
```

Return JSON:
{{
  "status": "review",
  "agent": "Y",
  "shard_id": {context["shard"]["index"]},
  "agree": false,
  "approved_candidate_ids": [],
  "rejected_candidates": [
    {{"candidate_id": "xy_s001_c001", "reason": "specific flaw", "required_revision": "specific fix"}}
  ],
  "counterproposal_candidate_specs": [],
  "experience_audit": {{
    "used_full_experience_book": true,
    "no_high_sun_prefilter_violation": false,
    "boundary_failures_checked": [],
    "template_consistency_checked": true,
    "comments": "concise audit"
  }},
  "overall_reasoning_summary": "concise summary"
}}
Set agree=true only when enough executable candidate_specs are acceptable for this shard and the audit passes.
"""


def prompt_y_counterproposal(context: Mapping[str, Any], proposal: Mapping[str, Any], review: Mapping[str, Any], cycle: int) -> str:
    return f"""Agent Y: you rejected Agent X's candidate proposal in cycle {cycle}. You must now propose your own executable candidate set that satisfies your critique.

{STRICT_COUNTERPROPOSAL_PROTOCOL}

Do not only object. Either produce Agent X-compatible candidate_specs, or produce candidate_specs=[] with a concise impossibility_certificate explaining why no faithful generator-executable candidate set can satisfy the shard target under the blind protocol.
For parser compatibility, set the JSON field agent to "X" and use the same output shape as Agent X: status, agent, shard_id, target_count, candidate_specs, experience_coverage, proposal_summary.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

{XY_CANDIDATE_SCHEMA}

CONTEXT_JSON:
```json
{prompt_json(context)}
```

PREVIOUS_AGENT_X_PROPOSAL:
```json
{prompt_json(proposal)}
```

AGENT_Y_REVIEW:
```json
{prompt_json(review)}
```

Return only Agent X JSON.
"""


def prompt_x_reverse_review(context: Mapping[str, Any], original_proposal: Mapping[str, Any], counterproposal: Mapping[str, Any], cycle: int) -> str:
    return f"""Agent X: critique Agent Y's candidate counterproposal for cycle {cycle}.

{STRICT_COUNTERPROPOSAL_PROTOCOL}

You are now the reviewer. If Agent Y's counterproposal is more faithful, executable, and less risky than yours, set agree=true. If not, provide exact required revisions.
For parser compatibility, set the JSON field agent to "Y" and use the same output shape as Agent Y: status, agent, shard_id, agree, approved_candidate_ids, rejected_candidates, counterproposal_candidate_specs, experience_audit, overall_reasoning_summary.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

CONTEXT_JSON:
```json
{prompt_json(context)}
```

ORIGINAL_AGENT_X_PROPOSAL:
```json
{prompt_json(original_proposal)}
```

AGENT_Y_COUNTERPROPOSAL_JSON:
```json
{prompt_json(counterproposal)}
```

Return only Agent Y JSON.
"""


def prompt_x_revision(context: Mapping[str, Any], proposal: Mapping[str, Any], review: Mapping[str, Any], cycle: int) -> str:
    return f"""Agent X revision cycle {cycle}.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

Revise the candidate set in response to Agent Y. Preserve accepted candidates when still valid, replace rejected ones, and keep the shard target count. Do not optimize for known SUN/e_hull or formation_energy_per_atom.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

PREVIOUS_X_PROPOSAL:
```json
{prompt_json(compact_material_description_for_prompt(proposal))}
```

AGENT_Y_REVIEW:
```json
{prompt_json(compact_review_for_prompt(review))}
```

Return the same Agent X proposal JSON shape with updated candidate_specs.
"""


def prompt_y_final(context: Mapping[str, Any], proposal: Mapping[str, Any], review: Mapping[str, Any]) -> str:
    return f"""Agent Y final consensus.

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

Write the final locked shard candidates. Do not introduce any candidate that X/Y have not reviewed. Candidate specs must remain generator-executable, mechanism-template faithful, and blind.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

LATEST_X_PROPOSAL:
```json
{prompt_json(compact_material_description_for_prompt(proposal))}
```

LATEST_Y_REVIEW:
```json
{prompt_json(compact_review_for_prompt(review))}
```

Return JSON:
{{
  "status": "consensus",
  "agent": "Y",
  "shard_id": {context["shard"]["index"]},
  "agreed_candidate_specs": ["candidate objects"],
  "rejected_candidates": [],
  "experience_coverage": {{
    "principle_ids_used": [],
    "rejected_or_boundary_lessons_used": [],
    "no_high_sun_prefilter": true
  }},
  "blind_lock_statement": "These candidates are locked before any new e_hull/SUN evaluation.",
  "debate_summary": "concise summary"
}}
"""


def prompt_z_proposal(context: Mapping[str, Any], design_consensus: Mapping[str, Any]) -> str:
    return f"""Agent Z concrete material proposal for one parallel shard.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

{ZW_GENERATION_POLICY}

{XY_CANDIDATE_SCHEMA}

Generate concrete candidate materials by applying the frozen X/Y design_experience_book. Do not invent a new principle. Every candidate must cite design_rule_ids from X/Y and cited_principle_ids from the design rule source principles.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

FROZEN_XY_DESIGN_CONSENSUS:
```json
{prompt_json(design_consensus)}
```

Return JSON:
{{
  "status": "proposal",
  "agent": "Z",
  "shard_id": {context["shard"]["index"]},
  "target_count": {context["shard"]["target_candidate_count"]},
  "candidate_specs": ["candidate objects"],
  "design_book_usage": {{
    "design_rule_ids_used": [],
    "rules_marked_unusable": [],
    "diversity_plan": "concise plan"
  }},
  "proposal_summary": "concise summary"
}}
"""


def prompt_w_review(
    context: Mapping[str, Any],
    design_consensus: Mapping[str, Any],
    proposal: Mapping[str, Any],
    cycle: int,
) -> str:
    return f"""Agent W concrete material review cycle {cycle}.

{STRICT_COUNTERPROPOSAL_PROTOCOL}

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

{ZW_GENERATION_POLICY}

Audit Agent Z's candidates against the frozen X/Y design_experience_book. Do not evaluate or infer e_hull/SUN. Reject candidates that do not cite design_rule_ids, violate allowed element/template/motif constraints, repeat formula families unnecessarily, or use an unfaithful generator template proxy.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

FROZEN_XY_DESIGN_CONSENSUS:
```json
{prompt_json(design_consensus)}
```

AGENT_Z_PROPOSAL:
```json
{prompt_json(proposal)}
```

Return JSON:
{{
  "status": "review",
  "agent": "W",
  "shard_id": {context["shard"]["index"]},
  "agree": false,
  "approved_candidate_ids": [],
  "rejected_candidates": [
    {{"candidate_id": "zw_s001_c001", "reason": "specific flaw", "required_revision": "specific fix"}}
  ],
  "counterproposal_candidate_specs": [],
  "design_book_audit": {{
    "all_candidates_have_design_rule_ids": false,
    "all_candidates_are_rule_faithful": false,
    "template_consistency_checked": true,
    "no_high_sun_prefilter_violation": false,
    "comments": "concise audit"
  }},
  "overall_reasoning_summary": "concise summary"
}}
Set agree=true only when enough executable candidate_specs are acceptable for this shard and the design-book audit passes.
"""


def prompt_w_counterproposal(
    context: Mapping[str, Any],
    design_consensus: Mapping[str, Any],
    proposal: Mapping[str, Any],
    review: Mapping[str, Any],
    cycle: int,
) -> str:
    return f"""Agent W: you rejected Agent Z's candidate proposal in cycle {cycle}. You must now propose your own executable candidate set that satisfies your critique.

{STRICT_COUNTERPROPOSAL_PROTOCOL}

Do not only object. Either produce Agent Z-compatible candidate_specs grounded in the frozen X/Y design book, or produce candidate_specs=[] with a concise impossibility_certificate explaining why no faithful executable candidate set can satisfy the design book and shard target.
For parser compatibility, set the JSON field agent to "Z" and use the same output shape as Agent Z: status, agent, shard_id, target_count, candidate_specs, design_book_usage, proposal_summary.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

{ZW_GENERATION_POLICY}

{XY_CANDIDATE_SCHEMA}

CONTEXT_JSON:
```json
{prompt_json(context)}
```

FROZEN_XY_DESIGN_CONSENSUS:
```json
{prompt_json(design_consensus)}
```

PREVIOUS_AGENT_Z_PROPOSAL:
```json
{prompt_json(proposal)}
```

AGENT_W_REVIEW:
```json
{prompt_json(review)}
```

Return only Agent Z JSON.
"""


def prompt_z_reverse_review(
    context: Mapping[str, Any],
    design_consensus: Mapping[str, Any],
    original_proposal: Mapping[str, Any],
    counterproposal: Mapping[str, Any],
    cycle: int,
) -> str:
    return f"""Agent Z: critique Agent W's candidate counterproposal for cycle {cycle}.

{STRICT_COUNTERPROPOSAL_PROTOCOL}

You are now the reviewer. If Agent W's counterproposal is more faithful to the frozen X/Y design book and more executable than yours, set agree=true. If not, provide exact required revisions.
For parser compatibility, set the JSON field agent to "W" and use the same output shape as Agent W: status, agent, shard_id, agree, approved_candidate_ids, rejected_candidates, counterproposal_candidate_specs, design_book_audit, overall_reasoning_summary.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

{ZW_GENERATION_POLICY}

CONTEXT_JSON:
```json
{prompt_json(context)}
```

FROZEN_XY_DESIGN_CONSENSUS:
```json
{prompt_json(design_consensus)}
```

ORIGINAL_AGENT_Z_PROPOSAL:
```json
{prompt_json(original_proposal)}
```

AGENT_W_COUNTERPROPOSAL_JSON:
```json
{prompt_json(counterproposal)}
```

Return only Agent W JSON.
"""


def prompt_z_revision(
    context: Mapping[str, Any],
    design_consensus: Mapping[str, Any],
    proposal: Mapping[str, Any],
    review: Mapping[str, Any],
    cycle: int,
) -> str:
    return f"""Agent Z concrete material revision cycle {cycle}.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

{ZW_GENERATION_POLICY}

Revise the candidate set in response to Agent W. Preserve accepted candidates when still valid, replace rejected ones, and keep the shard target count. Every candidate must remain grounded in X/Y design_rule_ids.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

FROZEN_XY_DESIGN_CONSENSUS:
```json
{prompt_json(design_consensus)}
```

PREVIOUS_Z_PROPOSAL:
```json
{prompt_json(proposal)}
```

AGENT_W_REVIEW:
```json
{prompt_json(review)}
```

Return the same Agent Z proposal JSON shape with updated candidate_specs.
"""


def prompt_w_final(
    context: Mapping[str, Any],
    design_consensus: Mapping[str, Any],
    proposal: Mapping[str, Any],
    review: Mapping[str, Any],
) -> str:
    return f"""Agent W final concrete-material consensus.

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

{ZW_GENERATION_POLICY}

Write the final locked shard candidates. Do not introduce any candidate that Z/W have not reviewed. Candidate specs must remain generator-executable, design-rule-grounded, mechanism-template faithful, and blind.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

FROZEN_XY_DESIGN_CONSENSUS:
```json
{prompt_json(design_consensus)}
```

LATEST_Z_PROPOSAL:
```json
{prompt_json(proposal)}
```

LATEST_W_REVIEW:
```json
{prompt_json(review)}
```

Return JSON:
{{
  "status": "consensus",
  "agent": "W",
  "shard_id": {context["shard"]["index"]},
  "agreed_candidate_specs": ["candidate objects"],
  "rejected_candidates": [],
  "design_book_usage": {{
    "design_rule_ids_used": [],
    "rules_marked_unusable": [],
    "no_high_sun_prefilter": true
  }},
  "blind_lock_statement": "These candidates are locked before any new e_hull/SUN evaluation.",
  "debate_summary": "concise summary"
}}
"""


def prompt_x_materialization_repair(
    context: Mapping[str, Any],
    proposal: Mapping[str, Any],
    feedback: Mapping[str, Any],
    repair_round: int,
) -> str:
    return f"""Agent X materialization repair round {repair_round}.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

{XY_CANDIDATE_SCHEMA}

The controller performed a pre-evaluation materialization dry-run. No new e_hull, SUN, stability, or formation-energy labels were exposed. The dry-run found that this shard does not yet lock enough executable candidates.

Repair only executability while preserving the scientific mechanism coverage:
- Copy every LOCKED_CANDIDATE_SPECS_TO_PRESERVE item exactly into candidate_specs; these candidates already passed blind X/Y review and controller materialization.
- Replace failed candidates with executable alternatives.
- Do not call query_candidate_pool; do not introduce MP-pool queries or material_ids.
- If formula_probe_strings or formula_probes fail, repair the generator string/template/oxidation-state roles or use a parseable structure_dict.
- Prefer a new generator string over copying a failed one; preserve the same mechanism and cited principle boundary.
- Avoid reusing formulas already locked in this shard.
- Do not repair by forcing the same mechanism into a template that cannot realize its local motif; change the template only when the template_consistency_audit remains faithful.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

PREVIOUS_X_PROPOSAL_OR_CONSENSUS:
```json
{prompt_json(proposal)}
```

MATERIALIZATION_DRY_RUN_FEEDBACK:
```json
{prompt_json(feedback)}
```

Return the same Agent X proposal JSON shape with candidate_specs containing at least the shard target count and preserving the locked candidate specs where possible.
"""


def prompt_y_materialization_repair_review(
    context: Mapping[str, Any],
    proposal: Mapping[str, Any],
    feedback: Mapping[str, Any],
    repair_round: int,
) -> str:
    return f"""Agent Y materialization repair review round {repair_round}.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

Audit Agent X's repaired proposal. The controller dry-run feedback contains only executability information, not e_hull/SUN/stability labels.

Require:
- enough executable candidate specs for this shard;
- preserved LOCKED_CANDIDATE_SPECS_TO_PRESERVE candidates are kept exactly unless they violate the blind protocol;
- no MP-pool material_ids, MP-pool queries, or query_candidate_pool dependence;
- no candidate chosen by e_hull/SUN/formation-energy/stable_count;
- no unnecessary abandonment of the cited material principles;
- failed candidates replaced by faithful, materializable alternatives.
- formula_probe_string/formula_probe replacements should be rejected when they repeat a generator input that already failed without fixing template, roles, oxidation states, or structure.
- every retained or replacement candidate has a passing template_consistency_audit and does not use a generator template as an unfaithful proxy for the claimed mechanism.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

AGENT_X_REPAIRED_PROPOSAL:
```json
{prompt_json(proposal)}
```

MATERIALIZATION_DRY_RUN_FEEDBACK:
```json
{prompt_json(feedback)}
```

Return the same Agent Y review JSON shape. Set agree=true only if the repaired candidate_specs should proceed to final consensus and materialization dry-run.
"""


def prompt_z_materialization_repair(
    context: Mapping[str, Any],
    design_consensus: Mapping[str, Any],
    proposal: Mapping[str, Any],
    feedback: Mapping[str, Any],
    repair_round: int,
) -> str:
    return f"""Agent Z materialization repair round {repair_round}.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

{ZW_GENERATION_POLICY}

{XY_CANDIDATE_SCHEMA}

The controller performed a pre-evaluation materialization dry-run. No new e_hull, SUN, stability, or formation-energy labels were exposed. Repair only executability while preserving X/Y design-rule grounding.

Rules:
- Copy every LOCKED_CANDIDATE_SPECS_TO_PRESERVE item exactly into candidate_specs.
- Replace failed candidates with executable alternatives that cite design_rule_ids from the frozen design book.
- Do not call query_candidate_pool; do not introduce MP-pool queries or material_ids.
- If a formula_probe_string fails, repair template/roles/oxidation states or switch to a faithful structure_dict.
- Avoid reusing formulas already locked in this shard.
- Do not force a design rule into a template that X/Y marked as forbidden or unfaithful.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

FROZEN_XY_DESIGN_CONSENSUS:
```json
{prompt_json(design_consensus)}
```

PREVIOUS_Z_PROPOSAL_OR_CONSENSUS:
```json
{prompt_json(proposal)}
```

MATERIALIZATION_DRY_RUN_FEEDBACK:
```json
{prompt_json(feedback)}
```

Return the same Agent Z proposal JSON shape with candidate_specs containing at least the shard target count and preserving locked candidate specs where possible.
"""


def prompt_w_materialization_repair_review(
    context: Mapping[str, Any],
    design_consensus: Mapping[str, Any],
    proposal: Mapping[str, Any],
    feedback: Mapping[str, Any],
    repair_round: int,
) -> str:
    return f"""Agent W materialization repair review round {repair_round}.

{XY_HISTORY_RAG_REQUIREMENT}

{XY_BLIND_EVALUATION_POLICY}

{XY_GENERATOR_ONLY_POLICY}

{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}

{ZW_GENERATION_POLICY}

Audit Agent Z's repaired proposal. The controller dry-run feedback contains only executability information, not e_hull/SUN/stability labels.

Require:
- enough executable candidate specs for this shard;
- preserved LOCKED_CANDIDATE_SPECS_TO_PRESERVE candidates are kept exactly unless they violate the blind protocol;
- every retained or replacement candidate cites design_rule_ids and source principles;
- no MP-pool material_ids, MP-pool queries, or query_candidate_pool dependence;
- no candidate chosen by e_hull/SUN/formation-energy/stable_count;
- failed candidates replaced by faithful, materializable alternatives.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

FROZEN_XY_DESIGN_CONSENSUS:
```json
{prompt_json(design_consensus)}
```

AGENT_Z_REPAIRED_PROPOSAL:
```json
{prompt_json(proposal)}
```

MATERIALIZATION_DRY_RUN_FEEDBACK:
```json
{prompt_json(feedback)}
```

Return the same Agent W review JSON shape. Set agree=true only if the repaired candidate_specs should proceed to final consensus and materialization dry-run.
"""


MATERIAL_DESCRIPTION_PROMPT_KEYS = (
    "natural_language_description",
    "reduced_formula",
    "target_reduced_formula",
    "preferred_reduced_formula",
    "chemical_system",
    "target_family",
    "target_chemical_family",
    "source_principle_ids",
    "history_lessons_used",
    "elements_or_families",
    "charge_or_valence_constraints",
    "expected_local_motif",
    "generator_template",
    "generator_role_mapping",
    "template_formula_family",
    "why_template_is_faithful",
    "why_sun_likely",
    "why_not_duplicate",
    "known_risks",
)

DEFAULT_XY_SUN_CANDIDATE_QUEUE_SIZE = 4


def xy_sun_candidate_queue_size_from_context(context: Mapping[str, Any]) -> int:
    constraints = context.get("controller_constraints")
    if isinstance(constraints, Mapping):
        try:
            value = int(constraints.get("xy_sun_candidate_queue_size"))
        except Exception:
            value = DEFAULT_XY_SUN_CANDIDATE_QUEUE_SIZE
    else:
        value = DEFAULT_XY_SUN_CANDIDATE_QUEUE_SIZE
    return max(2, min(6, value))


def sun_candidate_queue_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_queue = (
        payload.get("sun_candidate_queue")
        or payload.get("ranked_sun_candidate_queue")
        or payload.get("candidate_material_descriptions")
        or payload.get("ranked_material_descriptions")
    )
    if not isinstance(raw_queue, list):
        raw_description = payload.get("material_description")
        if isinstance(raw_description, Mapping):
            nested_queue = (
                raw_description.get("sun_candidate_queue")
                or raw_description.get("ranked_sun_candidate_queue")
                or raw_description.get("candidate_material_descriptions")
            )
            raw_queue = nested_queue if isinstance(nested_queue, list) else []
        else:
            raw_queue = []
    queue: list[dict[str, Any]] = []
    for item in raw_queue:
        if isinstance(item, Mapping):
            normalized_item = dict(item)
            if not normalized_item.get("reduced_formula"):
                alias_formula = normalized_item.get("formula") or normalized_item.get("crystal_llm_formula")
                if alias_formula:
                    normalized_item["reduced_formula"] = alias_formula
            material = normalized_item.get("material_description")
            if isinstance(material, Mapping):
                if not normalized_item.get("generator_template"):
                    template = material.get("generator_template") or material.get("template_expectation")
                    if template:
                        normalized_item["generator_template"] = template
                if not normalized_item.get("reduced_formula"):
                    formula = template_formula_from_material_description({"material_description": material})
                    if formula:
                        normalized_item["reduced_formula"] = formula
            queue.append(normalized_item)
    return queue


def _formula_hint_from_material_description(description: Mapping[str, Any]) -> str:
    for key in (
        "reduced_formula",
        "target_reduced_formula",
        "preferred_reduced_formula",
        "formula",
        "crystal_llm_formula",
    ):
        formula = _normalize_formula_text(description.get(key))
        if formula:
            return formula
    return ""


def queue_item_reduced_formula(item: Mapping[str, Any]) -> str:
    formula = _normalize_formula_text(
        item.get("reduced_formula")
        or item.get("target_reduced_formula")
        or item.get("preferred_reduced_formula")
        or item.get("formula")
        or item.get("crystal_llm_formula")
    )
    if formula:
        return formula
    material = item.get("material_description")
    if isinstance(material, Mapping):
        formula = _formula_hint_from_material_description(material)
        if formula:
            return formula
        return _normalize_formula_text(template_formula_from_material_description({"material_description": material}))
    return ""


def selected_queue_item_from_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    queue = sun_candidate_queue_from_payload(payload)
    selected_id = str(payload.get("selected_candidate_id") or payload.get("selected_material_id") or "").strip()
    if selected_id:
        for item in queue:
            item_id = str(item.get("candidate_id") or item.get("id") or "").strip()
            if item_id == selected_id:
                return item
    return queue[0] if len(queue) == 1 else None


def selected_reduced_formula_from_payload(payload: Mapping[str, Any]) -> str:
    selected = selected_queue_item_from_payload(payload)
    if isinstance(selected, Mapping):
        formula = queue_item_reduced_formula(selected)
        if formula:
            return formula
    description = material_description_from_payload(payload)
    formula = _formula_hint_from_material_description(description)
    if formula:
        return formula
    return _normalize_formula_text(template_formula_from_material_description(payload))


def material_consensus_with_source_queue(
    consensus: Mapping[str, Any],
    source_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Carry the approved queue into a compact final consensus when Y omits it."""

    merged = dict(consensus)
    if material_payload_declares_no_valid_description(merged):
        return merged
    for key in ("selected_candidate_id", "selection_rationale"):
        if merged.get(key) in (None, "", [], {}):
            value = source_payload.get(key)
            if value not in (None, "", [], {}):
                merged[key] = value
    if not material_description_from_payload(merged):
        source_description = material_description_from_payload(source_payload)
        if source_description:
            merged["material_description"] = dict(source_description)
    merged_queue = sun_candidate_queue_from_payload(merged)
    source_queue = sun_candidate_queue_from_payload(source_payload)
    if not merged_queue and source_queue:
        merged_queue = [dict(item) for item in source_queue]
    if merged_queue:
        selected_id = str(merged.get("selected_candidate_id") or "").strip()
        top_description = material_description_from_payload(merged)
        source_by_id = {
            str(item.get("candidate_id") or item.get("id") or "").strip(): item
            for item in source_queue
            if str(item.get("candidate_id") or item.get("id") or "").strip()
        }
        enriched_queue: list[dict[str, Any]] = []
        for item in merged_queue:
            enriched = dict(item)
            item_id = str(enriched.get("candidate_id") or enriched.get("id") or "").strip()
            if not isinstance(enriched.get("material_description"), Mapping):
                source_item = source_by_id.get(item_id)
                source_material = source_item.get("material_description") if isinstance(source_item, Mapping) else None
                if isinstance(source_material, Mapping):
                    enriched["material_description"] = dict(source_material)
                elif selected_id and item_id == selected_id and top_description:
                    enriched["material_description"] = dict(top_description)
            enriched_queue.append(enriched)
        merged["sun_candidate_queue"] = enriched_queue
    return merged


def payload_with_controller_iteration(payload: Mapping[str, Any], *, iteration: int) -> dict[str, Any]:
    """Trust the controller loop for iteration numbering, not model-authored JSON."""

    normalized = dict(payload)
    normalized["iteration"] = iteration
    for key in ("counterproposal_material_description",):
        nested = normalized.get(key)
        if isinstance(nested, Mapping) and (
            "iteration" in nested or "status" in nested or "agent" in nested
        ):
            nested_normalized = dict(nested)
            nested_normalized["iteration"] = iteration
            normalized[key] = nested_normalized
    return normalized


def approved_material_consensus_from_payload(
    proposal: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    iteration: int,
) -> dict[str, Any]:
    """Convert a clean peer approval into final X/Y consensus without another LLM call."""

    consensus = dict(proposal)
    consensus["status"] = "material_description_consensus"
    consensus["agent"] = "Y"
    consensus["iteration"] = iteration
    summary = (
        review.get("overall_reasoning_summary")
        or review.get("review_summary")
        or proposal.get("proposal_summary")
        or proposal.get("overall_reasoning_summary")
        or "X/Y approved the ranked material queue and selected material."
    )
    consensus.setdefault("debate_summary", short_text(summary, 1200))
    if review.get("risk_audit") not in (None, "", [], {}):
        consensus.setdefault("final_review_risk_audit", review.get("risk_audit"))
    return payload_with_controller_iteration(
        material_consensus_with_source_queue(consensus, proposal),
        iteration=iteration,
    )


def compact_sun_candidate_queue_for_prompt(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for item in sun_candidate_queue_from_payload(payload)[:4]:
        raw_material = item.get("material_description")
        material_payload = {"material_description": raw_material} if isinstance(raw_material, Mapping) else item
        full_material = compact_material_description_for_prompt(material_payload).get("material_description", {})
        compact_material = {
            key: value
            for key, value in {
                "natural_language_description": short_text(full_material.get("natural_language_description"), 160)
                if isinstance(full_material, Mapping)
                else None,
                "target_family": full_material.get("target_family") if isinstance(full_material, Mapping) else None,
                "generator_template": full_material.get("generator_template") if isinstance(full_material, Mapping) else None,
                "generator_role_mapping": full_material.get("generator_role_mapping") if isinstance(full_material, Mapping) else None,
                "expected_local_motif": short_text(full_material.get("expected_local_motif"), 120)
                if isinstance(full_material, Mapping)
                else None,
                "why_sun_likely": short_text(full_material.get("why_sun_likely"), 120)
                if isinstance(full_material, Mapping)
                else None,
            }.items()
            if value not in (None, "", [], {})
        }
        queue.append(
            {
                key: value
                for key, value in {
                    "candidate_id": item.get("candidate_id") or item.get("id"),
                    "rank": item.get("rank"),
                    "reduced_formula": item.get("reduced_formula") or item.get("formula"),
                    "acquisition_mode": item.get("acquisition_mode"),
                    "basin_key": item.get("basin_key"),
                    "estimated_sun_probability": short_text(item.get("estimated_sun_probability"), 100),
                    "expected_e_hull_band": item.get("expected_e_hull_band"),
                    "acquisition_rationale": short_text(item.get("acquisition_rationale") or item.get("why_ranked_here"), 160),
                    "duplicate_audit": short_text(item.get("duplicate_audit"), 120),
                    "template_feasibility_audit": short_text(item.get("template_feasibility_audit"), 120),
                    "main_risks": item.get("main_risks")[:2] if isinstance(item.get("main_risks"), list) else item.get("main_risks"),
                    "material_description": compact_material,
                }.items()
                if value not in (None, "", [], {})
            }
        )
    return queue


def compact_impossibility_certificate_for_prompt(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        return short_text(value, 260)
    if isinstance(value, Mapping):
        compact: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 16:
                break
            if item in (None, "", [], {}):
                continue
            compact[str(key)] = compact_impossibility_certificate_for_prompt(item, depth=depth + 1)
        return compact
    if isinstance(value, list):
        return [compact_impossibility_certificate_for_prompt(item, depth=depth + 1) for item in value[:12]]
    return short_text(value, 520) if isinstance(value, str) else value


def compact_material_description_for_prompt(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_description = payload.get("material_description")
    description = dict(raw_description) if isinstance(raw_description, Mapping) else dict(payload)
    compact_description: dict[str, Any] = {}
    for key in MATERIAL_DESCRIPTION_PROMPT_KEYS:
        value = description.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, str):
            compact_description[key] = short_text(value, 260 if key == "natural_language_description" else 180)
        elif isinstance(value, list):
            compact_description[key] = [short_text(item, 140) for item in value[:3]]
        elif isinstance(value, Mapping):
            compact_description[key] = dict(value)
        else:
            compact_description[key] = value
    compact_payload = {
        "status": payload.get("status"),
        "agent": payload.get("agent"),
        "iteration": payload.get("iteration"),
        "selected_candidate_id": payload.get("selected_candidate_id") or payload.get("selected_material_id"),
        "selection_rationale": short_text(payload.get("selection_rationale"), 220),
        "sun_candidate_queue": compact_sun_candidate_queue_for_prompt(payload),
        "queue_audit": payload.get("queue_audit") if isinstance(payload.get("queue_audit"), Mapping) else None,
        "material_description": compact_description,
        "impossibility_certificate": compact_impossibility_certificate_for_prompt(payload.get("impossibility_certificate"))
        if isinstance(payload.get("impossibility_certificate"), Mapping)
        else None,
        "proposal_summary": short_text(payload.get("proposal_summary"), 180),
        "sun_optimization_rationale": short_text(payload.get("sun_optimization_rationale"), 180),
        "debate_summary": short_text(payload.get("debate_summary"), 180),
    }
    filtered = {key: value for key, value in compact_payload.items() if value not in (None, "", [], {})}
    if str(payload.get("status") or "").strip().lower() in {"no_valid_material_description", "no_valid_material_consensus"}:
        filtered["material_description"] = compact_description
    return filtered


def compact_material_description_for_review_prompt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Smaller proposal view for X/Y peer review prompts."""

    compact = compact_material_description_for_prompt(payload)
    if compact.get("selection_rationale"):
        compact["selection_rationale"] = short_text(compact.get("selection_rationale"), 100)
    if compact.get("proposal_summary"):
        compact["proposal_summary"] = short_text(compact.get("proposal_summary"), 90)
    compact.pop("sun_optimization_rationale", None)
    compact.pop("debate_summary", None)
    compact.pop("queue_audit", None)
    material = compact.get("material_description")
    if isinstance(material, Mapping):
        elements = material.get("elements_or_families")
        if isinstance(elements, Mapping):
            elements = {
                key: [short_text(item, 40) for item in value[:6]] if isinstance(value, list) else short_text(value, 80)
                for key, value in {
                    "required": elements.get("required"),
                    "forbidden": elements.get("forbidden"),
                }.items()
                if value not in (None, "", [], {})
            }
        compact["material_description"] = {
            key: value
            for key, value in {
                "natural_language_description": short_text(material.get("natural_language_description"), 100),
                "target_family": material.get("target_family"),
                "elements_or_families": elements,
                "charge_or_valence_constraints": [short_text(item, 60) for item in material.get("charge_or_valence_constraints")[:2]]
                if isinstance(material.get("charge_or_valence_constraints"), list)
                else short_text(material.get("charge_or_valence_constraints"), 100),
                "expected_local_motif": short_text(material.get("expected_local_motif"), 80),
                "generator_template": material.get("generator_template"),
                "generator_role_mapping": material.get("generator_role_mapping"),
                "why_sun_likely": short_text(material.get("why_sun_likely"), 80),
                "known_risks": [short_text(item, 60) for item in material.get("known_risks")[:1]]
                if isinstance(material.get("known_risks"), list)
                else short_text(material.get("known_risks"), 80),
            }.items()
            if value not in (None, "", [], {})
        }
    queue = compact.get("sun_candidate_queue")
    if isinstance(queue, list):
        compact["sun_candidate_queue_count"] = len(queue)
        compact["sun_candidate_queue_formula_order"] = [
            {
                key: value
                for key, value in {
                    "candidate_id": item.get("candidate_id") if isinstance(item, Mapping) else None,
                    "rank": item.get("rank") if isinstance(item, Mapping) else None,
                    "reduced_formula": item.get("reduced_formula") if isinstance(item, Mapping) else None,
                }.items()
                if value not in (None, "", [], {})
            }
            for item in queue[:6]
            if isinstance(item, Mapping)
        ]
        compact["sun_candidate_queue"] = [
            {
                key: value
                for key, value in {
                    "candidate_id": item.get("candidate_id") if isinstance(item, Mapping) else None,
                    "rank": item.get("rank") if isinstance(item, Mapping) else None,
                    "reduced_formula": item.get("reduced_formula") if isinstance(item, Mapping) else None,
                    "acquisition_mode": item.get("acquisition_mode") if isinstance(item, Mapping) else None,
                    "expected_e_hull_band": item.get("expected_e_hull_band") if isinstance(item, Mapping) else None,
                    "summary": short_text(
                        (item.get("material_description") or {}).get("natural_language_description")
                        if isinstance(item, Mapping) and isinstance(item.get("material_description"), Mapping)
                        else "",
                        80,
                    ),
                }.items()
                if value not in (None, "", [], {})
            }
            for item in queue[:4]
            if isinstance(item, Mapping)
        ]
    return compact


def compact_material_description_for_generator_prompt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Minimal X/Y consensus view for Z/W generator translation."""

    compact = compact_material_description_for_review_prompt(payload)
    compact.pop("proposal_summary", None)
    compact.pop("selection_rationale", None)
    material = compact.get("material_description")
    if isinstance(material, Mapping):
        compact["material_description"] = {
            key: value
            for key, value in {
                "natural_language_description": short_text(material.get("natural_language_description"), 80),
                "target_family": short_text(material.get("target_family"), 80),
                "elements_or_families": material.get("elements_or_families"),
                "charge_or_valence_constraints": material.get("charge_or_valence_constraints"),
                "expected_local_motif": short_text(material.get("expected_local_motif"), 70),
                "generator_template": material.get("generator_template"),
                "generator_role_mapping": material.get("generator_role_mapping"),
                "known_risks": material.get("known_risks"),
            }.items()
            if value not in (None, "", [], {})
        }
    queue = compact.get("sun_candidate_queue")
    if isinstance(queue, list):
        compact["sun_candidate_queue"] = [
            {
                key: value
                for key, value in {
                    "candidate_id": item.get("candidate_id") if isinstance(item, Mapping) else None,
                    "rank": item.get("rank") if isinstance(item, Mapping) else None,
                    "reduced_formula": item.get("reduced_formula") if isinstance(item, Mapping) else None,
                    "acquisition_mode": item.get("acquisition_mode") if isinstance(item, Mapping) else None,
                    "summary": short_text(item.get("summary"), 60) if isinstance(item, Mapping) else None,
                }.items()
                if value not in (None, "", [], {})
            }
            for item in queue[:2]
            if isinstance(item, Mapping)
        ]
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_strategy_cooldowns_for_prompt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    blocked_systems = value.get("blocked_chemical_systems")
    blocked_patterns = value.get("blocked_family_patterns")

    def cooldown_prompt_rank(item: Mapping[str, Any]) -> tuple[int, int, str]:
        try:
            failure_count = int(item.get("failure_count") or 0)
        except Exception:
            failure_count = 0
        recent_iteration = 0
        iterations = item.get("source_iterations")
        if isinstance(iterations, list):
            numeric_iterations = [int(iteration) for iteration in iterations if isinstance(iteration, int)]
            if numeric_iterations:
                recent_iteration = max(numeric_iterations)
        label = str(item.get("chemical_system") or item.get("family_pattern") or "")
        return -failure_count, -recent_iteration, label

    compact_systems = []
    if isinstance(blocked_systems, list):
        sorted_systems = sorted(
            [item for item in blocked_systems if isinstance(item, Mapping)],
            key=cooldown_prompt_rank,
        )
        for item in sorted_systems[:XY_STRATEGY_COOLDOWN_VISIBLE_LIMIT]:
            if not isinstance(item, Mapping):
                continue
            compact_systems.append(
                {
                    key: val
                    for key, val in {
                        "chemical_system": item.get("chemical_system"),
                        "family_pattern": item.get("family_pattern"),
                        "failure_count": item.get("failure_count"),
                        "known_formula_examples": item.get("known_formula_examples", [])[:3]
                        if isinstance(item.get("known_formula_examples"), list)
                        else item.get("known_formula_examples"),
                        "failure_markers": item.get("failure_markers", [])[:3]
                        if isinstance(item.get("failure_markers"), list)
                        else item.get("failure_markers"),
                    }.items()
                    if val not in (None, "", [], {})
                }
            )
    compact_patterns = []
    if isinstance(blocked_patterns, list):
        sorted_patterns = sorted(
            [item for item in blocked_patterns if isinstance(item, Mapping)],
            key=cooldown_prompt_rank,
        )
        for item in sorted_patterns[:XY_STRATEGY_COOLDOWN_VISIBLE_LIMIT]:
            if not isinstance(item, Mapping):
                continue
            compact_patterns.append(
                {
                    key: val
                    for key, val in {
                        "family_pattern": item.get("family_pattern"),
                        "chemical_systems": item.get("chemical_systems", [])[:4]
                        if isinstance(item.get("chemical_systems"), list)
                        else item.get("chemical_systems"),
                        "failure_count": item.get("failure_count"),
                        "distinct_system_count": item.get("distinct_system_count"),
                    }.items()
                    if val not in (None, "", [], {})
                }
            )
    return {
        key: val
        for key, val in {
            "policy": value.get("policy"),
            "cooldown_reason": short_text(value.get("cooldown_reason"), 220),
            "blocked_chemical_systems": compact_systems,
            "blocked_family_patterns": compact_patterns,
            "required_next_move": short_text(value.get("required_next_move"), 260),
            "do_not_solve_by": value.get("do_not_solve_by"),
            "omitted_blocked_chemical_system_count": max(
                0, len(blocked_systems) - len(compact_systems)
            )
            if isinstance(blocked_systems, list)
            else None,
            "omitted_blocked_family_pattern_count": max(0, len(blocked_patterns) - len(compact_patterns))
            if isinstance(blocked_patterns, list)
            else None,
        }.items()
        if val not in (None, "", [], {})
    }


def compact_strategy_constraints_for_prompt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    blocked = value.get("blocked_candidate_formula_reasons")
    compact_blocked: dict[str, Any] = {}
    if isinstance(blocked, Mapping):
        for index, (formula, reasons) in enumerate(blocked.items()):
            if index >= 4:
                break
            compact_blocked[str(formula)] = [short_text(item, 90) for item in reasons[:2]] if isinstance(reasons, list) else short_text(reasons, 120)
    return {
        key: val
        for key, val in {
            "schema_version": value.get("schema_version"),
            "binding": value.get("binding"),
            "queue_min_legal_items": value.get("queue_min_legal_items"),
            "queue_max_items": value.get("queue_max_items"),
            "required_acquisition_mode": value.get("required_acquisition_mode"),
            "latest_strategy_order_enforced": value.get("latest_strategy_order_enforced"),
            "ordered_candidate_formulas": value.get("ordered_candidate_formulas")[:6]
            if isinstance(value.get("ordered_candidate_formulas"), list)
            else value.get("ordered_candidate_formulas"),
            "legal_ordered_candidate_formulas": value.get("legal_ordered_candidate_formulas")[:6]
            if isinstance(value.get("legal_ordered_candidate_formulas"), list)
            else value.get("legal_ordered_candidate_formulas"),
            "first_required_formula": value.get("first_required_formula"),
            "blocked_candidate_formula_reasons": compact_blocked,
            "search_policy_supersedes_latest_order": value.get("search_policy_supersedes_latest_order"),
            "requirements": value.get("requirements")[:5] if isinstance(value.get("requirements"), list) else value.get("requirements"),
            "mattergen_target_policy": short_text(value.get("mattergen_target_policy"), 180),
        }.items()
        if val not in (None, "", [], {})
    }


def compact_generator_context_for_prompt(context: Mapping[str, Any]) -> dict[str, Any]:
    constraints = context.get("controller_constraints")
    constraints = constraints if isinstance(constraints, Mapping) else {}
    search_policy = constraints.get("search_policy") if isinstance(constraints.get("search_policy"), Mapping) else {}
    mattergen_status = (
        constraints.get("mattergen_operational_status")
        if isinstance(constraints.get("mattergen_operational_status"), Mapping)
        else {}
    )
    latest = (
        context.get("latest_xy_strategy_constraints")
        if isinstance(context.get("latest_xy_strategy_constraints"), Mapping)
        else {}
    )
    active = context.get("active_principle_program") if isinstance(context.get("active_principle_program"), Mapping) else {}
    failed_formulas = constraints.get("failed_or_used_reduced_formula_hints")
    if not isinstance(failed_formulas, list):
        failed_formulas = constraints.get("failed_or_used_reduced_formulas")
    failed_volume = constraints.get("failed_volume_template_boundaries")
    compact_constraints = {
        "no_mp_pool": constraints.get("no_mp_pool"),
        "mattergen_backend_enabled": constraints.get("mattergen_backend_enabled"),
        "mattergen_request_defaults": constraints.get("mattergen_request_defaults"),
        "mattergen_operational_status": {
            key: mattergen_status.get(key)
            for key in ("backend", "evidence", "policy", "success_formula", "success_e_hull")
            if mattergen_status.get(key) not in (None, "", [], {})
        },
        "search_policy": {
            key: search_policy.get(key)
            for key in ("backend", "current_search_mode", "force_escape_triggered")
            if search_policy.get(key) not in (None, "", [], {})
        },
        "avoid_reduced_formula_repeats": constraints.get("avoid_reduced_formula_repeats"),
        "failed_or_used_reduced_formulas": failed_formulas[:XY_CONTEXT_FAILED_OR_USED_FORMULA_HINT_LIMIT]
        if isinstance(failed_formulas, list)
        else [],
        "failed_or_used_reduced_formula_total": constraints.get("failed_or_used_reduced_formula_total"),
        "failed_or_used_reduced_formula_hint_policy": constraints.get("failed_or_used_reduced_formula_hint_policy"),
        "failed_volume_template_boundaries": failed_volume[:2] if isinstance(failed_volume, list) else [],
        "forbidden_evaluator_null_elements": constraints.get("forbidden_evaluator_null_elements"),
        "strategy_cooldowns": _compact_strategy_cooldowns_for_prompt(constraints.get("strategy_cooldowns")),
        "strategy_constraints": compact_strategy_constraints_for_prompt(constraints.get("strategy_constraints")),
        "z_w_can_return_to_xy": constraints.get("z_w_can_return_to_xy"),
    }
    compact_latest = {
        key: latest.get(key)
        for key in ("source_iteration", "next_strategy", "candidate_formula_tokens", "failure_boundaries")
        if latest.get(key) not in (None, "", [], {})
    }
    if isinstance(compact_latest.get("failure_boundaries"), list):
        compact_latest["failure_boundaries"] = [short_text(item, 90) for item in compact_latest["failure_boundaries"][:3]]
    return {
        key: value
        for key, value in {
            "schema_version": context.get("schema_version"),
            "mode": context.get("mode"),
            "iteration": context.get("iteration"),
            "generation_protocol": context.get("generation_protocol"),
            "generator_backend": context.get("generator_backend"),
            "candidate_count_this_iteration": context.get("candidate_count_this_iteration"),
            "controller_constraints": {k: v for k, v in compact_constraints.items() if v not in (None, "", [], {})},
            "latest_xy_strategy_constraints": compact_latest,
            "active_principle_program": {
                key: short_text(active.get(key), 120)
                for key in ("current_principle_statement", "micro_mechanism")
                if active.get(key) not in (None, "", [], {})
            },
            "strict_sun_definition": context.get("strict_sun_definition"),
        }.items()
        if value not in (None, "", [], {})
    }


def compact_xy_material_context_for_prompt(context: Mapping[str, Any]) -> dict[str, Any]:
    """Small prompt-visible X/Y context; full controller state remains enforced in code."""

    constraints = context.get("controller_constraints")
    constraints = constraints if isinstance(constraints, Mapping) else {}
    search_policy = constraints.get("search_policy") if isinstance(constraints.get("search_policy"), Mapping) else {}
    mattergen_status = (
        constraints.get("mattergen_operational_status")
        if isinstance(constraints.get("mattergen_operational_status"), Mapping)
        else {}
    )
    latest = (
        context.get("latest_xy_strategy_constraints")
        if isinstance(context.get("latest_xy_strategy_constraints"), Mapping)
        else {}
    )
    failed_formulas = constraints.get("failed_or_used_reduced_formula_hints")
    if not isinstance(failed_formulas, list):
        failed_formulas = constraints.get("failed_or_used_reduced_formulas")
    failed_volume = constraints.get("failed_volume_template_boundaries")
    candidate_formulas = latest.get("next_strategy_candidate_formulas") or latest.get("candidate_formula_tokens")
    active = context.get("active_principle_program") if isinstance(context.get("active_principle_program"), Mapping) else {}
    return {
        key: value
        for key, value in {
            "schema_version": context.get("schema_version"),
            "mode": context.get("mode"),
            "iteration": context.get("iteration"),
            "generation_protocol": context.get("generation_protocol"),
            "generator_backend": context.get("generator_backend"),
            "candidate_count_this_iteration": context.get("candidate_count_this_iteration"),
            "strict_sun_definition": short_text(context.get("strict_sun_definition"), 220),
            "controller_constraints": {
                key: value
                for key, value in {
                    "control_candidate_requested": constraints.get("control_candidate_requested"),
                    "avoid_reduced_formula_repeats": constraints.get("avoid_reduced_formula_repeats"),
                    "search_policy": {
                        key: search_policy.get(key)
                        for key in (
                            "current_search_mode",
                            "backend",
                            "force_escape_triggered",
                            "search_policy_70_20_10",
                        )
                        if search_policy.get(key) not in (None, "", [], {})
                    },
                    "mattergen_request_defaults": constraints.get("mattergen_request_defaults"),
                    "mattergen_operational_status": {
                        key: mattergen_status.get(key)
                        for key in ("backend", "evidence", "policy", "success_formula", "success_e_hull")
                        if mattergen_status.get(key) not in (None, "", [], {})
                    },
                    "failed_or_used_reduced_formulas": failed_formulas[:XY_CONTEXT_FAILED_OR_USED_FORMULA_HINT_LIMIT]
                    if isinstance(failed_formulas, list)
                    else [],
                    "failed_or_used_reduced_formula_total": constraints.get("failed_or_used_reduced_formula_total"),
                    "failed_or_used_reduced_formula_hint_policy": constraints.get("failed_or_used_reduced_formula_hint_policy"),
                    "failed_volume_template_boundaries": failed_volume[:2] if isinstance(failed_volume, list) else [],
                    "failed_volume_template_boundary_total": constraints.get("failed_volume_template_boundary_total"),
                    "forbidden_evaluator_null_elements": constraints.get("forbidden_evaluator_null_elements"),
                    "strategy_cooldowns": _compact_strategy_cooldowns_for_prompt(
                        constraints.get("strategy_cooldowns")
                    ),
                    "strategy_constraints": compact_strategy_constraints_for_prompt(
                        constraints.get("strategy_constraints")
                    ),
                }.items()
                if value not in (None, "", [], {})
            },
            "latest_xy_strategy_constraints": {
                key: value
                for key, value in {
                    "source_iteration": latest.get("source_iteration"),
                    "next_strategy": short_text(latest.get("next_strategy"), 520),
                    "next_strategy_candidate_formulas": candidate_formulas[:12]
                    if isinstance(candidate_formulas, list)
                    else candidate_formulas,
                    "failure_boundaries": [short_text(item, 120) for item in latest.get("failure_boundaries", [])[:4]]
                    if isinstance(latest.get("failure_boundaries"), list)
                    else short_text(latest.get("failure_boundaries"), 240),
                }.items()
                if value not in (None, "", [], {})
            },
            "active_principle_program": {
                key: short_text(active.get(key), 120)
                for key in ("current_principle_statement", "micro_mechanism")
                if active.get(key) not in (None, "", [], {})
            },
        }.items()
        if value not in (None, "", [], {})
    }


def compact_review_for_prompt(review: Mapping[str, Any]) -> dict[str, Any]:
    raw_counterproposal = review.get("counterproposal_material_description")
    if isinstance(raw_counterproposal, Mapping):
        counterproposal_payload = (
            raw_counterproposal
            if "material_description" in raw_counterproposal or "impossibility_certificate" in raw_counterproposal
            else {"material_description": raw_counterproposal}
        )
        compact_counterproposal = compact_material_description_for_prompt(counterproposal_payload)
    else:
        compact_counterproposal = None
    return {
        key: value
        for key, value in {
            "status": review.get("status"),
            "agent": review.get("agent"),
            "iteration": review.get("iteration"),
            "agree": review.get("agree"),
            "approved": review.get("approved"),
            "required_revision": short_text(review.get("required_revision"), 520),
            "risk_audit": review.get("risk_audit") if isinstance(review.get("risk_audit"), Mapping) else None,
            "counterproposal_material_description": compact_counterproposal,
            "overall_reasoning_summary": short_text(review.get("overall_reasoning_summary"), 650),
        }.items()
        if value not in (None, "", [], {})
    }


def compact_review_for_counterproposal_prompt(review: Mapping[str, Any]) -> dict[str, Any]:
    """Keep Y's rejection rationale without echoing its draft counterproposal back to itself."""

    compact = compact_review_for_prompt(review)
    compact.pop("counterproposal_material_description", None)
    return compact


def compact_mattergen_request_for_prompt(request: Mapping[str, Any]) -> dict[str, Any]:
    properties = request.get("properties_to_condition_on")
    filters = request.get("filters")
    compact_properties: dict[str, Any] = {}
    if isinstance(properties, Mapping):
        compact_properties = {
            key: properties.get(key)
            for key in ("chemical_system", "energy_above_hull")
            if properties.get(key) not in (None, "", [], {})
        }
    compact_filters: dict[str, Any] = {}
    if isinstance(filters, Mapping):
        for key in (
            "chemical_system",
            "require_chemical_system_exact",
            "allowed_elements",
            "required_elements",
            "excluded_elements",
            "max_sites",
            "min_sites",
            "min_volume_per_atom",
            "max_volume_per_atom",
            "target_reduced_formula",
            "require_target_reduced_formula",
            "deduplicate_reduced_formula",
            "exclude_known_formulas",
        ):
            if filters.get(key) not in (None, "", [], {}):
                compact_filters[key] = filters.get(key)
        excluded = filters.get("exclude_reduced_formulas")
        if isinstance(excluded, list) and excluded:
            compact_filters["exclude_reduced_formulas"] = excluded[-8:]
    forbidden_top_level = [
        key for key in ("chemical_system", "energy_above_hull") if request.get(key) not in (None, "", [], {})
    ]
    return {
        key: value
        for key, value in {
            "backend": request.get("backend"),
            "request_id": short_text(request.get("request_id"), 100),
            "properties_to_condition_on": compact_properties or None,
            "filters": compact_filters or None,
            "target_count": request.get("target_count"),
            "batch_size": request.get("batch_size"),
            "num_batches": request.get("num_batches"),
            "diffusion_guidance_factor": request.get("diffusion_guidance_factor"),
            "target_reduced_formula": request.get("target_reduced_formula"),
            "require_target_reduced_formula": request.get("require_target_reduced_formula"),
            "forbidden_top_level_fields_present": forbidden_top_level or None,
        }.items()
        if value not in (None, "", [], {})
    }


def compact_candidate_spec_for_prompt(spec: Mapping[str, Any]) -> dict[str, Any]:
    audit = spec.get("template_consistency_audit")
    if isinstance(audit, Mapping):
        mechanism_local_motif = (
            audit.get("mechanism_local_motif")
            or audit.get("motif_notes")
            or audit.get("local_motif")
            or audit.get("claimed_motif")
        )
        required_coordination_or_polyhedra = (
            audit.get("required_coordination_or_polyhedra")
            or audit.get("polyhedra_notes")
            or audit.get("coordination_notes")
            or audit.get("coordination_or_polyhedra")
            or audit.get("required_coordination")
        )
        why_template_is_faithful = (
            audit.get("why_template_is_faithful")
            or audit.get("faithfulness_notes")
            or audit.get("faithfulness_reason")
            or audit.get("audit_reasoning")
            or audit.get("why_faithful")
        )
        compact_audit = {
            key: value
            for key, value in {
                "chosen_template": audit.get("chosen_template"),
                "template_realizes_motif": audit.get("template_realizes_motif"),
                "unsupported_motif_substitution": audit.get("unsupported_motif_substitution"),
                "mechanism_local_motif": short_text(mechanism_local_motif, 320),
                "required_coordination_or_polyhedra": required_coordination_or_polyhedra,
                "why_template_is_faithful": short_text(why_template_is_faithful, 420),
                "generator_limitations": audit.get("generator_limitations"),
            }.items()
            if value not in (None, "", [], {})
        }
    else:
        compact_audit = None
    mattergen_requests = spec.get("mattergen_requests")
    compact_mattergen_requests = None
    if isinstance(mattergen_requests, list):
        compact_mattergen_requests = [
            compact_mattergen_request_for_prompt(item) for item in mattergen_requests[:1] if isinstance(item, Mapping)
        ]

    return {
        key: value
        for key, value in {
            "id": spec.get("id") or spec.get("candidate_id"),
            "candidate_id": spec.get("candidate_id"),
            "source": spec.get("source"),
            "count": spec.get("count"),
            "target_reduced_formula": spec.get("target_reduced_formula"),
            "require_target_reduced_formula": spec.get("require_target_reduced_formula"),
            "formula_probe_strings": spec.get("formula_probe_strings"),
            "formula_probes": spec.get("formula_probes"),
            "mattergen_requests": compact_mattergen_requests,
            "generator_template": spec.get("generator_template"),
            "generator_role_mapping": spec.get("generator_role_mapping"),
            "template_formula_family": spec.get("template_formula_family"),
            "expected_local_motif": short_text(spec.get("expected_local_motif"), 420),
            "why_template_is_faithful": short_text(spec.get("why_template_is_faithful"), 520),
            "structure_dicts": "[present]" if spec.get("structure_dicts") else None,
            "design_rule_ids": spec.get("design_rule_ids"),
            "source_principle_ids": spec.get("source_principle_ids"),
            "mechanism_rationale": short_text(spec.get("mechanism_rationale"), 520),
            "template_consistency_audit": compact_audit,
        }.items()
        if value not in (None, "", [], {})
    }


def compact_candidate_payload_for_prompt(payload: Mapping[str, Any]) -> dict[str, Any]:
    specs = [compact_candidate_spec_for_prompt(spec) for spec in candidate_specs_from_payload(payload)[:3]]
    feasibility = payload.get("feasibility_assessment")
    return {
        key: value
        for key, value in {
            "status": payload.get("status"),
            "agent": payload.get("agent"),
            "iteration": payload.get("iteration"),
            "candidate_specs": specs,
            "feasibility_assessment": feasibility if isinstance(feasibility, Mapping) else None,
            "proposal_summary": short_text(payload.get("proposal_summary"), 420),
            "feasibility_summary": short_text(payload.get("feasibility_summary"), 420),
        }.items()
        if value not in (None, "", [], {})
    }


def compact_candidate_review_for_prompt(review: Mapping[str, Any]) -> dict[str, Any]:
    rejected = review.get("rejected_candidates")
    if isinstance(rejected, list):
        compact_rejected = [
            {
                key: value
                for key, value in {
                    "candidate_id": item.get("candidate_id") if isinstance(item, Mapping) else None,
                    "reason": short_text(item.get("reason"), 360) if isinstance(item, Mapping) else None,
                    "required_revision": short_text(item.get("required_revision"), 420) if isinstance(item, Mapping) else None,
                }.items()
                if value not in (None, "", [], {})
            }
            for item in rejected[:4]
            if isinstance(item, Mapping)
        ]
    else:
        compact_rejected = None
    xy_repair = review.get("xy_repair_feedback")
    return {
        key: value
        for key, value in {
            "status": review.get("status"),
            "agent": review.get("agent"),
            "iteration": review.get("iteration"),
            "agree": review.get("agree"),
            "return_to_xy": review.get("return_to_xy"),
            "approved_candidate_ids": review.get("approved_candidate_ids"),
            "rejected_candidates": compact_rejected,
            "xy_repair_feedback": xy_repair if isinstance(xy_repair, Mapping) else None,
            "candidate_audit": review.get("candidate_audit") if isinstance(review.get("candidate_audit"), Mapping) else None,
            "overall_reasoning_summary": short_text(review.get("overall_reasoning_summary"), 650),
        }.items()
        if value not in (None, "", [], {})
    }


def _compact_feedback_summary(value: Any, *, max_chars: int = 260) -> Any:
    if not isinstance(value, Mapping):
        return short_text(value, max_chars)
    keep: dict[str, Any] = {}
    for key in (
        "status",
        "agent",
        "iteration",
        "source",
        "candidate_count",
        "rejected_candidate_count",
        "approved_candidate_count",
        "candidate_ids",
        "design_rule_count",
        "agree",
        "return_to_xy",
    ):
        if value.get(key) not in (None, "", [], {}):
            keep[key] = value.get(key)
    for key in ("summary", "proposal_summary", "overall_reasoning_summary", "controller_feedback", "reason"):
        if value.get(key) not in (None, "", [], {}):
            keep[key] = short_text(value.get(key), max_chars)
    return keep or {"summary": short_text(value, max_chars)}


def _compact_executable_rules_for_generator_feedback(rules: Any) -> Any:
    if not isinstance(rules, Mapping):
        return None
    backend = str(rules.get("generator_backend") or "").strip().lower()
    if backend == "mattergen":
        return {
            "generator_backend": "mattergen",
            "candidate_contract": "source=generator; count=1; exactly_one_input_field=mattergen_requests",
            "request_required_paths": [
                "backend='mattergen'",
                "properties_to_condition_on.chemical_system",
                "properties_to_condition_on.energy_above_hull=0.0",
                "filters chemical_system/allowed_elements/required_elements/max_sites",
                "diffusion_guidance_factor",
            ],
            "forbidden_request_top_level_fields": ["chemical_system", "energy_above_hull"],
            "forbidden_with_mattergen": "formula_probes, structure_dicts, manual cells/coords, mp_pool/material_ids/query",
            "target_formula_policy": "sequential target_reduced_formula is soft; require_target_reduced_formula=false",
        }
    return {
        key: value
        for key, value in {
            "candidate_contract": rules.get("candidate_contract"),
            "template_only": rules.get("template_only"),
            "controller_rule": short_text(rules.get("controller_rule"), 360),
            "forbidden_common_aliases": rules.get("forbidden_common_aliases"),
            "template_consistency_audit_required_keys": rules.get("template_consistency_audit_required_keys"),
        }.items()
        if value not in (None, "", [], {})
    }


def compact_generator_feedback_for_prompt(feedback: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not feedback:
        return None
    if not isinstance(feedback, Mapping):
        return {"feedback": short_text(feedback, 1000)}
    errors = feedback.get("generator_errors")
    if not isinstance(errors, list):
        errors = feedback.get("materialization_errors") if isinstance(feedback.get("materialization_errors"), list) else []
    compact_errors = [short_text(item, 260) for item in errors[:8]]
    omitted_errors = int(feedback.get("omitted_error_count") or 0)
    if isinstance(errors, list) and len(errors) > 8:
        omitted_errors += len(errors) - 8
    return {
        key: value
        for key, value in {
            "source": feedback.get("source"),
            "iteration": feedback.get("iteration"),
            "description_attempt": feedback.get("description_attempt"),
            "failed_repair_round": feedback.get("failed_repair_round"),
            "template_only": feedback.get("template_only"),
            "generator_errors": compact_errors,
            "omitted_error_count": omitted_errors or None,
            "controller_feedback": short_text(feedback.get("controller_feedback"), 460),
            "strategy_cooldowns": _compact_strategy_cooldowns_for_prompt(feedback.get("strategy_cooldowns")),
            "last_z_proposal_summary": _compact_feedback_summary(feedback.get("last_z_proposal_summary")),
            "last_w_review_summary": _compact_feedback_summary(feedback.get("last_w_review_summary")),
            "executable_generator_rules": _compact_executable_rules_for_generator_feedback(
                feedback.get("executable_generator_rules")
            ),
        }.items()
        if value not in (None, "", [], {})
    }


def generator_feedback_prompt_block(feedback: Mapping[str, Any] | None) -> str:
    compact = compact_generator_feedback_for_prompt(feedback)
    if not compact:
        return ""
    return f"\nCONTROLLER_GENERATOR_OR_W_FEEDBACK:\n```json\n{prompt_json(compact)}\n```\n"


def prompt_x_sequential_material_proposal(
    context: Mapping[str, Any],
    *,
    iteration: int,
    return_feedback: Mapping[str, Any] | None = None,
    template_only: bool = False,
) -> str:
    feedback_block = xy_return_feedback_prompt_block(return_feedback)
    template_block = f"\n{generator_template_only_prompt_block()}" if template_only else ""
    mattergen_block = mattergen_xy_prompt_block(context)
    search_policy_instruction = xy_search_policy_instruction(context)
    queue_size = xy_sun_candidate_queue_size_from_context(context)
    return f"""Agent X sequential single-material proposal.

{XY_SEQUENTIAL_RAG_REQUIREMENT}

{XY_SEQUENTIAL_OPTIMIZATION_POLICY}
{XY_SEQUENTIAL_STRATEGY_BOUNDARY_POLICY}
{XY_SEQUENTIAL_MACHINE_STRATEGY_POLICY}
{template_block}
{mattergen_block}

Propose a ranked queue of up to {queue_size} natural-language material descriptions for iteration {iteration}; select one by strict SUN probability. No probes, structures, MP ids, or MP queries.
Each queue item: candidate_id, rank, reduced_formula, acquisition_mode, expected_e_hull_band, duplicate/template audit, risks, material_description.
Keep JSON iteration={iteration}; selected_candidate_id must match a queue item; top-level material_description must copy that item.
{search_policy_instruction}
Obey latest_xy_strategy_constraints unless search_policy supersedes it. Use the first unblocked next_strategy_candidate_formulas item. Avoid forbidden_evaluator_null_elements unless control_candidate_requested=true.
If no legal non-duplicate remains, return no_valid_material_description with impossibility_certificate auditing each next_strategy_candidate_formulas item.

	CONTEXT_JSON:
	```json
	{prompt_json(compact_xy_material_context_for_prompt(context))}
	```
	{feedback_block}
Return JSON object:
status="material_description_proposal"; agent="X"; iteration={iteration};
sun_candidate_queue=[{{candidate_id,rank,reduced_formula,acquisition_mode,expected_e_hull_band,material_description}}];
selected_candidate_id; selection_rationale; material_description={{natural_language_description,target_family,elements_or_families,expected_local_motif,generator_template,why_sun_likely,why_not_duplicate,known_risks}};
impossibility_certificate; strategy_update_from_history; proposal_summary.
	"""


def xy_return_feedback_prompt_block(return_feedback: Mapping[str, Any] | None) -> str:
    if not return_feedback:
        return ""
    return f"""
Z/W_RETURN_FEEDBACK_FROM_PREVIOUS_DESCRIPTION_ATTEMPT:
```json
{prompt_json(return_feedback)}
```

Treat this feedback as binding controller evidence for the current iteration. If it reports generator materialization failure, W.return_to_xy=true, or a no-faithful-candidate result for a formula/template/role mapping, that exact mapping is blocked and overrides stale next_strategy text. Do not counterpropose, approve, or finalize that blocked mapping. If the latest strategy allows no fallback, approve or emit no_valid_material_description with an impossibility_certificate instead of reviving the failed candidate.
"""


def prompt_y_sequential_material_review(
    context: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    iteration: int,
    cycle: int,
    return_feedback: Mapping[str, Any] | None = None,
    template_only: bool = False,
) -> str:
    feedback_block = xy_return_feedback_prompt_block(return_feedback)
    template_block = f"\n{generator_template_only_prompt_block()}" if template_only else ""
    mattergen_block = mattergen_xy_prompt_block(context)
    search_policy_instruction = xy_search_policy_instruction(context, reviewer=True)
    queue_size = xy_sun_candidate_queue_size_from_context(context)
    return f"""Agent Y sequential single-material review cycle {cycle}.

	{STRICT_COUNTERPROPOSAL_PROTOCOL}

	{XY_SEQUENTIAL_MACHINE_STRATEGY_POLICY}
	{template_block}
	BINDING_XY_STRATEGY_BOUNDARY_POLICY: review search_policy 70/20/10, latest_xy_strategy_constraints, failure_boundaries, failed/used formulas, forbidden evaluator-null elements, and control_candidate_requested; search_policy supersedes stale routes only with audit. Use no_valid_material_description only with a complete impossibility_certificate.
	MatterGen chemical-system acquisition budget: acquisition_mode is the controller current_search_mode label, not a free-form novelty label. In exploit mode, a legal same-mechanism replacement after cooldown-blocked stale formulas is still acquisition_mode="exploit"; no formula-probe template gates.

	Review Agent X's ranked SUN candidate queue. Approve only if selected material is the best legal strict-SUN item, uses A/B/X/Y evidence, avoids duplicates/null elements, and is specific enough for Z/W. agree=true and approved=true is terminal. Keep JSON iteration={iteration}.
	{search_policy_instruction}
	You must reject if selected_candidate_id mismatches material_description, higher-ranked legal items are skipped, latest strategy/failure boundaries are violated, first unblocked next_strategy_candidate_formulas item is skipped, forbidden_evaluator_null_elements appear without control_candidate_requested=true, or template-only faithfulness is a stoichiometry proxy.
	no_valid_material_description requires every next_strategy_candidate_formulas item audited as blocked.

	CONTEXT_JSON:
	```json
		{prompt_json(compact_xy_material_context_for_prompt(context))}
		```
		{feedback_block}

	AGENT_X_PROPOSAL:
```json
{prompt_json(compact_material_description_for_review_prompt(proposal))}
```

Return JSON:
{{
  "status": "material_description_review",
  "agent": "Y",
  "iteration": {iteration},
  "agree": false,
  "approved": false,
  "required_revision": "specific change needed, or empty when approved",
  "risk_audit": {{
    "ranked_queue_present": true,
    "selected_candidate_is_best_legal_sun_bet": true,
    "duplicate_formula_risk": "low|medium|high",
    "generator_feasibility_risk": "low|medium|high"
  }},
  "counterproposal_material_description": {{}},
  "overall_reasoning_summary": "concise summary"
}}
Set agree=true only when the material description should be sent to Z/W or when no_valid_material_description is justified and should terminate the iteration as strategy-blocked.
"""


def prompt_y_sequential_material_counterproposal(
    context: Mapping[str, Any],
    proposal: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    iteration: int,
    cycle: int,
    return_feedback: Mapping[str, Any] | None = None,
    template_only: bool = False,
) -> str:
    feedback_block = xy_return_feedback_prompt_block(return_feedback)
    template_block = f"\n{generator_template_only_prompt_block()}" if template_only else ""
    mattergen_block = mattergen_xy_prompt_block(context)
    queue_size = xy_sun_candidate_queue_size_from_context(context)
    return f"""Agent Y: you rejected Agent X's single-material description in cycle {cycle}. You must now propose your own one-material description that satisfies your critique.

{STRICT_COUNTERPROPOSAL_PROTOCOL}

{XY_SEQUENTIAL_RAG_REQUIREMENT}

{XY_SEQUENTIAL_COMPACT_DEBATE_POLICY}
{XY_SEQUENTIAL_MACHINE_STRATEGY_POLICY}
{template_block}
{mattergen_block}

Do not only object. Produce one Agent X-compatible ranked SUN candidate queue with up to {queue_size} concrete material descriptions plus one selected top-level material_description, or status="no_valid_material_description" with an empty material_description and concise impossibility_certificate.
Your counterproposal must obey CONTEXT_JSON.latest_xy_strategy_constraints unless CONTEXT_JSON.controller_constraints.search_policy explicitly supersedes that stale route; when only a named duplicate list is exhausted, keep the validated basin with at least two new concrete non-duplicate formulas.
Do not revive controller-removed tokens, Z/W-blocked formula/template/role mappings, or forbidden_evaluator_null_elements unless control_candidate_requested=true; do not counterpropose that candidate again when Z/W feedback blocks it.
For parser compatibility, set the JSON field agent to "X" and use Agent X's output shape: status, agent, iteration, sun_candidate_queue, selected_candidate_id, selection_rationale, material_description, strategy_update_from_history, proposal_summary, optional impossibility_certificate.
	In template-only mode, the counterproposal must name exactly one allowed generator_template and include complete generator_role_mapping unless it is explicitly returning status="no_valid_material_description".
In template-only mode, do not counterpropose a template that is merely a stoichiometry vehicle/proxy/container for the intended motif.

CONTEXT_JSON:
```json
	{prompt_json(compact_xy_material_context_for_prompt(context))}
	```
	{feedback_block}

	PREVIOUS_AGENT_X_PROPOSAL:
```json
{prompt_json(compact_material_description_for_review_prompt(proposal))}
```

AGENT_Y_REVIEW:
```json
{prompt_json(compact_review_for_counterproposal_prompt(review))}
```

Return only Agent X JSON.
"""


def prompt_x_sequential_material_reverse_review(
    context: Mapping[str, Any],
    original_proposal: Mapping[str, Any],
    counterproposal: Mapping[str, Any],
    *,
    iteration: int,
    cycle: int,
    return_feedback: Mapping[str, Any] | None = None,
    template_only: bool = False,
) -> str:
    feedback_block = xy_return_feedback_prompt_block(return_feedback)
    template_block = f"\n{generator_template_only_prompt_block()}" if template_only else ""
    mattergen_block = mattergen_xy_prompt_block(context)
    return f"""Agent X: critique Agent Y's single-material counterproposal for cycle {cycle}.

{STRICT_COUNTERPROPOSAL_PROTOCOL}

{XY_SEQUENTIAL_RAG_REQUIREMENT}

{XY_SEQUENTIAL_COMPACT_DEBATE_POLICY}
{XY_SEQUENTIAL_MACHINE_STRATEGY_POLICY}
{template_block}
{mattergen_block}

You are now the reviewer. If Agent Y's queue is the best legal strict-SUN bet and specific enough for Z/W, set agree=true and approved=true; that is terminal. Keep iteration={iteration}.
Reject no_valid_material_description unless the impossibility_certificate audits every next_strategy_candidate_formulas item. Reject counterproposals that contradict active search_policy/latest strategy, enter failure_boundaries, use forbidden_evaluator_null_elements without control_candidate_requested=true, or revive Z/W-blocked mappings.
	Reject it if the template is justified only as a stoichiometry vehicle/proxy/container rather than the topology/coordination motif being tested.
For parser compatibility, set the JSON field agent to "Y" and use Agent Y's review shape: status, agent, iteration, agree, approved, required_revision, risk_audit, counterproposal_material_description, overall_reasoning_summary.

CONTEXT_JSON:
```json
	{prompt_json(compact_xy_material_context_for_prompt(context))}
	```
	{feedback_block}

	ORIGINAL_AGENT_X_PROPOSAL:
```json
{prompt_json(compact_material_description_for_review_prompt(original_proposal))}
```

AGENT_Y_COUNTERPROPOSAL_JSON:
```json
{prompt_json(compact_material_description_for_review_prompt(counterproposal))}
```

Return only Agent Y JSON.
"""


def prompt_x_sequential_material_revision(
    context: Mapping[str, Any],
    proposal: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    iteration: int,
    cycle: int,
    return_feedback: Mapping[str, Any] | None = None,
    template_only: bool = False,
) -> str:
    feedback_block = xy_return_feedback_prompt_block(return_feedback)
    template_block = f"\n{generator_template_only_prompt_block()}" if template_only else ""
    mattergen_block = mattergen_xy_prompt_block(context)
    return f"""Agent X sequential material-description revision cycle {cycle}.

{XY_SEQUENTIAL_RAG_REQUIREMENT}

{XY_SEQUENTIAL_COMPACT_DEBATE_POLICY}
{XY_SEQUENTIAL_MACHINE_STRATEGY_POLICY}
{template_block}
{mattergen_block}

Revise the ranked SUN queue and selected natural-language material_description. Keep iteration={iteration}; do not increment for cycle {cycle}.
Obey active search_policy/latest strategy, choose the first unblocked next_strategy_candidate_formulas item, and do not revive controller-removed tokens, Z/W-blocked mappings, or forbidden_evaluator_null_elements unless control_candidate_requested=true.
If all allowed non-duplicates are impossible, return no_valid_material_description with an impossibility_certificate auditing every next_strategy_candidate_formulas item.
	In template-only mode, the revision must stay inside one allowed generator template and include complete generator_role_mapping unless it is explicitly returning status="no_valid_material_description".
	In template-only mode, a revision that keeps the chemistry by treating a template as a stoichiometry vehicle/proxy/container is invalid; change to a faithful route or no_valid_material_description.

CONTEXT_JSON:
```json
	{prompt_json(compact_xy_material_context_for_prompt(context))}
	```
	{feedback_block}

	PREVIOUS_X_PROPOSAL:
```json
{prompt_json(compact_material_description_for_review_prompt(proposal))}
```

AGENT_Y_REVIEW:
```json
{prompt_json(compact_review_for_prompt(review))}
```

Return the same Agent X proposal JSON shape with an updated sun_candidate_queue, selected_candidate_id, selection_rationale, and material_description.
"""


def prompt_y_sequential_material_final(
    context: Mapping[str, Any],
    proposal: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    iteration: int,
    return_feedback: Mapping[str, Any] | None = None,
    template_only: bool = False,
) -> str:
    feedback_block = xy_return_feedback_prompt_block(return_feedback)
    template_block = f"\n{generator_template_only_prompt_block()}" if template_only else ""
    mattergen_block = mattergen_xy_prompt_block(context)
    return f"""Agent Y final single-material description for iteration {iteration}.

{XY_SEQUENTIAL_RAG_REQUIREMENT}

{XY_SEQUENTIAL_FINAL_POLICY_REMINDER}
{XY_SEQUENTIAL_MACHINE_STRATEGY_POLICY}
{template_block}
{mattergen_block}

	Write final ranked SUN queue plus selected material_description for Z/W, or no_valid_material_consensus. No generator strings.
		The final material must obey latest_xy_strategy_constraints unless search_policy superseded that stale route during the reviewed debate. If Z/W feedback blocks the debated mapping, do not finalize that mapping; if no fallback is legal under the active search_policy and template constraints, return no_valid_material_consensus with impossibility_certificate.
	Template-only finals must preserve generator_template, generator_role_mapping, template_formula_family, expected_local_motif, and why_template_is_faithful. Reject proxy/container template justifications.

CONTEXT_JSON:
```json
	{prompt_json(compact_xy_material_context_for_prompt(context))}
	```
	{feedback_block}

	LATEST_X_PROPOSAL:
```json
{prompt_json(compact_material_description_for_review_prompt(proposal))}
```

LATEST_Y_REVIEW:
```json
{prompt_json(compact_review_for_prompt(review))}
```

Return JSON:
{{
  "status": "material_description_consensus|no_valid_material_consensus",
  "agent": "Y",
  "iteration": {iteration},
  "sun_candidate_queue": ["queue items"],
  "selected_candidate_id": "q1",
  "selection_rationale": "why selected item is best legal strict-SUN bet",
  "material_description": {{}},
  "impossibility_certificate": {{}},
  "ab_principle_ids_used": [],
  "xy_history_lessons_used": [],
  "sun_optimization_rationale": "why selected material is next best SUN bet",
  "debate_summary": "concise summary"
}}
"""


def prompt_z_sequential_candidate(
    context: Mapping[str, Any],
    material_consensus: Mapping[str, Any],
    *,
    iteration: int,
    feedback: Mapping[str, Any] | None = None,
    template_only: bool = False,
) -> str:
    backend = str(context.get("generator_backend") or DEFAULT_GENERATOR_BACKEND)
    feedback_block = generator_feedback_prompt_block(feedback)
    template_block = f"\n{generator_template_only_prompt_block()}" if template_only else ""
    if backend == "mattergen":
        policy_block = sequential_generator_policy_block(backend=backend, template_only=template_only)
        schema_block = sequential_schema_reminder(backend=backend, template_only=template_only)
        feedback_rule_block = ""
    else:
        policy_block = f"{XY_GENERATOR_ONLY_POLICY}\n\n{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}{template_block}"
        schema_block = generator_executable_schema_reminder(template_only=template_only, backend=backend)
        feedback_rule_block = GENERATOR_VALIDATION_FEEDBACK_RULE
    unrepresentable_rule = (
        "Return no candidate_specs only when the material description is unrepresentable by MatterGen chemical-system conditioning, every faithful MatterGen request is barred by binding feedback, or controller feedback marks the selected chemical basin saturated by repeated historical exclusions."
        if backend == "mattergen"
        else (
        "Return no candidate_specs only when the material description is unrepresentable by any allowed formula_probe template. "
        "Do not use structure_dicts in template-only mode."
        if template_only
        else "Return no candidate_specs only when the material description is unrepresentable by either an allowed formula_probe template or a faithful Pymatgen Structure.as_dict(), not merely because the previous JSON shape was wrong."
        )
    )
    repair_rule = (
        "Fix the MatterGen request fields, chemical_system, filters, and audit keys that failed. Do not repeat any rejected request. If excluded_reduced_formula dominates a no_accepted_structures report, one controlled rescue may increase sampling while preserving hidden exclusions; after repeated duplicate-pressure feedback, leave that chemical system or return candidate_specs=[]."
        if backend == "mattergen"
        else (
        "Fix the exact field names, templates, oxidation-state roles, and audit keys that failed. Do not repeat any rejected schema."
        if template_only
        else "Fix the exact field names, templates, audit keys, or Structure.as_dict() shape that failed. Do not repeat any rejected schema."
        )
    )
    return f"""Agent Z sequential generator translation for iteration {iteration}.

{policy_block}

{schema_block}

{feedback_rule_block}

Translate X/Y's selected natural-language material_description into exactly one generator candidate if faithful representation is possible.
Before translating, inspect XY_MATERIAL_DESCRIPTION_CONSENSUS.sun_candidate_queue when present. Confirm that the selected material is generator-feasible and is not obviously dominated by another queue item on executable-template fidelity or known controller constraints. Do not silently switch to another queue item; if the selected material is dominated or infeasible, explain that through feasibility_assessment.description_representation_gap so W can return the issue to X/Y.
If CONTROLLER_GENERATOR_OR_W_FEEDBACK is present, treat every materialization error as binding input for this repair. {repair_rule}
Schema/materialization errors are Z/W's responsibility. {unrepresentable_rule}
In template-only mode, do not propose a candidate whose audit says the template is a stoichiometry vehicle/proxy/container or may not capture the claimed topology; return candidate_specs=[] with can_generate_faithfully=false instead.
In MatterGen mode, do not invent a manual cell. Propose one MatterGen request with properties_to_condition_on.chemical_system and properties_to_condition_on.energy_above_hull=0.0; do not put either field at request top level. Use filters to enforce allowed elements, max_sites, duplicates, and any exact formula requirement.
In MatterGen mode, convert X/Y density language into filters: if the selected description or recent postmortem says high-volume/high volume, volume penalty/risk, volume_per_atom near the ceiling, open-framework, dense, or compact, set filters.max_volume_per_atom <= {DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM:g} unless current controller feedback says that tighter cap underfilled.

CONTEXT_JSON:
```json
{prompt_json(compact_generator_context_for_prompt(context))}
```

XY_MATERIAL_DESCRIPTION_CONSENSUS:
```json
{prompt_json(compact_material_description_for_generator_prompt(material_consensus))}
```
{feedback_block}
Return JSON:
{{
  "status": "candidate_proposal",
  "agent": "Z",
  "iteration": {iteration},
  "candidate_specs": ["exactly one candidate object, or [] when impossible"],
  "feasibility_assessment": {{
    "can_generate_faithfully": true,
    "selected_queue_candidate_id": "q1",
    "queue_screening_summary": "why the selected material survives Z feasibility screening against other queue items",
    "required_template_or_structure": "template/custom_structure",
    "description_representation_gap": []
  }},
  "proposal_summary": "concise summary"
}}
"""


def prompt_w_sequential_candidate_review(
    context: Mapping[str, Any],
    material_consensus: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    iteration: int,
    cycle: int,
    feedback: Mapping[str, Any] | None = None,
    template_only: bool = False,
) -> str:
    backend = str(context.get("generator_backend") or DEFAULT_GENERATOR_BACKEND)
    feedback_block = generator_feedback_prompt_block(feedback)
    template_block = f"\n{generator_template_only_prompt_block()}" if template_only else ""
    if backend == "mattergen":
        policy_block = sequential_generator_policy_block(backend=backend, template_only=template_only)
        schema_block = sequential_schema_reminder(backend=backend, template_only=template_only)
        feedback_rule_block = ""
        review_protocol = "COUNTERPROPOSAL_PROTOCOL: if rejecting, give exact required_revision; return_to_xy if no faithful MatterGen request remains or the selected basin has repeated/high-budget saturated-basin feedback."
        audit_rules = (
            f"Approve only if Z uses exact MatterGen schema, preserves selected queue chemistry/motif, repeats no controller feedback error, "
            f"and applies filters.max_volume_per_atom<={DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM:g} when dense/high-volume risk is present. "
            "Reject manual cells, formula probes, MP fields, top-level chemical_system/energy_above_hull, missing audit keys, missing required elements, or unfaithful queue switches."
        )
        return_shape = (
            '{"status":"candidate_review","agent":"W","iteration":%d,"agree":false,"return_to_xy":false,'
            '"approved_candidate_ids":[],"rejected_candidates":[{"candidate_id":"seq_001","reason":"specific flaw","required_revision":"specific fix"}],'
            '"xy_repair_feedback":{"reason":"","needed_description_changes":[]},"candidate_audit":{"faithful_to_description":false,'
            '"template_consistency_checked":true,"generator_executable":false,"selected_candidate_best_from_queue":false,'
            '"duplicate_formula_risk":"low|medium|high"},"overall_reasoning_summary":"concise"}'
        ) % iteration
    else:
        policy_block = f"{XY_GENERATOR_ONLY_POLICY}\n\n{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}{template_block}"
        schema_block = generator_executable_schema_reminder(template_only=template_only, backend=backend)
        feedback_rule_block = GENERATOR_VALIDATION_FEEDBACK_RULE
        review_protocol = STRICT_COUNTERPROPOSAL_PROTOCOL
        audit_rules = (
            "Audit Agent Z's one candidate against X/Y's selected natural-language material description, the ranked SUN candidate queue, and the latest controller feedback.\n"
            "Set agree=true only when the candidate uses the exact executable generator schema and does not repeat any materialization error in CONTROLLER_GENERATOR_OR_W_FEEDBACK.\n"
            "Reject or return_to_xy if the selected X/Y material is clearly not the best legal candidate in the queue because another queue item is equally specific, more template-faithful, and less blocked. In that case, xy_repair_feedback must name the better queue item and the reason.\n"
            "Reject any candidate whose template_consistency_audit calls the template a stoichiometry vehicle/proxy/container or otherwise admits the generated topology may not capture X/Y's intended local motif. Set return_to_xy=true when no faithful allowed template exists for the X/Y description."
        )
        return_shape = f"""{{
  "status": "candidate_review",
  "agent": "W",
  "iteration": {iteration},
  "agree": false,
  "return_to_xy": false,
  "approved_candidate_ids": [],
  "rejected_candidates": [
    {{"candidate_id": "seq_001", "reason": "specific flaw", "required_revision": "specific fix"}}
  ],
  "xy_repair_feedback": {{
    "reason": "why X/Y must revise the natural-language description, if return_to_xy=true",
    "needed_description_changes": []
  }},
  "candidate_audit": {{
    "faithful_to_description": false,
    "template_consistency_checked": true,
    "generator_executable": false,
    "selected_candidate_best_from_queue": false,
    "duplicate_formula_risk": "low|medium|high"
  }},
  "overall_reasoning_summary": "concise summary"
}}"""
    if backend == "mattergen":
        return_to_xy_rule = "return_to_xy=true only if no faithful MatterGen request remains after binding feedback, including repeated/high-budget controller saturated-basin feedback."
        rejection_rule = (
            f"Reject wrong MatterGen fields, missing properties_to_condition_on chemical_system/e_hull, missing filter/audit keys, missing required elements, "
            f"or missing max_volume_per_atom<={DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM:g} when dense/high-volume risk is present."
        )
    else:
        return_to_xy_rule = (
            "Set return_to_xy=true when the X/Y material description cannot be represented faithfully by any allowed formula_probe template, or when its only faithful allowed formula-probe representation is barred by binding generator feedback."
            if template_only
            else "Set return_to_xy=true when the natural-language description cannot be represented faithfully by any allowed formula_probe template or faithful Structure.as_dict(), or when every faithful generator representation is barred by binding generator feedback."
        )
        rejection_rule = (
            "If Z used a wrong field name, unsupported template, missing audit key, structure_dicts, or incomplete formula-probe roles, reject it with required_revision so Z repairs it inside the Z/W loop."
            if template_only
            else "If Z used a wrong field name, unsupported template, missing audit key, or invalid Structure.as_dict() shape, reject it with required_revision so Z repairs it inside the Z/W loop."
        )
    return f"""Agent W sequential candidate review cycle {cycle}.

{review_protocol}

{policy_block}

{schema_block}

{feedback_rule_block}

{audit_rules}
{rejection_rule}
{return_to_xy_rule}
In template-only mode, structure_dicts and structure_dict aliases are invalid; reject any candidate containing them.
In MatterGen mode, formula_probe_strings, formula_probes, structure_dicts, manual lattice vectors, manual fractional coordinates, material_ids, and MP queries are invalid.

CONTEXT_JSON:
```json
{prompt_json(compact_generator_context_for_prompt(context))}
```

XY_MATERIAL_DESCRIPTION_CONSENSUS:
```json
{prompt_json(compact_material_description_for_generator_prompt(material_consensus))}
```

AGENT_Z_PROPOSAL:
```json
{prompt_json(compact_candidate_payload_for_prompt(proposal))}
```
{feedback_block}

Return JSON:
{return_shape}
"""


def prompt_w_sequential_candidate_counterproposal(
    context: Mapping[str, Any],
    material_consensus: Mapping[str, Any],
    proposal: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    iteration: int,
    cycle: int,
    feedback: Mapping[str, Any] | None = None,
    template_only: bool = False,
) -> str:
    backend = str(context.get("generator_backend") or DEFAULT_GENERATOR_BACKEND)
    feedback_block = generator_feedback_prompt_block(feedback)
    template_block = f"\n{generator_template_only_prompt_block()}" if template_only else ""
    if backend == "mattergen":
        policy_block = sequential_generator_policy_block(backend=backend, template_only=template_only)
        schema_block = sequential_schema_reminder(backend=backend, template_only=template_only)
        feedback_rule_block = ""
    else:
        policy_block = f"{XY_GENERATOR_ONLY_POLICY}\n\n{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}{template_block}"
        schema_block = generator_executable_schema_reminder(template_only=template_only, backend=backend)
        feedback_rule_block = GENERATOR_VALIDATION_FEEDBACK_RULE
    unrepresentable_rule = (
        "Return no candidate_specs when the material description is unrepresentable by MatterGen chemical-system conditioning, every faithful MatterGen request is barred by binding generator feedback, or controller feedback marks the selected chemical basin saturated by repeated historical exclusions. Do not call a soft-target MatterGen repair unfaithful when it preserves the same chemical_system, required elements, motif, and target_reduced_formula preference after a hard-target materialization failure unless saturated-basin feedback is present."
        if backend == "mattergen"
        else (
        "Return no candidate_specs when the material description is unrepresentable by any allowed formula_probe template, or when every faithful allowed formula-probe representation is barred by binding generator feedback; structure_dicts are invalid."
        if template_only
        else "Return no candidate_specs when the material description is unrepresentable by any allowed formula_probe template or faithful Structure.as_dict(), or when every faithful generator representation is barred by binding generator feedback."
        )
    )
    return f"""Agent W: you rejected Agent Z's generator candidate in cycle {cycle}. You must now propose your own executable generator candidate that satisfies your critique.

{STRICT_COUNTERPROPOSAL_PROTOCOL}

{policy_block}

{schema_block}

{feedback_rule_block}

Do not only object. Either produce exactly one Agent Z-compatible candidate_specs object that faithfully represents X/Y's selected material description, or produce candidate_specs=[] with can_generate_faithfully=false and a specific description_representation_gap. {unrepresentable_rule}
If the selected material is dominated by another X/Y queue item, do not switch material silently; return no candidate_specs and explain the queue-screening problem so X/Y can revise.
Do not repair by relabeling an allowed template as a stoichiometry vehicle/proxy/container; that is an unfaithful representation gap, not an executable candidate.
For parser compatibility, set the JSON field agent to "Z" and use the same output shape as Agent Z: status, agent, iteration, candidate_specs, feasibility_assessment, proposal_summary.
If controller feedback reports a failed formula/template/role mapping or failed MatterGen request/filter, your counterproposal must not repeat it.

CONTEXT_JSON:
```json
{prompt_json(compact_generator_context_for_prompt(context))}
```

XY_MATERIAL_DESCRIPTION_CONSENSUS:
```json
{prompt_json(compact_material_description_for_generator_prompt(material_consensus))}
```

PREVIOUS_AGENT_Z_PROPOSAL:
```json
{prompt_json(compact_candidate_payload_for_prompt(proposal))}
```

AGENT_W_REVIEW:
```json
{prompt_json(compact_candidate_review_for_prompt(review))}
```
{feedback_block}
Return only Agent Z JSON.
"""


def prompt_z_sequential_candidate_reverse_review(
    context: Mapping[str, Any],
    material_consensus: Mapping[str, Any],
    original_proposal: Mapping[str, Any],
    counterproposal: Mapping[str, Any],
    *,
    iteration: int,
    cycle: int,
    feedback: Mapping[str, Any] | None = None,
    template_only: bool = False,
) -> str:
    backend = str(context.get("generator_backend") or DEFAULT_GENERATOR_BACKEND)
    feedback_block = generator_feedback_prompt_block(feedback)
    template_block = f"\n{generator_template_only_prompt_block()}" if template_only else ""
    if backend == "mattergen":
        policy_block = sequential_generator_policy_block(backend=backend, template_only=template_only)
        schema_block = sequential_schema_reminder(backend=backend, template_only=template_only)
        feedback_rule_block = ""
    else:
        policy_block = f"{XY_GENERATOR_ONLY_POLICY}\n\n{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}{template_block}"
        schema_block = generator_executable_schema_reminder(template_only=template_only, backend=backend)
        feedback_rule_block = GENERATOR_VALIDATION_FEEDBACK_RULE
    return f"""Agent Z: critique Agent W's generator candidate counterproposal for cycle {cycle}.

{STRICT_COUNTERPROPOSAL_PROTOCOL}

{policy_block}

{schema_block}

{feedback_rule_block}

You are now the reviewer. If Agent W's counterproposal is more faithful to X/Y's selected material description, generator-executable, queue-consistent, and avoids the latest controller feedback, set agree=true. If not, provide exact rejected_candidates and required_revision.
For parser compatibility, set the JSON field agent to "W" and use the same output shape as Agent W: status, agent, iteration, agree, return_to_xy, approved_candidate_ids, rejected_candidates, xy_repair_feedback, candidate_audit, overall_reasoning_summary.
    Set return_to_xy=true if Agent W correctly demonstrates that the X/Y material description is not representable by the allowed generator interface, that every faithful allowed generator representation is barred by binding generator feedback, or that controller saturated-basin feedback requires X/Y to leave the selected MatterGen chemical system.
Reject any counterproposal that treats template/charge/stoichiometry matching as sufficient while admitting a topology proxy. In MatterGen mode, reject any counterproposal that manually specifies a cell instead of one MatterGen request.

CONTEXT_JSON:
```json
{prompt_json(compact_generator_context_for_prompt(context))}
```

XY_MATERIAL_DESCRIPTION_CONSENSUS:
```json
{prompt_json(compact_material_description_for_generator_prompt(material_consensus))}
```

ORIGINAL_AGENT_Z_PROPOSAL:
```json
{prompt_json(compact_candidate_payload_for_prompt(original_proposal))}
```

AGENT_W_COUNTERPROPOSAL_JSON:
```json
{prompt_json(compact_candidate_payload_for_prompt(counterproposal))}
```
{feedback_block}
Return only Agent W JSON.
"""


def prompt_w_sequential_candidate_final(
    context: Mapping[str, Any],
    material_consensus: Mapping[str, Any],
    proposal: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    iteration: int,
    feedback: Mapping[str, Any] | None = None,
    template_only: bool = False,
) -> str:
    backend = str(context.get("generator_backend") or DEFAULT_GENERATOR_BACKEND)
    feedback_block = generator_feedback_prompt_block(feedback)
    template_block = f"\n{generator_template_only_prompt_block()}" if template_only else ""
    if backend == "mattergen":
        policy_block = sequential_generator_policy_block(backend=backend, template_only=template_only)
        schema_block = sequential_schema_reminder(backend=backend, template_only=template_only)
        feedback_rule_block = ""
    else:
        policy_block = f"{XY_GENERATOR_ONLY_POLICY}\n\n{XY_MECHANISM_TEMPLATE_CONSISTENCY_POLICY}{template_block}"
        schema_block = generator_executable_schema_reminder(template_only=template_only, backend=backend)
        feedback_rule_block = GENERATOR_VALIDATION_FEEDBACK_RULE
    final_rule = (
        "The final agreed_candidate_specs must contain exactly one candidate object with exactly one mattergen_requests entry."
        if backend == "mattergen"
        else (
        "The final agreed_candidate_specs must use formula_probe_strings or formula_probes only; structure_dicts are invalid."
        if template_only
        else "The final agreed_candidate_specs must contain exactly one candidate object using the exact executable generator schema."
        )
    )
    return f"""Agent W final sequential candidate lock for iteration {iteration}.

{policy_block}

{schema_block}

{feedback_rule_block}

Write the final single candidate for the selected X/Y queue item. Do not introduce any candidate Z/W have not reviewed.
{final_rule} Do not reintroduce singular alias fields, unsupported templates, missing template_consistency_audit keys, constructor-style structure shorthand, or invalid MatterGen request fields rejected by the controller.
Do not finalize a candidate whose template_consistency_audit contains stoichiometry vehicle/proxy/container wording or admits that the template may not capture the intended topology.
Do not finalize a candidate if W's own queue screening found that another queue item should have been selected; return_to_xy must have been used earlier instead.

CONTEXT_JSON:
```json
{prompt_json(compact_generator_context_for_prompt(context))}
```

XY_MATERIAL_DESCRIPTION_CONSENSUS:
```json
{prompt_json(compact_material_description_for_generator_prompt(material_consensus))}
```

LATEST_Z_PROPOSAL:
```json
{prompt_json(compact_candidate_payload_for_prompt(proposal))}
```

LATEST_W_REVIEW:
```json
{prompt_json(compact_candidate_review_for_prompt(review))}
```
{feedback_block}

Return JSON:
{{
  "status": "candidate_consensus",
  "agent": "W",
  "iteration": {iteration},
  "agreed_candidate_specs": ["exactly one candidate object"],
  "feasibility_summary": "concise summary"
}}
"""


def prompt_x_sequential_postmortem(
    context: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    iteration: int,
) -> str:
    return f"""Agent X postmortem for sequential SUN optimization iteration {iteration}.

{XY_SEQUENTIAL_RAG_REQUIREMENT}

Analyze the just-evaluated material or MatterGen batch, or the strategy-blocked no-valid-material certificate if no material was evaluated. When ITERATION_RECORD.evaluation_results is present, base the outcome on the batch distribution: sun_count, near_stable counts, best_rows, and failure/missing rows. Explain whether the result supports or weakens the chosen A/B principle interpretation and how X/Y should change the next material strategy. Use prior X/Y history as context; do not just restate one e_hull.
Use outcome_class thresholds strictly: "sun" means strict SUN/e_hull < 0; "near_miss" means non-SUN but e_hull < 0.03; "weak_near_miss" means 0.03 <= e_hull < 0.10; "high_e_hull" means e_hull >= 0.10. Do not call a weak_near_miss a near_miss or use it as strong evidence for continuing the same local repair branch.
Frame the next_strategy as a ranked acquisition queue for strict SUN discovery. Name at least two pre-audited non-duplicate formulas when a route remains viable, and explain why the first queue item is most likely to cross e_hull < 0 rather than merely remain near-stable.
Every next_strategy formula token must be a complete reduced formula with all intended elements, not an abstract label such as Ranked, HgBr4, queue, family, or route. Put abstract labels only in prose, never in candidate formula slots.
If the evaluated material is a near_miss with volume_per_atom >= {DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM:g} or the causal interpretation names high-volume/volume penalty/open-framework packing, include an explicit density filter instruction such as max_volume_per_atom <= {DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM:g} for the next same-route MatterGen request.
If status is strategy_blocked, do not repeat the exhausted branch in next_strategy. Choose a new explicit route outside the impossibility_certificate and failure_boundaries.
Before naming any formula in next_strategy or a fallback order, audit CONTEXT_JSON.controller_constraints.failed_or_used_reduced_formulas. This visible list is compact; the controller also enforces a hidden full failed/used set. A listed formula is illegal as a future candidate unless CONTEXT_JSON.controller_constraints.control_candidate_requested is explicitly true. Do not name a listed formula even as a conditional fallback. Used/failed formulas may appear only as "do not repeat" or "branch exhausted" boundaries.
Also audit CONTEXT_JSON.controller_constraints.failed_volume_template_boundaries before naming a formula/template/role route. This boundary list may also be compact; the controller enforces the full set. Do not emit a single named future formula unless it is explicitly pre-audited as absent from visible failed_or_used_reduced_formulas and outside visible failed_volume_template_boundaries. Prefer a duplicate-gated ordered set of two or more viable candidates from the same route; if no unblocked candidate remains, mark the whole route exhausted instead of naming a blocked single fallback.
Also audit CONTEXT_JSON.controller_constraints.forbidden_evaluator_null_elements; use them only as failure boundaries, not future candidates.
Controller enforcement: if next_strategy names a formula from failed_or_used_reduced_formulas as a future candidate, the controller will remove that formula before writing memory. Avoid losing route details by proposing only non-duplicate formulas in next_strategy.
Write next_strategy and failure_boundaries as operational instructions for the next X/Y proposal/review. Name concrete formulas, families, templates, roles, and variation axes to avoid or pursue; vague lessons are not enough because Y must enforce these fields as hard review constraints.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

ITERATION_RECORD:
```json
{prompt_json(record)}
```

Return JSON:
{{
  "status": "postmortem_proposal",
  "agent": "X",
  "iteration": {iteration},
  "outcome_class": "sun|near_miss|weak_near_miss|high_e_hull|materialization_failure|strategy_blocked|not_evaluated",
  "causal_interpretation": "why this happened in materials terms",
  "strategy_update": "what to exploit, avoid, or vary next",
  "next_strategy": "concise instruction for the next X/Y proposal",
  "principle_updates": [],
  "failure_boundaries": []
}}
"""


def prompt_y_sequential_postmortem_review(
    context: Mapping[str, Any],
    record: Mapping[str, Any],
    postmortem: Mapping[str, Any],
    *,
    iteration: int,
) -> str:
    return f"""Agent Y final postmortem review for sequential SUN optimization iteration {iteration}.

{XY_SEQUENTIAL_RAG_REQUIREMENT}

Audit Agent X's postmortem. Keep the strategy update factual and useful for the next X/Y proposal; in batch MatterGen mode, verify that X used ITERATION_RECORD.evaluation_results instead of judging only the first selected_record.
The next_strategy must be operational as a ranked strict-SUN acquisition queue, not only a narrative lesson. Prefer at least two concrete non-duplicate formulas from the route, with the first item being the best legal next material and later items serving as audited alternatives for X/Y queue construction.
Every next_strategy formula token must be a complete reduced formula with all intended elements, not an abstract label such as Ranked, HgBr4, queue, family, or route. Remove abstract tokens rather than preserving them as fallbacks.
If the evaluated material is a near_miss with volume_per_atom >= {DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM:g} or X names high-volume/volume penalty/open-framework packing, require the next same-route MatterGen strategy to state max_volume_per_atom <= {DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM:g} unless it declares the route exhausted.
Use outcome_class thresholds strictly: "sun" means strict SUN/e_hull < 0; "near_miss" means non-SUN but e_hull < 0.03; "weak_near_miss" means 0.03 <= e_hull < 0.10; "high_e_hull" means e_hull >= 0.10. If X labels a weak_near_miss as near_miss, correct the label and weaken any same-branch continuation.
If the iteration was strategy_blocked, verify that next_strategy leaves the exhausted branch instead of retrying a route certified impossible.
Before approving next_strategy or any fallback formula, audit CONTEXT_JSON.controller_constraints.failed_or_used_reduced_formulas. This visible list is compact; the controller also enforces a hidden full failed/used set. If X names a visible used/failed formula as a future candidate while control_candidate_requested is not true, replace it with a non-duplicate route or mark the branch exhausted; do not preserve the formula as a conditional fallback. Repeated formulas are allowed only inside failure_boundaries/do-not-repeat/branch-exhausted text.
Also audit CONTEXT_JSON.controller_constraints.failed_volume_template_boundaries before approving a future template/role route. This visible list may be compact; the controller enforces the full set. Reject postmortems that spend the next turn on a single unaudited formula; next_strategy should contain a pre-audited, duplicate-gated ordered set of viable candidates, or should declare the entire route exhausted when every visible candidate is blocked.
Also audit CONTEXT_JSON.controller_constraints.forbidden_evaluator_null_elements; remove such future candidates or mark the route exhausted.
Controller enforcement: if your final next_strategy still names formulas from failed_or_used_reduced_formulas as future candidates, the controller will remove them before writing memory. Fix the route yourself so the next turn receives useful, non-duplicate instructions.
Make next_strategy and failure_boundaries operational enough for the next X/Y debate to enforce. Preserve concrete avoid boundaries from the evaluated result and state the exact fallback order when the preferred chemistry may be template-blocked.

CONTEXT_JSON:
```json
{prompt_json(context)}
```

ITERATION_RECORD:
```json
{prompt_json(record)}
```

AGENT_X_POSTMORTEM:
```json
{prompt_json(postmortem)}
```

Return JSON:
{{
  "status": "postmortem_consensus",
  "agent": "Y",
  "iteration": {iteration},
  "outcome_class": "sun|near_miss|weak_near_miss|high_e_hull|materialization_failure|strategy_blocked|not_evaluated",
  "causal_interpretation": "audited materials interpretation",
  "strategy_update": "audited update",
  "next_strategy": "what X/Y should do next",
  "principle_updates": [],
  "failure_boundaries": [],
  "review_summary": "concise summary"
}}
"""


def candidate_specs_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("agreed_candidate_specs", "candidate_specs", "counterproposal_candidate_specs"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def candidate_payload_declares_no_faithful_generator_candidate(payload: Mapping[str, Any]) -> bool:
    """Return true when a Z-shaped payload explicitly says no faithful executable candidate remains."""

    if candidate_specs_from_payload(payload):
        return False
    feasibility = payload.get("feasibility_assessment")
    if not isinstance(feasibility, Mapping):
        return False
    return _bool_value(feasibility.get("can_generate_faithfully")) is False


def design_rules_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in (
        "design_experience_book",
        "agreed_design_experience_book",
        "design_rules",
        "candidate_feature_rules",
        "counterproposal_design_rules",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def design_book_agrees(review: Mapping[str, Any], proposal: Mapping[str, Any]) -> bool:
    if bool(review.get("agree")) is not True:
        return False
    rules = design_rules_from_payload(proposal)
    if not rules:
        return False
    approved = review.get("approved_design_rule_ids")
    if isinstance(approved, list) and approved:
        approved_set = {str(item).strip() for item in approved if str(item).strip()}
        rule_ids = {str(item.get("design_rule_id") or "").strip() for item in rules if isinstance(item, Mapping)}
        return bool(approved_set & rule_ids) or len(approved_set) >= len(rules)
    return True


def compact_payload_for_dialogue(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"payload_preview": short_text(payload, 240)}
    specs = candidate_specs_from_payload(payload)
    design_rules = design_rules_from_payload(payload)
    rejected = payload.get("rejected_candidates")
    rejected_rules = payload.get("rejected_design_rules")
    approved = payload.get("approved_candidate_ids")
    approved_rules = payload.get("approved_design_rule_ids")
    audit = payload.get("experience_audit")
    if not isinstance(audit, Mapping):
        audit = payload.get("design_book_audit")
    return {
        key: value
        for key, value in {
            "status": payload.get("status"),
            "agent": payload.get("agent"),
            "shard_id": payload.get("shard_id"),
            "agree": payload.get("agree"),
            "locked_count": payload.get("locked_count"),
            "target_count": payload.get("target_count"),
            "error_count": payload.get("error_count"),
            "error_categories": payload.get("error_categories"),
            "candidate_count": len(specs),
            "candidate_ids": [str(item.get("id") or "") for item in specs[:12] if isinstance(item, Mapping)],
            "design_rule_count": len(design_rules),
            "design_rule_ids": [str(item.get("design_rule_id") or "") for item in design_rules[:12] if isinstance(item, Mapping)],
            "approved_candidate_count": len(approved) if isinstance(approved, list) else None,
            "rejected_candidate_count": len(rejected) if isinstance(rejected, list) else None,
            "approved_design_rule_count": len(approved_rules) if isinstance(approved_rules, list) else None,
            "rejected_design_rule_count": len(rejected_rules) if isinstance(rejected_rules, list) else None,
            "used_full_experience_book": audit.get("used_full_experience_book") if isinstance(audit, Mapping) else None,
            "no_high_sun_prefilter_violation": audit.get("no_high_sun_prefilter_violation") if isinstance(audit, Mapping) else None,
            "summary": short_text(
                payload.get("proposal_summary")
                or payload.get("overall_reasoning_summary")
                or payload.get("debate_summary")
                or payload.get("reason"),
                500,
            ),
        }.items()
        if value not in (None, "", [])
    }


def compact_dialogue_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    compact = {
        key: artifact.get(key)
        for key in ("role", "cycle", "mode", "execution_feedback_round")
        if artifact.get(key) not in (None, "", [])
    }
    compact["payload_summary"] = compact_payload_for_dialogue(artifact.get("payload"))
    return compact


def compact_dialogue(artifacts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [compact_dialogue_artifact(item) for item in artifacts if isinstance(item, Mapping)]


def _mattergen_error_int_field(text: str, field: str) -> int | None:
    match = re.search(rf"\b{re.escape(field)}=(\d+)\b", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _mattergen_error_status_field(text: str) -> str:
    match = re.search(r"\breport_status=([^;,\s]+)", text)
    return match.group(1).strip() if match else ""


def _mattergen_error_reject_reasons(text: str) -> dict[str, int]:
    match = re.search(r"\breject_reasons=(\{[^\n]*\})", text)
    if not match:
        return {}
    raw = match.group(1).strip()
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
    if not isinstance(parsed, Mapping):
        return {}
    reasons: dict[str, int] = {}
    for key, value in parsed.items():
        reason = str(key or "").strip()
        if not reason:
            continue
        try:
            reasons[reason] = int(value)
        except Exception:
            continue
    return reasons


def mattergen_saturation_analysis_from_errors(
    errors: Sequence[str],
    *,
    request_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    systems = set(_extract_mattergen_request_chemical_systems_from_payload(request_payload or {}, max_items=8))
    duplicate_pressure_events: list[dict[str, Any]] = []
    for raw_error in errors:
        text = str(raw_error or "")
        if "MatterGen materialized" not in text or "reject_reasons=" not in text:
            continue
        systems.update(_chemical_systems_from_error_text(text))
        accepted_count = _mattergen_error_int_field(text, "accepted_count")
        status = _mattergen_error_status_field(text)
        reject_reasons = _mattergen_error_reject_reasons(text)
        rejected_total = sum(max(0, int(count)) for count in reject_reasons.values())
        excluded_count = max(0, int(reject_reasons.get("excluded_reduced_formula") or 0))
        excluded_ratio = (excluded_count / rejected_total) if rejected_total else 0.0
        saturated = (
            accepted_count == 0
            and status == "no_accepted_structures"
            and rejected_total >= XY_MATTERGEN_SATURATION_MIN_REJECTIONS
            and excluded_ratio >= XY_MATTERGEN_SATURATION_EXCLUDED_RATIO
        )
        if not saturated:
            continue
        duplicate_pressure_events.append(
            {
                "accepted_count": accepted_count,
                "rejected_total": rejected_total,
                "excluded_reduced_formula": excluded_count,
                "excluded_ratio": round(excluded_ratio, 4),
                "reject_reasons": reject_reasons,
            }
        )
    if not duplicate_pressure_events:
        return {}
    strongest_event = max(
        duplicate_pressure_events,
        key=lambda item: (float(item.get("excluded_ratio") or 0.0), int(item.get("rejected_total") or 0)),
    )
    event_count = len(duplicate_pressure_events)
    max_rejected_total = max(int(item.get("rejected_total") or 0) for item in duplicate_pressure_events)
    requires_xy_revision = event_count >= 2 or max_rejected_total >= (XY_MATTERGEN_SATURATION_MIN_REJECTIONS * 4)
    status = "saturated_mattergen_basin" if requires_xy_revision else "mattergen_duplicate_pressure"
    return {
        "status": status,
        "requires_xy_strategy_revision": requires_xy_revision,
        "controlled_rescue_allowed": not requires_xy_revision,
        "duplicate_pressure_event_count": event_count,
        "events": duplicate_pressure_events[-3:],
        "accepted_count": strongest_event["accepted_count"],
        "rejected_total": strongest_event["rejected_total"],
        "excluded_reduced_formula": strongest_event["excluded_reduced_formula"],
        "excluded_ratio": strongest_event["excluded_ratio"],
        "reject_reasons": strongest_event["reject_reasons"],
        "blocked_chemical_systems": sorted(system for system in systems if system),
        "required_next_move": (
            (
                "Return to X/Y and select a different chemical system or mechanism route. "
                "Do not repair by only increasing batch_size/num_batches, lowering guidance, or removing visible excludes; "
                "the controller will reapply historical duplicate exclusions."
            )
            if requires_xy_revision
            else (
                "Allow one controlled rescue in Z/W: keep exact required/allowed elements and hidden exclusions, "
                "increase sampling budget modestly, and do not claim that removing visible exclude_reduced_formulas changes "
                "the controller-enforced hidden history. If this rescue also returns no accepted structures, return to X/Y."
            )
        ),
    }


def mattergen_feedback_requires_xy_strategy_revision(feedback: Mapping[str, Any] | None) -> bool:
    if not isinstance(feedback, Mapping):
        return False
    saturation = feedback.get("mattergen_saturation")
    return isinstance(saturation, Mapping) and bool(saturation.get("requires_xy_strategy_revision"))


def sequential_generator_repair_feedback(
    *,
    iteration: int,
    description_attempt: int,
    repair_round: int,
    errors: Sequence[str],
    z_proposal: Mapping[str, Any],
    w_review: Mapping[str, Any],
    candidate_consensus: Mapping[str, Any] | None = None,
    template_only: bool = False,
    backend: str = DEFAULT_GENERATOR_BACKEND,
    strategy_cooldowns: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    mattergen_saturation = (
        mattergen_saturation_analysis_from_errors(
            errors,
            request_payload=candidate_consensus or z_proposal,
        )
        if backend == "mattergen"
        else {}
    )
    controller_instruction = (
        "Z and W must repair these generator/materialization errors in the next candidate attempt. "
        "Do not return to X/Y for schema aliases, unsupported formula-probe strings, missing audit keys, "
        "or invalid generator formatting; return to X/Y only for a genuinely unrepresentable material description. "
        "When template_only is true, structure_dicts are forbidden and Z/W must repair using formula_probe_strings or formula_probes. "
        "When strategy_cooldowns are present, MatterGen chemical_system and family-pattern cooldowns are hard preflight constraints."
    )
    if mattergen_saturation and mattergen_saturation.get("requires_xy_strategy_revision"):
        controller_instruction = (
            "Controller classified this MatterGen failure as saturated_mattergen_basin: accepted_count=0 and "
            "excluded_reduced_formula dominates the rejection reasons after historical duplicate exclusions were applied. "
            "Z/W must not repair by only increasing sampling budget, changing guidance, or dropping visible excludes. "
            "Return the selected material route to X/Y unless the next faithful request leaves the blocked chemical system."
        )
    elif mattergen_saturation:
        controller_instruction = (
            "Controller classified this MatterGen failure as mattergen_duplicate_pressure: accepted_count=0 and "
            "excluded_reduced_formula dominates the rejection reasons after historical duplicate exclusions were applied. "
            "Z/W may attempt exactly one controlled rescue by modestly increasing sampling budget while preserving the "
            "same exact required/allowed elements and controller hidden exclusions. Do not claim that dropping visible "
            "exclude_reduced_formulas changes the request; the controller will reapply hidden history. If the rescue also "
            "returns no accepted structures, return the selected material route to X/Y."
        )
    return {
        "source": "controller_generator_materialization",
        "iteration": iteration,
        "description_attempt": description_attempt,
        "failed_repair_round": repair_round,
        "template_only": template_only,
        "generator_errors": [short_text(message, 600) for message in errors[:20]],
        "omitted_error_count": max(0, len(errors) - 20),
        "last_z_proposal_summary": compact_payload_for_dialogue(z_proposal),
        "last_w_review_summary": compact_payload_for_dialogue(w_review),
        "last_candidate_consensus_summary": compact_payload_for_dialogue(candidate_consensus or {}),
        "strategy_cooldowns": strategy_cooldowns or {},
        "mattergen_saturation": mattergen_saturation,
        "executable_generator_rules": generator_executable_schema_rules(template_only=template_only, backend=backend),
        "controller_instruction": controller_instruction,
    }


def executable_generator_rule_from_locked_spec(
    locked_spec: Mapping[str, Any],
    selected_record: Mapping[str, Any],
    *,
    repair_round: int,
    preceding_errors: Sequence[str],
    template_only: bool = False,
    backend: str = DEFAULT_GENERATOR_BACKEND,
) -> dict[str, Any]:
    formula_probes = locked_spec.get("formula_probes")
    structure_dicts = locked_spec.get("structure_dicts")
    mattergen_requests = locked_spec.get("mattergen_requests")
    input_type = "unknown"
    input_field = ""
    template = ""
    if isinstance(mattergen_requests, list) and mattergen_requests and isinstance(mattergen_requests[0], Mapping):
        input_type = "mattergen_request"
        input_field = "mattergen_requests"
        template = "mattergen"
    elif isinstance(formula_probes, list) and formula_probes and isinstance(formula_probes[0], Mapping):
        input_type = "formula_probe"
        input_field = "formula_probes"
        template = str(formula_probes[0].get("template") or "")
    elif isinstance(structure_dicts, list) and structure_dicts:
        input_type = "structure_dict"
        input_field = "structure_dicts"
        audit = locked_spec.get("template_consistency_audit")
        if isinstance(audit, Mapping):
            template = str(audit.get("chosen_template") or "")
    raw_selected_spec = selected_record.get("xy_candidate_spec")
    selected_candidate_id = raw_selected_spec.get("id") if isinstance(raw_selected_spec, Mapping) else None
    return {
        "status": "generator_materialized",
        "created_at_utc": utc_now(),
        "candidate_id": locked_spec.get("id") or selected_candidate_id,
        "formula": selected_record.get("formula"),
        "material_id": selected_record.get("material_id"),
        "input_type": input_type,
        "input_field": input_field,
        "generator_template": template,
        "repair_round": repair_round,
        "accepted_schema_rules": generator_executable_schema_rules(template_only=template_only, backend=backend),
        "accepted_template_consistency_audit": locked_spec.get("template_consistency_audit", {}),
        "accepted_mattergen_request": mattergen_requests[0] if isinstance(mattergen_requests, list) and mattergen_requests else None,
        "normalization_notes": locked_spec.get("normalization_notes", []),
        "preceding_error_count": len(preceding_errors),
        "preceding_errors_tail": [short_text(message, 420) for message in preceding_errors[-12:]],
        "reusable_rule": (
            "Use source='generator', count=1, exactly one plural generator input field, and a complete "
            "template_consistency_audit whose chosen_template matches the generator input. "
            "In template-only mode, use formula_probe_strings/formula_probes only. "
            "For MatterGen backend, use one mattergen_requests entry with properties_to_condition_on.chemical_system "
            "and properties_to_condition_on.energy_above_hull=0.0; do not put either field at request top level. "
            "Otherwise, custom structures must be real Pymatgen Structure.as_dict() objects inside structure_dicts."
        ),
    }


def y_agrees(review: Mapping[str, Any], proposal: Mapping[str, Any], target_count: int) -> bool:
    if bool(review.get("agree")) is not True:
        return False
    approved = review.get("approved_candidate_ids")
    if isinstance(approved, list) and len(approved) >= target_count:
        return True
    return len(candidate_specs_from_payload(proposal)) >= target_count


def review_requires_counterproposal(
    review: Mapping[str, Any],
    *,
    rejection_streak: int,
    threshold: int = DEFAULT_CRITIC_COUNTERPROPOSAL_AFTER,
) -> bool:
    """Return true when a reviewer must switch from objection to counterproposal."""

    if threshold <= 0:
        return False
    for key in ("agree", "approved", "concede"):
        value = _bool_value(review.get(key))
        if value is True:
            return False
    status = str(review.get("status") or review.get("overall_verdict") or review.get("decision") or "").strip().lower()
    if status in {"accept", "accepted", "approved"}:
        return False
    if status == "consensus":
        approved_candidates = review.get("approved_candidate_ids")
        approved_rules = review.get("approved_design_rule_ids")
        if not (approved_candidates == [] or approved_rules == []):
            return False
    return rejection_streak + 1 >= threshold


def freeze_materialized_candidate_specs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Freeze dry-run selections so final evaluation uses the audited records."""
    frozen: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        raw_spec = record.get("xy_candidate_spec")
        spec = dict(raw_spec) if isinstance(raw_spec, Mapping) else {}
        spec["count"] = 1
        spec.setdefault("id", f"xy_locked_{index:03d}")
        source = str(record.get("crystal_llm_source") or spec.get("source") or "mp_pool")
        if source == "generator" and isinstance(record.get("structure_dict"), Mapping):
            spec["source"] = "generator"
            spec["material_ids"] = []
            spec["formula_probes"] = []
            spec["mattergen_requests"] = []
            spec["structure_dicts"] = [dict(record["structure_dict"])]
        else:
            material_id = str(record.get("material_id") or "")
            spec["source"] = "mp_pool"
            spec["material_ids"] = [material_id] if material_id else []
            spec["structure_dicts"] = []
            spec["mattergen_requests"] = []
        frozen.append(spec)
    return frozen


def _candidate_error_categories(errors: Sequence[str]) -> dict[str, int]:
    categories = {
        "unsupported_query_key": 0,
        "zero_pool_match": 0,
        "zero_materialized": 0,
        "generator_failed": 0,
        "duplicate_reduced_formula": 0,
        "other": 0,
    }
    for message in errors:
        if "unsupported query key:" in message:
            categories["unsupported_query_key"] += 1
        elif "query matched zero pool records" in message:
            categories["zero_pool_match"] += 1
        elif "materialized 0 records" in message or "requested" in message and "materialized" in message:
            categories["zero_materialized"] += 1
        elif "generator materialized" in message or "structure_dict" in message:
            categories["generator_failed"] += 1
        elif "duplicate reduced_formula" in message:
            categories["duplicate_reduced_formula"] += 1
        else:
            categories["other"] += 1
    return {key: value for key, value in categories.items() if value}


def materialization_dry_run_feedback(
    *,
    pool_records: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    target_count: int,
    seed: int,
    max_sites: int,
    known_formulas: set[str] | None,
    allowed_sources: set[str] | None = None,
    max_per_reduced_formula: int = 1,
    require_design_rule_ids: bool = False,
    allow_structure_dicts: bool = True,
    mattergen_config: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected, locked_specs, errors = materialize_candidate_specs(
        pool_records,
        specs,
        target_count=target_count,
        seed=seed,
        max_sites=max_sites,
        known_formulas=known_formulas,
        allowed_sources=allowed_sources,
        max_per_reduced_formula=max_per_reduced_formula,
        require_design_rule_ids=require_design_rule_ids,
        allow_structure_dicts=allow_structure_dicts,
        mattergen_config=mattergen_config,
    )
    frozen_specs = freeze_materialized_candidate_specs(selected[:target_count])
    feedback = {
        "status": "passed" if len(selected) >= target_count else "failed",
        "target_count": target_count,
        "locked_count": min(len(selected), target_count),
        "raw_candidate_count": len(specs),
        "locked_candidate_ids": [str(spec.get("id") or "") for spec in frozen_specs],
        "locked_materials": [
            {
                "candidate_id": (record.get("xy_candidate_spec") or {}).get("id")
                if isinstance(record.get("xy_candidate_spec"), Mapping)
                else None,
                "material_id": record.get("material_id"),
                "formula": record.get("formula"),
                "source": record.get("crystal_llm_source", "mp_pool"),
            }
            for record in selected[:target_count]
        ],
        "locked_candidate_specs_to_preserve": frozen_specs,
        "error_count": len(errors),
        "error_categories": _candidate_error_categories(errors),
        "errors": [short_text(message, 420) for message in errors[:30]],
        "omitted_error_count": max(0, len(errors) - 30),
        "deduplication": {
            "max_per_reduced_formula": max_per_reduced_formula,
            "hard_reduced_formula_dedup": max_per_reduced_formula == 1,
        },
        "two_stage_requirements": {
            "require_design_rule_ids": require_design_rule_ids,
        },
        "schema_reminders": {
            "supported_query_keys": [
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
                "density_min",
                "density_max",
                "volume_per_atom_min",
                "volume_per_atom_max",
                "crystal_system_in",
                "spacegroup_number_in",
                "spacegroup_number_min",
                "spacegroup_number_max",
                "preferred_order",
            ],
            "common_alias_repairs": {
                "num_sites_min/max": "nsites_min/nsites_max",
                "formula or formula_exact": "formula_in",
                "fields/random_seed/deduplicate_*": "remove from query",
                "formula exclusions": "top-level exclude_formulas",
            },
            "generator_candidate_schema": generator_executable_schema_rules(
                template_only=not allow_structure_dicts,
                backend="mattergen" if mattergen_config else DEFAULT_GENERATOR_BACKEND,
            ),
        },
    }
    return selected[:target_count], frozen_specs, feedback


def locked_candidate_specs_from_feedback(feedback: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(feedback, Mapping):
        return []
    specs = feedback.get("locked_candidate_specs_to_preserve")
    if not isinstance(specs, list):
        return []
    return [dict(item) for item in specs if isinstance(item, Mapping)]


def review_rejection_messages(review: Mapping[str, Any]) -> list[str]:
    rejected = review.get("rejected_candidates")
    if not isinstance(rejected, list):
        return []
    messages: list[str] = []
    for item in rejected[:12]:
        if isinstance(item, Mapping):
            candidate_id = str(item.get("candidate_id") or "").strip()
            reason = str(item.get("reason") or item.get("required_revision") or "").strip()
            messages.append(short_text(f"{candidate_id}: {reason}" if candidate_id else reason, 360))
        else:
            messages.append(short_text(item, 360))
    return [message for message in messages if message]


def require_design_rule_ids_for_args(args: argparse.Namespace) -> bool:
    return str(getattr(args, "generation_protocol", "direct") or "direct") == "two_stage"


def run_two_stage_shard_debate(
    *,
    args: argparse.Namespace,
    root: Path,
    work_dir: Path,
    state_path: Path,
    candidate_pool_path: Path,
    context: Mapping[str, Any],
    pool_records: Sequence[Mapping[str, Any]],
    known_formulas: set[str] | None,
    shard_index: int,
    shard_target: int,
) -> dict[str, Any]:
    shard_dir = work_dir / "debates" / f"shard_{shard_index:03d}"
    x_client = make_client(
        role="X",
        args=args,
        root=root,
        log_dir=shard_dir / "x_llm_calls",
        state_path=state_path,
        candidate_pool_path=candidate_pool_path,
    )
    y_client = make_client(
        role="Y",
        args=args,
        root=root,
        log_dir=shard_dir / "y_llm_calls",
        state_path=state_path,
        candidate_pool_path=candidate_pool_path,
    )
    z_client = make_client(
        role="Z",
        args=args,
        root=root,
        log_dir=shard_dir / "z_llm_calls",
        state_path=state_path,
        candidate_pool_path=candidate_pool_path,
    )
    w_client = make_client(
        role="W",
        args=args,
        root=root,
        log_dir=shard_dir / "w_llm_calls",
        state_path=state_path,
        candidate_pool_path=candidate_pool_path,
    )

    artifacts: list[dict[str, Any]] = []
    design_proposal = call_json_object(
        x_client,
        system=AGENT_X_SYSTEM,
        user=prompt_x_design_book_proposal(context),
        role="agent_x_design",
        metadata={"role": "agent_x", "stage": "xy_design_proposal", "shard": shard_index, "cycle": 1},
        json_repair_attempts=args.json_repair_attempts,
    )
    artifacts.append({"role": "X", "cycle": 1, "mode": "design_proposal", "payload": design_proposal})
    design_review: dict[str, Any] = {}
    design_consensus: dict[str, Any] | None = None
    max_rounds = max(1, int(args.max_debate_rounds))
    min_rounds = max(1, min(int(args.min_debate_rounds), max_rounds))
    design_rejection_streak = 0
    for cycle in range(1, max_rounds + 1):
        design_review = call_json_object(
            y_client,
            system=AGENT_Y_SYSTEM,
            user=prompt_y_design_book_review(context, design_proposal, cycle),
            role="agent_y_design",
            metadata={"role": "agent_y", "stage": "xy_design_review", "shard": shard_index, "cycle": cycle},
            json_repair_attempts=args.json_repair_attempts,
        )
        artifacts.append({"role": "Y", "cycle": cycle, "mode": "design_review", "payload": design_review})
        if cycle >= min_rounds and design_book_agrees(design_review, design_proposal):
            design_consensus = call_json_object(
                y_client,
                system=AGENT_Y_SYSTEM,
                user=prompt_y_design_book_final(context, design_proposal, design_review),
                role="xy_design_consensus",
                metadata={"role": "xy_design_consensus", "stage": "xy_design_final", "shard": shard_index, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "Y", "cycle": cycle, "mode": "design_final", "payload": dict(design_consensus)})
            if design_rules_from_payload(design_consensus):
                write_json(shard_dir / "xy_design_experience_book.json", design_consensus)
                break
            design_consensus = None
        if review_requires_counterproposal(
            design_review,
            rejection_streak=design_rejection_streak,
            threshold=args.critic_counterproposal_after,
        ):
            design_rejection_streak = 0
            previous_design_proposal = design_proposal
            design_counterproposal = call_json_object(
                y_client,
                system=AGENT_Y_SYSTEM,
                user=prompt_y_design_book_counterproposal(context, previous_design_proposal, design_review, cycle),
                role="agent_x_design",
                metadata={"role": "agent_y_design_counterproposal", "stage": "xy_design_counterproposal", "shard": shard_index, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "Y", "cycle": cycle, "mode": "design_counterproposal", "payload": design_counterproposal})
            reverse_design_review = call_json_object(
                x_client,
                system=AGENT_X_SYSTEM,
                user=prompt_x_design_book_reverse_review(context, previous_design_proposal, design_counterproposal, cycle),
                role="agent_y_design",
                metadata={"role": "agent_x_design_reverse_review", "stage": "xy_design_reverse_review", "shard": shard_index, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "X", "cycle": cycle, "mode": "design_reverse_review", "payload": reverse_design_review})
            if cycle >= min_rounds and design_book_agrees(reverse_design_review, design_counterproposal):
                design_consensus = call_json_object(
                    y_client,
                    system=AGENT_Y_SYSTEM,
                    user=prompt_y_design_book_final(context, design_counterproposal, reverse_design_review),
                    role="xy_design_consensus",
                    metadata={"role": "xy_design_reverse_consensus", "stage": "xy_design_reverse_final", "shard": shard_index, "cycle": cycle},
                    json_repair_attempts=args.json_repair_attempts,
                )
                artifacts.append({"role": "Y", "cycle": cycle, "mode": "design_reverse_final", "payload": dict(design_consensus)})
                if design_rules_from_payload(design_consensus):
                    write_json(shard_dir / "xy_design_experience_book.json", design_consensus)
                    break
                design_consensus = None
            design_proposal = design_counterproposal
            design_review = reverse_design_review
        else:
            design_rejection_streak += 1
        if cycle >= max_rounds:
            break
        design_proposal = call_json_object(
            x_client,
            system=AGENT_X_SYSTEM,
            user=prompt_x_design_book_revision(context, design_proposal, design_review, cycle + 1),
            role="agent_x_design",
            metadata={"role": "agent_x", "stage": "xy_design_revision", "shard": shard_index, "cycle": cycle + 1},
            json_repair_attempts=args.json_repair_attempts,
        )
        artifacts.append({"role": "X", "cycle": cycle + 1, "mode": "design_revision", "payload": design_proposal})

    if design_consensus is None:
        consensus = {
            "status": "unresolved",
            "agent": "controller",
            "shard_id": shard_index,
            "reason": "X/Y did not reach an executable design_experience_book consensus.",
            "design_experience_book": design_rules_from_payload(design_proposal),
            "last_design_review": design_review,
            "agreed_candidate_specs": [],
        }
        consensus["dialogue"] = compact_dialogue(artifacts)
        write_json(shard_dir / "xy_debate.json", consensus)
        return consensus

    proposal = call_json_object(
        z_client,
        system=AGENT_Z_SYSTEM,
        user=prompt_z_proposal(context, design_consensus),
        role="agent_z",
        metadata={"role": "agent_z", "stage": "zw_proposal", "shard": shard_index, "cycle": 1},
        json_repair_attempts=args.json_repair_attempts,
    )
    artifacts.append({"role": "Z", "cycle": 1, "mode": "candidate_proposal", "payload": proposal})
    review: dict[str, Any] = {}
    consensus: dict[str, Any] | None = None
    materialization_feedback: dict[str, Any] | None = None
    zw_rejection_streak = 0
    for cycle in range(1, max_rounds + 1):
        review = call_json_object(
            w_client,
            system=AGENT_W_SYSTEM,
            user=prompt_w_review(context, design_consensus, proposal, cycle),
            role="agent_w",
            metadata={"role": "agent_w", "stage": "zw_review", "shard": shard_index, "cycle": cycle},
            json_repair_attempts=args.json_repair_attempts,
        )
        artifacts.append({"role": "W", "cycle": cycle, "mode": "candidate_review", "payload": review})
        if cycle >= min_rounds and y_agrees(review, proposal, shard_target):
            consensus = call_json_object(
                w_client,
                system=AGENT_W_SYSTEM,
                user=prompt_w_final(context, design_consensus, proposal, review),
                role="zw_consensus",
                metadata={"role": "zw_consensus", "stage": "zw_final", "shard": shard_index, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "W", "cycle": cycle, "mode": "candidate_final", "payload": dict(consensus)})
            specs = candidate_specs_from_payload(consensus)
            _, frozen_specs, materialization_feedback = materialization_dry_run_feedback(
                pool_records=pool_records,
                specs=specs,
                target_count=shard_target,
                seed=args.seed_base + shard_index * 1000 + cycle,
                max_sites=args.max_sites,
                known_formulas=known_formulas,
                allowed_sources=allowed_candidate_sources(args),
                max_per_reduced_formula=args.max_per_reduced_formula,
                require_design_rule_ids=True,
                allow_structure_dicts=not args.generator_template_only,
                mattergen_config=mattergen_config_from_args(args, root, shard_dir / "mattergen_dry_run" / f"cycle_{cycle:02d}"),
            )
            artifacts.append(
                {
                    "role": "controller",
                    "cycle": cycle,
                    "mode": "materialization_dry_run",
                    "payload": materialization_feedback,
                }
            )
            if materialization_feedback.get("status") == "passed":
                consensus["agreed_candidate_specs"] = frozen_specs
                consensus["materialization_preflight"] = {
                    key: value
                    for key, value in materialization_feedback.items()
                    if key != "locked_candidate_specs_to_preserve"
                }
                break
            proposal = consensus
            consensus = None
            break
        if review_requires_counterproposal(
            review,
            rejection_streak=zw_rejection_streak,
            threshold=args.critic_counterproposal_after,
        ):
            zw_rejection_streak = 0
            previous_proposal = proposal
            counterproposal = call_json_object(
                w_client,
                system=AGENT_W_SYSTEM,
                user=prompt_w_counterproposal(context, design_consensus, previous_proposal, review, cycle),
                role="agent_z",
                metadata={"role": "agent_w_counterproposal", "stage": "zw_counterproposal", "shard": shard_index, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "W", "cycle": cycle, "mode": "candidate_counterproposal", "payload": counterproposal})
            reverse_review = call_json_object(
                z_client,
                system=AGENT_Z_SYSTEM,
                user=prompt_z_reverse_review(context, design_consensus, previous_proposal, counterproposal, cycle),
                role="agent_w",
                metadata={"role": "agent_z_reverse_review", "stage": "zw_reverse_review", "shard": shard_index, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "Z", "cycle": cycle, "mode": "candidate_reverse_review", "payload": reverse_review})
            if cycle >= min_rounds and y_agrees(reverse_review, counterproposal, shard_target):
                consensus = call_json_object(
                    w_client,
                    system=AGENT_W_SYSTEM,
                    user=prompt_w_final(context, design_consensus, counterproposal, reverse_review),
                    role="zw_consensus",
                    metadata={"role": "zw_reverse_consensus", "stage": "zw_reverse_final", "shard": shard_index, "cycle": cycle},
                    json_repair_attempts=args.json_repair_attempts,
                )
                artifacts.append({"role": "W", "cycle": cycle, "mode": "candidate_reverse_final", "payload": dict(consensus)})
                specs = candidate_specs_from_payload(consensus)
                _, frozen_specs, materialization_feedback = materialization_dry_run_feedback(
                    pool_records=pool_records,
                    specs=specs,
                    target_count=shard_target,
                    seed=args.seed_base + shard_index * 1000 + cycle,
                    max_sites=args.max_sites,
                    known_formulas=known_formulas,
                    allowed_sources=allowed_candidate_sources(args),
                    max_per_reduced_formula=args.max_per_reduced_formula,
                    require_design_rule_ids=True,
                    allow_structure_dicts=not args.generator_template_only,
                    mattergen_config=mattergen_config_from_args(args, root, shard_dir / "mattergen_dry_run" / f"reverse_cycle_{cycle:02d}"),
                )
                artifacts.append(
                    {
                        "role": "controller",
                        "cycle": cycle,
                        "mode": "reverse_materialization_dry_run",
                        "payload": materialization_feedback,
                    }
                )
                if materialization_feedback.get("status") == "passed":
                    consensus["agreed_candidate_specs"] = frozen_specs
                    consensus["materialization_preflight"] = {
                        key: value
                        for key, value in materialization_feedback.items()
                        if key != "locked_candidate_specs_to_preserve"
                    }
                    break
                proposal = consensus
                consensus = None
                break
            proposal = counterproposal
            review = reverse_review
        else:
            zw_rejection_streak += 1
        if cycle >= max_rounds:
            break
        proposal = call_json_object(
            z_client,
            system=AGENT_Z_SYSTEM,
            user=prompt_z_revision(context, design_consensus, proposal, review, cycle + 1),
            role="agent_z",
            metadata={"role": "agent_z", "stage": "zw_revision", "shard": shard_index, "cycle": cycle + 1},
            json_repair_attempts=args.json_repair_attempts,
        )
        artifacts.append({"role": "Z", "cycle": cycle + 1, "mode": "candidate_revision", "payload": proposal})

    if consensus is None and materialization_feedback is not None:
        for repair_round in range(1, max(0, int(args.max_materialization_repair_rounds)) + 1):
            proposal = call_json_object(
                z_client,
                system=AGENT_Z_SYSTEM,
                user=prompt_z_materialization_repair(context, design_consensus, proposal, materialization_feedback, repair_round),
                role="agent_z",
                metadata={
                    "role": "agent_z",
                    "stage": "zw_materialization_repair",
                    "shard": shard_index,
                    "repair_round": repair_round,
                },
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "Z", "mode": "materialization_repair", "cycle": repair_round, "payload": proposal})
            review = call_json_object(
                w_client,
                system=AGENT_W_SYSTEM,
                user=prompt_w_materialization_repair_review(context, design_consensus, proposal, materialization_feedback, repair_round),
                role="agent_w",
                metadata={
                    "role": "agent_w",
                    "stage": "zw_materialization_repair_review",
                    "shard": shard_index,
                    "repair_round": repair_round,
                },
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "W", "mode": "materialization_repair_review", "cycle": repair_round, "payload": review})
            if not y_agrees(review, proposal, shard_target):
                materialization_feedback = {
                    "status": "failed",
                    "target_count": shard_target,
                    "locked_count": 0,
                    "raw_candidate_count": len(candidate_specs_from_payload(proposal)),
                    "errors": [
                        "Agent W did not approve the repaired proposal.",
                        *review_rejection_messages(review),
                    ],
                    "locked_candidate_specs_to_preserve": materialization_feedback.get("locked_candidate_specs_to_preserve", []),
                    "two_stage_requirements": {"require_design_rule_ids": True},
                }
                continue
            consensus = call_json_object(
                w_client,
                system=AGENT_W_SYSTEM,
                user=prompt_w_final(context, design_consensus, proposal, review),
                role="zw_consensus",
                metadata={
                    "role": "zw_consensus",
                    "stage": "zw_materialization_repair_final",
                    "shard": shard_index,
                    "repair_round": repair_round,
                },
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "W", "mode": "materialization_repair_final", "cycle": repair_round, "payload": dict(consensus)})
            specs = candidate_specs_from_payload(consensus)
            _, frozen_specs, materialization_feedback = materialization_dry_run_feedback(
                pool_records=pool_records,
                specs=specs,
                target_count=shard_target,
                seed=args.seed_base + shard_index * 1000 + repair_round,
                max_sites=args.max_sites,
                known_formulas=known_formulas,
                allowed_sources=allowed_candidate_sources(args),
                max_per_reduced_formula=args.max_per_reduced_formula,
                require_design_rule_ids=True,
                allow_structure_dicts=not args.generator_template_only,
                mattergen_config=mattergen_config_from_args(args, root, shard_dir / "mattergen_repair" / f"round_{repair_round:02d}"),
            )
            artifacts.append(
                {
                    "role": "controller",
                    "mode": "materialization_repair_dry_run",
                    "cycle": repair_round,
                    "payload": materialization_feedback,
                }
            )
            if materialization_feedback.get("status") == "passed":
                consensus["agreed_candidate_specs"] = frozen_specs
                consensus["materialization_preflight"] = {
                    key: value
                    for key, value in materialization_feedback.items()
                    if key != "locked_candidate_specs_to_preserve"
                }
                break
            proposal = consensus
            consensus = None

    if consensus is None:
        preserved_specs = locked_candidate_specs_from_feedback(materialization_feedback)
        if preserved_specs:
            preflight = {
                key: value
                for key, value in (materialization_feedback or {}).items()
                if key != "locked_candidate_specs_to_preserve"
            }
            preflight["status"] = "partial"
            preflight["locked_count"] = len(preserved_specs)
            preflight["partial_reason"] = (
                "Preserving candidates that already passed blind Z/W consensus and controller "
                "materialization; failed candidates are omitted for oversampling/backfill."
            )
            consensus = {
                "status": "partial_consensus",
                "agent": "controller",
                "shard_id": shard_index,
                "reason": "Shard did not fully converge, but some candidates were already executable and locked.",
                "design_experience_book": design_rules_from_payload(design_consensus),
                "agreed_candidate_specs": preserved_specs,
                "materialization_preflight": preflight,
                "last_candidate_specs": candidate_specs_from_payload(proposal),
                "last_review": review,
                "last_materialization_feedback": materialization_feedback,
            }
        else:
            consensus = {
                "status": "unresolved",
                "agent": "controller",
                "shard_id": shard_index,
                "reason": "Z/W did not reach acceptable executable consensus within debate/materialization repair limits.",
                "design_experience_book": design_rules_from_payload(design_consensus),
                "last_candidate_specs": candidate_specs_from_payload(proposal),
                "last_review": review,
                "last_materialization_feedback": materialization_feedback,
            }
    consensus["generation_protocol"] = "two_stage"
    consensus["xy_design_consensus"] = {
        key: value
        for key, value in design_consensus.items()
        if key != "dialogue"
    }
    consensus.setdefault("design_experience_book", design_rules_from_payload(design_consensus))
    consensus["dialogue"] = compact_dialogue(artifacts)
    write_json(shard_dir / "xy_debate.json", consensus)
    return consensus


def run_shard_debate(
    *,
    args: argparse.Namespace,
    root: Path,
    work_dir: Path,
    state_path: Path,
    candidate_pool_path: Path,
    state: Mapping[str, Any],
    pool_summary: Mapping[str, Any],
    pool_records: Sequence[Mapping[str, Any]],
    known_formulas: set[str] | None,
    shard_index: int,
    shard_target: int,
    total_target: int,
) -> dict[str, Any]:
    shard_dir = work_dir / "debates" / f"shard_{shard_index:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    context = shard_context_payload(
        state=state,
        pool_summary=pool_summary,
        mode=args.mode,
        generation_protocol=str(getattr(args, "generation_protocol", "direct") or "direct"),
        candidate_source=str(getattr(args, "candidate_source", "generator") or "generator"),
        shard_index=shard_index,
        shard_target=shard_target,
        total_target=total_target,
        seed=args.seed_base + shard_index,
    )
    if require_design_rule_ids_for_args(args):
        return run_two_stage_shard_debate(
            args=args,
            root=root,
            work_dir=work_dir,
            state_path=state_path,
            candidate_pool_path=candidate_pool_path,
            context=context,
            pool_records=pool_records,
            known_formulas=known_formulas,
            shard_index=shard_index,
            shard_target=shard_target,
        )
    x_client = make_client(
        role="X",
        args=args,
        root=root,
        log_dir=shard_dir / "x_llm_calls",
        state_path=state_path,
        candidate_pool_path=candidate_pool_path,
    )
    y_client = make_client(
        role="Y",
        args=args,
        root=root,
        log_dir=shard_dir / "y_llm_calls",
        state_path=state_path,
        candidate_pool_path=candidate_pool_path,
    )

    artifacts: list[dict[str, Any]] = []
    proposal = call_json_object(
        x_client,
        system=AGENT_DIRECT_X_SYSTEM,
        user=prompt_x_proposal(context),
        role="agent_x",
        metadata={"role": "agent_x", "stage": "xy_proposal", "shard": shard_index, "cycle": 1},
        json_repair_attempts=args.json_repair_attempts,
    )
    artifacts.append({"role": "X", "cycle": 1, "payload": proposal})
    review: dict[str, Any] = {}
    consensus: dict[str, Any] | None = None
    materialization_feedback: dict[str, Any] | None = None
    max_rounds = max(1, int(args.max_debate_rounds))
    min_rounds = max(1, min(int(args.min_debate_rounds), max_rounds))
    xy_rejection_streak = 0
    for cycle in range(1, max_rounds + 1):
        review = call_json_object(
            y_client,
            system=AGENT_DIRECT_Y_SYSTEM,
            user=prompt_y_review(context, proposal, cycle),
            role="agent_y",
            metadata={"role": "agent_y", "stage": "xy_review", "shard": shard_index, "cycle": cycle},
            json_repair_attempts=args.json_repair_attempts,
        )
        artifacts.append({"role": "Y", "cycle": cycle, "payload": review})
        if cycle >= min_rounds and y_agrees(review, proposal, shard_target):
            consensus = call_json_object(
                y_client,
                system=AGENT_DIRECT_Y_SYSTEM,
                user=prompt_y_final(context, proposal, review),
                role="xy_consensus",
                metadata={"role": "xy_consensus", "stage": "xy_final", "shard": shard_index, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "Y", "cycle": cycle, "mode": "final", "payload": dict(consensus)})
            specs = candidate_specs_from_payload(consensus)
            _, frozen_specs, materialization_feedback = materialization_dry_run_feedback(
                pool_records=pool_records,
                specs=specs,
                target_count=shard_target,
                seed=args.seed_base + shard_index * 1000 + cycle,
                max_sites=args.max_sites,
                known_formulas=known_formulas,
                allowed_sources=allowed_candidate_sources(args),
                max_per_reduced_formula=args.max_per_reduced_formula,
                require_design_rule_ids=False,
                allow_structure_dicts=not args.generator_template_only,
                mattergen_config=mattergen_config_from_args(args, root, shard_dir / "mattergen_dry_run" / f"cycle_{cycle:02d}"),
            )
            artifacts.append(
                {
                    "role": "controller",
                    "cycle": cycle,
                    "mode": "materialization_dry_run",
                    "payload": materialization_feedback,
                }
            )
            if materialization_feedback.get("status") == "passed":
                consensus["agreed_candidate_specs"] = frozen_specs
                consensus["materialization_preflight"] = {
                    key: value
                    for key, value in materialization_feedback.items()
                    if key != "locked_candidate_specs_to_preserve"
                }
                break
            proposal = consensus
            consensus = None
            break
        if review_requires_counterproposal(
            review,
            rejection_streak=xy_rejection_streak,
            threshold=args.critic_counterproposal_after,
        ):
            xy_rejection_streak = 0
            previous_proposal = proposal
            counterproposal = call_json_object(
                y_client,
                system=AGENT_DIRECT_Y_SYSTEM,
                user=prompt_y_counterproposal(context, previous_proposal, review, cycle),
                role="agent_x",
                metadata={"role": "agent_y_counterproposal", "stage": "xy_counterproposal", "shard": shard_index, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "Y", "cycle": cycle, "mode": "counterproposal", "payload": counterproposal})
            reverse_review = call_json_object(
                x_client,
                system=AGENT_DIRECT_X_SYSTEM,
                user=prompt_x_reverse_review(context, previous_proposal, counterproposal, cycle),
                role="agent_y",
                metadata={"role": "agent_x_reverse_review", "stage": "xy_reverse_review", "shard": shard_index, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "X", "cycle": cycle, "mode": "reverse_review", "payload": reverse_review})
            if cycle >= min_rounds and y_agrees(reverse_review, counterproposal, shard_target):
                consensus = call_json_object(
                    y_client,
                    system=AGENT_DIRECT_Y_SYSTEM,
                    user=prompt_y_final(context, counterproposal, reverse_review),
                    role="xy_consensus",
                    metadata={"role": "xy_reverse_consensus", "stage": "xy_reverse_final", "shard": shard_index, "cycle": cycle},
                    json_repair_attempts=args.json_repair_attempts,
                )
                artifacts.append({"role": "Y", "cycle": cycle, "mode": "reverse_final", "payload": dict(consensus)})
                specs = candidate_specs_from_payload(consensus)
                _, frozen_specs, materialization_feedback = materialization_dry_run_feedback(
                    pool_records=pool_records,
                    specs=specs,
                    target_count=shard_target,
                    seed=args.seed_base + shard_index * 1000 + cycle,
                    max_sites=args.max_sites,
                    known_formulas=known_formulas,
                    allowed_sources=allowed_candidate_sources(args),
                    max_per_reduced_formula=args.max_per_reduced_formula,
                    require_design_rule_ids=False,
                    allow_structure_dicts=not args.generator_template_only,
                    mattergen_config=mattergen_config_from_args(args, root, shard_dir / "mattergen_dry_run" / f"reverse_cycle_{cycle:02d}"),
                )
                artifacts.append(
                    {
                        "role": "controller",
                        "cycle": cycle,
                        "mode": "reverse_materialization_dry_run",
                        "payload": materialization_feedback,
                    }
                )
                if materialization_feedback.get("status") == "passed":
                    consensus["agreed_candidate_specs"] = frozen_specs
                    consensus["materialization_preflight"] = {
                        key: value
                        for key, value in materialization_feedback.items()
                        if key != "locked_candidate_specs_to_preserve"
                    }
                    break
                proposal = consensus
                consensus = None
                break
            proposal = counterproposal
            review = reverse_review
        else:
            xy_rejection_streak += 1
        if cycle >= max_rounds:
            break
        proposal = call_json_object(
            x_client,
            system=AGENT_DIRECT_X_SYSTEM,
            user=prompt_x_revision(context, proposal, review, cycle + 1),
            role="agent_x",
            metadata={"role": "agent_x", "stage": "xy_revision", "shard": shard_index, "cycle": cycle + 1},
            json_repair_attempts=args.json_repair_attempts,
        )
        artifacts.append({"role": "X", "cycle": cycle + 1, "payload": proposal})

    if consensus is None and materialization_feedback is not None:
        for repair_round in range(1, max(0, int(args.max_materialization_repair_rounds)) + 1):
            proposal = call_json_object(
                x_client,
                system=AGENT_DIRECT_X_SYSTEM,
                user=prompt_x_materialization_repair(context, proposal, materialization_feedback, repair_round),
                role="agent_x",
                metadata={
                    "role": "agent_x",
                    "stage": "xy_materialization_repair",
                    "shard": shard_index,
                    "repair_round": repair_round,
                },
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "X", "mode": "materialization_repair", "cycle": repair_round, "payload": proposal})
            review = call_json_object(
                y_client,
                system=AGENT_DIRECT_Y_SYSTEM,
                user=prompt_y_materialization_repair_review(context, proposal, materialization_feedback, repair_round),
                role="agent_y",
                metadata={
                    "role": "agent_y",
                    "stage": "xy_materialization_repair_review",
                    "shard": shard_index,
                    "repair_round": repair_round,
                },
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "Y", "mode": "materialization_repair_review", "cycle": repair_round, "payload": review})
            if not y_agrees(review, proposal, shard_target):
                materialization_feedback = {
                    "status": "failed",
                    "target_count": shard_target,
                    "locked_count": 0,
                    "raw_candidate_count": len(candidate_specs_from_payload(proposal)),
                    "errors": [
                        "Agent Y did not approve the repaired proposal.",
                        *review_rejection_messages(review),
                    ],
                    "locked_candidate_specs_to_preserve": materialization_feedback.get("locked_candidate_specs_to_preserve", []),
                }
                continue
            consensus = call_json_object(
                y_client,
                system=AGENT_DIRECT_Y_SYSTEM,
                user=prompt_y_final(context, proposal, review),
                role="xy_consensus",
                metadata={
                    "role": "xy_consensus",
                    "stage": "xy_materialization_repair_final",
                    "shard": shard_index,
                    "repair_round": repair_round,
                },
                json_repair_attempts=args.json_repair_attempts,
            )
            artifacts.append({"role": "Y", "mode": "materialization_repair_final", "cycle": repair_round, "payload": dict(consensus)})
            specs = candidate_specs_from_payload(consensus)
            _, frozen_specs, materialization_feedback = materialization_dry_run_feedback(
                pool_records=pool_records,
                specs=specs,
                target_count=shard_target,
                seed=args.seed_base + shard_index * 1000 + repair_round,
                max_sites=args.max_sites,
                known_formulas=known_formulas,
                allowed_sources=allowed_candidate_sources(args),
                max_per_reduced_formula=args.max_per_reduced_formula,
                require_design_rule_ids=False,
                allow_structure_dicts=not args.generator_template_only,
                mattergen_config=mattergen_config_from_args(args, root, shard_dir / "mattergen_repair" / f"round_{repair_round:02d}"),
            )
            artifacts.append(
                {
                    "role": "controller",
                    "mode": "materialization_repair_dry_run",
                    "cycle": repair_round,
                    "payload": materialization_feedback,
                }
            )
            if materialization_feedback.get("status") == "passed":
                consensus["agreed_candidate_specs"] = frozen_specs
                consensus["materialization_preflight"] = {
                    key: value
                    for key, value in materialization_feedback.items()
                    if key != "locked_candidate_specs_to_preserve"
                }
                break
            proposal = consensus
            consensus = None

    if consensus is None:
        preserved_specs = locked_candidate_specs_from_feedback(materialization_feedback)
        if preserved_specs:
            preflight = {
                key: value
                for key, value in (materialization_feedback or {}).items()
                if key != "locked_candidate_specs_to_preserve"
            }
            preflight["status"] = "partial"
            preflight["locked_count"] = len(preserved_specs)
            preflight["partial_reason"] = (
                "Preserving candidates that already passed blind X/Y consensus and controller "
                "materialization; failed candidates are omitted for oversampling/backfill."
            )
            consensus = {
                "status": "partial_consensus",
                "agent": "controller",
                "shard_id": shard_index,
                "reason": "Shard did not fully converge, but some candidates were already executable and locked.",
                "agreed_candidate_specs": preserved_specs,
                "materialization_preflight": preflight,
                "last_candidate_specs": candidate_specs_from_payload(proposal),
                "last_review": review,
                "last_materialization_feedback": materialization_feedback,
            }
        else:
            consensus = {
                "status": "unresolved",
                "agent": "controller",
                "shard_id": shard_index,
                "reason": "X/Y did not reach acceptable executable consensus within debate/materialization repair limits.",
                "last_candidate_specs": candidate_specs_from_payload(proposal),
                "last_review": review,
                "last_materialization_feedback": materialization_feedback,
            }
    consensus["dialogue"] = compact_dialogue(artifacts)
    write_json(shard_dir / "xy_debate.json", consensus)
    return consensus


def forbidden_selection_reasons(value: Any, *, path: str = "candidate") -> list[str]:
    reasons: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            lower_key = key_text.lower()
            mattergen_conditioning_field = (
                lower_key == "energy_above_hull"
                and ".mattergen_requests[" in path
                and path.endswith(".properties_to_condition_on")
            )
            if not mattergen_conditioning_field and (
                lower_key in DEFAULT_FORBIDDEN_SELECTION_FIELDS or any(
                field in lower_key for field in DEFAULT_FORBIDDEN_SELECTION_FIELDS
                )
            ):
                reasons.append(f"{path}.{key_text} uses forbidden selection field")
            if lower_key == "preferred_order":
                orders = item if isinstance(item, list) else [item]
                for order in orders:
                    order_text = str(order or "").strip()
                    lower_order = order_text.lower()
                    if any(field in lower_order for field in DEFAULT_FORBIDDEN_SELECTION_FIELDS):
                        reasons.append(f"{path}.{key_text} uses forbidden preferred_order {order_text!r}")
                    elif lower_order not in DEFAULT_ALLOWED_PREFERRED_ORDERS:
                        reasons.append(f"{path}.{key_text} uses unsupported preferred_order {order_text!r}")
            reasons.extend(forbidden_selection_reasons(item, path=f"{path}.{key_text}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reasons.extend(forbidden_selection_reasons(item, path=f"{path}[{index}]"))
    return reasons


def _parse_generator_role_token(token: str) -> tuple[str, dict[str, Any]] | None:
    match = re.match(
        r"^(?P<role>A|B2|B|X)\s*[:=]\s*(?P<element>[A-Z][a-z]?)\s*(?:[:=]|\()?[\s]*(?P<oxi>[+-]?\d+)\)?$",
        token.strip(),
    )
    if not match:
        return None
    role = match.group("role")
    element = match.group("element")
    try:
        element = Element(element).symbol
        oxidation_state = int(match.group("oxi"))
    except Exception:
        return None
    return role, {"element": element, "oxidation_state": oxidation_state}


def parse_formula_probe_string(value: str, *, default_id: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse compact generator strings into the existing formula_probe object."""

    text = str(value or "").strip()
    if not text:
        return None, "formula_probe_string is empty"
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"formula_probe_string JSON parse failed: {exc}"
        if isinstance(parsed, Mapping):
            return dict(parsed), None
        return None, "formula_probe_string JSON must decode to an object"

    parts = [part.strip() for part in re.split(r"[;|]", text) if part.strip()]
    probe: dict[str, Any] = {"id": default_id, "roles": {}}
    role_errors: list[str] = []
    for index, part in enumerate(parts):
        lower = part.lower()
        if lower.startswith("template=") or lower.startswith("template:"):
            probe["template"] = part.split("=", 1)[1].strip() if "=" in part else part.split(":", 1)[1].strip()
            continue
        if lower.startswith("family=") or lower.startswith("family:"):
            probe["family"] = part.split("=", 1)[1].strip() if "=" in part else part.split(":", 1)[1].strip()
            continue
        if lower.startswith("id=") or lower.startswith("id:"):
            probe["id"] = part.split("=", 1)[1].strip() if "=" in part else part.split(":", 1)[1].strip()
            continue
        if index == 0 and "template" not in probe and part in ALLOWED_TEMPLATES:
            probe["template"] = part
            continue
        parsed_role = _parse_generator_role_token(part)
        if parsed_role is None:
            role_errors.append(part)
            continue
        role, role_payload = parsed_role
        probe["roles"][role] = role_payload

    template = str(probe.get("template") or "").strip()
    if template not in ALLOWED_TEMPLATES:
        return None, f"formula_probe_string template must be one of {list(ALLOWED_TEMPLATES)}, got {template!r}"
    required = set(required_roles(template))
    actual = set(probe["roles"])
    if missing := sorted(required - actual):
        return None, f"formula_probe_string missing roles for {template}: {missing}"
    if extra := sorted(actual - required):
        return None, f"formula_probe_string has extra roles for {template}: {extra}"
    if role_errors:
        return None, f"formula_probe_string has unparsed tokens: {role_errors}"
    if not str(probe.get("id") or "").strip():
        probe["id"] = default_id
    return probe, None


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0"}:
            return False
    return None


def _nonempty_string_or_list(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(bool(str(item).strip()) for item in value)
    return False


TEMPLATE_PROXY_REJECT_PHRASES = (
    "stoichiometry vehicle",
    "stoichiometric vehicle",
    "stoichiometry proxy",
    "stoichiometric proxy",
    "formula-only proxy",
    "formula only proxy",
    "generic formula container",
    "generic role container",
    "generic stoichiometry container",
    "template proxy",
    "proxy topology",
    "proxy for",
    "only a stoichiometry",
    "only a stoichiometric",
    "merely matching stoichiometry",
    "may not capture the ideal",
    "does not capture the ideal",
    "cannot capture the ideal",
    "may not capture the claimed",
    "does not capture the claimed",
    "cannot capture the claimed",
)


def _template_faithfulness_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(_template_faithfulness_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_template_faithfulness_text(item) for item in value)
    return ""


def _template_proxy_phrase_is_negated(normalized_text: str, phrase_start: int) -> bool:
    preceding = normalized_text[:phrase_start]
    clause = re.split(r"[.;:\n]", preceding)[-1]
    return bool(
        re.search(
            r"\b(not|no|never|without|nor)\b(?:\W+\w+){0,12}\W*$",
            clause,
        )
        or re.search(
            r"\b(rather than|instead of)\b(?:\W+\w+){0,12}\W*$",
            clause,
        )
    )


def template_proxy_rejection_reason(*values: Any) -> str:
    text = " ".join(_template_faithfulness_text(value) for value in values)
    normalized = re.sub(r"[\s_-]+", " ", text.strip().lower())
    if not normalized:
        return ""
    for phrase in TEMPLATE_PROXY_REJECT_PHRASES:
        for match in re.finditer(re.escape(phrase), normalized):
            if _template_proxy_phrase_is_negated(normalized, match.start()):
                continue
            return f"template faithfulness text admits an unfaithful proxy pattern: {phrase!r}"
    return ""


def _generator_input_template(candidate: Mapping[str, Any]) -> str:
    requests = candidate.get("mattergen_requests")
    if isinstance(requests, list) and requests:
        return "mattergen"
    probes = candidate.get("formula_probes")
    if isinstance(probes, list) and probes and isinstance(probes[0], Mapping):
        return str(probes[0].get("template") or "").strip()
    structures = candidate.get("structure_dicts")
    if isinstance(structures, list) and structures:
        return "structure_dict"
    return ""


def validate_template_consistency_audit(candidate: Mapping[str, Any], *, candidate_id: str) -> list[str]:
    """Require explicit mechanism-to-generator-template consistency before materialization."""

    errors: list[str] = []
    source = str(candidate.get("source") or "").strip().lower()
    if source != "generator":
        return errors
    audit = candidate.get("template_consistency_audit")
    if not isinstance(audit, Mapping):
        return [f"{candidate_id}: generator candidate requires template_consistency_audit"]

    input_template = _generator_input_template(candidate)
    chosen_template = str(audit.get("chosen_template") or audit.get("template") or "").strip()
    if not chosen_template:
        errors.append(f"{candidate_id}: template_consistency_audit.chosen_template is required")
    elif input_template and input_template != "structure_dict" and chosen_template != input_template:
        errors.append(
            f"{candidate_id}: template_consistency_audit.chosen_template {chosen_template!r} "
            f"does not match generator template {input_template!r}"
        )
    elif input_template == "mattergen" and chosen_template != "mattergen":
        errors.append(
            f"{candidate_id}: template_consistency_audit.chosen_template for MatterGen requests must be 'mattergen'"
        )
    elif input_template == "structure_dict" and chosen_template not in {"structure_dict", "custom_structure"} and chosen_template not in ALLOWED_TEMPLATES:
        errors.append(
            f"{candidate_id}: template_consistency_audit.chosen_template for structure_dict must be "
            f"'structure_dict', 'custom_structure', or an allowed template"
        )

    realizes = _bool_value(audit.get("template_realizes_motif"))
    if realizes is not True:
        errors.append(f"{candidate_id}: template_consistency_audit.template_realizes_motif must be true")

    unsupported = _bool_value(audit.get("unsupported_motif_substitution"))
    if unsupported is not False:
        errors.append(f"{candidate_id}: template_consistency_audit.unsupported_motif_substitution must be false")

    if not _nonempty_string_or_list(audit.get("mechanism_local_motif")):
        errors.append(f"{candidate_id}: template_consistency_audit.mechanism_local_motif is required")
    if not _nonempty_string_or_list(audit.get("required_coordination_or_polyhedra")):
        errors.append(f"{candidate_id}: template_consistency_audit.required_coordination_or_polyhedra is required")
    if not _nonempty_string_or_list(audit.get("why_template_is_faithful")):
        errors.append(f"{candidate_id}: template_consistency_audit.why_template_is_faithful is required")
    proxy_reason = template_proxy_rejection_reason(
        audit.get("why_template_is_faithful"),
        audit.get("mechanism_local_motif"),
        audit.get("required_coordination_or_polyhedra"),
        audit.get("generator_limitations"),
        candidate.get("why_template_is_faithful"),
        candidate.get("expected_local_motif"),
        candidate.get("mechanism_rationale"),
    )
    if proxy_reason:
        errors.append(f"{candidate_id}: {proxy_reason}; choose a template whose topology realizes the claimed motif")

    return errors


def _normalize_formula_probes(candidate: Mapping[str, Any], *, candidate_id: str) -> tuple[list[Any], list[str]]:
    errors: list[str] = []
    normalized: list[Any] = []
    raw_probes = candidate.get("formula_probes")
    if isinstance(raw_probes, list):
        for index, item in enumerate(raw_probes):
            if isinstance(item, Mapping):
                normalized.append(dict(item))
            elif isinstance(item, str):
                parsed, error = parse_formula_probe_string(item, default_id=f"{candidate_id}_probe_{index + 1:03d}")
                if error:
                    errors.append(f"{candidate_id}: formula_probes[{index}] {error}")
                elif parsed is not None:
                    normalized.append(parsed)
            else:
                errors.append(f"{candidate_id}: formula_probes[{index}] must be an object or generator string")
    raw_strings = candidate.get("formula_probe_strings")
    if isinstance(raw_strings, list):
        for index, item in enumerate(raw_strings):
            if not isinstance(item, str):
                errors.append(f"{candidate_id}: formula_probe_strings[{index}] must be a string")
                continue
            parsed, error = parse_formula_probe_string(item, default_id=f"{candidate_id}_string_{index + 1:03d}")
            if error:
                errors.append(f"{candidate_id}: formula_probe_strings[{index}] {error}")
            elif parsed is not None:
                normalized.append(parsed)
    return normalized, errors


def _normalize_mattergen_requests(candidate: Mapping[str, Any], *, candidate_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    raw_requests = candidate.get("mattergen_requests")
    candidate_target_formula = str(
        candidate.get("target_reduced_formula") or candidate.get("reduced_formula") or candidate.get("formula") or ""
    ).strip()
    candidate_has_require_target = "require_target_reduced_formula" in candidate
    candidate_require_target = candidate.get("require_target_reduced_formula")
    if raw_requests is None and isinstance(candidate.get("mattergen_request"), Mapping):
        raw_requests = [candidate.get("mattergen_request")]
    if raw_requests is None:
        return normalized, errors
    if not isinstance(raw_requests, list):
        return normalized, [f"{candidate_id}: mattergen_requests must be a list"]
    for index, item in enumerate(raw_requests):
        if not isinstance(item, Mapping):
            errors.append(f"{candidate_id}: mattergen_requests[{index}] must be an object")
            continue
        request = dict(item)
        backend = str(request.get("backend", "mattergen")).strip().lower()
        if backend != "mattergen":
            errors.append(f"{candidate_id}: mattergen_requests[{index}].backend must be 'mattergen'")
            continue
        request["backend"] = "mattergen"
        request.setdefault("request_id", f"{candidate_id}_mattergen_{index + 1:03d}")
        properties = request.get("properties_to_condition_on")
        if not isinstance(properties, Mapping):
            properties = {}
        properties = dict(properties)
        top_level_system = request.pop("chemical_system", None)
        filters = request.get("filters")
        if not isinstance(filters, Mapping):
            filters = {}
        filters = dict(filters)
        filter_system = filters.get("chemical_system")
        chemical_system = properties.get("chemical_system") or top_level_system or filter_system
        if not chemical_system:
            errors.append(f"{candidate_id}: mattergen_requests[{index}] requires properties_to_condition_on.chemical_system")
        else:
            properties["chemical_system"] = chemical_system
            if "chemical_system" not in filters:
                filters["chemical_system"] = chemical_system
        if properties.get("energy_above_hull") is None:
            properties["energy_above_hull"] = 0.0
        try:
            properties["energy_above_hull"] = float(properties["energy_above_hull"])
        except Exception:
            errors.append(f"{candidate_id}: mattergen_requests[{index}].energy_above_hull must be numeric")
        request.setdefault("diffusion_guidance_factor", DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR)
        filters.setdefault("min_volume_per_atom", DEFAULT_MATTERGEN_MIN_VOLUME_PER_ATOM)
        filters.setdefault("max_volume_per_atom", DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM)
        request_target_formula = str(request.get("target_reduced_formula") or "").strip()
        filter_target_formula = str(filters.get("target_reduced_formula") or "").strip()
        target_formula = filter_target_formula or request_target_formula or candidate_target_formula
        if target_formula:
            filters["target_reduced_formula"] = target_formula
            request["target_reduced_formula"] = target_formula
        if "require_target_reduced_formula" not in filters and "require_target_reduced_formula" in request:
            filters["require_target_reduced_formula"] = request.get("require_target_reduced_formula")
        elif "require_target_reduced_formula" not in filters and candidate_has_require_target:
            filters["require_target_reduced_formula"] = candidate_require_target
        filters["require_target_reduced_formula"] = (
            _bool_value(filters.get("require_target_reduced_formula")) is True
        )
        request["properties_to_condition_on"] = properties
        request["filters"] = filters
        normalized.append(request)
    return normalized, errors


def _mattergen_audit_target_label(request: Mapping[str, Any] | None) -> str:
    if not isinstance(request, Mapping):
        return "the requested chemical system"
    properties = request.get("properties_to_condition_on")
    filters = request.get("filters")
    properties = properties if isinstance(properties, Mapping) else {}
    filters = filters if isinstance(filters, Mapping) else {}
    chemical_system = properties.get("chemical_system") or filters.get("chemical_system") or "the requested chemical system"
    if isinstance(chemical_system, Sequence) and not isinstance(chemical_system, (str, bytes)):
        chemical_system = "-".join(str(item).strip() for item in chemical_system if str(item).strip())
    target_formula = filters.get("target_reduced_formula") or request.get("target_reduced_formula")
    if target_formula:
        return f"{target_formula} in {chemical_system}"
    return str(chemical_system)


def _coerce_mattergen_template_audit(
    raw_audit: Any,
    *,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(raw_audit, Mapping):
        if request is None:
            return None
        raw_audit = {}
    if not isinstance(raw_audit, Mapping):
        return None
    audit = dict(raw_audit)
    target_label = _mattergen_audit_target_label(request)
    default_narrative = (
        f"MatterGen samples full periodic {target_label} structures conditioned on chemical system and low hull "
        "energy; it is the structure generator rather than a fixed-template proxy or manual cell."
    )
    narrative = str(
        audit.get("why_template_is_faithful")
        or audit.get("audit")
        or audit.get("reasoning")
        or audit.get("summary")
        or default_narrative
    ).strip()
    motif = audit.get("mechanism_local_motif") or audit.get("local_motif") or narrative
    coordination = audit.get("required_coordination_or_polyhedra") or audit.get("coordination") or audit.get("polyhedra")
    if not _nonempty_string_or_list(coordination):
        coordination = str(motif)
    chosen_template = audit.get("chosen_template") or audit.get("template")
    return {
        **audit,
        "chosen_template": str(chosen_template).strip() if chosen_template else "mattergen",
        "template_realizes_motif": _bool_value(audit.get("template_realizes_motif", audit.get("is_faithful", True))) is not False,
        "unsupported_motif_substitution": _bool_value(audit.get("unsupported_motif_substitution", False)) is True,
        "mechanism_local_motif": motif,
        "required_coordination_or_polyhedra": coordination,
        "why_template_is_faithful": narrative,
        "generator_limitations": audit.get("generator_limitations") or [],
    }


def normalize_candidate_spec(
    spec: Mapping[str, Any],
    *,
    default_id: str,
    allowed_sources: set[str] | None = None,
    require_design_rule_ids: bool = False,
    allow_structure_dicts: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    candidate = dict(spec)
    normalization_notes: list[str] = []
    if "id" not in candidate and candidate.get("candidate_id"):
        candidate["id"] = candidate.get("candidate_id")
        normalization_notes.append("mapped candidate_id to id")
    if "formula_probe_strings" not in candidate and isinstance(candidate.get("formula_probe_string"), str):
        candidate["formula_probe_strings"] = [candidate.get("formula_probe_string")]
        normalization_notes.append("mapped formula_probe_string to formula_probe_strings")
    if "formula_probes" not in candidate and "formula_probe" in candidate:
        raw_probe = candidate.get("formula_probe")
        if isinstance(raw_probe, Mapping):
            candidate["formula_probes"] = [dict(raw_probe)]
            normalization_notes.append("mapped formula_probe object to formula_probes")
        elif isinstance(raw_probe, str):
            candidate["formula_probe_strings"] = [raw_probe]
            normalization_notes.append("mapped formula_probe string to formula_probe_strings")
    if "structure_dicts" not in candidate and isinstance(candidate.get("structure_dict"), Mapping):
        candidate["structure_dicts"] = [dict(candidate["structure_dict"])]
        normalization_notes.append("mapped structure_dict to structure_dicts")
    if "mattergen_requests" not in candidate and isinstance(candidate.get("mattergen_request"), Mapping):
        candidate["mattergen_requests"] = [dict(candidate["mattergen_request"])]
        normalization_notes.append("mapped mattergen_request to mattergen_requests")
    for alias_key in ("candidate_id", "formula_probe_string", "formula_probe", "structure_dict", "mattergen_request"):
        candidate.pop(alias_key, None)
    audit = candidate.get("template_consistency_audit")
    if isinstance(audit, Mapping):
        audit = dict(audit)
        audit_aliases = (
            ("chosen_generator_template", "chosen_template"),
            ("generator_template", "chosen_template"),
            ("selected_template", "chosen_template"),
            ("claimed_template", "chosen_template"),
            ("required_microscopic_motif", "mechanism_local_motif"),
            ("local_motif", "mechanism_local_motif"),
            ("claimed_motif", "mechanism_local_motif"),
            ("motif_notes", "mechanism_local_motif"),
            ("coordination_or_polyhedra", "required_coordination_or_polyhedra"),
            ("required_coordination", "required_coordination_or_polyhedra"),
            ("polyhedra_notes", "required_coordination_or_polyhedra"),
            ("coordination_notes", "required_coordination_or_polyhedra"),
            ("faithfulness_reason", "why_template_is_faithful"),
            ("audit_reasoning", "why_template_is_faithful"),
            ("why_faithful", "why_template_is_faithful"),
            ("faithfulness_notes", "why_template_is_faithful"),
        )
        for alias, canonical in audit_aliases:
            if canonical not in audit and alias in audit:
                audit[canonical] = audit[alias]
                normalization_notes.append(f"mapped template_consistency_audit.{alias} to {canonical}")
        bool_audit_aliases = (
            ("template_faithful", "template_realizes_motif"),
            ("template_realises_motif", "template_realizes_motif"),
            ("is_faithful", "template_realizes_motif"),
            ("unsupported_proxy", "unsupported_motif_substitution"),
        )
        for alias, canonical in bool_audit_aliases:
            if canonical not in audit and alias in audit:
                audit[canonical] = audit[alias]
                normalization_notes.append(f"mapped template_consistency_audit.{alias} to {canonical}")
        for alias, _canonical in (*audit_aliases, *bool_audit_aliases):
            audit.pop(alias, None)
        has_structure_dicts = isinstance(candidate.get("structure_dicts"), list) and bool(candidate.get("structure_dicts"))
        chosen_template = str(audit.get("chosen_template") or "").strip()
        if has_structure_dicts and chosen_template:
            if chosen_template.startswith("custom_structure") and chosen_template != "custom_structure":
                audit["chosen_template"] = "custom_structure"
                normalization_notes.append(
                    f"mapped template_consistency_audit.chosen_template {chosen_template!r} to custom_structure"
                )
            elif chosen_template.startswith("structure_dict") and chosen_template != "structure_dict":
                audit["chosen_template"] = "structure_dict"
                normalization_notes.append(
                    f"mapped template_consistency_audit.chosen_template {chosen_template!r} to structure_dict"
                )
        candidate["template_consistency_audit"] = audit
    if normalization_notes:
        existing_notes = candidate.get("normalization_notes")
        notes = list(existing_notes) if isinstance(existing_notes, list) else []
        candidate["normalization_notes"] = [*notes, *normalization_notes]
    candidate_id = str(candidate.get("id") or default_id).strip()
    candidate["id"] = candidate_id
    try:
        count = int(candidate.get("count") or 1)
    except Exception:
        count = 1
    candidate["count"] = 1 if count <= 0 else min(count, 1)
    source_raw = str(candidate.get("source") or "").strip().lower()
    has_generator_payload = any(
        isinstance(candidate.get(key), list) and bool(candidate.get(key))
        for key in ("formula_probes", "formula_probe_strings", "structure_dicts", "mattergen_requests")
    )
    source = source_raw or ("generator" if has_generator_payload else "mp_pool")
    allowed = allowed_sources or {"mp_pool", "generator"}
    if source not in {"mp_pool", "generator"}:
        errors.append(f"{candidate_id}: source must be mp_pool or generator")
        source = "mp_pool"
    if source not in allowed:
        errors.append(f"{candidate_id}: source {source!r} is not allowed in this run; allowed sources are {sorted(allowed)}")
    candidate["source"] = source
    design_rule_ids = candidate.get("design_rule_ids")
    if require_design_rule_ids and (
        not isinstance(design_rule_ids, list) or not any(str(item).strip() for item in design_rule_ids)
    ):
        errors.append(f"{candidate_id}: two-stage candidate requires nonempty design_rule_ids")

    formula_probes, probe_errors = _normalize_formula_probes(candidate, candidate_id=candidate_id)
    errors.extend(probe_errors)
    if formula_probes:
        candidate["formula_probes"] = formula_probes[:1]
    candidate.pop("formula_probe_strings", None)
    mattergen_requests, mattergen_errors = _normalize_mattergen_requests(candidate, candidate_id=candidate_id)
    errors.extend(mattergen_errors)
    if mattergen_requests:
        candidate["mattergen_requests"] = mattergen_requests[:1]
        raw_audit = candidate.get("template_consistency_audit")
        if not isinstance(raw_audit, Mapping):
            raw_audit = mattergen_requests[0].get("template_consistency_audit")
        completed_audit = _coerce_mattergen_template_audit(raw_audit, request=mattergen_requests[0])
        if completed_audit is not None:
            if completed_audit != candidate.get("template_consistency_audit"):
                existing_notes = candidate.get("normalization_notes")
                notes = list(existing_notes) if isinstance(existing_notes, list) else []
                candidate["normalization_notes"] = [
                    *notes,
                    "completed MatterGen template_consistency_audit defaults",
                ]
            candidate["template_consistency_audit"] = completed_audit

    material_ids = [str(item).strip() for item in candidate.get("material_ids", []) if str(item).strip()] if isinstance(candidate.get("material_ids"), list) else []
    if candidate.get("material_id") and not material_ids:
        material_ids = [str(candidate.get("material_id")).strip()]
    candidate["material_ids"] = material_ids[:1]
    candidate.pop("material_id", None)

    exclude_formulas = _extract_excluded_formulas(candidate)
    if exclude_formulas:
        candidate["exclude_formulas"] = sorted(exclude_formulas)

    query = candidate.get("query")
    if source == "generator" and isinstance(query, Mapping):
        if query:
            errors.append(f"{candidate_id}: generator candidate must not include MP-pool query fields")
        candidate.pop("query", None)
        query = None
    if isinstance(query, Mapping):
        query = dict(query)
        query_excludes = _extract_excluded_formulas(query)
        if query_excludes:
            exclude_formulas.update(query_excludes)
            candidate["exclude_formulas"] = sorted(exclude_formulas)
        normalization_notes: list[str] = []
        for alias, canonical in (("num_sites_min", "nsites_min"), ("num_sites_max", "nsites_max")):
            if alias in query:
                if canonical not in query:
                    query[canonical] = query[alias]
                query.pop(alias, None)
                normalization_notes.append(f"mapped query.{alias} to query.{canonical}")
        for formula_key in ("formula", "formula_exact", "formula_include"):
            if formula_key in query:
                raw_formula = query.pop(formula_key)
                formulas = raw_formula if isinstance(raw_formula, list) else [raw_formula]
                if "formula_in" not in query:
                    query["formula_in"] = [str(item) for item in formulas if str(item).strip()]
                normalization_notes.append(f"mapped query.{formula_key} to query.formula_in")
        for advisory_key in list(query):
            if advisory_key in {"fields", "random_seed"} or advisory_key.startswith("deduplicate_against_"):
                query.pop(advisory_key, None)
                normalization_notes.append(f"removed advisory query.{advisory_key}")
        for unsupported_exclusion_key in (
            "exclude_formula_probes",
            "exclude_formulas",
            "formula_exclude",
            "formulas_exclude",
            "formula_reduced_exclude",
            "exclude_formula_anonymous_or_reduced",
        ):
            query.pop(unsupported_exclusion_key, None)
        if normalization_notes:
            existing_notes = candidate.get("normalization_notes")
            notes = list(existing_notes) if isinstance(existing_notes, list) else []
            candidate["normalization_notes"] = [*notes, *normalization_notes]
        query.setdefault("preferred_order", "random")
        query_errors = validate_query(query)
        if query_errors:
            errors.extend(f"{candidate_id}: query error: {message}" for message in query_errors)
        candidate["query"] = query
    elif "query" in candidate:
        errors.append(f"{candidate_id}: query must be an object")
        candidate.pop("query", None)

    errors.extend(f"{candidate_id}: {message}" for message in forbidden_selection_reasons(candidate))

    if source == "mp_pool" and not candidate["material_ids"] and not isinstance(candidate.get("query"), Mapping):
        errors.append(f"{candidate_id}: mp_pool candidate requires material_ids or query")
    if source == "generator":
        if candidate["material_ids"]:
            errors.append(f"{candidate_id}: generator candidate must not include material_ids")
        formula_probes = candidate.get("formula_probes")
        structure_dicts = candidate.get("structure_dicts")
        mattergen_requests = candidate.get("mattergen_requests")
        if not allow_structure_dicts and isinstance(structure_dicts, list) and structure_dicts:
            errors.append(
                f"{candidate_id}: generator template-only run forbids structure_dicts; "
                "use formula_probe_strings or formula_probes with an allowed template"
            )
        input_count = sum(
            1
            for value in (formula_probes, structure_dicts, mattergen_requests)
            if isinstance(value, list) and bool(value)
        )
        if input_count == 0:
            errors.append(f"{candidate_id}: generator candidate requires one formula_probe, one structure_dict, or one mattergen_request")
        if input_count > 1:
            errors.append(f"{candidate_id}: generator candidate must include only one of formula_probes, structure_dicts, or mattergen_requests")
        if isinstance(formula_probes, list) and len(formula_probes) > 1:
            candidate["formula_probes"] = formula_probes[:1]
        if isinstance(structure_dicts, list) and len(structure_dicts) > 1:
            candidate["structure_dicts"] = structure_dicts[:1]
        if isinstance(mattergen_requests, list) and len(mattergen_requests) > 1:
            candidate["mattergen_requests"] = mattergen_requests[:1]
        errors.extend(validate_template_consistency_audit(candidate, candidate_id=candidate_id))

    if errors:
        return None, errors
    return candidate, []


def _extract_excluded_formulas(value: Mapping[str, Any]) -> set[str]:
    excluded: set[str] = set()
    for key in (
        "exclude_formulas",
        "exclude_formula_probes",
        "formula_exclude",
        "formulas_exclude",
        "formula_reduced_exclude",
        "exclude_formula_anonymous_or_reduced",
    ):
        raw = value.get(key)
        excluded.update(_normalised_formula_set(raw))
    return excluded


def _normalised_formula_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        items: Iterable[Any] = [value]
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, Mapping)):
        items = value
    else:
        items = [value]
    formulas: set[str] = set()
    for item in items:
        formula = _normalize_formula_text(item)
        if formula:
            formulas.add(formula)
    return formulas


def _formula_is_excluded(record: Mapping[str, Any], excluded_formulas: set[str]) -> bool:
    if not excluded_formulas:
        return False
    formula = _normalize_formula_text(
        record.get("formula") or record.get("reduced_formula") or record.get("pretty_formula") or ""
    )
    return formula in excluded_formulas


def _normalize_formula_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if evaluator_null_element_symbols_in_formula_text(text):
        return text
    try:
        return Composition(text).reduced_formula
    except Exception:
        return text


FORMULA_LIKE_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9()]*\b")
ELEMENT_SYMBOL_TOKEN_RE = re.compile(r"[A-Z][a-z]?")
EXPLICIT_STRATEGY_FORMULA_RE = re.compile(
    r"[\"']?(?:candidate_reduced_formula|reduced_formula|formula)[\"']?\s*[:=]\s*[\"']?([A-Z][A-Za-z0-9()]*)"
)
ORDERED_STRATEGY_FORMULA_RE = re.compile(r"(?:^|[\s(:,;])(?:\d+\.|[-*])\s*([A-Z][A-Za-z0-9()]*)\b")
STRATEGY_CANDIDATE_SEGMENT_RE = re.compile(
    r"(?:candidate\s+(?:set|order|formulas?)|formula\s+set|fallback\s+order|ordered\s+route|ordered\s+[^.:\n]{0,120}\bset).*?:\s*([^.\n]+)",
    re.IGNORECASE,
)
ABSTRACT_STRATEGY_FORMULA_TOKEN_RE = re.compile(
    r"^(?:[A-Z][a-z]?BO3|BO[0-9]+|A(?:[0-9A-Z][A-Za-z0-9()]*)?|[A-Z][A-Za-z0-9()]*X[0-9]*)$"
)
COORDINATION_CONTEXT_RE = re.compile(
    r"\b(?:coordination|coordinated|tetrahedra|tetrahedral|octahedra|octahedral|polyhedra|polyhedral)\b",
    re.IGNORECASE,
)
ALL_CAPS_FORMULA_CONTEXT_RE = re.compile(
    r"\b(?:formula|candidate|set|route|propose|fallback|template|wurtzite|zincblende|rocksalt|nitride|phosphide|boride|carbide|AX|III-V)\b",
    re.IGNORECASE,
)


def evaluator_null_element_symbols_in_formula_text(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    return {
        symbol
        for symbol in ELEMENT_SYMBOL_TOKEN_RE.findall(text)
        if symbol in EVALUATOR_NULL_E_HULL_ELEMENTS
    }


def _is_likely_coordination_formula_token(token: str, text: str, start: int, end: int) -> bool:
    if not any(char.isdigit() for char in token):
        return False
    context = text[max(0, start - 40) : min(len(text), end + 40)]
    if not COORDINATION_CONTEXT_RE.search(context):
        return False
    if evaluator_null_element_symbols_in_formula_text(token):
        return False
    try:
        composition = Composition(token)
    except Exception:
        return False
    if len(composition.elements) != 2:
        return False
    amounts = sorted(float(amount) for amount in composition.as_dict().values())
    return len(amounts) == 2 and abs(amounts[0] - 1.0) < 1e-8 and amounts[1] in {4.0, 6.0, 8.0, 12.0}


def _formula_matches_allowed_template_stoichiometry(composition: Composition) -> bool:
    formula_amounts: dict[str, int] = {}
    for element, raw_amount in composition.get_el_amt_dict().items():
        try:
            amount = float(raw_amount)
        except Exception:
            return False
        rounded = int(round(amount))
        if rounded <= 0 or abs(amount - rounded) > 1e-6:
            return False
        formula_amounts[str(element)] = rounded
    if not formula_amounts:
        return False
    normalized_counts = sorted(formula_amounts.values())
    for role_counts in TEMPLATE_ROLE_COUNTS.values():
        if sorted(int(count) for count in role_counts.values()) == normalized_counts:
            return True
    return False


def _is_likely_standalone_coordination_fragment(token: str, composition: Composition) -> bool:
    amounts = {
        str(element): float(amount)
        for element, amount in composition.get_el_amt_dict().items()
    }
    if len(amounts) != 2:
        return False
    if _formula_matches_allowed_template_stoichiometry(composition):
        return False
    anion_matches = [
        (element, amount)
        for element, amount in amounts.items()
        if element in COMMON_ANION_ELEMENTS and amount in {4.0, 6.0, 8.0, 12.0}
    ]
    if len(anion_matches) != 1:
        return False
    cation_amounts = [
        amount
        for element, amount in amounts.items()
        if element != anion_matches[0][0]
    ]
    return len(cation_amounts) == 1 and abs(cation_amounts[0] - 1.0) < 1e-8


def _has_formula_case_signal(token: str, composition: Composition, text: str, start: int, end: int) -> bool:
    if any(char.isdigit() or char.islower() for char in token):
        return True
    if len(token) > 3 or len(composition.elements) != 2:
        return False
    if any(len(str(element)) != 1 for element in composition.elements):
        return False
    context = text[max(0, start - 60) : min(len(text), end + 60)]
    return ALL_CAPS_FORMULA_CONTEXT_RE.search(context) is not None


def _is_concrete_formula_token(token: str, composition: Composition) -> bool:
    if ABSTRACT_STRATEGY_FORMULA_TOKEN_RE.fullmatch(token):
        return False
    if _is_likely_standalone_coordination_fragment(token, composition):
        return False
    for element in composition.elements:
        try:
            Element(str(element))
        except Exception:
            return False
    return True


def _composition_or_none_quietly(token: str) -> Composition | None:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No Pauling electronegativity")
            return Composition(token)
    except Exception:
        return None


def strategy_formula_candidates_from_text(value: Any) -> list[str]:
    """Extract concrete formula-like tokens from strategy text for no-valid audits."""

    return [
        formula
        for formula in (_normalize_formula_text(token) for token in strategy_formula_candidate_tokens_from_text(value))
        if formula
    ]


def strategy_formula_candidate_tokens_from_text(value: Any) -> list[str]:
    """Extract formula tokens from strategy text while preserving their original display order."""

    text = str(value or "")
    tokens_out: list[str] = []
    seen: set[str] = set()
    explicit_tokens = [(match.group(1), match.start(1), match.end(1)) for match in EXPLICIT_STRATEGY_FORMULA_RE.finditer(text)]
    ordered_tokens = [(match.group(1), match.start(1), match.end(1)) for match in ORDERED_STRATEGY_FORMULA_RE.finditer(text)]
    segment_tokens: list[tuple[str, int, int]] = []
    for segment_match in STRATEGY_CANDIDATE_SEGMENT_RE.finditer(text):
        segment = segment_match.group(1)
        segment_start = segment_match.start(1)
        segment_tokens.extend(
            (match.group(0), segment_start + match.start(), segment_start + match.end())
            for match in FORMULA_LIKE_TOKEN_RE.finditer(segment)
            if not _is_likely_coordination_formula_token(
                match.group(0),
                text,
                segment_start + match.start(),
                segment_start + match.end(),
            )
        )
    tokens = (explicit_tokens + ordered_tokens + segment_tokens) or [
        (match.group(0), match.start(), match.end())
        for match in FORMULA_LIKE_TOKEN_RE.finditer(text)
        if not _is_likely_coordination_formula_token(match.group(0), text, match.start(), match.end())
    ]
    for token, start, end in tokens:
        if evaluator_null_element_symbols_in_formula_text(token):
            if token not in seen:
                seen.add(token)
                tokens_out.append(token)
            continue
        try:
            composition = Composition(token)
        except Exception:
            continue
        if len(composition.elements) < 2:
            continue
        if not _has_formula_case_signal(token, composition, text, start, end):
            continue
        if not _is_concrete_formula_token(token, composition):
            continue
        formula = composition.reduced_formula
        if formula and formula not in seen:
            seen.add(formula)
            tokens_out.append(token)
    return tokens_out


def formula_mentions_from_text(value: Any) -> list[str]:
    """Extract all concrete composition tokens from free text for controller audits."""

    text = str(value or "")
    formulas: list[str] = []
    seen: set[str] = set()
    for match in FORMULA_LIKE_TOKEN_RE.finditer(text):
        token = match.group(0)
        if evaluator_null_element_symbols_in_formula_text(token):
            continue
        try:
            composition = Composition(token)
        except Exception:
            continue
        if len(composition.elements) < 2:
            continue
        if not _has_formula_case_signal(token, composition, text, match.start(), match.end()):
            continue
        if not _is_concrete_formula_token(token, composition):
            continue
        formula = composition.reduced_formula
        if formula and formula not in seen:
            seen.add(formula)
            formulas.append(formula)
    return formulas


def abstract_formula_tokens_from_text(value: Any) -> list[str]:
    text = str(value or "")
    tokens: list[str] = []
    seen: set[str] = set()
    for match in FORMULA_LIKE_TOKEN_RE.finditer(text):
        token = match.group(0)
        if not any(char.isdigit() or char.islower() for char in token):
            continue
        if _is_likely_coordination_formula_token(token, text, match.start(), match.end()):
            continue
        if evaluator_null_element_symbols_in_formula_text(token):
            if _composition_or_none_quietly(token) is None:
                continue
            if token not in seen:
                seen.add(token)
                tokens.append(token)
            continue
        try:
            composition = Composition(token)
        except Exception:
            continue
        if len(composition.elements) < 2:
            continue
        if _is_concrete_formula_token(token, composition):
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def _record_reduced_formula(record: Mapping[str, Any]) -> str:
    for key in ("reduced_formula", "formula", "pretty_formula", "formula_pretty", "composition"):
        formula = _normalize_formula_text(record.get(key))
        if formula:
            return formula
    return ""


def _claim_reduced_formula(
    *,
    candidate_id: str,
    formula: str,
    reduced_formula_counts: dict[str, int] | None,
    max_per_reduced_formula: int,
    errors: list[str],
) -> bool:
    if not formula or reduced_formula_counts is None or max_per_reduced_formula <= 0:
        return True
    existing = reduced_formula_counts.get(formula, 0)
    if existing >= max_per_reduced_formula:
        errors.append(
            f"{candidate_id}: duplicate reduced_formula {formula} skipped "
            f"(max_per_reduced_formula={max_per_reduced_formula})"
        )
        return False
    reduced_formula_counts[formula] = existing + 1
    return True


def _role_to_ion_for_diagnostic(value: Any, *, path: str, errors: list[str]) -> Any | None:
    if isinstance(value, Mapping):
        element = value.get("element")
        oxidation_state = value.get("oxidation_state", value.get("oxi"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        element = value[0]
        oxidation_state = value[1]
    else:
        errors.append(f"{path} must be an object or [element, oxidation_state]")
        return None
    try:
        return make_ion(str(element), int(oxidation_state))
    except Exception as exc:
        errors.append(f"{path} cannot be converted to an ion: {exc}")
        return None


def _formula_probe_failure_diagnostics(
    formula_probes: Sequence[Any],
    *,
    candidate_id: str,
    max_sites: int,
    seed: int,
    known_formulas: set[str] | None,
) -> list[str]:
    diagnostics: list[str] = []
    for index, probe in enumerate(formula_probes):
        path = f"{candidate_id}: formula_probes[{index}]"
        if not isinstance(probe, Mapping):
            diagnostics.append(f"{path} is not an object")
            continue
        template = str(probe.get("template") or "").strip()
        builder = BUILDERS.get(template)
        if builder is None:
            diagnostics.append(f"{path} template {template!r} is unsupported")
            continue
        roles_raw = probe.get("roles")
        if not isinstance(roles_raw, Mapping):
            diagnostics.append(f"{path}.roles must be an object")
            continue
        required = set(required_roles(template))
        actual = {str(role) for role in roles_raw}
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        if missing:
            diagnostics.append(f"{path}.roles missing required roles {missing} for {template}")
            continue
        if extra:
            diagnostics.append(f"{path}.roles has unsupported roles {extra} for {template}")
            continue
        role_errors: list[str] = []
        roles = {}
        for role in sorted(required):
            ion = _role_to_ion_for_diagnostic(roles_raw.get(role), path=f"{path}.roles.{role}", errors=role_errors)
            if ion is not None:
                roles[role] = ion
        if role_errors:
            diagnostics.extend(role_errors)
            continue
        try:
            candidate = builder(roles, random.Random(seed + 9_100_003 + index * 131_071))
        except Exception as exc:
            diagnostics.append(f"{path} builder failed for template {template}: {type(exc).__name__}: {exc}")
            continue
        structure = candidate.structure
        formula = reduced_formula(structure)
        validation = validate_structure(structure, max_sites=max_sites)
        if not validation.ok:
            diagnostics.append(
                f"{path} generated {formula or 'unknown_formula'} with template {template} but failed structure validation: "
                f"{', '.join(validation.reasons)} "
                f"(volume_per_atom={validation.volume_per_atom:.3f}, min_distance={validation.min_distance:.3f}, "
                f"required volume_per_atom=4.0..29.5, min_distance>=0.75, max_sites={max_sites})"
            )
            continue
        if known_formulas and formula in known_formulas:
            diagnostics.append(f"{path} generated {formula} but it is in the known/training formula set")
    return diagnostics


def _resolve_project_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return root / path


def mattergen_config_from_args(args: argparse.Namespace, root: Path, work_dir: Path) -> dict[str, Any] | None:
    if str(getattr(args, "generator_backend", DEFAULT_GENERATOR_BACKEND)) != "mattergen":
        return None
    mattergen_root = _resolve_project_path(root, getattr(args, "mattergen_root", DEFAULT_MATTERGEN_ROOT))
    mattergen_venv_arg = getattr(args, "mattergen_venv", None)
    mattergen_venv = (
        _resolve_project_path(root, mattergen_venv_arg)
        if mattergen_venv_arg
        else mattergen_root / ".venv"
    )
    mattergen_bin = getattr(args, "mattergen_bin", None) or str(mattergen_venv / "bin" / "mattergen-generate")
    return {
        "root": str(root),
        "work_dir": str(work_dir),
        "adapter_path": str(_resolve_project_path(root, getattr(args, "mattergen_adapter", "mattergen_backend_prototype/mattergen_adapter.py"))),
        "mattergen_root": str(mattergen_root),
        "mattergen_venv": str(mattergen_venv),
        "mattergen_python": str(mattergen_venv / "bin" / "python"),
        "mattergen_bin": str(mattergen_bin),
        "model_path": str(_resolve_project_path(root, getattr(args, "mattergen_model_path", DEFAULT_MATTERGEN_MODEL_PATH))),
        "checkpoint": str(getattr(args, "mattergen_checkpoint", DEFAULT_MATTERGEN_CHECKPOINT)),
        "target_count": max(1, int(getattr(args, "mattergen_target_count", DEFAULT_MATTERGEN_TARGET_COUNT))),
        "batch_size": max(1, int(getattr(args, "mattergen_batch_size", DEFAULT_MATTERGEN_BATCH_SIZE))),
        "num_batches": max(1, int(getattr(args, "mattergen_num_batches", DEFAULT_MATTERGEN_NUM_BATCHES))),
        "max_sites": max(1, int(getattr(args, "mattergen_max_sites", DEFAULT_MATTERGEN_MAX_SITES))),
        "min_volume_per_atom": max(
            0.0,
            float(getattr(args, "mattergen_min_volume_per_atom", DEFAULT_MATTERGEN_MIN_VOLUME_PER_ATOM)),
        ),
        "max_volume_per_atom": max(
            0.0,
            float(getattr(args, "mattergen_max_volume_per_atom", DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM)),
        ),
        "allow_hard_target_formula": bool(
            getattr(args, "mattergen_allow_hard_target_formula", DEFAULT_MATTERGEN_ALLOW_HARD_TARGET_FORMULA)
        ),
        "diffusion_guidance_factor": max(
            0.0,
            float(getattr(args, "mattergen_diffusion_guidance_factor", DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR)),
        ),
        "partition": str(getattr(args, "mattergen_partition", DEFAULT_MATTERGEN_PARTITION)),
        "gres": str(getattr(args, "mattergen_gres", DEFAULT_MATTERGEN_GRES)),
        "cpus_per_task": max(1, int(getattr(args, "mattergen_cpus_per_task", DEFAULT_MATTERGEN_CPUS_PER_TASK))),
        "module_init": str(getattr(args, "mattergen_module_init", DEFAULT_MATTERGEN_MODULE_INIT) or ""),
        "modules": str(getattr(args, "mattergen_modules", DEFAULT_MATTERGEN_MODULES) or ""),
        "cuda_home": str(getattr(args, "mattergen_cuda_home", DEFAULT_MATTERGEN_CUDA_HOME) or ""),
        "runner": str(getattr(args, "mattergen_runner", "slurm")),
        "job_timeout": max(0, int(getattr(args, "mattergen_job_timeout", 0))),
        "poll_sec": max(1.0, float(getattr(args, "mattergen_poll_sec", 10.0))),
    }


def _chemical_system_elements(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        elements: list[str] = []
        for key in (
            "chemical_system",
            "elements",
            "allowed",
            "required",
            "allowed_elements",
            "required_elements",
        ):
            for element in _chemical_system_elements(value.get(key)):
                if element and element not in elements:
                    elements.append(element)
        return elements
    if isinstance(value, str):
        return [part.strip() for part in value.split("-") if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _xy_density_volume_cap_from_material_description(material_consensus: Mapping[str, Any] | None) -> float | None:
    """Return a conservative MatterGen VPA cap when X/Y has made high volume a binding failure boundary."""

    if not isinstance(material_consensus, Mapping):
        return None
    text = json.dumps(material_consensus, ensure_ascii=False, sort_keys=True).lower()
    if not any(pattern in text for pattern in XY_DENSITY_EDGE_TRIGGER_PATTERNS):
        return None
    return DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM


def _mattergen_config_with_xy_density_cap(
    mattergen_config: Mapping[str, Any] | None,
    material_consensus: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if mattergen_config is None:
        return None
    adjusted = dict(mattergen_config)
    adjusted["force_soft_target_reduced_formula"] = True
    density_cap = _xy_density_volume_cap_from_material_description(material_consensus)
    if density_cap is None:
        return adjusted
    adjusted["max_volume_per_atom_cap"] = min(
        float(adjusted.get("max_volume_per_atom", DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM)),
        density_cap,
    )
    return adjusted


def _structure_element_symbols(structure: Structure) -> set[str]:
    return {str(element.symbol) for element in structure.composition.elements}


def _mattergen_request_element_guard(request: Mapping[str, Any]) -> dict[str, Any]:
    filters = request.get("filters")
    filters = filters if isinstance(filters, Mapping) else {}
    chemical_system = set(_chemical_system_elements(filters.get("chemical_system")))
    allowed_elements = set(_chemical_system_elements(filters.get("allowed_elements"))) or chemical_system
    required_elements = set(_chemical_system_elements(filters.get("required_elements")))
    if not required_elements and (_bool_value(filters.get("require_chemical_system_exact")) is True):
        required_elements = set(chemical_system)
    return {
        "chemical_system": chemical_system,
        "allowed_elements": allowed_elements,
        "required_elements": required_elements,
        "require_chemical_system_exact": _bool_value(filters.get("require_chemical_system_exact")) is True,
    }


def _mattergen_structure_element_rejection_reason(structure: Structure, request: Mapping[str, Any]) -> str | None:
    guard = _mattergen_request_element_guard(request)
    structure_elements = _structure_element_symbols(structure)
    allowed_elements = guard["allowed_elements"]
    if allowed_elements and not structure_elements <= allowed_elements:
        return (
            "outside_allowed_elements:"
            f" structure_elements={sorted(structure_elements)} allowed_elements={sorted(allowed_elements)}"
        )
    chemical_system = guard["chemical_system"]
    if guard["require_chemical_system_exact"] and chemical_system and structure_elements != chemical_system:
        return (
            "chemical_system_not_exact:"
            f" structure_elements={sorted(structure_elements)} chemical_system={sorted(chemical_system)}"
        )
    required_elements = guard["required_elements"]
    missing = sorted(required_elements - structure_elements)
    if missing:
        return (
            "missing_required_elements:"
            f" missing={missing} structure_elements={sorted(structure_elements)}"
        )
    return None


def _normalise_mattergen_request_for_run(
    raw_request: Mapping[str, Any],
    *,
    candidate_id: str,
    mattergen_config: Mapping[str, Any],
    excluded_formulas: set[str],
    max_sites: int,
    count: int,
    candidate_target_reduced_formula: Any | None = None,
    candidate_require_target_reduced_formula: Any | None = None,
) -> dict[str, Any]:
    def positive_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = int(default)
        return parsed if parsed > 0 else int(default)

    def ensure_normalization_notes() -> list[Any]:
        notes = request.get("controller_normalization_notes")
        if not isinstance(notes, list):
            notes = []
            request["controller_normalization_notes"] = notes
        return notes

    request = dict(raw_request)
    request["backend"] = "mattergen"
    request.setdefault("request_id", f"{candidate_id}_mattergen")
    request.setdefault("checkpoint", mattergen_config.get("checkpoint") or DEFAULT_MATTERGEN_CHECKPOINT)
    request.setdefault("model_path", mattergen_config.get("model_path"))
    request["target_count"] = max(
        int(count),
        positive_int(request.get("target_count"), int(mattergen_config.get("target_count") or count)),
        positive_int(mattergen_config.get("target_count"), int(count)),
    )
    request_batch_size = positive_int(request.get("batch_size"), DEFAULT_MATTERGEN_BATCH_SIZE)
    configured_batch_size = positive_int(mattergen_config.get("batch_size"), DEFAULT_MATTERGEN_BATCH_SIZE)
    request_num_batches = positive_int(request.get("num_batches"), DEFAULT_MATTERGEN_NUM_BATCHES)
    configured_num_batches = positive_int(mattergen_config.get("num_batches"), DEFAULT_MATTERGEN_NUM_BATCHES)
    request["batch_size"] = max(request_batch_size, configured_batch_size)
    request["num_batches"] = max(request_num_batches, configured_num_batches)
    if request["batch_size"] != request_batch_size or request["num_batches"] != request_num_batches:
        ensure_normalization_notes().append(
            "raised MatterGen batch_size/num_batches to the controller-configured sampling budget"
        )
    configured_guidance_factor = float(
        mattergen_config.get("diffusion_guidance_factor", DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR)
    )
    raw_guidance_factor = request.get("diffusion_guidance_factor")
    if raw_guidance_factor not in (None, ""):
        try:
            raw_guidance_float = float(raw_guidance_factor)
        except Exception:
            raw_guidance_float = configured_guidance_factor
        if abs(raw_guidance_float - configured_guidance_factor) > 1e-9:
            ensure_normalization_notes().append("forced diffusion_guidance_factor to configured controller value")
    request["diffusion_guidance_factor"] = configured_guidance_factor
    properties = dict(request.get("properties_to_condition_on") or {})
    properties.setdefault("energy_above_hull", 0.0)
    request["properties_to_condition_on"] = properties
    filters = dict(request.get("filters") or {})
    chemical_system = filters.get("chemical_system") or properties.get("chemical_system")
    elements = _chemical_system_elements(chemical_system)
    if elements:
        filters["chemical_system"] = elements
        properties["chemical_system"] = "-".join(elements)
    allow_subset_fallback = _bool_value(
        filters.pop("allow_subset_fallback", request.pop("allow_subset_fallback", None))
    ) is True
    required_elements = _chemical_system_elements(
        filters.get("required_elements") or request.get("required_elements")
    )
    allowed_elements = _chemical_system_elements(
        filters.get("allowed_elements") or request.get("allowed_elements")
    )
    if elements and not required_elements and not allow_subset_fallback:
        required_elements = list(elements)
    if elements and not allowed_elements:
        allowed_elements = list(elements)
    if allowed_elements:
        filters["allowed_elements"] = allowed_elements
    if required_elements:
        filters["required_elements"] = required_elements
    if elements and not allow_subset_fallback:
        filters["require_chemical_system_exact"] = True
    else:
        default_exact = False if allow_subset_fallback else bool(elements)
        filters["require_chemical_system_exact"] = _bool_value(
            filters.get("require_chemical_system_exact", default_exact)
        ) is True
    request_target_formula = str(request.get("target_reduced_formula") or "").strip()
    filter_target_formula = str(filters.get("target_reduced_formula") or "").strip()
    candidate_target_formula = str(candidate_target_reduced_formula or "").strip()
    target_formula = filter_target_formula or request_target_formula or candidate_target_formula
    if target_formula:
        filters["target_reduced_formula"] = target_formula
        request["target_reduced_formula"] = target_formula
    if "require_target_reduced_formula" not in filters and "require_target_reduced_formula" in request:
        filters["require_target_reduced_formula"] = request.get("require_target_reduced_formula")
    elif "require_target_reduced_formula" not in filters and candidate_require_target_reduced_formula is not None:
        filters["require_target_reduced_formula"] = candidate_require_target_reduced_formula
    normalization_notes = list(request.get("controller_normalization_notes") or [])
    filters["require_target_reduced_formula"] = (
        _bool_value(filters.get("require_target_reduced_formula")) is True
    )
    force_soft_target = _bool_value(mattergen_config.get("force_soft_target_reduced_formula")) is True
    if filters["require_target_reduced_formula"] and force_soft_target:
        filters["require_target_reduced_formula"] = False
        normalization_notes.append(
            "controller_forced_soft_target_formula: sequential X/Y MatterGen blind SUN optimization "
            "uses target_reduced_formula only as a soft preference; exact chemical-system conditioning, "
            "required elements, exclusions, and density/site filters remain binding."
        )
    if filters["require_target_reduced_formula"] and len(elements) > DEFAULT_MATTERGEN_HARD_TARGET_MAX_EXACT_ELEMENTS:
        filters["require_target_reduced_formula"] = False
        request["num_batches"] = max(
            int(request.get("num_batches") or DEFAULT_MATTERGEN_NUM_BATCHES),
            DEFAULT_MATTERGEN_HARD_TARGET_MIN_NUM_BATCHES,
        )
        normalization_notes.append(
            "controller_softened_hard_target_formula: X/Y MatterGen SUN optimization treats "
            f"chemical systems with more than {DEFAULT_MATTERGEN_HARD_TARGET_MAX_EXACT_ELEMENTS} elements "
            "as exact-chemical-system searches with target_reduced_formula only as a soft preference."
        )
    if filters["require_target_reduced_formula"]:
        request["num_batches"] = max(
            int(request.get("num_batches") or DEFAULT_MATTERGEN_NUM_BATCHES),
            DEFAULT_MATTERGEN_HARD_TARGET_MIN_NUM_BATCHES,
        )
    filters.setdefault("deduplicate_reduced_formula", True)
    filters["max_sites"] = min(max_sites, int(filters.get("max_sites") or mattergen_config.get("max_sites") or max_sites))
    filters.setdefault("min_volume_per_atom", mattergen_config.get("min_volume_per_atom", DEFAULT_MATTERGEN_MIN_VOLUME_PER_ATOM))
    filters.setdefault("max_volume_per_atom", mattergen_config.get("max_volume_per_atom", DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM))
    if mattergen_config.get("max_volume_per_atom_cap") is not None:
        try:
            cap = float(mattergen_config["max_volume_per_atom_cap"])
            current_max = float(filters.get("max_volume_per_atom", DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM))
        except Exception:
            cap = None
            current_max = None
        if cap is not None and cap > 0 and current_max is not None and current_max > cap:
            filters["max_volume_per_atom"] = cap
            request["max_volume_per_atom"] = cap
            normalization_notes.append(
                "controller_density_volume_cap: max_volume_per_atom reduced "
                f"from {current_max:g} to {cap:g} because the X/Y material description or postmortem "
                "made high-volume/low-density structures a binding failure boundary."
            )
    existing_excluded = _normalised_formula_set(filters.get("exclude_reduced_formulas"))
    filters["exclude_reduced_formulas"] = sorted(existing_excluded | _normalised_formula_set(excluded_formulas))
    if normalization_notes:
        request["controller_normalization_notes"] = normalization_notes
    request["filters"] = filters
    return request


def _tail_text(path: Path, *, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _write_mattergen_sbatch(
    *,
    run_dir: Path,
    request_path: Path,
    mattergen_config: Mapping[str, Any],
) -> Path:
    root = Path(str(mattergen_config["root"]))
    adapter_path = Path(str(mattergen_config["adapter_path"]))
    mattergen_root = Path(str(mattergen_config["mattergen_root"]))
    mattergen_venv = Path(str(mattergen_config.get("mattergen_venv") or mattergen_root / ".venv"))
    mattergen_bin = Path(str(mattergen_config["mattergen_bin"]))
    module_init = str(mattergen_config.get("module_init") or "").strip()
    module_names = [
        item
        for item in re.split(r"[\s,]+", str(mattergen_config.get("modules") or "").strip())
        if item
    ]
    cuda_home = str(mattergen_config.get("cuda_home") or "").strip()
    module_setup_lines: list[str] = []
    if module_init:
        module_setup_lines.append(f"source {shlex.quote(module_init)}")
    for module_name in module_names:
        module_setup_lines.append(f"module load {shlex.quote(module_name)}")
    cuda_setup_lines: list[str] = []
    if cuda_home:
        cuda_setup_lines.extend(
            [
                f"export CUDA_HOME={shlex.quote(cuda_home)}",
                'export PATH="${CUDA_HOME}/bin:${PATH}"',
                'export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"',
            ]
        )
    module_setup = "\n".join(module_setup_lines)
    cuda_setup = "\n".join(cuda_setup_lines)
    script_path = run_dir / "run_mattergen.sbatch"
    script = f"""#!/bin/bash
#SBATCH --job-name=xy_mattergen
#SBATCH --partition={mattergen_config.get("partition", DEFAULT_MATTERGEN_PARTITION)}
#SBATCH --gres={mattergen_config.get("gres", DEFAULT_MATTERGEN_GRES)}
#SBATCH --cpus-per-task={mattergen_config.get("cpus_per_task", DEFAULT_MATTERGEN_CPUS_PER_TASK)}
#SBATCH --output={run_dir / "slurm-%j.out"}
#SBATCH --error={run_dir / "slurm-%j.err"}

set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
{module_setup}
{cuda_setup}
export PATH=${{HOME}}/.local/bin:{mattergen_venv / "bin"}:${{PATH}}
source {mattergen_venv / "bin" / "activate"}
cd {mattergen_root}

echo "host=$(hostname)"
echo "started=$(date -Is)"
echo "mattergen_venv={mattergen_venv}"
echo "mattergen_bin={mattergen_bin}"
echo "CUDA_HOME=${{CUDA_HOME:-}}"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
    print("cuda_capability", torch.cuda.get_device_capability(0))
    x = torch.ones((8, 8), device="cuda")
    print("cuda_tensor_sum", float((x @ x).sum().item()))
PY

PYTHONPATH={root}/src:${{PYTHONPATH:-}} python {adapter_path} \\
  --request {request_path} \\
  --work-dir {run_dir} \\
  --mattergen-bin {mattergen_bin} \\
  > {run_dir / "adapter_stdout.log"} 2> {run_dir / "adapter_stderr.log"}

echo "finished=$(date -Is)"
"""
    script_path.write_text(script, encoding="utf-8")
    return script_path


def _submit_and_wait_mattergen(script_path: Path, *, mattergen_config: Mapping[str, Any], run_dir: Path) -> str:
    submit = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        cwd=str(Path(str(mattergen_config["root"]))),
        text=True,
        capture_output=True,
        check=False,
    )
    if submit.returncode != 0:
        raise RuntimeError(f"sbatch failed: {submit.stderr.strip() or submit.stdout.strip()}")
    job_id = submit.stdout.strip().split(";", 1)[0]
    (run_dir / "slurm_job_id.txt").write_text(job_id + "\n", encoding="utf-8")
    print(f"[{utc_now()}] mattergen_submitted job_id={job_id} run_dir={run_dir}", flush=True)
    started = time.time()
    timeout = int(mattergen_config.get("job_timeout") or 0)
    poll_sec = float(mattergen_config.get("poll_sec") or 10.0)
    while True:
        queued = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T|%r"],
            text=True,
            capture_output=True,
            check=False,
        )
        queued_text = queued.stdout.strip()
        if not queued_text:
            break
        state_reason = queued_text.splitlines()[0].strip()
        state, _, reason = state_reason.partition("|")
        reason_lower = reason.lower()
        if state.upper() == "PENDING" and any(pattern in reason_lower for pattern in MATTERGEN_SLURM_FATAL_PENDING_REASONS):
            raise RuntimeError(f"MatterGen Slurm job {job_id} is pending with fatal reason: {reason or state_reason}")
        if timeout and time.time() - started > timeout:
            raise TimeoutError(f"MatterGen Slurm job {job_id} exceeded timeout {timeout}s")
        time.sleep(poll_sec)
    print(f"[{utc_now()}] mattergen_finished job_id={job_id} elapsed_sec={time.time() - started:.1f}", flush=True)
    return job_id


def _run_mattergen_request(
    request: Mapping[str, Any],
    *,
    candidate_id: str,
    mattergen_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(str(mattergen_config["root"]))
    base_work_dir = Path(str(mattergen_config["work_dir"]))
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id)[:80] or "candidate"
    run_dir = base_work_dir / safe_id
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "request.json"
    write_json(request_path, dict(request))
    runner = str(mattergen_config.get("runner") or "slurm")
    if runner == "local":
        command = [
            str(mattergen_config.get("mattergen_python") or sys.executable),
            str(_resolve_project_path(root, mattergen_config["adapter_path"])),
            "--request",
            str(request_path),
            "--work-dir",
            str(run_dir),
            "--mattergen-bin",
            str(mattergen_config["mattergen_bin"]),
        ]
        for source_path in mattergen_config.get("from_existing", []) or []:
            command.extend(["--from-existing", str(source_path)])
        completed = subprocess.run(
            command,
            cwd=str(root),
            text=True,
            stdout=(run_dir / "adapter_stdout.log").open("w", encoding="utf-8"),
            stderr=(run_dir / "adapter_stderr.log").open("w", encoding="utf-8"),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"MatterGen adapter failed with exit code {completed.returncode}")
    else:
        script_path = _write_mattergen_sbatch(
            run_dir=run_dir,
            request_path=request_path,
            mattergen_config=mattergen_config,
        )
        _submit_and_wait_mattergen(script_path, mattergen_config=mattergen_config, run_dir=run_dir)

    report_path = run_dir / "generation_report.json"
    input_path = run_dir / "input.json"
    records_path = run_dir / "selected_records.json"
    if not report_path.exists() or not input_path.exists() or not records_path.exists():
        raise RuntimeError(
            "MatterGen adapter did not produce expected outputs; "
            f"stdout_tail={_tail_text(run_dir / 'adapter_stdout.log')!r}; "
            f"stderr_tail={_tail_text(run_dir / 'adapter_stderr.log')!r}"
        )
    report = read_json(report_path, {})
    input_structures = read_json(input_path, [])
    adapter_records = read_json(records_path, [])
    if not isinstance(input_structures, list) or not isinstance(adapter_records, list):
        raise RuntimeError("MatterGen adapter outputs must be JSON lists")
    return list(input_structures), [dict(item) for item in adapter_records if isinstance(item, Mapping)], dict(report)


def _formula_halogen_to_nonhalogen_ratio(formula: Any) -> float | None:
    try:
        composition = Composition(str(formula or ""))
    except Exception:
        return None
    amounts = composition.get_el_amt_dict()
    halogen_amount = sum(float(amounts.get(element) or 0.0) for element in COMMON_HALOGEN_ELEMENTS)
    total_amount = sum(float(value or 0.0) for value in amounts.values())
    nonhalogen_amount = total_amount - halogen_amount
    if halogen_amount <= 0 or nonhalogen_amount <= 0:
        return None
    return halogen_amount / nonhalogen_amount


def _mattergen_halogen_stoichiometry_rejection(formula: Any, request: Mapping[str, Any]) -> str:
    filters = request.get("filters") if isinstance(request.get("filters"), Mapping) else {}
    target_formula = (
        filters.get("target_reduced_formula")
        or request.get("target_reduced_formula")
        or request.get("reduced_formula")
        or ""
    )
    target_ratio = _formula_halogen_to_nonhalogen_ratio(target_formula)
    if target_ratio is None or target_ratio < 0.75:
        return ""
    generated_ratio = _formula_halogen_to_nonhalogen_ratio(formula)
    if generated_ratio is None:
        return ""
    min_ratio = max(0.5, min(1.0, 0.5 * target_ratio))
    if generated_ratio + 1e-9 >= min_ratio:
        return ""
    return (
        "halogen_stoichiometry_guard:"
        f" generated_formula={formula} halogen_to_nonhalogen_ratio={generated_ratio:.3g} "
        f"is below required_min={min_ratio:.3g} inferred from target_reduced_formula={target_formula} "
        f"(target_ratio={target_ratio:.3g}); reject Br/F/Cl/I-deficient exact-system drift."
    )


def materialize_one_candidate(
    pool_records: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    *,
    seed: int,
    seen_material_ids: set[str],
    reduced_formula_counts: dict[str, int] | None,
    max_per_reduced_formula: int,
    max_sites: int,
    known_formulas: set[str] | None,
    mattergen_config: Mapping[str, Any] | None = None,
    additional_excluded_formulas: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    candidate_id = str(spec.get("id") or "candidate")
    source = str(spec.get("source") or "mp_pool")
    count = int(spec.get("count") or 1)
    excluded_formulas = set(str(item) for item in spec.get("exclude_formulas", []) if str(item).strip()) if isinstance(spec.get("exclude_formulas"), list) else set()
    if additional_excluded_formulas:
        excluded_formulas.update(_normalised_formula_set(additional_excluded_formulas))
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    if source == "mp_pool":
        material_ids = [str(item) for item in spec.get("material_ids", []) if str(item).strip()] if isinstance(spec.get("material_ids"), list) else []
        if material_ids:
            matches = [
                dict(record)
                for record in pool_records
                if str(record.get("material_id")) in set(material_ids) and not _formula_is_excluded(record, excluded_formulas)
            ]
        else:
            query = spec.get("query")
            if not isinstance(query, Mapping):
                return [], [f"{candidate_id}: mp_pool query is missing"]
            if not any(query_matches(record, query) for record in pool_records):
                return [], [f"{candidate_id}: query matched zero pool records"]
            matches = [
                record
                for record in select_matches(pool_records, query, count=100, seed=seed)
                if not _formula_is_excluded(record, excluded_formulas)
            ]
        for record in matches:
            material_id = str(record.get("material_id") or "")
            if not material_id or material_id in seen_material_ids:
                continue
            item = dict(record)
            formula = _record_reduced_formula(item)
            if formula:
                item["reduced_formula"] = formula
            if not _claim_reduced_formula(
                candidate_id=candidate_id,
                formula=formula,
                reduced_formula_counts=reduced_formula_counts,
                max_per_reduced_formula=max_per_reduced_formula,
                errors=errors,
            ):
                continue
            item["physics_bundle_id"] = candidate_id
            item["physics_prediction_ids"] = [candidate_id]
            item["physics_role"] = "xy_candidate"
            item["physics_expected_relation"] = "blind_generation"
            item["physics_selection_order"] = str(spec.get("query", {}).get("preferred_order", "random")) if isinstance(spec.get("query"), Mapping) else "exact_material_id"
            item["crystal_llm_source"] = "mp_pool"
            item["xy_candidate_spec"] = dict(spec)
            seen_material_ids.add(material_id)
            records.append(item)
            if len(records) >= count:
                break
        if len(records) < count:
            errors.append(f"{candidate_id}: materialized {len(records)} records but requested {count}")
        return records, errors

    if source == "generator":
        structure_dicts = spec.get("structure_dicts")
        formula_probes = spec.get("formula_probes")
        mattergen_requests = spec.get("mattergen_requests")
        if isinstance(mattergen_requests, list) and mattergen_requests:
            if mattergen_config is None:
                return [], [f"{candidate_id}: mattergen_requests require --generator-backend mattergen and MatterGen configuration"]
            request = _normalise_mattergen_request_for_run(
                mattergen_requests[0],
                candidate_id=candidate_id,
                mattergen_config=mattergen_config,
                excluded_formulas=excluded_formulas,
                max_sites=max_sites,
                count=count,
                candidate_target_reduced_formula=(
                    spec.get("target_reduced_formula") or spec.get("reduced_formula") or spec.get("formula")
                ),
                candidate_require_target_reduced_formula=spec.get("require_target_reduced_formula"),
            )
            try:
                input_structures, adapter_records, report = _run_mattergen_request(
                    request,
                    candidate_id=candidate_id,
                    mattergen_config=mattergen_config,
                )
            except Exception as exc:
                return [], [f"{candidate_id}: MatterGen adapter failed: {type(exc).__name__}: {exc}"]
            for index, raw_structure in enumerate(input_structures):
                try:
                    structure = Structure.from_dict(dict(raw_structure))
                except Exception as exc:
                    errors.append(f"{candidate_id}: MatterGen structure[{index}] parse failed: {exc}")
                    continue
                min_volume = float(mattergen_config.get("min_volume_per_atom", DEFAULT_MATTERGEN_MIN_VOLUME_PER_ATOM))
                max_volume = float(mattergen_config.get("max_volume_per_atom", DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM))
                validation = validate_structure(
                    structure,
                    max_sites=max_sites,
                    min_volume_per_atom=min_volume,
                    max_volume_per_atom=max_volume,
                )
                if not validation.ok:
                    errors.append(
                        f"{candidate_id}: MatterGen structure[{index}] validation failed: {', '.join(validation.reasons)} "
                        f"(volume_per_atom={validation.volume_per_atom:.3f}, min_distance={validation.min_distance:.3f}, "
                        f"required volume_per_atom={min_volume:g}..{max_volume:g}, min_distance>=0.75, max_sites={max_sites})"
                    )
                    continue
                element_rejection = _mattergen_structure_element_rejection_reason(structure, request)
                if element_rejection:
                    errors.append(f"{candidate_id}: MatterGen structure[{index}] rejected by element guard: {element_rejection}")
                    continue
                formula = reduced_formula(structure)
                halogen_rejection = _mattergen_halogen_stoichiometry_rejection(formula, request)
                if halogen_rejection:
                    errors.append(f"{candidate_id}: MatterGen generated formula {formula} rejected by {halogen_rejection}")
                    continue
                if formula in excluded_formulas:
                    errors.append(f"{candidate_id}: MatterGen generated formula {formula} is excluded")
                    continue
                if known_formulas and formula in known_formulas:
                    errors.append(f"{candidate_id}: MatterGen generated formula {formula} but it is in the known/training formula set")
                    continue
                material_id = f"mattergen::{candidate_id}::{index + 1:03d}::{formula}"
                if material_id in seen_material_ids:
                    continue
                if not _claim_reduced_formula(
                    candidate_id=candidate_id,
                    formula=formula,
                    reduced_formula_counts=reduced_formula_counts,
                    max_per_reduced_formula=max_per_reduced_formula,
                    errors=errors,
                ):
                    continue
                seen_material_ids.add(material_id)
                adapter_record = adapter_records[index] if index < len(adapter_records) else {}
                records.append(
                    {
                        **dict(adapter_record),
                        "material_id": material_id,
                        "formula": formula,
                        "cif_path": None,
                        "structure_dict": structure.as_dict(),
                        "physics_bundle_id": candidate_id,
                        "physics_prediction_ids": [candidate_id],
                        "physics_role": "xy_candidate",
                        "physics_expected_relation": "blind_generation",
                        "physics_selection_order": "mattergen",
                        "crystal_llm_source": "generator",
                        "crystal_llm_generator_backend": "mattergen",
                        "crystal_llm_generated_from_mattergen_requests": [dict(request)],
                        "crystal_llm_mattergen_report": report,
                        "xy_candidate_spec": dict(spec),
                    }
                )
                if len(records) >= count:
                    break
            if len(records) < count:
                request_systems = sorted(_extract_mattergen_request_chemical_systems_from_payload(request, max_items=4))
                request_filters = request.get("filters") if isinstance(request.get("filters"), Mapping) else {}
                request_target = (
                    request_filters.get("target_reduced_formula")
                    or request.get("target_reduced_formula")
                    or spec.get("target_reduced_formula")
                    or spec.get("reduced_formula")
                    or spec.get("formula")
                    or ""
                )
                errors.append(
                    f"{candidate_id}: MatterGen materialized {len(records)} records but requested {count}; "
                    f"chemical_system={','.join(request_systems) or 'unknown'}; "
                    f"target_reduced_formula={request_target or 'none'}; "
                    f"report_status={report.get('status')}; accepted_count={report.get('accepted_count')}; "
                    f"reject_reasons={report.get('reject_reasons')}"
                )
        elif isinstance(structure_dicts, list) and structure_dicts:
            for index, raw_structure in enumerate(structure_dicts[:count]):
                try:
                    structure = Structure.from_dict(dict(raw_structure))
                except Exception as exc:
                    errors.append(f"{candidate_id}: structure_dict[{index}] parse failed: {exc}")
                    continue
                validation = validate_structure(structure, max_sites=max_sites)
                if not validation.ok:
                    errors.append(
                        f"{candidate_id}: structure_dict[{index}] validation failed: {', '.join(validation.reasons)} "
                        f"(volume_per_atom={validation.volume_per_atom:.3f}, min_distance={validation.min_distance:.3f}, "
                        f"required volume_per_atom=4.0..29.5, min_distance>=0.75, max_sites={max_sites})"
                    )
                    continue
                formula = reduced_formula(structure)
                if formula in excluded_formulas:
                    errors.append(f"{candidate_id}: generated formula {formula} is excluded")
                    continue
                material_id = f"generated::{candidate_id}::{index + 1}::{formula}"
                if material_id in seen_material_ids:
                    continue
                if not _claim_reduced_formula(
                    candidate_id=candidate_id,
                    formula=formula,
                    reduced_formula_counts=reduced_formula_counts,
                    max_per_reduced_formula=max_per_reduced_formula,
                    errors=errors,
                ):
                    continue
                seen_material_ids.add(material_id)
                records.append(
                    {
                        "material_id": material_id,
                        "formula": formula,
                        "cif_path": None,
                        "structure_dict": structure.as_dict(),
                        "physics_bundle_id": candidate_id,
                        "physics_prediction_ids": [candidate_id],
                        "physics_role": "xy_candidate",
                        "physics_expected_relation": "blind_generation",
                        "physics_selection_order": "generator_structure",
                        "crystal_llm_source": "generator",
                        "crystal_llm_generated_from_structure_dicts": [dict(raw_structure)],
                        "xy_candidate_spec": dict(spec),
                    }
                )
        elif isinstance(formula_probes, list) and formula_probes:
            candidates = load_formula_probes({"formula_probes": formula_probes[:count]}, max_sites=max_sites, base_seed=seed, known_formulas=known_formulas)
            if len(candidates) < min(count, len(formula_probes)):
                errors.extend(
                    _formula_probe_failure_diagnostics(
                        formula_probes[:count],
                        candidate_id=candidate_id,
                        max_sites=max_sites,
                        seed=seed,
                        known_formulas=known_formulas,
                    )
                )
            for candidate in candidates[:count]:
                structure = candidate.structure
                formula = reduced_formula(structure)
                if formula in excluded_formulas:
                    errors.append(f"{candidate_id}: generated formula {formula} is excluded")
                    continue
                material_id = f"generated::{candidate_id}::{formula}"
                if material_id in seen_material_ids:
                    continue
                if not _claim_reduced_formula(
                    candidate_id=candidate_id,
                    formula=formula,
                    reduced_formula_counts=reduced_formula_counts,
                    max_per_reduced_formula=max_per_reduced_formula,
                    errors=errors,
                ):
                    continue
                seen_material_ids.add(material_id)
                records.append(
                    {
                        "material_id": material_id,
                        "formula": formula,
                        "cif_path": None,
                        "structure_dict": structure.as_dict(),
                        "physics_bundle_id": candidate_id,
                        "physics_prediction_ids": [candidate_id],
                        "physics_role": "xy_candidate",
                        "physics_expected_relation": "blind_generation",
                        "physics_selection_order": "generator_probe",
                        "crystal_llm_source": "generator",
                        "crystal_llm_generated_from_formula_probes": [dict(probe) for probe in formula_probes if isinstance(probe, Mapping)],
                        "xy_candidate_spec": dict(spec),
                    }
                )
        if len(records) < count:
            errors.append(f"{candidate_id}: generator materialized {len(records)} records but requested {count}")
        return records, errors

    return [], [f"{candidate_id}: unsupported source {source}"]


def materialize_candidate_specs(
    pool_records: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    *,
    target_count: int,
    seed: int,
    max_sites: int,
    known_formulas: set[str] | None,
    allowed_sources: set[str] | None = None,
    max_per_reduced_formula: int = 1,
    require_design_rule_ids: bool = False,
    allow_structure_dicts: bool = True,
    mattergen_config: Mapping[str, Any] | None = None,
    additional_excluded_formulas: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    accepted_specs: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_material_ids: set[str] = set()
    reduced_formula_counts: dict[str, int] = {}
    for index, raw_spec in enumerate(specs, start=1):
        normalized, normalize_errors = normalize_candidate_spec(
            raw_spec,
            default_id=f"xy_candidate_{index:03d}",
            allowed_sources=allowed_sources,
            require_design_rule_ids=require_design_rule_ids,
            allow_structure_dicts=allow_structure_dicts,
        )
        if normalize_errors or normalized is None:
            errors.extend(normalize_errors)
            continue
        if (
            mattergen_config is not None
            and target_count > 1
            and str(normalized.get("source") or "") == "generator"
            and isinstance(normalized.get("mattergen_requests"), list)
            and normalized.get("mattergen_requests")
        ):
            normalized = dict(normalized)
            normalized["count"] = max(
                int(normalized.get("count") or 1),
                min(
                    int(target_count),
                    int(mattergen_config.get("target_count") or target_count),
                ),
            )
        records, materialization_errors = materialize_one_candidate(
            pool_records,
            normalized,
            seed=seed + index,
            seen_material_ids=seen_material_ids,
            reduced_formula_counts=reduced_formula_counts,
            max_per_reduced_formula=max_per_reduced_formula,
            max_sites=max_sites,
            known_formulas=known_formulas,
            mattergen_config=mattergen_config,
            additional_excluded_formulas=additional_excluded_formulas,
        )
        errors.extend(materialization_errors)
        if records:
            accepted_specs.append(normalized)
            selected.extend(records)
        if len(selected) >= target_count:
            break
    return selected[:target_count], accepted_specs[:target_count], errors


def write_input_structures(records: Sequence[Mapping[str, Any]], input_path: Path) -> None:
    structures: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if isinstance(record.get("structure_dict"), Mapping):
            structure = Structure.from_dict(dict(record["structure_dict"]))
        else:
            cif_path = Path(str(record.get("cif_path") or ""))
            structure = Structure.from_file(str(cif_path))
        payload = structure.as_dict()
        payload["properties"] = dict(payload.get("properties") or {})
        payload["properties"].update(
            {
                "crystal_llm_material_id": record.get("material_id"),
                "crystal_llm_formula": record.get("formula"),
                "crystal_llm_source": record.get("crystal_llm_source", "mp_pool"),
                "crystal_llm_pool_metadata": {
                    "bundle_id": record.get("physics_bundle_id"),
                    "prediction_ids": record.get("physics_prediction_ids", []),
                    "role": record.get("physics_role"),
                    "expected_relation": record.get("physics_expected_relation"),
                    "selection_order": record.get("physics_selection_order"),
                    "candidate_pool_index": record.get("material_id"),
                    "xy_index": index,
                },
                "crystal_llm_xy_candidate_spec": record.get("xy_candidate_spec", {}),
            }
        )
        if record.get("crystal_llm_generated_from_formula_probes"):
            payload["properties"]["crystal_llm_generator_formula_probes"] = record.get("crystal_llm_generated_from_formula_probes")
        if record.get("crystal_llm_generated_from_structure_dicts"):
            payload["properties"]["crystal_llm_generator_structure_dicts"] = record.get("crystal_llm_generated_from_structure_dicts")
        if record.get("crystal_llm_generator_backend"):
            payload["properties"]["crystal_llm_generator_backend"] = record.get("crystal_llm_generator_backend")
        if record.get("crystal_llm_generated_from_mattergen_requests"):
            payload["properties"]["crystal_llm_generator_mattergen_requests"] = record.get("crystal_llm_generated_from_mattergen_requests")
        if record.get("crystal_llm_mattergen_report"):
            payload["properties"]["crystal_llm_mattergen_report"] = record.get("crystal_llm_mattergen_report")
        structures.append(payload)
    write_json(input_path, structures)


def compact_selected_record_for_memory(record: Mapping[str, Any]) -> dict[str, Any]:
    formula = _record_reduced_formula(record) or _normalize_formula_text(record.get("formula"))
    compact: dict[str, Any] = {
        "material_id": short_text(str(record.get("material_id") or ""), 180),
        "formula": formula,
        "reduced_formula": formula,
        "crystal_llm_source": record.get("crystal_llm_source"),
        "crystal_llm_generator_backend": record.get("crystal_llm_generator_backend"),
        "physics_bundle_id": record.get("physics_bundle_id"),
        "physics_prediction_ids": record.get("physics_prediction_ids", []),
        "physics_role": record.get("physics_role"),
        "physics_expected_relation": record.get("physics_expected_relation"),
        "physics_selection_order": record.get("physics_selection_order"),
    }
    spec = record.get("xy_candidate_spec")
    if isinstance(spec, Mapping):
        compact["xy_candidate_spec"] = compact_candidate_spec_for_prompt(spec)
    requests = record.get("crystal_llm_generated_from_mattergen_requests")
    if isinstance(requests, list):
        compact["crystal_llm_generated_from_mattergen_requests"] = [
            compact_mattergen_request_for_prompt(item) for item in requests[:2] if isinstance(item, Mapping)
        ]
    report = record.get("crystal_llm_mattergen_report")
    if isinstance(report, Mapping):
        rejection_counts = report.get("rejection_counts")
        reject_reasons = report.get("reject_reasons")
        accepted_formulas = report.get("accepted_formulas")
        if not isinstance(accepted_formulas, list):
            accepted_formulas = report.get("accepted_reduced_formulas")
        compact["crystal_llm_mattergen_report"] = {
            "status": report.get("status"),
            "target_count": report.get("target_count"),
            "accepted_count": report.get("accepted_count"),
            "generated_count": report.get("generated_count"),
            "batch_size": report.get("batch_size"),
            "num_batches": report.get("num_batches"),
            "rejection_counts": dict(rejection_counts) if isinstance(rejection_counts, Mapping) else rejection_counts,
            "reject_reasons": reject_reasons[:10] if isinstance(reject_reasons, list) else reject_reasons,
            "accepted_formulas": accepted_formulas[:24] if isinstance(accepted_formulas, list) else accepted_formulas,
        }
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def build_locked_report(
    *,
    args: argparse.Namespace,
    state: Mapping[str, Any],
    debates: Sequence[Mapping[str, Any]],
    locked_specs: Sequence[Mapping[str, Any]],
    selected_records: Sequence[Mapping[str, Any]],
    materialization_errors: Sequence[str],
    evaluation_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    principle_counts: dict[str, int] = {}
    design_rule_counts: dict[str, int] = {}
    for spec in locked_specs:
        ids = spec.get("cited_principle_ids")
        if not isinstance(ids, list):
            ids = []
        for principle_id in ids:
            text = str(principle_id).strip()
            if text:
                principle_counts[text] = principle_counts.get(text, 0) + 1
        rule_ids = spec.get("design_rule_ids")
        if not isinstance(rule_ids, list):
            rule_ids = []
        for rule_id in rule_ids:
            text = str(rule_id).strip()
            if text:
                design_rule_counts[text] = design_rule_counts.get(text, 0) + 1
    design_rule_total = sum(len(design_rules_from_payload(debate)) for debate in debates)
    return {
        "schema_version": "xy_blind_generation_report.v2",
        "created_at_utc": utc_now(),
        "mode": args.mode,
        "generation_protocol": args.generation_protocol,
        "candidate_count_requested": args.candidate_count,
        "candidate_count_locked": len(selected_records),
        "strict_sun_definition": STRICT_SUN_NOTE,
        "blind_protocol": {
            "candidate_lock_before_evaluation": True,
            "no_controller_high_sun_prefilter": True,
            "full_principle_book_visible_to_xy": args.mode == "experience_xy",
            "new_candidate_e_hull_visible_during_debate": False,
            "generation_protocol": args.generation_protocol,
            "two_stage_design_book_before_candidates": args.generation_protocol == "two_stage",
            "two_stage_candidates_require_design_rule_ids": require_design_rule_ids_for_args(args),
            "candidate_source": args.candidate_source,
            "generator_backend": getattr(args, "generator_backend", DEFAULT_GENERATOR_BACKEND),
            "allowed_material_sources": sorted(allowed_candidate_sources(args)),
            "mp_pool_final_candidates_allowed": args.candidate_source != "generator",
            "allowed_selection_orders": sorted(order for order in DEFAULT_ALLOWED_PREFERRED_ORDERS if order),
            "forbidden_selection_fields": sorted(DEFAULT_FORBIDDEN_SELECTION_FIELDS),
            "max_per_reduced_formula": args.max_per_reduced_formula,
            "hard_reduced_formula_dedup": args.max_per_reduced_formula == 1,
        },
        "source_state": {
            "current_round": state.get("current_round"),
            "principle_book_len": len(state.get("principle_book", [])) if isinstance(state.get("principle_book"), list) else 0,
        },
        "debate": {
            "shard_count": len(debates),
            "consensus_shards": sum(1 for item in debates if item.get("status") == "consensus"),
            "partial_consensus_shards": sum(1 for item in debates if item.get("status") == "partial_consensus"),
            "executable_shards": sum(1 for item in debates if item.get("status") in {"consensus", "partial_consensus"}),
            "max_debate_rounds": args.max_debate_rounds,
            "min_debate_rounds": args.min_debate_rounds,
            "design_rule_count": design_rule_total,
        },
        "experience_usage": {
            "cited_principle_counts": dict(sorted(principle_counts.items())),
            "cited_design_rule_counts": dict(sorted(design_rule_counts.items())),
            "candidate_without_principle_citation": sum(
                1
                for spec in locked_specs
                if not isinstance(spec.get("cited_principle_ids"), list) or not spec.get("cited_principle_ids")
            ),
            "candidate_without_design_rule_id": sum(
                1
                for spec in locked_specs
                if not isinstance(spec.get("design_rule_ids"), list) or not spec.get("design_rule_ids")
            ),
        },
        "materialization": {
            "errors": list(materialization_errors)[:200],
            "omitted_error_count": max(0, len(materialization_errors) - 200),
            "source_counts": _count_values(selected_records, "crystal_llm_source"),
            "formula_count": len({_record_reduced_formula(record) for record in selected_records if _record_reduced_formula(record)}),
        },
        "evaluation_summary": dict(evaluation_summary or {}),
    }


def _count_values(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def summarize_evaluation(round_dir: Path) -> dict[str, Any]:
    ranked_path = round_dir / "analysis" / "e_hull_ranked.csv"
    if not ranked_path.exists():
        return {}
    rows = load_e_hull_rows(ranked_path)
    e_hulls = [float(row["e_hull"]) for row in rows]
    if not e_hulls:
        return {"evaluated_count": 0}
    sun_count = sum(1 for value in e_hulls if value < 0.0)
    lt003 = sum(1 for value in e_hulls if value < 0.03)
    lt010 = sum(1 for value in e_hulls if value < 0.10)
    return {
        "evaluated_count": len(e_hulls),
        "sun_count": sun_count,
        "sun_ratio": sun_count / len(e_hulls),
        "e_hull_lt_0_03_count": lt003,
        "e_hull_lt_0_10_count": lt010,
        "min_e_hull": min(e_hulls),
        "mean_e_hull": sum(e_hulls) / len(e_hulls),
    }


def write_markdown_report(path: Path, report: Mapping[str, Any]) -> None:
    if report.get("generation_protocol") == "sequential_single":
        lines = [
            "# X/Y Sequential Single-Material SUN Optimizer Report",
            "",
            f"- mode: `{report.get('mode')}`",
            f"- generation_protocol: `{report.get('generation_protocol')}`",
            f"- completed_records: `{report.get('completed_records')}` / `{report.get('requested_iterations')}`",
            f"- evaluated_count: `{report.get('evaluated_count')}`",
            f"- sun_count: `{report.get('sun_count')}`",
            f"- sun_ratio: `{report.get('sun_ratio')}`",
            f"- near_stable_0_03_count: `{report.get('near_stable_0_03_count')}`",
            f"- SUN definition: `{STRICT_SUN_NOTE}`",
            "",
            "## Tail Records",
            "",
            "```json",
            prompt_json(report.get("records_tail", [])),
            "```",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    evaluation = report.get("evaluation_summary") if isinstance(report.get("evaluation_summary"), Mapping) else {}
    lines = [
        "# X/Y Blind Generation Report",
        "",
        f"- mode: `{report.get('mode')}`",
        f"- generation_protocol: `{report.get('generation_protocol')}`",
        f"- locked candidates: `{report.get('candidate_count_locked')}` / `{report.get('candidate_count_requested')}`",
        f"- protocol: candidate lock before evaluation, no controller high-SUN prefilter",
        f"- SUN definition: `{STRICT_SUN_NOTE}`",
        "",
        "## Evaluation",
        "",
        f"- evaluated_count: `{evaluation.get('evaluated_count')}`",
        f"- sun_count: `{evaluation.get('sun_count')}`",
        f"- sun_ratio: `{evaluation.get('sun_ratio')}`",
        f"- min_e_hull: `{evaluation.get('min_e_hull')}`",
        f"- mean_e_hull: `{evaluation.get('mean_e_hull')}`",
        "",
        "## Experience Usage",
        "",
        "```json",
        prompt_json(report.get("experience_usage", {})),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def material_description_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("material_description")
    return dict(value) if isinstance(value, Mapping) else {}


def material_payload_declares_no_valid_description(payload: Mapping[str, Any]) -> bool:
    """Return true when X/Y explicitly certify that no legal material can be proposed."""

    status = str(payload.get("status") or "").strip().lower()
    if status not in {"no_valid_material_description", "no_valid_material_consensus"}:
        return False
    if material_description_from_payload(payload):
        return False
    certificate = payload.get("impossibility_certificate")
    if isinstance(certificate, Mapping) and certificate:
        return True
    summary = payload.get("proposal_summary") or payload.get("debate_summary") or payload.get("overall_reasoning_summary")
    return bool(str(summary or "").strip())


def template_formula_from_material_description(payload: Mapping[str, Any]) -> str:
    description = material_description_from_payload(payload)
    raw_template = description.get("generator_template")
    if raw_template in (None, "") and description.get("template_expectation") in ALLOWED_TEMPLATES:
        raw_template = description.get("template_expectation")
    template = str(raw_template or "").strip()
    role_mapping = description.get("generator_role_mapping")
    if not isinstance(role_mapping, Mapping):
        role_mapping = description.get("role_mapping")
    if template not in ALLOWED_TEMPLATES or not isinstance(role_mapping, Mapping):
        return ""
    composition: dict[str, int] = {}
    for role, count in TEMPLATE_ROLE_COUNTS[template].items():
        raw_role = role_mapping.get(role)
        if not isinstance(raw_role, Mapping):
            return ""
        element = str(raw_role.get("element") or "").strip()
        if not element:
            return ""
        try:
            Element(element)
        except Exception:
            return ""
        composition[element] = composition.get(element, 0) + int(count)
    try:
        return Composition(composition).reduced_formula
    except Exception:
        return ""


def evaluator_null_elements_from_formula(formula_text: Any) -> set[str]:
    direct = evaluator_null_element_symbols_in_formula_text(formula_text)
    if direct:
        return direct
    normalized = _normalize_formula_text(formula_text)
    if not normalized:
        return set()
    direct = evaluator_null_element_symbols_in_formula_text(normalized)
    if direct:
        return direct
    try:
        composition = Composition(normalized)
    except Exception:
        return set()
    return {str(element) for element in composition.elements if str(element) in EVALUATOR_NULL_E_HULL_ELEMENTS}


def material_description_role_elements(payload: Mapping[str, Any]) -> set[str]:
    description = material_description_from_payload(payload)
    role_mapping = description.get("generator_role_mapping")
    if not isinstance(role_mapping, Mapping):
        role_mapping = description.get("role_mapping")
    if not isinstance(role_mapping, Mapping):
        return set()
    elements: set[str] = set()
    for raw_role in role_mapping.values():
        if not isinstance(raw_role, Mapping):
            continue
        element = str(raw_role.get("element") or "").strip()
        if element:
            elements.add(element)
    return elements


def evaluator_null_elements_from_material_payload(payload: Mapping[str, Any]) -> set[str]:
    elements = set(material_description_role_elements(payload))
    formula = template_formula_from_material_description(payload)
    if formula:
        elements.update(evaluator_null_elements_from_formula(formula))
    return {element for element in elements if element in EVALUATOR_NULL_E_HULL_ELEMENTS}


def _record_e_hull(record: Mapping[str, Any]) -> float | None:
    result = record.get("evaluation_result")
    if not isinstance(result, Mapping):
        return None
    value = result.get("e_hull")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _record_is_sun(record: Mapping[str, Any], e_hull: float | None = None) -> bool:
    result = record.get("evaluation_result")
    if not isinstance(result, Mapping):
        return False
    if bool(result.get("is_sun")):
        return True
    if e_hull is None:
        e_hull = _record_e_hull(record)
    return isinstance(e_hull, (int, float)) and e_hull < 0


def _record_formula(record: Mapping[str, Any]) -> str:
    for container_name in ("evaluation_result", "selected_record", "executable_generator_rule"):
        container = record.get(container_name)
        if isinstance(container, Mapping):
            formula = _normalize_formula_text(container.get("formula") or container.get("reduced_formula"))
            if formula:
                return formula
    formula = template_formula_from_material_description({"material_description": record.get("material_description")})
    return formula or ""


def _record_generator_template(record: Mapping[str, Any]) -> str:
    for container_name in ("executable_generator_rule", "candidate_spec", "material_description"):
        container = record.get(container_name)
        if isinstance(container, Mapping):
            template = str(container.get("generator_template") or container.get("template") or "").strip()
            if template in ALLOWED_TEMPLATES:
                return template
    selected = record.get("selected_record")
    if isinstance(selected, Mapping):
        spec = selected.get("xy_candidate_spec")
        if isinstance(spec, Mapping):
            template = str(spec.get("generator_template") or spec.get("template") or "").strip()
            if template in ALLOWED_TEMPLATES:
                return template
    return ""


def _candidate_spec_role_mapping(spec: Mapping[str, Any]) -> Mapping[str, Any] | None:
    role_mapping = spec.get("generator_role_mapping") or spec.get("role_mapping")
    if isinstance(role_mapping, Mapping) and role_mapping:
        return role_mapping
    probes = spec.get("formula_probes")
    if isinstance(probes, list):
        for probe in probes:
            if isinstance(probe, Mapping) and isinstance(probe.get("roles"), Mapping):
                return probe["roles"]
    return None


def _record_role_mapping(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    material = record.get("material_description")
    if isinstance(material, Mapping):
        role_mapping = material.get("generator_role_mapping") or material.get("role_mapping")
        if isinstance(role_mapping, Mapping) and role_mapping:
            return role_mapping
    spec = record.get("candidate_spec")
    if isinstance(spec, Mapping):
        role_mapping = _candidate_spec_role_mapping(spec)
        if role_mapping:
            return role_mapping
    selected = record.get("selected_record")
    if isinstance(selected, Mapping):
        spec = selected.get("xy_candidate_spec")
        if isinstance(spec, Mapping):
            role_mapping = _candidate_spec_role_mapping(spec)
            if role_mapping:
                return role_mapping
    return None


def _role_element(role_mapping: Mapping[str, Any] | None, role: str) -> str:
    if not isinstance(role_mapping, Mapping):
        return ""
    raw_role = role_mapping.get(role)
    if not isinstance(raw_role, Mapping):
        return ""
    element = str(raw_role.get("element") or "").strip()
    try:
        Element(element)
    except Exception:
        return ""
    return element


def _record_x_element(record: Mapping[str, Any]) -> str:
    return _role_element(_record_role_mapping(record), "X")


def _record_template_family(record: Mapping[str, Any]) -> str:
    material = record.get("material_description")
    if isinstance(material, Mapping):
        for key in ("template_formula_family", "target_family", "target_chemical_family"):
            value = short_text(material.get(key), 80)
            if value:
                return value
    return ""


def _record_basin_key(record: Mapping[str, Any]) -> str:
    template = _record_generator_template(record) or "unknown_template"
    x_element = _record_x_element(record) or "unknown_X"
    return f"template={template};X={x_element}"


def _new_basin_stats(key: str, *, template: str = "", x_element: str = "") -> dict[str, Any]:
    return {
        "basin_key": key,
        "template": template,
        "x_element": x_element,
        "attempts": 0,
        "sun_count": 0,
        "near_0_03_count": 0,
        "weak_0_10_count": 0,
        "high_e_hull_count": 0,
        "best_e_hull": None,
        "best_formula": None,
        "recent_formulas": [],
        "template_families": [],
    }


def _update_basin_stats(stats: dict[str, Any], record: Mapping[str, Any], e_hull: float) -> None:
    stats["attempts"] = int(stats.get("attempts") or 0) + 1
    if _record_is_sun(record, e_hull):
        stats["sun_count"] = int(stats.get("sun_count") or 0) + 1
    elif e_hull < 0.03:
        stats["near_0_03_count"] = int(stats.get("near_0_03_count") or 0) + 1
    elif e_hull < 0.10:
        stats["weak_0_10_count"] = int(stats.get("weak_0_10_count") or 0) + 1
    else:
        stats["high_e_hull_count"] = int(stats.get("high_e_hull_count") or 0) + 1
    best_e_hull = stats.get("best_e_hull")
    if not isinstance(best_e_hull, (int, float)) or e_hull < float(best_e_hull):
        stats["best_e_hull"] = e_hull
        stats["best_formula"] = _record_formula(record)
    formula = _record_formula(record)
    recent_formulas = stats.setdefault("recent_formulas", [])
    if formula and isinstance(recent_formulas, list) and formula not in recent_formulas:
        recent_formulas.insert(0, formula)
        del recent_formulas[5:]
    family = _record_template_family(record)
    families = stats.setdefault("template_families", [])
    if family and isinstance(families, list) and family not in families:
        families.append(family)
        del families[4:]


def _basin_score(stats: Mapping[str, Any]) -> float:
    attempts = max(1, int(stats.get("attempts") or 0))
    sun = int(stats.get("sun_count") or 0)
    near = int(stats.get("near_0_03_count") or 0)
    weak = int(stats.get("weak_0_10_count") or 0)
    high = int(stats.get("high_e_hull_count") or 0)
    best = stats.get("best_e_hull")
    best_bonus = 0.0
    if isinstance(best, (int, float)):
        best_bonus = max(0.0, 0.03 - float(best)) * 8.0
    return round((8.0 * sun + 3.0 * near + 0.7 * weak + best_bonus - 0.35 * high) / math.sqrt(attempts), 4)


def _compact_basin_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    attempts = int(stats.get("attempts") or 0)
    success_like = int(stats.get("sun_count") or 0) + int(stats.get("near_0_03_count") or 0)
    return {
        "basin_key": stats.get("basin_key"),
        "template": stats.get("template"),
        "x_element": stats.get("x_element"),
        "attempts": attempts,
        "sun_count": stats.get("sun_count", 0),
        "near_0_03_count": stats.get("near_0_03_count", 0),
        "weak_0_10_count": stats.get("weak_0_10_count", 0),
        "high_e_hull_count": stats.get("high_e_hull_count", 0),
        "success_like_rate": round(success_like / attempts, 4) if attempts else None,
        "best_e_hull": stats.get("best_e_hull"),
        "best_formula": stats.get("best_formula"),
        "recent_formulas": stats.get("recent_formulas", []),
        "template_families": stats.get("template_families", []),
        "score": _basin_score(stats),
    }


def _search_mode_for_iteration(iteration: int) -> tuple[str, int]:
    slot = (max(1, int(iteration)) - 1) % 10
    if slot < 7:
        return "exploit", slot
    if slot < 9:
        return "explore_adjacent", slot
    return "orthogonal_jump", slot


def xy_search_policy_from_memory(memory: Mapping[str, Any], *, iteration: int) -> dict[str, Any]:
    """Build the controller budget policy that keeps X/Y from either drifting or getting stuck."""

    records = memory.get("records")
    if not isinstance(records, list):
        records = []
    evaluated_records: list[Mapping[str, Any]] = []
    basin_stats: dict[str, dict[str, Any]] = {}
    template_stats: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        e_hull = _record_e_hull(record)
        if e_hull is None:
            continue
        evaluated_records.append(record)
        template = _record_generator_template(record) or "unknown_template"
        x_element = _record_x_element(record) or "unknown_X"
        basin_key = _record_basin_key(record)
        basin = basin_stats.setdefault(basin_key, _new_basin_stats(basin_key, template=template, x_element=x_element))
        _update_basin_stats(basin, record, e_hull)
        template_key = f"template={template}"
        template_entry = template_stats.setdefault(template_key, _new_basin_stats(template_key, template=template, x_element="any"))
        _update_basin_stats(template_entry, record, e_hull)

    recent_records = evaluated_records[-24:]
    recent_basin_stats: dict[str, dict[str, Any]] = {}
    recent_template_stats: dict[str, dict[str, Any]] = {}
    for record in recent_records:
        e_hull = _record_e_hull(record)
        if e_hull is None:
            continue
        template = _record_generator_template(record) or "unknown_template"
        x_element = _record_x_element(record) or "unknown_X"
        basin_key = _record_basin_key(record)
        basin = recent_basin_stats.setdefault(basin_key, _new_basin_stats(basin_key, template=template, x_element=x_element))
        _update_basin_stats(basin, record, e_hull)
        template_key = f"template={template}"
        template_entry = recent_template_stats.setdefault(template_key, _new_basin_stats(template_key, template=template, x_element="any"))
        _update_basin_stats(template_entry, record, e_hull)

    def is_cooled(stats: Mapping[str, Any], *, min_attempts: int) -> bool:
        attempts = int(stats.get("attempts") or 0)
        if attempts < min_attempts:
            return False
        good = int(stats.get("sun_count") or 0) + int(stats.get("near_0_03_count") or 0)
        high = int(stats.get("high_e_hull_count") or 0)
        return good == 0 and high / max(attempts, 1) >= 0.65

    cooled_basin_keys = sorted(key for key, stats in recent_basin_stats.items() if is_cooled(stats, min_attempts=3))
    cooled_templates = sorted(
        str(stats.get("template") or "")
        for stats in recent_template_stats.values()
        if is_cooled(stats, min_attempts=4) and str(stats.get("template") or "") in ALLOWED_TEMPLATES
    )

    top_basins = sorted(
        (_compact_basin_stats(stats) for stats in basin_stats.values()),
        key=lambda item: (
            item.get("score") if isinstance(item.get("score"), (int, float)) else -999,
            -float(item.get("best_e_hull")) if isinstance(item.get("best_e_hull"), (int, float)) else -999,
        ),
        reverse=True,
    )
    preferred_exploitation_basins = [
        item
        for item in top_basins
        if item.get("basin_key") not in cooled_basin_keys
        and item.get("template") not in cooled_templates
        and (int(item.get("sun_count") or 0) + int(item.get("near_0_03_count") or 0)) > 0
    ][:6]
    exploit_template_allowlist: list[str] = []
    for item in preferred_exploitation_basins:
        template = str(item.get("template") or "")
        if template in ALLOWED_TEMPLATES and template not in exploit_template_allowlist:
            exploit_template_allowlist.append(template)
        if len(exploit_template_allowlist) >= 3:
            break

    recent_templates = [
        template
        for template, _count in Counter(
            _record_generator_template(record) for record in evaluated_records[-12:] if _record_generator_template(record)
        ).most_common()
    ]
    preferred_jump_templates = [
        template
        for template in ALLOWED_TEMPLATES
        if template not in recent_templates and template not in cooled_templates and template not in exploit_template_allowlist
    ][:5]
    if len(preferred_jump_templates) < 2:
        for template in ALLOWED_TEMPLATES:
            if template not in cooled_templates and template not in preferred_jump_templates:
                preferred_jump_templates.append(template)
            if len(preferred_jump_templates) >= 5:
                break

    scheduled_mode, mode_slot = _search_mode_for_iteration(iteration)
    recent_tail = evaluated_records[-10:]
    recent_tail_ehulls = [_record_e_hull(record) for record in recent_tail]
    recent_tail_ehulls = [value for value in recent_tail_ehulls if isinstance(value, (int, float))]
    force_escape = bool(
        len(recent_tail_ehulls) >= 8
        and min(recent_tail_ehulls) >= 0.03
        and sum(1 for value in recent_tail_ehulls if value >= 0.10) >= 6
    )
    current_mode = "orthogonal_jump" if force_escape else scheduled_mode
    if current_mode == "exploit":
        mode_instruction = (
            "Use the best SUN/near-SUN basin allowlist; do not drift into cooled or high-e_hull-dense templates."
        )
    elif current_mode == "explore_adjacent":
        mode_instruction = (
            "Keep one successful/near-success mechanism fixed and change one meaningful axis; avoid cooled templates."
        )
    else:
        mode_instruction = (
            "Escape local minima with an underexplored native template/family; still require charge-neutral faithful templates and at least two legal queue items."
        )

    return {
        "schema_version": "xy_search_policy.v1",
        "budget": {"exploit": 0.70, "explore_adjacent": 0.20, "orthogonal_jump": 0.10},
        "current_search_mode": current_mode,
        "scheduled_search_mode": scheduled_mode,
        "cycle_slot_0_to_9": mode_slot,
        "force_escape_triggered": force_escape,
        "mode_instruction": mode_instruction,
        "controller_priority": (
            "search_policy overrides stale latest_xy_strategy_constraints when the latest route enters cooled templates "
            "or conflicts with the current 70/20/10 acquisition mode."
        ),
        "preferred_exploitation_basins": preferred_exploitation_basins,
        "exploit_template_allowlist": exploit_template_allowlist,
        "cooled_basin_keys": cooled_basin_keys[:12],
        "cooled_templates": cooled_templates,
        "preferred_jump_templates": preferred_jump_templates,
        "recent_templates": recent_templates,
        "recent_window_evaluated": len(recent_records),
        "top_basin_summary": top_basins[:8],
        "usage_note": (
            "For exploit mode, select from preferred_exploitation_basins/exploit_template_allowlist. "
            "For explore_adjacent, make one-axis variants near those basins while avoiding cooled templates. "
            "For orthogonal_jump, use preferred_jump_templates or another underexplored non-cooled native template."
        ),
    }


def mattergen_search_policy_from_template_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Convert template-budget search policy into MatterGen-compatible chemistry guidance."""

    current_mode = str(policy.get("current_search_mode") or "explore_adjacent")
    scheduled_mode = str(policy.get("scheduled_search_mode") or current_mode)

    def compact_basin(item: Any) -> dict[str, Any]:
        if not isinstance(item, Mapping):
            return {"value": str(item)[:120]}
        keep = (
            "basin_key",
            "template",
            "x_element",
            "attempts",
            "sun_count",
            "near_0_03_count",
            "best_formula",
            "best_e_hull",
            "score",
        )
        compact = {key: item.get(key) for key in keep if item.get(key) not in (None, [], {})}
        recent = item.get("recent_formulas")
        if isinstance(recent, list):
            compact["recent_formulas"] = [str(value) for value in recent[:3]]
        return compact

    def compact_list(value: Any, *, limit: int = XY_CONTEXT_MATTERGEN_LEGACY_BASIN_LIMIT) -> list[Any]:
        if not isinstance(value, list):
            return []
        return value[:limit]

    legacy_summary: dict[str, Any] = {}
    preferred = policy.get("preferred_exploitation_basins")
    if isinstance(preferred, list) and preferred:
        basin = compact_basin(preferred[0])
        legacy_summary["best_formula"] = basin.get("best_formula")
        legacy_summary["best_e_hull"] = basin.get("best_e_hull")
        legacy_summary["basin_key"] = basin.get("basin_key")
    top = policy.get("top_basin_summary")
    if isinstance(top, list) and top and not legacy_summary.get("best_formula"):
        basin = compact_basin(top[0])
        legacy_summary["best_formula"] = basin.get("best_formula")
        legacy_summary["best_e_hull"] = basin.get("best_e_hull")
        legacy_summary["basin_key"] = basin.get("basin_key")
    return {
        "schema_version": "xy_search_policy.mattergen.v1",
        "backend": "mattergen",
        "current_search_mode": current_mode,
        "force_escape_triggered": bool(policy.get("force_escape_triggered")),
        "template_policy_enforcement": "disabled_for_mattergen",
        "mode_instruction": "exploit SUN-near; explore_adjacent one-axis; orthogonal_jump new basin.",
        "legacy_best": legacy_summary,
    }


def _context_search_policy(context: Mapping[str, Any]) -> Mapping[str, Any]:
    constraints = context.get("controller_constraints")
    if isinstance(constraints, Mapping) and isinstance(constraints.get("search_policy"), Mapping):
        return constraints["search_policy"]
    return {}


def _policy_string_set(policy: Mapping[str, Any], key: str) -> set[str]:
    value = policy.get(key)
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value if str(item)}


def _latest_strategy_template_mentions(context: Mapping[str, Any] | None) -> set[str]:
    latest = context.get("latest_xy_strategy_constraints") if isinstance(context, Mapping) else None
    if not isinstance(latest, Mapping):
        return set()
    text = _jsonish_text_blob(
        latest.get("next_strategy"),
        latest.get("failure_boundaries"),
        latest.get("controller_postmortem_audit_errors"),
        latest.get("notes"),
    ).lower()
    normalized_text = re.sub(r"[_-]+", " ", text)
    mentions: set[str] = set()
    for template in ALLOWED_TEMPLATES:
        aliases = {template.lower(), template.replace("_", " ").lower()}
        for alias in aliases:
            alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
            if re.search(rf"(?<![a-z]){alias_pattern}(?![a-z])", normalized_text):
                mentions.add(template)
                break
    return mentions


def latest_strategy_search_policy_supersedes(context: Mapping[str, Any] | None) -> bool:
    """Return true when the current acquisition policy should override stale latest_strategy routing."""

    search_policy = _context_search_policy(context) if isinstance(context, Mapping) else {}
    search_mode = str(search_policy.get("current_search_mode") or "").strip()
    if not search_mode:
        return False
    latest_templates = _latest_strategy_template_mentions(context)
    if not latest_templates:
        return False
    cooled_templates = _policy_string_set(search_policy, "cooled_templates")
    if latest_templates.intersection(cooled_templates):
        return True
    if search_mode == "exploit":
        exploit_template_allowlist = _policy_string_set(search_policy, "exploit_template_allowlist")
        return bool(exploit_template_allowlist and latest_templates.isdisjoint(exploit_template_allowlist))
    if search_mode == "orthogonal_jump":
        preferred_jump_templates = _policy_string_set(search_policy, "preferred_jump_templates")
        return bool(preferred_jump_templates and latest_templates.isdisjoint(preferred_jump_templates))
    return False


def latest_strategy_viable_formula_queue(
    context: Mapping[str, Any] | None,
    return_feedback: Mapping[str, Any] | None = None,
    forbidden_formulas: set[str] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    ordered_formulas = latest_strategy_candidate_formulas(context)
    if not ordered_formulas:
        return [], {}
    blocked_by_feedback = blocked_formulas_from_return_feedback(return_feedback)
    blocked_by_boundaries = latest_strategy_failure_boundary_formulas(context)
    forbidden = _context_failed_or_used_formulas(context)
    if forbidden_formulas:
        forbidden.update(
            formula
            for formula in (_normalize_formula_text(item) for item in forbidden_formulas if str(item).strip())
            if formula
        )
    forbidden_null_elements = _context_forbidden_evaluator_null_elements(context)
    viable: list[str] = []
    blocked_reasons: dict[str, list[str]] = {}
    for formula in ordered_formulas:
        reasons: list[str] = []
        if formula in blocked_by_feedback:
            reasons.append("blocked_by_current_generator_feedback")
        if formula in blocked_by_boundaries:
            reasons.append("blocked_by_latest_failure_boundaries")
        if formula in forbidden:
            reasons.append("already_failed_or_used")
        null_elements = sorted(evaluator_null_elements_from_formula(formula).intersection(forbidden_null_elements))
        if null_elements:
            reasons.append(f"forbidden_evaluator_null_elements={null_elements}")
        cooldown_errors = strategy_cooldown_errors_for_formula(
            formula,
            context=context,
            label="latest next_strategy candidate",
        )
        if cooldown_errors:
            reasons.extend(cooldown_errors)
        if reasons:
            blocked_reasons[formula] = reasons
        else:
            viable.append(formula)
    return viable, blocked_reasons


def xy_strategy_constraints_from_context(
    context: Mapping[str, Any] | None,
    return_feedback: Mapping[str, Any] | None = None,
    forbidden_formulas: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    search_policy = _context_search_policy(context)
    latest = context.get("latest_xy_strategy_constraints")
    if not isinstance(latest, Mapping) and not search_policy:
        return {}
    queue_size = xy_sun_candidate_queue_size_from_context(context)
    search_mode = str(search_policy.get("current_search_mode") or "").strip()
    ordered = latest_strategy_candidate_formulas(context)
    viable, blocked = latest_strategy_viable_formula_queue(
        context,
        return_feedback,
        forbidden_formulas=forbidden_formulas,
    )
    latest_order_enforced = bool(ordered and viable and not latest_strategy_search_policy_supersedes(context))
    requirements = [
        "rank a queue before selecting one material",
        "avoid failed_or_used_reduced_formulas and strategy_cooldowns",
        "keep at least two legal candidates unless returning no_valid_material_description",
    ]
    if search_mode:
        requirements.append(f"set every acquisition_mode to {search_mode}")
    if latest_order_enforced:
        requirements.append("select the first legal latest-strategy formula before lower-priority alternatives")
    elif ordered and latest_strategy_search_policy_supersedes(context):
        requirements.append("search_policy supersedes stale latest-strategy formula order for this iteration")
    return {
        key: value
        for key, value in {
            "schema_version": "xy_strategy_constraints.v1",
            "binding": True,
            "queue_min_legal_items": 2,
            "queue_max_items": queue_size,
            "required_acquisition_mode": search_mode or None,
            "latest_strategy_order_enforced": latest_order_enforced,
            "ordered_candidate_formulas": ordered[:queue_size],
            "legal_ordered_candidate_formulas": viable[:queue_size],
            "first_required_formula": viable[0] if latest_order_enforced and viable else None,
            "blocked_candidate_formula_reasons": {
                formula: reasons[:3] for formula, reasons in list(blocked.items())[:queue_size]
            },
            "source_iteration": latest.get("source_iteration") if isinstance(latest, Mapping) else None,
            "source_formula": latest.get("source_formula") if isinstance(latest, Mapping) else None,
            "source_outcome_class": latest.get("outcome_class") if isinstance(latest, Mapping) else None,
            "search_policy_supersedes_latest_order": latest_strategy_search_policy_supersedes(context),
            "requirements": requirements,
            "mattergen_target_policy": (
                "reduced_formula is a soft MatterGen target, but X/Y must still choose and copy it consistently"
                if str(context.get("generator_backend") or DEFAULT_GENERATOR_BACKEND) == "mattergen"
                else None
            ),
        }.items()
        if value not in (None, "", [], {})
    }


def _queue_item_generator_template(item: Mapping[str, Any]) -> str:
    item_template = str(item.get("generator_template") or "").strip()
    if item_template in ALLOWED_TEMPLATES:
        return item_template
    material = item.get("material_description")
    if isinstance(material, Mapping):
        material_template = str(material.get("generator_template") or material.get("template_expectation") or "").strip()
        if material_template in ALLOWED_TEMPLATES:
            return material_template
    return item_template


def _queue_item_basin_key(item: Mapping[str, Any], template: str) -> str:
    material = item.get("material_description")
    role_mapping = None
    if isinstance(material, Mapping):
        role_mapping = material.get("generator_role_mapping") or material.get("role_mapping")
    x_element = _role_element(role_mapping if isinstance(role_mapping, Mapping) else None, "X")
    return f"template={template or 'unknown_template'};X={x_element or 'unknown_X'}"


def validate_xy_sun_candidate_queue(
    payload: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    forbidden_reduced_formulas: set[str] | None = None,
) -> list[str]:
    """Controller-side guard that X/Y actually compare a small candidate queue before selecting one."""

    if material_payload_declares_no_valid_description(payload):
        return []
    errors: list[str] = []
    queue_size = xy_sun_candidate_queue_size_from_context(context)
    queue = sun_candidate_queue_from_payload(payload)
    if not queue:
        errors.append(
            "X/Y output must include sun_candidate_queue with ranked concrete material descriptions before selecting one material"
        )
        return errors
    if len(queue) > queue_size:
        errors.append(f"sun_candidate_queue has {len(queue)} items but controller allows at most {queue_size}")
    if len(queue) < 2:
        errors.append("sun_candidate_queue must include at least two pre-audited concrete candidates, or return no_valid_material_description")
    forbidden_null_elements = _context_forbidden_evaluator_null_elements(context)
    forbidden_formulas = _context_failed_or_used_formulas(context)
    if forbidden_reduced_formulas:
        forbidden_formulas.update(
            formula
            for formula in (_normalize_formula_text(item) for item in forbidden_reduced_formulas if str(item).strip())
            if formula
        )
    forbidden_volume_boundaries = _context_failed_volume_boundaries(context)
    search_policy = _context_search_policy(context)
    search_mode = str(search_policy.get("current_search_mode") or "").strip()
    cooled_templates = _policy_string_set(search_policy, "cooled_templates")
    cooled_basin_keys = _policy_string_set(search_policy, "cooled_basin_keys")
    exploit_template_allowlist = _policy_string_set(search_policy, "exploit_template_allowlist")
    preferred_jump_templates = _policy_string_set(search_policy, "preferred_jump_templates")
    legal_item_ids: list[str] = []
    for index, item in enumerate(queue, start=1):
        if not isinstance(item, Mapping):
            continue
        item_id = str(item.get("candidate_id") or item.get("id") or f"rank {index}").strip()
        item_errors: list[str] = []
        item_null_elements: set[str] = set()
        item_material = item.get("material_description")
        if isinstance(item_material, Mapping):
            item_null_elements.update(evaluator_null_elements_from_material_payload({"material_description": item_material}))
        item_formula = (
            item.get("reduced_formula")
            or item.get("target_reduced_formula")
            or item.get("preferred_reduced_formula")
            or item.get("formula")
            or item.get("crystal_llm_formula")
        )
        item_null_elements.update(evaluator_null_elements_from_formula(item_formula))
        blocked_null_elements = sorted(item_null_elements.intersection(forbidden_null_elements))
        if blocked_null_elements:
            item_errors.append(
                f"sun_candidate_queue item {item_id!r} uses evaluator-null/high-risk elements "
                f"{blocked_null_elements}; remove that candidate from the ranked queue"
            )
        normalized_formula = queue_item_reduced_formula(item)
        if normalized_formula and normalized_formula in forbidden_formulas:
            item_errors.append(
                f"sun_candidate_queue item {item_id!r} reduced_formula {normalized_formula} is already failed/used "
                "(possibly from the hidden full failed/used set); remove it"
            )
        if normalized_formula:
            item_errors.extend(
                strategy_cooldown_errors_for_formula(
                    normalized_formula,
                    context=context,
                    label=f"sun_candidate_queue item {item_id!r}",
                )
            )
        if item_formula and abstract_formula_tokens_from_text(item_formula):
            item_errors.append(
                f"sun_candidate_queue item {item_id!r} formula {item_formula!r} is an abstract or coordination-fragment token"
            )
        item_template = str(item.get("generator_template") or "").strip()
        if not item_template and isinstance(item_material, Mapping):
            item_template = str(item_material.get("generator_template") or item_material.get("template_expectation") or "").strip()
        item_template = _queue_item_generator_template(item) or item_template
        item_basin_key = _queue_item_basin_key(item, item_template)
        if item_template in cooled_templates and search_mode != "orthogonal_jump":
            item_errors.append(
                f"sun_candidate_queue item {item_id!r} uses cooled template {item_template!r} "
                f"while current_search_mode={search_mode!r}; choose a non-cooled basin or return no_valid_material_description"
            )
        if item_basin_key in cooled_basin_keys and search_mode != "orthogonal_jump":
            item_errors.append(
                f"sun_candidate_queue item {item_id!r} uses cooled basin {item_basin_key!r} "
                f"while current_search_mode={search_mode!r}; choose a higher-yield basin"
            )
        if search_mode == "exploit" and exploit_template_allowlist and item_template not in exploit_template_allowlist:
            item_errors.append(
                f"sun_candidate_queue item {item_id!r} uses template {item_template!r}, but exploit mode requires "
                f"one of {sorted(exploit_template_allowlist)} unless all exploit basins are blocked"
            )
        if search_mode == "orthogonal_jump" and preferred_jump_templates and item_template not in preferred_jump_templates:
            item_errors.append(
                f"sun_candidate_queue item {item_id!r} uses template {item_template!r}, but orthogonal_jump mode "
                f"prefers underexplored templates {sorted(preferred_jump_templates)}"
            )
        boundary_matches: set[str] = set()
        if isinstance(item_material, Mapping):
            boundary_matches.update(template_volume_boundary_keys({"material_description": item_material}).intersection(forbidden_volume_boundaries))
        if item_template and normalized_formula:
            boundary_matches.update(template_volume_boundary_keys_from_formula(item_template, normalized_formula).intersection(forbidden_volume_boundaries))
        if boundary_matches:
            item_errors.append(
                f"sun_candidate_queue item {item_id!r} hits failed volume/template boundaries {sorted(boundary_matches)}"
            )
        if item_template in ALLOWED_TEMPLATES and isinstance(item_material, Mapping):
            role_mapping = item_material.get("generator_role_mapping")
            if not isinstance(role_mapping, Mapping):
                item_errors.append(f"sun_candidate_queue item {item_id!r} lacks generator_role_mapping for template {item_template}")
            else:
                missing_roles = [role for role in required_roles(item_template) if role not in role_mapping]
                if missing_roles:
                    item_errors.append(
                        f"sun_candidate_queue item {item_id!r} missing template roles {missing_roles} for {item_template}"
                    )
                else:
                    try:
                        charge = sum(
                            int(role_mapping[role].get("oxidation_state")) * int(count)
                            for role, count in TEMPLATE_ROLE_COUNTS[item_template].items()
                            if isinstance(role_mapping.get(role), Mapping)
                        )
                    except Exception:
                        charge = 999999
                    if charge != 0:
                        item_errors.append(
                            f"sun_candidate_queue item {item_id!r} is not charge-neutral under {item_template} role stoichiometry"
                        )
        if item_errors:
            errors.extend(item_errors)
        else:
            legal_item_ids.append(item_id)
    if len(legal_item_ids) < 2:
        errors.append(
            "sun_candidate_queue has fewer than two controller-legal candidates after sanitation "
            f"(legal={legal_item_ids}); provide a different route with at least two legal candidates or return no_valid_material_description"
        )
    selected_id = str(payload.get("selected_candidate_id") or payload.get("selected_material_id") or "").strip()
    if not selected_id:
        errors.append("X/Y output must include selected_candidate_id naming the queue item copied into material_description")
    ids = [str(item.get("candidate_id") or item.get("id") or "").strip() for item in queue]
    if selected_id and selected_id not in ids:
        errors.append(f"selected_candidate_id {selected_id!r} is not present in sun_candidate_queue ids {ids}")
    description = material_description_from_payload(payload)
    if not description:
        errors.append("X/Y selected material_description is missing or empty")
    selected_items = [item for item, item_id in zip(queue, ids) if item_id == selected_id]
    if selected_items:
        selected_item = selected_items[0]
        selected_material = selected_item.get("material_description")
        if not isinstance(selected_material, Mapping):
            selected_formula = _normalize_formula_text(
                selected_item.get("reduced_formula")
                or selected_item.get("formula")
                or selected_item.get("crystal_llm_formula")
                or ""
            )
            top_formula = _normalize_formula_text(template_formula_from_material_description(payload))
            if not (description and selected_formula and top_formula and selected_formula == top_formula):
                errors.append(f"selected queue item {selected_id!r} is missing material_description")
        elif description:
            selected_formula = template_formula_from_material_description({"material_description": selected_material})
            top_formula = template_formula_from_material_description(payload)
            if selected_formula and top_formula and selected_formula != top_formula:
                errors.append(
                    f"selected queue item formula {selected_formula} does not match top-level material_description formula {top_formula}"
                )
    if not str(payload.get("selection_rationale") or "").strip():
        errors.append("X/Y output must include selection_rationale explaining why the selected queue item is the best legal SUN bet")
    errors.extend(
        validate_xy_strategy_constraints(
            payload,
            context=context,
            forbidden_reduced_formulas=forbidden_formulas,
        )
    )
    return errors


def validate_xy_strategy_constraints(
    payload: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    forbidden_reduced_formulas: set[str] | None = None,
) -> list[str]:
    if material_payload_declares_no_valid_description(payload):
        return []
    controller_constraints = context.get("controller_constraints") if isinstance(context, Mapping) else None
    strategy_constraints = (
        controller_constraints.get("strategy_constraints")
        if isinstance(controller_constraints, Mapping) and isinstance(controller_constraints.get("strategy_constraints"), Mapping)
        else None
    )
    if not isinstance(strategy_constraints, Mapping):
        return []
    errors: list[str] = []
    queue = sun_candidate_queue_from_payload(payload)
    queue_formulas = [formula for formula in (queue_item_reduced_formula(item) for item in queue) if formula]
    selected_formula = selected_reduced_formula_from_payload(payload)
    forbidden_formulas = _context_failed_or_used_formulas(context)
    if forbidden_reduced_formulas:
        forbidden_formulas.update(
            formula
            for formula in (_normalize_formula_text(item) for item in forbidden_reduced_formulas if str(item).strip())
            if formula
        )
    for formula in queue_formulas:
        if formula in forbidden_formulas:
            errors.append(
                f"sun_candidate_queue formula {formula} is already failed/used in the full controller history; remove it"
            )
    if selected_formula and selected_formula in forbidden_formulas:
        errors.append(
            f"selected material reduced_formula {selected_formula} is already failed/used in the full controller history"
        )
    required_mode = str(strategy_constraints.get("required_acquisition_mode") or "").strip()
    if required_mode:
        for index, item in enumerate(queue, start=1):
            item_id = str(item.get("candidate_id") or item.get("id") or f"rank {index}").strip()
            item_mode = str(item.get("acquisition_mode") or "").strip()
            if item_mode != required_mode:
                errors.append(
                    f"sun_candidate_queue item {item_id!r} acquisition_mode={item_mode!r}; "
                    f"strategy_constraints requires acquisition_mode={required_mode!r}"
                )
    ordered = [
        formula
        for formula in (_normalize_formula_text(item) for item in strategy_constraints.get("legal_ordered_candidate_formulas", []))
        if formula
    ] if isinstance(strategy_constraints.get("legal_ordered_candidate_formulas"), list) else []
    order_enforced = bool(strategy_constraints.get("latest_strategy_order_enforced"))
    if not order_enforced or not ordered:
        return errors
    queue_size = xy_sun_candidate_queue_size_from_context(context)
    expected_visible = ordered[:queue_size]
    if len(ordered) < int(strategy_constraints.get("queue_min_legal_items") or 2):
        errors.append(
            "strategy_constraints latest route has fewer than two legal ordered formulas; "
            "return no_valid_material_description with an impossibility_certificate instead of selecting a singleton"
        )
    if queue_formulas:
        if queue_formulas[0] != ordered[0]:
            errors.append(
                f"sun_candidate_queue must start with first legal strategy formula {ordered[0]}, got {queue_formulas[0]}"
            )
        missing = [formula for formula in expected_visible if formula not in queue_formulas]
        if missing:
            errors.append(
                "sun_candidate_queue omitted legal strategy formula(s) "
                f"{missing}; include the visible legal_ordered_candidate_formulas before inventing lower-priority alternatives"
            )
    if selected_formula and selected_formula != ordered[0]:
        errors.append(
            f"selected material reduced_formula {selected_formula} violates strategy_constraints; "
            f"select first_required_formula {ordered[0]} unless binding feedback blocks it"
        )
    elif not selected_formula:
        errors.append(
            "selected material must expose reduced_formula/preferred_reduced_formula/target_reduced_formula "
            "so strategy_constraints can verify the selected first_required_formula"
        )
    return errors


def validate_template_only_material_description(
    payload: Mapping[str, Any],
    *,
    forbidden_reduced_formulas: set[str] | None = None,
    forbidden_volume_boundaries: set[str] | None = None,
    forbidden_evaluator_null_elements: set[str] | None = None,
) -> list[str]:
    description = material_description_from_payload(payload)
    errors: list[str] = []
    if not description:
        return ["template-only X/Y material_description is missing or not an object"]
    raw_template = description.get("generator_template")
    if raw_template in (None, "") and description.get("template_expectation") in ALLOWED_TEMPLATES:
        raw_template = description.get("template_expectation")
    template = str(raw_template or "").strip()
    if not template:
        errors.append("template-only material_description requires generator_template")
    elif template.startswith("custom_structure") or template in {"structure_dict", "custom"}:
        errors.append("template-only material_description forbids custom_structure/structure_dict templates")
    elif template not in ALLOWED_TEMPLATES:
        errors.append(
            f"template-only material_description generator_template {template!r} is not allowed; "
            f"allowed templates are {list(ALLOWED_TEMPLATES)}"
        )

    role_mapping = description.get("generator_role_mapping")
    if not isinstance(role_mapping, Mapping):
        role_mapping = description.get("role_mapping")
    if not isinstance(role_mapping, Mapping):
        errors.append("template-only material_description requires generator_role_mapping object")
    elif template in ALLOWED_TEMPLATES:
        required = required_roles(template)
        missing = [role for role in required if role not in role_mapping]
        if missing:
            errors.append(f"template-only generator_role_mapping for {template} is missing roles {missing}")
        oxidation_states: dict[str, int] = {}
        for role in required:
            raw_role = role_mapping.get(role)
            if not isinstance(raw_role, Mapping):
                errors.append(f"template-only generator_role_mapping.{role} must be an object with element and oxidation_state")
                continue
            element = str(raw_role.get("element") or "").strip()
            if not element:
                errors.append(f"template-only generator_role_mapping.{role}.element is required")
            else:
                try:
                    Element(element)
                except Exception:
                    errors.append(f"template-only generator_role_mapping.{role}.element {element!r} is not a valid element")
            try:
                oxidation_states[role] = int(raw_role.get("oxidation_state"))
            except Exception:
                errors.append(f"template-only generator_role_mapping.{role}.oxidation_state must be an integer")
        if all(role in oxidation_states for role in required):
            stoichiometry = TEMPLATE_ROLE_COUNTS[template]
            net_charge = sum(oxidation_states[role] * int(stoichiometry[role]) for role in required)
            if net_charge != 0:
                details = ", ".join(
                    f"{role}:{stoichiometry[role]}*{oxidation_states[role]}" for role in required
                )
                errors.append(
                    f"template-only generator_role_mapping for {template} is not charge-neutral "
                    f"(net_charge={net_charge}; {details})"
                )
            for role in required:
                ox = oxidation_states[role]
                if role == "X" and ox >= 0:
                    errors.append(f"template-only generator_role_mapping.{role}.oxidation_state must be negative")
                if role != "X" and ox <= 0:
                    errors.append(f"template-only generator_role_mapping.{role}.oxidation_state must be positive")

    proposed_formula = template_formula_from_material_description(payload)
    normalized_forbidden_formulas = {
        formula for formula in (_normalize_formula_text(item) for item in (forbidden_reduced_formulas or set())) if formula
    }
    if normalized_forbidden_formulas and proposed_formula and proposed_formula in normalized_forbidden_formulas:
        errors.append(
            f"template-only material_description proposed reduced_formula {proposed_formula}, "
            "which repeats a prior failed or used X/Y history formula"
        )
    boundary_matches = sorted(template_volume_boundary_keys(payload).intersection(forbidden_volume_boundaries or set()))
    if boundary_matches:
        errors.append(
            "template-only material_description repeats a prior volume_per_atom_too_large generator boundary "
            f"({boundary_matches[0]}); choose a materially different, more compact allowed-template route"
        )
    null_elements = sorted(
        evaluator_null_elements_from_material_payload(payload).intersection(
            forbidden_evaluator_null_elements or EVALUATOR_NULL_E_HULL_ELEMENTS
        )
    )
    if null_elements:
        errors.append(
            "template-only material_description uses evaluator-null/high-risk elements "
            f"{null_elements}; choose elements with prior evaluator e_hull support instead"
        )

    for key in ("expected_local_motif", "why_template_is_faithful"):
        if not str(description.get(key) or "").strip():
            errors.append(f"template-only material_description requires nonempty {key}")
    proxy_reason = template_proxy_rejection_reason(
        description.get("why_template_is_faithful"),
        description.get("expected_local_motif"),
        description.get("natural_language_description"),
        description.get("known_risks"),
    )
    if proxy_reason:
        errors.append(
            f"template-only material_description {proxy_reason}; "
            "choose a template whose topology/coordination is the intended local motif"
        )
    return errors


def template_volume_boundary_key(payload: Mapping[str, Any]) -> str:
    keys = sorted(template_volume_boundary_keys(payload))
    return keys[0] if keys else ""


def _volume_boundary_keys_from_role_elements(template: str, role_elements: Mapping[str, str]) -> set[str]:
    x_element = str(role_elements.get("X") or "").strip()
    if template not in ALLOWED_TEMPLATES or not x_element:
        return set()
    keys: set[str] = set()
    for role, element in role_elements.items():
        element_text = str(element or "").strip()
        if role == "X" or not element_text:
            continue
        keys.add(f"template={template};{role}={element_text};X={x_element};failure=volume_per_atom_too_large")
    for element in sorted({str(item or "").strip() for item in role_elements.values()}):
        if element in COMMON_ALKALI_ELEMENTS:
            keys.add(f"template={template};contains={element};X={x_element};failure=volume_per_atom_too_large")
    return keys


def template_volume_boundary_keys(payload: Mapping[str, Any]) -> set[str]:
    description = material_description_from_payload(payload)
    raw_template = description.get("generator_template")
    if raw_template in (None, "") and description.get("template_expectation") in ALLOWED_TEMPLATES:
        raw_template = description.get("template_expectation")
    template = str(raw_template or "").strip()
    role_mapping = description.get("generator_role_mapping")
    if not isinstance(role_mapping, Mapping):
        role_mapping = description.get("role_mapping")
    if template not in ALLOWED_TEMPLATES or not isinstance(role_mapping, Mapping):
        return set()
    raw_a = role_mapping.get("A")
    raw_x = role_mapping.get("X")
    if not isinstance(raw_x, Mapping):
        return set()
    x_element = str(raw_x.get("element") or "").strip()
    if not x_element:
        return set()
    role_elements = {
        str(role): str(raw_role.get("element") or "").strip()
        for role, raw_role in role_mapping.items()
        if isinstance(raw_role, Mapping) and str(raw_role.get("element") or "").strip()
    }
    role_elements["X"] = x_element
    return _volume_boundary_keys_from_role_elements(template, role_elements)


def _formula_amounts(formula_text: str) -> dict[str, int]:
    try:
        amounts = Composition(formula_text).get_el_amt_dict()
    except Exception:
        return {}
    normalized: dict[str, int] = {}
    for element, raw_amount in amounts.items():
        try:
            amount = float(raw_amount)
        except Exception:
            return {}
        rounded = int(round(amount))
        if rounded <= 0 or abs(amount - rounded) > 1e-6:
            return {}
        normalized[str(element)] = rounded
    return normalized


def _likely_x_elements_for_formula(template: str, amounts: Mapping[str, int]) -> list[str]:
    role_counts = TEMPLATE_ROLE_COUNTS.get(template)
    if not role_counts or "X" not in role_counts:
        return []
    x_count = int(role_counts["X"])
    exact = [element for element, amount in amounts.items() if int(amount) == x_count]
    exact_anions = [element for element in exact if element in COMMON_ANION_ELEMENTS]
    if exact_anions:
        return sorted(exact_anions)
    if exact:
        return sorted(exact)
    anions = [element for element in amounts if element in COMMON_ANION_ELEMENTS]
    if anions:
        max_amount = max(int(amounts[element]) for element in anions)
        return sorted(element for element in anions if int(amounts[element]) == max_amount)
    if amounts:
        max_amount = max(int(amount) for amount in amounts.values())
        return sorted(element for element, amount in amounts.items() if int(amount) == max_amount)
    return []


def template_volume_boundary_keys_from_formula(template: str, formula_text: str) -> set[str]:
    if template not in ALLOWED_TEMPLATES:
        return set()
    amounts = _formula_amounts(formula_text)
    if not amounts:
        return set()
    role_counts = TEMPLATE_ROLE_COUNTS[template]
    count_to_roles: dict[int, list[str]] = {}
    for role, count in role_counts.items():
        count_to_roles.setdefault(int(count), []).append(str(role))
    keys: set[str] = set()
    for x_element in _likely_x_elements_for_formula(template, amounts):
        role_elements: dict[str, str] = {"X": x_element}
        for role, count in role_counts.items():
            role = str(role)
            if role == "X":
                continue
            matching = [
                element
                for element, amount in amounts.items()
                if element != x_element and int(amount) == int(count)
            ]
            if len(count_to_roles.get(int(count), [])) == 1 and len(matching) == 1:
                role_elements[role] = matching[0]
        if "A" in role_counts and "A" not in role_elements:
            alkali_candidates = [
                element
                for element in amounts
                if element != x_element and element in COMMON_ALKALI_ELEMENTS
            ]
            if len(alkali_candidates) == 1:
                role_elements["A"] = alkali_candidates[0]
            else:
                non_x = [element for element in amounts if element != x_element]
                if len(non_x) == 1:
                    role_elements["A"] = non_x[0]
        keys.update(_volume_boundary_keys_from_role_elements(template, role_elements))
    return keys


def sequential_description_agrees(review: Mapping[str, Any], proposal: Mapping[str, Any]) -> bool:
    return bool(review.get("agree")) is True and bool(review.get("approved", review.get("agree"))) is True and bool(
        material_description_from_payload(proposal)
    )


def sequential_no_valid_description_agrees(
    review: Mapping[str, Any],
    proposal: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> bool:
    return sequential_no_valid_description_agrees_with_context(review, proposal, context)


def _jsonish_text_blob(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, str):
            parts.append(value)
            continue
        try:
            parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        except Exception:
            parts.append(str(value))
    return "\n".join(parts)


def missing_no_valid_strategy_formula_audits(
    proposal: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> list[str]:
    if not material_payload_declares_no_valid_description(proposal):
        return []
    latest = context.get("latest_xy_strategy_constraints") if isinstance(context, Mapping) else None
    if not isinstance(latest, Mapping):
        return []
    raw_required = latest.get("next_strategy_candidate_formulas")
    if isinstance(raw_required, list):
        required = [
            formula
            for formula in (_normalize_formula_text(item) for item in raw_required)
            if formula
        ]
    else:
        required = strategy_formula_candidates_from_text(latest.get("next_strategy"))
    if not required:
        return []
    certificate_text = _jsonish_text_blob(
        proposal.get("impossibility_certificate"),
        proposal.get("proposal_summary"),
        proposal.get("debate_summary"),
        proposal.get("overall_reasoning_summary"),
        proposal.get("strategy_update_from_history"),
    )
    mentioned = set(strategy_formula_candidates_from_text(certificate_text))
    missing: list[str] = []
    seen: set[str] = set()
    for formula in required:
        if formula in seen:
            continue
        seen.add(formula)
        if formula not in mentioned:
            missing.append(formula)
    return missing


def sequential_no_valid_description_agrees_with_context(
    review: Mapping[str, Any],
    proposal: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> bool:
    if latest_strategy_search_policy_supersedes(context):
        return False
    if missing_no_valid_strategy_formula_audits(proposal, context):
        return False
    return (
        bool(review.get("agree")) is True
        and bool(review.get("approved", review.get("agree"))) is True
        and material_payload_declares_no_valid_description(proposal)
    )


def enforce_no_valid_strategy_audit_guard(
    review: Mapping[str, Any],
    proposal: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    guarded = dict(review)
    if bool(guarded.get("agree")) is not True:
        return guarded
    if bool(guarded.get("approved", guarded.get("agree"))) is not True:
        return guarded
    missing = missing_no_valid_strategy_formula_audits(proposal, context)
    if not missing:
        if not material_payload_declares_no_valid_description(proposal) or not latest_strategy_search_policy_supersedes(context):
            return guarded
        guarded["agree"] = False
        guarded["approved"] = False
        guarded["required_revision"] = (
            "no_valid_material_description only audits stale latest_xy_strategy_constraints, but "
            "controller_constraints.search_policy supersedes that route for this iteration; propose or approve "
            "a current-search-policy queue with at least two legal non-duplicate candidates instead."
        )
        guarded["controller_no_valid_audit_errors"] = [
            "search_policy supersedes stale latest strategy; stale-route no_valid cannot terminate this iteration"
        ]
        existing_summary = str(guarded.get("overall_reasoning_summary") or "").strip()
        guard_summary = "Controller rejected no-valid approval because search_policy supersedes the stale route."
        guarded["overall_reasoning_summary"] = (
            f"{guard_summary} {existing_summary}".strip() if existing_summary else guard_summary
        )
        return guarded
    missing_text = ", ".join(missing)
    guarded["agree"] = False
    guarded["approved"] = False
    guarded["required_revision"] = (
        "no_valid_material_description is incomplete: impossibility_certificate must audit every "
        f"formula named by latest_xy_strategy_constraints.next_strategy_candidate_formulas; missing {missing_text}."
    )
    guarded["controller_no_valid_audit_errors"] = [
        f"missing no-valid audit for next_strategy candidate {formula}" for formula in missing
    ]
    existing_summary = str(guarded.get("overall_reasoning_summary") or "").strip()
    guard_summary = (
        f"Controller rejected no-valid approval because the impossibility_certificate omitted {missing_text}."
    )
    guarded["overall_reasoning_summary"] = (
        f"{guard_summary} {existing_summary}".strip() if existing_summary else guard_summary
    )
    return guarded


def exhausted_latest_strategy_requires_new_route_audit(context: Mapping[str, Any] | None) -> bool:
    latest = context.get("latest_xy_strategy_constraints") if isinstance(context, Mapping) else None
    if not isinstance(latest, Mapping):
        return False
    candidate_formulas = latest.get("next_strategy_candidate_formulas")
    if isinstance(candidate_formulas, list) and candidate_formulas:
        return False
    text = str(latest.get("next_strategy") or "").lower()
    return (
        "route as exhausted" in text
        or "route exhausted" in text
        or "candidate list is exhausted" in text
        or "named candidate list exhausted" in text
    )


def proposal_new_route_candidate_formulas(
    proposal: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> list[str]:
    text = _jsonish_text_blob(
        proposal.get("proposal_summary"),
        proposal.get("debate_summary"),
        proposal.get("overall_reasoning_summary"),
        proposal.get("strategy_update_from_history"),
        proposal.get("material_description"),
    )
    formulas = strategy_formula_candidates_from_text(text)
    template_formula = template_formula_from_material_description(proposal)
    if template_formula:
        formulas.append(template_formula)
    for item in sun_candidate_queue_from_payload(proposal):
        if not isinstance(item, Mapping):
            continue
        raw_formula = item.get("reduced_formula") or item.get("formula") or item.get("crystal_llm_formula")
        if raw_formula:
            formulas.append(str(raw_formula))
        item_material = item.get("material_description")
        if isinstance(item_material, Mapping):
            item_formula = template_formula_from_material_description({"material_description": item_material})
            if item_formula:
                formulas.append(item_formula)

    constraints = context.get("controller_constraints") if isinstance(context, Mapping) else None
    raw_forbidden = constraints.get("failed_or_used_reduced_formulas") if isinstance(constraints, Mapping) else []
    forbidden = {
        formula
        for formula in (_normalize_formula_text(item) for item in raw_forbidden if str(item).strip())
        if formula
    } if isinstance(raw_forbidden, list) else set()
    result: list[str] = []
    seen: set[str] = set()
    for formula in formulas:
        normalized = _normalize_formula_text(formula)
        if not normalized or normalized in seen or normalized in forbidden:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def enforce_exhausted_strategy_route_audit_guard(
    review: Mapping[str, Any],
    proposal: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    guarded = dict(review)
    if bool(guarded.get("agree")) is not True:
        return guarded
    if bool(guarded.get("approved", guarded.get("agree"))) is not True:
        return guarded
    if material_payload_declares_no_valid_description(proposal):
        return guarded
    if not exhausted_latest_strategy_requires_new_route_audit(context):
        return guarded
    formulas = proposal_new_route_candidate_formulas(proposal, context)
    if len(formulas) >= 2:
        return guarded
    formula_text = ", ".join(formulas) if formulas else "none"
    guarded["agree"] = False
    guarded["approved"] = False
    guarded["required_revision"] = (
        "latest_xy_strategy_constraints marks the previous route exhausted and names no valid next candidates. "
        "Before approving a new single material, the proposal must explicitly audit at least two concrete, "
        f"non-duplicate formulas in the new route; currently found {formula_text}."
    )
    guarded["controller_exhausted_strategy_audit_errors"] = [
        "exhausted latest strategy requires at least two concrete non-duplicate formulas audited before selecting one"
    ]
    existing_summary = str(guarded.get("overall_reasoning_summary") or "").strip()
    guard_summary = "Controller rejected approval because the exhausted-strategy replacement route audited fewer than two formulas."
    guarded["overall_reasoning_summary"] = (
        f"{guard_summary} {existing_summary}".strip() if existing_summary else guard_summary
    )
    return guarded


def latest_strategy_candidate_formulas(context: Mapping[str, Any] | None) -> list[str]:
    latest = context.get("latest_xy_strategy_constraints") if isinstance(context, Mapping) else None
    if not isinstance(latest, Mapping):
        return []
    raw_candidates = latest.get("next_strategy_candidate_formulas")
    if isinstance(raw_candidates, list):
        formulas = [_normalize_formula_text(item) for item in raw_candidates]
    else:
        formulas = strategy_formula_candidates_from_text(latest.get("next_strategy"))
    result: list[str] = []
    seen: set[str] = set()
    for formula in formulas:
        normalized = _normalize_formula_text(formula)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def latest_strategy_failure_boundary_formulas(context: Mapping[str, Any] | None) -> set[str]:
    latest = context.get("latest_xy_strategy_constraints") if isinstance(context, Mapping) else None
    if not isinstance(latest, Mapping):
        return set()
    raw_boundaries = latest.get("failure_boundaries")
    texts = raw_boundaries if isinstance(raw_boundaries, list) else [raw_boundaries]
    formulas: set[str] = set()
    for text in texts:
        formulas.update(formula_mentions_from_text(text))
    return {
        formula
        for formula in (_normalize_formula_text(item) for item in formulas if str(item).strip())
        if formula
    }


def _context_failed_or_used_formulas(context: Mapping[str, Any] | None) -> set[str]:
    constraints = context.get("controller_constraints") if isinstance(context, Mapping) else None
    raw_forbidden = constraints.get("failed_or_used_reduced_formulas") if isinstance(constraints, Mapping) else []
    if not isinstance(raw_forbidden, list):
        return set()
    return {
        formula
        for formula in (_normalize_formula_text(item) for item in raw_forbidden if str(item).strip())
        if formula
    }


def _context_failed_volume_boundaries(context: Mapping[str, Any] | None) -> set[str]:
    constraints = context.get("controller_constraints") if isinstance(context, Mapping) else None
    raw_boundaries = constraints.get("failed_volume_template_boundaries") if isinstance(constraints, Mapping) else []
    if not isinstance(raw_boundaries, list):
        return set()
    return {str(item).strip() for item in raw_boundaries if str(item).strip()}


def _context_forbidden_evaluator_null_elements(context: Mapping[str, Any] | None) -> set[str]:
    constraints = context.get("controller_constraints") if isinstance(context, Mapping) else None
    raw_forbidden = constraints.get("forbidden_evaluator_null_elements") if isinstance(constraints, Mapping) else None
    if isinstance(raw_forbidden, list):
        return {
            str(element).strip()
            for element in raw_forbidden
            if str(element).strip()
        }
    return set(EVALUATOR_NULL_E_HULL_ELEMENTS)


def _context_strategy_cooldown_sets(context: Mapping[str, Any] | None) -> tuple[set[str], set[str]]:
    constraints = context.get("controller_constraints") if isinstance(context, Mapping) else None
    cooldowns = constraints.get("strategy_cooldowns") if isinstance(constraints, Mapping) else None
    if not isinstance(cooldowns, Mapping):
        return set(), set()
    raw_systems = cooldowns.get("blocked_chemical_systems")
    raw_patterns = cooldowns.get("blocked_family_patterns")
    blocked_systems = {
        str(item.get("chemical_system") or "")
        for item in raw_systems or []
        if isinstance(item, Mapping) and item.get("chemical_system")
    }
    blocked_patterns = {
        str(item.get("family_pattern") or "")
        for item in raw_patterns or []
        if isinstance(item, Mapping) and item.get("family_pattern")
    }
    return blocked_systems, blocked_patterns


def strategy_cooldown_errors_for_formula(
    formula: Any,
    *,
    context: Mapping[str, Any] | None,
    label: str,
) -> list[str]:
    normalized = _normalize_formula_text(formula)
    if not normalized:
        return []
    system = _chemical_system_from_formula(normalized)
    if not system:
        return []
    pattern = _chemical_system_family_pattern(system)
    blocked_systems, blocked_patterns = _context_strategy_cooldown_sets(context)
    errors: list[str] = []
    if system in blocked_systems:
        errors.append(
            f"{label} reduced_formula {normalized} has chemical_system={system}, "
            "which is under strategy_cooldowns.blocked_chemical_systems"
        )
    if pattern and pattern in blocked_patterns:
        errors.append(
            f"{label} reduced_formula {normalized} has family_pattern={pattern}, "
            "which is under strategy_cooldowns.blocked_family_patterns"
        )
    return errors


def _collect_error_strings(value: Any, *, active: bool = False) -> list[str]:
    strings: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child_active = active or key_text in {
                "generator_errors",
                "materialization_errors",
                "template_only_errors",
                "last_generator_repair_feedback",
            }
            strings.extend(_collect_error_strings(item, active=child_active))
        return strings
    if isinstance(value, list):
        for item in value:
            strings.extend(_collect_error_strings(item, active=active))
        return strings
    if active and isinstance(value, str):
        strings.append(value)
    return strings


def blocked_formulas_from_return_feedback(return_feedback: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(return_feedback, Mapping):
        return set()
    formulas: set[str] = set()
    for text in _collect_error_strings(return_feedback):
        formulas.update(strategy_formula_candidates_from_text(text))
    return {formula for formula in (_normalize_formula_text(item) for item in formulas) if formula}


def latest_strategy_selection_order_errors(
    proposal: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
    return_feedback: Mapping[str, Any] | None = None,
) -> list[str]:
    if material_payload_declares_no_valid_description(proposal):
        return []
    if latest_strategy_search_policy_supersedes(context):
        return []
    ordered_formulas = latest_strategy_candidate_formulas(context)
    if len(ordered_formulas) < 2:
        return []
    proposed_formula = selected_reduced_formula_from_payload(proposal)
    if not proposed_formula:
        return []
    viable, _blocked_reasons = latest_strategy_viable_formula_queue(context, return_feedback)
    if return_feedback and len(viable) < 2:
        viable_text = ", ".join(viable) if viable else "none"
        blocked_text = ", ".join(formula for formula in ordered_formulas if formula not in viable)
        return [
            "latest ordered route has fewer than two currently viable formulas after binding feedback "
            f"(viable={viable_text}; blocked={blocked_text}); return no_valid_material_description instead of selecting a singleton"
        ]
    if not viable:
        return []
    expected = viable[0]
    if proposed_formula not in ordered_formulas:
        return [
            f"proposed reduced_formula {proposed_formula}, but latest next_strategy_candidate_formulas "
            f"requires the ordered route {', '.join(ordered_formulas)} and first viable formula {expected}"
        ]
    if proposed_formula != expected:
        return [
            f"proposed reduced_formula {proposed_formula}, but latest next_strategy_candidate_formulas is an ordered queue; "
            f"select {expected} first and use later formulas only after earlier candidates are blocked by binding feedback"
        ]
    return []


def enforce_latest_strategy_selection_order_guard(
    review: Mapping[str, Any],
    proposal: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
    return_feedback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    guarded = dict(review)
    if bool(guarded.get("agree")) is not True:
        return guarded
    if bool(guarded.get("approved", guarded.get("agree"))) is not True:
        return guarded
    errors = latest_strategy_selection_order_errors(proposal, context, return_feedback)
    if not errors:
        return guarded
    guarded["agree"] = False
    guarded["approved"] = False
    guarded["required_revision"] = errors[0]
    guarded["controller_strategy_order_errors"] = errors
    existing_summary = str(guarded.get("overall_reasoning_summary") or "").strip()
    guard_summary = "Controller rejected approval because the proposal violated the ordered next_strategy candidate queue."
    guarded["overall_reasoning_summary"] = (
        f"{guard_summary} {existing_summary}".strip() if existing_summary else guard_summary
    )
    return guarded


def formulas_from_iteration_record(record: Mapping[str, Any]) -> set[str]:
    formulas: set[str] = set()
    for source_key in ("evaluation_result", "selected_record", "executable_generator_rule", "candidate_spec"):
        source = record.get(source_key)
        if not isinstance(source, Mapping):
            continue
        for key in ("formula", "reduced_formula"):
            formula = _normalize_formula_text(source.get(key))
            if formula:
                formulas.add(formula)
    for key in ("formula", "reduced_formula"):
        formula = _normalize_formula_text(record.get(key))
        if formula:
            formulas.add(formula)
    for source_key in ("selected_records", "evaluation_results"):
        values = record.get(source_key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, Mapping):
                continue
            for key in ("formula", "reduced_formula"):
                formula = _normalize_formula_text(item.get(key))
                if formula:
                    formulas.add(formula)
    batch_result = record.get("evaluation_results")
    if isinstance(batch_result, Mapping):
        best_rows = batch_result.get("best_rows")
        if isinstance(best_rows, list):
            for item in best_rows:
                if not isinstance(item, Mapping):
                    continue
                for key in ("formula", "reduced_formula"):
                    formula = _normalize_formula_text(item.get(key))
                    if formula:
                        formulas.add(formula)
    return formulas


def postmortem_blocked_next_strategy_formulas(
    postmortem: Mapping[str, Any],
    forbidden_reduced_formulas: set[str],
) -> list[str]:
    forbidden = {
        formula for formula in (_normalize_formula_text(item) for item in forbidden_reduced_formulas) if formula
    }
    boundary_forbidden = {
        formula
        for formula in (
            _normalize_formula_text(item)
            for item in formula_mentions_from_text(postmortem.get("failure_boundaries"))
        )
        if formula
    }
    if not forbidden:
        forbidden = set()
    seen: set[str] = set()
    blocked: list[str] = []
    for formula in formula_mentions_from_text(postmortem.get("next_strategy")):
        if (formula in forbidden or formula in boundary_forbidden) and formula not in seen:
            seen.add(formula)
            blocked.append(formula)
    return blocked


def enforce_postmortem_next_strategy_guard(
    postmortem: Mapping[str, Any],
    *,
    forbidden_reduced_formulas: set[str],
) -> dict[str, Any]:
    guarded = dict(postmortem)
    blocked = postmortem_blocked_next_strategy_formulas(guarded, forbidden_reduced_formulas)
    original_next_strategy = str(guarded.get("next_strategy") or "")
    abstract_tokens = abstract_formula_tokens_from_text(original_next_strategy)
    evaluator_null_tokens: list[str] = []
    allowed_mentions: list[str] = []
    seen_allowed: set[str] = set()
    forbidden = {
        formula for formula in (_normalize_formula_text(item) for item in forbidden_reduced_formulas) if formula
    }
    boundary_forbidden = {
        formula
        for formula in (
            _normalize_formula_text(item)
            for item in formula_mentions_from_text(guarded.get("failure_boundaries"))
        )
        if formula
    }
    for token in strategy_formula_candidate_tokens_from_text(original_next_strategy):
        formula = _normalize_formula_text(token)
        if formula in forbidden or formula in boundary_forbidden:
            continue
        if evaluator_null_elements_from_formula(formula):
            evaluator_null_tokens.append(token)
            continue
        if formula and formula not in seen_allowed:
            seen_allowed.add(formula)
            allowed_mentions.append(token)
    has_candidate_segment = STRATEGY_CANDIDATE_SEGMENT_RE.search(original_next_strategy) is not None
    if not blocked and not abstract_tokens and not evaluator_null_tokens:
        return guarded
    if not blocked and not evaluator_null_tokens and abstract_tokens and not allowed_mentions and not has_candidate_segment:
        return guarded

    blocked_text = ", ".join(blocked)
    abstract_text = ", ".join(abstract_tokens)
    evaluator_null_text = ", ".join(evaluator_null_tokens)
    singleton_allowed_mentions = list(allowed_mentions) if len(allowed_mentions) == 1 else []
    if singleton_allowed_mentions:
        allowed_mentions = []
    removed_items = blocked + abstract_tokens + evaluator_null_tokens + singleton_allowed_mentions
    removed_text = ", ".join(removed_items)
    errors = [
        f"next_strategy named prior failed/used or failure-boundary reduced_formula {formula}; removed from binding strategy"
        for formula in blocked
    ]
    errors.extend(
        f"next_strategy named abstract/non-concrete formula token {token}; removed from binding strategy"
        for token in abstract_tokens
    )
    errors.extend(
        f"next_strategy named evaluator-null/high-risk element formula {token}; removed from binding strategy"
        for token in evaluator_null_tokens
    )
    if singleton_allowed_mentions:
        singleton_text = ", ".join(singleton_allowed_mentions)
        errors.append(
            "next_strategy retained fewer than two non-duplicate concrete candidate formulas after guard "
            f"({singleton_text}); route marked exhausted instead of writing a singleton fallback"
        )
    existing_errors = guarded.get("controller_postmortem_audit_errors")
    if isinstance(existing_errors, list):
        errors = [str(item) for item in existing_errors] + errors
    guarded["controller_postmortem_audit_errors"] = errors
    guarded["controller_original_next_strategy"] = short_text(original_next_strategy, 3200)

    if allowed_mentions:
        allowed_text = ", ".join(allowed_mentions)
        guarded["next_strategy"] = (
            "Controller-sanitized next_strategy: X/Y's postmortem named already failed/used, abstract, "
            "or evaluator-null/high-risk formulas, so those entries are removed and are not future candidates. "
            f"Continue only with this ordered non-duplicate formula set from the same route/fallbacks: {allowed_text}. "
            "Before approving any next material, reconstruct a complete generator_template and role mapping for the chosen "
            "formula, require the template to faithfully realize the motif, and require at least two formulas from the "
            "active route to clear failed_or_used_reduced_formulas plus failed_volume_template_boundaries. "
            "If fewer than two formulas clear, mark that route exhausted instead of preserving blocked formulas as fallbacks."
        )
    else:
        guarded["next_strategy"] = (
            "Controller-sanitized next_strategy: X/Y's postmortem named only formulas already failed/used, abstract, "
            "or evaluator-null/high-risk, so the named candidate list is exhausted. This does not by itself prove "
            "the validated mechanism basin is physically exhausted. X/Y may stay in that basin only if they audit "
            "at least two new concrete non-duplicate formulas outside every failure_boundary; otherwise name a "
            "materially different route with at least two pre-audited non-duplicate formulas or return "
            "no_valid_material_description with a complete audit of the blocked strategy."
        )

    boundaries = guarded.get("failure_boundaries")
    if not isinstance(boundaries, list):
        boundaries = []
    else:
        boundaries = list(boundaries)
    boundaries.append(
        "Controller postmortem guard: do not use removed next_strategy entries "
        f"({removed_text}) unless controller_constraints.control_candidate_requested=true and the entry is a concrete formula."
    )
    guarded["failure_boundaries"] = boundaries

    existing_summary = str(guarded.get("review_summary") or guarded.get("overall_reasoning_summary") or "").strip()
    guard_parts = []
    if blocked_text:
        guard_parts.append(f"duplicate formulas: {blocked_text}")
    if abstract_text:
        guard_parts.append(f"abstract formula tokens: {abstract_text}")
    if evaluator_null_text:
        guard_parts.append(f"evaluator-null/high-risk formulas: {evaluator_null_text}")
    guard_summary = f"Controller removed {'; '.join(guard_parts)} from next_strategy."
    guarded["review_summary"] = f"{guard_summary} {existing_summary}".strip() if existing_summary else guard_summary
    return guarded


def no_valid_material_consensus_from_payload(
    proposal: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    iteration: int,
) -> dict[str, Any]:
    certificate = proposal.get("impossibility_certificate")
    if not isinstance(certificate, Mapping):
        certificate = {
            "summary": short_text(
                proposal.get("proposal_summary")
                or proposal.get("debate_summary")
                or proposal.get("overall_reasoning_summary")
                or "X/Y certified that no valid material exists under the current constraints.",
                1200,
            )
        }
    return {
        "status": "no_valid_material_consensus",
        "agent": "Y",
        "iteration": iteration,
        "material_description": {},
        "impossibility_certificate": dict(certificate),
        "xy_history_lessons_used": proposal.get("xy_history_lessons_used", []),
        "debate_summary": short_text(
            review.get("overall_reasoning_summary")
            or proposal.get("proposal_summary")
            or proposal.get("debate_summary")
            or "X/Y agreed that no legal one-material candidate can be sent to Z/W.",
            1200,
        ),
    }


def sequential_candidate_agrees(review: Mapping[str, Any], proposal: Mapping[str, Any]) -> bool:
    if bool(review.get("return_to_xy")) is True:
        return False
    if bool(review.get("agree")) is not True:
        return False
    approved = review.get("approved_candidate_ids")
    if isinstance(approved, list) and not approved:
        return False
    return len(candidate_specs_from_payload(proposal)) == 1


def compact_failure_boundary_for_strategy(item: Any) -> Any:
    if isinstance(item, Mapping):
        compact: dict[str, Any] = {}
        for key in ("boundary", "formula", "family", "template", "role_mapping", "axis", "lesson", "reason", "evidence"):
            value = item.get(key)
            if value in (None, "", [], {}):
                continue
            compact[key] = short_text(value, 260) if isinstance(value, str) else value
        return compact or short_text(item, 260)
    return short_text(item, 260)


def stale_mattergen_backend_gate_text(value: Any) -> bool:
    text = str(value or "").lower()
    normalized = re.sub(r"[\s_\-]+", " ", text)
    if any(
        (
            "backend gate" in normalized and ("unrepaired" in normalized or "repair" in normalized),
            "backend remains unrepaired" in normalized,
            "backend is unrepaired" in normalized,
            "until backend repair" in normalized,
        )
    ):
        return True
    if "mattergen" not in normalized:
        return False
    return any(
        (
            "backend repair" in normalized,
            "repair confirmation" in normalized,
            "repair/cuda" in normalized,
            "cuda compatibility" in normalized and ("repair" in normalized or "confirmed" in normalized),
            "no kernel" in normalized,
            "nokernel" in normalized,
            "kernel image" in normalized,
            "do not dispatch" in normalized and ("repair" in normalized or "confirmed" in normalized),
            "dispatch nothing" in normalized and ("repair" in normalized or "confirmed" in normalized),
            "not explicitly confirmed" in normalized and ("repair" in normalized or "backend" in normalized),
        )
    )


def _strip_stale_mattergen_backend_gate_sentences(text: str) -> tuple[str, bool]:
    if not text:
        return "", False
    chunks = re.split(r"(?<=[.!?;])\s+|\n+", text)
    kept: list[str] = []
    removed = False
    for chunk in chunks:
        stripped = chunk.strip()
        if not stripped:
            continue
        if stale_mattergen_backend_gate_text(stripped):
            removed = True
            continue
        stripped = re.sub(r"^(if|once|after)\s+repaired,?\s+", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"^after\s+repair,?\s+", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"^after\s+backend\s+repair,?\s+", "", stripped, flags=re.IGNORECASE)
        if stripped:
            kept.append(stripped)
    sanitized = re.sub(r"\s+", " ", " ".join(kept)).strip()
    return sanitized, removed


def _clean_residual_mattergen_repair_qualifiers(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _clean_residual_mattergen_repair_qualifiers(item) if isinstance(item, (str, Mapping, list)) else item
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clean_residual_mattergen_repair_qualifiers(item) for item in value]
    if not isinstance(value, str):
        return value
    cleaned = re.sub(
        r"\bafter\s+(backend\s+)?repair\s+use\s+mattergen",
        "use MatterGen",
        value,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bonly\s+after\s+(backend\s+)?repair\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(after|once)\s+(backend\s+)?repair(?:ed)?[:,]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\buse\s+mattergen\b", "use MatterGen", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def mattergen_record_has_success(record: Mapping[str, Any]) -> bool:
    result = record.get("evaluation_result")
    if not isinstance(result, Mapping):
        return False
    if str(record.get("status") or "").strip().lower() != "evaluated" and not result.get("formula"):
        return False
    evidence_text = _jsonish_text_blob(
        record.get("candidate_spec"),
        record.get("candidate_consensus"),
        record.get("executable_generator_rule"),
        record.get("selected_record"),
        record.get("material_description"),
        record.get("materialization_errors"),
    ).lower()
    return "mattergen" in evidence_text


def mattergen_operational_status_from_memory(
    memory: Mapping[str, Any],
    *,
    generator_backend: str = DEFAULT_GENERATOR_BACKEND,
) -> dict[str, Any]:
    status = {
        "backend": generator_backend,
        "verified": False,
        "evidence": "",
        "policy": "verified supersedes stale backend gates.",
    }
    if str(generator_backend).strip().lower() != "mattergen":
        return status
    records = memory.get("records")
    if not isinstance(records, list):
        return status
    for record in reversed(records):
        if not isinstance(record, Mapping) or not mattergen_record_has_success(record):
            continue
        result = record.get("evaluation_result") if isinstance(record.get("evaluation_result"), Mapping) else {}
        formula = result.get("formula") or result.get("reduced_formula")
        e_hull = result.get("e_hull")
        status.update(
            {
                "verified": True,
                "success_formula": formula,
                "success_e_hull": e_hull,
                "evidence": "prior_mattergen_evaluated_success",
            }
        )
        return status
    return status


def _sanitize_stale_mattergen_backend_gate_strategy(
    postmortem: Mapping[str, Any],
    *,
    mattergen_backend_verified: bool,
) -> dict[str, Any]:
    sanitized = dict(postmortem)
    if not mattergen_backend_verified:
        return sanitized

    audit_errors = list(sanitized.get("controller_postmortem_audit_errors") or [])
    raw_next_strategy = str(sanitized.get("next_strategy") or "")
    cleaned_strategy, removed_strategy_gate = _strip_stale_mattergen_backend_gate_sentences(raw_next_strategy)
    if not removed_strategy_gate:
        cleaned_strategy = raw_next_strategy
    formulas = strategy_formula_candidates_from_text(raw_next_strategy)
    cleaned_strategy = re.sub(
        r"\bbackend-gated\s+(ranked\s+)?",
        r"\1",
        cleaned_strategy,
        flags=re.IGNORECASE,
    ).strip()
    cleaned_strategy = re.sub(r"\bstage\s+\d+\s+", "", cleaned_strategy, flags=re.IGNORECASE)
    cleaned_strategy = _clean_residual_mattergen_repair_qualifiers(cleaned_strategy)
    cleaned_strategy = re.sub(r"^\s*use\s+mattergen", "Use MatterGen", cleaned_strategy, flags=re.IGNORECASE)
    cleaned_strategy = re.sub(
        r"^\s*ranked\s+strict-sun\s+acquisition\s+queue\.?\s*",
        "Use this strict-SUN acquisition queue. ",
        cleaned_strategy,
        flags=re.IGNORECASE,
    )
    cleaned_strategy = re.sub(
        r"^\s*ranked\s+",
        "Use this ranked ",
        cleaned_strategy,
        flags=re.IGNORECASE,
    )
    strategy_changed = cleaned_strategy != raw_next_strategy
    if strategy_changed:
        if not cleaned_strategy and formulas:
            cleaned_strategy = (
                "Continue with the non-duplicate MatterGen chemical-system queue after controller audits: "
                + ", ".join(formulas)
                + "."
            )
        prefix = (
            "Controller override: stale MatterGen backend repair/CUDA gate is superseded by prior successful "
            "MatterGen materialization/evaluation; dispatch legal MatterGen requests normally. "
        )
        sanitized["next_strategy"] = short_text(prefix + cleaned_strategy, 3200)
        sanitized["mattergen_backend_gate_superseded"] = True
        if removed_strategy_gate:
            audit_errors.append(
                "stale MatterGen backend repair/CUDA no-dispatch gate removed after successful MatterGen jobs"
            )
        else:
            audit_errors.append(
                "stale MatterGen backend repair/CUDA qualifiers removed after successful MatterGen jobs"
            )

    raw_boundaries = sanitized.get("failure_boundaries")
    if isinstance(raw_boundaries, list):
        kept_boundaries: list[Any] = []
        removed_boundary_gate = False
        for item in raw_boundaries:
            if stale_mattergen_backend_gate_text(_jsonish_text_blob(item)):
                removed_boundary_gate = True
                continue
            kept_boundaries.append(_clean_residual_mattergen_repair_qualifiers(item))
        if removed_boundary_gate:
            sanitized["failure_boundaries"] = kept_boundaries
            sanitized["mattergen_backend_gate_superseded"] = True
            audit_errors.append(
                "stale MatterGen backend repair/CUDA failure_boundary removed after successful MatterGen jobs"
            )
        elif kept_boundaries != raw_boundaries:
            sanitized["failure_boundaries"] = kept_boundaries

    if sanitized.get("mattergen_backend_gate_superseded"):
        sanitized["controller_postmortem_audit_errors"] = audit_errors
    return sanitized


def latest_xy_strategy_constraints_from_memory(
    memory: Mapping[str, Any],
    *,
    mattergen_backend_verified: bool = False,
) -> dict[str, Any]:
    records = memory.get("records")
    if not isinstance(records, list):
        return {}
    for index in range(len(records) - 1, -1, -1):
        record = records[index]
        if not isinstance(record, Mapping):
            continue
        postmortem = record.get("xy_postmortem")
        if not isinstance(postmortem, Mapping):
            continue
        try:
            prior_memory = {"records": records[:index]}
            forbidden_formulas = failed_or_used_formulas_from_memory(prior_memory)
            forbidden_formulas.update(formulas_from_iteration_record(record))
            postmortem = enforce_postmortem_next_strategy_guard(
                postmortem,
                forbidden_reduced_formulas=forbidden_formulas,
            )
        except Exception:
            postmortem = dict(postmortem)
        postmortem = _sanitize_stale_mattergen_backend_gate_strategy(
            postmortem,
            mattergen_backend_verified=mattergen_backend_verified,
        )
        raw_next_strategy = str(postmortem.get("next_strategy") or "")
        next_strategy_candidate_tokens = strategy_formula_candidate_tokens_from_text(raw_next_strategy)
        next_strategy_candidate_formulas = strategy_formula_candidates_from_text(raw_next_strategy)
        if next_strategy_candidate_formulas:
            display_formulas = next_strategy_candidate_tokens or next_strategy_candidate_formulas
            next_strategy = "Use candidates: " + ", ".join(display_formulas) + "."
        else:
            next_strategy = short_text(raw_next_strategy, 220)
        failure_items = postmortem.get("failure_boundaries")
        if not next_strategy and not failure_items:
            continue
        result = record.get("evaluation_result")
        formula = None
        e_hull = None
        if isinstance(result, Mapping):
            formula = result.get("formula") or result.get("reduced_formula")
            e_hull = result.get("e_hull")
        boundaries = (
            [compact_failure_boundary_for_strategy(item) for item in failure_items[:8]]
            if isinstance(failure_items, list)
            else []
        )
        audit_errors = postmortem.get("controller_postmortem_audit_errors", [])
        compact_audit_errors = [short_text(item, 100) for item in audit_errors[:1]] if isinstance(audit_errors, list) else []
        return {
            "source_iteration": record.get("iteration") or postmortem.get("iteration"),
            "source_formula": formula,
            "source_e_hull": e_hull,
            "outcome_class": postmortem.get("outcome_class"),
            "next_strategy": next_strategy,
            "next_strategy_candidate_formulas": next_strategy_candidate_formulas,
            "failure_boundaries": boundaries,
            "controller_postmortem_audit_errors": compact_audit_errors,
            "controller_postmortem_audit_error_total": len(audit_errors) if isinstance(audit_errors, list) else 0,
            "mattergen_backend_gate_superseded": bool(postmortem.get("mattergen_backend_gate_superseded")),
            "binding_policy": ["failure_boundaries are reject conditions"],
        }
    return {}


def sequential_context_payload(
    *,
    state: Mapping[str, Any],
    mode: str,
    iteration: int,
    memory_path: Path,
    candidate_source: str,
    seed: int,
    generator_template_only: bool = False,
    generator_backend: str = DEFAULT_GENERATOR_BACKEND,
    xy_sun_candidate_queue_size: int = DEFAULT_XY_SUN_CANDIDATE_QUEUE_SIZE,
    sequential_materialization_target_count: int = 1,
) -> dict[str, Any]:
    include_experience = mode == "experience_xy"
    context = compact_sequential_state_context(state, include_experience=include_experience)
    context.update(
        {
            "generation_protocol": "sequential_single",
            "iteration": iteration,
            "seed": seed,
            "candidate_source": candidate_source,
            "generator_backend": generator_backend,
            "candidate_count_this_iteration": max(1, int(sequential_materialization_target_count)),
            "controller_constraints": {
                "xy_sun_candidate_queue_size": max(2, min(6, int(xy_sun_candidate_queue_size))),
                "no_mp_pool": candidate_source == "generator",
                "avoid_reduced_formula_repeats": True,
                "z_w_can_return_to_xy": True,
                "control_candidate_requested": False,
                "generator_template_only": generator_template_only,
                "mattergen_backend_enabled": generator_backend == "mattergen",
                "mattergen_request_defaults": {
                    "target_count": max(
                        DEFAULT_MATTERGEN_TARGET_COUNT,
                        int(sequential_materialization_target_count),
                    ),
                    "batch_size": DEFAULT_MATTERGEN_BATCH_SIZE,
                    "num_batches": DEFAULT_MATTERGEN_NUM_BATCHES,
                }
                if generator_backend == "mattergen"
                else None,
                "allowed_generator_templates": list(ALLOWED_TEMPLATES) if generator_template_only else None,
                "generator_template_role_requirements": {
                    template: list(required_roles(template)) for template in ALLOWED_TEMPLATES
                }
                if generator_template_only
                else None,
                "generator_template_role_stoichiometry": {
                    template: dict(TEMPLATE_ROLE_COUNTS[template]) for template in ALLOWED_TEMPLATES
                }
                if generator_template_only
                else None,
                "structure_dicts_allowed": not generator_template_only,
            },
        }
    )
    memory = read_sequential_memory(memory_path)
    mattergen_status = mattergen_operational_status_from_memory(
        memory,
        generator_backend=generator_backend,
    )
    controller_constraints = context.get("controller_constraints")
    if isinstance(controller_constraints, MutableMapping):
        controller_constraints["mattergen_operational_status"] = (
            mattergen_status if generator_backend == "mattergen" else None
        )
    latest_strategy = latest_xy_strategy_constraints_from_memory(
        memory,
        mattergen_backend_verified=bool(mattergen_status.get("verified")),
    )
    if latest_strategy:
        context["latest_xy_strategy_constraints"] = latest_strategy
    if isinstance(controller_constraints, MutableMapping):
        template_policy = xy_search_policy_from_memory(memory, iteration=iteration)
        controller_constraints["search_policy"] = (
            mattergen_search_policy_from_template_policy(template_policy)
            if generator_backend == "mattergen"
            else template_policy
        )
    return context


def read_sequential_memory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": "xy_sequential_single_memory.v1", "records": []}
    raw = read_json(path, {})
    if isinstance(raw, Mapping):
        records = raw.get("records", [])
        if not isinstance(records, list):
            records = []
        return {**dict(raw), "records": records}
    if isinstance(raw, list):
        return {"schema_version": "xy_sequential_single_memory.v1", "records": raw}
    return {"schema_version": "xy_sequential_single_memory.v1", "records": []}


def append_sequential_memory(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    memory = read_sequential_memory(path)
    records = list(memory.get("records", [])) if isinstance(memory.get("records"), list) else []
    records.append(dict(record))
    memory.update(
        {
            "schema_version": "xy_sequential_single_memory.v1",
            "updated_at_utc": utc_now(),
            "records": records,
        }
    )
    write_json(path, memory)
    return memory


def memory_with_transient_materialization_errors(
    memory: Mapping[str, Any],
    materialization_errors: Sequence[str],
) -> dict[str, Any]:
    if not materialization_errors:
        return dict(memory)
    records = list(memory.get("records", [])) if isinstance(memory.get("records"), list) else []
    records.append({"status": "transient_current_iteration", "materialization_errors": list(materialization_errors)})
    return {**dict(memory), "records": records}


def seed_sequential_memory_if_needed(memory_path: Path, seed_path: Path | None) -> dict[str, Any] | None:
    if seed_path is None:
        return None
    if not seed_path.exists():
        raise FileNotFoundError(f"--seed-sequential-memory does not exist: {seed_path}")
    if memory_path.exists():
        existing = read_sequential_memory(memory_path)
        records = existing.get("records")
        if isinstance(records, list) and records:
            return {
                "seeded": False,
                "reason": "target memory already has records",
                "target": str(memory_path),
                "existing_record_count": len(records),
            }
    seed = read_sequential_memory(seed_path)
    records = list(seed.get("records", [])) if isinstance(seed.get("records"), list) else []
    seeded = {
        **dict(seed),
        "schema_version": "xy_sequential_single_memory.v1",
        "updated_at_utc": utc_now(),
        "seeded_from": str(seed_path),
        "records": records,
    }
    write_json(memory_path, seeded)
    return {
        "seeded": True,
        "source": str(seed_path),
        "target": str(memory_path),
        "record_count": len(records),
    }


def summarize_single_evaluation(round_dir: Path, selected_record: Mapping[str, Any]) -> dict[str, Any]:
    ranked_path = round_dir / "analysis" / "e_hull_ranked.csv"
    summary_path = round_dir / "analysis" / "summary.json"
    rows = load_e_hull_rows(ranked_path) if ranked_path.exists() else []
    summary = read_json(summary_path, {}) if summary_path.exists() else {}
    row = rows[0] if rows else {}
    e_hull = None
    if row.get("e_hull") is not None:
        try:
            e_hull = float(row["e_hull"])
        except Exception:
            e_hull = None
    formula = row.get("formula") or selected_record.get("formula")
    result = {
        "formula": formula,
        "e_hull": e_hull,
        "is_sun": bool(e_hull is not None and e_hull < 0.0),
        "near_stable_0_03": bool(e_hull is not None and e_hull < 0.03),
        "near_stable_0_10": bool(e_hull is not None and e_hull < 0.10),
        "ranked_row": row,
    }
    missing_rows = summary.get("missing_e_hull_rows") if isinstance(summary, Mapping) else None
    if e_hull is None and isinstance(missing_rows, list) and missing_rows:
        first_missing = missing_rows[0] if isinstance(missing_rows[0], Mapping) else {}
        result["evaluation_error"] = "missing_e_hull"
        result["evaluation_error_reason"] = first_missing.get("reason") or "e_hull_not_found_in_evaluator_log"
        result["missing_e_hull_count"] = summary.get("missing_e_hull_count")
        result["missing_e_hull_row"] = dict(first_missing)
        if not formula and first_missing.get("formula"):
            result["formula"] = first_missing.get("formula")
    return result


def summarize_batch_evaluation(round_dir: Path, selected_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranked_path = round_dir / "analysis" / "e_hull_ranked.csv"
    summary_path = round_dir / "analysis" / "summary.json"
    rows = load_e_hull_rows(ranked_path) if ranked_path.exists() else []
    summary = read_json(summary_path, {}) if summary_path.exists() else {}
    evaluated_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        e_hull = None
        if item.get("e_hull") is not None:
            try:
                e_hull = float(item["e_hull"])
            except Exception:
                e_hull = None
        item["e_hull"] = e_hull
        item["is_sun"] = bool(e_hull is not None and e_hull < 0.0)
        item["near_stable_0_03"] = bool(e_hull is not None and e_hull < 0.03)
        item["near_stable_0_10"] = bool(e_hull is not None and e_hull < 0.10)
        evaluated_rows.append(item)
    sun_rows = [item for item in evaluated_rows if item.get("is_sun") is True]
    near003_rows = [item for item in evaluated_rows if item.get("near_stable_0_03") is True]
    near010_rows = [item for item in evaluated_rows if item.get("near_stable_0_10") is True]
    missing_rows = summary.get("missing_e_hull_rows") if isinstance(summary, Mapping) else None
    result = {
        "materialized_count": len(selected_records),
        "evaluated_count": len(evaluated_rows),
        "sun_count": len(sun_rows),
        "near_stable_0_03_count": len(near003_rows),
        "near_stable_0_10_count": len(near010_rows),
        "sun_ratio": (len(sun_rows) / len(evaluated_rows)) if evaluated_rows else None,
        "best_rows": evaluated_rows[:20],
        "summary": {
            "count": summary.get("count") if isinstance(summary, Mapping) else None,
            "sun_strict_e_hull_lt_0": summary.get("sun_strict_e_hull_lt_0") if isinstance(summary, Mapping) else None,
            "e_hull_lt_0_03": summary.get("e_hull_lt_0_03") if isinstance(summary, Mapping) else None,
            "e_hull_lt_0_10": summary.get("e_hull_lt_0_10") if isinstance(summary, Mapping) else None,
            "min_e_hull": summary.get("min_e_hull") if isinstance(summary, Mapping) else None,
            "mean_e_hull": summary.get("mean_e_hull") if isinstance(summary, Mapping) else None,
            "max_e_hull": summary.get("max_e_hull") if isinstance(summary, Mapping) else None,
        },
    }
    if isinstance(missing_rows, list) and missing_rows:
        result["missing_e_hull_count"] = summary.get("missing_e_hull_count")
        result["missing_e_hull_rows"] = missing_rows[:10]
    return result


def sequential_fallback_postmortem(
    record: Mapping[str, Any],
    *,
    iteration: int,
    reason: str,
) -> dict[str, Any]:
    result = record.get("evaluation_result") if isinstance(record.get("evaluation_result"), Mapping) else {}
    selected = record.get("selected_record") if isinstance(record.get("selected_record"), Mapping) else {}
    material_description = (
        record.get("material_description") if isinstance(record.get("material_description"), Mapping) else {}
    )
    formula = _normalize_formula_text(result.get("formula") or selected.get("formula") or record.get("formula"))
    e_hull: float | None = None
    if result.get("e_hull") is not None:
        try:
            e_hull = float(result["e_hull"])
        except Exception:
            e_hull = None
    missing_e_hull = str(result.get("evaluation_error") or "") == "missing_e_hull"
    status = str(record.get("status") or "")
    if status == "strategy_blocked":
        outcome = "strategy_blocked"
    elif e_hull is None:
        outcome = "materialization_failure" if record.get("materialization_errors") else "not_evaluated"
    elif bool(result.get("is_sun")) or e_hull < 0.0:
        outcome = "sun"
    elif e_hull < 0.03:
        outcome = "near_miss"
    elif e_hull < 0.10:
        outcome = "weak_near_miss"
    else:
        outcome = "high_e_hull"

    template = str(material_description.get("generator_template") or "").strip()
    roles = material_description.get("generator_role_mapping")
    role_summary = ""
    if isinstance(roles, Mapping):
        pieces = []
        for role, payload in roles.items():
            if not isinstance(payload, Mapping):
                continue
            element = payload.get("element")
            ox = payload.get("oxidation_state")
            if element is not None and ox is not None:
                pieces.append(f"{role}={element}:{ox:+g}" if isinstance(ox, (int, float)) else f"{role}={element}:{ox}")
        if pieces:
            role_summary = "; roles " + ", ".join(pieces)
    formula_text = formula or "the attempted material"
    null_elements = sorted(evaluator_null_elements_from_formula(formula))
    template_text = f" in template={template}" if template else ""
    reason_text = short_text(reason, 500)

    if outcome == "sun":
        next_strategy = (
            f"Do not repeat {formula_text}. Treat this as a SUN hit; if more candidates are required, propose exactly "
            f"one non-duplicate close analog using the same faithful allowed-template motif{template_text}{role_summary}, "
            "varying one chemically adjacent role at a time and preserving charge neutrality."
        )
    elif outcome == "near_miss":
        next_strategy = (
            f"Do not repeat {formula_text}. Because the evaluated material is below 0.03 eV/atom but not SUN, keep the "
            f"successful local motif only if the next proposal is a non-duplicate, charge-neutral, faithful "
            f"allowed-template analog{template_text}{role_summary}. Prefer the first explicit non-duplicate fallback "
            "already named in prior memory; otherwise vary exactly one non-redox role and reject all failed "
            "volume/template boundaries."
        )
    elif outcome == "weak_near_miss":
        next_strategy = (
            f"Do not repeat {formula_text}. Because the evaluated material is below 0.10 eV/atom but not below "
            "0.03 eV/atom and not SUN, treat this as weak support only. Do not keep enumerating the same local "
            "repair axis unless the latest memory names exactly one duplicate-gated final test; otherwise pivot to "
            f"a new explicit allowed-template branch with charge-neutral roles{template_text}{role_summary}."
        )
    elif outcome == "high_e_hull":
        next_strategy = (
            f"Do not repeat {formula_text}. Mark this chemistry/size axis as weakened by high e_hull and pivot to a "
            "new explicit allowed-template branch with charge-neutral roles, rather than enumerating more formulas "
            "from the same failed axis."
        )
    elif outcome == "strategy_blocked":
        next_strategy = (
            "Do not repeat the no-valid-material branch certified by X/Y. Choose a new explicit route whose formula, "
            "template, role mapping, and motif are outside the impossibility certificate and prior failed boundaries."
        )
    elif missing_e_hull:
        null_text = ", ".join(null_elements) if null_elements else "the elements in this evaluator-null formula"
        next_strategy = (
            f"Do not repeat {formula_text}. Treat this as an evaluator-null failure rather than a near miss: "
            f"avoid {null_text} in future generator-template candidates and pivot to a route whose elements have "
            "prior evaluator e_hull support."
        )
    elif outcome == "materialization_failure":
        next_strategy = (
            "Return to X/Y before Z/W. The next proposal must be representable by exactly one allowed generator "
            "template with charge-neutral roles and must avoid every materialization error boundary from this record."
        )
    else:
        next_strategy = (
            "The iteration was not evaluated. Continue with exactly one non-duplicate, charge-neutral material that "
            "uses an allowed generator template and explicitly audits template faithfulness."
        )

    failure_boundaries = []
    if formula:
        failure_boundaries.append(f"Do not repeat reduced_formula={formula}.")
    if template:
        failure_boundaries.append(f"Require a faithful, charge-neutral use of template={template}; do not use it as a formula-only proxy.")
    if record.get("materialization_errors"):
        failure_boundaries.append("Preserve all materialization_errors from this record as hard generator boundaries.")
    if missing_e_hull:
        failure_boundaries.append(
            "Evaluator returned missing_e_hull for this candidate; avoid evaluator-null/high-risk elements "
            f"{null_elements or [formula_text]} in future template-only proposals."
        )

    return {
        "status": "postmortem_consensus",
        "agent": "controller_fallback",
        "iteration": iteration,
        "outcome_class": outcome,
        "causal_interpretation": (
            f"Fallback postmortem because X/Y postmortem LLM failed: {reason_text}. "
            f"Evaluation summary: formula={formula_text}, e_hull={e_hull}, status={status or 'unknown'}."
        ),
        "strategy_update": "Preserve the evaluator signal and continue with one audited, non-duplicate, template-faithful proposal.",
        "next_strategy": next_strategy,
        "principle_updates": [],
        "failure_boundaries": failure_boundaries,
        "review_summary": "Controller fallback generated after postmortem transport failure.",
    }


def used_formulas_from_memory(memory: Mapping[str, Any]) -> set[str]:
    formulas: set[str] = set()
    records = memory.get("records")
    if not isinstance(records, list):
        return formulas
    for record in records:
        if not isinstance(record, Mapping):
            continue
        result = record.get("evaluation_result")
        selected = record.get("selected_record")
        for source in (result, selected, record):
            if not isinstance(source, Mapping):
                continue
            formula = _normalize_formula_text(source.get("formula") or source.get("reduced_formula"))
            if formula:
                formulas.add(formula)
        for source_key in ("selected_records", "evaluation_results"):
            values = record.get(source_key)
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, Mapping):
                    continue
                formula = _normalize_formula_text(item.get("formula") or item.get("reduced_formula"))
                if formula:
                    formulas.add(formula)
        batch_result = record.get("evaluation_results")
        if isinstance(batch_result, Mapping):
            best_rows = batch_result.get("best_rows")
            if isinstance(best_rows, list):
                for item in best_rows:
                    if not isinstance(item, Mapping):
                        continue
                    formula = _normalize_formula_text(item.get("formula") or item.get("reduced_formula"))
                    if formula:
                        formulas.add(formula)
    return formulas


def failed_or_used_formulas_from_memory(memory: Mapping[str, Any]) -> set[str]:
    formulas = set(used_formulas_from_memory(memory))
    records = memory.get("records")
    if not isinstance(records, list):
        return formulas
    for record in records:
        if not isinstance(record, Mapping):
            continue
        for key in ("formula", "reduced_formula"):
            formula = _normalize_formula_text(record.get(key))
            if formula:
                formulas.add(formula)
        postmortem = record.get("xy_postmortem")
        if isinstance(postmortem, Mapping):
            formulas.update(
                formula
                for formula in (
                    _normalize_formula_text(item)
                    for item in formula_mentions_from_text(postmortem.get("failure_boundaries"))
                )
                if formula
            )
        payloads: list[Any] = [record, record.get("selected_record")]
        selected_records = record.get("selected_records")
        if isinstance(selected_records, list):
            payloads.extend(selected_records)
        for payload in payloads:
            if not isinstance(payload, Mapping):
                continue
            report = payload.get("crystal_llm_mattergen_report")
            if not isinstance(report, Mapping):
                continue
            accepted_formulas = report.get("accepted_formulas")
            if isinstance(accepted_formulas, list):
                for item in accepted_formulas:
                    formula = _normalize_formula_text(item)
                    if formula:
                        formulas.add(formula)
        failed_generated = record.get("failed_generated_formulas")
        if isinstance(failed_generated, list):
            for item in failed_generated:
                if isinstance(item, Mapping):
                    formula = _normalize_formula_text(item.get("formula"))
                    if formula:
                        formulas.add(formula)
        errors = record.get("materialization_errors")
        if isinstance(errors, list):
            for item in errors:
                text = str(item or "")
                for pattern in (GENERATED_FORMULA_ERROR_RE, GENERATED_FORMULA_EXCLUDED_RE):
                    for match in pattern.finditer(text):
                        formula = _normalize_formula_text(match.group(1))
                        if formula:
                            formulas.add(formula)
                match = re.search(r"generated\s+([A-Za-z0-9()]+)\s+with template\s+([A-Za-z0-9_]+)", text)
                if match:
                    formula = _normalize_formula_text(match.group(1))
                    if formula:
                        formulas.add(formula)
    return formulas


def _canonical_element_symbol(value: Any) -> str:
    token = str(value or "").strip().strip("'\"")
    if not token:
        return ""
    try:
        return str(Element(token).symbol)
    except Exception:
        return ""


def _canonical_chemical_system(elements: Iterable[Any]) -> str:
    canonical: list[str] = []
    for item in elements:
        symbol = _canonical_element_symbol(item)
        if symbol and symbol not in canonical:
            canonical.append(symbol)
    return "-".join(sorted(canonical))


def _chemical_system_elements_from_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        elements: list[str] = []
        for key in (
            "chemical_system",
            "elements",
            "allowed",
            "required",
            "allowed_elements",
            "required_elements",
        ):
            for element in _chemical_system_elements_from_text(value.get(key)):
                if element and element not in elements:
                    elements.append(element)
        return elements
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical_element_symbol(item) for item in value if _canonical_element_symbol(item)]
    text = str(value or "").strip()
    if not text:
        return []
    text = re.split(
        r"\s+(?:allowed_elements|required_elements|structure_elements|filters|request|reason|status)\b",
        text,
        maxsplit=1,
    )[0]
    text = text.strip("[](){}")
    text = text.replace("'", "").replace('"', "")
    return [
        symbol
        for symbol in (_canonical_element_symbol(part) for part in re.split(r"[-,;/\s]+", text))
        if symbol
    ]


def _chemical_system_from_formula(formula: Any) -> str:
    normalized = _normalize_formula_text(formula)
    if not normalized:
        return ""
    try:
        composition = Composition(normalized)
    except Exception:
        return ""
    return _canonical_chemical_system(str(element.symbol) for element in composition.elements)


def _failed_or_used_formula_hints_for_context(
    context: Mapping[str, Any],
    forbidden_formulas: set[str],
    recent_formulas: Sequence[str],
    *,
    limit: int = XY_CONTEXT_FAILED_OR_USED_FORMULA_HINT_LIMIT,
) -> list[str]:
    """Return a compact prompt-visible duplicate list biased to the active chemistry."""

    hint_systems: set[str] = set()
    for formula in latest_strategy_candidate_formulas(context):
        system = _chemical_system_from_formula(formula)
        if system:
            hint_systems.add(system)
    latest = context.get("latest_xy_strategy_constraints")
    if isinstance(latest, Mapping):
        for formula in (latest.get("source_formula"),):
            system = _chemical_system_from_formula(formula)
            if system:
                hint_systems.add(system)
    constraints = context.get("controller_constraints")
    if isinstance(constraints, Mapping):
        status = constraints.get("mattergen_operational_status")
        if isinstance(status, Mapping):
            system = _chemical_system_from_formula(status.get("success_formula"))
            if system:
                hint_systems.add(system)
    for formula in recent_formulas[:6]:
        system = _chemical_system_from_formula(formula)
        if system:
            hint_systems.add(system)

    hints: list[str] = []
    for formula in recent_formulas:
        normalized = _normalize_formula_text(formula)
        if normalized and normalized not in hints:
            hints.append(normalized)
        if len(hints) >= limit:
            return hints
    if hint_systems:
        for formula in sorted(forbidden_formulas):
            normalized = _normalize_formula_text(formula)
            if not normalized or normalized in hints:
                continue
            if _chemical_system_from_formula(normalized) in hint_systems:
                hints.append(normalized)
            if len(hints) >= limit:
                break
        return hints
    for formula in sorted(forbidden_formulas):
        normalized = _normalize_formula_text(formula)
        if normalized and normalized not in hints:
            hints.append(normalized)
        if len(hints) >= limit:
            break
    return hints


def _extract_generated_formulas_from_error(text: Any) -> list[str]:
    seen: list[str] = []
    raw = str(text or "")
    patterns = (
        GENERATED_FORMULA_ERROR_RE,
        GENERATED_FORMULA_EXCLUDED_RE,
        re.compile(r"\b(?:formula|reduced_formula)\s*[:=]\s*([A-Z][A-Za-z0-9()]+)\b"),
    )
    for pattern in patterns:
        for match in pattern.finditer(raw):
            formula = _normalize_formula_text(match.group(1))
            if formula and formula not in seen:
                seen.append(formula)
    return seen


def _chemical_systems_from_error_text(text: Any) -> list[str]:
    systems: list[str] = []
    raw = str(text or "")
    for formula in _extract_generated_formulas_from_error(raw):
        system = _chemical_system_from_formula(formula)
        if system and system not in systems:
            systems.append(system)
    for match in CHEMICAL_SYSTEM_ERROR_RE.finditer(raw):
        system = _canonical_chemical_system(_chemical_system_elements_from_text(match.group(1)))
        if system and system not in systems:
            systems.append(system)
    return systems


def _mattergen_request_chemical_systems(request: Mapping[str, Any]) -> list[str]:
    systems: list[str] = []
    filters = request.get("filters") if isinstance(request.get("filters"), Mapping) else {}
    properties = (
        request.get("properties_to_condition_on")
        if isinstance(request.get("properties_to_condition_on"), Mapping)
        else {}
    )
    for value in (
        filters.get("chemical_system"),
        properties.get("chemical_system"),
        request.get("chemical_system"),
    ):
        system = _canonical_chemical_system(_chemical_system_elements_from_text(value))
        if system and system not in systems:
            systems.append(system)
    for value in (
        filters.get("target_reduced_formula"),
        properties.get("target_reduced_formula"),
        request.get("target_reduced_formula"),
    ):
        system = _chemical_system_from_formula(value)
        if system and system not in systems:
            systems.append(system)
    return systems


def _extract_mattergen_request_chemical_systems_from_payload(payload: Any, *, max_items: int = 24) -> list[str]:
    systems: list[str] = []

    def add(system: str) -> None:
        if system and system not in systems and len(systems) < max_items:
            systems.append(system)

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 7 or len(systems) >= max_items:
            return
        if isinstance(value, Mapping):
            requests = value.get("mattergen_requests")
            if isinstance(requests, Mapping):
                for system in _mattergen_request_chemical_systems(requests):
                    add(system)
            elif isinstance(requests, list):
                for request in requests:
                    if isinstance(request, Mapping):
                        for system in _mattergen_request_chemical_systems(request):
                            add(system)
            if (
                str(value.get("backend") or "").lower() == "mattergen"
                or isinstance(value.get("filters"), Mapping)
                or isinstance(value.get("properties_to_condition_on"), Mapping)
            ):
                for system in _mattergen_request_chemical_systems(value):
                    add(system)
            chemical_system = value.get("chemical_system")
            if chemical_system is not None:
                add(_canonical_chemical_system(_chemical_system_elements_from_text(chemical_system)))
            for key in ("target_reduced_formula", "reduced_formula", "formula"):
                if key in value:
                    add(_chemical_system_from_formula(value.get(key)))
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, list):
            for child in value[:40]:
                visit(child, depth + 1)
        elif isinstance(value, str):
            for system in _chemical_systems_from_error_text(value):
                add(system)

    visit(payload)
    return systems


def _strategy_family_label(element_symbol: str) -> str:
    symbol = _canonical_element_symbol(element_symbol)
    if symbol in COMMON_ALKALI_ELEMENTS:
        return "alkali"
    if symbol in COMMON_ALKALINE_EARTH_ELEMENTS:
        return "alkaline_earth"
    if symbol in COMMON_HALOGEN_ELEMENTS:
        return "halide"
    if symbol in COMMON_CHALCOGEN_ELEMENTS:
        return "chalcogen"
    if symbol in COMMON_PNICTOGEN_ELEMENTS:
        return "pnictide"
    if symbol in COMMON_GROUP_12_ELEMENTS:
        return symbol
    return symbol


def _strategy_family_sort_key(label: str) -> tuple[int, str]:
    priorities = {
        "alkali": 0,
        "alkaline_earth": 0,
        "pnictide": 6,
        "chalcogen": 7,
        "halide": 8,
    }
    return priorities.get(label, 4), label


def _chemical_system_family_pattern(system: str) -> str:
    labels = [_strategy_family_label(part) for part in str(system or "").split("-") if part]
    labels = [label for label in labels if label]
    return "-".join(sorted(set(labels), key=_strategy_family_sort_key))


def _materialization_error_texts(record: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in ("materialization_errors", "generator_errors"):
        value = record.get(key)
        if isinstance(value, list):
            texts.extend(str(item or "") for item in value if str(item or "").strip())
    value = record.get("failure_reason")
    if value not in (None, "", [], {}):
        texts.append(str(value))
    return texts


def _cooldown_failure_markers(errors: Sequence[str], status: str) -> list[str]:
    joined = "\n".join(str(item or "") for item in errors).lower()
    markers = [marker for marker in XY_STRATEGY_COOLDOWN_ERROR_MARKERS if marker in joined]
    if "not_materialized" in status or "transient_current_iteration" in status:
        markers.append(status)
    return sorted(set(markers))


def mattergen_strategy_cooldowns_from_memory(memory: Mapping[str, Any]) -> dict[str, Any]:
    records = memory.get("records")
    if not isinstance(records, list):
        return {}
    exact_entries: dict[str, dict[str, Any]] = {}
    recent_records = [item for item in records if isinstance(item, Mapping)][-XY_STRATEGY_COOLDOWN_RECENT_RECORD_LIMIT:]
    for record in recent_records:
        status = str(record.get("status") or record.get("outcome_class") or "").lower()
        if "transient_current_iteration" in status:
            continue
        selected_record = record.get("selected_record")
        selected_records = record.get("selected_records")
        has_materialized_success = bool(selected_record) or (
            isinstance(selected_records, list) and any(isinstance(item, Mapping) for item in selected_records)
        )
        if has_materialized_success and any(token in status for token in ("evaluated", "materialized")):
            continue
        errors = _materialization_error_texts(record)
        markers = _cooldown_failure_markers(errors, status)
        failure_like = bool(markers) or any(
            token in status for token in ("not_materialized", "transient_current_iteration")
        )
        if not failure_like:
            continue
        systems = set(_extract_mattergen_request_chemical_systems_from_payload(record))
        generated_formulas: list[str] = []
        for error in errors:
            generated_formulas.extend(_extract_generated_formulas_from_error(error))
            systems.update(_chemical_systems_from_error_text(error))
        systems = {system for system in systems if system}
        if not systems:
            continue
        iteration = record.get("iteration")
        source_iteration = iteration if iteration not in (None, "") else "transient"
        for system in sorted(systems):
            entry = exact_entries.setdefault(
                system,
                {
                    "chemical_system": system,
                    "family_pattern": _chemical_system_family_pattern(system),
                    "failure_count": 0,
                    "source_iterations": [],
                    "known_formula_examples": [],
                    "failure_markers": set(),
                },
            )
            entry["failure_count"] += 1
            if source_iteration not in entry["source_iterations"]:
                entry["source_iterations"].append(source_iteration)
            for formula in generated_formulas:
                if formula and formula not in entry["known_formula_examples"]:
                    entry["known_formula_examples"].append(formula)
            entry["failure_markers"].update(markers or ["materialization_failure"])

    blocked_systems = []
    for system, entry in sorted(exact_entries.items()):
        if int(entry["failure_count"]) < XY_STRATEGY_COOLDOWN_EXACT_FAILURE_THRESHOLD:
            continue
        blocked_systems.append(
            {
                "chemical_system": system,
                "family_pattern": entry["family_pattern"],
                "failure_count": entry["failure_count"],
                "source_iterations": entry["source_iterations"][-5:],
                "known_formula_examples": entry["known_formula_examples"][:5],
                "failure_markers": sorted(entry["failure_markers"])[:6],
                "required_escape": "Choose a different chemical system; adding only exclude_reduced_formulas is not enough.",
            }
        )

    pattern_entries: dict[str, dict[str, Any]] = {}
    for entry in blocked_systems:
        pattern = str(entry.get("family_pattern") or "")
        if not pattern:
            continue
        pattern_entry = pattern_entries.setdefault(
            pattern,
            {
                "family_pattern": pattern,
                "chemical_systems": [],
                "failure_count": 0,
                "failure_markers": set(),
            },
        )
        system = str(entry.get("chemical_system") or "")
        if system and system not in pattern_entry["chemical_systems"]:
            pattern_entry["chemical_systems"].append(system)
        pattern_entry["failure_count"] += int(entry.get("failure_count") or 0)
        pattern_entry["failure_markers"].update(entry.get("failure_markers") or [])

    blocked_patterns = []
    for pattern, entry in sorted(pattern_entries.items()):
        distinct_system_count = len(entry["chemical_systems"])
        if distinct_system_count < XY_STRATEGY_COOLDOWN_FAMILY_DISTINCT_SYSTEM_THRESHOLD:
            continue
        blocked_patterns.append(
            {
                "family_pattern": pattern,
                "chemical_systems": sorted(entry["chemical_systems"])[:8],
                "distinct_system_count": distinct_system_count,
                "failure_count": entry["failure_count"],
                "failure_markers": sorted(entry["failure_markers"])[:6],
                "required_escape": "Leave this family pattern; change at least one role family or the mechanism family.",
            }
        )

    if not blocked_systems and not blocked_patterns:
        return {}
    return {
        "policy": "hard_preflight",
        "cooldown_reason": (
            "Recent MatterGen/materialization feedback shows repeated known/training, duplicate, missing-element, "
            "or zero-accepted failures in these systems. The controller blocks the basin before GPU submission."
        ),
        "blocked_chemical_systems": blocked_systems,
        "blocked_chemical_system_total": len(blocked_systems),
        "blocked_family_patterns": blocked_patterns,
        "blocked_family_pattern_total": len(blocked_patterns),
        "required_next_move": (
            "X/Y must select a materially different SUN route outside blocked systems/patterns. "
            "Z/W must repair by changing the MatterGen chemical_system, not by adding only exclude_reduced_formulas."
        ),
        "do_not_solve_by": [
            "adding only more exclude_reduced_formulas inside the same chemical system",
            "switching among alkali variants inside a blocked alkali-Cd-halide style family pattern",
            "dropping required elements or allowing subset fallback to make MatterGen accept something else",
        ],
    }


def mattergen_strategy_cooldown_preflight_errors(
    specs: Sequence[Mapping[str, Any]],
    strategy_cooldowns: Mapping[str, Any] | None,
) -> list[str]:
    if not strategy_cooldowns:
        return []
    blocked_system_entries = strategy_cooldowns.get("blocked_chemical_systems")
    blocked_pattern_entries = strategy_cooldowns.get("blocked_family_patterns")
    blocked_systems = {
        str(item.get("chemical_system") or "")
        for item in blocked_system_entries or []
        if isinstance(item, Mapping) and item.get("chemical_system")
    }
    blocked_patterns = {
        str(item.get("family_pattern") or "")
        for item in blocked_pattern_entries or []
        if isinstance(item, Mapping) and item.get("family_pattern")
    }
    if not blocked_systems and not blocked_patterns:
        return []
    errors: list[str] = []
    for index, spec in enumerate(specs, start=1):
        if not isinstance(spec, Mapping):
            continue
        candidate_id = str(spec.get("id") or spec.get("candidate_id") or f"candidate_{index}")
        for system in _extract_mattergen_request_chemical_systems_from_payload(spec, max_items=8):
            pattern = _chemical_system_family_pattern(system)
            if system in blocked_systems:
                errors.append(
                    "strategy_cooldown_block:"
                    f" candidate={candidate_id} chemical_system={system} family_pattern={pattern} "
                    "matches blocked_chemical_system from recent MatterGen failures; change the MatterGen chemical_system."
                )
            if pattern and pattern in blocked_patterns:
                errors.append(
                    "strategy_cooldown_block:"
                    f" candidate={candidate_id} chemical_system={system} family_pattern={pattern} "
                    "matches blocked_family_pattern from repeated recent MatterGen failures; leave this chemistry basin."
                )
    return errors


def mattergen_excluded_target_preflight_errors(
    specs: Sequence[Mapping[str, Any]],
    excluded_formulas: set[str] | None,
) -> list[str]:
    excluded = _normalised_formula_set(excluded_formulas)
    if not excluded:
        return []
    errors: list[str] = []
    for index, spec in enumerate(specs, start=1):
        if not isinstance(spec, Mapping):
            continue
        candidate_id = str(spec.get("id") or spec.get("candidate_id") or f"candidate_{index}")
        raw_targets: list[Any] = [
            spec.get("target_reduced_formula"),
            spec.get("reduced_formula"),
            spec.get("formula"),
        ]
        requests = spec.get("mattergen_requests")
        if isinstance(requests, list):
            for request in requests:
                if not isinstance(request, Mapping):
                    continue
                filters = request.get("filters") if isinstance(request.get("filters"), Mapping) else {}
                raw_targets.extend(
                    [
                        request.get("target_reduced_formula"),
                        filters.get("target_reduced_formula") if isinstance(filters, Mapping) else None,
                    ]
                )
        for formula in (_normalize_formula_text(item) for item in raw_targets if str(item or "").strip()):
            if formula and formula in excluded:
                errors.append(
                    "mattergen_target_excluded:"
                    f" candidate={candidate_id} target_reduced_formula={formula} is already failed/used or known/training; "
                    "do not submit a MatterGen request whose soft strategy target is also excluded. "
                    "Return to X/Y for a non-duplicate queue item or choose a reviewed non-excluded queue candidate."
                )
                break
    return errors


def failed_volume_boundaries_from_memory(memory: Mapping[str, Any]) -> set[str]:
    boundaries: set[str] = set()
    records = memory.get("records")
    if not isinstance(records, list):
        return boundaries
    for record in records:
        if not isinstance(record, Mapping):
            continue
        errors = record.get("materialization_errors")
        if not isinstance(errors, list):
            continue
        for item in errors:
            text = str(item or "")
            if "volume_per_atom_too_large" not in text:
                continue
            match = re.search(r"generated\s+([A-Za-z0-9()]+)\s+with template\s+([A-Za-z0-9_]+)", text)
            if not match:
                continue
            formula_text = _normalize_formula_text(match.group(1))
            template = str(match.group(2) or "").strip()
            if template not in ALLOWED_TEMPLATES or not formula_text:
                continue
            boundaries.update(template_volume_boundary_keys_from_formula(template, formula_text))
    return boundaries


def evaluator_null_elements_from_memory(memory: Mapping[str, Any]) -> set[str]:
    elements = set(EVALUATOR_NULL_E_HULL_ELEMENTS)
    records = memory.get("records")
    if not isinstance(records, list):
        return elements
    for record in records:
        if not isinstance(record, Mapping):
            continue
        result = record.get("evaluation_result")
        if not isinstance(result, Mapping):
            continue
        if result.get("e_hull") is not None and result.get("evaluation_error") != "missing_e_hull":
            continue
        formula = result.get("formula") or result.get("reduced_formula")
        elements.update(evaluator_null_elements_from_formula(formula))
    return {element for element in elements if element in EVALUATOR_NULL_E_HULL_ELEMENTS}


def populate_sequential_controller_constraints(
    context: MutableMapping[str, Any],
    memory: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    forbidden_history_formulas = failed_or_used_formulas_from_memory(memory)
    forbidden_volume_boundaries = failed_volume_boundaries_from_memory(memory)
    forbidden_null_elements = evaluator_null_elements_from_memory(memory)
    controller_constraints = context.get("controller_constraints")
    if isinstance(controller_constraints, MutableMapping):
        latest_candidates = [
            formula for formula in latest_strategy_candidate_formulas(context) if formula in forbidden_history_formulas
        ]
        recent_formulas: list[str] = []
        records = memory.get("records")
        if isinstance(records, list):
            for record in reversed(records):
                if not isinstance(record, Mapping):
                    continue
                for formula in sorted(formulas_from_iteration_record(record)):
                    if formula in forbidden_history_formulas and formula not in recent_formulas:
                        recent_formulas.append(formula)
                if len(recent_formulas) >= XY_CONTEXT_FAILED_OR_USED_FORMULA_LIMIT:
                    break
        compact_failed_formulas: list[str] = []
        for formula in latest_candidates + recent_formulas + sorted(forbidden_history_formulas):
            if formula and formula not in compact_failed_formulas:
                compact_failed_formulas.append(formula)
            if len(compact_failed_formulas) >= XY_CONTEXT_FAILED_OR_USED_FORMULA_LIMIT:
                break
        failed_formula_hints = _failed_or_used_formula_hints_for_context(
            context,
            forbidden_history_formulas,
            recent_formulas,
        )
        compact_volume_boundaries = sorted(forbidden_volume_boundaries)[:XY_CONTEXT_FAILED_VOLUME_BOUNDARY_LIMIT]
        controller_constraints["failed_or_used_reduced_formulas"] = compact_failed_formulas
        controller_constraints["failed_or_used_reduced_formula_hints"] = failed_formula_hints
        controller_constraints["failed_or_used_reduced_formula_total"] = len(forbidden_history_formulas)
        controller_constraints["failed_or_used_reduced_formula_omitted_count"] = max(
            0, len(forbidden_history_formulas) - len(compact_failed_formulas)
        )
        controller_constraints["failed_or_used_reduced_formula_context_policy"] = "Visible subset; controller enforces full hidden failed/used formula set."
        controller_constraints["failed_or_used_reduced_formula_hint_policy"] = (
            "Visible list is recent plus active-chemical-system hidden failed/used hints; "
            "controller still enforces the complete hidden set and will return exact conflicts."
        )
        controller_constraints["failed_volume_template_boundaries"] = compact_volume_boundaries
        controller_constraints["failed_volume_template_boundary_total"] = len(forbidden_volume_boundaries)
        controller_constraints["failed_volume_template_boundary_omitted_count"] = max(
            0, len(forbidden_volume_boundaries) - len(compact_volume_boundaries)
        )
        controller_constraints["forbidden_evaluator_null_elements"] = sorted(forbidden_null_elements)
        controller_constraints["forbidden_evaluator_null_element_policy"] = "Hard avoid: e_hull=null cannot be SUN."
        if str(context.get("generator_backend") or DEFAULT_GENERATOR_BACKEND) == "mattergen":
            strategy_cooldowns = mattergen_strategy_cooldowns_from_memory(memory)
            if strategy_cooldowns:
                controller_constraints["strategy_cooldowns"] = strategy_cooldowns
            else:
                controller_constraints.pop("strategy_cooldowns", None)
        controller_constraints["strategy_constraints"] = xy_strategy_constraints_from_context(
            context,
            forbidden_formulas=forbidden_history_formulas,
        )
    return forbidden_history_formulas, forbidden_volume_boundaries


def build_sequential_report(
    *,
    args: argparse.Namespace,
    state: Mapping[str, Any],
    memory: Mapping[str, Any],
    materialization_errors: Sequence[str],
) -> dict[str, Any]:
    records = [item for item in memory.get("records", []) if isinstance(item, Mapping)] if isinstance(memory.get("records"), list) else []
    evaluated = [item for item in records if isinstance(item.get("evaluation_result"), Mapping)]
    evaluated_structure_count = 0
    batch_sun_count = 0
    batch_near003_count = 0
    for item in records:
        batch_result = item.get("evaluation_results")
        if isinstance(batch_result, Mapping):
            try:
                evaluated_structure_count += int(batch_result.get("evaluated_count") or 0)
                batch_sun_count += int(batch_result.get("sun_count") or 0)
                batch_near003_count += int(batch_result.get("near_stable_0_03_count") or 0)
            except Exception:
                pass
        elif isinstance(item.get("evaluation_result"), Mapping):
            evaluated_structure_count += 1
            if item["evaluation_result"].get("is_sun") is True:
                batch_sun_count += 1
            if item["evaluation_result"].get("near_stable_0_03") is True:
                batch_near003_count += 1
    sun = [
        item
        for item in evaluated
        if isinstance(item.get("evaluation_result"), Mapping) and item["evaluation_result"].get("is_sun") is True
    ]
    near003 = [
        item
        for item in evaluated
        if isinstance(item.get("evaluation_result"), Mapping) and item["evaluation_result"].get("near_stable_0_03") is True
    ]
    return {
        "schema_version": "xy_sequential_single_report.v1",
        "created_at_utc": utc_now(),
        "mode": args.mode,
        "generation_protocol": args.generation_protocol,
        "requested_iterations": args.candidate_count,
        "completed_records": len(records),
        "evaluated_count": len(evaluated),
        "sun_count": len(sun),
        "sun_ratio": (len(sun) / len(evaluated)) if evaluated else None,
        "near_stable_0_03_count": len(near003),
        "evaluated_structure_count": evaluated_structure_count,
        "structure_sun_count": batch_sun_count,
        "structure_sun_ratio": (batch_sun_count / evaluated_structure_count) if evaluated_structure_count else None,
        "structure_near_stable_0_03_count": batch_near003_count,
        "strict_sun_definition": STRICT_SUN_NOTE,
        "source_state": {
            "current_round": state.get("current_round"),
            "principle_book_len": len(state.get("principle_book", [])) if isinstance(state.get("principle_book"), list) else 0,
        },
        "materialization": {
            "errors": list(materialization_errors)[:200],
            "omitted_error_count": max(0, len(materialization_errors) - 200),
        },
        "records_tail": records[-12:],
    }


def sequential_iteration_plan(candidate_count: int | None) -> tuple[Any, int | None]:
    if candidate_count is None:
        return itertools.count(1), None
    try:
        limit = int(candidate_count)
    except Exception:
        return itertools.count(1), None
    if limit <= 0:
        return itertools.count(1), None
    return range(1, limit + 1), limit


def run_sequential_single_optimizer(
    *,
    args: argparse.Namespace,
    root: Path,
    work_dir: Path,
    state_path: Path,
    candidate_pool_path: Path,
    state: Mapping[str, Any],
    training_data: Path,
    ppd_path: Path,
    known_formulas: set[str] | None,
) -> int:
    pool_records: list[dict[str, Any]] = []
    memory_path = work_dir / "sequential_memory.json"
    seed_memory_report = seed_sequential_memory_if_needed(
        memory_path, Path(args.seed_sequential_memory) if getattr(args, "seed_sequential_memory", None) else None
    )
    materialization_errors_all: list[str] = []
    iteration_numbers, iteration_limit = sequential_iteration_plan(args.candidate_count)
    iteration_label = str(iteration_limit) if iteration_limit is not None else "unbounded"
    print(
        f"[{utc_now()}] xy_sequential_start mode={args.mode} iterations={iteration_label} work_dir={work_dir} memory={memory_path}",
        flush=True,
    )
    if seed_memory_report:
        print(f"[{utc_now()}] xy_sequential_memory_seed {prompt_json(seed_memory_report)}", flush=True)
    for iteration in iteration_numbers:
        iter_dir = work_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        sequential_batch_target = max(1, int(getattr(args, "sequential_materialization_target_count", 1)))
        context = sequential_context_payload(
            state=state,
            mode=args.mode,
            iteration=iteration,
            memory_path=memory_path,
            candidate_source=args.candidate_source,
            seed=args.seed_base + iteration,
            generator_template_only=args.generator_template_only,
            generator_backend=args.generator_backend,
            xy_sun_candidate_queue_size=args.xy_sun_candidate_queue_size,
            sequential_materialization_target_count=sequential_batch_target,
        )
        memory_snapshot = read_sequential_memory(memory_path)
        forbidden_history_formulas, forbidden_volume_boundaries = populate_sequential_controller_constraints(
            context,
            memory_snapshot,
        )
        x_client = make_client(
            role="X",
            args=args,
            root=root,
            log_dir=iter_dir / "x_llm_calls",
            state_path=state_path,
            candidate_pool_path=candidate_pool_path,
            xy_history_path=memory_path,
        )
        y_client = make_client(
            role="Y",
            args=args,
            root=root,
            log_dir=iter_dir / "y_llm_calls",
            state_path=state_path,
            candidate_pool_path=candidate_pool_path,
            xy_history_path=memory_path,
        )
        z_client = make_client(
            role="Z",
            args=args,
            root=root,
            log_dir=iter_dir / "z_llm_calls",
            state_path=state_path,
            candidate_pool_path=candidate_pool_path,
            xy_history_path=memory_path,
        )
        w_client = make_client(
            role="W",
            args=args,
            root=root,
            log_dir=iter_dir / "w_llm_calls",
            state_path=state_path,
            candidate_pool_path=candidate_pool_path,
            xy_history_path=memory_path,
        )
        return_feedback: dict[str, Any] | None = None
        material_consensus: dict[str, Any] | None = None
        candidate_consensus: dict[str, Any] | None = None
        selected_records: list[dict[str, Any]] = []
        locked_specs: list[dict[str, Any]] = []
        executable_generator_rule: dict[str, Any] | None = None
        iteration_errors: list[str] = []
        xy_dialogue: list[dict[str, Any]] = []
        zw_dialogue: list[dict[str, Any]] = []

        for description_attempt in range(1, max(1, int(args.max_description_revision_rounds)) + 1):
            proposal = call_json_object(
                x_client,
                system=AGENT_X_SEQUENTIAL_SYSTEM,
                user=prompt_x_sequential_material_proposal(
                    context,
                    iteration=iteration,
                    return_feedback=return_feedback,
                    template_only=args.generator_template_only,
                ),
                role="agent_x_sequential",
                metadata={
                    "role": "agent_x_sequential",
                    "stage": "xy_sequential_material_proposal",
                    "iteration": iteration,
                    "description_attempt": description_attempt,
                    "cycle": 1,
                },
                json_repair_attempts=args.json_repair_attempts,
            )
            proposal = payload_with_controller_iteration(proposal, iteration=iteration)
            xy_dialogue.append({"role": "X", "mode": "material_description_proposal", "cycle": 1, "payload": proposal})
            review: dict[str, Any] = {}
            max_rounds = max(1, int(args.max_debate_rounds))
            min_rounds = max(1, min(int(args.min_debate_rounds), max_rounds))
            xy_rejection_streak = 0
            for cycle in range(1, max_rounds + 1):
                review = call_json_object(
                    y_client,
                    system=AGENT_Y_SEQUENTIAL_SYSTEM,
                    user=prompt_y_sequential_material_review(
                        context,
                        proposal,
                        iteration=iteration,
                        cycle=cycle,
                        return_feedback=return_feedback,
                        template_only=args.generator_template_only,
                    ),
                    role="agent_y_sequential",
                    metadata={
                        "role": "agent_y_sequential",
                        "stage": "xy_sequential_material_review",
                        "iteration": iteration,
                        "description_attempt": description_attempt,
                        "cycle": cycle,
                    },
                    json_repair_attempts=args.json_repair_attempts,
                )
                review = payload_with_controller_iteration(review, iteration=iteration)
                review = enforce_no_valid_strategy_audit_guard(review, proposal, context)
                review = enforce_exhausted_strategy_route_audit_guard(review, proposal, context)
                review = enforce_latest_strategy_selection_order_guard(review, proposal, context, return_feedback)
                review = payload_with_controller_iteration(review, iteration=iteration)
                xy_dialogue.append({"role": "Y", "mode": "material_description_review", "cycle": cycle, "payload": review})
                if sequential_no_valid_description_agrees_with_context(review, proposal, context):
                    material_consensus = no_valid_material_consensus_from_payload(
                        proposal,
                        review,
                        iteration=iteration,
                    )
                    xy_dialogue.append(
                        {
                            "role": "Y",
                            "mode": "no_valid_material_consensus",
                            "cycle": cycle,
                            "payload": material_consensus,
                        }
                    )
                    break
                if sequential_description_agrees(review, proposal):
                    if cycle < min_rounds:
                        xy_dialogue.append(
                            {
                                "role": "controller",
                                "mode": "xy_early_clean_approval",
                                "cycle": cycle,
                                "payload": {
                                    "reason": (
                                        "Y explicitly approved the ranked queue and selected material; "
                                        "controller finalized without forcing extra debate cycles."
                                    ),
                                    "configured_min_debate_rounds": min_rounds,
                                },
                            }
                        )
                    material_consensus = approved_material_consensus_from_payload(
                        proposal,
                        review,
                        iteration=iteration,
                    )
                    xy_dialogue.append({"role": "Y", "mode": "material_description_final", "cycle": cycle, "payload": material_consensus})
                    break
                if review_requires_counterproposal(
                    review,
                    rejection_streak=xy_rejection_streak,
                    threshold=args.critic_counterproposal_after,
                ):
                    xy_rejection_streak = 0
                    previous_proposal = proposal
                    counterproposal = call_json_object(
                        y_client,
                        system=AGENT_Y_SEQUENTIAL_SYSTEM,
                        user=prompt_y_sequential_material_counterproposal(
                            context,
                            previous_proposal,
                            review,
                            iteration=iteration,
                            cycle=cycle,
                            return_feedback=return_feedback,
                            template_only=args.generator_template_only,
                        ),
                        role="agent_y_sequential_counterproposal",
                        metadata={
                            "role": "agent_y_sequential_counterproposal",
                            "stage": "xy_sequential_material_counterproposal",
                            "iteration": iteration,
                            "description_attempt": description_attempt,
                            "cycle": cycle,
                        },
                        json_repair_attempts=args.json_repair_attempts,
                    )
                    counterproposal = payload_with_controller_iteration(counterproposal, iteration=iteration)
                    xy_dialogue.append({"role": "Y", "mode": "material_description_counterproposal", "cycle": cycle, "payload": counterproposal})
                    reverse_review = call_json_object(
                        x_client,
                        system=AGENT_X_SEQUENTIAL_SYSTEM,
                        user=prompt_x_sequential_material_reverse_review(
                            context,
                            previous_proposal,
                            counterproposal,
                            iteration=iteration,
                            cycle=cycle,
                            return_feedback=return_feedback,
                            template_only=args.generator_template_only,
                        ),
                        role="agent_x_sequential_reverse_review",
                        metadata={
                            "role": "agent_x_sequential_reverse_review",
                            "stage": "xy_sequential_material_reverse_review",
                            "iteration": iteration,
                            "description_attempt": description_attempt,
                            "cycle": cycle,
                        },
                        json_repair_attempts=args.json_repair_attempts,
                    )
                    reverse_review = payload_with_controller_iteration(reverse_review, iteration=iteration)
                    reverse_review = enforce_no_valid_strategy_audit_guard(reverse_review, counterproposal, context)
                    reverse_review = enforce_exhausted_strategy_route_audit_guard(reverse_review, counterproposal, context)
                    reverse_review = enforce_latest_strategy_selection_order_guard(
                        reverse_review,
                        counterproposal,
                        context,
                        return_feedback,
                    )
                    reverse_review = payload_with_controller_iteration(reverse_review, iteration=iteration)
                    xy_dialogue.append({"role": "X", "mode": "material_description_reverse_review", "cycle": cycle, "payload": reverse_review})
                    if sequential_no_valid_description_agrees_with_context(
                        reverse_review,
                        counterproposal,
                        context,
                    ):
                        material_consensus = no_valid_material_consensus_from_payload(
                            counterproposal,
                            reverse_review,
                            iteration=iteration,
                        )
                        xy_dialogue.append(
                            {
                                "role": "Y",
                                "mode": "no_valid_material_reverse_consensus",
                                "cycle": cycle,
                                "payload": material_consensus,
                            }
                        )
                        break
                    if sequential_description_agrees(reverse_review, counterproposal):
                        if cycle < min_rounds:
                            xy_dialogue.append(
                                {
                                    "role": "controller",
                                    "mode": "xy_early_clean_reverse_approval",
                                    "cycle": cycle,
                                    "payload": {
                                        "reason": (
                                            "X explicitly approved Y's counterproposal; controller finalized "
                                            "without forcing extra debate cycles."
                                        ),
                                        "configured_min_debate_rounds": min_rounds,
                                    },
                                }
                            )
                        material_consensus = approved_material_consensus_from_payload(
                            counterproposal,
                            reverse_review,
                            iteration=iteration,
                        )
                        xy_dialogue.append({"role": "Y", "mode": "material_description_reverse_final", "cycle": cycle, "payload": material_consensus})
                        break
                    proposal = counterproposal
                    review = reverse_review
                else:
                    xy_rejection_streak += 1
                if cycle >= max_rounds:
                    break
                proposal = call_json_object(
                    x_client,
                    system=AGENT_X_SEQUENTIAL_SYSTEM,
                    user=prompt_x_sequential_material_revision(
                        context,
                        proposal,
                        review,
                        iteration=iteration,
                        cycle=cycle + 1,
                        return_feedback=return_feedback,
                        template_only=args.generator_template_only,
                    ),
                    role="agent_x_sequential",
                    metadata={
                        "role": "agent_x_sequential",
                        "stage": "xy_sequential_material_revision",
                        "iteration": iteration,
                        "description_attempt": description_attempt,
                        "cycle": cycle + 1,
                    },
                    json_repair_attempts=args.json_repair_attempts,
                )
                proposal = payload_with_controller_iteration(proposal, iteration=iteration)
                xy_dialogue.append({"role": "X", "mode": "material_description_revision", "cycle": cycle + 1, "payload": proposal})
            if material_consensus is None:
                return_feedback = {
                    "reason": "X/Y did not reach a one-material description consensus within max_debate_rounds.",
                    "last_review": review,
                }
                continue
            if material_payload_declares_no_valid_description(material_consensus):
                return_feedback = {
                    "reason": "X/Y certified that no allowed non-duplicate material exists under the current strategy constraints.",
                    "xy_no_valid_consensus": material_consensus,
                }
                break
            strategy_order_errors = latest_strategy_selection_order_errors(
                material_consensus,
                context,
                return_feedback,
            )
            if strategy_order_errors:
                return_feedback = {
                    "reason": "Controller rejected X/Y material consensus because it violated the ordered latest next_strategy candidate queue.",
                    "latest_strategy_selection_errors": strategy_order_errors,
                    "material_consensus_summary": compact_payload_for_dialogue(material_consensus),
                    "latest_xy_strategy_constraints": context.get("latest_xy_strategy_constraints"),
                }
                xy_dialogue.append(
                    {
                        "role": "controller",
                        "mode": "latest_strategy_order_feedback",
                        "cycle": description_attempt,
                        "payload": return_feedback,
                    }
                )
                material_consensus = None
                continue
            queue_errors = validate_xy_sun_candidate_queue(
                material_consensus,
                context=context,
                forbidden_reduced_formulas=forbidden_history_formulas,
            )
            if queue_errors:
                return_feedback = {
                    "reason": "Controller rejected X/Y output because sequential SUN optimization now requires a ranked candidate queue before selecting one material.",
                    "sun_candidate_queue_errors": queue_errors,
                    "queue_size_target": xy_sun_candidate_queue_size_from_context(context),
                    "search_policy": _context_search_policy(context),
                    "material_consensus_summary": compact_payload_for_dialogue(material_consensus),
                    "required_fix": (
                        "Return sun_candidate_queue with at least two concrete non-duplicate candidates, "
                        "selected_candidate_id, selection_rationale, acquisition_mode matching current_search_mode "
                        "(in exploit mode, legal cooldown escapes that preserve the mechanism are still labeled exploit), "
                        "and a top-level material_description copied from the selected queue item."
                    ),
                }
                xy_dialogue.append(
                    {
                        "role": "controller",
                        "mode": "xy_sun_candidate_queue_feedback",
                        "cycle": description_attempt,
                        "payload": return_feedback,
                    }
                )
                material_consensus = None
                continue
            if args.generator_template_only:
                memory_snapshot = memory_with_transient_materialization_errors(
                    read_sequential_memory(memory_path),
                    iteration_errors,
                )
                template_description_errors = validate_template_only_material_description(
                    material_consensus,
                    forbidden_reduced_formulas=failed_or_used_formulas_from_memory(memory_snapshot),
                    forbidden_volume_boundaries=failed_volume_boundaries_from_memory(memory_snapshot),
                    forbidden_evaluator_null_elements=evaluator_null_elements_from_memory(memory_snapshot),
                )
                if template_description_errors:
                    return_feedback = {
                        "reason": "Controller rejected X/Y material description because template-only mode requires an allowed generator template and complete role mapping.",
                        "template_only_errors": template_description_errors,
                        "template_only_rules": generator_template_only_rules(),
                        "material_consensus_summary": compact_payload_for_dialogue(material_consensus),
                    }
                    xy_dialogue.append(
                        {
                            "role": "controller",
                            "mode": "template_only_material_description_feedback",
                            "cycle": description_attempt,
                            "payload": return_feedback,
                        }
                    )
                    material_consensus = None
                    continue

            z_feedback: dict[str, Any] | None = None
            max_zw_generator_repairs = (
                args.max_zw_generator_repair_rounds
                if args.max_zw_generator_repair_rounds is not None
                else args.max_materialization_repair_rounds
            )
            zw_rejection_streak = 0
            for repair_round in range(0, max(0, int(max_zw_generator_repairs)) + 1):
                try:
                    z_proposal = call_json_object(
                        z_client,
                        system=AGENT_Z_SEQUENTIAL_SYSTEM,
                        user=prompt_z_sequential_candidate(
                            context,
                            material_consensus,
                            iteration=iteration,
                            feedback=z_feedback,
                            template_only=args.generator_template_only,
                        ),
                        role="agent_z_sequential",
                        metadata={
                            "role": "agent_z_sequential",
                            "stage": "zw_sequential_candidate_proposal"
                            if repair_round == 0
                            else "zw_sequential_candidate_repair",
                            "iteration": iteration,
                            "description_attempt": description_attempt,
                            "repair_round": repair_round,
                        },
                        json_repair_attempts=args.json_repair_attempts,
                    )
                except JSONOutputRepairFailure as exc:
                    z_feedback = json_output_repair_feedback(
                        exc=exc,
                        iteration=iteration,
                        description_attempt=description_attempt,
                        repair_round=repair_round,
                        template_only=args.generator_template_only,
                        backend=args.generator_backend,
                    )
                    write_json(iter_dir / f"zw_generator_repair_feedback_round_{repair_round:02d}.json", z_feedback)
                    zw_dialogue.append(
                        {
                            "role": "controller",
                            "mode": "z_json_parse_feedback",
                            "cycle": repair_round,
                            "payload": z_feedback,
                        }
                    )
                    iteration_errors.append(
                        f"agent_z_sequential invalid JSON at repair_round {repair_round}: {exc.error}"
                    )
                    continue
                z_proposal = payload_with_controller_iteration(z_proposal, iteration=iteration)
                zw_dialogue.append({"role": "Z", "mode": "candidate_proposal" if repair_round == 0 else "candidate_repair", "cycle": repair_round, "payload": z_proposal})
                w_review = call_json_object(
                    w_client,
                    system=AGENT_W_SEQUENTIAL_SYSTEM,
                    user=prompt_w_sequential_candidate_review(
                        context,
                        material_consensus,
                        z_proposal,
                        iteration=iteration,
                        cycle=repair_round + 1,
                        feedback=z_feedback,
                        template_only=args.generator_template_only,
                    ),
                    role="agent_w_sequential",
                    metadata={
                        "role": "agent_w_sequential",
                        "stage": "zw_sequential_candidate_review",
                        "iteration": iteration,
                        "description_attempt": description_attempt,
                        "repair_round": repair_round,
                    },
                    json_repair_attempts=args.json_repair_attempts,
                )
                w_review = payload_with_controller_iteration(w_review, iteration=iteration)
                zw_dialogue.append({"role": "W", "mode": "candidate_review", "cycle": repair_round, "payload": w_review})
                accepted_proposal: dict[str, Any] | None = None
                accepted_review: dict[str, Any] | None = None
                z_declared_no_faithful_candidate = candidate_payload_declares_no_faithful_generator_candidate(z_proposal)
                if bool(w_review.get("return_to_xy")) is True:
                    if z_declared_no_faithful_candidate:
                        return_feedback = {
                            "reason": "Z/W returned the material description to X/Y as infeasible or too vague.",
                            "w_review": w_review,
                            "z_proposal": z_proposal,
                            "last_generator_repair_feedback": z_feedback,
                            "materialization_errors": iteration_errors[-12:],
                        }
                        material_consensus = None
                        break
                    z_feedback = {
                        "source": "agent_w_review_return_to_xy_overridden",
                        "controller_feedback": (
                            "W requested return_to_xy, but Z/W must repair executable generator/schema issues themselves "
                            "unless Z declares the description unrepresentable and provides no candidate_specs."
                        ),
                        "w_review": w_review,
                        "last_z_proposal_summary": compact_payload_for_dialogue(z_proposal),
                        "executable_generator_rules": generator_executable_schema_rules(
                            template_only=args.generator_template_only,
                            backend=args.generator_backend,
                        ),
                    }
                    zw_dialogue.append({"role": "controller", "mode": "candidate_review_feedback", "cycle": repair_round, "payload": z_feedback})
                if sequential_candidate_agrees(w_review, z_proposal):
                    accepted_proposal = z_proposal
                    accepted_review = w_review
                    zw_rejection_streak = 0
                elif review_requires_counterproposal(
                    w_review,
                    rejection_streak=zw_rejection_streak,
                    threshold=args.critic_counterproposal_after,
                ):
                    zw_rejection_streak = 0
                    counterproposal = call_json_object(
                        w_client,
                        system=AGENT_W_SEQUENTIAL_SYSTEM,
                        user=prompt_w_sequential_candidate_counterproposal(
                            context,
                            material_consensus,
                            z_proposal,
                            w_review,
                            iteration=iteration,
                            cycle=repair_round + 1,
                            feedback=z_feedback,
                            template_only=args.generator_template_only,
                        ),
                        role="agent_w_sequential_counterproposal",
                        metadata={
                            "role": "agent_w_sequential_counterproposal",
                            "stage": "zw_sequential_candidate_counterproposal",
                            "iteration": iteration,
                            "description_attempt": description_attempt,
                            "repair_round": repair_round,
                        },
                        json_repair_attempts=args.json_repair_attempts,
                    )
                    counterproposal = payload_with_controller_iteration(counterproposal, iteration=iteration)
                    zw_dialogue.append({"role": "W", "mode": "candidate_counterproposal", "cycle": repair_round, "payload": counterproposal})
                    reverse_review = call_json_object(
                        z_client,
                        system=AGENT_Z_SEQUENTIAL_SYSTEM,
                        user=prompt_z_sequential_candidate_reverse_review(
                            context,
                            material_consensus,
                            z_proposal,
                            counterproposal,
                            iteration=iteration,
                            cycle=repair_round + 1,
                            feedback=z_feedback,
                            template_only=args.generator_template_only,
                        ),
                        role="agent_z_sequential_reverse_review",
                        metadata={
                            "role": "agent_z_sequential_reverse_review",
                            "stage": "zw_sequential_candidate_reverse_review",
                            "iteration": iteration,
                            "description_attempt": description_attempt,
                            "repair_round": repair_round,
                        },
                        json_repair_attempts=args.json_repair_attempts,
                    )
                    reverse_review = payload_with_controller_iteration(reverse_review, iteration=iteration)
                    zw_dialogue.append({"role": "Z", "mode": "candidate_reverse_review", "cycle": repair_round, "payload": reverse_review})
                    if bool(reverse_review.get("return_to_xy")) is True and not candidate_specs_from_payload(counterproposal):
                        return_feedback = {
                            "reason": "Z/W agreed that the material description is infeasible for the generator.",
                            "w_counterproposal": counterproposal,
                            "z_reverse_review": reverse_review,
                            "original_z_proposal": z_proposal,
                            "last_generator_repair_feedback": z_feedback,
                            "materialization_errors": iteration_errors[-12:],
                        }
                        material_consensus = None
                        break
                    counterproposal_declared_no_faithful_candidate = (
                        candidate_payload_declares_no_faithful_generator_candidate(counterproposal)
                    )
                    if counterproposal_declared_no_faithful_candidate and (
                        z_declared_no_faithful_candidate or _bool_value(reverse_review.get("agree")) is True
                    ):
                        return_feedback = {
                            "reason": (
                                "Z and W both declared that no non-repeating faithful generator candidate remains; "
                                "returning the material description to X/Y."
                            ),
                            "w_counterproposal": counterproposal,
                            "z_reverse_review": reverse_review,
                            "original_z_proposal": z_proposal,
                            "last_generator_repair_feedback": z_feedback,
                            "materialization_errors": iteration_errors[-12:],
                        }
                        material_consensus = None
                        break
                    if sequential_candidate_agrees(reverse_review, counterproposal):
                        accepted_proposal = counterproposal
                        accepted_review = reverse_review
                    else:
                        z_feedback = {
                            "source": "agent_z_reverse_review",
                            "w_counterproposal_summary": compact_payload_for_dialogue(counterproposal),
                            "z_reverse_review": reverse_review,
                            "last_z_proposal_summary": compact_payload_for_dialogue(z_proposal),
                            "executable_generator_rules": generator_executable_schema_rules(
                                template_only=args.generator_template_only,
                                backend=args.generator_backend,
                            ),
                        }
                        zw_dialogue.append({"role": "controller", "mode": "candidate_reverse_review_feedback", "cycle": repair_round, "payload": z_feedback})
                        continue
                else:
                    zw_rejection_streak += 1
                    z_feedback = {
                        "source": "agent_w_review",
                        "w_review": w_review,
                        "last_z_proposal_summary": compact_payload_for_dialogue(z_proposal),
                        "executable_generator_rules": generator_executable_schema_rules(
                            template_only=args.generator_template_only,
                            backend=args.generator_backend,
                        ),
                    }
                    continue
                if accepted_proposal is None or accepted_review is None:
                    continue
                candidate_consensus = call_json_object(
                    w_client,
                    system=AGENT_W_SEQUENTIAL_SYSTEM,
                    user=prompt_w_sequential_candidate_final(
                        context,
                        material_consensus,
                        accepted_proposal,
                        accepted_review,
                        iteration=iteration,
                        feedback=z_feedback,
                        template_only=args.generator_template_only,
                    ),
                    role="zw_candidate_consensus",
                    metadata={
                        "role": "zw_candidate_consensus",
                        "stage": "zw_sequential_candidate_final",
                        "iteration": iteration,
                        "description_attempt": description_attempt,
                        "repair_round": repair_round,
                    },
                    json_repair_attempts=args.json_repair_attempts,
                )
                candidate_consensus = payload_with_controller_iteration(candidate_consensus, iteration=iteration)
                zw_dialogue.append({"role": "W", "mode": "candidate_final", "cycle": repair_round, "payload": candidate_consensus})
                specs = candidate_specs_from_payload(candidate_consensus)
                pre_materialization_errors: list[str] = []
                memory_for_preflight: dict[str, Any] | None = None
                preflight_strategy_cooldowns: dict[str, Any] = {}
                if len(specs) != 1:
                    pre_materialization_errors.append(
                        f"candidate_consensus.agreed_candidate_specs must contain exactly one candidate object, got {len(specs)}"
                    )
                if not pre_materialization_errors and args.generator_backend == "mattergen":
                    memory_for_preflight = read_sequential_memory(memory_path)
                    transient_memory_for_preflight = memory_with_transient_materialization_errors(
                        memory_for_preflight,
                        iteration_errors,
                    )
                    preflight_strategy_cooldowns = mattergen_strategy_cooldowns_from_memory(
                        transient_memory_for_preflight
                    )
                    pre_materialization_errors.extend(
                        mattergen_strategy_cooldown_preflight_errors(specs, preflight_strategy_cooldowns)
                    )
                    preflight_excluded_formulas = failed_or_used_formulas_from_memory(transient_memory_for_preflight)
                    if known_formulas:
                        preflight_excluded_formulas.update(known_formulas)
                    pre_materialization_errors.extend(
                        mattergen_excluded_target_preflight_errors(specs, preflight_excluded_formulas)
                    )
                if pre_materialization_errors:
                    selected_records, locked_specs, errors = [], [], pre_materialization_errors
                else:
                    memory = memory_for_preflight or read_sequential_memory(memory_path)
                    used_formulas = used_formulas_from_memory(memory)
                    materialization_known_formulas = set(known_formulas or set())
                    materialization_known_formulas.update(used_formulas)
                    materialization_mattergen_config = _mattergen_config_with_xy_density_cap(
                        mattergen_config_from_args(
                            args,
                            root,
                            iter_dir / "mattergen_materialization" / f"repair_{repair_round:02d}",
                        ),
                        material_consensus,
                    )
                    selected_records, locked_specs, errors = materialize_candidate_specs(
                        pool_records,
                        specs,
                        target_count=sequential_batch_target,
                        seed=args.seed_base + iteration * 1000 + repair_round,
                        max_sites=args.max_sites,
                        known_formulas=materialization_known_formulas or None,
                        allowed_sources=allowed_candidate_sources(args),
                        max_per_reduced_formula=args.max_per_reduced_formula,
                        require_design_rule_ids=False,
                        allow_structure_dicts=not args.generator_template_only,
                        mattergen_config=materialization_mattergen_config,
                        additional_excluded_formulas=failed_or_used_formulas_from_memory(memory),
                    )
                iteration_errors.extend(errors)
                memory = read_sequential_memory(memory_path)
                used_formulas = used_formulas_from_memory(memory)
                if selected_records:
                    unique_records: list[dict[str, Any]] = []
                    seen_batch_formulas: set[str] = set()
                    for record_item in selected_records:
                        formula = _normalize_formula_text(record_item.get("formula"))
                        if formula and formula in used_formulas:
                            duplicate_error = (
                                f"iteration {iteration}: duplicate reduced_formula {formula} already exists in sequential memory"
                            )
                            iteration_errors.append(duplicate_error)
                            errors.append(duplicate_error)
                            continue
                        if formula and formula in seen_batch_formulas:
                            duplicate_error = f"iteration {iteration}: duplicate reduced_formula {formula} repeated inside batch"
                            iteration_errors.append(duplicate_error)
                            errors.append(duplicate_error)
                            continue
                        if formula:
                            seen_batch_formulas.add(formula)
                        unique_records.append(record_item)
                    selected_records = unique_records
                if selected_records:
                    if locked_specs:
                        executable_generator_rule = executable_generator_rule_from_locked_spec(
                            locked_specs[0],
                            selected_records[0],
                            repair_round=repair_round,
                            preceding_errors=iteration_errors,
                            template_only=args.generator_template_only,
                            backend=args.generator_backend,
                        )
                        write_json(iter_dir / "executable_generator_rule.json", executable_generator_rule)
                    break
                z_feedback = sequential_generator_repair_feedback(
                    iteration=iteration,
                    description_attempt=description_attempt,
                    repair_round=repair_round,
                    errors=iteration_errors[-20:] or errors,
                    z_proposal=accepted_proposal or z_proposal,
                    w_review=accepted_review or w_review,
                    candidate_consensus=candidate_consensus,
                    template_only=args.generator_template_only,
                    backend=args.generator_backend,
                    strategy_cooldowns=mattergen_strategy_cooldowns_from_memory(
                        memory_with_transient_materialization_errors(
                            read_sequential_memory(memory_path), iteration_errors[-20:] or errors
                        )
                    )
                    if args.generator_backend == "mattergen"
                    else None,
                )
                write_json(iter_dir / f"zw_generator_repair_feedback_round_{repair_round:02d}.json", z_feedback)
                zw_dialogue.append({"role": "controller", "mode": "generator_materialization_feedback", "cycle": repair_round, "payload": z_feedback})
                if mattergen_feedback_requires_xy_strategy_revision(z_feedback):
                    return_feedback = {
                        "reason": (
                            "Controller classified the MatterGen request as a saturated chemical basin: no structures were "
                            "accepted and excluded_reduced_formula dominated the rejection reasons after historical duplicate "
                            "exclusions were applied."
                        ),
                        "required_fix": (
                            "X/Y must select a different chemical system, mechanism route, or ranked queue item outside the "
                            "blocked basin. Do not ask Z/W to repair this by only increasing batch_size/num_batches, changing "
                            "diffusion_guidance_factor, or removing visible exclude_reduced_formulas; the controller will "
                            "reapply historical exclusions."
                        ),
                        "mattergen_saturation": z_feedback.get("mattergen_saturation"),
                        "material_consensus_summary": compact_payload_for_dialogue(material_consensus),
                        "zw_dialogue_tail": compact_dialogue(zw_dialogue[-6:]),
                        "materialization_errors": iteration_errors[-12:],
                    }
                    xy_dialogue.append(
                        {
                            "role": "controller",
                            "mode": "mattergen_saturated_basin_feedback",
                            "cycle": description_attempt,
                            "payload": return_feedback,
                        }
                    )
                    material_consensus = None
                    break
            if selected_records:
                break
            if material_consensus is None:
                continue
            return_feedback = {
                "reason": "Z/W did not produce one materializable candidate within generator repair limits.",
                "material_consensus": material_consensus,
                "zw_dialogue_tail": compact_dialogue(zw_dialogue[-6:]),
                "materialization_errors": iteration_errors[-12:],
                "last_generator_repair_feedback": z_feedback,
            }
            material_consensus = None

        strategy_blocked = bool(material_consensus) and material_payload_declares_no_valid_description(material_consensus)
        memory_selected_records = [compact_selected_record_for_memory(item) for item in selected_records]
        record: dict[str, Any] = {
            "iteration": iteration,
            "created_at_utc": utc_now(),
            "status": "materialized" if selected_records else ("strategy_blocked" if strategy_blocked else "not_materialized"),
            "material_description": material_description_from_payload(material_consensus or {}),
            "xy_material_consensus": material_consensus,
            "candidate_consensus": candidate_consensus,
            "candidate_spec": locked_specs[0] if locked_specs else (candidate_specs_from_payload(candidate_consensus or {})[0] if candidate_specs_from_payload(candidate_consensus or {}) else {}),
            "candidate_specs": locked_specs if locked_specs else candidate_specs_from_payload(candidate_consensus or {}),
            "selected_record": memory_selected_records[0] if memory_selected_records else {},
            "selected_records": memory_selected_records,
            "executable_generator_rule": executable_generator_rule or {},
            "materialization_errors": iteration_errors,
            "xy_dialogue": compact_dialogue(xy_dialogue),
            "zw_dialogue": compact_dialogue(zw_dialogue),
        }

        if selected_records:
            input_path = iter_dir / "input.json"
            write_input_structures(selected_records, input_path)
            write_json(iter_dir / "selected_records.json", selected_records)
            write_json(iter_dir / "candidate_specs.locked.json", locked_specs)
            if not args.skip_evaluation:
                results_path = iter_dir / "results.json"
                eval_log = iter_dir / "eval" / "full_cpu_0.out"
                run_evaluator(
                    root=root,
                    round_dir=iter_dir,
                    input_path=input_path,
                    results_path=results_path,
                    training_data=training_data,
                    ppd_path=ppd_path,
                    eval_log=eval_log,
                    args=args,
                )
                run_analysis(root=root, round_dir=iter_dir, input_path=input_path, results_path=results_path, eval_log=eval_log)
                record["evaluation_summary"] = summarize_evaluation(iter_dir)
                record["evaluation_result"] = summarize_single_evaluation(iter_dir, selected_records[0])
                record["evaluation_results"] = summarize_batch_evaluation(iter_dir, selected_records)
                record["status"] = "evaluated"
            else:
                record["status"] = "materialized_skip_evaluation"
        else:
            record["failure_reason"] = (return_feedback or {}).get("reason") or "No materializable candidate produced."

        postmortem_context = sequential_context_payload(
            state=state,
            mode=args.mode,
            iteration=iteration,
            memory_path=memory_path,
            candidate_source=args.candidate_source,
            seed=args.seed_base + iteration,
            generator_template_only=args.generator_template_only,
            generator_backend=args.generator_backend,
            xy_sun_candidate_queue_size=args.xy_sun_candidate_queue_size,
            sequential_materialization_target_count=sequential_batch_target,
        )
        try:
            x_postmortem = call_json_object(
                x_client,
                system=AGENT_X_SEQUENTIAL_SYSTEM,
                user=prompt_x_sequential_postmortem(postmortem_context, record, iteration=iteration),
                role="agent_x_sequential_postmortem",
                metadata={"role": "agent_x_sequential_postmortem", "stage": "xy_sequential_postmortem", "iteration": iteration},
                json_repair_attempts=args.json_repair_attempts,
            )
            y_postmortem = call_json_object(
                y_client,
                system=AGENT_Y_SEQUENTIAL_SYSTEM,
                user=prompt_y_sequential_postmortem_review(postmortem_context, record, x_postmortem, iteration=iteration),
                role="agent_y_sequential_postmortem",
                metadata={"role": "agent_y_sequential_postmortem", "stage": "xy_sequential_postmortem_review", "iteration": iteration},
                json_repair_attempts=args.json_repair_attempts,
            )
            postmortem_forbidden_formulas = failed_or_used_formulas_from_memory(read_sequential_memory(memory_path))
            postmortem_forbidden_formulas.update(formulas_from_iteration_record(record))
            y_postmortem = enforce_postmortem_next_strategy_guard(
                y_postmortem,
                forbidden_reduced_formulas=postmortem_forbidden_formulas,
            )
            record["xy_postmortem"] = y_postmortem
        except Exception as exc:
            record["xy_postmortem_error"] = f"{type(exc).__name__}: {exc}"
            record["xy_postmortem"] = sequential_fallback_postmortem(
                record,
                iteration=iteration,
                reason=record["xy_postmortem_error"],
            )

        write_json(iter_dir / "iteration_record.json", record)
        memory = append_sequential_memory(memory_path, record)
        materialization_errors_all.extend(iteration_errors)
        result = record.get("evaluation_result") if isinstance(record.get("evaluation_result"), Mapping) else {}
        batch_result = record.get("evaluation_results") if isinstance(record.get("evaluation_results"), Mapping) else {}
        print(
            f"[{utc_now()}] xy_sequential_iter_done iteration={iteration} status={record.get('status')} "
            f"formula={result.get('formula') or (record.get('selected_record') or {}).get('formula')} "
            f"e_hull={result.get('e_hull')} sun={result.get('is_sun')} "
            f"batch_evaluated={batch_result.get('evaluated_count')} batch_sun={batch_result.get('sun_count')} "
            f"memory_records={len(memory.get('records', []))}",
            flush=True,
        )

    memory = read_sequential_memory(memory_path)
    report = build_sequential_report(args=args, state=state, memory=memory, materialization_errors=materialization_errors_all)
    write_json(work_dir / "report.json", report)
    write_markdown_report(work_dir / "report.md", report)
    print(
        f"[{utc_now()}] xy_sequential_done evaluated={report.get('evaluated_count')} sun={report.get('sun_count')} ratio={report.get('sun_ratio')} work_dir={work_dir}",
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.generation_protocol != "sequential_single" and args.candidate_count is None:
        args.candidate_count = DEFAULT_CANDIDATE_COUNT
    root = Path(args.root).resolve()
    state_path = (root / args.state).resolve()
    candidate_pool_path = (root / args.candidate_pool).resolve()
    if args.work_dir:
        work_dir = (root / args.work_dir).resolve()
    else:
        work_dir = (root / "xy_runs" / timestamp_id()).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    state = read_json(state_path, {})
    if not isinstance(state, Mapping):
        raise ValueError(f"{state_path} must contain a JSON object")
    if args.candidate_source == "generator":
        pool_records = []
        pool_summary = {
            "count": 0,
            "available_to_xy": False,
            "reason": "MP-pool loading skipped for generator-only X/Y candidate design.",
        }
    else:
        pool_records = load_candidate_pool(candidate_pool_path)
        pool_summary = pool_digest(pool_records)
    training_data = Path(args.training_data).resolve() if args.training_data else default_training_data(root)
    ppd_path = Path(args.ppd_path).resolve() if args.ppd_path else default_ppd_path(root)
    known_formulas = load_known_formulas(str(training_data))

    print(
        f"[{utc_now()}] xy_start mode={args.mode} generation_protocol={args.generation_protocol} candidate_source={args.candidate_source} work_dir={work_dir} pool_count={len(pool_records)} "
        f"candidate_count={args.candidate_count} shards={args.shards} max_debate_rounds={args.max_debate_rounds}",
        flush=True,
    )
    write_json(
        work_dir / "config.json",
        {
            "args": {key: value for key, value in vars(args).items() if "key" not in key.lower()},
            "root": str(root),
            "state_path": str(state_path),
            "candidate_pool_path": str(candidate_pool_path),
            "command": command_text([sys.executable, "-m", "crystal_llm.run_xy_experience_debate", *(argv or [])]),
        },
    )

    if args.generation_protocol == "sequential_single":
        try:
            return run_sequential_single_optimizer(
                args=args,
                root=root,
                work_dir=work_dir,
                state_path=state_path,
                candidate_pool_path=candidate_pool_path,
                state=state,
                training_data=training_data,
                ppd_path=ppd_path,
                known_formulas=known_formulas,
            )
        except RecoverableLLMFailure as exc:
            memory_path = work_dir / "sequential_memory.json"
            pause = recoverable_llm_pause_payload(
                exc=exc,
                args=args,
                work_dir=work_dir,
                memory_path=memory_path,
            )
            write_json(work_dir / "recoverable_pause.json", pause)
            iteration = exc.metadata.get("iteration")
            if isinstance(iteration, int):
                iter_dir = work_dir / f"iter_{iteration:03d}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                write_json(iter_dir / "recoverable_pause.json", pause)
            print(
                f"[{utc_now()}] xy_recoverable_llm_pause role={exc.role} "
                f"stage={exc.metadata.get('stage')} iteration={exc.metadata.get('iteration')} "
                f"exit_code={RECOVERABLE_LLM_EXIT_CODE} error={exc.error}",
                flush=True,
            )
            return RECOVERABLE_LLM_EXIT_CODE

    debates_by_shard: dict[int, dict[str, Any]] = {}

    def _shard_id(debate: Mapping[str, Any]) -> int:
        try:
            return int(debate.get("shard_id") or 0)
        except Exception:
            return 0

    def _sorted_debates() -> list[dict[str, Any]]:
        return [debates_by_shard[key] for key in sorted(debates_by_shard)]

    def _write_debate_snapshot() -> list[dict[str, Any]]:
        debates = _sorted_debates()
        write_json(work_dir / "xy_debates.json", debates)
        return debates

    def _materialize_snapshot() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        debates = _write_debate_snapshot()
        raw_specs: list[dict[str, Any]] = []
        for debate in debates:
            raw_specs.extend(candidate_specs_from_payload(debate))
        write_json(work_dir / "candidate_specs.raw.json", raw_specs)
        selected, locked, errors = materialize_candidate_specs(
            pool_records,
            raw_specs,
            target_count=args.candidate_count,
            seed=args.seed_base,
            max_sites=args.max_sites,
            known_formulas=known_formulas,
            allowed_sources=allowed_candidate_sources(args),
            max_per_reduced_formula=args.max_per_reduced_formula,
            require_design_rule_ids=require_design_rule_ids_for_args(args),
            allow_structure_dicts=not args.generator_template_only,
            mattergen_config=mattergen_config_from_args(args, root, work_dir / "mattergen_materialization"),
        )
        return debates, raw_specs, selected, locked, errors

    def _run_batch(targets: Sequence[int], *, start_index: int, batch_label: str) -> None:
        if not targets:
            return
        pending: list[tuple[int, int]] = []
        for offset, target in enumerate(targets):
            index = start_index + offset
            shard_path = work_dir / "debates" / f"shard_{index:03d}" / "xy_debate.json"
            if args.resume_existing_shards and shard_path.exists():
                existing = read_json(shard_path, {})
                if isinstance(existing, Mapping):
                    debate = dict(existing)
                    debates_by_shard[_shard_id(debate) or index] = debate
                    print(
                        f"[{utc_now()}] xy_shard_reused batch={batch_label} shard={index} "
                        f"status={debate.get('status')} candidates={len(candidate_specs_from_payload(debate))}",
                        flush=True,
                    )
                    continue
            if (
                args.resume_existing_shards
                and args.skip_missing_initial_shards_on_resume
                and batch_label == "initial"
            ):
                debate = {
                    "status": "unresolved",
                    "agent": "controller",
                    "shard_id": index,
                    "reason": "Skipped missing initial shard during resume so dynamic backfill can fill the global candidate deficit.",
                    "agreed_candidate_specs": [],
                    "dialogue": [],
                }
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(shard_path, debate)
                debates_by_shard[index] = debate
                print(
                    f"[{utc_now()}] xy_shard_skipped batch={batch_label} shard={index} "
                    "status=unresolved candidates=0",
                    flush=True,
                )
                continue
            pending.append((index, target))
        if not pending:
            _write_debate_snapshot()
            return
        worker_count = args.parallel_workers or min(len(pending), max(1, len(pending)))
        with ThreadPoolExecutor(max_workers=max(1, min(worker_count, len(pending)))) as executor:
            futures = [
                executor.submit(
                    run_shard_debate,
                    args=args,
                    root=root,
                    work_dir=work_dir,
                    state_path=state_path,
                    candidate_pool_path=candidate_pool_path,
                    state=state,
                    pool_summary=pool_summary,
                    pool_records=pool_records,
                    known_formulas=known_formulas,
                    shard_index=index,
                    shard_target=target,
                    total_target=args.candidate_count,
                )
                for index, target in pending
            ]
            for future in as_completed(futures):
                debate = future.result()
                debates_by_shard[_shard_id(debate)] = debate
                _write_debate_snapshot()
                print(
                    f"[{utc_now()}] xy_shard_done batch={batch_label} shard={debate.get('shard_id')} "
                    f"status={debate.get('status')} candidates={len(candidate_specs_from_payload(debate))}",
                    flush=True,
                )

    shard_targets = split_shards(args.candidate_count, args.oversample, args.shards)
    _run_batch(shard_targets, start_index=1, batch_label="initial")
    debates, raw_specs, selected_records, locked_specs, materialization_errors = _materialize_snapshot()

    for backfill_batch in range(1, max(0, int(args.max_backfill_batches)) + 1):
        if len(selected_records) >= args.candidate_count:
            break
        deficit = args.candidate_count - len(selected_records)
        backfill_targets = split_shards(
            deficit,
            max(1.0, float(args.backfill_oversample)),
            max(1, int(args.backfill_shards)),
        )
        start_index = (max(debates_by_shard) if debates_by_shard else 0) + 1
        print(
            f"[{utc_now()}] xy_backfill_start batch={backfill_batch} deficit={deficit} "
            f"shards={len(backfill_targets)} requested_specs={sum(backfill_targets)} start_shard={start_index}",
            flush=True,
        )
        _run_batch(backfill_targets, start_index=start_index, batch_label=f"backfill_{backfill_batch}")
        debates, raw_specs, selected_records, locked_specs, materialization_errors = _materialize_snapshot()
        print(
            f"[{utc_now()}] xy_backfill_done batch={backfill_batch} locked={len(selected_records)} "
            f"errors={len(materialization_errors)}",
            flush=True,
        )

    write_json(work_dir / "candidate_specs.locked.json", locked_specs)
    write_json(work_dir / "selected_records.json", selected_records)
    write_json(work_dir / "materialization_errors.json", list(materialization_errors))
    input_path = work_dir / "input.json"
    write_input_structures(selected_records, input_path)
    print(
        f"[{utc_now()}] xy_locked candidates={len(selected_records)} errors={len(materialization_errors)} input={input_path}",
        flush=True,
    )
    if len(selected_records) < args.candidate_count:
        print(
            f"[{utc_now()}] xy_warning locked fewer candidates than requested: {len(selected_records)} < {args.candidate_count}",
            flush=True,
        )

    evaluation_summary: dict[str, Any] = {}
    if not args.skip_evaluation and selected_records:
        eval_dir = work_dir / "eval"
        results_path = work_dir / "results.json"
        eval_log = eval_dir / "full_cpu_0.out"
        if args.force or not json_ok(results_path) or not eval_log.exists():
            run_evaluator(
                root=root,
                round_dir=work_dir,
                input_path=input_path,
                results_path=results_path,
                training_data=training_data,
                ppd_path=ppd_path,
                eval_log=eval_log,
                args=args,
            )
        run_analysis(root=root, round_dir=work_dir, input_path=input_path, results_path=results_path, eval_log=eval_log)
        evaluation_summary = summarize_evaluation(work_dir)
        print(
            f"[{utc_now()}] xy_evaluated count={evaluation_summary.get('evaluated_count')} "
            f"sun={evaluation_summary.get('sun_count')} ratio={evaluation_summary.get('sun_ratio')}",
            flush=True,
        )

    report = build_locked_report(
        args=args,
        state=state,
        debates=debates,
        locked_specs=locked_specs,
        selected_records=selected_records,
        materialization_errors=materialization_errors,
        evaluation_summary=evaluation_summary,
    )
    write_json(work_dir / "report.json", report)
    write_markdown_report(work_dir / "report.md", report)
    print(f"[{utc_now()}] xy_done work_dir={work_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
