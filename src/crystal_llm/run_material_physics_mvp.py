"""Run the new materials-physics MVP.

This workflow asks the LLMs to:
1. debate mechanism hypotheses about material stability,
2. convert surviving mechanisms into falsifiable predictions,
3. turn predictions into concrete materialization bundles, and
4. evaluate the selected materials with the existing evaluator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import orjson
from pymatgen.core import Composition, Structure

from crystal_llm.filters import load_known_formulas, reduced_formula, validate_structure
from crystal_llm.generate import load_formula_probes
from crystal_llm.hypothesis_schema import schema_reference_json as generator_schema_reference_json
from crystal_llm.material_physics_schema import (
    MATERIAL_PHYSICS_DIRECTIVE,
    MaterializationSelection,
    query_matches,
    schema_reference_json,
    select_matches,
    validate_execution_payload,
    validate_mechanism_payload,
    validate_prediction_payload,
)
from crystal_llm.llm_client import LLMConfig, LLMError, ResponsesClient, extract_json_object
from crystal_llm.local_agent_runtime import LocalAgentRuntime
from crystal_llm.memory import read_json, round_label, write_json


MAX_JSON_REPAIR_ATTEMPTS = 2
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
AMU_PER_A3_TO_G_CM3 = 1.66053906660
DEFAULT_GENERATOR_VOLUME_PER_ATOM_TARGET = 18.0

COMMON_FORMAL_OXIDATION_STATES = {
    "Li": 1,
    "Na": 1,
    "K": 1,
    "Rb": 1,
    "Cs": 1,
    "Be": 2,
    "Mg": 2,
    "Ca": 2,
    "Sr": 2,
    "Ba": 2,
    "Sc": 3,
    "Y": 3,
    "La": 3,
    "Al": 3,
    "Ga": 3,
    "In": 3,
    "Ti": 4,
    "Zr": 4,
    "Hf": 4,
    "O": -2,
    "S": -2,
    "Se": -2,
    "Te": -2,
    "F": -1,
    "Cl": -1,
    "Br": -1,
    "I": -1,
    "N": -3,
    "P": -3,
}

AUDIT_STATUSES = {
    "critique",
    "review",
    "audit",
    "audit_complete",
    "partial_rejection",
    "needs_repair",
    "rejected",
    "reject",
    "reject_partial",
    "accepted_with_revisions",
}


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

NONCONSENSUS_STATUSES = {
    "rejected",
    "reject",
    "unresolved",
    "no_consensus",
    "no_prediction_consensus",
    "no_mechanism_consensus",
    "non_executable",
}

NONFATAL_ROUND_STATUSES = {
    "unresolved_mechanism",
    "unresolved_prediction",
    "unresolved_execution",
}

PREDICTION_DESIGN_INFEASIBLE_STATUS = "prediction_design_infeasible"
PROMPT_MATERIALIZATION_ITEM_LIMIT = 12

DEFAULT_CRITIC_COUNTERPROPOSAL_AFTER = 1
DEFAULT_MECHANISM_MAX_TOKENS = 3072
DEFAULT_PREDICTION_MAX_TOKENS = 4096
DEFAULT_EXECUTION_MAX_TOKENS = 4096
DEFAULT_AGENT_MAX_STEPS = 12
DEFAULT_AGENT_MAX_TOOL_CALLS = 8
PROMPT_JSON_SEPARATORS = (",", ":")
DEFAULT_PRINCIPLE_PROGRAM_MAX_INNER_ROUNDS = 8
DEFAULT_MATERIALIZATION_BACKEND = "mixed"
SUPPORTED_MATTERGEN_FILTER_KEYS = {
    "chemical_system",
    "require_chemical_system_exact",
    "min_sites",
    "max_sites",
    "min_volume_per_atom",
    "max_volume_per_atom",
    "deduplicate_reduced_formula",
    "target_reduced_formula",
    "require_target_reduced_formula",
    "exclude_reduced_formulas",
}
SUPPORTED_MATTERGEN_REQUEST_KEYS = {
    "backend",
    "request_id",
    "checkpoint",
    "model_path",
    "target_count",
    "batch_size",
    "num_batches",
    "diffusion_guidance_factor",
    "properties_to_condition_on",
    "filters",
    "target_reduced_formula",
    "require_target_reduced_formula",
}
SUPPORTED_MATTERGEN_PROPERTY_KEYS = {"chemical_system", "energy_above_hull"}
UNSUPPORTED_MATTERGEN_GATE_HINTS = {
    "stoichiometry_gate",
    "branch_predicate",
    "derived_predicate",
    "post_generation_filter",
    "post_generation_filters",
    "acceptance_predicate",
    "formula_ratio",
    "formula_ratio_gate",
    "oxidation_state_gate",
    "valence_gate",
}
DEFAULT_MATTERGEN_ROOT = os.environ.get("MATTERGEN_ROOT", "external/mattergen")
DEFAULT_MATTERGEN_MODEL_PATH = os.environ.get("MATTERGEN_MODEL_PATH", "external/mattergen/checkpoints/chemical_system_energy_above_hull")
DEFAULT_MATTERGEN_CHECKPOINT = "chemical_system_energy_above_hull"
DEFAULT_MATTERGEN_TARGET_COUNT = 4
DEFAULT_MATTERGEN_BATCH_SIZE = 8
DEFAULT_MATTERGEN_NUM_BATCHES = 2
DEFAULT_MATTERGEN_HARD_TARGET_MIN_NUM_BATCHES = 8
DEFAULT_MATTERGEN_MAX_TARGET_COUNT = 8
DEFAULT_MATTERGEN_MAX_BATCH_SIZE = 16
DEFAULT_MATTERGEN_MAX_NUM_BATCHES = 16
DEFAULT_MATTERGEN_MAX_RAW_SAMPLES = 256
DEFAULT_MATTERGEN_MAX_SITES = 20
DEFAULT_MATTERGEN_MIN_VOLUME_PER_ATOM = 4.0
DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM = 45.0
DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR = 1.0
DEFAULT_MATTERGEN_PARTITION = "8-5090"
DEFAULT_MATTERGEN_GRES = "gpu:1"
DEFAULT_MATTERGEN_CPUS_PER_TASK = 16
DEFAULT_MATTERGEN_MODULE_INIT = ""
DEFAULT_MATTERGEN_MODULES = ""
DEFAULT_MATTERGEN_CUDA_HOME = ""
MATTERGEN_NATIVE_TOOL_GATE_MARKER = "MATTERGEN_NATIVE_NO_MANDATORY_MP_POOL"
MATTERGEN_SLURM_FATAL_PENDING_REASONS = (
    "launch_failed",
    "held",
    "jobheld",
    "badconstraints",
    "invalidaccount",
    "invalidqos",
    "qosmax",
)

MECHANISM_AGENT_A_COMPACT_RULES = """COMPACT_OUTPUT_RULES:
- Return valid JSON only, using Agent A shape: status, agent, mechanisms, agree, concede.
- Include at most 3 mechanisms; prefer 2 if enough.
- Each mechanism: id, claim, rationale_summary, causal_driver, intervention, controlled_variables, descriptor_predictions, failure_modes, evidence_chain, scope, confidence.
- Keep claim <= 30 words.
- Keep fields concise; Never truncate JSON.
"""

MECHANISM_CONSENSUS_COMPACT_RULES = """COMPACT_OUTPUT_RULES:
- Return valid JSON only, using final consensus shape: status, accepted_mechanisms, rejected_mechanisms, consensus_summary, mechanism_requirements.
- Include at most 3 accepted_mechanisms; prefer 2 if that is enough.
- For each accepted mechanism, keep claim <= 30 words and rationale_summary <= 60 words.
- Accepted mechanisms should preserve microscopic fields when available: causal_driver, intervention, controlled_variables, descriptor_predictions, and failure_modes.
- Keep evidence_chain, counterevidence_considered, expected_material_signs, do_not_generalize_to, testable_implications, controlled_variables, descriptor_predictions, and failure_modes to at most 2 items each; each item <= 30 words.
- If the answer risks exceeding the output budget, remove weaker mechanisms or shorten fields. Never truncate JSON.
"""

MECHANISM_RAG_REQUIREMENT = """MANDATORY_MECHANISM_RAG:
- Use local RAG before final JSON: summarize_mechanism_evidence {"limit":6,"offset":0,"include_failed":true,"brief":true}.
- Read evidence_views including top_successes, near_misses, failure_boundaries, underexplored_clusters, recent_repetition, supported_predictions, failed_predictions.
- Distill evidence into concise fields; do not paste raw tool output.
"""

PRINCIPLE_DISCOVERY_POLICY = """PRINCIPLE_DISCOVERY_POLICY:
- Objective: validated materials-principle discovery; SUN is downstream evidence, not the mechanism target.
- Keep one microscopic causal theme per active principle program; refine/narrow/reject before starting a duplicate theme.
- State causal_driver, intervention axis, controls, expected descriptors/e_hull consequences, failure modes, and boundaries.
- Control-branch SUN is counterevidence or a new lead unless the primary/control test supports the original mechanism.
"""

PRINCIPLE_POSTMORTEM_COMPACT_RULES = """COMPACT_OUTPUT_RULES:
- Return valid JSON only, using shape: status, program_id, round, hypothesis_status, principle_update_action, current_principle_statement, micro_mechanism, e_hull_evidence, sun_accounting, causal_interpretation, failure_boundaries, unresolved_contradictions, next_test_focus, experience_book_entry.
- status must be one of continue, finalize, reject. principle_update_action must be one of refine, narrow, reject, finalize, promote_control_mechanism, start_new.
- hypothesis_status must be supported, contradicted, ambiguous, or execution_failed.
- If status is finalize or reject, include experience_book_entry with principle_statement, reasoning_chain, evidence_rounds, boundaries, and residual_risks.
- If this result refines or rejects the same causal theme as an existing principle_book entry, set experience_book_entry.updates_principle_id to that existing program_id. Only omit it when the causal theme is genuinely new.
- Keep causal_interpretation <= 120 words and each list to at most 4 concise items. Do not paste raw tool output.
"""

MECHANISM_EXPLORATION_REQUIREMENT = """MECHANISM_SEARCH_POLICY:
- A/B are peer debate agents. The current prompt may assign proposal, critique, or consensus duties, but neither side has privileged authority over the scientific conclusion.
- Read CONTEXT_JSON.mechanism_search_policy.current_mode before writing final mechanism JSON.
- exploitation: refine supported mechanism with one failure_boundary/near_miss.
- neighbor_exploration: mutate one axis of a supported mechanism.
- far_exploration: use underexplored_clusters/near_misses outside recent_repetition.
- Reviewer must reject mode collapse without materials-physics reason.
"""

CDEF_EXPLORATION_POLICY = """C/D/E/F_EXPLORATION_ANTI_REPETITION_POLICY:
- Read CONTEXT_JSON.mechanism_search_policy.current_mode before writing final prediction or execution JSON.
- In neighbor_exploration or far_exploration mode, do not collapse A/B's exploration mechanism back to the dominant recent material_id, formula, cation-family, or anion-motif unless explicitly using it as a baseline/control.
- You may use material_id and formula only as bookkeeping to avoid duplicate selections or high-overlap bundles, not as stability reasoning features.
- C/D exploration is only experimental-design exploration: preserve A/B's accepted mechanism, changed axis, and falsification intent. Do not introduce a new materials principle or pivot to an easier stable family merely because the pool looks promising.
- C/D must turn the exploration mechanism into executable primary/control queries that preserve the intended changed axis and check recent_repetition plus underexplored_clusters from historical RAG before accepting.
- D must reject prediction designs whose candidate pools or selection rules are likely to rematerialize the same recent chemistry without a materials-physics reason, but must not replace them with unrelated comparisons.
- E/F must audit final materialization for repetition risk: in exploration modes, reject or revise MP-pool bundles that simply take the same top low-formation-energy rows, formulas, or clusters from recent rounds.
- When an MP-pool branch has enough candidates, choose deterministic filters/order or explicit material_ids that favor untested examples while preserving the accepted prediction. If only repeated candidates are available, return prediction_design_infeasible or require C/D to revise the design.
"""

CDEF_MANDATORY_EXPLORATION_RAG = """MANDATORY_HISTORY_RAG:
- This is an exploration-mode C/D/E/F call. Before final prediction or execution JSON, request local RAG tools to inspect historical evidence and repetition risk.
- Your first tool batch must include summarize_mechanism_evidence with arguments {"limit": 6, "offset": 0, "include_failed": true, "brief": true}. It should also include query_candidate_pool calls for the primary/control pools or MP-pool branches you intend to use.
- Use evidence_views.recent_repetition, underexplored_clusters, near_misses, failure_boundaries, supported_predictions, failed_predictions, support_rate, min_e_hull, stable_count, and sun_score to decide whether the design truly explores or just repeats.
- Distill the result into matching_notes, required_revisions, rationale_summary, selection_notes, rejected_bundles, or prediction_design_feedback. Do not paste raw tool output.
"""

CD_PREDICTION_EXPLORATION_POLICY = """C/D/E/F_EXPLORATION_ANTI_REPETITION_POLICY:
- Read CONTEXT_JSON.mechanism_search_policy.current_mode.
- In exploration modes, preserve A/B's changed axis and falsification intent; avoid recent repeated formulas/clusters only inside that faithful test space.
- Reject designs likely to rematerialize the same recent chemistry unless it is an explicit baseline/control.
- Do not pivot to an unrelated stable family or easier SUN-rich pool. If faithful non-repeating materialization is impossible, say so explicitly.
"""

CD_MANDATORY_EXPLORATION_RAG = """MANDATORY_HISTORY_RAG:
- In neighbor/far exploration, first use summarize_mechanism_evidence {"limit":6,"offset":0,"include_failed":true,"brief":true}.
- Distill recent_repetition, underexplored_clusters, near_misses, and failure_boundaries into matching_notes or required_revisions; do not paste raw results.
"""

EF_EVIDENCE_QUALITY_POLICY = """E/F_EVIDENCE_QUALITY_POLICY:
- E/F are evidence-quality auditors, not SUN hunters. Materialize bundles only if they produce credible evidence about the active A/B principle.
- Audit whether primary/control branches are comparable enough that a lower e_hull can be interpreted mechanistically, not merely as candidate-pool bias.
- Flag strong controls explicitly: if the control branch embodies another plausible mechanism or produces SUN, it is a failure boundary or new mechanism lead, not proof of the original principle.
- Reject or repair bundles with obvious confounds in element count, nsites, density/volume scale, cation family, anion framework, or duplicate/repeated formulas unless those variables are the intended intervention.
"""

EF_COMPACT_PAYLOAD_POLICY = """E/F_COMPACT_PAYLOAD_POLICY:
- Some prior Agent E/F payloads in CONTEXT_JSON, FINAL_AGENT_E_JSON, and FINAL_AGENT_F_JSON are compact controller summaries, not raw executable JSON.
- Fields such as rationale_summary_stored_by_controller, selection_notes_stored_by_controller, *_chars, executable_*_stored_by_controller, and compact_payload_is_not_executable_original mean the controller has the original payload and will restore it by bundle id.
- Do not reject solely because compact summaries omit rationale_summary, selection_notes, structure_dicts, formula_probes, mattergen_requests, material_ids, or show stored-by-controller metadata.
- Only treat truncation as a schema blocker when the current raw Agent JSON you are revising contains a literal "..." or incomplete object inside an executable field.
"""

CD_HYPOTHESIS_VALIDATION_POLICY = """C/D_HYPOTHESIS_VALIDATION_POLICY:
- A/B own scientific exploration and mechanism invention; C/D only design matched primary/control falsification tests.
- Do not introduce a new materials principle; return infeasible/no consensus if the accepted mechanism cannot be faithfully tested.
- Stay inside the active principle program and accepted mechanism_ids; do not chase SUN, low formation energy, or unrelated stable basins.
- Use history/pool tools only to test executability, confounds, controls, underfill, and nearest faithful analogues.
- Each accepted prediction must state the tested mechanism_id, held-fixed variables, falsification criterion, counts, and primary/control contrast.
- If no faithful executable test remains, return no consensus or prediction_design_infeasible rather than changing the mechanism.
"""


PREDICTION_AGENT_C_COMPACT_RULES = """COMPACT_OUTPUT_RULES:
- JSON only. Agent C shape: status, agent, predictions, agree, concede, planned_material_count.
- Predictive design only: no formula_probes, structure_dicts, bundles, material_ids, or selected rows.
- Anchor every prediction to accepted mechanism_ids and target_count; prefer one faithful primary/control test whose counts sum to target_count.
- Keep claims <=35 words; keep matching_notes/falsification_criteria to two concise items.
- Use tools only for compact evidence such as pool counts, confounds, executable predicates, or blockers; do not paste raw tool output.
- When revising or reviewing a counterproposal, stay close to the immediately prior proposal unless a blocker makes that impossible.
- Include distilled evidence, not raw examples.
- Never truncate JSON.
"""

PREDICTION_AGENT_D_COMPACT_RULES = """COMPACT_OUTPUT_RULES:
- JSON only. Agent D shape: status, agent, required_revisions, agree, concede.
- Accept only predictions implied by accepted A/B mechanisms and executable as matched tests.
- Keep required_revisions to at most 4 blocker-level items, each <=70 words; do not repeat full prediction JSON in prose.
- In MatterGen-native mode, audit chemical-system requestability directly; candidate-pool tools are optional sanity evidence.
- If counterproposing, use Agent C shape and keep the same mechanism/falsification intent unless a concrete blocker forces the smallest faithful change.
- Cite distilled evidence, not raw tool output.
- Never truncate JSON.
"""


def _context_materialization_backend(context_json: str) -> str:
    try:
        context = json.loads(context_json)
    except Exception:
        return DEFAULT_MATERIALIZATION_BACKEND
    if not isinstance(context, Mapping):
        return DEFAULT_MATERIALIZATION_BACKEND
    backend = context.get("materialization_backend")
    if backend is None and isinstance(context.get("current_inputs"), Mapping):
        backend = context["current_inputs"].get("materialization_backend")
    return str(backend or DEFAULT_MATERIALIZATION_BACKEND)


def mattergen_native_prompt_block(context_json: str) -> str:
    if _context_materialization_backend(context_json) != "mattergen":
        return ""
    return f"""{MATTERGEN_NATIVE_TOOL_GATE_MARKER}:
- MatterGen-native mode is active. Do not perform an MP-pool query only to satisfy legacy C/D/E/F candidate-pool gates.
- You may still use query_candidate_pool as optional RAG/sanity evidence, especially for controls or confound checks, but final prediction/execution JSON must be MatterGen-testable without MP-pool rows.
- C/D prediction shape: keep primary_query/control_query only as optional observable summaries; add comparison_design.primary_mattergen and comparison_design.control_mattergen with chemical_system, count, filters, mechanism_alignment/control_alignment, held_fixed_variables, and drift_rejection_criteria.
- E/F execution branch shape: source="mattergen", count=<accepted structures to take>, mattergen_requests=[{{"backend":"mattergen","properties_to_condition_on":{{"chemical_system":"A-B[-C]","energy_above_hull":0.0}},"diffusion_guidance_factor":1.0,"filters":{{"chemical_system":["A","B"],"require_chemical_system_exact":true,"max_sites":20,"deduplicate_reduced_formula":true}}}}].
- MatterGen filters are not an arbitrary predicate DSL. Supported filters only: chemical_system, require_chemical_system_exact, min_sites, max_sites, min_volume_per_atom, max_volume_per_atom, deduplicate_reduced_formula, target_reduced_formula, require_target_reduced_formula, exclude_reduced_formulas.
- Keep filters.exclude_reduced_formulas compact and limited to formulas inside the active chemical_system. Do not paste unrelated history formulas from other element systems.
- The executable exclusion key is filters.exclude_reduced_formulas (a list). Do not invent or critique a synthetic filters.exclude_reduced_formulas_count key from compact summaries or earlier critique prose.
- Do not put stoichiometry_gate, branch_predicate, derived_predicate, post_generation_filter, acceptance_predicate, formula-ratio gates, oxidation-state gates, or natural-language predicates in filters.
- If the accepted mechanism requires O/P, Li/Mn, oxidation-state, or formula-ratio gates that the controller does not implement, do not fake them in filters. Choose a faithful chemical-system test the controller can execute, put non-executable drift concerns in drift_rejection_criteria, or return prediction_design_infeasible.
- Do not make primary_mattergen and control_mattergen differ only by unsupported post-generation predicates. They must be distinguishable by controller-supported MatterGen fields, or the prediction design is infeasible.
- A-F matched experiments must set filters.require_chemical_system_exact=true so element-subset drift is rejected, e.g. Li-P-O is invalid for a Li-Mn-P-O branch.
- Hard target_reduced_formula filtering is expensive and brittle in MatterGen. Prefer exact chemical_system comparisons without require_target_reduced_formula unless exact stoichiometry is the causal variable being tested. If a hard target is necessary, use bounded higher sampling (num_batches at least {DEFAULT_MATTERGEN_HARD_TARGET_MIN_NUM_BATCHES}, but no request above target_count={DEFAULT_MATTERGEN_MAX_TARGET_COUNT}, batch_size={DEFAULT_MATTERGEN_MAX_BATCH_SIZE}, num_batches={DEFAULT_MATTERGEN_MAX_NUM_BATCHES}, or {DEFAULT_MATTERGEN_MAX_RAW_SAMPLES} raw samples). If that still underfills, send the design back to C/D instead of escalating sampling indefinitely.
- Do not output formula_probes, structure_dicts, lattice vectors, fractional coordinates, or material_ids for MatterGen branches.
- In MatterGen-native mode, do not switch failed MatterGen branches to source="generator", source="mp_pool", formula_probes, structure_dicts, or material_ids. Repair by revising MatterGen filters/sampling, or return prediction_design_infeasible if no faithful MatterGen test remains.
- Avoid narrow lower site-count filters unless the mechanism strictly requires them. If materialization feedback reports too_few_sites, lower or remove filters.min_sites and/or increase MatterGen sampling; do not repeat the same underfilled filters.
"""


def mattergen_prediction_prompt_block(context_json: str) -> str:
    if _context_materialization_backend(context_json) != "mattergen":
        return ""
    return f"""{MATTERGEN_NATIVE_TOOL_GATE_MARKER}:
- C/D design matched primary/control chemical_system tests; E/F later writes executable mattergen_requests.
- Use comparison_design.primary_mattergen/control_mattergen with chemical_system, count, filters, held_fixed_variables, mechanism_alignment/control_alignment, and drift_rejection_criteria.
- Supported C/D filters: chemical_system, require_chemical_system_exact, min_sites, max_sites, min_volume_per_atom, max_volume_per_atom, deduplicate_reduced_formula, target_reduced_formula, require_target_reduced_formula.
- Set require_chemical_system_exact=true. Do not encode stoichiometry, oxidation-state, formula-ratio, or natural-language predicates as filters.
- primary_mattergen and control_mattergen must differ by a controller-supported condition or the design is infeasible.
- Do not output formula_probes, structure_dicts, lattice vectors, material_ids, bundles, or executable mattergen_requests in C/D.
"""


MECHANISM_AGENT_A_SYSTEM = (
    "You are Agent A in a materials-physics discovery MVP. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " You are one peer side of an A/B materials-physics debate. Propose mechanistic explanations for why some materials are more stable than others. "
    "Focus on charge balance, coordination, bonding, lattice strain, packing, and other observable materials features. "
    "When the controller enables MatterGen-native materialization, state whether the mechanism is testable by chemical-system conditioning and what generated-structure drift would falsify the test. "
    "Do not mention SUN as the objective. The evaluator reports e_hull; lower e_hull is better. "
    "Before final mechanism JSON, use local RAG tools to inspect past round evidence, including exploration cues, and fold the lessons into your concise fields."
)

MECHANISM_AGENT_B_SYSTEM = (
    "You are Agent B, one peer side and the rigorous critic in a materials-physics discovery MVP. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Attack unsupported mechanism claims with materials knowledge, prior evaluator outcomes, counterexamples, and exploration-collapse checks. "
    "When MatterGen-native materialization is enabled, audit whether the mechanism can actually be tested by chemical-system conditioned generation without losing the causal variable. "
    "Before final critique or counterproposal JSON, use local RAG tools to inspect past round evidence and exploration cues. "
    "Do not reveal hidden chain-of-thought. Provide concise evidence summaries only."
)

PRINCIPLE_POSTMORTEM_SYSTEM = (
    "You are writing the joint A/B postmortem for a materials-principle program. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Decide whether the active principle was supported, contradicted, or remains ambiguous after the latest primary/control test. "
    "Separate mechanism validation from SUN discovery. A control-branch SUN is counterevidence or a new mechanism lead, not success of the original hypothesis. "
    "Same material family plus same microscopic causal driver must update the existing principle-book experience instead of creating a duplicate entry. "
    "Update the microscopic causal explanation, boundaries, and next test focus without revealing hidden chain-of-thought."
)

PREDICTION_AGENT_C_SYSTEM = (
    "You are Agent C in a materials-physics discovery MVP. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Convert accepted A/B mechanisms into concise falsifiable matched primary/control predictions. "
    "In MatterGen mode design chemical_system conditions and counts; E/F will materialize. "
    "Use tools only for compact evidence, confounds, and executability. Do not chase SUN or unrelated stable families."
)

PREDICTION_AGENT_D_SYSTEM = (
    "You are Agent D, the adversarial auditor for prediction design in a materials-physics discovery MVP. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Audit whether C's matched prediction is implied by accepted mechanisms and executable as a test. "
    "In MatterGen mode audit chemical_system conditions/counts and confounds. "
    "Counterpropose only the nearest faithful fix; never replace A/B's hypothesis with an unrelated stable basin."
)

EXECUTION_AGENT_E_SYSTEM = (
    "You are Agent E in a materials-physics discovery MVP. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Translate the accepted predictions into concrete, executable materialization blueprints. "
    "Each accepted prediction must become its own bundle; never combine multiple prediction_ids into one bundle. "
    "When MatterGen-native materialization is enabled, prefer source='mattergen' branches with MatterGen requests over MP-pool rows or hand-built cells. "
    "Use candidate-pool fields only when you intentionally want MP-pool selection. "
    "If MatterGen-native materialization is enabled, express concrete branches as source='mattergen' requests. Otherwise, use generator-readable formula_probes or structure_dicts when appropriate. "
    "When the MP pool cannot satisfy a scientifically necessary branch, use source='mattergen' in MatterGen-native mode or source='generator' in legacy mode. "
    "When you provide generator structures, keep the compositions charge-balanced under the common formal valence model implied by the accepted hypothesis. "
    "If the accepted prediction had density_min/density_max or volume_per_atom_min/volume_per_atom_max filters, preserve those numeric filters in the generator branch query; the controller can rescale generator lattices to those windows during preflight. "
    "Do not invent final selected rows; the controller materializes the bundles after consensus. "
    "Before final execution JSON, request query_candidate_pool for every MP-pool branch you intend to use, or for the closest branch pools needed to decide whether MP-pool materialization is feasible. "
    "Do not write a final MP-pool bundle before seeing those tool results. "
    "In exploration rounds, choose executable branches that reduce recent material_id, formula, and cluster overlap whenever the accepted prediction allows it. "
    "Your output must be machine-readable and suitable for programmatic materialization."
)

EXECUTION_AGENT_F_SYSTEM = (
    "You are Agent F, the adversarial auditor for test materialization in a materials-physics discovery MVP. "
    + MATERIAL_PHYSICS_DIRECTIVE
    + " Reject bundles that do not faithfully operationalize the accepted predictions. "
    "Audit the bundle blueprint, not the final row-level materialization, because the controller performs the concrete selection after consensus. "
    "When MatterGen-native materialization is enabled, require executable source='mattergen' request branches unless E can justify MP-pool/generator materialization as the faithful test. "
    "Require Agent E to use generator formula_probes or structure_dicts by default when a branch can be stated concretely, and use MP-pool materialization only when that is the chosen branch type. "
    "Reject generator structures with impossible composition or formal charge imbalance, but do not spend repeated dialogue turns manually tuning lattice scale; the controller performs deterministic density/volume preflight and rescaling when the branch carries numeric query bounds. "
    "Before final audit JSON, request query_candidate_pool for every MP-pool branch in Agent E's proposal or for the nearest branch pools needed to audit materializability. "
    "Do not approve or reject an MP-pool bundle before seeing those tool results unless the blocker is purely logical/schema-level. "
    "In exploration rounds, reject bundles whose MP-pool plan is likely to select the same recent top rows, formulas, or cation-family/anion-motif clusters without a stated baseline/control role. "
    "Do not reveal hidden chain-of-thought. Provide concise audit summaries only."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the materials-physics MVP.")
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--work-dir", default="physics_mvp_runs/current", help="Working directory.")
    parser.add_argument("--memory-dir", default="physics_memory", help="Memory directory.")
    parser.add_argument("--candidate-pool", default="data/mp_candidate_pool/mp_candidates_filtered.jsonl")
    parser.add_argument("--training-data", default=None)
    parser.add_argument("--ppd-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=20260509)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--max-sites", type=int, default=80)
    parser.add_argument(
        "--materialization-backend",
        choices=("mixed", "mp_pool", "generator", "mattergen"),
        default=DEFAULT_MATERIALIZATION_BACKEND,
        help=(
            "Preferred A-F execution materialization backend. mixed preserves the legacy MP-pool/generator behavior; "
            "mattergen asks C/D/E/F for MatterGen-native primary/control chemical-system tests."
        ),
    )
    parser.add_argument("--max-dialogue-rounds", type=int, default=100)
    parser.add_argument(
        "--max-prediction-execution-feedback-rounds",
        type=int,
        default=2,
        help=(
            "Maximum times E/F may send a jointly accepted prediction-design infeasibility report "
            "back to C/D for prediction redesign within one round."
        ),
    )
    parser.add_argument(
        "--critic-counterproposal-after",
        type=int,
        default=DEFAULT_CRITIC_COUNTERPROPOSAL_AFTER,
        help=(
            "Ask the critic to produce a proposer-shaped counterproposal after this many consecutive rejected critiques. "
            "Default 1 implements strict alternating proposal/review; use 0 to disable."
        ),
    )
    parser.add_argument("--json-repair-attempts", type=int, default=2)
    parser.add_argument("--mechanism-model", default=None)
    parser.add_argument("--prediction-model", default=None)
    parser.add_argument("--execution-model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--mechanism-max-tokens",
        type=int,
        default=DEFAULT_MECHANISM_MAX_TOKENS,
        help="Mechanism-stage LLM max output tokens when --max-tokens is not set. Use 0 to fall back to LLM_MAX_TOKENS.",
    )
    parser.add_argument(
        "--prediction-max-tokens",
        type=int,
        default=DEFAULT_PREDICTION_MAX_TOKENS,
        help="Prediction-stage LLM max output tokens when --max-tokens is not set. Use 0 to fall back to LLM_MAX_TOKENS.",
    )
    parser.add_argument(
        "--execution-max-tokens",
        type=int,
        default=DEFAULT_EXECUTION_MAX_TOKENS,
        help="Execution-stage LLM max output tokens when --max-tokens is not set. Use 0 to fall back to LLM_MAX_TOKENS.",
    )
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--disable-local-agents", action="store_true", help="Disable controller-managed local agent tools for A/B/C/D/E/F.")
    parser.add_argument("--agent-allow-project-writes", action="store_true", help="Allow local agents to write project files. Disabled by default; agent artifacts remain writable.")
    parser.add_argument(
        "--agent-max-steps",
        type=int,
        default=DEFAULT_AGENT_MAX_STEPS,
        help="Safety cap for local tool-use turns per LLM role call. Raise this to allow deeper agent exploration.",
    )
    parser.add_argument(
        "--agent-max-tool-calls",
        type=int,
        default=DEFAULT_AGENT_MAX_TOOL_CALLS,
        help="Safety cap for local tool calls executed in one agent turn. Raise this to allow broader RAG batches.",
    )
    parser.add_argument(
        "--agent-max-tool-result-chars",
        type=int,
        default=6000,
        help="Maximum serialized tool-result characters returned to the LLM per agent turn.",
    )
    parser.add_argument("--evaluator-backend", choices=("local", "slurm"), default="slurm")
    parser.add_argument("--slurm-evaluator-script", default="slurm_evaluate_round_4090.sbatch")
    parser.add_argument("--slurm-partition", default="8-4090")
    parser.add_argument("--slurm-gres", default="gpu:rtx4090:1")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=16)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--mattergen-root", default=DEFAULT_MATTERGEN_ROOT)
    parser.add_argument(
        "--mattergen-venv",
        default=None,
        help="MatterGen Python environment. Defaults to <mattergen-root>/.venv; use the CUDA-12.8 venv for RTX 5090.",
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
        "--mattergen-diffusion-guidance-factor",
        type=float,
        default=DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR,
        help="Classifier-free guidance scale for conditional MatterGen sampling. Keep >0 for chemical-system conditioning.",
    )
    parser.add_argument("--mattergen-partition", default=DEFAULT_MATTERGEN_PARTITION)
    parser.add_argument("--mattergen-gres", default=DEFAULT_MATTERGEN_GRES)
    parser.add_argument("--mattergen-cpus-per-task", type=int, default=DEFAULT_MATTERGEN_CPUS_PER_TASK)
    parser.add_argument("--mattergen-module-init", default=DEFAULT_MATTERGEN_MODULE_INIT)
    parser.add_argument("--mattergen-modules", default=DEFAULT_MATTERGEN_MODULES)
    parser.add_argument("--mattergen-cuda-home", default=DEFAULT_MATTERGEN_CUDA_HOME)
    parser.add_argument("--mattergen-runner", choices=("slurm", "local"), default="slurm")
    parser.add_argument("--mattergen-job-timeout", type=int, default=0)
    parser.add_argument("--mattergen-poll-sec", type=float, default=10.0)
    parser.add_argument("--sleep-between-rounds", type=float, default=0.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_training_data(root: Path) -> Path:
    return root / "archive" / "matllmsearch_evaluator" / "data" / "a_training.json"


def default_ppd_path(root: Path) -> Path:
    return root / "archive" / "matllmsearch_evaluator" / "data" / "2024-08-07-ppd-mp.pkl"


def _resolve_project_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def command_text(cmd: Sequence[str]) -> str:
    return " ".join(str(part) for part in cmd)


def log_event(round_dir: Path, message: str) -> None:
    round_dir.mkdir(parents=True, exist_ok=True)
    line = f"[{utc_now()}] {message}\n"
    with (round_dir / "controller.log").open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = orjson.loads(line)
            if isinstance(item, dict):
                records.append(item)
    return records


def load_candidate_pool(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for record in records:
            handle.write(orjson.dumps(dict(record), option=orjson.OPT_APPEND_NEWLINE))


def json_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return True


def pool_digest(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nelements_hist = Counter()
    crystal_system_hist = Counter()
    element_hist = Counter()
    band_gap_values: list[float] = []
    for record in records:
        nelements_hist[int(record.get("nelements") or 0)] += 1
        crystal_system_hist[str(record.get("crystal_system") or "unknown")] += 1
        for element in record.get("elements", []) or []:
            element_hist[str(element)] += 1
        band_gap = record.get("band_gap")
        if isinstance(band_gap, (int, float)):
            band_gap_values.append(float(band_gap))
    common_elements = [item for item, _ in element_hist.most_common(20)]
    return {
        "count": len(records),
        "nelements_histogram": dict(sorted(nelements_hist.items())),
        "crystal_system_histogram": dict(crystal_system_hist.most_common(20)),
        "common_elements": common_elements,
        "band_gap": {
            "min": min(band_gap_values) if band_gap_values else None,
            "median": sorted(band_gap_values)[len(band_gap_values) // 2] if band_gap_values else None,
            "max": max(band_gap_values) if band_gap_values else None,
        },
    }


def summary_of_round(history_entry: Mapping[str, Any]) -> dict[str, Any]:
    summary = history_entry.get("evaluation_summary")
    if isinstance(summary, Mapping):
        return dict(summary)
    return {}


def schema_reference_for_stage(stage: str) -> dict[str, Any]:
    if stage == "mechanism":
        return {
            "material_physics_mechanism_consensus": {
                "status": "consensus|no_consensus",
                "accepted_mechanisms": [
                    {
                        "id": "m001",
                        "claim": "<=30 words",
                        "rationale_summary": "concise observable-materials rationale",
                        "causal_driver": "charge/coordination/bond strain/packing/etc.",
                        "intervention": "single variable to test",
                        "controlled_variables": ["fixed confounds"],
                        "descriptor_predictions": ["expected descriptor changes"],
                        "failure_modes": ["what disproves/narrows"],
                        "evidence_chain": ["<=2 evidence items"],
                        "scope": "where it applies",
                        "confidence": "low|medium|high",
                    }
                ],
                "rejected_mechanisms": [{"id": "m002", "rejection_reason": "why"}],
                "consensus_summary": "short summary",
            }
        }
    if stage == "prediction":
        return {
            "material_physics_prediction_consensus": {
                "status": "consensus|rejected|no_consensus",
                "accepted_predictions": [
                    {
                        "id": "p001",
                        "mechanism_ids": ["m001"],
                        "claim": "short falsifiable prediction",
                        "predicted_relation": "primary_lower_e_hull_than_control|primary_higher_e_hull_than_control",
                    "comparison_design": {
                        "primary_query": {"elements_all": ["Li"], "preferred_order": ["formation_energy_per_atom asc"]},
                        "control_query": {"elements_all": ["Na"], "preferred_order": ["formation_energy_per_atom asc"]},
                        "primary_mattergen": {
                            "chemical_system": "Li-P-S",
                            "count": 5,
                            "filters": {"max_sites": 20, "deduplicate_reduced_formula": True, "require_chemical_system_exact": True},
                            "mechanism_alignment": "why this chemical system expresses the accepted mechanism",
                        },
                        "control_mattergen": {
                            "chemical_system": "Na-P-S",
                            "count": 5,
                            "filters": {"max_sites": 20, "deduplicate_reduced_formula": True, "require_chemical_system_exact": True},
                            "control_alignment": "what matched variable is changed or held fixed",
                        },
                        "primary_count": 5,
                        "control_count": 5,
                        "matching_notes": ["short deterministic matching note"],
                    },
                        "falsification_criteria": ["short criterion"],
                        "scope": "tested family only",
                        "confidence": "low|medium|high",
                    }
                ],
                "rejected_predictions": [{"id": "p002", "claim": "short rejected claim", "rejection_reason": "why"}],
                "consensus_summary": "short summary",
            }
        }
    if stage == "execution":
        return {
            "material_physics_execution_plan": {
                "status": "consensus|materialization_conflict|no_materialized_consensus|prediction_design_infeasible",
                "accepted_bundles": [
                    {
                        "id": "b001",
                        "prediction_ids": ["p001"],
                        "expected_relation": "primary_lower_e_hull_than_control",
                        "primary": {
                            "source": "mp_pool|generator|mattergen",
                            "count": 5,
                            "query": {"elements_all": ["Li"], "preferred_order": ["formation_energy_per_atom asc"]},
                            "mattergen_requests": [
                                {
                                    "backend": "mattergen",
                                    "properties_to_condition_on": {
                                        "chemical_system": "Li-P-S",
                                        "energy_above_hull": 0.0,
                                    },
                                    "diffusion_guidance_factor": 1.0,
                                    "filters": {"chemical_system": ["Li", "P", "S"], "require_chemical_system_exact": True, "max_sites": 20},
                                }
                            ],
                            "formula_probes_or_structure_dicts": "required only when source=generator",
                        },
                        "control": {
                            "source": "mp_pool|generator|mattergen",
                            "count": 5,
                            "query": {"elements_all": ["Na"], "preferred_order": ["formation_energy_per_atom asc"]},
                            "mattergen_requests": [
                                {
                                    "backend": "mattergen",
                                    "properties_to_condition_on": {
                                        "chemical_system": "Na-P-S",
                                        "energy_above_hull": 0.0,
                                    },
                                    "diffusion_guidance_factor": 1.0,
                                    "filters": {"chemical_system": ["Na", "P", "S"], "require_chemical_system_exact": True, "max_sites": 20},
                                }
                            ],
                        },
                        "rationale_summary": "short materialization rationale",
                        "selection_notes": "short deterministic selection rule",
                    }
                ],
                "rejected_bundles": [{"id": "b002", "reason": "why"}],
                "consensus_summary": "short summary",
                "prediction_design_feedback": {
                    "use_only_when_status_is_prediction_design_infeasible": True,
                    "prediction_ids": ["p001"],
                    "blocking_issue": "why the accepted C/D prediction cannot be faithfully materialized",
                    "required_cd_reconsideration": ["what C/D must revise"],
                },
                "materialization_constraints": {"allowed_sources": ["mp_pool", "generator", "mattergen"]},
            }
        }
    if stage == "postmortem":
        return {
            "material_physics_principle_postmortem": {
                "status": "continue|finalize|reject",
                "program_id": "principle_program_001",
                "round": 1,
                "hypothesis_status": "supported|contradicted|ambiguous|execution_failed",
                "principle_update_action": "refine|narrow|reject|finalize|promote_control_mechanism|start_new",
                "current_principle_statement": "current best principle statement",
                "micro_mechanism": "microscopic causal explanation",
                "e_hull_evidence": {
                    "primary_mean": 0.0,
                    "control_mean": 0.0,
                    "support_rate": 1.0,
                },
                "sun_accounting": {
                    "primary_sun_count": 0,
                    "control_sun_count": 0,
                    "mechanism_validated_sun_count": 0,
                },
                "causal_interpretation": "why the data support, contradict, or narrow the principle",
                "failure_boundaries": ["where not to generalize"],
                "unresolved_contradictions": ["remaining evidence the principle cannot yet explain"],
                "next_test_focus": "next falsifiable refinement test",
                "experience_book_entry": {
                    "required_when_finalizing_or_rejecting": True,
                    "principle_identity": "new_principle|update_existing_principle",
                    "updates_principle_id": "existing principle_program id when updating the same causal theme",
                    "topic_key": "short stable identity such as Mg-Fe-O-F::Fe-O-Fe-network-disruption",
                    "principle_statement": "final or rejected principle",
                    "reasoning_chain": ["concise evidence-grounded steps"],
                    "evidence_rounds": [1],
                    "boundaries": ["scope limits"],
                    "residual_risks": ["what remains uncertain"],
                },
            }
        }
    reference = json.loads(schema_reference_json())
    return reference


def compact_prediction_schema_reference_for_repair() -> dict[str, Any]:
    return {
        "material_physics_prediction_consensus": {
            "status": "consensus|rejected|no_consensus",
            "accepted_predictions_or_predictions": [
                {
                    "id": "p001",
                    "mechanism_ids": ["accepted mechanism id"],
                    "claim": "short falsifiable prediction",
                    "predicted_relation": "primary_lower_e_hull_than_control|primary_higher_e_hull_than_control",
                    "comparison_design": {
                        "primary_count": "integer",
                        "control_count": "integer",
                        "primary_query_or_mattergen": "supported query object or primary_mattergen chemical_system design",
                        "control_query_or_mattergen": "supported query object or control_mattergen chemical_system design",
                    },
                    "falsification_criteria": ["short criterion"],
                }
            ],
            "planned_material_count": "sum(primary_count + control_count) over accepted predictions",
        }
    }


def _short_text(value: Any, max_chars: int = 360) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _compact_query_for_context(query: Any) -> Any:
    if not isinstance(query, Mapping):
        return query
    compact: dict[str, Any] = {}
    allowed_keys = {
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
    for key, value in query.items():
        if key not in allowed_keys:
            continue
        compact[str(key)] = value
    return compact


def _compact_mechanism_for_context(item: Any) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {"claim": _short_text(item)}
    return {
        key: value
        for key, value in {
            "id": item.get("id"),
            "claim": _short_text(item.get("claim"), 240),
            "rationale_summary": _short_text(item.get("rationale_summary"), 360),
            "causal_driver": _short_text(item.get("causal_driver"), 180),
            "intervention": _short_text(item.get("intervention"), 180),
            "controlled_variables": [_short_text(raw, 120) for raw in item.get("controlled_variables", [])[:3]]
            if isinstance(item.get("controlled_variables"), list)
            else None,
            "descriptor_predictions": [_short_text(raw, 120) for raw in item.get("descriptor_predictions", [])[:3]]
            if isinstance(item.get("descriptor_predictions"), list)
            else None,
            "failure_modes": [_short_text(raw, 120) for raw in item.get("failure_modes", [])[:3]]
            if isinstance(item.get("failure_modes"), list)
            else None,
            "scope": _short_text(item.get("scope"), 240),
            "confidence": item.get("confidence"),
        }.items()
        if value not in (None, "")
    }


def _compact_mechanism_for_prompt(item: Any) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {"claim": _short_text(item, 160)}
    return {
        key: value
        for key, value in {
            "id": item.get("id"),
            "claim": _short_text(item.get("claim"), 120),
            "causal_driver": _short_text(item.get("causal_driver"), 80),
            "intervention": _short_text(item.get("intervention"), 80),
            "descriptor_predictions": [_short_text(raw, 60) for raw in item.get("descriptor_predictions", [])[:1]]
            if isinstance(item.get("descriptor_predictions"), list)
            else None,
            "failure_modes": [_short_text(raw, 60) for raw in item.get("failure_modes", [])[:1]]
            if isinstance(item.get("failure_modes"), list)
            else None,
            "confidence": item.get("confidence"),
        }.items()
        if value not in (None, "", [], {})
    }


def _compact_mechanism_payload_for_prompt(value: Any, *, max_mechanisms: int = 2) -> Any:
    if not isinstance(value, Mapping):
        return compact_dialogue_payload_for_prompt(value, max_list_items=max_mechanisms)
    compact: dict[str, Any] = {
        key: value.get(key)
        for key in ("status", "agent", "agree", "concede")
        if value.get(key) not in (None, "", [], {})
    }
    for key in ("mechanisms", "accepted_mechanisms"):
        items = value.get(key)
        if isinstance(items, list):
            compact[key] = [_compact_mechanism_for_prompt(item) for item in items[:max_mechanisms]]
            if len(items) > max_mechanisms:
                compact[f"{key}_omitted_count"] = len(items) - max_mechanisms
    rejected = value.get("rejected_mechanisms")
    if isinstance(rejected, list):
        compact["rejected_mechanisms"] = [_compact_rejected_item_for_context(item) for item in rejected[:2]]
    revisions = value.get("required_revisions")
    if isinstance(revisions, list):
        compact["required_revisions"] = [_compact_rejected_item_for_context(item) for item in revisions[:3]]
    for key, limit in (
        ("critique_summary", 220),
        ("consensus_summary", 220),
        ("evidence_summary", 180),
        ("overall_reasoning_summary", 160),
        ("proposal_summary", 140),
        ("impossibility_certificate", 180),
    ):
        item = value.get(key)
        if item not in (None, "", [], {}):
            compact[key] = compact_repair_feedback(item, max_list_items=2) if isinstance(item, (Mapping, list)) else _short_text(item, limit)
    return {key: item for key, item in compact.items() if item not in (None, "", [], {})}


def _compact_prediction_for_context(item: Any) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {"claim": _short_text(item)}
    comparison = item.get("comparison_design")
    compact_comparison: dict[str, Any] = {}
    if isinstance(comparison, Mapping):
        for key in ("primary_count", "control_count"):
            if key in comparison:
                compact_comparison[key] = comparison[key]
        if isinstance(comparison.get("matching_notes"), list):
            compact_comparison["matching_notes"] = [_short_text(note, 240) for note in comparison.get("matching_notes", [])[:2]]
        for key in ("primary_query", "control_query"):
            if key in comparison:
                compact_comparison[key] = _compact_query_for_context(comparison.get(key))
        for key in ("primary_mattergen", "control_mattergen"):
            if isinstance(comparison.get(key), Mapping):
                compact_comparison[key] = compact_repair_feedback(comparison.get(key), max_list_items=4)
    falsification_criteria = item.get("falsification_criteria")
    return {
        key: value
        for key, value in {
            "id": item.get("id"),
            "mechanism_ids": item.get("mechanism_ids"),
            "claim": _short_text(item.get("claim"), 360),
            "predicted_relation": item.get("predicted_relation"),
            "comparison_design": compact_comparison if compact_comparison else None,
            "falsification_criteria": [_short_text(raw, 240) for raw in falsification_criteria[:2]]
            if isinstance(falsification_criteria, list)
            else None,
            "scope": _short_text(item.get("scope"), 240),
            "confidence": item.get("confidence"),
        }.items()
        if value not in (None, "", [])
    }


def compact_current_inputs_for_stage(stage: str, current_inputs: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(current_inputs or {})
    mechanisms = result.get("accepted_mechanisms")
    if isinstance(mechanisms, list):
        if stage == "prediction":
            result["accepted_mechanisms"] = [_compact_mechanism_for_prompt(item) for item in mechanisms[:3]]
        else:
            result["accepted_mechanisms"] = [_compact_mechanism_for_context(item) for item in mechanisms[:4]]
    predictions = result.get("accepted_predictions")
    if isinstance(predictions, list):
        result["accepted_predictions"] = [
            _compact_prediction_for_context(item) for item in predictions[:PROMPT_MATERIALIZATION_ITEM_LIMIT]
        ]
        if len(predictions) > PROMPT_MATERIALIZATION_ITEM_LIMIT:
            result["accepted_predictions_omitted_count"] = len(predictions) - PROMPT_MATERIALIZATION_ITEM_LIMIT
    return result


def compact_pool_summary_for_context(pool_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Keep pool-scale facts without replaying full histograms in every prompt."""

    compact: dict[str, Any] = {}
    for key in ("count", "band_gap"):
        if key in pool_summary:
            compact[key] = pool_summary[key]
    nelements_histogram = pool_summary.get("nelements_histogram")
    if isinstance(nelements_histogram, Mapping):
        compact["nelements_histogram"] = dict(nelements_histogram)
    common_elements = pool_summary.get("common_elements")
    if isinstance(common_elements, list):
        compact["common_elements"] = common_elements[:16]
        if len(common_elements) > 16:
            compact["common_elements_omitted_count"] = len(common_elements) - 16
    crystal_system_histogram = pool_summary.get("crystal_system_histogram")
    if isinstance(crystal_system_histogram, Mapping):
        compact["crystal_system_histogram_top"] = dict(list(crystal_system_histogram.items())[:4])
    return compact or dict(pool_summary)


def _compact_principle_program_for_context(program: Any) -> dict[str, Any] | None:
    if not isinstance(program, Mapping):
        return None
    evidence_rounds = program.get("evidence_rounds")
    compact_evidence: list[dict[str, Any]] = []
    if isinstance(evidence_rounds, list):
        for item in evidence_rounds[-6:]:
            if not isinstance(item, Mapping):
                continue
            compact_evidence.append(
                {
                    key: value
                    for key, value in {
                        "round": item.get("round"),
                        "hypothesis_status": item.get("hypothesis_status"),
                        "principle_update_action": item.get("principle_update_action"),
                        "support_rate": item.get("support_rate"),
                        "primary_sun_count": item.get("primary_sun_count"),
                        "control_sun_count": item.get("control_sun_count"),
                        "mechanism_validated_sun_count": item.get("mechanism_validated_sun_count"),
                        "delta": item.get("delta"),
                        "short_interpretation": _short_text(item.get("causal_interpretation"), 240),
                    }.items()
                    if value not in (None, "", [])
                }
            )
    return {
        key: value
        for key, value in {
            "program_id": program.get("program_id"),
            "status": program.get("status"),
            "started_round": program.get("started_round"),
            "inner_iteration": program.get("inner_iteration"),
            "max_inner_rounds": program.get("max_inner_rounds"),
            "current_principle_statement": _short_text(program.get("current_principle_statement"), 420),
            "micro_mechanism": _short_text(program.get("micro_mechanism"), 420),
            "active_mechanism_ids": program.get("active_mechanism_ids"),
            "principle_identity": program.get("principle_identity"),
            "principle_book_update_target": program.get("principle_book_update_target"),
            "failure_boundaries": [
                _short_text(item, 180) for item in program.get("failure_boundaries", [])[:4]
            ]
            if isinstance(program.get("failure_boundaries"), list)
            else None,
            "unresolved_contradictions": [
                _short_text(item, 180) for item in program.get("unresolved_contradictions", [])[:4]
            ]
            if isinstance(program.get("unresolved_contradictions"), list)
            else None,
            "recent_evidence_rounds": compact_evidence,
        }.items()
        if value not in (None, "", [])
    }


def _compact_principle_book_for_context(
    book: Any,
    *,
    entry_limit: int = 2,
    evidence_limit: int = 3,
) -> list[dict[str, Any]]:
    if not isinstance(book, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in book[-max(1, entry_limit) :]:
        if not isinstance(item, Mapping):
            continue
        evidence_rounds = item.get("evidence_rounds")
        compact_evidence_rounds: list[Any] | None = None
        if isinstance(evidence_rounds, list):
            compact_evidence_rounds = []
            for raw in evidence_rounds[-max(1, evidence_limit) :]:
                if isinstance(raw, Mapping):
                    compact_evidence_rounds.append(
                        {
                            key: value
                            for key, value in {
                                "round": raw.get("round"),
                                "status": raw.get("hypothesis_status") or raw.get("status"),
                                "support_rate": raw.get("support_rate"),
                                "primary_sun_count": raw.get("primary_sun_count"),
                                "control_sun_count": raw.get("control_sun_count"),
                                "summary": _short_text(raw.get("causal_interpretation") or raw.get("summary"), 80),
                            }.items()
                            if value not in (None, "", [])
                        }
                    )
                else:
                    compact_evidence_rounds.append(_short_text(raw, 120))
        compact.append(
            {
                key: value
                for key, value in {
                    "program_id": item.get("program_id"),
                    "status": item.get("status"),
                    "topic_key": _short_text(item.get("topic_key"), 120),
                    "last_updated_round": item.get("last_updated_round"),
                    "principle_statement": _short_text(item.get("principle_statement"), 120),
                    "micro_mechanism": _short_text(item.get("micro_mechanism"), 120),
                    "recent_evidence_rounds": compact_evidence_rounds,
                    "evidence_round_count": len(evidence_rounds) if isinstance(evidence_rounds, list) else None,
                    "boundaries": [_short_text(raw, 80) for raw in item.get("boundaries", [])[:1]]
                    if isinstance(item.get("boundaries"), list)
                    else None,
                    "residual_risks": [_short_text(raw, 80) for raw in item.get("residual_risks", [])[:1]]
                    if isinstance(item.get("residual_risks"), list)
                    else None,
                }.items()
                if value not in (None, "", [])
            }
        )
    return compact


def compact_generator_schema_reference() -> dict[str, Any]:
    return {
        "allowed_generator_fields": ["formula_probes", "structure_dicts"],
        "formula_probe_note": "Use only when the existing generator template can faithfully express the branch chemistry.",
        "structure_dicts_note": "Use pymatgen Structure.as_dict() objects for mixed-anion or explicit structures.",
        "mattergen_note": "Use source='mattergen' with mattergen_requests when testing a chemical system by conditional MatterGen generation.",
        "mattergen_request_required_fields": {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "A-B[-C]", "energy_above_hull": 0.0},
            "diffusion_guidance_factor": DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR,
            "filters": {"chemical_system": ["A", "B"], "max_sites": DEFAULT_MATTERGEN_MAX_SITES},
        },
        "branch_rules": [
            "source='generator' requires non-empty formula_probes or structure_dicts.",
            "source='mattergen' requires exactly one mattergen_requests object and forbids formula_probes/structure_dicts.",
            "branch count must equal len(formula_probes) or len(structure_dicts).",
            "For MatterGen branches, branch count is the number of accepted structures the controller should take from the generated batch.",
            "formula_probe_count and structure_dict_count are controller feedback summaries, not executable output fields.",
            "Generated structures must pass formal charge balance and pymatgen parsing.",
        ],
    }


def build_context_payload(
    *,
    state: Mapping[str, Any],
    pool_summary: Mapping[str, Any],
    stage: str,
    repair_feedback: Mapping[str, Any] | None = None,
    current_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    materialization_backend = str((current_inputs or {}).get("materialization_backend") or DEFAULT_MATERIALIZATION_BACKEND)
    history = state.get("history", [])
    if not isinstance(history, list):
        history = []
    recent_history = history[-3:]
    recent_history_compact = []
    for item in recent_history:
        if not isinstance(item, Mapping):
            continue
        recent_history_compact.append(
            {
                "round": item.get("round"),
                "status": item.get("status", "complete"),
                "mechanism_count": len(item.get("accepted_mechanisms", []) or []),
                "prediction_count": len(item.get("accepted_predictions", []) or []),
                "bundle_count": len(item.get("accepted_bundles", []) or []),
                "support_rate": item.get("evaluation_summary", {}).get("support_rate")
                if isinstance(item.get("evaluation_summary"), Mapping)
                else None,
                "mean_e_hull": item.get("evaluation_summary", {}).get("mean_e_hull")
                if isinstance(item.get("evaluation_summary"), Mapping)
                else None,
                "skip_summary": item.get("skip_summary") if isinstance(item.get("skip_summary"), Mapping) else None,
            }
        )
    base_instructions = [
        MATERIAL_PHYSICS_DIRECTIVE,
        "Primary objective: discover, refine, and validate reusable materials principles. SUN discovery is a downstream metric, not the optimization target.",
        "Keep mechanism validation separate from SUN accounting: a control-branch SUN is counterevidence or a new mechanism lead unless the primary/control test supports the original mechanism.",
        "Use only observable materials features. Do not use is_stable, e_hull, SUN, novelty, material_id, cif_path, or properties_path as reasoning inputs.",
        "The evaluator measures e_hull. Lower e_hull means greater thermodynamic stability.",
        "Prediction-stage outputs must be deterministic matched-pair plans: fix the bundle size, fix the primary/control counts, and spell out a finite selection order.",
        "Prediction-stage primary_query and control_query must differ by an executable observable predicate; do not leave them identical or describe the difference only in prose.",
        "Avoid open-ended language such as 'prioritize', 'if possible', or 'when available' unless you also provide the exact tie-break order that makes the rule deterministic.",
        "For any mixed-anion mechanism, prefer a direct mixed-anion versus single-anion observable predicate over a symmetry proxy; do not use spacegroup_number or crystal_system as the main causal split unless the mechanism itself is explicitly about symmetry.",
        "Execution-stage outputs must use only the supported query schema keys: material_ids, formula_in, formula_regex, elements_all, elements_any, elements_none, nelements_min, nelements_max, nsites_min, nsites_max, band_gap_min, band_gap_max, formation_energy_per_atom_min, formation_energy_per_atom_max, density_min, density_max, volume_per_atom_min, volume_per_atom_max, crystal_system_in, spacegroup_number_in, spacegroup_number_min, spacegroup_number_max, and preferred_order.",
        "Do not invent elements_exact, nelements_exact, derived_predicate, branch_predicate, include_ranks, materialization_guards, or any other custom DSL fields in execution outputs.",
    ]
    if stage == "prediction":
        base_instructions = [
            MATERIAL_PHYSICS_DIRECTIVE,
            "C/D are validation-design agents: design discriminating faithful tests as falsifiable matched primary/control plans for accepted A/B mechanism_ids.",
            "Do not chase SUN, low formation energy, or unrelated stable basins.",
            "Use observable features only; e_hull is evaluator output, lower is better.",
            "Prediction plans require deterministic matched-pair exact counts, held-fixed variables, falsification criteria, and executable primary/control conditions.",
            "Supported MP-pool query keys are material_ids, formula_in/regex, elements_all/any/none, nelements/nsites/band_gap/formation_energy/density/volume_per_atom/crystal_system/spacegroup filters, and preferred_order.",
        ]
    if stage == "postmortem":
        base_instructions = [
            MATERIAL_PHYSICS_DIRECTIVE,
            "Primary objective: update the active materials principle so it explains the latest evidence without post-hoc overfitting.",
            "Separate three outcomes: mechanism support, SUN discovery, and control-derived leads. Do not count control-branch SUN as success of the original A/B mechanism.",
            "Use the latest round's primary/control e_hull evidence, SUN accounting, accepted mechanisms, predictions, and prior principle-program evidence.",
            "The postmortem must choose continue, finalize, or reject. Continue means the next inner iteration must refine or narrow the same principle and propose another falsifiable test.",
            "Finalize only when the principle can explain this program's supporting data, contradictions, control-SUNs, and boundaries. Reject when contradictions cannot be resolved without abandoning the principle.",
            "Before finalizing or rejecting, compare the current principle against principle_book_tail. If it has the same material family, intervention axis, and microscopic causal driver as an existing entry, update that entry by setting experience_book_entry.principle_identity='update_existing_principle' and experience_book_entry.updates_principle_id to the existing program_id.",
            "Only mark experience_book_entry.principle_identity='new_principle' when A/B can state why the current causal theme cannot be absorbed into any existing principle-book entry.",
        ]
    if stage == "mechanism":
        base_instructions = [
            MATERIAL_PHYSICS_DIRECTIVE,
            "Primary objective: validated materials-principle discovery.",
            "Use only observable materials features; e_hull is evaluator output, lower is better.",
            "Work inside current_principle_program when present; refine/narrow/reject before starting a duplicate theme.",
            "State microscopic causal logic: causal_driver, intervention axis, controls, expected signs, failure modes, and boundaries.",
            "Use mechanism_search_policy and RAG evidence_views: near_misses, failure_boundaries, underexplored_clusters, recent_repetition.",
        ]
    if stage == "prediction":
        base_instructions.extend(
            [
                "Read mechanism_search_policy.current_mode and current_principle_program before accepting a design.",
                "In exploration modes, reduce recent repetition only within the faithful A/B mechanism test space.",
            ]
        )
    if stage == "execution":
        base_instructions.extend(
            [
                "C/D/E/F must read mechanism_search_policy. In exploration modes, prediction and execution designs must visibly reduce recent repetition while preserving the accepted mechanism.",
                "C/D/E/F must also read current_principle_program. Do not replace the active principle with an easier SUN-rich family; design evidence that refines, narrows, or falsifies the active principle.",
                "Use local historical RAG evidence_views.recent_repetition and underexplored_clusters before accepting exploration-mode designs; use material_id/formula only as duplicate bookkeeping, not as stability reasoning.",
                "In exploration modes, reject designs or bundles that are likely to rematerialize the same recent material IDs, formulas, or cation-family/anion-motif clusters unless they are explicitly designated baselines or controls.",
            ]
        )
    context: dict[str, Any] = {
        "schema_version": "material_physics_context.v1",
        "stage": stage,
        "materialization_backend": materialization_backend,
        "directive": MATERIAL_PHYSICS_DIRECTIVE,
        "candidate_pool_summary": compact_pool_summary_for_context(pool_summary),
        "required_output_schema_reference": compact_prediction_schema_reference_for_repair()
        if stage == "prediction"
        else schema_reference_for_stage(stage),
        "state": {
            "schema_version": state.get("schema_version"),
            "status": state.get("status"),
            "current_round": state.get("current_round"),
            "best_round": state.get("best_round"),
            "best_support_rate": state.get("best_support_rate"),
            "latest_support_rate": state.get("latest_support_rate"),
            "history_tail": recent_history_compact,
        },
        "current_inputs": compact_current_inputs_for_stage(stage, current_inputs),
        "repair_feedback": compact_prediction_design_feedback_for_prompt(repair_feedback)
        if stage == "prediction" and repair_feedback
        else compact_repair_feedback(repair_feedback),
        "instructions": base_instructions,
    }
    active_program = _compact_principle_program_for_context(state.get("current_principle_program"))
    if active_program:
        context["current_principle_program"] = active_program
    principle_book_tail = _compact_principle_book_for_context(
        state.get("principle_book"),
        entry_limit=1 if stage in {"mechanism", "prediction"} else 2,
        evidence_limit=1 if stage in {"mechanism", "prediction"} else 3,
    )
    if principle_book_tail:
        context["principle_book_tail"] = principle_book_tail
    if stage in {"mechanism", "prediction", "execution"}:
        context["mechanism_search_policy"] = mechanism_search_policy(state.get("current_round"))
    accepted_mechanisms = context["current_inputs"].get("accepted_mechanisms")
    if isinstance(accepted_mechanisms, list):
        accepted_ids = [
            str(item.get("id") or "").strip()
            for item in accepted_mechanisms
            if isinstance(item, Mapping) and str(item.get("id") or "").strip()
        ]
        if accepted_ids:
            context["current_inputs"]["accepted_mechanism_ids"] = accepted_ids
    if stage == "execution":
        context["generator_formula_probe_schema_reference"] = compact_generator_schema_reference()
        execution_instructions = [
            "A/B should propose and critique mechanisms.",
            "C/D should turn mechanisms into falsifiable predictions.",
            "E/F should turn predictions into concrete materialization bundles.",
            "Execution bundles must remain within the supported query schema and branch source schema; do not invent new query DSL fields.",
            "If E and F agree that the accepted C/D prediction itself cannot be faithfully materialized without changing the scientific test, return status='prediction_design_infeasible' with prediction_design_feedback so the controller can send the issue back to C/D.",
            "Only use mechanism_ids that appear in current_inputs.accepted_mechanism_ids; do not cite stale or rejected mechanism ids.",
        ]
        if materialization_backend == "mattergen":
            execution_instructions.extend(
                [
                    "MatterGen-native execution is strict: E/F must use source='mattergen' for every primary/control branch, or return prediction_design_infeasible. Do not fall back to source='generator' or source='mp_pool'.",
                    "If MatterGen materialization underfills, repair by changing MatterGen filters or sampling only. Increasing target_count/batch_size/num_batches on a failed branch is an execution-effort repair, not a physical confound, as long as chemical_system and physical filters stay fixed.",
                    f"For hard target_reduced_formula underfill or not_target_reduced_formula rejects, do not repeat the same exact-formula request with equal/lower sampling; increase sampling only within the controller cap target_count<={DEFAULT_MATTERGEN_MAX_TARGET_COUNT}, batch_size<={DEFAULT_MATTERGEN_MAX_BATCH_SIZE}, num_batches<={DEFAULT_MATTERGEN_MAX_NUM_BATCHES}, raw samples<={DEFAULT_MATTERGEN_MAX_RAW_SAMPLES}. If that capped budget is still insufficient, ask C/D for a chemical-system-only design if exact stoichiometry is not the causal variable.",
                    "MatterGen filters may only use controller-supported keys: chemical_system, require_chemical_system_exact, min_sites, max_sites, min_volume_per_atom, max_volume_per_atom, deduplicate_reduced_formula, target_reduced_formula, require_target_reduced_formula, exclude_reduced_formulas. Unsupported gates such as stoichiometry_gate, branch_predicate, derived_predicate, post_generation_filter, formula_ratio, or oxidation_state_gate make the plan invalid.",
                    "MatterGen exclude_reduced_formulas must be compact and limited to the active chemical_system; do not copy unrelated history formulas into a request.",
                ]
            )
        else:
            execution_instructions.extend(
                [
                    "If an MP-pool branch is not materializable but the test is scientifically necessary, E/F may use source='generator' with generator formula_probes/structure_dicts in legacy mode.",
                    "Mixed materialization is allowed: primary and control branches may independently use source='mp_pool', source='generator', or source='mattergen'.",
                    "When generator structure_dicts replace an underfilled MP-pool query, preserve numeric density/volume bounds in branch.query. The controller performs deterministic generator preflight: charge-balance checks and lattice rescaling to query density/volume windows.",
                    "Do not spend E/F debate turns repeatedly hand-scaling lattices. Fix composition and schema problems; let controller preflight handle density/volume scaling.",
                ]
            )
        context["instructions"].extend(execution_instructions)
    if materialization_backend == "mattergen" and stage in {"mechanism", "prediction", "execution"}:
        context["mattergen_native_mode"] = {
            "enabled": True,
            "role_boundary": (
                "A/B state mechanisms and MatterGen-testability; C/D design matched chemical-system tests; "
                "E/F emit MatterGen request blueprints only at execution stage."
            ),
            "do_not_use_as_shortcut": "Do not replace the accepted mechanism with a historically easy stable basin.",
        }
        context["instructions"].extend(
            [
                "MatterGen-native mode is enabled: use MatterGen as a conditional structure generator backend, not as a replacement for A/B mechanism reasoning.",
                "A/B should state whether the mechanism can be tested by chemical_system conditioning and what element/stoichiometry drift would invalidate the mechanism.",
                "C/D should express primary/control designs as matched chemical_system tests when possible, with exact counts, held-fixed variables, control chemistry, and drift/falsification criteria.",
                "E/F should use source='mattergen' with exactly one mattergen_requests object per branch when the accepted prediction is MatterGen-testable.",
                "MatterGen request must include backend='mattergen', properties_to_condition_on.chemical_system, properties_to_condition_on.energy_above_hull=0.0, diffusion_guidance_factor>0, and matching filters.chemical_system.",
                "MatterGen filters are not a predicate DSL. Do not encode stoichiometry, oxidation-state, O/P, Li/Mn, or formula-ratio gates in filters unless the controller explicitly supports that filter key.",
                "For A-F matched experiments, filters.require_chemical_system_exact must be true. Do not accept element-subset drift such as Li-P-O for a Li-Mn-P-O branch.",
                "Avoid narrow filters.min_sites lower bounds in MatterGen requests unless required by the mechanism. If prior MatterGen feedback reports too_few_sites, lower/remove min_sites rather than changing materialization source.",
                "MatterGen samples full structures and varied stoichiometries inside a chemical system; target_reduced_formula is a soft preference unless filters.require_target_reduced_formula=true.",
                f"Use hard require_target_reduced_formula only when exact stoichiometry is the causal variable; after hard-target underfill, prefer larger sampling within controller caps target_count<={DEFAULT_MATTERGEN_MAX_TARGET_COUNT}, batch_size<={DEFAULT_MATTERGEN_MAX_BATCH_SIZE}, num_batches<={DEFAULT_MATTERGEN_MAX_NUM_BATCHES}, raw samples<={DEFAULT_MATTERGEN_MAX_RAW_SAMPLES}, or a C/D redesign to exact chemical_system-only rather than repeating sibling exact formulas.",
                "MP-pool queries are optional sanity/RAG checks in MatterGen mode, not the primary materialization contract.",
            ]
        )
    if stage == "prediction" and isinstance(context["current_inputs"].get("target_count"), int):
        target_count = int(context["current_inputs"]["target_count"])
        context["instructions"].append(
            f"The downstream execution budget for this round is target_count={target_count}; C/D must make the accepted prediction set materializable with planned_material_count >= target_count, computed as sum(primary_count + control_count) over accepted predictions. Prefer exactly target_count when scientifically faithful."
        )
    if stage == "prediction" and repair_feedback:
        context["instructions"].extend(
            [
                "This prediction stage is being revisited after E/F execution review. Treat repair_feedback as binding execution feedback, not as optional commentary.",
                "If repair_feedback.status is prediction_design_infeasible, C/D must not repeat the same infeasible primary/control design. Revise the control family, primary family, counts, scope, or withdraw the prediction.",
                "Preserve the accepted A/B mechanism where possible, but the revised prediction must have at least one faithful materialization path through the enabled backend: MP pool, generator, or MatterGen.",
            ]
        )
    return context


def mechanism_search_policy(round_number: Any) -> dict[str, Any]:
    try:
        number = int(round_number)
    except (TypeError, ValueError):
        number = 1
    if number <= 0:
        number = 1
    slot = (number - 1) % 10
    if slot < 7:
        mode = "exploitation"
        directive = "Refine or falsify the strongest supported mechanism; include at least one near-miss or failure boundary so the proposal does not become blind repetition."
    elif slot < 9:
        mode = "neighbor_exploration"
        directive = "Explore a nearby mutation of a supported mechanism by changing one observable axis: cation family, anion set, element count, compactness, or control family."
    else:
        mode = "far_exploration"
        directive = "Explore a chemically plausible under-tested region outside recent repetition, guided by underexplored_clusters or near_misses from RAG."
    return {
        "schedule": "10_round_cycle",
        "cycle_fraction": {"exploitation": 0.7, "neighbor_exploration": 0.2, "far_exploration": 0.1},
        "round": number,
        "cycle_slot": slot + 1,
        "current_mode": mode,
        "directive": directive,
        "reviewer_obligation": "A/B reviewers must reject proposals that ignore the current mode without a concrete materials-physics reason.",
    }


def compact_repair_feedback(value: Any, *, max_list_items: int = 8) -> Any:
    """Keep execution repair feedback useful without replaying huge structures."""

    if isinstance(value, Mapping):
        if _looks_like_structure_dict(value):
            return summarize_structure_dict(value)
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if key == "structure_dicts" and isinstance(item, list):
                # In repair feedback these are summaries only. Do not keep the
                # executable field name, or later agents may copy summaries as
                # invalid generator structure_dicts.
                limit = max(max_list_items, PROMPT_MATERIALIZATION_ITEM_LIMIT)
                compact["structure_dict_summaries"] = [summarize_structure_dict(raw) for raw in item[:limit]]
                compact["structure_dicts_available_in_prior_payload"] = True
                if len(item) > limit:
                    compact["structure_dicts_omitted_count"] = len(item) - limit
                continue
            if key == "formula_probes" and isinstance(item, list):
                limit = max(max_list_items, PROMPT_MATERIALIZATION_ITEM_LIMIT)
                compact[key] = [compact_repair_feedback(raw, max_list_items=max_list_items) for raw in item[:limit]]
                if len(item) > limit:
                    compact[f"{key}_omitted_count"] = len(item) - limit
                continue
            compact[str(key)] = compact_repair_feedback(item, max_list_items=max_list_items)
        return compact
    if isinstance(value, list):
        return [compact_repair_feedback(item, max_list_items=max_list_items) for item in value[:max_list_items]] + (
            [{"omitted_count": len(value) - max_list_items}] if len(value) > max_list_items else []
        )
    if isinstance(value, str):
        return _short_text(value, 900)
    return value


def _looks_like_structure_dict(value: Mapping[str, Any]) -> bool:
    return (
        value.get("@class") == "Structure"
        or ("lattice" in value and "sites" in value)
        or ("charge" in value and "sites" in value and "lattice" in value)
    )


def summarize_structure_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        structure = Structure.from_dict(dict(value))
    except Exception as exc:
        return {"structure_parse_error": str(exc)[:240]}
    return structure_preflight_summary(structure)


def compact_structure_dict_for_prompt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep a parseable, minimal Structure.as_dict for agent-to-agent review."""

    try:
        structure = Structure.from_dict(dict(value))
    except Exception as exc:
        return {"structure_parse_error": str(exc)[:240], "raw_keys": sorted(str(key) for key in value.keys())[:16]}
    raw = structure.as_dict()
    lattice = raw.get("lattice", {})
    matrix = lattice.get("matrix", [])
    compact_matrix = [
        [round(float(cell), 6) for cell in row]
        for row in matrix
        if isinstance(row, list)
    ]
    sites: list[dict[str, Any]] = []
    for site in raw.get("sites", []):
        if not isinstance(site, Mapping):
            continue
        compact_site: dict[str, Any] = {
            "species": site.get("species"),
            "abc": [round(float(coord), 6) for coord in site.get("abc", [])],
        }
        sites.append({key: item for key, item in compact_site.items() if item not in (None, [], "")})
    return {
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "charge": raw.get("charge", 0),
        "lattice": {
            "matrix": compact_matrix,
            "pbc": lattice.get("pbc", [True, True, True]),
        },
        "sites": sites,
        "_prompt_summary": structure_preflight_summary(structure),
    }


def compact_formula_probe_for_prompt(value: Any) -> Any:
    if isinstance(value, Mapping):
        return compact_repair_feedback(value, max_list_items=PROMPT_MATERIALIZATION_ITEM_LIMIT)
    if isinstance(value, list):
        return [compact_formula_probe_for_prompt(item) for item in value[:PROMPT_MATERIALIZATION_ITEM_LIMIT]]
    if isinstance(value, str):
        return _short_text(value, 240)
    return value


def _compact_rejected_item_for_context(item: Any) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {"reason": _short_text(item, 360)}
    return {
        key: value
        for key, value in {
            "id": item.get("id"),
            "claim": _short_text(item.get("claim"), 240),
            "reason": _short_text(item.get("reason") or item.get("rejection_reason") or item.get("required_revision"), 360),
        }.items()
        if value not in (None, "")
    }


def _compact_execution_branch_for_context(branch: Any, bundle: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(branch, Mapping):
        return {"summary": _short_text(branch, 360)}
    compact_branch: dict[str, Any] = {
        "source": branch.get("source") or (bundle.get("source") if isinstance(bundle, Mapping) else None) or "mp_pool",
        "count": branch.get("count"),
    }
    if isinstance(branch.get("query"), Mapping):
        compact_branch["query"] = _compact_query_for_context(branch.get("query"))
    if isinstance(branch.get("material_ids"), list):
        material_ids = [str(item) for item in branch.get("material_ids", []) if str(item).strip()]
        compact_branch["material_ids"] = material_ids[:16]
        if len(material_ids) > 16:
            compact_branch["material_ids_omitted_count"] = len(material_ids) - 16
    if isinstance(branch.get("selection_order"), str):
        compact_branch["selection_order"] = branch.get("selection_order")
    if isinstance(branch.get("formula_probes"), list):
        formula_probes = branch.get("formula_probes") or []
        compact_branch["formula_probes"] = [
            compact_formula_probe_for_prompt(item) for item in formula_probes[:PROMPT_MATERIALIZATION_ITEM_LIMIT]
        ]
        if len(formula_probes) > PROMPT_MATERIALIZATION_ITEM_LIMIT:
            compact_branch["formula_probes_omitted_count"] = len(formula_probes) - PROMPT_MATERIALIZATION_ITEM_LIMIT
    if isinstance(branch.get("structure_dicts"), list):
        structure_dicts = branch.get("structure_dicts") or []
        compact_branch["structure_dicts"] = [
            compact_structure_dict_for_prompt(raw) if isinstance(raw, Mapping) else {"structure_parse_error": "structure_dict item is not an object"}
            for raw in structure_dicts[:PROMPT_MATERIALIZATION_ITEM_LIMIT]
        ]
        if len(structure_dicts) > PROMPT_MATERIALIZATION_ITEM_LIMIT:
            compact_branch["structure_dicts_omitted_count"] = len(structure_dicts) - PROMPT_MATERIALIZATION_ITEM_LIMIT
    if isinstance(branch.get("mattergen_requests"), list):
        requests = branch.get("mattergen_requests") or []
        compact_branch["mattergen_requests"] = [
            compact_mattergen_request_for_feedback(raw) for raw in requests[:2]
        ]
        if len(requests) > 2:
            compact_branch["mattergen_requests_omitted_count"] = len(requests) - 2
    return {key: value for key, value in compact_branch.items() if value not in (None, "", [])}


def compact_mattergen_request_for_feedback(request: Any) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        return {"summary": _short_text(request, 240)}
    properties = request.get("properties_to_condition_on")
    filters = request.get("filters")
    sampling = request.get("sampling") if isinstance(request.get("sampling"), Mapping) else {}
    result: dict[str, Any] = {
        "backend": request.get("backend") or "mattergen",
        "diffusion_guidance_factor": request.get("diffusion_guidance_factor"),
        "target_count": request.get("target_count", sampling.get("target_count")),
        "batch_size": request.get("batch_size", sampling.get("batch_size")),
        "num_batches": request.get("num_batches", sampling.get("num_batches")),
    }
    if isinstance(properties, Mapping):
        result["properties_to_condition_on"] = {
            key: properties.get(key)
            for key in ("chemical_system", "energy_above_hull")
            if properties.get(key) is not None
        }
    if isinstance(filters, Mapping):
        filter_keys = (
            "chemical_system",
            "require_chemical_system_exact",
            "min_sites",
            "max_sites",
            "min_volume_per_atom",
            "max_volume_per_atom",
            "deduplicate_reduced_formula",
            "target_reduced_formula",
            "require_target_reduced_formula",
        )
        compact_filters = {key: filters.get(key) for key in filter_keys if filters.get(key) is not None}
        excluded = filters.get("exclude_reduced_formulas")
        if isinstance(excluded, list) and excluded:
            compact_filters["exclude_reduced_formulas"] = [str(item) for item in excluded[:6]]
        result["filters"] = compact_filters
    return {key: value for key, value in result.items() if value not in (None, "", [], {})}


def _flatten_mattergen_sampling_aliases(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy/natural LLM `sampling` aliases into executable request fields."""

    result = dict(payload)
    for bundles_key in ("bundles", "accepted_bundles"):
        bundles = result.get(bundles_key)
        if not isinstance(bundles, list):
            continue
        normalized_bundles: list[Any] = []
        for bundle in bundles:
            if not isinstance(bundle, Mapping):
                normalized_bundles.append(bundle)
                continue
            normalized_bundle = dict(bundle)
            for role in ("primary", "control"):
                branch = normalized_bundle.get(role)
                if not isinstance(branch, Mapping):
                    continue
                normalized_branch = dict(branch)
                requests = normalized_branch.get("mattergen_requests")
                if isinstance(requests, list):
                    normalized_requests: list[Any] = []
                    for request in requests:
                        if not isinstance(request, Mapping):
                            normalized_requests.append(request)
                            continue
                        normalized_request = dict(request)
                        sampling = normalized_request.pop("sampling", None)
                        if isinstance(sampling, Mapping):
                            for key in ("target_count", "batch_size", "num_batches"):
                                if normalized_request.get(key) is None and sampling.get(key) is not None:
                                    normalized_request[key] = sampling.get(key)
                        normalized_requests.append(normalized_request)
                    normalized_branch["mattergen_requests"] = normalized_requests
                normalized_bundle[role] = normalized_branch
            normalized_bundles.append(normalized_bundle)
        result[bundles_key] = normalized_bundles
    return result


def _normalize_execution_bundle_item(item: Any, index: int) -> dict[str, Any]:
    """Normalize common LLM bundle aliases into the controller execution contract."""

    if isinstance(item, Mapping):
        normalized = dict(item)
    else:
        normalized = {"rationale_summary": str(item)}

    ordinal_id = f"b{index:03d}"
    bundle_id = normalized.get("id") or normalized.get("bundle_id")
    normalized["id"] = str(bundle_id or ordinal_id)

    if not isinstance(normalized.get("prediction_ids"), list):
        prediction_id = normalized.get("prediction_id")
        if isinstance(prediction_id, list):
            normalized["prediction_ids"] = [str(value) for value in prediction_id if str(value).strip()]
        elif prediction_id not in (None, ""):
            normalized["prediction_ids"] = [str(prediction_id)]

    for role in ("primary", "control"):
        if not isinstance(normalized.get(role), Mapping):
            for alias_key in (f"{role}_branch", f"{role}_group"):
                branch = normalized.get(alias_key)
                if isinstance(branch, Mapping):
                    normalized[role] = dict(branch)
                    break

    branches = normalized.get("branches")
    if isinstance(branches, Mapping):
        for role in ("primary", "control"):
            branch = branches.get(role)
            if isinstance(branch, Mapping) and not isinstance(normalized.get(role), Mapping):
                normalized[role] = dict(branch)

    return normalized


def _normalize_execution_bundle_list(items: Sequence[Any]) -> list[dict[str, Any]]:
    return [_normalize_execution_bundle_item(item, index) for index, item in enumerate(items, start=1)]


def _execution_bundle_aliases(bundle: Mapping[str, Any], index: int) -> set[str]:
    aliases: set[str] = set()
    for key in ("id", "bundle_id"):
        value = bundle.get(key)
        if value not in (None, ""):
            aliases.add(str(value))
    aliases.add(f"b{index:03d}")
    aliases.add(f"bundle_{index:03d}")
    prediction_id = bundle.get("prediction_id")
    if prediction_id not in (None, "") and not isinstance(prediction_id, list):
        aliases.add(str(prediction_id))
    prediction_ids = bundle.get("prediction_ids")
    if isinstance(prediction_ids, list) and len(prediction_ids) == 1 and prediction_ids[0] not in (None, ""):
        aliases.add(str(prediction_ids[0]))
    return {alias for alias in aliases if alias.strip()}


def _compact_execution_branch_for_final_prompt(branch: Any, bundle: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Summarize an execution branch for final consensus writing.

    The final consensus only needs enough information to identify the accepted
    bundle. Executable generator payloads are restored from the accepted E/F
    proposal by ``reconcile_execution_consensus``.
    """

    compact_branch = _compact_execution_branch_for_context(branch, bundle)
    if not isinstance(branch, Mapping):
        return compact_branch
    formula_probes = branch.get("formula_probes")
    if isinstance(formula_probes, list) and formula_probes:
        compact_branch.pop("formula_probes", None)
        compact_branch["formula_probe_count_in_prior_proposal"] = len(formula_probes)
        compact_branch["executable_formula_probes_stored_by_controller"] = True
    structure_dicts = branch.get("structure_dicts")
    if isinstance(structure_dicts, list) and structure_dicts:
        compact_branch.pop("structure_dicts", None)
        compact_branch["structure_dict_count_in_prior_proposal"] = len(structure_dicts)
        compact_branch["structure_summaries"] = [
            summarize_structure_dict(raw)
            for raw in structure_dicts[: min(3, PROMPT_MATERIALIZATION_ITEM_LIMIT)]
            if isinstance(raw, Mapping)
        ]
        compact_branch["executable_structure_dicts_stored_by_controller"] = True
    material_ids = branch.get("material_ids")
    if isinstance(material_ids, list) and material_ids:
        compact_branch.pop("material_ids", None)
        compact_branch["material_id_count_in_prior_proposal"] = len(material_ids)
        compact_branch["executable_material_ids_stored_by_controller"] = True
    mattergen_requests = branch.get("mattergen_requests")
    if isinstance(mattergen_requests, list) and mattergen_requests:
        compact_branch.pop("mattergen_requests", None)
        compact_branch["mattergen_request_count_in_prior_proposal"] = len(mattergen_requests)
        compact_branch["executable_mattergen_requests_stored_by_controller"] = True
    return {key: value for key, value in compact_branch.items() if value not in (None, "", [])}


def _compact_execution_bundle_for_context(bundle: Any) -> dict[str, Any]:
    if not isinstance(bundle, Mapping):
        return {"summary": _short_text(bundle, 360), "compact_payload_is_not_executable_original": True}
    rationale_text = str(bundle.get("rationale_summary") or bundle.get("rationale") or "").strip()
    selection_text = str(bundle.get("selection_notes") or "").strip()
    compact_bundle: dict[str, Any] = {
        "id": bundle.get("id"),
        "prediction_ids": bundle.get("prediction_ids"),
        "expected_relation": bundle.get("expected_relation"),
        "compact_payload_is_not_executable_original": True,
    }
    if rationale_text:
        compact_bundle["rationale_summary_stored_by_controller"] = True
        compact_bundle["rationale_summary_chars"] = len(rationale_text)
    if selection_text:
        compact_bundle["selection_notes_stored_by_controller"] = True
        compact_bundle["selection_notes_chars"] = len(selection_text)
    for role in ("primary", "control"):
        if role in bundle:
            compact_bundle[role] = _compact_execution_branch_for_context(bundle.get(role), bundle)
    return {key: value for key, value in compact_bundle.items() if value not in (None, "", [])}


def _compact_execution_bundle_for_final_prompt(bundle: Any) -> dict[str, Any]:
    if not isinstance(bundle, Mapping):
        return {"summary": _short_text(bundle, 360), "compact_payload_is_not_executable_original": True}
    rationale_text = str(bundle.get("rationale_summary") or bundle.get("rationale") or "").strip()
    selection_text = str(bundle.get("selection_notes") or "").strip()
    compact_bundle: dict[str, Any] = {
        "id": bundle.get("id"),
        "prediction_ids": bundle.get("prediction_ids"),
        "expected_relation": bundle.get("expected_relation"),
        "compact_payload_is_not_executable_original": True,
        "final_consensus_should_not_reject_missing_or_truncated_notes": True,
    }
    if rationale_text:
        compact_bundle["rationale_summary_stored_by_controller"] = True
        compact_bundle["rationale_summary_chars"] = len(rationale_text)
    if selection_text:
        compact_bundle["selection_notes_stored_by_controller"] = True
        compact_bundle["selection_notes_chars"] = len(selection_text)
    for role in ("primary", "control"):
        if role in bundle:
            compact_bundle[role] = _compact_execution_branch_for_final_prompt(bundle.get(role), bundle)
    return {key: value for key, value in compact_bundle.items() if value not in (None, "", [])}


def compact_execution_payload_for_final_prompt(value: Any, *, max_list_items: int = 4) -> Any:
    """Compact E/F payloads for final consensus prompts.

    This deliberately removes executable row/structure payloads from the prompt
    so the final consensus writer cannot copy huge ``structure_dicts`` back into
    a fragile JSON response. The controller keeps the original proposal in
    memory and restores executable branches by bundle id.
    """

    if isinstance(value, Mapping):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if key == "dialogue" and isinstance(item, list):
                compact["dialogue_turn_count"] = len(item)
                continue
            if key in {"bundles", "accepted_bundles"} and isinstance(item, list):
                limit = max(max_list_items, PROMPT_MATERIALIZATION_ITEM_LIMIT)
                compact[key] = [_compact_execution_bundle_for_final_prompt(raw) for raw in item[:limit]]
                if len(item) > limit:
                    compact[f"{key}_omitted_count"] = len(item) - limit
                continue
            if key == "rejected_bundles" and isinstance(item, list):
                compact[key] = [_compact_rejected_item_for_context(raw) for raw in item[:max_list_items]]
                if len(item) > max_list_items:
                    compact[f"{key}_omitted_count"] = len(item) - max_list_items
                continue
            if key in {
                "status",
                "agent",
                "agree",
                "concede",
                "accepted_bundle_ids",
                "rejected_bundle_ids",
                "required_revisions",
                "risk_flags",
            }:
                compact[key] = compact_repair_feedback(item, max_list_items=max_list_items)
                continue
            if key in {"consensus_summary", "critique_summary", "summary", "audit_summary", "impossibility_certificate"}:
                compact[key] = _short_text(item, 520)
                continue
            if key in {"formula_probes", "structure_dicts", "mattergen_requests", "material_ids"}:
                compact[f"{key}_count_in_prior_payload"] = len(item) if isinstance(item, list) else 1
                compact[f"executable_{key}_stored_by_controller"] = True
                continue
            compact[str(key)] = compact_repair_feedback(item, max_list_items=max_list_items)
        return {key: item for key, item in compact.items() if item not in (None, "", [])}
    if isinstance(value, list):
        return [compact_execution_payload_for_final_prompt(item, max_list_items=max_list_items) for item in value[:max_list_items]] + (
            [{"omitted_count": len(value) - max_list_items}] if len(value) > max_list_items else []
        )
    if isinstance(value, str):
        return _short_text(value, 900)
    return value


def _looks_like_prediction_payload(value: Mapping[str, Any]) -> bool:
    if str(value.get("agent") or "").strip() in {"C", "D"}:
        return True
    return any(
        key in value
        for key in (
            "predictions",
            "accepted_predictions",
            "rejected_predictions",
            "accepted_prediction_ids",
            "rejected_prediction_ids",
            "planned_material_count",
        )
    )


def _compact_prediction_payload_for_prompt(value: Mapping[str, Any], *, max_list_items: int = 4) -> dict[str, Any]:
    """Summarize C/D payloads without duplicating accepted_predictions and predictions."""

    compact: dict[str, Any] = {}
    for key in ("schema_version", "status", "agent", "agree", "concede", "planned_material_count", "target_count"):
        item = value.get(key)
        if item not in (None, "", []):
            compact[key] = item

    prediction_limit = max(max_list_items, PROMPT_MATERIALIZATION_ITEM_LIMIT)
    prediction_key = "predictions" if isinstance(value.get("predictions"), list) else None
    if prediction_key is None and isinstance(value.get("accepted_predictions"), list):
        prediction_key = "accepted_predictions"
    if prediction_key:
        predictions = value.get(prediction_key) or []
        compact[prediction_key] = [
            _compact_prediction_for_context(raw) for raw in predictions[:prediction_limit]
        ]
        if len(predictions) > prediction_limit:
            compact[f"{prediction_key}_omitted_count"] = len(predictions) - prediction_limit

    required_revisions = value.get("required_revisions")
    if isinstance(required_revisions, list):
        compact["required_revisions"] = [_short_text(item, 520) for item in required_revisions[:4]]
        if len(required_revisions) > 4:
            compact["required_revisions_omitted_count"] = len(required_revisions) - 4
    elif isinstance(value.get("rejected_predictions"), list):
        rejected = value.get("rejected_predictions") or []
        compact["rejected_predictions"] = [_compact_rejected_item_for_context(raw) for raw in rejected[:4]]
        if len(rejected) > 4:
            compact["rejected_predictions_omitted_count"] = len(rejected) - 4

    for key in ("accepted_prediction_ids", "rejected_prediction_ids"):
        item = value.get(key)
        if isinstance(item, list):
            compact[key] = item[:prediction_limit]
            if len(item) > prediction_limit:
                compact[f"{key}_omitted_count"] = len(item) - prediction_limit

    for key in ("consensus_summary", "critique_summary", "summary", "audit_summary", "impossibility_certificate"):
        item = value.get(key)
        if item not in (None, "", []):
            compact[key] = _short_text(item, 520)

    return {key: item for key, item in compact.items() if item not in (None, "", [])}


def _compact_previous_feedback_prediction(item: Any) -> dict[str, Any]:
    compact = _compact_prediction_for_context(item)
    compact.pop("falsification_criteria", None)
    compact.pop("scope", None)
    compact.pop("confidence", None)
    comparison = compact.get("comparison_design")
    if isinstance(comparison, dict) and isinstance(comparison.get("matching_notes"), list):
        comparison["matching_notes"] = comparison["matching_notes"][:1]
    return compact


def compact_prediction_design_feedback_for_prompt(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return compact_repair_feedback(value)
    if str(value.get("status") or "").strip().lower() != PREDICTION_DESIGN_INFEASIBLE_STATUS:
        return compact_repair_feedback(value)

    compact: dict[str, Any] = {
        "status": PREDICTION_DESIGN_INFEASIBLE_STATUS,
        "feedback_round": value.get("feedback_round"),
        "prediction_ids": value.get("prediction_ids"),
        "blocking_issue": _short_text(value.get("blocking_issue"), 700),
        "why_execution_cannot_fix_it": compact_repair_feedback(
            value.get("why_execution_cannot_fix_it"),
            max_list_items=4,
        ),
        "required_cd_reconsideration": compact_repair_feedback(
            value.get("required_cd_reconsideration"),
            max_list_items=4,
        ),
        "execution_consensus_summary": _short_text(value.get("execution_consensus_summary"), 520),
    }
    previous_prediction_summary = value.get("previous_prediction_summary")
    if isinstance(previous_prediction_summary, Mapping):
        previous_compact = _compact_prediction_payload_for_prompt(
            previous_prediction_summary,
            max_list_items=PROMPT_MATERIALIZATION_ITEM_LIMIT,
        )
        for redundant_key in ("rejected_predictions", "consensus_summary", "critique_summary", "summary", "audit_summary"):
            previous_compact.pop(redundant_key, None)
        for prediction_key in ("predictions", "accepted_predictions"):
            predictions = previous_compact.get(prediction_key)
            if isinstance(predictions, list):
                previous_compact[prediction_key] = [
                    _compact_previous_feedback_prediction(item) for item in predictions
                ]
        compact["previous_prediction_summary"] = previous_compact
    return {key: item for key, item in compact.items() if item not in (None, "", [])}


def compact_dialogue_payload_for_prompt(value: Any, *, max_list_items: int = 2) -> Any:
    """Summarize prior agent outputs before embedding them into another LLM prompt."""

    if isinstance(value, Mapping):
        if _looks_like_prediction_payload(value):
            return _compact_prediction_payload_for_prompt(value, max_list_items=max_list_items)
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if key == "dialogue" and isinstance(item, list):
                compact["dialogue_turn_count"] = len(item)
                continue
            if key in {"mechanisms", "accepted_mechanisms"} and isinstance(item, list):
                compact[key] = [_compact_mechanism_for_context(raw) for raw in item[:max_list_items]]
                if len(item) > max_list_items:
                    compact[f"{key}_omitted_count"] = len(item) - max_list_items
                continue
            if key == "rejected_mechanisms" and isinstance(item, list):
                compact[key] = [_compact_rejected_item_for_context(raw) for raw in item[:max_list_items]]
                if len(item) > max_list_items:
                    compact[f"{key}_omitted_count"] = len(item) - max_list_items
                continue
            if key in {"predictions", "accepted_predictions"} and isinstance(item, list):
                limit = max(max_list_items, PROMPT_MATERIALIZATION_ITEM_LIMIT)
                compact[key] = [_compact_prediction_for_context(raw) for raw in item[:limit]]
                if len(item) > limit:
                    compact[f"{key}_omitted_count"] = len(item) - limit
                continue
            if key == "rejected_predictions" and isinstance(item, list):
                compact[key] = [_compact_rejected_item_for_context(raw) for raw in item[:max_list_items]]
                if len(item) > max_list_items:
                    compact[f"{key}_omitted_count"] = len(item) - max_list_items
                continue
            if key in {"bundles", "accepted_bundles"} and isinstance(item, list):
                limit = max(max_list_items, PROMPT_MATERIALIZATION_ITEM_LIMIT)
                compact[key] = [_compact_execution_bundle_for_context(raw) for raw in item[:limit]]
                if len(item) > limit:
                    compact[f"{key}_omitted_count"] = len(item) - limit
                continue
            if key == "rejected_bundles" and isinstance(item, list):
                compact[key] = [_compact_rejected_item_for_context(raw) for raw in item[:max_list_items]]
                if len(item) > max_list_items:
                    compact[f"{key}_omitted_count"] = len(item) - max_list_items
                continue
            compact[str(key)] = compact_repair_feedback(item, max_list_items=max_list_items)
        return compact
    if isinstance(value, list):
        return [compact_dialogue_payload_for_prompt(item, max_list_items=max_list_items) for item in value[:max_list_items]] + (
            [{"omitted_count": len(value) - max_list_items}] if len(value) > max_list_items else []
        )
    if isinstance(value, str):
        return _short_text(value, 900)
    return value


def prompt_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=PROMPT_JSON_SEPARATORS)


def context_with_debate_history(
    context_json: str,
    artifacts: Sequence[Mapping[str, Any]],
    *,
    max_turns: int = 4,
) -> str:
    """Add compact debate history without replaying raw data payloads."""

    try:
        context = json.loads(context_json)
    except Exception:
        context = {"base_context": _short_text(context_json, 2000)}
    if not isinstance(context, dict):
        context = {"base_context": compact_dialogue_payload_for_prompt(context)}
    stage = str(context.get("stage") or "").strip().lower()
    if stage == "mechanism":
        for noisy_key in ("candidate_pool_summary", "required_output_schema_reference", "instructions", "directive"):
            context.pop(noisy_key, None)
        state_summary = context.get("state")
        if isinstance(state_summary, Mapping):
            context["state"] = {
                key: state_summary.get(key)
                for key in ("round", "current_round", "history_len", "status")
                if state_summary.get(key) not in (None, "", [], {})
            }
    if stage == "prediction":
        for noisy_key in ("instructions", "directive", "mattergen_native_mode"):
            context.pop(noisy_key, None)
        current_inputs = context.get("current_inputs")
        if isinstance(current_inputs, Mapping):
            compact_inputs = compact_current_inputs_for_stage("prediction", current_inputs)
            context["current_inputs"] = {
                key: compact_inputs.get(key)
                for key in (
                    "target_count",
                    "materialization_backend",
                    "accepted_mechanism_ids",
                    "accepted_mechanisms",
                )
                if compact_inputs.get(key) not in (None, "", [], {})
            }
        context["required_output_schema_reference"] = compact_prediction_schema_reference_for_repair()
        state_summary = context.get("state")
        if isinstance(state_summary, Mapping):
            context["state"] = {
                key: state_summary.get(key)
                for key in ("current_round", "status", "best_support_rate", "latest_support_rate")
                if state_summary.get(key) not in (None, "", [], {})
            }
        policy = context.get("mechanism_search_policy")
        if isinstance(policy, Mapping):
            context["mechanism_search_policy"] = {
                key: policy.get(key)
                for key in ("current_mode", "directive", "round")
                if policy.get(key) not in (None, "", [], {})
            }
        pool = context.get("candidate_pool_summary")
        if isinstance(pool, Mapping):
            context["candidate_pool_summary"] = {
                key: pool.get(key)
                for key in ("count", "common_elements", "common_elements_omitted_count")
                if pool.get(key) not in (None, "", [], {})
            }
        program = context.get("current_principle_program")
        if isinstance(program, Mapping):
            context["current_principle_program"] = {
                key: (
                    _short_text(program.get(key), 120)
                    if key in {"current_principle_statement", "micro_mechanism"}
                    else program.get(key)
                )
                for key in ("program_id", "current_principle_statement", "micro_mechanism", "active_mechanism_ids")
                if program.get(key) not in (None, "", [], {})
            }
        book = context.get("principle_book_tail")
        if isinstance(book, list):
            compact_book: list[dict[str, Any]] = []
            for entry in book[-1:]:
                if not isinstance(entry, Mapping):
                    continue
                compact_book.append(
                    {
                        key: _short_text(entry.get(key), 120) if key in {"topic_key", "principle_statement", "micro_mechanism"} else entry.get(key)
                        for key in ("program_id", "topic_key", "principle_statement", "micro_mechanism", "status")
                        if entry.get(key) not in (None, "", [], {})
                    }
                )
            if compact_book:
                context["principle_book_tail"] = compact_book
            else:
                context.pop("principle_book_tail", None)
        history_turns = min(max_turns, 2)
    else:
        history_turns = 1 if stage == "mechanism" else max_turns
    compact_turns: list[dict[str, Any]] = []
    for turn in artifacts[-history_turns:]:
        if not isinstance(turn, Mapping):
            continue
        item: dict[str, Any] = {
            "role": turn.get("role"),
            "cycle": turn.get("cycle", turn.get("attempt")),
            "mode": turn.get("mode", "proposal_or_review"),
        }
        payload = turn.get("payload")
        if isinstance(payload, Mapping):
            if stage == "mechanism":
                item["payload"] = _compact_mechanism_payload_for_prompt(payload, max_mechanisms=1)
            elif any(key in payload for key in ("bundles", "accepted_bundles", "structure_dicts", "formula_probes")):
                item["payload"] = compact_execution_payload_for_final_prompt(payload, max_list_items=3)
            else:
                item["payload"] = compact_dialogue_payload_for_prompt(payload, max_list_items=3)
        compact_turns.append({key: value for key, value in item.items() if value not in (None, "", [])})
    if stage == "mechanism":
        context["debate_protocol"] = {
            "mode": "strict_alternating_proposal_review",
            "rule": "Rejected reviewer must counterpropose; other side reviews; stop on acceptance or budget.",
            "rag_policy": "Use local RAG tools; final JSON cites compact evidence only.",
        }
    elif stage == "prediction":
        context["debate_protocol"] = {
            "mode": "strict_alternating_proposal_review",
            "rule": "Reviewer may counterpropose; other side reviews; stop on acceptance or budget.",
            "rag_policy": "Use RAG/tools for compact evidence only; final JSON must not paste raw search output.",
        }
    else:
        context["debate_protocol"] = {
            "mode": "strict_alternating_proposal_review",
            "rule": (
                "If a reviewer rejects the current proposal, the reviewer must give a constructive counterproposal. "
                "The other side then reviews that counterproposal. Stop only when one side accepts the other side's proposal, "
                "or when max_dialogue_rounds is exhausted."
            ),
            "rag_policy": (
                "Do not ask the controller to paste raw datasets into the prompt. Use local agent tools for RAG or availability checks, "
                "then cite only compact evidence summaries in the final JSON."
            ),
            "compact_payload_policy": (
                "Debate-history and final-consensus payloads marked compact_payload_is_not_executable_original are controller summaries, "
                "not the raw executable proposal. Do not reject a plan solely because rationale_summary or selection_notes are stored by the "
                "controller, omitted from the compact prompt, or represented by *_chars metadata. Only treat literal truncation as a blocker "
                "when it appears in the current raw Agent JSON field that the controller asks you to revise."
            ),
        }
    context["debate_history"] = compact_turns
    return prompt_json_dumps(context)


def _prompt_json(value: Mapping[str, Any]) -> str:
    compact = compact_dialogue_payload_for_prompt(value)
    if isinstance(value, Mapping) and _looks_like_prediction_payload(value):
        return prompt_json_dumps(compact)
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _compact_prediction_payload_for_peer_prompt(
    value: Mapping[str, Any], *, max_predictions: int | None = None
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("schema_version", "status", "agent", "agree", "concede", "planned_material_count", "target_count"):
        item = value.get(key)
        if item not in (None, "", []):
            compact[key] = item
    prediction_key = "predictions" if isinstance(value.get("predictions"), list) else None
    if prediction_key is None and isinstance(value.get("accepted_predictions"), list):
        prediction_key = "accepted_predictions"
    if prediction_key:
        predictions = value.get(prediction_key) or []
        selected_predictions = predictions if max_predictions is None else predictions[:max_predictions]
        compact[prediction_key] = [_compact_prediction_for_context(raw) for raw in selected_predictions]
        if max_predictions is not None and len(predictions) > max_predictions:
            compact[f"{prediction_key}_omitted_count"] = len(predictions) - max_predictions
    revisions = value.get("required_revisions")
    if isinstance(revisions, list):
        compact["required_revisions"] = [_short_text(item, 260) for item in revisions[:2]]
        if len(revisions) > 2:
            compact["required_revisions_omitted_count"] = len(revisions) - 2
    for key in ("rejected_predictions", "accepted_prediction_ids", "rejected_prediction_ids"):
        item = value.get(key)
        if isinstance(item, list):
            compact[key] = compact_repair_feedback(item[:2], max_list_items=2)
            if len(item) > 2:
                compact[f"{key}_omitted_count"] = len(item) - 2
    for key in ("consensus_summary", "critique_summary", "summary", "audit_summary", "impossibility_certificate"):
        item = value.get(key)
        if item not in (None, "", []):
            compact[key] = _short_text(item, 320)
    return {key: item for key, item in compact.items() if item not in (None, "", [], {})}


def _prompt_prediction_peer_json(value: Mapping[str, Any]) -> str:
    return prompt_json_dumps(_compact_prediction_payload_for_peer_prompt(value))


def compact_prediction_context_json_for_prompt(context_json: str) -> str:
    try:
        context = json.loads(context_json)
    except Exception:
        return _short_text(context_json, 2400)
    if not isinstance(context, Mapping):
        return prompt_json_dumps(compact_dialogue_payload_for_prompt(context))
    current_inputs = context.get("current_inputs")
    compact_inputs: dict[str, Any] = {}
    if isinstance(current_inputs, Mapping):
        base_inputs = compact_current_inputs_for_stage("prediction", current_inputs)
        compact_inputs = {
            key: base_inputs.get(key)
            for key in ("target_count", "materialization_backend", "accepted_mechanism_ids", "accepted_mechanisms")
            if base_inputs.get(key) not in (None, "", [], {})
        }
    repair_feedback = context.get("repair_feedback")
    program = context.get("current_principle_program")
    policy = context.get("mechanism_search_policy")
    compact: dict[str, Any] = {
        key: value
        for key, value in {
            "schema_version": context.get("schema_version"),
            "stage": context.get("stage"),
            "materialization_backend": context.get("materialization_backend"),
            "current_inputs": compact_inputs,
            "repair_feedback": compact_repair_feedback(repair_feedback, max_list_items=3)
            if repair_feedback not in (None, "", [], {})
            else None,
            "current_principle_program": {
                key: _short_text(program.get(key), 140) if key in {"current_principle_statement", "micro_mechanism"} else program.get(key)
                for key in ("program_id", "current_principle_statement", "micro_mechanism", "active_mechanism_ids")
                if isinstance(program, Mapping) and program.get(key) not in (None, "", [], {})
            },
            "mechanism_search_policy": {
                key: policy.get(key)
                for key in ("current_mode", "directive", "round")
                if isinstance(policy, Mapping) and policy.get(key) not in (None, "", [], {})
            },
            "debate_protocol": context.get("debate_protocol"),
        }.items()
        if value not in (None, "", [], {})
    }
    history = context.get("debate_history")
    if isinstance(history, list) and history:
        compact_history: list[dict[str, Any]] = []
        for turn in history[-1:]:
            if not isinstance(turn, Mapping):
                continue
            item = {
                "role": turn.get("role"),
                "cycle": turn.get("cycle"),
                "mode": turn.get("mode"),
            }
            payload = turn.get("payload")
            if isinstance(payload, Mapping) and _looks_like_prediction_payload(payload):
                item["payload"] = _compact_prediction_payload_for_peer_prompt(payload, max_predictions=1)
            elif isinstance(payload, Mapping):
                item["payload"] = compact_dialogue_payload_for_prompt(payload, max_list_items=1)
            compact_history.append({key: value for key, value in item.items() if value not in (None, "", [], {})})
        if compact_history:
            compact["debate_history"] = compact_history
    return prompt_json_dumps(compact)


def _compact_execution_payload_for_peer_prompt(
    value: Mapping[str, Any], *, max_bundles: int | None = None
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("schema_version", "status", "agent", "agree", "concede"):
        item = value.get(key)
        if item not in (None, "", []):
            compact[key] = item

    for key in ("accepted_bundle_ids", "audited_bundle_ids", "rejected_bundle_ids"):
        item = value.get(key)
        if isinstance(item, list):
            compact[key] = item[:12]
            if len(item) > 12:
                compact[f"{key}_omitted_count"] = len(item) - 12

    bundle_key = "bundles" if isinstance(value.get("bundles"), list) else None
    if bundle_key is None and isinstance(value.get("accepted_bundles"), list):
        bundle_key = "accepted_bundles"
    if bundle_key:
        bundles = value.get(bundle_key) or []
        selected_bundles = bundles if max_bundles is None else bundles[:max_bundles]
        compact[bundle_key] = [_compact_execution_bundle_for_context(raw) for raw in selected_bundles]
        if max_bundles is not None and len(bundles) > max_bundles:
            compact[f"{bundle_key}_omitted_count"] = len(bundles) - max_bundles

    rejected = value.get("rejected_bundles")
    if isinstance(rejected, list):
        compact["rejected_bundles"] = [_compact_rejected_item_for_context(raw) for raw in rejected[:4]]
        if len(rejected) > 4:
            compact["rejected_bundles_omitted_count"] = len(rejected) - 4

    revisions = value.get("required_revisions")
    if isinstance(revisions, list):
        compact["required_revisions"] = [_short_text(item, 420) for item in revisions[:4]]
        if len(revisions) > 4:
            compact["required_revisions_omitted_count"] = len(revisions) - 4

    counterproposal = value.get("constructive_counterproposal")
    if isinstance(counterproposal, Mapping):
        compact["constructive_counterproposal"] = _compact_execution_payload_for_peer_prompt(
            counterproposal,
            max_bundles=2,
        )

    feedback = value.get("prediction_design_feedback")
    if feedback not in (None, "", [], {}):
        compact["prediction_design_feedback"] = compact_prediction_design_feedback_for_prompt(feedback)

    for key in ("matching_notes", "materialization_constraints"):
        item = value.get(key)
        if item not in (None, "", [], {}):
            compact[key] = compact_repair_feedback(item, max_list_items=3)

    for key in ("consensus_summary", "critique_summary", "summary", "audit_summary", "impossibility_certificate"):
        item = value.get(key)
        if item not in (None, "", []):
            compact[key] = _short_text(item, 420)
    return {key: item for key, item in compact.items() if item not in (None, "", [], {})}


def _prompt_execution_peer_json(value: Mapping[str, Any]) -> str:
    return json.dumps(_compact_execution_payload_for_peer_prompt(value), ensure_ascii=False, indent=2)


def compact_execution_context_json_for_prompt(context_json: str) -> str:
    try:
        context = json.loads(context_json)
    except Exception:
        return _short_text(context_json, 3200)
    if not isinstance(context, Mapping):
        return prompt_json_dumps(compact_dialogue_payload_for_prompt(context))

    current_inputs = context.get("current_inputs")
    compact_inputs: dict[str, Any] = {}
    if isinstance(current_inputs, Mapping):
        base_inputs = compact_current_inputs_for_stage("execution", current_inputs)
        for key in ("accepted_predictions", "target_count", "materialization_backend", "accepted_mechanism_ids"):
            item = base_inputs.get(key)
            if item not in (None, "", [], {}):
                compact_inputs[key] = item
        defaults = base_inputs.get("mattergen_defaults")
        if defaults not in (None, "", [], {}):
            compact_inputs["mattergen_defaults"] = compact_repair_feedback(defaults, max_list_items=4)

    repair_feedback = context.get("repair_feedback")
    compact_repair = None
    if repair_feedback not in (None, "", [], {}):
        compact_repair = (
            compact_prediction_design_feedback_for_prompt(repair_feedback)
            if isinstance(repair_feedback, Mapping)
            and str(repair_feedback.get("status") or "").strip().lower() == PREDICTION_DESIGN_INFEASIBLE_STATUS
            else compact_repair_feedback(repair_feedback, max_list_items=4)
        )

    state = context.get("state")
    compact_state: dict[str, Any] = {}
    if isinstance(state, Mapping):
        compact_state = {
            key: state.get(key)
            for key in ("status", "current_round", "best_round", "best_support_rate", "latest_support_rate")
            if state.get(key) not in (None, "", [], {})
        }
        history_tail = state.get("history_tail")
        if isinstance(history_tail, list) and history_tail:
            compact_state["history_tail"] = compact_repair_feedback(history_tail[-1:], max_list_items=1)

    program = context.get("current_principle_program")
    compact_program: dict[str, Any] = {}
    if isinstance(program, Mapping):
        compact_program = {
            key: (
                _short_text(program.get(key), 180)
                if key in {"current_principle_statement", "micro_mechanism"}
                else compact_repair_feedback(program.get(key), max_list_items=2)
                if key in {"failure_boundaries", "unresolved_contradictions"}
                else program.get(key)
            )
            for key in (
                "program_id",
                "current_principle_statement",
                "micro_mechanism",
                "active_mechanism_ids",
                "failure_boundaries",
                "unresolved_contradictions",
            )
            if program.get(key) not in (None, "", [], {})
        }

    policy = context.get("mechanism_search_policy")
    compact_policy: dict[str, Any] = {}
    if isinstance(policy, Mapping):
        compact_policy = {
            key: policy.get(key)
            for key in ("current_mode", "directive", "round", "cycle_slot")
            if policy.get(key) not in (None, "", [], {})
        }

    compact: dict[str, Any] = {
        key: value
        for key, value in {
            "schema_version": context.get("schema_version"),
            "stage": context.get("stage"),
            "materialization_backend": context.get("materialization_backend"),
            "current_inputs": compact_inputs,
            "repair_feedback": compact_repair,
            "state": compact_state,
            "current_principle_program": compact_program,
            "mechanism_search_policy": compact_policy,
            "mattergen_native_mode": compact_repair_feedback(context.get("mattergen_native_mode"), max_list_items=3)
            if context.get("mattergen_native_mode") not in (None, "", [], {})
            else None,
            "candidate_pool_summary": compact_pool_summary_for_context(context.get("candidate_pool_summary"))
            if isinstance(context.get("candidate_pool_summary"), Mapping)
            else None,
            "debate_protocol": compact_repair_feedback(context.get("debate_protocol"), max_list_items=3)
            if context.get("debate_protocol") not in (None, "", [], {})
            else None,
        }.items()
        if value not in (None, "", [], {})
    }

    history = context.get("debate_history")
    if isinstance(history, list) and history:
        compact_history: list[dict[str, Any]] = []
        for turn in history[-1:]:
            if not isinstance(turn, Mapping):
                continue
            item: dict[str, Any] = {
                "role": turn.get("role"),
                "cycle": turn.get("cycle"),
                "mode": turn.get("mode"),
            }
            payload = turn.get("payload")
            if isinstance(payload, Mapping):
                item["payload_status"] = payload.get("status")
                item["payload_agent"] = payload.get("agent")
                item["payload_agree"] = payload.get("agree")
                if isinstance(payload.get("bundles"), list):
                    item["bundle_count"] = len(payload.get("bundles") or [])
                elif isinstance(payload.get("accepted_bundles"), list):
                    item["accepted_bundle_count"] = len(payload.get("accepted_bundles") or [])
                if isinstance(payload.get("required_revisions"), list):
                    item["required_revision_count"] = len(payload.get("required_revisions") or [])
            compact_history.append({key: value for key, value in item.items() if value not in (None, "", [], {})})
        if compact_history:
            compact["debate_history"] = compact_history
    return prompt_json_dumps(compact)


def _prompt_mechanism_json(value: Mapping[str, Any], *, max_mechanisms: int = 1) -> str:
    return prompt_json_dumps(_compact_mechanism_payload_for_prompt(value, max_mechanisms=max_mechanisms))


def _prompt_execution_final_json(value: Mapping[str, Any]) -> str:
    return json.dumps(compact_execution_payload_for_final_prompt(value), ensure_ascii=False, indent=2)


def _context_search_mode(context_json: str) -> str:
    try:
        context = json.loads(context_json)
    except Exception:
        return ""
    if not isinstance(context, Mapping):
        return ""
    policy = context.get("mechanism_search_policy")
    if not isinstance(policy, Mapping):
        return ""
    return str(policy.get("current_mode") or "").strip().lower()


def _cdef_exploration_requirement(context_json: str) -> str:
    mode = _context_search_mode(context_json)
    if mode in {"neighbor_exploration", "far_exploration"}:
        return f"{CDEF_EXPLORATION_POLICY}\n\n{CDEF_MANDATORY_EXPLORATION_RAG}"
    return CDEF_EXPLORATION_POLICY


def _cd_prediction_exploration_requirement(context_json: str) -> str:
    mode = _context_search_mode(context_json)
    if mode in {"neighbor_exploration", "far_exploration"}:
        return f"{CD_PREDICTION_EXPLORATION_POLICY}\n\n{CD_MANDATORY_EXPLORATION_RAG}"
    return CD_PREDICTION_EXPLORATION_POLICY


def make_client(
    *,
    role: str,
    dotenv: Path,
    model: str | None,
    temperature: float | None,
    max_tokens: int | None,
    log_dir: Path | None = None,
) -> ResponsesClient:
    return ResponsesClient(
        LLMConfig.from_env(
            dotenv=dotenv,
            role=role,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        log_dir=log_dir,
    )


def call_json(
    client: ResponsesClient,
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
    role_recoverable_retries = max(0, _env_int("LLM_ROLE_RECOVERABLE_RETRIES", 0))
    role_recoverable_retry_sleep = max(0.0, _env_float("LLM_ROLE_RECOVERABLE_RETRY_SLEEP", 30.0))
    role_recoverable_attempts = 0
    attempt = 0
    while attempt <= json_repair_attempts:
        try:
            call_metadata = {**dict(metadata), "json_retry_attempt": attempt}
            local_agent_runtime = getattr(client, "local_agent_runtime", None)
            if isinstance(local_agent_runtime, LocalAgentRuntime):
                text = local_agent_runtime.complete_text(
                    client,
                    system=system,
                    user=prompt,
                    metadata=call_metadata,
                )
            else:
                text = client.complete_text(system=system, user=prompt, metadata=call_metadata)
            last_text = text
        except LLMError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if (
                is_recoverable_llm_error_message(last_error)
                and role_recoverable_attempts < role_recoverable_retries
            ):
                role_recoverable_attempts += 1
                sleep_seconds = role_recoverable_retry_sleep * min(role_recoverable_attempts, 3)
                print(
                    f"[{utc_now()}] recoverable_llm_role_retry role={role} "
                    f"attempt={role_recoverable_attempts}/{role_recoverable_retries} "
                    f"json_retry_attempt={attempt} sleep_sec={sleep_seconds:.1f} "
                    f"error={last_error.replace(chr(10), ' ')[:500]}",
                    flush=True,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                continue
            if attempt < json_repair_attempts:
                prompt = transient_llm_retry_prompt(role, user, last_error)
                attempt += 1
                continue
            if is_recoverable_llm_error_message(last_error):
                raise RecoverableLLMFailure(
                    role=role,
                    metadata=metadata,
                    error=last_error,
                    attempts=json_repair_attempts + 1 + role_recoverable_attempts,
                ) from exc
            raise ValueError(f"LLM call for {role} failed after retry: {last_error}") from exc
        try:
            parsed = extract_json_object(text)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            parsed = normalize_role_payload(parsed, role)
            if valid_shape(parsed, role):
                return dict(parsed)
            last_error = f"output did not match required shape for {role}"
        if attempt < json_repair_attempts:
            prompt = retry_prompt(role, user, last_text, last_error)
        attempt += 1
    raise ValueError(f"LLM output for {role} could not be repaired: {last_error}")


def normalize_role_payload(parsed: Any, role: str) -> Any:
    """Accept semantically valid LLM JSON variants and coerce to our control shape."""

    if not isinstance(parsed, Mapping):
        return parsed
    role = _schema_role_for_output(role)
    payload = dict(parsed)
    role_wrapper_keys = {
        "mechanism_agent_a": "material_physics_mechanism_consensus",
        "mechanism_agent_b": "material_physics_mechanism_consensus",
        "mechanism_consensus": "material_physics_mechanism_consensus",
        "prediction_agent_c": "material_physics_prediction_consensus",
        "prediction_agent_d": "material_physics_prediction_consensus",
        "prediction_consensus": "material_physics_prediction_consensus",
        "execution_agent_e": "material_physics_execution_plan",
        "execution_agent_f": "material_physics_execution_plan",
        "execution_consensus": "material_physics_execution_plan",
        "principle_postmortem": "material_physics_principle_postmortem",
    }
    wrapper_key = role_wrapper_keys.get(role)
    if wrapper_key and isinstance(payload.get(wrapper_key), Mapping):
        payload = dict(payload[wrapper_key])
    if len(payload) == 1:
        only_key, only_value = next(iter(payload.items()))
        if isinstance(only_key, str) and only_key.startswith("material_physics_") and isinstance(only_value, Mapping):
            payload = dict(only_value)

    if role == "mechanism_agent_a":
        _ensure_agent(payload, "A", "mechanism_agent_a")
        if str(payload.get("status") or "").strip().lower() in {"success", "ok"} and isinstance(payload.get("mechanisms"), list):
            payload["status"] = "proposal"
        if payload.get("status") == "consensus" and "agent" not in payload:
            payload["agent"] = "A"
        mechanisms = _first_list(
            payload,
            (
                "mechanisms",
                "mechanism_hypotheses",
                "hypotheses",
                "accepted_mechanisms",
                "accepted_hypotheses",
            ),
        )
        if mechanisms is not None:
            payload["mechanisms"] = [_normalize_mechanism_item(item, index) for index, item in enumerate(mechanisms, start=1)]
        if not isinstance(payload.get("mechanisms"), list) and payload.get("status") == "consensus":
            accepted = payload.get("accepted_mechanisms")
            if isinstance(accepted, list):
                payload["mechanisms"] = [_normalize_mechanism_item(item, index) for index, item in enumerate(accepted, start=1)]
        if payload.get("status") == "consensus":
            payload["agree"] = True
        payload.setdefault("agree", False)
        payload.setdefault("concede", False)

    if role == "mechanism_agent_b":
        _ensure_agent(payload, "B", "mechanism_agent_b")
        if payload.get("status") in {"critique", "review", "audit", "consensus"} | NONCONSENSUS_STATUSES and "agent" not in payload:
            payload["agent"] = "B"
    if role == "mechanism_agent_b" and payload.get("agent") == "B":
        critiques = payload.get("critiques")
        if not isinstance(critiques, list):
            critiques = payload.get("mechanism_critiques")
        if not isinstance(critiques, list):
            critiques = payload.get("major_issues")
        if not isinstance(critiques, list) and payload.get("status") == "consensus":
            critiques = []
            for key in ("accepted_mechanisms", "rejected_mechanisms"):
                value = payload.get(key)
                if isinstance(value, list):
                    critiques.extend(value)
        if not isinstance(critiques, list):
            critiques = list(payload.get("accepted_mechanisms") or []) + list(payload.get("rejected_mechanisms") or [])
        if isinstance(critiques, list) and "required_revisions" not in payload:
            revisions: list[str] = []
            counterexamples: list[str] = []
            evidence_chain: list[dict[str, str]] = []
            for item in critiques:
                if not isinstance(item, Mapping):
                    revisions.append(str(item))
                    continue
                issue = str(item.get("issue") or item.get("id") or item.get("verdict") or "critique")
                critique = str(item.get("critique") or item.get("reason") or item.get("evidence_summary") or "")
                refinement = str(item.get("needed_refinement") or item.get("required_revision") or item.get("proposed_revision") or "")
                if refinement:
                    revisions.append(f"{issue}: {refinement}")
                elif critique:
                    revisions.append(f"{issue}: {critique}")
                if item.get("counterexample"):
                    counterexamples.append(str(item["counterexample"]))
                evidence_chain.append(
                    {
                        "premise": issue,
                        "implication": critique or refinement or "Needs sharper mechanism boundary.",
                        "confidence": "medium",
                    }
                )
            payload.setdefault("agree", False)
            payload.setdefault("concede", False)
            if payload.get("status") == "consensus":
                payload["agree"] = True
            payload.setdefault("critique_summary", str(payload.get("overall_assessment") or "Mechanism critique provided."))
            payload["required_revisions"] = revisions or ["Narrow the mechanisms into directly testable, observable claims."]
            payload.setdefault("evidence_chain", evidence_chain)
            payload.setdefault("counterexamples", counterexamples)
            payload.setdefault("accepted_mechanism_ids", [])
            payload.setdefault("rejected_mechanism_ids", [])
            payload.setdefault("risk_flags", ["schema_normalized"])
        if payload.get("status") == "consensus":
            accepted = payload.get("accepted_mechanisms")
            rejected = payload.get("rejected_mechanisms") or payload.get("rejected_mechanism_ids")
            payload["agree"] = not (isinstance(accepted, list) and not accepted and bool(rejected))
            payload.setdefault("concede", False)

    if role == "mechanism_consensus":
        if _first_list(payload, ("accepted_mechanisms", "mechanisms", "accepted_hypotheses", "mechanism_hypotheses")) is not None:
            payload["accepted_mechanisms"] = [
                _normalize_mechanism_item(item, index)
                for index, item in enumerate(
                    _first_list(payload, ("accepted_mechanisms", "mechanisms", "accepted_hypotheses", "mechanism_hypotheses")) or [],
                    start=1,
                )
            ]
        status = str(payload.get("status") or "").strip().lower()
        if status in NONCONSENSUS_STATUSES:
            payload.setdefault("accepted_mechanisms", [])
        if status == "consensus" and not (payload.get("accepted_mechanisms") or []) and (
            payload.get("rejected_mechanisms") or payload.get("rejected_mechanism_ids")
        ):
            payload["status"] = "no_consensus"
        payload.setdefault("status", "consensus")
        payload.setdefault("consensus_summary", str(payload.get("summary") or payload.get("overall_assessment") or "Mechanism consensus normalized from LLM output."))

    if role == "principle_postmortem":
        status = str(payload.get("status") or "").strip().lower()
        if status not in {"continue", "finalize", "reject"}:
            status = "continue"
        payload["status"] = status
        action = str(payload.get("principle_update_action") or payload.get("update_action") or "").strip().lower()
        allowed_actions = {"refine", "narrow", "reject", "finalize", "promote_control_mechanism", "start_new"}
        payload["principle_update_action"] = action if action in allowed_actions else (
            "finalize" if status == "finalize" else "reject" if status == "reject" else "refine"
        )
        hypothesis_status = str(payload.get("hypothesis_status") or payload.get("mechanism_status") or "").strip().lower()
        if hypothesis_status not in {"supported", "contradicted", "ambiguous", "execution_failed"}:
            hypothesis_status = "ambiguous"
        payload["hypothesis_status"] = hypothesis_status
        payload.setdefault("program_id", str(payload.get("principle_program_id") or "principle_program_unknown"))
        payload.setdefault("round", payload.get("round_number"))
        payload.setdefault("current_principle_statement", str(payload.get("principle_statement") or payload.get("claim") or "Unspecified active principle."))
        payload.setdefault("micro_mechanism", str(payload.get("mechanism") or payload.get("causal_mechanism") or "Microscopic mechanism not yet resolved."))
        payload.setdefault("e_hull_evidence", {})
        payload.setdefault("sun_accounting", {})
        payload.setdefault("causal_interpretation", str(payload.get("summary") or "Postmortem did not provide a causal interpretation."))
        for key in ("failure_boundaries", "unresolved_contradictions"):
            value = payload.get(key)
            if isinstance(value, str):
                payload[key] = [value]
            elif not isinstance(value, list):
                payload[key] = []
        payload.setdefault("next_test_focus", str(payload.get("next_step") or payload.get("next_prediction_focus") or "Refine with another matched primary/control test."))

    if role == "prediction_agent_c":
        _ensure_agent(payload, "C", "prediction_agent_c")
        if payload.get("status") == "consensus" and "agent" not in payload:
            payload["agent"] = "C"
        predictions = _first_list(
            payload,
            (
                "predictions",
                "testable_predictions",
                "prediction_hypotheses",
                "accepted_predictions",
                "hypotheses",
            ),
        )
        if predictions is not None:
            payload["predictions"] = [_normalize_prediction_item(item, index) for index, item in enumerate(predictions, start=1)]
        if not isinstance(payload.get("predictions"), list) and payload.get("status") == "consensus":
            accepted = payload.get("accepted_predictions")
            if isinstance(accepted, list):
                payload["predictions"] = [_normalize_prediction_item(item, index) for index, item in enumerate(accepted, start=1)]
        payload.setdefault("agree", False)
        payload.setdefault("concede", False)

    if role == "prediction_agent_d":
        _ensure_agent(payload, "D", "prediction_agent_d")
        if (
            payload.get("status") in AUDIT_STATUSES
            or payload.get("status") == "consensus"
            or any(key in payload for key in ("verdict", "overall_verdict", "decision", "issues", "summary"))
        ) and "agent" not in payload:
            payload["agent"] = "D"
    if role == "prediction_agent_d" and payload.get("agent") == "D":
        if "required_revisions" not in payload:
            raw = (
                payload.get("critiques")
                or payload.get("prediction_critiques")
                or payload.get("major_issues")
                or payload.get("issues")
                or payload.get("prediction_audits")
                or payload.get("audits")
                or list(payload.get("accepted_predictions") or []) + list(payload.get("rejected_predictions") or [])
                or []
            )
            verdict = str(
                payload.get("verdict")
                or payload.get("overall_verdict")
                or payload.get("decision")
                or payload.get("status")
                or ""
            ).lower()
            if verdict in {"accept", "accepted", "approved", "consensus"}:
                payload.setdefault("agree", True)
            else:
                payload.setdefault("agree", False)
            accepted = payload.get("accepted_predictions")
            rejected = payload.get("rejected_predictions") or payload.get("rejected_prediction_ids")
            if isinstance(accepted, list) and not accepted and bool(rejected):
                payload["agree"] = False
            payload.setdefault("concede", False)
            payload.setdefault("critique_summary", str(payload.get("summary") or payload.get("overall_assessment") or "Prediction critique provided."))
            payload["required_revisions"] = _revision_strings(raw) or ["Revise predictions into explicit primary-vs-control tests."]
            payload.setdefault("evidence_chain", [])
            payload.setdefault("counterexamples", [])
            payload.setdefault("accepted_prediction_ids", [])
            payload.setdefault("rejected_prediction_ids", [])
            payload.setdefault("risk_flags", ["schema_normalized"])

    if role == "prediction_consensus":
        if _first_list(payload, ("accepted_predictions", "predictions", "testable_predictions", "prediction_hypotheses")) is not None:
            payload["accepted_predictions"] = [
                _normalize_prediction_item(item, index)
                for index, item in enumerate(
                    _first_list(payload, ("accepted_predictions", "predictions", "testable_predictions", "prediction_hypotheses")) or [],
                    start=1,
                )
            ]
        status = str(payload.get("status") or "").strip().lower()
        if status in {"ok", "accepted", "approved", "accept"} and payload.get("accepted_predictions"):
            payload["status"] = "consensus"
        if str(payload.get("status") or "").strip().lower() in NONCONSENSUS_STATUSES:
            payload.setdefault("accepted_predictions", [])
        payload.setdefault("status", "consensus")
        payload.setdefault("consensus_summary", str(payload.get("summary") or payload.get("overall_assessment") or "Prediction consensus normalized from LLM output."))

    if role == "execution_agent_e":
        _ensure_agent(payload, "E", "execution_agent_e")
        if (
            payload.get("status") in {
                "ok",
                "consensus",
                "materialization_conflict",
                "hypothesis_conflict",
                PREDICTION_DESIGN_INFEASIBLE_STATUS,
                "non_executable",
                "conditional_accept_only_after_preflight",
            }
            or any(
                key in payload
                for key in (
                    "feasible",
                    "conflicting_constraints",
                    "minimal_fix_needed",
                    "intended_bundles",
                )
            )
        ) and "agent" not in payload:
            payload["agent"] = "E"
        bundles = _first_list(
            payload,
            (
                "bundles",
                "test_bundles",
                "accepted_bundles",
                "materialization_bundles",
                "execution_bundles",
            ),
        )
        if bundles is not None:
            payload["bundles"] = _normalize_execution_bundle_list(bundles)
            payload.setdefault("agent", "E")
        if payload.get("status") == "consensus" or isinstance(payload.get("bundles"), list):
            payload["agree"] = True
        payload.setdefault("agree", False)
        payload.setdefault("concede", False)
        payload = _flatten_mattergen_sampling_aliases(payload)

    if role == "execution_agent_f":
        _ensure_agent(payload, "F", "execution_agent_f")
        if (
            payload.get("status") in AUDIT_STATUSES
            or payload.get("status") == "consensus"
            or payload.get("status") == PREDICTION_DESIGN_INFEASIBLE_STATUS
            or any(key in payload for key in ("verdict", "overall_verdict", "decision", "issues", "summary"))
        ) and "agent" not in payload:
            payload["agent"] = "F"
    if role == "execution_agent_f" and payload.get("agent") == "F":
        if "required_revisions" not in payload:
            raw = (
                payload.get("critiques")
                or payload.get("bundle_critiques")
                or payload.get("major_issues")
                or payload.get("issues")
                or payload.get("bundle_audits")
                or payload.get("audits")
                or list(payload.get("accepted_bundles") or []) + list(payload.get("rejected_bundles") or [])
                or []
            )
            verdict = str(
                payload.get("verdict")
                or payload.get("overall_verdict")
                or payload.get("decision")
                or payload.get("status")
                or ""
            ).lower()
            if verdict in {"accept", "accepted", "approved", "consensus"}:
                payload.setdefault("agree", True)
            else:
                payload.setdefault("agree", False)
            accepted = payload.get("accepted_bundles")
            rejected = payload.get("rejected_bundles") or payload.get("rejected_bundle_ids")
            if isinstance(accepted, list) and not accepted and bool(rejected):
                payload["agree"] = False
            payload.setdefault("concede", False)
            payload.setdefault("critique_summary", str(payload.get("summary") or payload.get("overall_assessment") or "Execution-plan critique provided."))
            payload["required_revisions"] = _revision_strings(raw) or ["Revise bundles so every query materializes and faithfully tests the prediction."]
            payload.setdefault("evidence_chain", [])
            payload.setdefault("counterexamples", [])
            payload.setdefault("accepted_bundle_ids", [])
            payload.setdefault("rejected_bundle_ids", [])
            payload.setdefault("risk_flags", ["schema_normalized"])

    if role == "execution_consensus":
        if _first_list(payload, ("accepted_bundles", "bundles", "test_bundles", "materialization_bundles", "execution_bundles")) is not None:
            payload["accepted_bundles"] = _normalize_execution_bundle_list(
                _first_list(payload, ("accepted_bundles", "bundles", "test_bundles", "materialization_bundles", "execution_bundles")) or []
            )
        elif payload.get("status") == "consensus" and (
            isinstance(payload.get("accepted_bundle_ids"), list) or isinstance(payload.get("bundle_reviews"), list)
        ):
            payload["accepted_bundles"] = []
        status = str(payload.get("status") or "").strip().lower()
        has_accepted_bundle_refs = bool(payload.get("accepted_bundles")) or bool(payload.get("accepted_bundle_ids"))
        if status in {"ok", "accepted", "approved", "accept"} and has_accepted_bundle_refs:
            payload["status"] = "consensus"
        if payload.get("status") == "consensus" and not isinstance(payload.get("accepted_bundles"), list) and (
            isinstance(payload.get("accepted_bundle_ids"), list) or isinstance(payload.get("bundle_reviews"), list)
        ):
            payload["accepted_bundles"] = []
        payload.setdefault("status", "consensus")
        payload.setdefault("consensus_summary", str(payload.get("summary") or payload.get("overall_assessment") or "Execution consensus normalized from LLM output."))
        payload = _flatten_mattergen_sampling_aliases(payload)

    if role in {
        "mechanism_agent_a",
        "mechanism_agent_b",
        "prediction_agent_c",
        "prediction_agent_d",
        "execution_agent_e",
        "execution_agent_f",
    }:
        payload["agree"] = _dialogue_bool(payload.get("agree", False))
        payload["concede"] = _dialogue_bool(payload.get("concede", False))

    return payload


def _ensure_agent(payload: dict[str, Any], letter: str, role_name: str) -> None:
    agent = payload.get("agent")
    if agent == letter:
        return
    normalized_agent = str(agent or "").strip().lower().replace("-", "_").replace(" ", "_")
    if payload.get("role") == role_name or normalized_agent in {role_name, f"agent_{letter.lower()}"}:
        payload["agent"] = letter


def _dialogue_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "agree", "agreed", "accept", "accepted", "concede", "conceded"}
    return False


def _first_list(payload: Mapping[str, Any], keys: Sequence[str]) -> list[Any] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return None


def _normalize_mechanism_item(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        text = str(item)
        return {
            "id": f"m{index:03d}",
            "claim": text,
            "rationale_summary": text,
            "evidence_chain": [text],
            "scope": "Unspecified by the LLM output.",
            "confidence": "medium",
        }
    normalized = dict(item)
    name = str(normalized.get("name") or normalized.get("id") or f"m{index:03d}").strip()
    if not str(normalized.get("id") or "").strip():
        normalized["id"] = name if name.startswith("m") else f"m{index:03d}"
    claim = str(normalized.get("claim") or normalized.get("hypothesis") or normalized.get("mechanism") or "").strip()
    if claim:
        normalized["claim"] = claim
    if not str(normalized.get("rationale_summary") or "").strip() and claim:
        normalized["rationale_summary"] = claim
    rationale = str(
        normalized.get("rationale_summary")
        or normalized.get("rationale")
        or normalized.get("physical_basis")
        or normalized.get("reasoning_summary")
        or normalized.get("expected_trend")
        or ""
    ).strip()
    if rationale:
        normalized["rationale_summary"] = rationale
    evidence = normalized.get("evidence_chain")
    if not isinstance(evidence, list) or not evidence:
        evidence_items = _first_list(normalized, ("observable_features", "expected_material_signs", "testable_implications"))
        if evidence_items:
            normalized["evidence_chain"] = [str(value) for value in evidence_items if str(value).strip()]
        elif rationale:
            normalized["evidence_chain"] = [rationale]
        elif claim:
            normalized["evidence_chain"] = [claim]
    if not str(normalized.get("scope") or "").strip():
        normalized["scope"] = str(normalized.get("applicability") or normalized.get("domain") or "General candidate-pool materials where the observable features apply.")
    confidence = str(normalized.get("confidence") or "medium").strip().lower()
    normalized["confidence"] = confidence if confidence in {"low", "medium-low", "medium", "medium-high", "high"} else "medium"
    return normalized


def mechanism_consensus_from_accepted_proposal(
    proposal: Mapping[str, Any],
    *,
    round_number: int | None = None,
    dialogue: Sequence[Mapping[str, Any]] | None = None,
    consensus_summary: str | None = None,
) -> dict[str, Any]:
    mechanisms = proposal.get("mechanisms")
    accepted = [
        _normalize_mechanism_item(item, index)
        for index, item in enumerate(mechanisms if isinstance(mechanisms, list) else [], start=1)
    ]
    result: dict[str, Any] = {
        "status": "consensus" if accepted else "unresolved",
        "accepted_mechanisms": accepted,
        "rejected_mechanisms": [],
        "consensus_summary": consensus_summary
        or "Mechanism consensus reconstructed directly from an accepted proposal after A/B agreement.",
    }
    if round_number is not None:
        result["round"] = round_number
    if dialogue is not None:
        result["dialogue"] = list(dialogue)
    return result


def _normalize_prediction_item(item: Any, index: int) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        claim = str(item).strip()
        return {
            "id": f"p{index:03d}",
            "mechanism_ids": ["m001"],
            "claim": claim or f"Prediction {index}: primary branch should outperform matched controls.",
            "predicted_relation": "primary_lower_e_hull_than_control",
            "comparison_design": {
                "primary": "materials satisfying the accepted mechanism",
                "control": "matched controls that weaken the mechanism",
            },
            "falsification_criteria": [
                "The evaluated primary/control e_hull ordering does not match the predicted relation."
            ],
        }
    normalized = dict(item)
    if not str(normalized.get("id") or "").strip():
        normalized["id"] = f"p{index:03d}"
    mechanism_ids = _first_list(normalized, ("mechanism_ids", "mechanisms"))
    if mechanism_ids:
        normalized["mechanism_ids"] = [str(value) for value in mechanism_ids if str(value).strip()]
    elif "mechanism_ids" not in normalized:
        normalized["mechanism_ids"] = ["m001"]
    claim = str(normalized.get("claim") or normalized.get("prediction") or normalized.get("hypothesis") or "").strip()
    if claim:
        normalized["claim"] = claim
    else:
        normalized["claim"] = f"Prediction {index}: primary branch should differ from matched controls."
    relation = str(normalized.get("predicted_relation") or normalized.get("expected_relation") or "").strip()
    if relation not in {
        "primary_lower_e_hull_than_control",
        "primary_higher_e_hull_than_control",
        "primary_lower_form_e_than_control",
        "primary_higher_form_e_than_control",
    }:
        lower_text = " ".join(
            str(normalized.get(key) or "") for key in ("claim", "prediction", "expected_trend")
        ).lower()
        relation = "primary_higher_e_hull_than_control" if ("higher" in lower_text and "e_hull" in lower_text) else "primary_lower_e_hull_than_control"
    normalized["predicted_relation"] = relation
    if not isinstance(normalized.get("comparison_design"), Mapping):
        normalized["comparison_design"] = {
            "primary": str(normalized.get("primary") or "materials satisfying the accepted mechanism"),
            "control": str(normalized.get("control") or "matched controls that weaken the mechanism"),
        }
    criteria = _first_list(normalized, ("falsification_criteria", "failure_criteria"))
    if criteria:
        normalized["falsification_criteria"] = [str(value) for value in criteria if str(value).strip()]
    elif "falsification_criteria" not in normalized:
        normalized["falsification_criteria"] = [
            "The evaluated primary/control e_hull ordering does not match the predicted relation."
        ]
    return normalized


def _revision_strings(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    revisions: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            issue = str(item.get("issue") or item.get("type") or item.get("id") or item.get("verdict") or "critique")
            repairs = item.get("required_repairs")
            if isinstance(repairs, list):
                repair_text = "; ".join(str(value) for value in repairs if str(value).strip())
            else:
                repair_text = str(repairs or "")
            revision = str(
                item.get("needed_refinement")
                or item.get("required_revision")
                or item.get("proposed_revision")
                or repair_text
                or item.get("critique")
                or item.get("reason")
                or item.get("rejection_reason")
                or item.get("audit_summary")
                or item.get("detail")
                or item.get("description")
                or ""
            )
            revisions.append(f"{issue}: {revision}" if revision else issue)
        else:
            revisions.append(str(item))
    return [item for item in revisions if item.strip()]


ROLE_SCHEMA_ALIASES = {
    "mechanism_agent_b_counterproposal": "mechanism_agent_a",
    "mechanism_agent_a_reverse_critique": "mechanism_agent_b",
    "prediction_agent_d_counterproposal": "prediction_agent_c",
    "prediction_agent_c_reverse_critique": "prediction_agent_d",
    "execution_agent_f_counterproposal": "execution_agent_e",
    "execution_agent_e_reverse_critique": "execution_agent_f",
}


def _schema_role_for_output(role: str) -> str:
    """Map logical debate sub-roles to the parser-compatible output shape."""

    return ROLE_SCHEMA_ALIASES.get(role, role)


def compact_invalid_output_for_repair_prompt(role: str, schema_role: str, invalid_output: str) -> str:
    raw = str(invalid_output or "")
    if schema_role == "execution_consensus":
        return (
            f"Previous invalid execution_consensus output omitted ({len(raw)} chars). "
            "It likely copied executable branch payloads. Reconstruct a compact final consensus from ORIGINAL_TASK using accepted_bundle_ids only."
        )
    try:
        parsed = extract_json_object(raw)
    except Exception:
        preview = _short_text(raw, 1800)
    else:
        if isinstance(parsed, Mapping):
            if schema_role in {"mechanism_agent_a", "mechanism_agent_b", "mechanism_consensus"} or schema_role.startswith(
                "mechanism_"
            ):
                compact: Any = _compact_mechanism_payload_for_prompt(parsed, max_mechanisms=1)
            elif schema_role in {"prediction_agent_c", "prediction_agent_d", "prediction_consensus"} or schema_role.startswith(
                "prediction_"
            ):
                compact = _compact_prediction_payload_for_prompt(parsed, max_list_items=2)
            elif schema_role in {"execution_agent_e", "execution_agent_f"} or schema_role.startswith("execution_"):
                compact = compact_execution_payload_for_final_prompt(parsed, max_list_items=2)
            elif schema_role == "principle_postmortem":
                compact = {
                    key: value
                    for key, value in {
                        "status": parsed.get("status"),
                        "program_id": parsed.get("program_id"),
                        "round": parsed.get("round"),
                        "hypothesis_status": parsed.get("hypothesis_status"),
                        "principle_update_action": parsed.get("principle_update_action"),
                        "current_principle_statement": _short_text(parsed.get("current_principle_statement"), 420),
                        "micro_mechanism": _short_text(parsed.get("micro_mechanism"), 360),
                        "failure_boundaries": compact_repair_feedback(parsed.get("failure_boundaries"), max_list_items=3),
                    }.items()
                    if value not in (None, "", [], {})
                }
            else:
                compact = compact_dialogue_payload_for_prompt(parsed, max_list_items=2)
            preview = _short_text(prompt_json_dumps(compact), 1800)
        else:
            preview = _short_text(prompt_json_dumps(parsed), 1800)
    if len(raw) > 1800:
        preview += f"\n[invalid_output_omitted_chars={len(raw) - 1800}]"
    return preview


def retry_prompt(role: str, original_user: str, invalid_output: str, error: str) -> str:
    schema_role = _schema_role_for_output(role)
    schema_hint = ""
    if schema_role == "prediction_agent_c":
        schema_hint = (
            "\nEXPECTED_SHAPE:\n"
            "Return exactly one JSON object with keys: status, agent, predictions, agree, concede.\n"
            "The `predictions` field must be a list of prediction objects. Do not output formula_probes, bundles, or any generator materialization fields.\n"
            "Each prediction must include a deterministic matched-pair design with exact counts. MatterGen-native predictions may use non-identical primary_mattergen and control_mattergen fields; legacy predictions use non-identical primary_query and control_query plus a finite selection_order.\n"
            "Use only these selection_order fields: formation_energy_per_atom, band_gap, density, volume_per_atom, nelements, nsites, spacegroup_number, formula.\n"
            "For mixed-anion mechanisms, the primary/control split should prefer a direct mixed-anion predicate rather than a low- vs high-symmetry proxy.\n"
        )
    elif schema_role == "prediction_consensus":
        schema_hint = (
            "\nEXPECTED_SHAPE:\n"
            "Return exactly one JSON object with keys: status, accepted_predictions, consensus_summary.\n"
            "Use status=\"consensus\" when accepted_predictions is non-empty. Never use status=\"ok\" for final prediction consensus.\n"
            "If Agent C and Agent D do not jointly accept any prediction, return status=\"rejected\" and accepted_predictions=[] instead of forcing a false consensus.\n"
        )
    elif schema_role == "execution_consensus":
        schema_hint = (
            "\nEXPECTED_SHAPE:\n"
            "Return exactly one JSON object for the final execution consensus.\n"
            "Preferred compact consensus shape: {\"status\":\"consensus\", \"accepted_bundle_ids\":[\"b001\"], \"rejected_bundles\":[], \"consensus_summary\":\"...\"}.\n"
            "Do not output structure_dicts, formula_probes, mattergen_requests, material_ids, selected rows, or full branch payloads in this final consensus. The controller restores executable bundle details from the accepted proposal by bundle id.\n"
            "If no bundle is accepted, return status=\"no_materialized_consensus\" or \"prediction_design_infeasible\" with concise feedback instead of malformed partial JSON.\n"
        )
    elif schema_role == "principle_postmortem":
        schema_hint = (
            "\nEXPECTED_SHAPE:\n"
            "Return exactly one JSON object with keys: status, program_id, round, hypothesis_status, principle_update_action, current_principle_statement, micro_mechanism, e_hull_evidence, sun_accounting, causal_interpretation, failure_boundaries, unresolved_contradictions, next_test_focus.\n"
            "status must be continue, finalize, or reject. Do not count control-branch SUN as original-mechanism success.\n"
        )
    invalid_output_for_prompt = compact_invalid_output_for_repair_prompt(role, schema_role, invalid_output)
    return f"""Your previous output for role `{role}` was invalid. Return exactly one valid JSON object and preserve the intended content when possible.
Do not request tools in this JSON repair response. Do not output status="tool_request". Do not concatenate a tool_request JSON object with a final role JSON object; return only the corrected role JSON object.

DIRECTIVE:
{MATERIAL_PHYSICS_DIRECTIVE}
{schema_hint}

ERROR:
```text
{error}
```

ORIGINAL_TASK:
```text
{original_user}
```

INVALID_PREVIOUS_OUTPUT:
```text
{invalid_output_for_prompt}
```
"""


def transient_llm_retry_prompt(role: str, original_user: str, error: str) -> str:
    return f"""The previous LLM call for role `{role}` failed before usable text was extracted.
Retry the original task now and return exactly one valid JSON object.

DIRECTIVE:
{MATERIAL_PHYSICS_DIRECTIVE}

TRANSIENT_LLM_ERROR:
```text
{error}
```

ORIGINAL_TASK:
```text
{original_user}
```
"""


def valid_shape(parsed: Any, role: str) -> bool:
    role = _schema_role_for_output(role)
    if not isinstance(parsed, Mapping):
        return False
    if role == "mechanism_agent_a":
        return parsed.get("agent") == "A" and isinstance(parsed.get("mechanisms"), list)
    if role == "mechanism_agent_b":
        return parsed.get("agent") == "B" and isinstance(parsed.get("required_revisions"), list)
    if role == "mechanism_consensus":
        status = str(parsed.get("status") or "").strip().lower()
        return isinstance(parsed.get("accepted_mechanisms"), list) and (
            status == "consensus" or status in NONCONSENSUS_STATUSES
        )
    if role == "principle_postmortem":
        return (
            str(parsed.get("status") or "").strip().lower() in {"continue", "finalize", "reject"}
            and str(parsed.get("hypothesis_status") or "").strip().lower()
            in {"supported", "contradicted", "ambiguous", "execution_failed"}
            and str(parsed.get("current_principle_statement") or "").strip() != ""
        )
    if role == "prediction_agent_c":
        return parsed.get("agent") == "C" and isinstance(parsed.get("predictions"), list)
    if role == "prediction_agent_d":
        return parsed.get("agent") == "D" and isinstance(parsed.get("required_revisions"), list)
    if role == "prediction_consensus":
        status = str(parsed.get("status") or "").strip().lower()
        return isinstance(parsed.get("accepted_predictions"), list) and (
            status == "consensus" or status in NONCONSENSUS_STATUSES
        )
    if role == "execution_agent_e":
        return parsed.get("agent") == "E" and (
            parsed.get("status")
            in {
                "materialization_conflict",
                "hypothesis_conflict",
                PREDICTION_DESIGN_INFEASIBLE_STATUS,
                "non_executable",
                "conditional_accept_only_after_preflight",
            }
            or isinstance(parsed.get("bundles"), list)
        )
    if role == "execution_agent_f":
        return parsed.get("agent") == "F" and isinstance(parsed.get("required_revisions"), list)
    if role == "execution_consensus":
        return parsed.get("status") in {
            "consensus",
            "materialization_conflict",
            "no_materialized_consensus",
            "no_materialized_consensus_pending_preflight",
            "no_materialized_consensus_pending_generator",
            PREDICTION_DESIGN_INFEASIBLE_STATUS,
            "non_executable",
            "conditional_accept_only_after_preflight",
        } and (
            parsed.get("status") != "consensus"
            or isinstance(parsed.get("accepted_bundles"), list)
        )
    return True


def agreement_reached(proposal: Mapping[str, Any], critique: Mapping[str, Any]) -> bool:
    proposal_status = str(proposal.get("status") or "").strip().lower()
    critique_status = str(critique.get("status") or critique.get("overall_verdict") or critique.get("decision") or "").strip().lower()
    blocking_statuses = NONCONSENSUS_STATUSES | {
        "partial_rejection",
        "needs_repair",
        "reject_partial",
        "accepted_with_revisions",
    }
    if proposal_status in NONCONSENSUS_STATUSES or critique_status in blocking_statuses:
        return False
    return _dialogue_bool(critique.get("agree")) or _dialogue_bool(critique.get("concede"))


def critique_requires_counterproposal(
    critique: Mapping[str, Any],
    *,
    rejection_streak: int,
    threshold: int = DEFAULT_CRITIC_COUNTERPROPOSAL_AFTER,
) -> bool:
    """Return true when the critic must stop only objecting and propose instead.

    ``rejection_streak`` is the number of consecutive rejected critiques before
    the current critique. The current critique is counted by this helper.
    """

    if threshold <= 0:
        return False
    if _dialogue_bool(critique.get("agree")) or _dialogue_bool(critique.get("concede")):
        return False
    status = str(critique.get("status") or critique.get("overall_verdict") or critique.get("decision") or "").lower()
    accepted = _first_list(
        critique,
        (
            "accepted_mechanisms",
            "accepted_predictions",
            "accepted_bundles",
            "accepted_mechanism_ids",
            "accepted_prediction_ids",
            "accepted_bundle_ids",
        ),
    )
    rejected = _first_list(
        critique,
        (
            "rejected_mechanisms",
            "rejected_predictions",
            "rejected_bundles",
            "rejected_mechanism_ids",
            "rejected_prediction_ids",
            "rejected_bundle_ids",
        ),
    )
    empty_consensus_rejection = status == "consensus" and isinstance(accepted, list) and not accepted and bool(rejected)
    if status in {"accept", "accepted", "approved", "consensus"}:
        return empty_consensus_rejection and rejection_streak + 1 >= threshold
    return rejection_streak + 1 >= threshold


def prompt_mechanism_proposal(context_json: str) -> str:
    return f"""Agent A: propose one or more general mechanism hypotheses about why some materials are more stable than others.

{MECHANISM_RAG_REQUIREMENT}

{PRINCIPLE_DISCOVERY_POLICY}

{MECHANISM_EXPLORATION_REQUIREMENT}

{MECHANISM_AGENT_A_COMPACT_RULES}

CONTEXT_JSON:
```json
{context_json}
```

Return only Agent A JSON.
"""


def prompt_mechanism_counterproposal(context_json: str, proposal: Mapping[str, Any], critique: Mapping[str, Any], cycle: int) -> str:
    return f"""Agent B: you rejected Agent A's mechanism proposal in cycle {cycle}. You must now propose your own mechanism set that satisfies your own critique.

Do not only object. Either produce Agent A-compatible mechanism hypotheses, or produce a concise impossibility_certificate explaining why no defensible mechanism can be stated from the available observables.
For parser compatibility, set the JSON field agent to "A" and use the same output shape as Agent A: status, agent, mechanisms, agree, concede.

{MECHANISM_RAG_REQUIREMENT}

{PRINCIPLE_DISCOVERY_POLICY}

{MECHANISM_EXPLORATION_REQUIREMENT}

{MECHANISM_AGENT_A_COMPACT_RULES}

CONTEXT_JSON:
```json
{context_json}
```

PREVIOUS_AGENT_A_JSON:
```json
{_prompt_mechanism_json(proposal)}
```

AGENT_B_JSON:
```json
{_prompt_mechanism_json(critique, max_mechanisms=1)}
```

Return only Agent A JSON.
"""


def prompt_mechanism_reverse_critique(context_json: str, original_proposal: Mapping[str, Any], counterproposal: Mapping[str, Any], cycle: int) -> str:
    return f"""Agent A: critique Agent B's mechanism counterproposal for cycle {cycle}.

You are now the reviewer. If Agent B's counterproposal is stronger than yours, say agree=true. If not, explain exact required revisions.
For parser compatibility, set the JSON field agent to "B" and use the same output shape as Agent B: status, agent, required_revisions, agree, concede.

{MECHANISM_RAG_REQUIREMENT}

{PRINCIPLE_DISCOVERY_POLICY}

{MECHANISM_EXPLORATION_REQUIREMENT}

CONTEXT_JSON:
```json
{context_json}
```

ORIGINAL_AGENT_A_JSON:
```json
{_prompt_mechanism_json(original_proposal)}
```

AGENT_B_COUNTERPROPOSAL_JSON:
```json
{_prompt_mechanism_json(counterproposal)}
```

Return only Agent B JSON.
"""


def prompt_mechanism_reverse_final(context_json: str, counterproposal: Mapping[str, Any], reverse_critique: Mapping[str, Any]) -> str:
    return f"""Agent A: write the final mechanism consensus after auditing Agent B's counterproposal.

Only include mechanisms that both agents accept.

{MECHANISM_RAG_REQUIREMENT}

{PRINCIPLE_DISCOVERY_POLICY}

{MECHANISM_EXPLORATION_REQUIREMENT}

{MECHANISM_CONSENSUS_COMPACT_RULES}

CONTEXT_JSON:
```json
{context_json}
```

FINAL_AGENT_B_COUNTERPROPOSAL_JSON:
```json
{_prompt_mechanism_json(counterproposal)}
```

FINAL_AGENT_A_REVIEW_JSON:
```json
{_prompt_mechanism_json(reverse_critique, max_mechanisms=1)}
```

Return only final consensus JSON.
"""


def prompt_mechanism_critique(context_json: str, proposal: Mapping[str, Any], cycle: int) -> str:
    return f"""Agent B: critique Agent A's mechanism hypotheses for cycle {cycle}.

{MECHANISM_RAG_REQUIREMENT}

{PRINCIPLE_DISCOVERY_POLICY}

{MECHANISM_EXPLORATION_REQUIREMENT}

CONTEXT_JSON:
```json
{context_json}
```

AGENT_A_JSON:
```json
{_prompt_mechanism_json(proposal)}
```

Return only Agent B JSON.
"""


def prompt_mechanism_revision(context_json: str, proposal: Mapping[str, Any], critique: Mapping[str, Any], cycle: int) -> str:
    return f"""Agent A: revise after Agent B critique in cycle {cycle}.

Keep only mechanism claims that are actually supported by materials knowledge and the historical evidence. Narrow or remove weak claims.

{MECHANISM_RAG_REQUIREMENT}

{PRINCIPLE_DISCOVERY_POLICY}

{MECHANISM_EXPLORATION_REQUIREMENT}

{MECHANISM_AGENT_A_COMPACT_RULES}

CONTEXT_JSON:
```json
{context_json}
```

PREVIOUS_AGENT_A_JSON:
```json
{_prompt_mechanism_json(proposal)}
```

AGENT_B_JSON:
```json
{_prompt_mechanism_json(critique, max_mechanisms=1)}
```

Return only Agent A JSON.
"""


def prompt_mechanism_final(context_json: str, proposal: Mapping[str, Any], critique: Mapping[str, Any]) -> str:
    return f"""Agent B: write the final mechanism consensus.

Only include mechanisms that both agents accept. The final output will be the basis for later predictions.

{MECHANISM_RAG_REQUIREMENT}

{PRINCIPLE_DISCOVERY_POLICY}

{MECHANISM_EXPLORATION_REQUIREMENT}

{MECHANISM_CONSENSUS_COMPACT_RULES}

CONTEXT_JSON:
```json
{context_json}
```

FINAL_AGENT_A_JSON:
```json
{_prompt_mechanism_json(proposal)}
```

FINAL_AGENT_B_JSON:
```json
{_prompt_mechanism_json(critique, max_mechanisms=1)}
```

Return only final consensus JSON.
"""


def prompt_prediction_proposal(context_json: str) -> str:
    context_prompt = compact_prediction_context_json_for_prompt(context_json)
    return f"""Agent C: turn the accepted mechanism hypotheses into falsifiable predictions.

{PREDICTION_AGENT_C_COMPACT_RULES}

{CD_HYPOTHESIS_VALIDATION_POLICY}

{_cd_prediction_exploration_requirement(context_prompt)}

{mattergen_prediction_prompt_block(context_prompt)}

Rules:
- Output predictions only; no formula_probes, bundles, material IDs, or materialization blueprint.
- Each prediction is a deterministic matched-pair design. MatterGen mode uses non-identical primary_mattergen/control_mattergen; other modes use non-identical primary_query and control_query plus finite selection_order.
- planned_material_count must be at least current_inputs.target_count and should equal it when faithful.
- Keep one faithful execution path for each branch; if E/F repair feedback marked the design infeasible, directly change that primary/control design or return no consensus.
- Use only executable sort fields: formation_energy_per_atom, band_gap, density, volume_per_atom, nelements, nsites, spacegroup_number, formula.
- Prefer a direct mixed-anion versus single-anion split unless the mechanism is explicitly about symmetry.
- Only cite current_inputs.accepted_mechanism_ids; when revising/reviewing, stay close to the immediately prior proposal.

CONTEXT_JSON:
```json
{context_prompt}
```

Return only Agent C JSON.
"""


def prompt_prediction_counterproposal(context_json: str, proposal: Mapping[str, Any], critique: Mapping[str, Any], cycle: int) -> str:
    context_prompt = compact_prediction_context_json_for_prompt(context_json)
    return f"""Agent D: you rejected Agent C's prediction design in cycle {cycle}. You must now propose your own prediction design that satisfies your own critique.

{PREDICTION_AGENT_C_COMPACT_RULES}

{CD_HYPOTHESIS_VALIDATION_POLICY}

{_cd_prediction_exploration_requirement(context_prompt)}

{mattergen_prediction_prompt_block(context_prompt)}

Do not only object. Either produce Agent C-compatible predictions, or produce a concise impossibility_certificate explaining why no prediction can faithfully test the accepted mechanisms with the current observables and target_count.
Stay in the prediction stage only. Do not output formula_probes, bundles, structure_dicts, material IDs, or any materialization blueprint.
If CONTEXT_JSON.repair_feedback contains E/F execution feedback, your counterproposal must address it directly and avoid any primary/control design that E/F jointly marked prediction_design_infeasible.
Stay close to Agent C's immediately prior proposal: keep the same accepted mechanism, target chemistry, and falsification intent unless your critique identified a concrete blocker. Do not replace C's proposal with an unrelated easier pool comparison or new materials principle; if a branch must change, choose the nearest faithful analogue and summarize the specific evidence that forced the change.
Each prediction must include predictions as a list, exact counts, falsification_criteria, and a comparison_design. In MatterGen-native mode use comparison_design.primary_mattergen/control_mattergen; otherwise use comparison_design.primary_query/control_query with deterministic selection_order.
Materialization budget requirement: compute planned_material_count = sum(comparison_design.primary_count + comparison_design.control_count) over the accepted prediction set. Your counterproposal must be able to materialize at least current_inputs.target_count materials; prefer exactly current_inputs.target_count when scientifically faithful. Do not counterpropose an undersized prediction set unless you return an impossibility_certificate explaining why no faithful >= target_count prediction design exists.
Every accepted prediction must have at least one faithful execution path through MP-pool selection, generator materialization, or MatterGen conditional generation for both branches.
For parser compatibility, set the JSON field agent to "C" and use the same output shape as Agent C: status, agent, predictions, agree, concede.

CONTEXT_JSON:
```json
{context_prompt}
```

PREVIOUS_AGENT_C_JSON:
```json
{_prompt_prediction_peer_json(proposal)}
```

AGENT_D_JSON:
```json
{_prompt_prediction_peer_json(critique)}
```

Return only Agent C JSON.
"""


def prompt_prediction_reverse_critique(context_json: str, original_proposal: Mapping[str, Any], counterproposal: Mapping[str, Any], cycle: int) -> str:
    context_prompt = compact_prediction_context_json_for_prompt(context_json)
    return f"""Agent C: critique Agent D's prediction counterproposal for cycle {cycle}.

{PREDICTION_AGENT_D_COMPACT_RULES}

{CD_HYPOTHESIS_VALIDATION_POLICY}

{_cd_prediction_exploration_requirement(context_prompt)}

{mattergen_prediction_prompt_block(context_prompt)}

You are now the reviewer. If Agent D's counterproposal is more faithful, executable, and less confounded than your proposal, say agree=true. If not, provide exact required revisions.
Reject any counterproposal that introduces a new materials principle or unrelated stable basin instead of testing the accepted A/B mechanism.
Reject any counterproposal whose accepted prediction set has planned_material_count < current_inputs.target_count, where planned_material_count = sum(comparison_design.primary_count + comparison_design.control_count). If it is under budget, require D to add independent predictions or increase exact counts without changing the falsification design.
If CONTEXT_JSON.repair_feedback says a previous design was prediction_design_infeasible, reject any counterproposal that repeats the same infeasible branch chemistry or control definition without a faithful materialization path.
For parser compatibility, set the JSON field agent to "D" and use the same output shape as Agent D: status, agent, required_revisions, agree, concede.

CONTEXT_JSON:
```json
{context_prompt}
```

ORIGINAL_AGENT_C_JSON:
```json
{_prompt_prediction_peer_json(original_proposal)}
```

AGENT_D_COUNTERPROPOSAL_JSON:
```json
{_prompt_prediction_peer_json(counterproposal)}
```

Return only Agent D JSON.
"""


def prompt_prediction_reverse_final(context_json: str, counterproposal: Mapping[str, Any], reverse_critique: Mapping[str, Any]) -> str:
    context_prompt = compact_prediction_context_json_for_prompt(context_json)
    return f"""Agent C: write the final prediction consensus after auditing Agent D's counterproposal.

{PREDICTION_AGENT_C_COMPACT_RULES}

{CD_HYPOTHESIS_VALIDATION_POLICY}

{_cd_prediction_exploration_requirement(context_prompt)}

{mattergen_prediction_prompt_block(context_prompt)}

Only include predictions that both agents accept. These predictions will be turned into concrete materialization tests.
The final consensus must remain anchored to accepted A/B mechanism_ids; do not add a prediction that merely searches a historically promising family without testing those mechanisms.
The final accepted_predictions must have planned_material_count >= current_inputs.target_count, computed as sum(comparison_design.primary_count + comparison_design.control_count). Prefer exactly current_inputs.target_count when scientifically faithful; otherwise include enough accepted predictions/counts to exceed it without changing the stated falsification designs.
Do not finalize any prediction design that repeats a CONTEXT_JSON.repair_feedback prediction_design_infeasible branch/control issue without a faithful materialization path.

CONTEXT_JSON:
```json
{context_prompt}
```

FINAL_AGENT_D_COUNTERPROPOSAL_JSON:
```json
{_prompt_prediction_peer_json(counterproposal)}
```

FINAL_AGENT_C_REVIEW_JSON:
```json
{_prompt_prediction_peer_json(reverse_critique)}
```

Return only final consensus JSON.
"""


def prompt_prediction_critique(context_json: str, proposal: Mapping[str, Any], cycle: int) -> str:
    context_prompt = compact_prediction_context_json_for_prompt(context_json)
    return f"""Agent D: critique Agent C's prediction design for cycle {cycle}.

{PREDICTION_AGENT_D_COMPACT_RULES}

{CD_HYPOTHESIS_VALIDATION_POLICY}

{_cd_prediction_exploration_requirement(context_prompt)}

{mattergen_prediction_prompt_block(context_prompt)}

Check the materialization budget before accepting. Compute planned_material_count = sum(comparison_design.primary_count + comparison_design.control_count) over Agent C's accepted/proposed prediction set. If planned_material_count < current_inputs.target_count, reject and require more independent predictions or larger exact counts that preserve the falsification design. Prefer exactly current_inputs.target_count when scientifically faithful.
Also check execution feasibility. If CONTEXT_JSON.repair_feedback contains E/F prediction_design_infeasible feedback, reject any proposal that repeats the same infeasible primary/control design. Require a prediction whose branches can be faithfully materialized by MP-pool selection, generator, or MatterGen without changing the scientific test.
Reject if Agent C drifted from validation into independent exploration: new mechanism, unrelated target chemistry, low-energy-pool chasing, or a primary/control contrast that no longer falsifies an accepted mechanism_id.

CONTEXT_JSON:
```json
{context_prompt}
```

AGENT_C_JSON:
```json
{_prompt_prediction_peer_json(proposal)}
```

Return only Agent D JSON.
"""


def prompt_prediction_revision(context_json: str, proposal: Mapping[str, Any], critique: Mapping[str, Any], cycle: int) -> str:
    context_prompt = compact_prediction_context_json_for_prompt(context_json)
    return f"""Agent C: revise after Agent D critique in cycle {cycle}.

{PREDICTION_AGENT_C_COMPACT_RULES}

{CD_HYPOTHESIS_VALIDATION_POLICY}

{_cd_prediction_exploration_requirement(context_prompt)}

{mattergen_prediction_prompt_block(context_prompt)}

Stay in the prediction stage only. Revise the prediction claims, comparison design, scope, or falsification criteria.
Do not turn the answer into generator formula_probes, bundle blueprints, or any other materialization output.
Keep the revision anchored to the accepted A/B mechanism. If D's critique exposes an untestable or confounded mechanism, choose the nearest faithful validation analogue or return no consensus rather than pivoting to a different materials principle.
Keep the design deterministic: exact counts plus either a non-identical primary/control MatterGen chemical-system pair or a non-identical primary/control query pair with an explicit selection order.
If CONTEXT_JSON.repair_feedback contains E/F prediction_design_infeasible feedback, directly fix that issue. Do not repeat the same infeasible branch chemistry or control family unless you narrow the scope and show a faithful materialization path.
Materialization budget requirement: compute planned_material_count = sum(comparison_design.primary_count + comparison_design.control_count) over the prediction set. The revised prediction set must be able to materialize at least current_inputs.target_count materials; prefer exactly current_inputs.target_count when scientifically faithful. If the current accepted set is under budget, add independent predictions or increase exact primary/control counts without changing the falsification design.
Selection orders must use only executable sort fields: formation_energy_per_atom, band_gap, density, volume_per_atom, nelements, nsites, spacegroup_number, formula. Use volume_per_atom, not volume. Do not use material_id or spacegroup_symbol in prediction-stage selection_order.
For mixed-anion mechanisms, if the current design uses symmetry as its main split, replace it with a direct mixed-anion observable predicate. Only fall back to a symmetry proxy if you explicitly narrow the claim to a symmetry-proxy hypothesis.
Only use mechanism_ids that appear in current_inputs.accepted_mechanism_ids; do not cite stale or rejected mechanism ids.

CONTEXT_JSON:
```json
{context_prompt}
```

PREVIOUS_AGENT_C_JSON:
```json
{_prompt_prediction_peer_json(proposal)}
```

AGENT_D_JSON:
```json
{_prompt_prediction_peer_json(critique)}
```

Return only Agent C JSON.
"""


def prompt_prediction_final(context_json: str, proposal: Mapping[str, Any], critique: Mapping[str, Any]) -> str:
    context_prompt = compact_prediction_context_json_for_prompt(context_json)
    return f"""Agent D: write the final prediction consensus.

{PREDICTION_AGENT_C_COMPACT_RULES}

{CD_HYPOTHESIS_VALIDATION_POLICY}

{_cd_prediction_exploration_requirement(context_prompt)}

{mattergen_prediction_prompt_block(context_prompt)}

Only include predictions that both agents accept. These predictions will be turned into concrete materialization tests.
Return the final prediction_consensus shape: top-level status, accepted_predictions, consensus_summary, and optional rejected_predictions/materialization_constraints. Use status="consensus" when accepted_predictions is non-empty; use status="rejected" or status="no_consensus" with accepted_predictions=[] when no prediction is jointly accepted. Never use status="ok" in final prediction consensus.
Final prediction consensus is a validation contract for A/B mechanisms, not a new discovery proposal. Exclude any branch that does not directly test or falsify an accepted mechanism_id.
The final accepted_predictions must have planned_material_count >= current_inputs.target_count, computed as sum(comparison_design.primary_count + comparison_design.control_count). Do not finalize an undersized prediction set; require revision or return no consensus if no faithful >= target_count design exists. Prefer exactly current_inputs.target_count when scientifically faithful.
Do not finalize any prediction design that repeats a CONTEXT_JSON.repair_feedback prediction_design_infeasible branch/control issue without a faithful materialization path.

CONTEXT_JSON:
```json
{context_prompt}
```

FINAL_AGENT_C_JSON:
```json
{_prompt_prediction_peer_json(proposal)}
```

FINAL_AGENT_D_JSON:
```json
{_prompt_prediction_peer_json(critique)}
```

Return only final consensus JSON.
"""


def prompt_execution_proposal(context_json: str) -> str:
    context_prompt = compact_execution_context_json_for_prompt(context_json)
    return f"""Agent E: translate the accepted predictions into concrete materialization blueprints.

{_cdef_exploration_requirement(context_prompt)}

{EF_EVIDENCE_QUALITY_POLICY}

{EF_COMPACT_PAYLOAD_POLICY}

{mattergen_native_prompt_block(context_prompt)}

Each bundle should contain a primary group and a matched control group, and each bundle must correspond to exactly one accepted prediction_id. Do not combine multiple predictions into one bundle. The total selected count across all bundles must equal the requested target count.
For each primary/control branch:
- In MatterGen-native mode, use source "mattergen" with exactly one mattergen_requests object per branch; do not hand-build cells or formula_probes for MatterGen branches.
- In MatterGen-native mode, every executable branch must use source "mattergen"; do not use source "generator", source "mp_pool", formula_probes, structure_dicts, or material_ids. If MatterGen cannot faithfully materialize the accepted prediction, return prediction_design_infeasible.
- In legacy/mixed modes, prefer generator formula_probes whenever you can express the branch concretely enough to generate the material directly.
- In legacy/mixed modes, if formula_probes cannot faithfully represent the needed chemistry or structure, use source "generator" with structure_dicts: a list of pymatgen Structure.as_dict() objects, one per generated material.
- In legacy/mixed modes, use source "mp_pool" with query/material_ids only when you intentionally want to select from the candidate pool.
- Use source "mattergen" with mattergen_requests when the accepted prediction is a chemical-system conditioned generation test.
- MP-pool query may use only supported keys: material_ids, formula_in, formula_regex, elements_all, elements_any, elements_none, nelements_min, nelements_max, nsites_min, nsites_max, band_gap_min, band_gap_max, formation_energy_per_atom_min, formation_energy_per_atom_max, density_min, density_max, volume_per_atom_min, volume_per_atom_max, crystal_system_in, spacegroup_number_in, spacegroup_number_min, spacegroup_number_max, preferred_order.
- Do not use unsupported query keys such as elements_exact, nelements_exact, branch_predicate, derived_predicate, include_ranks, materialization_guards, or object/list-valued selection_order.
- For deterministic MP-pool branches, omit selection_order and put the full deterministic sort specification in query.preferred_order as an ordered list such as ["formation_energy_per_atom asc", "band_gap desc", "density asc", "volume_per_atom asc", "formula asc"].
- Only use selection_order="random" if you explicitly want nondeterministic fallback selection.
- If you omit source and provide formula_probes or structure_dicts, the controller will treat that branch as generator.
- If you omit source and provide mattergen_requests, the controller will treat that branch as mattergen.
- In legacy/mixed modes, mixed bundles are allowed: branches may independently use MP-pool, generator, or MatterGen if Agent F agrees that this is the scientifically faithful materialization.
- Generator formula_probes must use the existing generator interface: template plus roles with element and oxidation_state.
- Generator structure_dicts must be valid pymatgen Structure.as_dict() objects; each branch count must equal the number of formula_probes or structure_dicts.
- MatterGen requests must condition on properties_to_condition_on.chemical_system and energy_above_hull=0.0, set diffusion_guidance_factor>0, and use matching filters.chemical_system.
- Put MatterGen sampling knobs directly in each request as top-level target_count, batch_size, and num_batches. Do not use a nested sampling object in new output; if prior dialogue shows nested sampling, treat it as an alias that the controller normalizes, not as missing sampling.
- Keep MatterGen sampling bounded: target_count <= {DEFAULT_MATTERGEN_MAX_TARGET_COUNT}, batch_size <= {DEFAULT_MATTERGEN_MAX_BATCH_SIZE}, num_batches <= {DEFAULT_MATTERGEN_MAX_NUM_BATCHES}, and batch_size*num_batches <= {DEFAULT_MATTERGEN_MAX_RAW_SAMPLES}. If a hard exact formula still underfills at that bounded budget, return prediction_design_infeasible for C/D redesign instead of escalating sampling further.
- MatterGen requests in A-F matched experiments must set filters.require_chemical_system_exact=true; element-subset products are drift and must not be used as primary/control evidence.
- MatterGen filters should be broad enough to sample. Avoid filters.min_sites unless required by the mechanism; if prior feedback reports too_few_sites, lower or remove min_sites and/or increase target_count/num_batches rather than changing source.
- In MatterGen-native mode, target_reduced_formula is a soft preference by default. Set require_target_reduced_formula=true only when the accepted prediction explicitly tests exact stoichiometry; after not_target_reduced_formula/no_accepted_structures feedback, do not repeat the same hard exact formula with equal or lower sampling.
- Increasing target_count, batch_size, or num_batches on a failed MatterGen branch is an execution-effort repair, not a new physical variable. It is allowed even when the matched control branch is already locked, provided chemical_system and physical filters are unchanged.
- Never output formula_probe_count or structure_dict_count as branch fields. They are compact feedback summaries only; executable generator branches must contain actual formula_probes or structure_dicts.
- Generator structures must be formal-charge-balanced under the oxidation states implied by the accepted mechanism. State the charge-balance relation you used when it is non-obvious.
- In legacy/mixed modes, if you switch an MP-pool query branch to generator because the MP pool underfills, preserve any density_min/density_max and volume_per_atom_min/volume_per_atom_max filters in branch.query so the controller can preflight and rescale lattices programmatically.
- Do not provide final selected rows; provide the executable blueprint that the controller will materialize after consensus.
- If no faithful execution blueprint exists because the accepted C/D prediction itself is not materializable without changing the scientific comparison, return status="prediction_design_infeasible", accepted_bundles=[], and prediction_design_feedback with prediction_ids, blocking_issue, why_execution_cannot_fix_it, and required_cd_reconsideration.

CONTEXT_JSON:
```json
{context_prompt}
```

Return only Agent E JSON.
"""


def prompt_execution_repair_proposal(context_json: str) -> str:
    context_prompt = compact_execution_context_json_for_prompt(context_json)
    return f"""Agent E: repair the previous execution materialization blueprint.

{_cdef_exploration_requirement(context_prompt)}

{EF_EVIDENCE_QUALITY_POLICY}

{EF_COMPACT_PAYLOAD_POLICY}

{mattergen_native_prompt_block(context_prompt)}

Use only the compact CONTEXT_JSON. Focus on current_inputs.accepted_predictions and repair_feedback.
Rules:
- Return the Agent E proposal wrapper: status, agent="E", bundles, agree, concede.
- Keep exactly one bundle per accepted prediction_id and keep the total branch counts equal to current_inputs.target_count when present.
- In MatterGen-native mode, every executable branch must remain source="mattergen". Do not switch a failed MatterGen branch to source="generator", source="mp_pool", formula_probes, structure_dicts, or material_ids.
- If repair_feedback reports an MP-pool branch underfilled, do not repeat the same source/query for that branch.
- In legacy/mixed modes, for an underfilled MP-pool branch, either provide a different executable MP-pool query that preserves the prediction, or switch that branch to source="generator".
- In MatterGen-native mode, repair failed branches with source="mattergen" and revised MatterGen chemical_system/filters/sampling unless the accepted C/D prediction itself is infeasible.
- Put sampling knobs directly in the MatterGen request as top-level target_count, batch_size, and num_batches. Do not use nested sampling in new output; if prior payloads had sampling={...}, the controller treats it as an alias for those top-level fields.
- Keep MatterGen sampling bounded: target_count <= {DEFAULT_MATTERGEN_MAX_TARGET_COUNT}, batch_size <= {DEFAULT_MATTERGEN_MAX_BATCH_SIZE}, num_batches <= {DEFAULT_MATTERGEN_MAX_NUM_BATCHES}, and batch_size*num_batches <= {DEFAULT_MATTERGEN_MAX_RAW_SAMPLES}. If a hard exact formula still underfills at that bounded budget, return prediction_design_infeasible for C/D redesign instead of escalating sampling further.
- If repair_feedback.successful_branches_to_preserve is present, those branches are controller-locked because they already materialized successfully. Preserve their source, count, query, MatterGen requests, filters, sampling knobs, prediction_ids, and expected_relation exactly; repair only failed branches.
- You may increase target_count, batch_size, or num_batches only on failed MatterGen branches; this does not mutate locked successful branches and does not by itself make the prediction design infeasible.
- If a locked successful branch must change for scientific comparability, do not mutate it in E/F. Return status="prediction_design_infeasible", accepted_bundles=[], and prediction_design_feedback so C/D can redesign the matched comparison.
- If MatterGen feedback reports too_few_sites, lower or remove filters.min_sites and/or increase target_count/num_batches. Do not repeat the same min_sites/max_sites/VPA filters that just underfilled.
- If MatterGen feedback reports not_target_reduced_formula or no_accepted_structures for a hard target_reduced_formula, increase sampling substantially or remove the hard target if exact stoichiometry is not the causal variable. Do not retry a sibling exact formula with the same small sampling budget.
- If source="generator", provide non-empty formula_probes or structure_dicts; for mixed-anion chemistry that formula_probes cannot express, use structure_dicts.
- If source="mattergen", provide exactly one mattergen_requests object and do not include formula_probes or structure_dicts.
- If repair_feedback says generator formula_probes materialized zero structures, do not repeat those same formula_probes. Use parseable structure_dicts, switch to a materially faithful MP-pool branch, or return status="prediction_design_infeasible".
- Never output formula_probe_count or structure_dict_count as branch fields. If feedback mentions structure_dict_summaries, use them only as audit clues and output actual formula_probes or actual parseable structure_dicts, not the summaries.
- If repair shows that every faithful execution path would change the accepted C/D prediction, return status="prediction_design_infeasible", accepted_bundles=[], and prediction_design_feedback instead of repeatedly proposing invalid bundles.
- Keep branches concise. Do not include long explanations or full dialogue history.
- Use only supported MP-pool query keys and generator branch fields.
- Use only supported MP-pool query keys, generator branch fields, and MatterGen mattergen_requests fields.

CONTEXT_JSON:
```json
{context_prompt}
```

Return only Agent E JSON.
"""


def prompt_execution_counterproposal(context_json: str, proposal: Mapping[str, Any], critique: Mapping[str, Any], cycle: int) -> str:
    context_prompt = compact_execution_context_json_for_prompt(context_json)
    return f"""Agent F: you rejected Agent E's execution plan in cycle {cycle}. You must now propose your own executable materialization blueprint that satisfies your own critique.

{_cdef_exploration_requirement(context_prompt)}

{EF_EVIDENCE_QUALITY_POLICY}

{EF_COMPACT_PAYLOAD_POLICY}

{mattergen_native_prompt_block(context_prompt)}

Do not only object. Either produce Agent E-compatible bundles, or produce a concise impossibility_certificate explaining why no faithful execution plan can satisfy the accepted predictions, target_count, supported query schema, and enabled materialization backend.
In MatterGen-native mode, your counterproposal must keep every executable branch as source="mattergen"; do not replace a failed MatterGen branch with generator formula_probes, structure_dicts, MP-pool queries, or material_ids.
If repair_feedback.successful_branches_to_preserve is present, preserve those successful branches exactly and repair only failed branches. If you need to change a preserved branch, return status="prediction_design_infeasible" with prediction_design_feedback instead of a mutated bundle.
Use the same output shape as Agent E: bundles with primary/control branches, one prediction_id per bundle, and backend-appropriate supported fields.
For MatterGen requests, put target_count, batch_size, and num_batches directly at the request top level. Do not use nested sampling in new output; prior nested sampling is a controller-normalized alias and is not a missing-field error.
Keep MatterGen sampling bounded: target_count <= {DEFAULT_MATTERGEN_MAX_TARGET_COUNT}, batch_size <= {DEFAULT_MATTERGEN_MAX_BATCH_SIZE}, num_batches <= {DEFAULT_MATTERGEN_MAX_NUM_BATCHES}, and batch_size*num_batches <= {DEFAULT_MATTERGEN_MAX_RAW_SAMPLES}. If a hard exact formula still underfills at that bounded budget, return prediction_design_infeasible for C/D redesign instead of escalating sampling further.
If your critique is that the accepted C/D prediction itself is infeasible, produce status="prediction_design_infeasible", accepted_bundles=[], and prediction_design_feedback rather than an Agent E bundle.
Never use formula_probe_count or structure_dict_count in a counterproposal; those are compact feedback summaries, not executable branch fields.
In legacy/mixed modes, if you use generator structure_dicts as a fallback from an underfilled MP-pool branch, keep the original physical numeric filters in branch.query and make the formulas formal-charge-balanced. Do not use the debate to repeatedly hand-tune lattice scale; the controller will preflight and rescale density/volume when bounds are present.
For parser compatibility, set the JSON field agent to "E" and use the same output shape as Agent E: status, agent, bundles, agree, concede.

CONTEXT_JSON:
```json
{context_prompt}
```

PREVIOUS_AGENT_E_JSON:
```json
{_prompt_execution_peer_json(proposal)}
```

AGENT_F_JSON:
```json
{_prompt_execution_peer_json(critique)}
```

Return only Agent E JSON.
"""


def prompt_execution_reverse_critique(context_json: str, original_proposal: Mapping[str, Any], counterproposal: Mapping[str, Any], cycle: int) -> str:
    context_prompt = compact_execution_context_json_for_prompt(context_json)
    return f"""Agent E: critique Agent F's execution counterproposal for cycle {cycle}.

{_cdef_exploration_requirement(context_prompt)}

{EF_EVIDENCE_QUALITY_POLICY}

{EF_COMPACT_PAYLOAD_POLICY}

{mattergen_native_prompt_block(context_prompt)}

You are now the reviewer. If Agent F's blueprint is faithful and executable, say agree=true. If not, provide exact required revisions.
Reject for composition, charge-balance, missing counts, or schema problems. Do not reject solely for small lattice-scale density/volume adjustments when branch.query includes the numeric bounds, because controller preflight can rescale generator lattices.
In MatterGen-native mode, reject any Agent F counterproposal that uses source="generator", source="mp_pool", formula_probes, structure_dicts, or material_ids instead of revised MatterGen requests.
If CONTEXT_JSON.repair_feedback.successful_branches_to_preserve is present, reject any counterproposal that changes a locked successful branch's source, count, query, MatterGen requests, filters, sampling knobs, prediction_ids, or expected_relation.
Embedded structure_dicts may be compact but parseable Structure.as_dict objects for prompt review: lattice.matrix, species, and abc coordinates are sufficient for the controller to parse them; _prompt_summary is non-executable audit metadata.
If Agent F returns prediction_design_infeasible and you agree that no faithful execution exists without changing the accepted C/D prediction, set agree=true and require the final status prediction_design_infeasible so the controller can send feedback back to C/D.
Agent F counterproposals intentionally use the Agent E proposal wrapper with top-level agent, bundles, agree, and concede fields for parser compatibility. Do not reject solely because that proposal wrapper is not the final execution_consensus schema. Judge whether the wrapped bundles can become accepted_bundles in the final consensus.
In that parser-compatible counterproposal wrapper, top-level agent="E" is correct even though Agent F authored it. Do not require agent="F" on the counterproposal payload itself.
If a prior MatterGen request used nested sampling={...}, the controller normalizes it into top-level target_count, batch_size, and num_batches. Do not reject solely for the alias when those values are present.
When auditing unsupported MatterGen filter keys, inspect only the current AGENT_F_COUNTERPROPOSAL_JSON executable filters. Do not infer that an unsupported key exists from earlier critique prose, rejected_bundles text, or compact feedback summaries. In particular, filters.exclude_reduced_formulas is valid, while filters.exclude_reduced_formulas_count is invalid only if it literally appears inside the current request.filters object.
For parser compatibility, set the JSON field agent to "F" and use the same output shape as Agent F: status, agent, required_revisions, agree, concede.

CONTEXT_JSON:
```json
{context_prompt}
```

ORIGINAL_AGENT_E_JSON:
```json
{_prompt_execution_peer_json(original_proposal)}
```

AGENT_F_COUNTERPROPOSAL_JSON:
```json
{_prompt_execution_peer_json(counterproposal)}
```

Return only Agent F JSON.
"""


def prompt_execution_reverse_final(context_json: str, counterproposal: Mapping[str, Any], reverse_critique: Mapping[str, Any]) -> str:
    context_prompt = compact_execution_context_json_for_prompt(context_json)
    return f"""Agent E: write the final execution plan consensus after auditing Agent F's counterproposal.

{_cdef_exploration_requirement(context_prompt)}

{EF_EVIDENCE_QUALITY_POLICY}

{EF_COMPACT_PAYLOAD_POLICY}

{mattergen_native_prompt_block(context_prompt)}

Only include bundles that both agents accept and that can be materialized by the enabled backend. In MatterGen-native mode, accepted bundles must use source="mattergen" for every branch.
If both agents agree that the accepted C/D prediction itself is not faithfully materializable, return status="prediction_design_infeasible", accepted_bundles=[], rejected_bundles, consensus_summary, and prediction_design_feedback. The controller will send that feedback back to C/D.
Return the final execution_consensus shape, not the Agent E proposal wrapper: top-level status, accepted_bundle_ids or accepted_bundles, rejected_bundles, consensus_summary, prediction_design_feedback when status is prediction_design_infeasible, and materialization_constraints if needed. Do not include top-level agent, agree, concede, or bundles in the final consensus.
Prefer accepted_bundle_ids over accepted_bundles. Do not repeat branch-level structure_dicts, formula_probes, mattergen_requests, material_ids, selected rows, or full generator payloads in final consensus; the controller will restore executable branches from the accepted counterproposal by bundle id.

CONTEXT_JSON:
```json
{context_prompt}
```

FINAL_AGENT_F_COUNTERPROPOSAL_JSON:
```json
{_prompt_execution_final_json(counterproposal)}
```

FINAL_AGENT_E_REVIEW_JSON:
```json
{_prompt_execution_final_json(reverse_critique)}
```

Return only final consensus JSON.
"""


def prompt_execution_critique(context_json: str, proposal: Mapping[str, Any], cycle: int) -> str:
    context_prompt = compact_execution_context_json_for_prompt(context_json)
    return f"""Agent F: critique Agent E's execution plan for cycle {cycle}.

{_cdef_exploration_requirement(context_prompt)}

{EF_EVIDENCE_QUALITY_POLICY}

{EF_COMPACT_PAYLOAD_POLICY}

{mattergen_native_prompt_block(context_prompt)}

Agent E proposals intentionally use a proposal wrapper with top-level agent, bundles, agree, and concede fields for parser compatibility. Do not reject solely because that proposal wrapper is not the final execution_consensus schema. Audit the scientific and executable content of the wrapped bundles; the controller will normalize accepted proposal bundles into final accepted_bundles after consensus.
In legacy/mixed modes, embedded structure_dicts may be compact but parseable Structure.as_dict objects for prompt review: lattice.matrix, species, and abc coordinates are sufficient for the controller to parse them; _prompt_summary is non-executable audit metadata. Do not reject a generator branch merely because xyz coordinates or verbose lattice scalars were omitted.
In MatterGen-native mode, reject any executable branch whose source is not "mattergen". A failed MatterGen branch must be repaired by revised MatterGen filters/sampling or escalated as prediction_design_infeasible, not by generator or MP-pool fallback.
For MatterGen requests, sampling knobs must be top-level target_count, batch_size, and num_batches in new output. If Agent E used nested sampling={...}, treat it as a controller-normalized alias and audit the numeric values instead of declaring sampling absent.
When auditing unsupported MatterGen filter keys, inspect only the current AGENT_E_JSON executable filters. Do not infer that an unsupported key exists from earlier critique prose, rejected_bundles text, or compact feedback summaries. In particular, filters.exclude_reduced_formulas is valid, while filters.exclude_reduced_formulas_count is invalid only if it literally appears inside the current request.filters object.
If CONTEXT_JSON.repair_feedback.successful_branches_to_preserve is present, reject any Agent E proposal that changes a locked successful branch's source, count, query, MatterGen requests, filters, sampling knobs, prediction_ids, or expected_relation. Locked branches must be copied exactly while only failed branches are repaired.
If the real problem is that the accepted C/D prediction cannot be faithfully materialized by MP pool, generator, or MatterGen without changing the scientific comparison, say so explicitly and require status="prediction_design_infeasible" with prediction_design_feedback. Do not keep asking E to fix an impossible execution blueprint.

CONTEXT_JSON:
```json
{context_prompt}
```

AGENT_E_JSON:
```json
{_prompt_execution_peer_json(proposal)}
```

Return only Agent F JSON.
"""


def prompt_execution_revision(context_json: str, proposal: Mapping[str, Any], critique: Mapping[str, Any], cycle: int) -> str:
    context_prompt = compact_execution_context_json_for_prompt(context_json)
    return f"""Agent E: revise after Agent F critique in cycle {cycle}.

{_cdef_exploration_requirement(context_prompt)}

{EF_EVIDENCE_QUALITY_POLICY}

{EF_COMPACT_PAYLOAD_POLICY}

{mattergen_native_prompt_block(context_prompt)}

If Agent F finds a query impossible, narrow or replace it. If F says a bundle does not really test the mechanism, fix the comparison design.
If F rejects the selection order, remove branch selection_order for deterministic plans and express the exact sort order in query.preferred_order as an ordered list with explicit asc/desc directions.
If MP-pool selection or formula_probes cannot represent a scientifically required branch, use source "mattergen" with mattergen_requests in MatterGen-native mode, or source "generator" with structure_dicts in legacy mode, instead of returning no_materialized_consensus.
In MatterGen-native mode, do not use generator or MP-pool fallback. If MatterGen underfilled, revise MatterGen filters/sampling; for too_few_sites rejects lower/remove filters.min_sites or increase target_count/num_batches.
If repair_feedback.successful_branches_to_preserve is present, copy those successful branches exactly and revise only failed branches. Changing a locked branch is not an execution repair; if that change is necessary, return status="prediction_design_infeasible" with prediction_design_feedback for C/D.
For generator revisions, fix formula/charge-balance errors directly. Preserve density/volume numeric bounds in branch.query and let controller preflight handle lattice rescaling.
Never output formula_probe_count or structure_dict_count as branch fields. If Agent F asks for explicit formula_probes, copy those exact formula strings into formula_probes.
If repair_feedback shows those formula_probes already materialized zero structures, do not copy them again; use parseable structure_dicts, a faithful executable MP-pool branch, or status="prediction_design_infeasible".
If Agent F has shown that the accepted C/D prediction itself is infeasible, return status="prediction_design_infeasible", accepted_bundles=[], and prediction_design_feedback instead of repeating an invalid bundle.
Keep using the Agent E proposal wrapper in this revision: top-level status, agent, bundles, agree, and concede. If Agent F only objected that proposal-wrapper fields differ from the final execution_consensus schema, preserve the accepted bundles and clarify that the controller will normalize them to accepted_bundles in the final consensus.

CONTEXT_JSON:
```json
{context_prompt}
```

PREVIOUS_AGENT_E_JSON:
```json
{_prompt_execution_peer_json(proposal)}
```

AGENT_F_JSON:
```json
{_prompt_execution_peer_json(critique)}
```

Return only Agent E JSON.
"""


def prompt_execution_final(context_json: str, proposal: Mapping[str, Any], critique: Mapping[str, Any]) -> str:
    context_prompt = compact_execution_context_json_for_prompt(context_json)
    return f"""Agent F: write the final execution plan consensus.

{_cdef_exploration_requirement(context_prompt)}

{EF_EVIDENCE_QUALITY_POLICY}

{EF_COMPACT_PAYLOAD_POLICY}

{mattergen_native_prompt_block(context_prompt)}

Only include bundles that both agents accept and that can be materialized by the enabled backend. In MatterGen-native mode, every accepted branch must use source="mattergen"; generator and MP-pool fallback are invalid. Keep one prediction_id per bundle; do not aggregate multiple predictions into one bundle.
You are auditing the executable blueprint, not the final row-level selection, because the controller will materialize concrete rows after consensus.
If MatterGen-native mode is active and a branch can be expressed as chemical-system conditioning, require source="mattergen" with mattergen_requests.
If a branch can be expressed concretely as generator-readable formula_probes or structure_dicts in legacy mode, prefer that path even if source is omitted.
If MP-pool materialization is insufficient but the test is still scientifically required, require source "mattergen" in MatterGen-native mode or source "generator" formula_probes/structure_dicts in legacy mode instead of returning no_materialized_consensus.
If MatterGen materialization underfilled, require revised MatterGen filters/sampling. For too_few_sites rejects, lower/remove filters.min_sites or increase target_count/num_batches; do not approve a switch to generator or MP-pool.
If a failed MatterGen branch only needs more target_count/batch_size/num_batches, approve that repair even when another branch is locked; sampling effort is not a physical confound.
If repair_feedback.successful_branches_to_preserve is present, final consensus must preserve those successful branches exactly and only change failed branches. If preserving them makes the comparison impossible, return prediction_design_infeasible with prediction_design_feedback.
Return no_materialized_consensus only if MP-pool selection, generator formula_probes/structure_dicts, and MatterGen mattergen_requests cannot faithfully implement the accepted predictions.
Return prediction_design_infeasible instead of no_materialized_consensus when the obstacle is the accepted C/D prediction design itself: any faithful execution would require changing the primary/control definition, chemistry, or scope. Include prediction_design_feedback with prediction_ids, blocking_issue, why_execution_cannot_fix_it, and required_cd_reconsideration.
Return the final execution_consensus shape, not the Agent E proposal wrapper: top-level status, accepted_bundle_ids or accepted_bundles, rejected_bundles, consensus_summary, prediction_design_feedback when status is prediction_design_infeasible, and materialization_constraints if needed. Do not include top-level agent, agree, concede, or bundles in the final consensus.
Prefer accepted_bundle_ids over accepted_bundles. Do not repeat branch-level structure_dicts, formula_probes, mattergen_requests, material_ids, selected rows, or full generator payloads in final consensus; the controller will restore executable branches from Agent E's accepted proposal by bundle id.

CONTEXT_JSON:
```json
{context_prompt}
```

FINAL_AGENT_E_JSON:
```json
{_prompt_execution_final_json(proposal)}
```

FINAL_AGENT_F_JSON:
```json
{_prompt_execution_final_json(critique)}
```

Return only final consensus JSON.
"""


def compact_round_summary_for_postmortem(summary: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = summary.get("evaluation_summary") if isinstance(summary.get("evaluation_summary"), Mapping) else {}
    accepted_mechanisms = summary.get("accepted_mechanisms")
    accepted_predictions = summary.get("accepted_predictions")
    compact_bundles: list[dict[str, Any]] = []
    for item in summary.get("bundle_results", []) if isinstance(summary.get("bundle_results"), list) else []:
        if not isinstance(item, Mapping):
            continue
        compact_bundles.append(
            {
                key: value
                for key, value in {
                    "bundle_id": item.get("bundle_id"),
                    "prediction_ids": item.get("prediction_ids"),
                    "expected_relation": item.get("expected_relation"),
                    "supported": item.get("supported"),
                    "delta": item.get("delta"),
                    "primary_mean_e_hull": item.get("primary_mean_e_hull"),
                    "control_mean_e_hull": item.get("control_mean_e_hull"),
                    "primary_min_e_hull": item.get("primary_min_e_hull"),
                    "control_min_e_hull": item.get("control_min_e_hull"),
                    "primary_sun_count": item.get("primary_sun_count"),
                    "control_sun_count": item.get("control_sun_count"),
                    "mechanism_validated_sun_count": item.get("mechanism_validated_sun_count"),
                }.items()
                if value not in (None, "", [])
            }
        )
    return {
        "round": summary.get("round"),
        "accepted_mechanisms": [_compact_mechanism_for_context(item) for item in accepted_mechanisms[:4]]
        if isinstance(accepted_mechanisms, list)
        else [],
        "accepted_predictions": [
            _compact_prediction_for_context(item) for item in accepted_predictions[:PROMPT_MATERIALIZATION_ITEM_LIMIT]
        ]
        if isinstance(accepted_predictions, list)
        else [],
        "evaluation_summary": {
            key: evaluation.get(key)
            for key in (
                "count",
                "mean_e_hull",
                "min_e_hull",
                "max_e_hull",
                "stable_count",
                "primary_sun_count",
                "control_sun_count",
                "mechanism_validated_sun_count",
                "support_rate",
                "supported_bundle_count",
                "bundle_count",
            )
        },
        "bundle_results": compact_bundles,
    }


def prompt_principle_postmortem(context_json: str, round_summary: Mapping[str, Any]) -> str:
    return f"""A/B joint postmortem: update the active materials-principle program after the latest evaluated round.

{PRINCIPLE_POSTMORTEM_COMPACT_RULES}

CONTEXT_JSON:
```json
{context_json}
```

LATEST_ROUND_SUMMARY_JSON:
```json
{prompt_json_dumps(compact_round_summary_for_postmortem(round_summary))}
```

Required reasoning discipline:
- Decide whether the active principle was supported, contradicted, or remains ambiguous.
- Use primary/control mean e_hull, support_rate, and SUN accounting separately.
- If control_sun_count > 0, explain whether it is a failure boundary or a control-derived new mechanism lead.
- If continuing, narrow or refine the current principle and name the next falsifiable test focus.
- If finalizing, write an experience_book_entry that explains all evidence from this principle program, not only the latest round.
- Before finalizing or rejecting, compare against CONTEXT_JSON.principle_book_tail. Same material family plus same microscopic causal driver must update that existing experience, not append a duplicate. Set experience_book_entry.updates_principle_id when updating.
- A new experience-book entry is allowed only for a distinct causal mechanism or distinct material/descriptor scope that cannot be represented as a boundary, transfer test, or refinement of an existing entry.

Return only principle postmortem JSON.
"""


def reconcile_execution_consensus(consensus: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(consensus)
    if payload.get("status") != "consensus":
        return payload

    proposal_bundles = proposal.get("bundles")
    if not isinstance(proposal_bundles, list):
        proposal_bundles = proposal.get("accepted_bundles")
    if isinstance(proposal_bundles, list):
        proposal_bundles = _normalize_execution_bundle_list(proposal_bundles)
    accepted_bundles = payload.get("accepted_bundles")
    if isinstance(accepted_bundles, list) and accepted_bundles:
        if isinstance(proposal_bundles, list) and proposal_bundles:
            by_alias: dict[str, dict[str, Any]] = {}
            for index, proposal_bundle in enumerate(proposal_bundles, start=1):
                if not isinstance(proposal_bundle, Mapping):
                    continue
                normalized_proposal_bundle = dict(proposal_bundle)
                for alias in _execution_bundle_aliases(normalized_proposal_bundle, index):
                    by_alias.setdefault(alias, normalized_proposal_bundle)
            restored_bundles: list[dict[str, Any]] = []
            for index, bundle in enumerate(_normalize_execution_bundle_list(accepted_bundles), start=1):
                proposal_bundle = None
                for alias in _execution_bundle_aliases(bundle, index):
                    proposal_bundle = by_alias.get(alias)
                    if proposal_bundle is not None:
                        break
                restored_bundles.append(_merge_execution_bundle_from_proposal(bundle, proposal_bundle))
            payload["accepted_bundles"] = restored_bundles
        return payload

    if not isinstance(proposal_bundles, list) or not proposal_bundles:
        return payload

    accepted_ids = {str(item) for item in payload.get("accepted_bundle_ids", []) if str(item).strip()} if isinstance(payload.get("accepted_bundle_ids"), list) else set()
    if accepted_ids:
        selected = [
            dict(bundle)
            for index, bundle in enumerate(proposal_bundles, start=1)
            if isinstance(bundle, Mapping) and (_execution_bundle_aliases(bundle, index) & accepted_ids)
        ]
    else:
        selected = [dict(bundle) for bundle in proposal_bundles]
    if selected:
        payload["accepted_bundles"] = selected
    return payload


def _merge_execution_bundle_from_proposal(bundle: Mapping[str, Any], proposal_bundle: Mapping[str, Any] | None) -> dict[str, Any]:
    """Restore materialization details when final consensus only summarizes a bundle."""

    result = dict(bundle)
    if not isinstance(proposal_bundle, Mapping):
        return result
    for key in ("prediction_ids", "expected_relation", "rationale_summary", "selection_notes"):
        if key not in result and key in proposal_bundle:
            result[key] = proposal_bundle[key]
    for role in ("primary", "control"):
        branch = result.get(role)
        proposal_branch = proposal_bundle.get(role)
        if isinstance(proposal_branch, Mapping) and not _branch_has_materialization_spec(branch):
            result[role] = dict(proposal_branch)
    return result


def _branch_has_materialization_spec(branch: Any) -> bool:
    if not isinstance(branch, Mapping):
        return False
    formula_probes = branch.get("formula_probes")
    structure_dicts = branch.get("structure_dicts")
    mattergen_requests = branch.get("mattergen_requests")
    material_ids = branch.get("material_ids")
    source = str(branch.get("source") or "").strip()
    if isinstance(formula_probes, list) and formula_probes:
        return True
    if isinstance(structure_dicts, list) and structure_dicts:
        return True
    if isinstance(mattergen_requests, list) and mattergen_requests:
        return True
    if isinstance(material_ids, list) and material_ids:
        return True
    if source in {"generator", "mattergen"}:
        return False
    if isinstance(branch.get("query"), Mapping):
        return True
    return False


def _constraint_text(value: Any) -> str:
    if value in (None, "", []):
        return ""
    if isinstance(value, Mapping):
        return " ".join(_constraint_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_constraint_text(item) for item in value)
    return str(value)


def _branch_constraint_text(
    *,
    bundle: Mapping[str, Any],
    branch: Mapping[str, Any],
    role: str,
    plan_constraints: Mapping[str, Any] | None,
) -> str:
    parts = [
        _constraint_text(branch.get("acceptance_constraint")),
        _constraint_text(branch.get("drift_rejection_criteria")),
        _constraint_text(branch.get("selection_notes")),
        _constraint_text(bundle.get("selection_notes")),
        _constraint_text(bundle.get("rationale_summary")),
    ]
    if isinstance(plan_constraints, Mapping):
        parts.extend(
            [
                _constraint_text(plan_constraints.get(f"{role}_acceptance_constraint")),
                _constraint_text(plan_constraints.get(f"{role}_drift_rejection_criteria")),
                _constraint_text(plan_constraints.get("materialization_constraints")),
            ]
        )
    return " ".join(part for part in parts if part)


def _requires_phosphate_like_p_o(text: str) -> bool:
    lowered = text.lower()
    return any(
        term in lowered
        for term in (
            "phosphate",
            "pyrophosphate",
            "metaphosphate",
            "oxophosphate",
            "p-o",
            "po4",
            "p2o7",
        )
    )


def _ordered_site_symbol(site: Any) -> str:
    specie = getattr(site, "specie", None)
    symbol = getattr(specie, "symbol", None)
    if symbol:
        return str(symbol)
    species = getattr(site, "species", None)
    if species:
        try:
            return str(next(iter(species.elements)).symbol)
        except Exception:
            return str(species)
    return ""


def phosphate_like_p_o_rejection_reason(structure: Structure, *, max_p_o_distance: float = 2.25) -> str | None:
    amounts = structure.composition.get_el_amt_dict()
    p_count = float(amounts.get("P") or 0.0)
    o_count = float(amounts.get("O") or 0.0)
    if p_count <= 0 or o_count <= 0:
        return "phosphate_like_p_o_missing_P_or_O"
    ratio = o_count / p_count
    if ratio < 3.0:
        return f"phosphate_like_p_o_low_o_to_p_ratio:{ratio:.3g}"
    p_coordination: list[int] = []
    for site in structure:
        if _ordered_site_symbol(site) != "P":
            continue
        oxygen_neighbors = [
            neighbor
            for neighbor in structure.get_neighbors(site, max_p_o_distance)
            if _ordered_site_symbol(neighbor) == "O"
        ]
        p_coordination.append(len(oxygen_neighbors))
    if not p_coordination:
        return "phosphate_like_p_o_missing_P_sites"
    min_coordination = min(p_coordination)
    if min_coordination < 3:
        return f"phosphate_like_p_o_undercoordinated_P:min_o_neighbors={min_coordination}"
    return None


def mattergen_structure_drift_rejection_reason(
    structure: Structure,
    *,
    bundle: Mapping[str, Any],
    branch: Mapping[str, Any],
    role: str,
    plan_constraints: Mapping[str, Any] | None,
) -> str | None:
    text = _branch_constraint_text(
        bundle=bundle,
        branch=branch,
        role=role,
        plan_constraints=plan_constraints,
    )
    if _requires_phosphate_like_p_o(text):
        amounts = structure.composition.get_el_amt_dict()
        if amounts.get("P") and amounts.get("O"):
            reason = phosphate_like_p_o_rejection_reason(structure)
            if reason:
                return reason
    return None


def _append_constraint_items(existing: Any, additions: Sequence[str]) -> list[Any]:
    items: list[Any] = []
    if isinstance(existing, list):
        items.extend(existing)
    elif existing not in (None, "", []):
        items.append(existing)
    for item in additions:
        if item and item not in items:
            items.append(item)
    return items


def execution_plan_with_prediction_drift_constraints(
    plan: Mapping[str, Any],
    prediction_consensus: Mapping[str, Any],
) -> dict[str, Any]:
    """Carry C/D materialization drift criteria into E/F executable branches."""

    result = dict(plan)
    bundles = result.get("accepted_bundles")
    predictions = prediction_consensus.get("accepted_predictions") if isinstance(prediction_consensus, Mapping) else None
    if not isinstance(bundles, list) or not isinstance(predictions, list):
        return result
    prediction_by_id = {
        str(prediction.get("id") or ""): prediction
        for prediction in predictions
        if isinstance(prediction, Mapping) and str(prediction.get("id") or "").strip()
    }
    updated_bundles: list[Any] = []
    for raw_bundle in bundles:
        if not isinstance(raw_bundle, Mapping):
            updated_bundles.append(raw_bundle)
            continue
        bundle = dict(raw_bundle)
        prediction_ids = [str(item) for item in bundle.get("prediction_ids", []) if str(item).strip()]
        for role in ("primary", "control"):
            branch = bundle.get(role)
            if not isinstance(branch, Mapping):
                continue
            drift_items: list[str] = []
            provenance: list[dict[str, Any]] = []
            for prediction_id in prediction_ids:
                prediction = prediction_by_id.get(prediction_id)
                if not isinstance(prediction, Mapping):
                    continue
                comparison = prediction.get("comparison_design")
                if not isinstance(comparison, Mapping):
                    continue
                mattergen_design = comparison.get(f"{role}_mattergen")
                if not isinstance(mattergen_design, Mapping):
                    continue
                prediction_drift_items: list[str] = []
                for key in ("drift_rejection_criteria", "mechanism_alignment", "control_alignment", "held_fixed_variables"):
                    value = mattergen_design.get(key)
                    if isinstance(value, list):
                        prediction_drift_items.extend(str(item) for item in value if str(item).strip())
                    elif value not in (None, "", []):
                        prediction_drift_items.append(str(value))
                if prediction_drift_items:
                    drift_items.extend(prediction_drift_items)
                    provenance.append(
                        {
                            "prediction_id": prediction_id,
                            "source": f"comparison_design.{role}_mattergen",
                            "chemical_system": mattergen_design.get("chemical_system"),
                        }
                    )
            if not drift_items:
                continue
            branch_copy = dict(branch)
            branch_copy["drift_rejection_criteria"] = _append_constraint_items(
                branch_copy.get("drift_rejection_criteria"),
                drift_items,
            )
            existing_provenance = branch_copy.get("prediction_drift_constraint_provenance")
            if isinstance(existing_provenance, list):
                branch_copy["prediction_drift_constraint_provenance"] = existing_provenance + provenance
            else:
                branch_copy["prediction_drift_constraint_provenance"] = provenance
            bundle[role] = branch_copy
        updated_bundles.append(bundle)
    result["accepted_bundles"] = updated_bundles
    return result


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _branch_numeric_constraint(branch: Mapping[str, Any], key: str) -> float | None:
    value = _as_float(branch.get(key))
    if value is not None:
        return value
    query = branch.get("query")
    if isinstance(query, Mapping):
        return _as_float(query.get(key))
    return None


def formal_charge_imbalance(structure: Structure) -> float | None:
    total = 0.0
    for element, amount in structure.composition.get_el_amt_dict().items():
        if element not in COMMON_FORMAL_OXIDATION_STATES:
            return None
        total += COMMON_FORMAL_OXIDATION_STATES[element] * float(amount)
    return total


def structure_preflight_summary(structure: Structure) -> dict[str, Any]:
    charge = formal_charge_imbalance(structure)
    return {
        "formula": reduced_formula(structure),
        "nsites": len(structure),
        "volume": round(float(structure.volume), 6),
        "volume_per_atom": round(float(structure.volume) / max(1, len(structure)), 6),
        "density": round(float(structure.density), 6),
        "formal_charge_imbalance": None if charge is None else round(float(charge), 6),
    }


def _target_volume_for_generator_structure(
    structure: Structure,
    branch: Mapping[str, Any],
) -> tuple[float | None, list[str]]:
    notes: list[str] = []
    nsites = max(1, len(structure))
    density_min = _branch_numeric_constraint(branch, "density_min")
    density_max = _branch_numeric_constraint(branch, "density_max")
    volume_per_atom_min = _branch_numeric_constraint(branch, "volume_per_atom_min")
    volume_per_atom_max = _branch_numeric_constraint(branch, "volume_per_atom_max")

    lower = 0.0
    upper = float("inf")
    if volume_per_atom_min is not None:
        lower = max(lower, nsites * volume_per_atom_min)
    if volume_per_atom_max is not None:
        upper = min(upper, nsites * volume_per_atom_max)

    cell_mass_amu = float(structure.composition.weight)
    if density_max is not None and density_max > 0:
        lower = max(lower, cell_mass_amu * AMU_PER_A3_TO_G_CM3 / density_max)
    if density_min is not None and density_min > 0:
        upper = min(upper, cell_mass_amu * AMU_PER_A3_TO_G_CM3 / density_min)

    if lower > upper:
        notes.append(
            "incompatible density/volume constraints: "
            f"minimum allowed volume {lower:.3f} A^3 exceeds maximum allowed volume {upper:.3f} A^3"
        )
        return None, notes

    has_explicit_window = any(
        value is not None
        for value in (density_min, density_max, volume_per_atom_min, volume_per_atom_max)
    )
    if not has_explicit_window:
        lower = max(lower, nsites * 8.0)
        upper = min(upper, nsites * 28.0)

    current = float(structure.volume)
    if lower <= current <= upper:
        return None, notes

    target = nsites * DEFAULT_GENERATOR_VOLUME_PER_ATOM_TARGET
    target = max(lower, min(upper, target))
    if target <= 0 or target == float("inf"):
        return None, notes
    notes.append(
        f"rescaled generator lattice volume from {current:.3f} to {target:.3f} A^3 "
        "to satisfy density/volume constraints"
    )
    return target, notes


def prepare_generator_structure(
    structure: Structure,
    branch: Mapping[str, Any],
) -> tuple[Structure, list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []

    charge = formal_charge_imbalance(structure)
    if charge is not None and abs(charge) > 1e-6:
        errors.append(
            f"formal charge imbalance is {charge:g} under common oxidation states "
            f"for formula {reduced_formula(structure)}"
        )

    target_volume, volume_notes = _target_volume_for_generator_structure(structure, branch)
    notes.extend(volume_notes)
    if target_volume is not None:
        structure = structure.copy()
        structure.scale_lattice(target_volume)

    return structure, notes, errors


def mattergen_config_from_args(args: argparse.Namespace, root: Path, work_dir: Path) -> dict[str, Any]:
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


def mattergen_prompt_defaults(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": "mattergen",
        "checkpoint": getattr(args, "mattergen_checkpoint", DEFAULT_MATTERGEN_CHECKPOINT),
        "conditioning_required": {
            "chemical_system": "required, e.g. Rb-Cd-Br",
            "energy_above_hull": 0.0,
        },
        "diffusion_guidance_factor": float(
            getattr(args, "mattergen_diffusion_guidance_factor", DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR)
        ),
        "filters": {
            "max_sites": int(getattr(args, "mattergen_max_sites", DEFAULT_MATTERGEN_MAX_SITES)),
            "min_volume_per_atom": float(
                getattr(args, "mattergen_min_volume_per_atom", DEFAULT_MATTERGEN_MIN_VOLUME_PER_ATOM)
            ),
            "max_volume_per_atom": float(
                getattr(args, "mattergen_max_volume_per_atom", DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM)
            ),
            "deduplicate_reduced_formula": True,
            "require_chemical_system_exact": True,
            "require_target_reduced_formula": False,
        },
        "sampling": {
            "target_count_default": int(getattr(args, "mattergen_target_count", DEFAULT_MATTERGEN_TARGET_COUNT)),
            "batch_size": int(getattr(args, "mattergen_batch_size", DEFAULT_MATTERGEN_BATCH_SIZE)),
            "num_batches": int(getattr(args, "mattergen_num_batches", DEFAULT_MATTERGEN_NUM_BATCHES)),
            "max_target_count_per_request": DEFAULT_MATTERGEN_MAX_TARGET_COUNT,
            "max_batch_size_per_request": DEFAULT_MATTERGEN_MAX_BATCH_SIZE,
            "max_num_batches_per_request": DEFAULT_MATTERGEN_MAX_NUM_BATCHES,
            "max_raw_samples_per_request": DEFAULT_MATTERGEN_MAX_RAW_SAMPLES,
        },
    }


def _chemical_system_elements(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split("-") if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _formula_element_symbols(formula: Any) -> set[str]:
    text = str(formula or "").strip()
    if not text:
        return set()
    try:
        return {str(element.symbol) for element in Composition(text).elements}
    except Exception:
        return set()


def _filter_excluded_formulas_for_chemical_system(
    formulas: set[str],
    *,
    elements: Sequence[str],
    require_exact: bool,
) -> list[str]:
    system = {str(element).strip() for element in elements if str(element).strip()}
    if not system:
        return sorted(str(formula).strip() for formula in formulas if str(formula).strip())
    filtered: list[str] = []
    for formula in formulas:
        text = str(formula or "").strip()
        if not text:
            continue
        formula_elements = _formula_element_symbols(text)
        if not formula_elements:
            continue
        if require_exact:
            if formula_elements == system:
                filtered.append(text)
        elif formula_elements <= system:
            filtered.append(text)
    return sorted(set(filtered))


def _normalise_mattergen_request_for_physics(
    raw_request: Mapping[str, Any],
    *,
    branch_id: str,
    mattergen_config: Mapping[str, Any],
    excluded_formulas: set[str],
    max_sites: int,
    count: int,
) -> dict[str, Any]:
    request = dict(raw_request)
    request["backend"] = "mattergen"
    request.setdefault("request_id", f"{branch_id}_mattergen")
    request.setdefault("checkpoint", mattergen_config.get("checkpoint") or DEFAULT_MATTERGEN_CHECKPOINT)
    request.setdefault("model_path", mattergen_config.get("model_path"))
    normalization_notes: list[str] = []
    existing_notes = request.get("controller_normalization_notes")
    if isinstance(existing_notes, list):
        normalization_notes.extend(str(note) for note in existing_notes if note)
    elif existing_notes:
        normalization_notes.append(str(existing_notes))
    sampling = request.pop("sampling", None)
    if isinstance(sampling, Mapping):
        for key in ("target_count", "batch_size", "num_batches"):
            if request.get(key) is None and sampling.get(key) is not None:
                request[key] = sampling.get(key)
    request["target_count"] = max(count, int(request.get("target_count") or mattergen_config.get("target_count") or count))
    request["batch_size"] = int(request.get("batch_size") or mattergen_config.get("batch_size") or DEFAULT_MATTERGEN_BATCH_SIZE)
    request["num_batches"] = int(request.get("num_batches") or mattergen_config.get("num_batches") or DEFAULT_MATTERGEN_NUM_BATCHES)
    request.setdefault(
        "diffusion_guidance_factor",
        mattergen_config.get("diffusion_guidance_factor", DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR),
    )
    properties = dict(request.get("properties_to_condition_on") or {})
    properties.setdefault("energy_above_hull", 0.0)
    request["properties_to_condition_on"] = properties
    filters = dict(request.get("filters") or {})
    chemical_system = filters.get("chemical_system") or properties.get("chemical_system")
    elements = _chemical_system_elements(chemical_system)
    if elements:
        filters["chemical_system"] = elements
    filters["require_chemical_system_exact"] = _bool_value(filters.get("require_chemical_system_exact", True)) is True
    if "target_reduced_formula" not in filters and request.get("target_reduced_formula"):
        filters["target_reduced_formula"] = request.get("target_reduced_formula")
    if "require_target_reduced_formula" not in filters and "require_target_reduced_formula" in request:
        filters["require_target_reduced_formula"] = request.get("require_target_reduced_formula")
    filters["require_target_reduced_formula"] = _bool_value(filters.get("require_target_reduced_formula")) is True
    if filters["require_target_reduced_formula"]:
        before = int(request.get("num_batches") or DEFAULT_MATTERGEN_NUM_BATCHES)
        request["num_batches"] = max(
            before,
            DEFAULT_MATTERGEN_HARD_TARGET_MIN_NUM_BATCHES,
        )
        if request["num_batches"] != before:
            normalization_notes.append(
                f"raised hard-target num_batches from {before} to {request['num_batches']}"
            )
    target_count_cap = max(count, DEFAULT_MATTERGEN_MAX_TARGET_COUNT)
    if request["target_count"] > target_count_cap:
        normalization_notes.append(
            f"capped target_count from {request['target_count']} to {target_count_cap}"
        )
        request["target_count"] = target_count_cap
    if request["batch_size"] > DEFAULT_MATTERGEN_MAX_BATCH_SIZE:
        normalization_notes.append(
            f"capped batch_size from {request['batch_size']} to {DEFAULT_MATTERGEN_MAX_BATCH_SIZE}"
        )
        request["batch_size"] = DEFAULT_MATTERGEN_MAX_BATCH_SIZE
    if request["num_batches"] > DEFAULT_MATTERGEN_MAX_NUM_BATCHES:
        normalization_notes.append(
            f"capped num_batches from {request['num_batches']} to {DEFAULT_MATTERGEN_MAX_NUM_BATCHES}"
        )
        request["num_batches"] = DEFAULT_MATTERGEN_MAX_NUM_BATCHES
    raw_batch_cap = max(1, DEFAULT_MATTERGEN_MAX_RAW_SAMPLES // max(1, int(request["batch_size"])))
    if request["num_batches"] > raw_batch_cap:
        normalization_notes.append(
            f"capped num_batches from {request['num_batches']} to {raw_batch_cap} to keep raw samples <= {DEFAULT_MATTERGEN_MAX_RAW_SAMPLES}"
        )
        request["num_batches"] = raw_batch_cap
    filters.setdefault("deduplicate_reduced_formula", True)
    filters["max_sites"] = min(max_sites, int(filters.get("max_sites") or mattergen_config.get("max_sites") or max_sites))
    filters.setdefault("min_volume_per_atom", mattergen_config.get("min_volume_per_atom", DEFAULT_MATTERGEN_MIN_VOLUME_PER_ATOM))
    filters.setdefault("max_volume_per_atom", mattergen_config.get("max_volume_per_atom", DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM))
    existing_excluded = set(_chemical_system_elements(filters.get("exclude_reduced_formulas")))
    filters["exclude_reduced_formulas"] = _filter_excluded_formulas_for_chemical_system(
        existing_excluded | excluded_formulas,
        elements=elements,
        require_exact=filters["require_chemical_system_exact"],
    )
    request["filters"] = filters
    if normalization_notes:
        request["controller_normalization_notes"] = normalization_notes
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
#SBATCH --job-name=af_mattergen
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
    print(f"[{utc_now()}] af_mattergen_submitted job_id={job_id} run_dir={run_dir}", flush=True)
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
    print(f"[{utc_now()}] af_mattergen_finished job_id={job_id} elapsed_sec={time.time() - started:.1f}", flush=True)
    return job_id


def _run_mattergen_request(
    request: Mapping[str, Any],
    *,
    branch_id: str,
    mattergen_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(str(mattergen_config["root"]))
    base_work_dir = Path(str(mattergen_config["work_dir"]))
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", branch_id)[:80] or "branch"
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
        with (run_dir / "adapter_stdout.log").open("w", encoding="utf-8") as stdout_handle, (
            run_dir / "adapter_stderr.log"
        ).open("w", encoding="utf-8") as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=str(root),
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
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


def _execution_branch_cache_ref(bundle_id: str, role: str) -> str:
    return f"{bundle_id}.{role}"


def _execution_branch_fingerprint(
    *,
    bundle: Mapping[str, Any],
    role: str,
    branch: Mapping[str, Any],
    plan_constraints: Mapping[str, Any] | None,
) -> str:
    payload = {
        "bundle_id": str(bundle.get("id") or ""),
        "role": role,
        "prediction_ids": [str(item) for item in bundle.get("prediction_ids", []) if str(item).strip()],
        "expected_relation": str(bundle.get("expected_relation") or ""),
        "branch": dict(branch),
        "plan_constraints": dict(plan_constraints or {}),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)


def select_bundle_records(
    pool_records: Sequence[Mapping[str, Any]],
    *,
    bundle: Mapping[str, Any],
    seed: int,
    seen_material_ids: set[str],
    max_sites: int,
    known_formulas: set[str] | None,
    mattergen_config: Mapping[str, Any] | None = None,
    plan_constraints: Mapping[str, Any] | None = None,
    branch_cache_records: dict[str, list[dict[str, Any]]] | None = None,
    branch_cache_fingerprints: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    bundle_id = str(bundle.get("id") or "")
    prediction_ids = [str(item) for item in bundle.get("prediction_ids", []) if str(item).strip()]
    selected: list[dict[str, Any]] = []
    for role in ("primary", "control"):
        branch = bundle.get(role)
        if not isinstance(branch, Mapping):
            errors.append(f"{bundle_id}.{role} must be an object")
            continue
        formula_probes = branch.get("formula_probes")
        structure_dicts = branch.get("structure_dicts")
        mattergen_requests = branch.get("mattergen_requests")
        has_formula_probes = isinstance(formula_probes, list) and bool(formula_probes)
        has_structure_dicts = isinstance(structure_dicts, list) and bool(structure_dicts)
        has_mattergen_requests = isinstance(mattergen_requests, list) and bool(mattergen_requests)
        source = str(branch.get("source") or bundle.get("source") or "mp_pool")
        if (not branch.get("source")) and (has_formula_probes or has_structure_dicts):
            source = "generator"
        if (not branch.get("source")) and has_mattergen_requests:
            source = "mattergen"
        count = int(branch.get("count") or 0)
        if source == "generator" and (has_formula_probes or has_structure_dicts) and count <= 0:
            count = len(structure_dicts) if has_structure_dicts else len(formula_probes)
        query = branch.get("query")
        material_ids = [str(item) for item in branch.get("material_ids", []) if str(item).strip()]
        matches: list[dict[str, Any]] = []
        cache_ref = _execution_branch_cache_ref(bundle_id, role)
        cache_fingerprint = _execution_branch_fingerprint(
            bundle=bundle,
            role=role,
            branch=branch,
            plan_constraints=plan_constraints,
        )
        if source == "mattergen" and branch_cache_fingerprints is not None and cache_ref in branch_cache_fingerprints:
            previous_fingerprint = branch_cache_fingerprints.get(cache_ref)
            if previous_fingerprint != cache_fingerprint:
                errors.append(
                    f"{bundle_id}.{role} was already materialized successfully; repair changed this branch. "
                    "Preserve successful branches exactly and repair only failed branches, or return "
                    "status='prediction_design_infeasible' so C/D can redesign the matched comparison."
                )
                continue
            cached_records = list((branch_cache_records or {}).get(cache_ref, []))
            if len(cached_records) >= count:
                matches = [dict(record, crystal_llm_materialization_cache_reused=True) for record in cached_records[:count]]
                source = "cached_mattergen"
        if source == "generator":
            if has_structure_dicts:
                if has_formula_probes:
                    errors.append(f"{bundle_id}.{role} must not include both formula_probes and structure_dicts for source=generator")
                    continue
                for index, raw_structure in enumerate(structure_dicts):
                    if not isinstance(raw_structure, Mapping):
                        errors.append(f"{bundle_id}.{role}.structure_dicts[{index}] must be an object")
                        continue
                    try:
                        structure = Structure.from_dict(dict(raw_structure))
                    except Exception as exc:
                        errors.append(f"{bundle_id}.{role}.structure_dicts[{index}] could not be parsed: {exc}")
                        continue
                    structure, preflight_notes, preflight_errors = prepare_generator_structure(structure, branch)
                    if preflight_errors:
                        errors.extend(
                            f"{bundle_id}.{role}.structure_dicts[{index}] failed generator preflight: {message}"
                            for message in preflight_errors
                        )
                        continue
                    validation = validate_structure(structure, max_sites=max_sites)
                    if not validation.ok:
                        errors.append(
                            f"{bundle_id}.{role}.structure_dicts[{index}] failed validation: {', '.join(validation.reasons)}"
                        )
                        continue
                    formula = reduced_formula(structure)
                    record = {
                        "material_id": f"generated::{bundle_id}::{role}::{index + 1}::{formula}",
                        "formula": formula,
                        "cif_path": None,
                        "structure_dict": structure.as_dict(),
                        "physics_bundle_id": bundle_id,
                        "physics_prediction_ids": prediction_ids,
                        "physics_role": role,
                        "physics_expected_relation": str(bundle.get("expected_relation") or ""),
                        "physics_selection_order": str(branch.get("selection_order") or "material_id"),
                        "crystal_llm_source": "generator",
                        "crystal_llm_generated_from_structure_dicts": [dict(raw_structure)],
                        "crystal_llm_generator_preflight": {
                            "notes": preflight_notes,
                            "summary": structure_preflight_summary(structure),
                        },
                    }
                    matches.append(record)
            else:
                raw_strategy = {"formula_probes": formula_probes or []}
                candidates = load_formula_probes(raw_strategy, max_sites=max_sites, base_seed=seed, known_formulas=known_formulas)
                if len(candidates) < count:
                    errors.append(
                        f"{bundle_id}.{role} requested {count} generator structures but only {len(candidates)} formula_probes materialized"
                    )
                    continue
                for candidate in candidates[:count]:
                    structure = candidate.structure
                    formula = reduced_formula(structure)
                    record = {
                        "material_id": f"generated::{bundle_id}::{role}::{formula}",
                        "formula": formula,
                        "cif_path": None,
                        "structure_dict": structure.as_dict(),
                        "physics_bundle_id": bundle_id,
                        "physics_prediction_ids": prediction_ids,
                        "physics_role": role,
                        "physics_expected_relation": str(bundle.get("expected_relation") or ""),
                        "physics_selection_order": str(branch.get("selection_order") or "material_id"),
                        "crystal_llm_source": "generator",
                        "crystal_llm_generated_from_formula_probes": [dict(probe) for probe in raw_strategy["formula_probes"] if isinstance(probe, Mapping)],
                    }
                    matches.append(record)
        elif source == "mattergen":
            if mattergen_config is None:
                errors.append(f"{bundle_id}.{role} requested source=mattergen but MatterGen configuration is unavailable")
                continue
            if not has_mattergen_requests:
                errors.append(f"{bundle_id}.{role} must include mattergen_requests when source=mattergen")
                continue
            if len(mattergen_requests) != 1 or not isinstance(mattergen_requests[0], Mapping):
                errors.append(f"{bundle_id}.{role}.mattergen_requests must contain exactly one object")
                continue
            excluded_formulas = set(str(item) for item in branch.get("exclude_formulas", []) if str(item).strip()) if isinstance(branch.get("exclude_formulas"), list) else set()
            if known_formulas:
                excluded_formulas |= set(known_formulas)
            branch_id = f"{bundle_id}_{role}"
            request = _normalise_mattergen_request_for_physics(
                mattergen_requests[0],
                branch_id=branch_id,
                mattergen_config=mattergen_config,
                excluded_formulas=excluded_formulas,
                max_sites=max_sites,
                count=count,
            )
            try:
                input_structures, adapter_records, report = _run_mattergen_request(
                    request,
                    branch_id=branch_id,
                    mattergen_config=mattergen_config,
                )
            except Exception as exc:
                errors.append(f"{bundle_id}.{role} MatterGen adapter failed: {type(exc).__name__}: {exc}")
                continue
            motif_rejects: list[dict[str, Any]] = []
            for index, raw_structure in enumerate(input_structures):
                try:
                    structure = Structure.from_dict(dict(raw_structure))
                except Exception as exc:
                    errors.append(f"{bundle_id}.{role}.mattergen[{index}] could not be parsed: {exc}")
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
                        f"{bundle_id}.{role}.mattergen[{index}] failed validation: {', '.join(validation.reasons)} "
                        f"(volume_per_atom={validation.volume_per_atom:.3f}, min_distance={validation.min_distance:.3f}, "
                        f"required volume_per_atom={min_volume:g}..{max_volume:g}, min_distance>=0.75, max_sites={max_sites})"
                    )
                    continue
                formula = reduced_formula(structure)
                if known_formulas and formula in known_formulas:
                    errors.append(f"{bundle_id}.{role}.mattergen[{index}] generated known/training formula {formula}")
                    continue
                motif_rejection = mattergen_structure_drift_rejection_reason(
                    structure,
                    bundle=bundle,
                    branch=branch,
                    role=role,
                    plan_constraints=plan_constraints,
                )
                if motif_rejection:
                    motif_rejects.append(
                        {
                            "index": index,
                            "formula": formula,
                            "reason": motif_rejection,
                        }
                    )
                    continue
                adapter_record = adapter_records[index] if index < len(adapter_records) else {}
                report_with_controller_checks = dict(report)
                if motif_rejects:
                    report_with_controller_checks["controller_motif_rejects"] = list(motif_rejects)
                record = {
                    **dict(adapter_record),
                    "material_id": f"mattergen::{bundle_id}::{role}::{index + 1:03d}::{formula}",
                    "formula": formula,
                    "cif_path": None,
                    "structure_dict": structure.as_dict(),
                    "physics_bundle_id": bundle_id,
                    "physics_prediction_ids": prediction_ids,
                    "physics_role": role,
                    "physics_expected_relation": str(bundle.get("expected_relation") or ""),
                    "physics_selection_order": "mattergen",
                    "crystal_llm_source": "mattergen",
                    "crystal_llm_generator_backend": "mattergen",
                    "crystal_llm_generated_from_mattergen_requests": [dict(request)],
                    "crystal_llm_mattergen_report": report_with_controller_checks,
                    "crystal_llm_mattergen_controller_motif_checks": {
                        "constraint_text": _short_text(
                            _branch_constraint_text(
                                bundle=bundle,
                                branch=branch,
                                role=role,
                                plan_constraints=plan_constraints,
                            ),
                            600,
                        ),
                        "rejected_before_this_record": list(motif_rejects),
                    },
                }
                matches.append(record)
                if len(matches) >= count:
                    break
            if len(matches) < count and motif_rejects:
                errors.append(
                    f"{bundle_id}.{role} rejected {len(motif_rejects)} MatterGen structures by controller motif drift checks: "
                    f"{json.dumps(motif_rejects[:5], ensure_ascii=False)}"
                )
            if len(matches) < count:
                report_summary = {
                    "adapter_status": report.get("status"),
                    "adapter_target_count": report.get("target_count"),
                    "adapter_accepted_count": report.get("accepted_count"),
                    "accepted_formulas": report.get("accepted_formulas"),
                    "reject_reasons": report.get("reject_reasons"),
                    "reject_examples": report.get("reject_examples"),
                    "request": compact_mattergen_request_for_feedback(request),
                }
                errors.append(
                    f"{bundle_id}.{role} requested {count} records but only {len(matches)} materialized from the mattergen; "
                    f"mattergen_report={json.dumps(report_summary, ensure_ascii=False)[:1200]}; "
                    "repair must keep source='mattergen', avoid repeating the same underfilled request, and lower/remove "
                    "filters.min_sites or increase target_count/batch_size/num_batches when reject_reasons include "
                    "too_few_sites, not_target_reduced_formula, or no_accepted_structures. Sampling-only increases on "
                    "a failed branch are execution repairs, not physical confounds."
                )
        elif source == "cached_mattergen":
            pass
        elif material_ids:
            matches = [dict(record) for record in pool_records if str(record.get("material_id")) in set(material_ids)]
        elif isinstance(query, Mapping):
            matches = select_matches(pool_records, query, count=count, seed=seed)
        else:
            matches = []
        if len(matches) < count:
            errors.append(
                f"{bundle_id}.{role} requested {count} records but only {len(matches)} materialized from the {source}"
            )
            continue
        if isinstance(query, Mapping) and isinstance(query.get("preferred_order"), list) and query.get("preferred_order"):
            order = "preferred_order"
        else:
            order = str(branch.get("selection_order") or "material_id")
        if order == "random":
            # select_matches already shuffled; no extra work
            chosen = matches[:count]
        else:
            chosen = matches[:count]
        if (
            branch_cache_records is not None
            and branch_cache_fingerprints is not None
            and (source == "mattergen" or source == "cached_mattergen")
        ):
            branch_cache_records[cache_ref] = [dict(record) for record in chosen]
            branch_cache_fingerprints[cache_ref] = cache_fingerprint
        for record in chosen:
            material_id = str(record.get("material_id") or "")
            if not material_id:
                errors.append(f"{bundle_id}.{role} produced a record without material_id")
                continue
            if material_id in seen_material_ids:
                errors.append(f"{bundle_id}.{role} duplicates material_id {material_id}")
                continue
            seen_material_ids.add(material_id)
            item = dict(record)
            item["physics_bundle_id"] = bundle_id
            item["physics_prediction_ids"] = prediction_ids
            item["physics_role"] = role
            item["physics_expected_relation"] = str(bundle.get("expected_relation") or "")
            item["physics_selection_order"] = order
            item["crystal_llm_source"] = item.get("crystal_llm_source") or ("mp_pool" if source != "generator" else "generator")
            selected.append(item)
    return selected, errors


def materialize_plan(
    pool_records: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    *,
    seed: int,
    max_sites: int = 80,
    known_formulas: set[str] | None = None,
    mattergen_config: Mapping[str, Any] | None = None,
    branch_cache_records: dict[str, list[dict[str, Any]]] | None = None,
    branch_cache_fingerprints: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_material_ids: set[str] = set()
    bundles = plan.get("accepted_bundles", [])
    if not isinstance(bundles, list):
        return [], ["accepted_bundles must be a list"]
    raw_constraints = plan.get("materialization_constraints")
    plan_constraints = raw_constraints if isinstance(raw_constraints, Mapping) else None
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            errors.append("bundle must be an object")
            continue
        chosen, bundle_errors = select_bundle_records(
            pool_records,
            bundle=bundle,
            seed=seed,
            seen_material_ids=seen_material_ids,
            max_sites=max_sites,
            known_formulas=known_formulas,
            mattergen_config=mattergen_config,
            plan_constraints=plan_constraints,
            branch_cache_records=branch_cache_records,
            branch_cache_fingerprints=branch_cache_fingerprints,
        )
        selected.extend(chosen)
        errors.extend(bundle_errors)
    return selected, errors


def validate_mattergen_native_request_contract(request: Mapping[str, Any], *, path: str) -> list[str]:
    """Reject MatterGen request fields the controller does not implement."""

    errors: list[str] = []
    unknown_request_keys = sorted(str(key) for key in request if str(key) not in SUPPORTED_MATTERGEN_REQUEST_KEYS)
    if unknown_request_keys:
        errors.append(
            f"{path} contains unsupported MatterGen request keys {unknown_request_keys}. "
            "MatterGen requests are not an arbitrary predicate DSL; use only backend, request_id, checkpoint, model_path, "
            "target_count, batch_size, num_batches, diffusion_guidance_factor, properties_to_condition_on, filters, "
            "target_reduced_formula, and require_target_reduced_formula. If a faithful test needs unsupported gates, "
            "return status='prediction_design_infeasible'."
        )
    properties = request.get("properties_to_condition_on")
    if isinstance(properties, Mapping):
        unknown_properties = sorted(str(key) for key in properties if str(key) not in SUPPORTED_MATTERGEN_PROPERTY_KEYS)
        if unknown_properties:
            errors.append(
                f"{path}.properties_to_condition_on contains unsupported MatterGen condition keys {unknown_properties}. "
                "The active controller supports chemical_system and energy_above_hull only."
            )
    filters = request.get("filters")
    if isinstance(filters, Mapping):
        unknown_filters = sorted(str(key) for key in filters if str(key) not in SUPPORTED_MATTERGEN_FILTER_KEYS)
        if unknown_filters:
            gate_keys = sorted(str(key) for key in filters if str(key) in UNSUPPORTED_MATTERGEN_GATE_HINTS)
            gate_note = f" Unsupported gate-like keys detected: {gate_keys}." if gate_keys else ""
            errors.append(
                f"{path}.filters contains unsupported MatterGen filter keys {unknown_filters}.{gate_note} "
                "Supported filters are chemical_system, require_chemical_system_exact, min_sites, max_sites, "
                "min_volume_per_atom, max_volume_per_atom, deduplicate_reduced_formula, target_reduced_formula, "
                "require_target_reduced_formula, and exclude_reduced_formulas. Move non-executable drift concerns to "
                "drift_rejection_criteria or return status='prediction_design_infeasible'."
            )
    return errors


def validate_mattergen_native_execution_sources(plan: Mapping[str, Any]) -> list[str]:
    """Enforce the A-F MatterGen-native contract after generic schema validation."""

    errors: list[str] = []
    bundles = plan.get("accepted_bundles")
    if not isinstance(bundles, list):
        return errors
    for bundle_index, bundle in enumerate(bundles):
        if not isinstance(bundle, Mapping):
            continue
        bundle_source = str(bundle.get("source") or "mp_pool")
        for role in ("primary", "control"):
            branch = bundle.get(role)
            if not isinstance(branch, Mapping):
                continue
            path = f"accepted_bundles[{bundle_index}].{role}"
            formula_probes = branch.get("formula_probes")
            structure_dicts = branch.get("structure_dicts")
            material_ids = branch.get("material_ids")
            mattergen_requests = branch.get("mattergen_requests")
            has_formula_probes = isinstance(formula_probes, list) and bool(formula_probes)
            has_structure_dicts = isinstance(structure_dicts, list) and bool(structure_dicts)
            has_mattergen_requests = isinstance(mattergen_requests, list) and bool(mattergen_requests)
            source = str(branch.get("source") or bundle_source)
            if (not branch.get("source")) and (has_formula_probes or has_structure_dicts):
                source = "generator"
            if (not branch.get("source")) and has_mattergen_requests:
                source = "mattergen"
            if source != "mattergen":
                errors.append(
                    f"{path} must use source='mattergen' in MatterGen-native mode; got source={source!r}. "
                    "Repair with revised MatterGen filters/sampling or return status='prediction_design_infeasible'; "
                    "do not fall back to generator, mp_pool, formula_probes, structure_dicts, or material_ids."
                )
            if has_formula_probes or has_structure_dicts or (isinstance(material_ids, list) and material_ids):
                errors.append(
                    f"{path} contains non-MatterGen materialization payloads in MatterGen-native mode. "
                    "Use exactly one mattergen_requests object or mark the prediction design infeasible."
                )
            if isinstance(mattergen_requests, list):
                for request_index, request in enumerate(mattergen_requests):
                    if isinstance(request, Mapping):
                        errors.extend(
                            validate_mattergen_native_request_contract(
                                request,
                                path=f"{path}.mattergen_requests[{request_index}]",
                            )
                        )
    return errors


def compact_execution_plan_for_feedback(plan: Mapping[str, Any]) -> dict[str, Any]:
    bundles: list[dict[str, Any]] = []
    raw_bundles = plan.get("accepted_bundles")
    if not isinstance(raw_bundles, list):
        raw_bundles = plan.get("bundles")
    if isinstance(raw_bundles, list):
        for bundle in raw_bundles[:4]:
            bundles.append(_compact_execution_bundle_for_context(bundle))
    return {
        "status": plan.get("status"),
        "accepted_bundles": bundles,
        "consensus_summary": _short_text(plan.get("consensus_summary"), 360),
    }


def summarize_materialization_feedback(
    *,
    plan: Mapping[str, Any],
    materialization_errors: Sequence[str],
    proposal: Mapping[str, Any],
    critique: Mapping[str, Any],
) -> dict[str, Any]:
    failed_branches: list[dict[str, Any]] = []
    branch_pattern = re.compile(
        r"^(?P<bundle>[^.]+)\.(?P<role>primary|control) requested (?P<requested>\d+) records "
        r"but only (?P<actual>\d+) materialized from the (?P<source>[A-Za-z0-9_:-]+)"
    )
    generator_formula_pattern = re.compile(
        r"^(?P<bundle>[^.]+)\.(?P<role>primary|control) requested (?P<requested>\d+) generator structures "
        r"but only (?P<actual>\d+) formula_probes materialized"
    )
    raw_bundles = plan.get("accepted_bundles")
    if not isinstance(raw_bundles, list):
        raw_bundles = []
    bundle_by_id = {str(bundle.get("id") or ""): bundle for bundle in raw_bundles if isinstance(bundle, Mapping)}
    for message in materialization_errors[:8]:
        match = branch_pattern.search(str(message))
        generator_match = generator_formula_pattern.search(str(message))
        if not match:
            if not generator_match:
                failed_branches.append({"error": _short_text(message, 360)})
                continue
            bundle_id = generator_match.group("bundle")
            role = generator_match.group("role")
            source = "generator_formula_probes"
            requested = int(generator_match.group("requested"))
            materialized = int(generator_match.group("actual"))
        else:
            bundle_id = match.group("bundle")
            role = match.group("role")
            source = match.group("source")
            requested = int(match.group("requested"))
            materialized = int(match.group("actual"))
        bundle = bundle_by_id.get(bundle_id)
        branch = bundle.get(role) if isinstance(bundle, Mapping) else None
        branch_summary: dict[str, Any] = {
            "bundle_id": bundle_id,
            "role": role,
            "requested": requested,
            "materialized": materialized,
            "source": source,
            "error": _short_text(message, 360),
        }
        if isinstance(branch, Mapping):
            branch_summary["branch"] = {
                "source": branch.get("source") or (bundle.get("source") if isinstance(bundle, Mapping) else None) or "mp_pool",
                "count": branch.get("count"),
                "query": _compact_query_for_context(branch.get("query")),
            }
            mattergen_requests = branch.get("mattergen_requests")
            if isinstance(mattergen_requests, list) and mattergen_requests:
                branch_summary["failed_mattergen_request"] = compact_mattergen_request_for_feedback(mattergen_requests[0])
            formula_probes = branch.get("formula_probes")
            if isinstance(formula_probes, list) and formula_probes:
                branch_summary["failed_formula_probes"] = [
                    compact_formula_probe_for_prompt(item) for item in formula_probes[:PROMPT_MATERIALIZATION_ITEM_LIMIT]
                ]
        if branch_summary["source"] == "mp_pool":
            branch_summary["required_action"] = (
                "Do not repeat the same MP-pool query. Replace this branch with source='generator' "
                "or provide a different MP-pool query that the controller can materialize and that preserves the prediction."
            )
        elif branch_summary["source"] == "generator_formula_probes":
            branch_summary["required_action"] = (
                "Do not repeat the same generator formula_probes: the controller already materialized zero structures from them. "
                "Use parseable generator structure_dicts or a different executable MP-pool branch that preserves the prediction. "
                "If neither exists, return status='prediction_design_infeasible' with prediction_design_feedback."
            )
        elif branch_summary["source"] == "mattergen":
            branch_summary["required_action"] = (
                "Keep source='mattergen'. Do not switch to generator or MP-pool in MatterGen-native mode. "
                "Do not repeat the same MatterGen filters. If reject_reasons mention too_few_sites, lower or remove "
                "filters.min_sites and/or increase target_count/num_batches; if volume filters dominate, widen the VPA window. "
                "Return prediction_design_infeasible only if no faithful MatterGen-conditioned test remains."
            )
        failed_branches.append(branch_summary)
    return {
        "materialization_errors": [_short_text(message, 360) for message in materialization_errors[:8]],
        "failed_branches": failed_branches,
        "required_repair": (
            "Repair only the failed branches. Preserve accepted prediction ids, expected_relation, and counts. "
            "Do not replay the same underfilled MP-pool query, failed generator formula_probes, or underfilled MatterGen filters."
        ),
        "previous_plan_summary": compact_execution_plan_for_feedback(plan),
        "previous_proposal_summary": compact_execution_plan_for_feedback(proposal),
        "latest_critique_summary": {
            "agent": critique.get("agent"),
            "agree": critique.get("agree"),
            "required_revisions": compact_repair_feedback(critique.get("required_revisions", []), max_list_items=4),
            "summary": _short_text(critique.get("summary") or critique.get("audit_summary") or critique.get("execution_audit"), 500),
        },
    }


def is_prediction_design_infeasible(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return str(payload.get("status") or "").strip().lower() == PREDICTION_DESIGN_INFEASIBLE_STATUS


def prediction_feedback_from_execution(
    *,
    execution_debate: Mapping[str, Any],
    prediction_debate: Mapping[str, Any],
    feedback_round: int,
) -> dict[str, Any]:
    feedback = execution_debate.get("prediction_design_feedback")
    if not isinstance(feedback, Mapping):
        feedback = {}
    return {
        "status": PREDICTION_DESIGN_INFEASIBLE_STATUS,
        "feedback_round": feedback_round,
        "prediction_ids": feedback.get("prediction_ids") or _prediction_ids_from_payload(prediction_debate),
        "blocking_issue": _short_text(
            feedback.get("blocking_issue")
            or feedback.get("reason")
            or execution_debate.get("consensus_summary")
            or "E/F agreed that the accepted prediction cannot be faithfully materialized.",
            700,
        ),
        "why_execution_cannot_fix_it": compact_repair_feedback(
            feedback.get("why_execution_cannot_fix_it")
            or feedback.get("why_execution_cannot_fix")
            or execution_debate.get("rejected_bundles")
            or [],
            max_list_items=4,
        ),
        "required_cd_reconsideration": compact_repair_feedback(
            feedback.get("required_cd_reconsideration")
            or [
                "Revise the infeasible primary/control design.",
                "Keep planned_material_count >= current_inputs.target_count.",
                "Only accept predictions whose branches have a faithful MP-pool or generator materialization path.",
            ],
            max_list_items=4,
        ),
        "previous_prediction_summary": compact_dialogue_payload_for_prompt(prediction_debate, max_list_items=4),
        "execution_consensus_summary": _short_text(execution_debate.get("consensus_summary"), 700),
    }


def _prediction_ids_from_payload(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("accepted_predictions") or payload.get("predictions") or []
    if not isinstance(raw, list):
        return []
    ids: list[str] = []
    for item in raw:
        if isinstance(item, Mapping) and str(item.get("id") or "").strip():
            ids.append(str(item["id"]).strip())
    return ids


def attach_structure_metadata(record: Mapping[str, Any], bundle_index: int, item_index: int) -> dict[str, Any]:
    payload = dict(record)
    payload["crystal_llm_material_physics"] = {
        "bundle_id": record.get("physics_bundle_id"),
        "prediction_ids": record.get("physics_prediction_ids", []),
        "role": record.get("physics_role"),
        "expected_relation": record.get("physics_expected_relation"),
        "selection_order": record.get("physics_selection_order"),
        "bundle_index": bundle_index,
        "item_index": item_index,
    }
    payload["crystal_llm_source"] = "mp_candidate_pool"
    payload["crystal_llm_material_id"] = record.get("material_id")
    payload["crystal_llm_formula"] = record.get("formula")
    payload["crystal_llm_pool_metadata"] = {
        "crystal_system": record.get("crystal_system"),
        "spacegroup_number": record.get("spacegroup_number"),
        "band_gap": record.get("band_gap"),
        "density": record.get("density"),
        "formation_energy_per_atom": record.get("formation_energy_per_atom"),
        "volume": record.get("volume"),
        "nsites": record.get("nsites"),
        "nelements": record.get("nelements"),
    }
    return payload


def round_output_dir(work_dir: Path, round_number: int) -> Path:
    return work_dir / round_label(round_number)


def default_evaluator_input_path(round_dir: Path) -> Path:
    return round_dir / "input.json"


def default_results_path(round_dir: Path) -> Path:
    return round_dir / "results.json"


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    round_dir: Path,
    step_name: str,
    env: Mapping[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import os

    process_env = os.environ.copy()
    process_env["PYTHONPATH"] = f"{cwd / 'src'}:{process_env.get('PYTHONPATH', '')}"
    if env:
        process_env.update(dict(env))

    log_event(round_dir, f"START {step_name}: {command_text(cmd)}")
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# started_at_utc={utc_now()}\n")
        handle.write(f"# cwd={cwd}\n")
        handle.write(f"# command={command_text(cmd)}\n")
        handle.flush()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return_code = proc.wait()
        handle.write(f"# finished_at_utc={utc_now()}\n")
        handle.write(f"# return_code={return_code}\n")
    elapsed = time.monotonic() - started
    if return_code != 0:
        log_event(round_dir, f"FAIL {step_name}: return_code={return_code} elapsed_sec={elapsed:.1f}")
        raise subprocess.CalledProcessError(return_code, list(cmd))
    log_event(round_dir, f"DONE {step_name}: elapsed_sec={elapsed:.1f}")


def mechanism_context(state: Mapping[str, Any], pool_summary: Mapping[str, Any], repair_feedback: Mapping[str, Any] | None) -> str:
    return prompt_json_dumps(
        build_context_payload(state=state, pool_summary=pool_summary, stage="mechanism", repair_feedback=repair_feedback),
    )


def prediction_context(
    state: Mapping[str, Any],
    pool_summary: Mapping[str, Any],
    repair_feedback: Mapping[str, Any] | None,
    current_inputs: Mapping[str, Any] | None = None,
) -> str:
    return prompt_json_dumps(
        build_context_payload(
            state=state,
            pool_summary=pool_summary,
            stage="prediction",
            repair_feedback=repair_feedback,
            current_inputs=current_inputs,
        ),
    )


def execution_context(
    state: Mapping[str, Any],
    pool_summary: Mapping[str, Any],
    repair_feedback: Mapping[str, Any] | None,
    current_inputs: Mapping[str, Any] | None = None,
) -> str:
    return prompt_json_dumps(
        build_context_payload(
            state=state,
            pool_summary=pool_summary,
            stage="execution",
            repair_feedback=repair_feedback,
            current_inputs=current_inputs,
        ),
    )


def postmortem_context(
    state: Mapping[str, Any],
    pool_summary: Mapping[str, Any],
    current_inputs: Mapping[str, Any] | None = None,
) -> str:
    return prompt_json_dumps(
        build_context_payload(
            state=state,
            pool_summary=pool_summary,
            stage="postmortem",
            repair_feedback=None,
            current_inputs=current_inputs,
        ),
    )


def summarize_round(
    *,
    round_number: int,
    mechanism_payload: Mapping[str, Any],
    prediction_payload: Mapping[str, Any],
    execution_payload: Mapping[str, Any],
    analysis_summary: Mapping[str, Any],
    selected_records: Sequence[Mapping[str, Any]],
    bundle_results: list[dict[str, Any]],
) -> dict[str, Any]:
    primary_sun_count = sum(int(item.get("primary_sun_count") or 0) for item in bundle_results)
    control_sun_count = sum(int(item.get("control_sun_count") or 0) for item in bundle_results)
    mechanism_validated_sun_count = sum(
        int(item.get("primary_sun_count") or 0) for item in bundle_results if item.get("supported") is True
    )
    return {
        "round": round_number,
        "created_at_utc": utc_now(),
        "accepted_mechanisms": mechanism_payload.get("accepted_mechanisms", []),
        "accepted_predictions": prediction_payload.get("accepted_predictions", []),
        "accepted_bundles": execution_payload.get("accepted_bundles", []),
        "selected_material_ids": [str(record.get("material_id")) for record in selected_records],
        "bundle_results": bundle_results,
        "evaluation_summary": {
            "count": analysis_summary.get("count"),
            "mean_e_hull": analysis_summary.get("mean_e_hull"),
            "min_e_hull": analysis_summary.get("min_e_hull"),
            "max_e_hull": analysis_summary.get("max_e_hull"),
            "stable_count": analysis_summary.get("sun_strict_e_hull_lt_0"),
            "primary_sun_count": primary_sun_count,
            "control_sun_count": control_sun_count,
            "mechanism_validated_sun_count": mechanism_validated_sun_count,
            "support_rate": _support_rate(bundle_results),
            "supported_bundle_count": sum(1 for item in bundle_results if item.get("supported")),
            "bundle_count": len(bundle_results),
        },
    }


def summarize_skipped_round(
    *,
    round_number: int,
    status: str,
    artifact: Mapping[str, Any],
    mechanism_payload: Mapping[str, Any] | None = None,
    prediction_payload: Mapping[str, Any] | None = None,
    execution_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    skip_summary: dict[str, Any] = {
        "stage_status": artifact.get("status"),
        "reason": artifact.get("reason"),
        "consensus_summary": artifact.get("consensus_summary"),
    }
    if isinstance(artifact.get("validation_errors"), list):
        skip_summary["validation_errors"] = artifact.get("validation_errors")
    if isinstance(artifact.get("dialogue"), list):
        skip_summary["dialogue_turns"] = len(artifact["dialogue"])
    return {
        "round": round_number,
        "status": status,
        "created_at_utc": utc_now(),
        "accepted_mechanisms": (mechanism_payload or artifact).get("accepted_mechanisms", []),
        "accepted_predictions": (prediction_payload or artifact).get("accepted_predictions", []),
        "accepted_bundles": (execution_payload or artifact).get("accepted_bundles", []),
        "selected_material_ids": [],
        "bundle_results": [],
        "skip_summary": {key: value for key, value in skip_summary.items() if value not in (None, [], "")},
        "evaluation_summary": {
            "count": 0,
            "mean_e_hull": None,
            "min_e_hull": None,
            "max_e_hull": None,
            "stable_count": None,
            "primary_sun_count": 0,
            "control_sun_count": 0,
            "mechanism_validated_sun_count": 0,
            "support_rate": None,
            "supported_bundle_count": 0,
            "bundle_count": 0,
        },
    }


def record_skipped_round(
    state: dict[str, Any],
    summary: Mapping[str, Any],
) -> None:
    state.setdefault("history", [])
    if not isinstance(state["history"], list):
        state["history"] = []
    state["history"].append(dict(summary))
    state["updated_at_utc"] = utc_now()
    state["status"] = "running"
    state["latest_support_rate"] = None


def ensure_round_completed(result: Mapping[str, Any]) -> None:
    if result.get("status") != "complete" and result.get("status") not in NONFATAL_ROUND_STATUSES:
        round_number = result.get("round", "unknown")
        raise RuntimeError(f"round {round_number} did not reach evaluator completion: {result.get('status')}")


def finalize_controller_state(state: dict[str, Any]) -> None:
    """Mark a normally exhausted controller run as terminal without masking hard errors."""

    if state.get("status") == "running":
        state["status"] = "completed"
    state["updated_at_utc"] = utc_now()


def _support_rate(bundle_results: Sequence[Mapping[str, Any]]) -> float | None:
    if not bundle_results:
        return None
    supported = sum(1 for item in bundle_results if item.get("supported"))
    return supported / len(bundle_results)


def bundle_results_from_analysis(
    selected_records: Sequence[Mapping[str, Any]],
    analysis_rows: Sequence[Mapping[str, Any]],
    bundles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_index = {int(row["index"]): row for row in analysis_rows if "index" in row}
    bundle_results: list[dict[str, Any]] = []
    index = 1
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            continue
        bundle_id = str(bundle.get("id") or f"bundle_{index:03d}")
        prediction_ids = [str(item) for item in bundle.get("prediction_ids", []) if str(item).strip()]
        bundle_item_indices: dict[str, list[int]] = {"primary": [], "control": []}
        for role in ("primary", "control"):
            count = int(bundle.get(role, {}).get("count") or 0) if isinstance(bundle.get(role), Mapping) else 0
            bundle_item_indices[role] = list(range(index, index + count))
            index += count
        primary_values = [float(rows_by_index[i]["e_hull"]) for i in bundle_item_indices["primary"] if i in rows_by_index]
        control_values = [float(rows_by_index[i]["e_hull"]) for i in bundle_item_indices["control"] if i in rows_by_index]
        primary_sun_count = sum(1 for value in primary_values if value < 0)
        control_sun_count = sum(1 for value in control_values if value < 0)
        if primary_values and control_values:
            primary_mean = sum(primary_values) / len(primary_values)
            control_mean = sum(control_values) / len(control_values)
            primary_min = min(primary_values)
            control_min = min(control_values)
            relation = str(bundle.get("expected_relation") or "")
            if relation == "primary_lower_e_hull_than_control":
                supported = primary_mean < control_mean
            elif relation == "primary_higher_e_hull_than_control":
                supported = primary_mean > control_mean
            else:
                supported = None
            delta = primary_mean - control_mean
        else:
            primary_mean = None
            control_mean = None
            primary_min = None
            control_min = None
            delta = None
            supported = None
        bundle_results.append(
            {
                "bundle_id": bundle_id,
                "prediction_ids": prediction_ids,
                "primary_indices": bundle_item_indices["primary"],
                "control_indices": bundle_item_indices["control"],
                "primary_mean_e_hull": primary_mean,
                "control_mean_e_hull": control_mean,
                "primary_min_e_hull": primary_min,
                "control_min_e_hull": control_min,
                "primary_sun_count": primary_sun_count,
                "control_sun_count": control_sun_count,
                "mechanism_validated_sun_count": primary_sun_count if supported is True else 0,
                "delta": delta,
                "supported": supported,
                "expected_relation": bundle.get("expected_relation"),
            }
        )
    return bundle_results


def write_round_report(round_dir: Path, summary: Mapping[str, Any]) -> None:
    analysis_dir = round_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_json(analysis_dir / "physics_summary.json", summary)
    lines = [
        "# Materials Physics Round Summary",
        "",
        f"- round: `{summary.get('round')}`",
        f"- mean e_hull: `{summary.get('evaluation_summary', {}).get('mean_e_hull')}`",
        f"- min e_hull: `{summary.get('evaluation_summary', {}).get('min_e_hull')}`",
        f"- max e_hull: `{summary.get('evaluation_summary', {}).get('max_e_hull')}`",
        f"- support rate: `{summary.get('evaluation_summary', {}).get('support_rate')}`",
        f"- supported bundles: `{summary.get('evaluation_summary', {}).get('supported_bundle_count')}` / `{summary.get('evaluation_summary', {}).get('bundle_count')}`",
        f"- primary SUN count: `{summary.get('evaluation_summary', {}).get('primary_sun_count')}`",
        f"- control SUN count: `{summary.get('evaluation_summary', {}).get('control_sun_count')}`",
        f"- mechanism-validated SUN count: `{summary.get('evaluation_summary', {}).get('mechanism_validated_sun_count')}`",
        "",
    ]
    postmortem = summary.get("principle_postmortem")
    if isinstance(postmortem, Mapping):
        lines.extend(
            [
                "## Principle Postmortem",
                "",
                f"- program: `{postmortem.get('program_id')}`",
                f"- status: `{postmortem.get('status')}`",
                f"- hypothesis status: `{postmortem.get('hypothesis_status')}`",
                f"- update action: `{postmortem.get('principle_update_action')}`",
                f"- current principle: {postmortem.get('current_principle_statement')}",
                f"- next test focus: {postmortem.get('next_test_focus')}",
                "",
            ]
        )
    lines.extend(["## Material IDs", ""])
    for material_id in summary.get("selected_material_ids", []):
        lines.append(f"- `{material_id}`")
    (analysis_dir / "physics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fallback_principle_postmortem(
    *,
    state: Mapping[str, Any],
    round_summary: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    program = state.get("current_principle_program") if isinstance(state.get("current_principle_program"), Mapping) else {}
    evaluation = round_summary.get("evaluation_summary") if isinstance(round_summary.get("evaluation_summary"), Mapping) else {}
    support_rate = evaluation.get("support_rate")
    if support_rate == 1:
        hypothesis_status = "supported"
    elif support_rate == 0:
        hypothesis_status = "contradicted"
    else:
        hypothesis_status = "ambiguous"
    return {
        "status": "continue",
        "program_id": program.get("program_id") or "principle_program_unknown",
        "round": round_summary.get("round"),
        "hypothesis_status": hypothesis_status,
        "principle_update_action": "refine" if hypothesis_status == "supported" else "narrow",
        "current_principle_statement": program.get("current_principle_statement") or "Active principle not specified.",
        "micro_mechanism": program.get("micro_mechanism") or "Microscopic mechanism requires further refinement.",
        "e_hull_evidence": {
            "support_rate": support_rate,
            "mean_e_hull": evaluation.get("mean_e_hull"),
            "min_e_hull": evaluation.get("min_e_hull"),
        },
        "sun_accounting": {
            "primary_sun_count": evaluation.get("primary_sun_count"),
            "control_sun_count": evaluation.get("control_sun_count"),
            "mechanism_validated_sun_count": evaluation.get("mechanism_validated_sun_count"),
        },
        "causal_interpretation": f"Fallback postmortem because LLM postmortem failed: {reason}",
        "failure_boundaries": [],
        "unresolved_contradictions": ["A/B postmortem was not completed; require explicit causal review next round."],
        "next_test_focus": "Repeat A/B postmortem review and refine the same active principle.",
    }


def run_principle_postmortem(
    *,
    args: argparse.Namespace,
    root: Path,
    round_dir: Path,
    llm_log_dir: Path,
    state: Mapping[str, Any],
    pool_summary: Mapping[str, Any],
    round_summary: Mapping[str, Any],
) -> dict[str, Any]:
    context_json = postmortem_context(
        state,
        pool_summary,
        current_inputs={"latest_round_summary": compact_round_summary_for_postmortem(round_summary)},
    )
    client = _client("MECHANISM_B", args, root, log_dir=llm_log_dir / "ab")
    try:
        postmortem = call_json(
            client,
            system=PRINCIPLE_POSTMORTEM_SYSTEM,
            user=prompt_principle_postmortem(context_json, round_summary),
            role="principle_postmortem",
            metadata={"role": "principle_postmortem", "round": round_summary.get("round")},
            json_repair_attempts=args.json_repair_attempts,
        )
    except Exception as exc:
        log_event(round_dir, f"principle_postmortem_fallback: {type(exc).__name__}: {exc}")
        postmortem = fallback_principle_postmortem(state=state, round_summary=round_summary, reason=str(exc))
    write_json(round_dir / "principle_postmortem.json", postmortem)
    return postmortem


ELEMENT_SYMBOLS_FOR_TOPIC_MATCH = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
}

PRINCIPLE_TOPIC_STOPWORDS = {
    "and",
    "are",
    "because",
    "between",
    "branch",
    "branches",
    "compact",
    "compactness",
    "compared",
    "control",
    "controls",
    "derived",
    "discovery",
    "does",
    "finite",
    "from",
    "higher",
    "hull",
    "into",
    "lower",
    "matched",
    "mean",
    "only",
    "primary",
    "relative",
    "result",
    "scope",
    "show",
    "shows",
    "stable",
    "stabilized",
    "stability",
    "tested",
    "than",
    "that",
    "the",
    "this",
    "under",
    "versus",
    "when",
    "while",
    "with",
    "within",
}


def _principle_identity_text(entry: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("topic_key", "principle_statement", "current_principle_statement", "micro_mechanism"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    for key in ("boundaries", "failure_boundaries", "residual_risks", "unresolved_contradictions"):
        value = entry.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value[:4] if str(item).strip())
    return " ".join(parts)


def _extract_topic_elements(text: str) -> set[str]:
    elements: set[str] = set()
    for token in re.findall(r"\b[A-Z][a-z]?\b", text):
        if token in ELEMENT_SYMBOLS_FOR_TOPIC_MATCH:
            elements.add(token)
    return elements


def _principle_topic_terms(text: str) -> set[str]:
    lower_elements = {item.lower() for item in ELEMENT_SYMBOLS_FOR_TOPIC_MATCH}
    terms: set[str] = set()
    normalized = text.lower().replace("oxyfluorides", "oxyfluoride").replace("oxides", "oxide")
    normalized = normalized.replace("phosphates", "phosphate").replace("silicates", "silicate")
    for token in re.findall(r"[a-z][a-z0-9]+", normalized):
        if len(token) < 3 or token in lower_elements or token in PRINCIPLE_TOPIC_STOPWORDS:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        terms.add(token)
    return terms


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _explicit_principle_update_id(*items: Any) -> str:
    keys = (
        "updates_principle_id",
        "update_principle_id",
        "parent_principle_id",
        "existing_principle_id",
        "principle_book_update_target",
    )
    for item in items:
        if not isinstance(item, Mapping):
            continue
        for key in keys:
            value = str(item.get(key) or "").strip()
            if value:
                return value
    return ""


def find_principle_book_update_index(
    book: Any,
    incoming_entry: Mapping[str, Any],
    *,
    postmortem: Mapping[str, Any] | None = None,
    program: Mapping[str, Any] | None = None,
) -> int | None:
    if not isinstance(book, list):
        return None
    raw_experience = postmortem.get("experience_book_entry") if isinstance(postmortem, Mapping) else None
    experience = raw_experience if isinstance(raw_experience, Mapping) else {}
    explicit_id = _explicit_principle_update_id(incoming_entry, experience, postmortem, program)
    if explicit_id:
        for index, existing in enumerate(book):
            if isinstance(existing, Mapping) and str(existing.get("program_id") or "").strip() == explicit_id:
                return index
    identity = str(incoming_entry.get("principle_identity") or experience.get("principle_identity") or "").strip().lower()
    requested_new = identity == "new_principle"
    incoming_text = _principle_identity_text(incoming_entry)
    incoming_elements = _extract_topic_elements(incoming_text)
    incoming_terms = _principle_topic_terms(incoming_text)
    if not incoming_text.strip() or not incoming_elements:
        return None
    best_index: int | None = None
    best_score = 0.0
    for index, existing in enumerate(book):
        if not isinstance(existing, Mapping):
            continue
        existing_text = _principle_identity_text(existing)
        existing_elements = _extract_topic_elements(existing_text)
        if not existing_elements:
            continue
        element_score = _jaccard(incoming_elements, existing_elements)
        if element_score < 0.75:
            continue
        term_score = _jaccard(incoming_terms, _principle_topic_terms(existing_text))
        score = 0.65 * element_score + 0.35 * term_score
        if score > best_score:
            best_score = score
            best_index = index
    if best_index is None:
        return None
    if requested_new and best_score < 0.82:
        return None
    return best_index if best_score >= 0.72 else None


def _short_text_list(value: Any, *, limit: int = 20, max_chars: int = 240) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _short_text(item, max_chars)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _merge_text_lists(existing: Any, incoming: Any, *, limit: int = 24, max_chars: int = 240) -> list[str]:
    merged: list[str] = []
    for text in _short_text_list(existing, limit=limit, max_chars=max_chars) + _short_text_list(
        incoming, limit=limit, max_chars=max_chars
    ):
        if text and text not in merged:
            merged.append(text)
    return merged[-limit:]


def _merge_evidence_rounds(existing: Any, incoming: Any) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for value in (existing if isinstance(existing, list) else []) + (incoming if isinstance(incoming, list) else []):
        key = json.dumps(value, sort_keys=True, default=str) if isinstance(value, Mapping) else str(value)
        if key in seen:
            continue
        seen.add(key)
        merged.append(value)
    return merged[-80:]


def merge_principle_book_entry(existing: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    existing_id = str(existing.get("program_id") or "").strip()
    incoming_id = str(incoming.get("program_id") or "").strip()
    merged["program_id"] = existing_id or incoming_id
    if existing.get("closed_at_round") is not None:
        merged.setdefault("first_closed_at_round", existing.get("closed_at_round"))
    else:
        merged["closed_at_round"] = incoming.get("closed_at_round")
    if incoming.get("closed_at_round") is not None:
        merged["last_updated_round"] = incoming.get("closed_at_round")
    status_history = merged.get("status_history")
    if not isinstance(status_history, list):
        status_history = []
    status_history.append(
        {
            "round": incoming.get("closed_at_round"),
            "status": incoming.get("status"),
            "program_id": incoming_id,
        }
    )
    merged["status_history"] = [item for item in status_history[-20:] if isinstance(item, Mapping)]
    incoming_status = str(incoming.get("status") or "").strip()
    existing_status = str(existing.get("status") or "").strip()
    if incoming_status == "validated_principle" or existing_status != "validated_principle":
        merged["status"] = incoming_status or existing_status
    else:
        merged["status"] = existing_status
        merged["latest_update_status"] = incoming_status
    for key in ("principle_statement", "micro_mechanism"):
        value = str(incoming.get(key) or "").strip()
        if value:
            merged[key] = value
    merged["reasoning_chain"] = _merge_text_lists(
        existing.get("reasoning_chain"), incoming.get("reasoning_chain"), limit=24, max_chars=320
    )
    merged["evidence_rounds"] = _merge_evidence_rounds(existing.get("evidence_rounds"), incoming.get("evidence_rounds"))
    merged["boundaries"] = _merge_text_lists(existing.get("boundaries"), incoming.get("boundaries"), limit=24)
    merged["residual_risks"] = _merge_text_lists(
        existing.get("residual_risks"), incoming.get("residual_risks"), limit=24
    )
    if incoming.get("topic_key"):
        merged["topic_key"] = incoming.get("topic_key")
    merged_program_ids = merged.get("merged_program_ids")
    if not isinstance(merged_program_ids, list):
        merged_program_ids = []
    if incoming_id and incoming_id != merged.get("program_id") and incoming_id not in merged_program_ids:
        merged_program_ids.append(incoming_id)
    if merged_program_ids:
        merged["merged_program_ids"] = merged_program_ids[-20:]
    return merged


def update_principle_program_after_postmortem(
    state: dict[str, Any],
    *,
    round_summary: Mapping[str, Any],
    postmortem: Mapping[str, Any],
) -> None:
    program = state.get("current_principle_program")
    if not isinstance(program, dict):
        program = ensure_active_principle_program(state, int(round_summary.get("round") or 0))
    evaluation = round_summary.get("evaluation_summary") if isinstance(round_summary.get("evaluation_summary"), Mapping) else {}
    bundle_results = round_summary.get("bundle_results") if isinstance(round_summary.get("bundle_results"), list) else []
    deltas = [float(item["delta"]) for item in bundle_results if isinstance(item, Mapping) and isinstance(item.get("delta"), (int, float))]
    evidence_entry = {
        "round": round_summary.get("round"),
        "hypothesis_status": postmortem.get("hypothesis_status"),
        "principle_update_action": postmortem.get("principle_update_action"),
        "support_rate": evaluation.get("support_rate"),
        "primary_sun_count": evaluation.get("primary_sun_count"),
        "control_sun_count": evaluation.get("control_sun_count"),
        "mechanism_validated_sun_count": evaluation.get("mechanism_validated_sun_count"),
        "delta": sum(deltas) / len(deltas) if deltas else None,
        "causal_interpretation": _short_text(postmortem.get("causal_interpretation"), 500),
    }
    evidence_rounds = program.get("evidence_rounds")
    if not isinstance(evidence_rounds, list):
        evidence_rounds = []
    evidence_rounds.append({key: value for key, value in evidence_entry.items() if value not in (None, "", [])})
    program["evidence_rounds"] = evidence_rounds[-50:]
    if str(postmortem.get("current_principle_statement") or "").strip():
        program["current_principle_statement"] = str(postmortem["current_principle_statement"]).strip()
    if str(postmortem.get("micro_mechanism") or "").strip():
        program["micro_mechanism"] = str(postmortem["micro_mechanism"]).strip()
    for key in ("failure_boundaries", "unresolved_contradictions"):
        merged = program.get(key)
        if not isinstance(merged, list):
            merged = []
        postmortem_items = postmortem.get(key)
        if not isinstance(postmortem_items, list):
            postmortem_items = []
        for item in postmortem_items:
            text = _short_text(item, 240)
            if text and text not in merged:
                merged.append(text)
        program[key] = merged[-20:]
    status = str(postmortem.get("status") or "").strip().lower()
    action = str(postmortem.get("principle_update_action") or "").strip().lower()
    if status in {"finalize", "reject"} or action in {"finalize", "reject"}:
        book = state.get("principle_book")
        if not isinstance(book, list):
            book = []
        experience = postmortem.get("experience_book_entry") if isinstance(postmortem.get("experience_book_entry"), Mapping) else {}
        book_entry = {
            "program_id": program.get("program_id"),
            "status": "validated_principle" if status == "finalize" or action == "finalize" else "rejected_principle",
            "closed_at_round": round_summary.get("round"),
            "principle_identity": experience.get("principle_identity")
            or ("update_existing_principle" if program.get("principle_book_update_target") else None),
            "updates_principle_id": experience.get("updates_principle_id")
            or postmortem.get("updates_principle_id")
            or program.get("principle_book_update_target"),
            "topic_key": experience.get("topic_key"),
            "principle_statement": experience.get("principle_statement")
            or postmortem.get("current_principle_statement")
            or program.get("current_principle_statement"),
            "micro_mechanism": postmortem.get("micro_mechanism") or program.get("micro_mechanism"),
            "reasoning_chain": experience.get("reasoning_chain") if isinstance(experience.get("reasoning_chain"), list) else [],
            "evidence_rounds": experience.get("evidence_rounds")
            if isinstance(experience.get("evidence_rounds"), list)
            else [item.get("round") for item in program.get("evidence_rounds", []) if isinstance(item, Mapping)],
            "boundaries": experience.get("boundaries")
            if isinstance(experience.get("boundaries"), list)
            else program.get("failure_boundaries", []),
            "residual_risks": experience.get("residual_risks")
            if isinstance(experience.get("residual_risks"), list)
            else program.get("unresolved_contradictions", []),
        }
        update_index = find_principle_book_update_index(book, book_entry, postmortem=postmortem, program=program)
        if update_index is None:
            book.append(book_entry)
        else:
            book[update_index] = merge_principle_book_entry(
                book[update_index] if isinstance(book[update_index], Mapping) else {},
                book_entry,
            )
        state["principle_book"] = book
        state["current_principle_program"] = None
    else:
        program["status"] = "active"
        state["current_principle_program"] = program


def load_e_hull_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "index": int(row["index"]),
                    "formula": row["formula"],
                    "e_hull": float(row["e_hull"]),
                    "nsites": int(row["nsites"]),
                    "template_guess": row["template_guess"],
                    "volume_per_atom": float(row["volume_per_atom"]),
                }
            )
    return rows


def initial_state(work_dir: Path, seed_base: int) -> dict[str, Any]:
    return {
        "schema_version": "material_physics_mvp.v1",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status": "initialized",
        "seed_base": seed_base,
        "current_round": 0,
        "best_round": None,
        "best_support_rate": None,
        "latest_support_rate": None,
        "history": [],
        "principle_book": [],
        "current_principle_program": None,
        "work_dir": str(work_dir),
    }


def _next_principle_program_id(state: Mapping[str, Any]) -> str:
    book = state.get("principle_book")
    book_count = len(book) if isinstance(book, list) else 0
    current = state.get("current_principle_program")
    current_count = 1 if isinstance(current, Mapping) else 0
    return f"principle_program_{book_count + current_count + 1:03d}"


def ensure_active_principle_program(state: dict[str, Any], round_number: int) -> dict[str, Any]:
    book = state.get("principle_book")
    if not isinstance(book, list):
        state["principle_book"] = []
    current = state.get("current_principle_program")
    if isinstance(current, Mapping) and str(current.get("status") or "") == "active":
        program = dict(current)
    else:
        program = {
            "program_id": _next_principle_program_id(state),
            "status": "active",
            "started_round": round_number,
            "inner_iteration": 0,
            "max_inner_rounds": DEFAULT_PRINCIPLE_PROGRAM_MAX_INNER_ROUNDS,
            "current_principle_statement": "New principle program: A/B must propose a microscopic materials principle.",
            "micro_mechanism": "",
            "active_mechanism_ids": [],
            "evidence_rounds": [],
            "failure_boundaries": [],
            "unresolved_contradictions": [],
        }
    program["inner_iteration"] = int(program.get("inner_iteration") or 0) + 1
    program["current_round"] = round_number
    state["current_principle_program"] = program
    return program


def update_principle_program_from_mechanism(state: dict[str, Any], mechanism_payload: Mapping[str, Any]) -> None:
    program = state.get("current_principle_program")
    if not isinstance(program, dict):
        return
    mechanisms = mechanism_payload.get("accepted_mechanisms")
    if not isinstance(mechanisms, list) or not mechanisms:
        return
    primary = mechanisms[0] if isinstance(mechanisms[0], Mapping) else {}
    claim = str(primary.get("claim") or "").strip()
    if claim:
        program["current_principle_statement"] = claim
    micro = str(
        primary.get("micro_mechanism")
        or primary.get("causal_driver")
        or primary.get("rationale_summary")
        or ""
    ).strip()
    if micro:
        program["micro_mechanism"] = micro
    program["active_mechanism_ids"] = [
        str(item.get("id"))
        for item in mechanisms
        if isinstance(item, Mapping) and str(item.get("id") or "").strip()
    ]
    book = state.get("principle_book")
    if isinstance(book, list):
        candidate_entry = {
            "program_id": program.get("program_id"),
            "principle_statement": program.get("current_principle_statement"),
            "micro_mechanism": program.get("micro_mechanism"),
        }
        update_index = find_principle_book_update_index(book, candidate_entry, program=program)
        if update_index is not None and isinstance(book[update_index], Mapping):
            target_id = str(book[update_index].get("program_id") or "").strip()
            if target_id:
                program["principle_book_update_target"] = target_id
                program["principle_identity"] = "update_existing_principle"
    state["current_principle_program"] = program


def load_state(state_path: Path) -> dict[str, Any]:
    state = read_json(state_path, {})
    if not isinstance(state, dict):
        raise ValueError(f"{state_path} must contain a JSON object")
    return state


def completed_round_numbers(state: Mapping[str, Any]) -> set[int]:
    history = state.get("history")
    if not isinstance(history, list):
        return set()
    rounds: set[int] = set()
    for item in history:
        if not isinstance(item, Mapping):
            continue
        try:
            rounds.add(int(item.get("round")))
        except Exception:
            continue
    return rounds


def next_round_to_run(state: Mapping[str, Any]) -> int:
    try:
        current = int(state.get("current_round") or 0)
    except Exception:
        current = 0
    status = str(state.get("status") or "").strip()
    if current > 0 and status in {"running", "error", "recoverable_llm_failure"}:
        if current not in completed_round_numbers(state):
            return current
    return current + 1


def _client(role: str, args: argparse.Namespace, root: Path, *, log_dir: Path | None = None) -> ResponsesClient:
    model = {
        "MECHANISM_A": args.mechanism_model,
        "MECHANISM_B": args.mechanism_model,
        "PREDICTION_C": args.prediction_model,
        "PREDICTION_D": args.prediction_model,
        "EXECUTION_E": args.execution_model,
        "EXECUTION_F": args.execution_model,
    }[role]
    max_tokens = args.max_tokens
    if max_tokens is None and role in {"MECHANISM_A", "MECHANISM_B"}:
        mechanism_max_tokens = getattr(args, "mechanism_max_tokens", DEFAULT_MECHANISM_MAX_TOKENS)
        max_tokens = mechanism_max_tokens if mechanism_max_tokens and mechanism_max_tokens > 0 else None
    if max_tokens is None and role in {"PREDICTION_C", "PREDICTION_D"}:
        prediction_max_tokens = getattr(args, "prediction_max_tokens", DEFAULT_PREDICTION_MAX_TOKENS)
        max_tokens = prediction_max_tokens if prediction_max_tokens and prediction_max_tokens > 0 else None
    if max_tokens is None and role in {"EXECUTION_E", "EXECUTION_F"}:
        execution_max_tokens = getattr(args, "execution_max_tokens", DEFAULT_EXECUTION_MAX_TOKENS)
        max_tokens = execution_max_tokens if execution_max_tokens and execution_max_tokens > 0 else None
    client = ResponsesClient(
        LLMConfig.from_env(
            dotenv=root / args.dotenv,
            role=role,
            model=model,
            temperature=args.temperature,
            max_tokens=max_tokens,
        ),
        log_dir=log_dir,
    )
    if not bool(getattr(args, "disable_local_agents", False)):
        role_key = role.lower()
        candidate_pool_arg = getattr(args, "candidate_pool", "data/mp_candidate_pool/mp_candidates_filtered.jsonl")
        work_dir_arg = getattr(args, "work_dir", "physics_mvp_runs/current")
        trace_dir = (log_dir / "agent_traces") if log_dir else (root / "agent_artifacts" / "material_physics_agents" / role_key / "traces")
        writable_dir = root / "agent_artifacts" / "material_physics_agents" / role_key
        client.local_agent_runtime = LocalAgentRuntime(  # type: ignore[attr-defined]
            root=root,
            trace_dir=trace_dir,
            writable_dir=writable_dir,
            candidate_pool_path=(root / candidate_pool_arg).resolve(),
            state_path=(root / work_dir_arg / "state.json").resolve(),
            max_steps=max(0, int(getattr(args, "agent_max_steps", DEFAULT_AGENT_MAX_STEPS))),
            max_tool_calls_per_step=max(1, int(getattr(args, "agent_max_tool_calls", DEFAULT_AGENT_MAX_TOOL_CALLS))),
            max_tool_result_chars=max(1000, int(getattr(args, "agent_max_tool_result_chars", 6000))),
            allow_project_writes=bool(getattr(args, "agent_allow_project_writes", False)),
        )
    return client


def run_stage_debate(
    *,
    round_number: int,
    stage_name: str,
    stage_role_a: str,
    stage_role_b: str,
    stage_consensus_role: str,
    system_a: str,
    system_b: str,
    make_context: Any,
    make_proposal_prompt: Any,
    make_critique_prompt: Any,
    make_revision_prompt: Any,
    make_final_prompt: Any,
    args: argparse.Namespace,
    root: Path,
    state: Mapping[str, Any],
    pool_summary: Mapping[str, Any],
    llm_log_dir: Path,
    repair_feedback: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    proposer = _client(stage_role_a, args, root, log_dir=llm_log_dir / "a")
    critic = _client(stage_role_b, args, root, log_dir=llm_log_dir / "b")
    context_json = make_context(state, pool_summary, repair_feedback)
    artifacts: list[dict[str, Any]] = []
    proposal = call_json(
        proposer,
        system=system_a,
        user=make_proposal_prompt(context_with_debate_history(context_json, artifacts)),
        role=stage_role_a.lower(),
        metadata={"role": stage_role_a.lower(), "round": round_number, "stage": stage_name, "cycle": 1},
        json_repair_attempts=args.json_repair_attempts,
    )
    artifacts.append({"role": "A", "cycle": 1, "payload": proposal})
    critique: dict[str, Any] = {}
    consensus: dict[str, Any] | None = None
    for cycle in range(1, max(1, args.max_dialogue_rounds) + 1):
        critique = call_json(
            critic,
            system=system_b,
            user=make_critique_prompt(context_with_debate_history(context_json, artifacts), proposal, cycle),
            role=stage_role_b.lower(),
            metadata={"role": stage_role_b.lower(), "round": round_number, "stage": stage_name, "cycle": cycle},
            json_repair_attempts=args.json_repair_attempts,
        )
        artifacts.append({"role": "B", "cycle": cycle, "payload": critique})
        if agreement_reached(proposal, critique):
            consensus = call_json(
                critic,
                system=system_b,
                user=make_final_prompt(context_with_debate_history(context_json, artifacts), proposal, critique),
                role=stage_consensus_role.lower(),
                metadata={"role": stage_consensus_role.lower(), "round": round_number, "stage": stage_name, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            break
        if cycle >= args.max_dialogue_rounds:
            break
        proposal = call_json(
            proposer,
            system=system_a,
            user=make_revision_prompt(context_with_debate_history(context_json, artifacts), proposal, critique, cycle + 1),
            role=stage_role_a.lower(),
            metadata={"role": stage_role_a.lower(), "round": round_number, "stage": stage_name, "cycle": cycle + 1},
            json_repair_attempts=args.json_repair_attempts,
        )
        artifacts.append({"role": "A", "cycle": cycle + 1, "payload": proposal})
    if consensus is None:
        return {
            "status": "unresolved",
            "round": round_number,
            "stage": stage_name,
            "reason": f"{stage_name} debate reached max_dialogue_rounds without explicit consensus.",
            "dialogue": artifacts,
        }, artifacts
    consensus["round"] = round_number
    consensus["dialogue"] = artifacts
    return consensus, artifacts


def materialize_with_repair(
    *,
    round_number: int,
    root: Path,
    args: argparse.Namespace,
    state: Mapping[str, Any],
    pool_records: Sequence[Mapping[str, Any]],
    pool_summary: Mapping[str, Any],
    llm_log_dir: Path,
    prediction_consensus: Mapping[str, Any],
    known_formulas: set[str] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    repair_feedback: Mapping[str, Any] | None = None
    run_work_dir = (root / args.work_dir).resolve()
    round_dir = round_output_dir(run_work_dir, round_number)
    mattergen_config = mattergen_config_from_args(
        args,
        root,
        round_dir / "mattergen_materialization",
    )
    execution_critic = _client("EXECUTION_F", args, root, log_dir=llm_log_dir / "f")
    execution_rejection_streak = 0
    execution_artifacts: list[dict[str, Any]] = []
    branch_cache_records: dict[str, list[dict[str, Any]]] = {}
    branch_cache_fingerprints: dict[str, str] = {}

    def _attach_execution_dialogue(payload: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        result.setdefault("round", round_number)
        result.setdefault("dialogue", execution_artifacts)
        return result

    def _resolve_execution_consensus(
        consensus_payload: Mapping[str, Any],
        proposal_payload: Mapping[str, Any],
        critique_payload: Mapping[str, Any],
    ) -> tuple[tuple[dict[str, Any], list[dict[str, Any]], list[str]] | None, dict[str, Any]]:
        consensus = _attach_execution_dialogue(reconcile_execution_consensus(consensus_payload, proposal_payload))

        def _try_materialize(candidate: dict[str, Any]) -> tuple[tuple[dict[str, Any], list[dict[str, Any]], list[str]] | None, dict[str, Any]]:
            candidate = execution_plan_with_prediction_drift_constraints(candidate, prediction_consensus)
            errors = validate_execution_payload(candidate, target_count=args.target_count)
            if args.materialization_backend == "mattergen":
                errors.extend(validate_mattergen_native_execution_sources(candidate))
            if errors:
                return None, {
                    "validation_errors": errors,
                    "required_repair": (
                        "Repair validation failures before materialization. In MatterGen-native mode, every executable "
                        "branch must remain source='mattergen'; do not switch failed branches to generator or MP-pool."
                    )
                    if args.materialization_backend == "mattergen"
                    else "Repair validation failures before materialization.",
                    "proposal": proposal_payload,
                    "critique": critique_payload,
                }
            selected, materialization_errors = materialize_plan(
                pool_records,
                candidate,
                seed=args.seed_base + round_number,
                max_sites=args.max_sites,
                known_formulas=known_formulas,
                mattergen_config=mattergen_config,
                branch_cache_records=branch_cache_records,
                branch_cache_fingerprints=branch_cache_fingerprints,
            )
            if materialization_errors:
                feedback = summarize_materialization_feedback(
                    plan=candidate,
                    materialization_errors=materialization_errors,
                    proposal=proposal_payload,
                    critique=critique_payload,
                )
                if branch_cache_fingerprints:
                    feedback["successful_branches_to_preserve"] = sorted(branch_cache_fingerprints)
                    feedback["successful_branch_policy"] = (
                        "These branches already materialized successfully and are locked by the controller. "
                        "Do not change their source, count, MatterGen filters, request fields, or sampling knobs in repair. "
                        "Repair only failed branches; if preserving a successful branch makes the matched comparison infeasible, "
                        "return status='prediction_design_infeasible' for C/D redesign."
                    )
                return None, feedback
            return (candidate, selected, materialization_errors), {}

        if consensus.get("status") == "materialization_conflict":
            return None, {"materialization_conflict": consensus, "proposal": proposal_payload, "critique": critique_payload}
        if is_prediction_design_infeasible(consensus):
            return (consensus, [], []), {}
        if consensus.get("status") in {
            "no_materialized_consensus",
            "no_materialized_consensus_pending_preflight",
            "no_materialized_consensus_pending_generator",
            "non_executable",
            "conditional_accept_only_after_preflight",
        }:
            if agreement_reached(proposal_payload, critique_payload):
                fallback = _attach_execution_dialogue(
                    reconcile_execution_consensus(
                        {
                            "status": "consensus",
                            "accepted_bundle_ids": critique_payload.get("accepted_bundle_ids", []),
                            "consensus_summary": (
                                "Controller fallback: final execution consensus contradicted an agreed "
                                "proposal/critique pair, so the agreed proposal was validated directly."
                            ),
                        },
                        proposal_payload,
                    )
                )
                result, feedback = _try_materialize(fallback)
                if result is not None:
                    return result, {}
                feedback["contradictory_final_consensus"] = consensus
                return None, feedback
            return None, {
                "materialization_conflict": consensus,
                "proposal": proposal_payload,
                "critique": critique_payload,
                "needs_generator_fallback": True,
            }
        if consensus.get("status") == "consensus":
            return _try_materialize(consensus)
        return None, {"proposal": proposal_payload, "critique": critique_payload, "consensus": consensus}

    for attempt in range(1, max(1, args.max_dialogue_rounds) + 1):
        execution_proposer = _client("EXECUTION_E", args, root, log_dir=llm_log_dir / "e")
        context_json = execution_context(
            state,
            pool_summary,
            repair_feedback,
            current_inputs={
                "accepted_predictions": prediction_consensus.get("accepted_predictions", []),
                "target_count": args.target_count,
                "materialization_backend": args.materialization_backend,
                "mattergen_defaults": mattergen_prompt_defaults(args),
            },
        )
        proposal = call_json(
            execution_proposer,
            system=EXECUTION_AGENT_E_SYSTEM,
            user=prompt_execution_repair_proposal(context_with_debate_history(context_json, execution_artifacts))
            if repair_feedback
            else prompt_execution_proposal(context_with_debate_history(context_json, execution_artifacts)),
            role="execution_agent_e",
            metadata={"role": "execution_agent_e", "round": round_number, "attempt": attempt},
            json_repair_attempts=args.json_repair_attempts,
        )
        execution_artifacts.append({"role": "E", "attempt": attempt, "payload": proposal})
        if proposal.get("status") in {"materialization_conflict", "hypothesis_conflict", PREDICTION_DESIGN_INFEASIBLE_STATUS}:
            critique = call_json(
                execution_critic,
                system=EXECUTION_AGENT_F_SYSTEM,
                user=prompt_execution_critique(context_with_debate_history(context_json, execution_artifacts), proposal, attempt),
                role="execution_agent_f",
                metadata={"role": "execution_agent_f", "round": round_number, "attempt": attempt},
                json_repair_attempts=args.json_repair_attempts,
            )
            execution_artifacts.append({"role": "F", "attempt": attempt, "payload": critique})
            if agreement_reached(proposal, critique):
                consensus = call_json(
                    execution_critic,
                    system=EXECUTION_AGENT_F_SYSTEM,
                    user=prompt_execution_final(context_with_debate_history(context_json, execution_artifacts), proposal, critique),
                    role="execution_consensus",
                    metadata={"role": "execution_consensus", "round": round_number, "attempt": attempt},
                    json_repair_attempts=args.json_repair_attempts,
                )
                if is_prediction_design_infeasible(consensus):
                    return _attach_execution_dialogue(consensus), [], []
                return _attach_execution_dialogue(consensus), [], []
            if critique_requires_counterproposal(
                critique,
                rejection_streak=execution_rejection_streak,
                threshold=args.critic_counterproposal_after,
            ):
                execution_rejection_streak = 0
                counterproposal = call_json(
                    execution_critic,
                    system=EXECUTION_AGENT_F_SYSTEM,
                    user=prompt_execution_counterproposal(context_with_debate_history(context_json, execution_artifacts), proposal, critique, attempt),
                    role="execution_agent_f_counterproposal",
                    metadata={"role": "execution_agent_f_counterproposal", "round": round_number, "attempt": attempt},
                    json_repair_attempts=args.json_repair_attempts,
                )
                execution_artifacts.append({"role": "F", "attempt": attempt, "mode": "counterproposal", "payload": counterproposal})
                reverse_critique = call_json(
                    execution_proposer,
                    system=EXECUTION_AGENT_E_SYSTEM,
                    user=prompt_execution_reverse_critique(context_with_debate_history(context_json, execution_artifacts), proposal, counterproposal, attempt),
                    role="execution_agent_e_reverse_critique",
                    metadata={"role": "execution_agent_e_reverse_critique", "round": round_number, "attempt": attempt},
                    json_repair_attempts=args.json_repair_attempts,
                )
                execution_artifacts.append({"role": "E", "attempt": attempt, "mode": "reverse_critique", "payload": reverse_critique})
                if agreement_reached(counterproposal, reverse_critique):
                    consensus = call_json(
                        execution_proposer,
                        system=EXECUTION_AGENT_E_SYSTEM,
                        user=prompt_execution_reverse_final(context_with_debate_history(context_json, execution_artifacts), counterproposal, reverse_critique),
                        role="execution_consensus",
                        metadata={"role": "execution_reverse_consensus", "round": round_number, "attempt": attempt},
                        json_repair_attempts=args.json_repair_attempts,
                    )
                    result, feedback = _resolve_execution_consensus(consensus, counterproposal, reverse_critique)
                    if result is not None:
                        return result
                    repair_feedback = feedback
                    continue
                repair_feedback = {
                    "agent_f_counterproposal": counterproposal,
                    "agent_e_reverse_critique": reverse_critique,
                    "proposal": proposal,
                    "critique": critique,
                }
                continue
            execution_rejection_streak += 1
            repair_feedback = {
                "agent_f_rejection": critique,
                "proposal": proposal,
            }
            continue

        if proposal.get("status") in {
            "non_executable",
            "conditional_accept_only_after_preflight",
        }:
            repair_feedback = {"proposal": proposal, "validation_errors": ["execution plan is not yet executable"], "needs_generator_fallback": True}
            continue

        if not isinstance(proposal.get("bundles"), list):
            repair_feedback = {"validation_errors": ["execution proposal must include bundles"], "proposal": proposal}
            continue
        critique = call_json(
            execution_critic,
            system=EXECUTION_AGENT_F_SYSTEM,
            user=prompt_execution_critique(context_with_debate_history(context_json, execution_artifacts), proposal, attempt),
            role="execution_agent_f",
            metadata={"role": "execution_agent_f", "round": round_number, "attempt": attempt},
            json_repair_attempts=args.json_repair_attempts,
        )
        execution_artifacts.append({"role": "F", "attempt": attempt, "payload": critique})
        if not agreement_reached(proposal, critique):
            if critique_requires_counterproposal(
                critique,
                rejection_streak=execution_rejection_streak,
                threshold=args.critic_counterproposal_after,
            ):
                execution_rejection_streak = 0
                counterproposal = call_json(
                    execution_critic,
                    system=EXECUTION_AGENT_F_SYSTEM,
                    user=prompt_execution_counterproposal(context_with_debate_history(context_json, execution_artifacts), proposal, critique, attempt),
                    role="execution_agent_f_counterproposal",
                    metadata={"role": "execution_agent_f_counterproposal", "round": round_number, "attempt": attempt},
                    json_repair_attempts=args.json_repair_attempts,
                )
                execution_artifacts.append({"role": "F", "attempt": attempt, "mode": "counterproposal", "payload": counterproposal})
                reverse_critique = call_json(
                    execution_proposer,
                    system=EXECUTION_AGENT_E_SYSTEM,
                    user=prompt_execution_reverse_critique(context_with_debate_history(context_json, execution_artifacts), proposal, counterproposal, attempt),
                    role="execution_agent_e_reverse_critique",
                    metadata={"role": "execution_agent_e_reverse_critique", "round": round_number, "attempt": attempt},
                    json_repair_attempts=args.json_repair_attempts,
                )
                execution_artifacts.append({"role": "E", "attempt": attempt, "mode": "reverse_critique", "payload": reverse_critique})
                if agreement_reached(counterproposal, reverse_critique):
                    consensus = call_json(
                        execution_proposer,
                        system=EXECUTION_AGENT_E_SYSTEM,
                        user=prompt_execution_reverse_final(context_with_debate_history(context_json, execution_artifacts), counterproposal, reverse_critique),
                        role="execution_consensus",
                        metadata={"role": "execution_reverse_consensus", "round": round_number, "attempt": attempt},
                        json_repair_attempts=args.json_repair_attempts,
                    )
                    result, feedback = _resolve_execution_consensus(consensus, counterproposal, reverse_critique)
                    if result is not None:
                        return result
                    repair_feedback = feedback
                    continue
                repair_feedback = {
                    "agent_f_counterproposal": counterproposal,
                    "agent_e_reverse_critique": reverse_critique,
                    "proposal": proposal,
                    "critique": critique,
                }
                continue
            execution_rejection_streak += 1
            repair_feedback = {"agent_f_rejection": critique, "proposal": proposal}
            continue
        consensus = call_json(
            execution_critic,
            system=EXECUTION_AGENT_F_SYSTEM,
            user=prompt_execution_final(context_with_debate_history(context_json, execution_artifacts), proposal, critique),
            role="execution_consensus",
            metadata={"role": "execution_consensus", "round": round_number, "attempt": attempt},
            json_repair_attempts=args.json_repair_attempts,
        )
        result, feedback = _resolve_execution_consensus(consensus, proposal, critique)
        if result is not None:
            return result
        repair_feedback = feedback
    return {
        "status": "unresolved",
        "reason": "execution plan did not converge",
        "round": round_number,
        "dialogue": execution_artifacts,
    }, [], ["execution plan did not converge"]


def run_prediction_feedback_debate(
    *,
    round_number: int,
    root: Path,
    args: argparse.Namespace,
    state: Mapping[str, Any],
    pool_summary: Mapping[str, Any],
    prediction_llm_dir: Path,
    mechanism_debate: Mapping[str, Any],
    execution_feedback: Mapping[str, Any],
    feedback_round: int,
) -> dict[str, Any]:
    prediction_context_json = prediction_context(
        state,
        pool_summary,
        execution_feedback,
        current_inputs={
            "accepted_mechanisms": mechanism_debate.get("accepted_mechanisms", []),
            "target_count": args.target_count,
            "materialization_backend": args.materialization_backend,
            "mattergen_defaults": mattergen_prompt_defaults(args),
        },
    )
    prediction_artifacts: list[dict[str, Any]] = []
    prediction_proposer = _client("PREDICTION_C", args, root, log_dir=prediction_llm_dir / "c")
    prediction_critic = _client("PREDICTION_D", args, root, log_dir=prediction_llm_dir / "d")
    prediction_proposal = call_json(
        prediction_proposer,
        system=PREDICTION_AGENT_C_SYSTEM,
        user=prompt_prediction_proposal(context_with_debate_history(prediction_context_json, prediction_artifacts)),
        role="prediction_agent_c",
        metadata={
            "role": "prediction_agent_c",
            "round": round_number,
            "cycle": 1,
            "execution_feedback_round": feedback_round,
        },
        json_repair_attempts=args.json_repair_attempts,
    )
    prediction_artifacts.append({"role": "C", "cycle": 1, "execution_feedback_round": feedback_round, "payload": prediction_proposal})
    prediction_debate: dict[str, Any] | None = None
    prediction_rejection_streak = 0
    prediction_critique: dict[str, Any] = {}
    for cycle in range(1, max(1, args.max_dialogue_rounds) + 1):
        prediction_critique = call_json(
            prediction_critic,
            system=PREDICTION_AGENT_D_SYSTEM,
            user=prompt_prediction_critique(context_with_debate_history(prediction_context_json, prediction_artifacts), prediction_proposal, cycle),
            role="prediction_agent_d",
            metadata={
                "role": "prediction_agent_d",
                "round": round_number,
                "cycle": cycle,
                "execution_feedback_round": feedback_round,
            },
            json_repair_attempts=args.json_repair_attempts,
        )
        prediction_artifacts.append({"role": "D", "cycle": cycle, "execution_feedback_round": feedback_round, "payload": prediction_critique})
        if agreement_reached(prediction_proposal, prediction_critique):
            prediction_debate = call_json(
                prediction_critic,
                system=PREDICTION_AGENT_D_SYSTEM,
                user=prompt_prediction_final(context_with_debate_history(prediction_context_json, prediction_artifacts), prediction_proposal, prediction_critique),
                role="prediction_consensus",
                metadata={
                    "role": "prediction_consensus",
                    "round": round_number,
                    "cycle": cycle,
                    "execution_feedback_round": feedback_round,
                },
                json_repair_attempts=args.json_repair_attempts,
            )
            break
        if critique_requires_counterproposal(
            prediction_critique,
            rejection_streak=prediction_rejection_streak,
            threshold=args.critic_counterproposal_after,
        ):
            prediction_rejection_streak = 0
            previous_proposal = prediction_proposal
            prediction_proposal = call_json(
                prediction_critic,
                system=PREDICTION_AGENT_D_SYSTEM,
                user=prompt_prediction_counterproposal(context_with_debate_history(prediction_context_json, prediction_artifacts), previous_proposal, prediction_critique, cycle),
                role="prediction_agent_d_counterproposal",
                metadata={
                    "role": "prediction_agent_d_counterproposal",
                    "round": round_number,
                    "cycle": cycle,
                    "execution_feedback_round": feedback_round,
                },
                json_repair_attempts=args.json_repair_attempts,
            )
            prediction_artifacts.append(
                {"role": "D", "cycle": cycle, "mode": "counterproposal", "execution_feedback_round": feedback_round, "payload": prediction_proposal}
            )
            prediction_critique = call_json(
                prediction_proposer,
                system=PREDICTION_AGENT_C_SYSTEM,
                user=prompt_prediction_reverse_critique(context_with_debate_history(prediction_context_json, prediction_artifacts), previous_proposal, prediction_proposal, cycle),
                role="prediction_agent_c_reverse_critique",
                metadata={
                    "role": "prediction_agent_c_reverse_critique",
                    "round": round_number,
                    "cycle": cycle,
                    "execution_feedback_round": feedback_round,
                },
                json_repair_attempts=args.json_repair_attempts,
            )
            prediction_artifacts.append(
                {"role": "C", "cycle": cycle, "mode": "reverse_critique", "execution_feedback_round": feedback_round, "payload": prediction_critique}
            )
            if agreement_reached(prediction_proposal, prediction_critique):
                prediction_debate = call_json(
                    prediction_proposer,
                    system=PREDICTION_AGENT_C_SYSTEM,
                    user=prompt_prediction_reverse_final(context_with_debate_history(prediction_context_json, prediction_artifacts), prediction_proposal, prediction_critique),
                    role="prediction_consensus",
                    metadata={
                        "role": "prediction_reverse_consensus",
                        "round": round_number,
                        "cycle": cycle,
                        "execution_feedback_round": feedback_round,
                    },
                    json_repair_attempts=args.json_repair_attempts,
                )
                break
        else:
            prediction_rejection_streak += 1
        if cycle >= args.max_dialogue_rounds:
            break
        prediction_proposal = call_json(
            prediction_proposer,
            system=PREDICTION_AGENT_C_SYSTEM,
            user=prompt_prediction_revision(context_with_debate_history(prediction_context_json, prediction_artifacts), prediction_proposal, prediction_critique, cycle + 1),
            role="prediction_agent_c",
            metadata={
                "role": "prediction_agent_c",
                "round": round_number,
                "cycle": cycle + 1,
                "execution_feedback_round": feedback_round,
            },
            json_repair_attempts=args.json_repair_attempts,
        )
        prediction_artifacts.append({"role": "C", "cycle": cycle + 1, "execution_feedback_round": feedback_round, "payload": prediction_proposal})
    if prediction_debate is None:
        prediction_debate = {
            "status": "unresolved",
            "round": round_number,
            "execution_feedback_round": feedback_round,
            "repair_feedback": compact_repair_feedback(execution_feedback),
            "dialogue": prediction_artifacts,
        }
    else:
        prediction_debate.setdefault("round", round_number)
        prediction_debate.setdefault("dialogue", prediction_artifacts)
        prediction_debate["execution_feedback_round"] = feedback_round
        prediction_debate["repair_feedback"] = compact_repair_feedback(execution_feedback)
    return prediction_debate


def run_evaluator(
    *,
    root: Path,
    round_dir: Path,
    input_path: Path,
    results_path: Path,
    training_data: Path,
    ppd_path: Path,
    eval_log: Path,
    args: argparse.Namespace,
) -> None:
    if args.evaluator_backend == "local":
        run_command(
            ["timeout", str(args.eval_timeout), str(root / "evaluate_full.sh")],
            cwd=root,
            log_path=eval_log,
            round_dir=round_dir,
            step_name="evaluate_full_local",
            env={
                "INPUT": str(input_path),
                "OUTPUT": str(results_path),
                "TRAINING_DATA": str(training_data),
                "PPD_PATH": str(ppd_path),
                "DEVICE": str(args.device),
            },
        )
        return

    export_values = {
        "ALL": None,
        "ROOT_DIR": str(root),
        "INPUT": str(input_path),
        "OUTPUT": str(results_path),
        "TRAINING_DATA": str(training_data),
        "PPD_PATH": str(ppd_path),
        "DEVICE": str(args.device),
        "EVAL_LOG": str(eval_log),
        "EVAL_TIMEOUT": str(args.eval_timeout),
    }
    export_arg = ",".join(key if value is None else f"{key}={value}" for key, value in export_values.items())
    cmd = [
        "sbatch",
        "--wait",
        "--parsable",
        f"--partition={args.slurm_partition}",
        f"--cpus-per-task={args.slurm_cpus_per_task}",
        f"--export={export_arg}",
    ]
    if args.slurm_gres:
        cmd.append(f"--gres={args.slurm_gres}")
    cmd.append(str(root / args.slurm_evaluator_script))
    run_command(
        cmd,
        cwd=root,
        log_path=round_dir / "evaluate_submit.log",
        round_dir=round_dir,
        step_name="evaluate_full_slurm",
    )


def run_analysis(
    *,
    root: Path,
    round_dir: Path,
    input_path: Path,
    results_path: Path,
    eval_log: Path,
) -> None:
    analysis_dir = round_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_dir / "summary.json"
    ranked_path = analysis_dir / "e_hull_ranked.csv"
    if json_ok(summary_path) and json_ok(ranked_path):
        return
    run_command(
        [
            sys.executable,
            "-m",
            "crystal_llm.analyze_evaluator_run",
            "--input",
            str(input_path),
            "--logs",
            str(eval_log),
            "--analysis-dir",
            str(analysis_dir),
            "--result-json",
            str(results_path),
        ],
        cwd=root,
        log_path=round_dir / "analysis.log",
        round_dir=round_dir,
        step_name="analyze_evaluator_run",
    )


def run_round(
    *,
    args: argparse.Namespace,
    root: Path,
    work_dir: Path,
    memory_dir: Path,
    pool_records: Sequence[Mapping[str, Any]],
    pool_summary: Mapping[str, Any],
    state: dict[str, Any],
    round_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = round_label(round_number)
    round_dir = work_dir / label
    analysis_dir = round_dir / "analysis"
    eval_dir = round_dir / "eval"
    training_data = Path(args.training_data).resolve() if args.training_data else default_training_data(root)
    ppd_path = Path(args.ppd_path).resolve() if args.ppd_path else default_ppd_path(root)
    known_formulas = load_known_formulas(str(training_data))
    seed = args.seed_base + round_number
    state["status"] = "running"
    state["current_round"] = round_number
    log_event(round_dir, f"ROUND {label} begin seed={seed}")

    mechanism_path = round_dir / "mechanism_debate.json"
    prediction_path = round_dir / "prediction_debate.json"
    execution_path = round_dir / "execution_debate.json"
    input_path = round_dir / "input.json"
    results_path = round_dir / "results.json"
    eval_log = eval_dir / "full_cpu_0.out"
    mechanism_llm_dir = round_dir / "mechanism_llm_calls"
    prediction_llm_dir = round_dir / "prediction_llm_calls"
    execution_llm_dir = round_dir / "execution_llm_calls"
    postmortem_llm_dir = round_dir / "postmortem_llm_calls"
    mechanism_llm_dir.mkdir(parents=True, exist_ok=True)
    prediction_llm_dir.mkdir(parents=True, exist_ok=True)
    execution_llm_dir.mkdir(parents=True, exist_ok=True)
    postmortem_llm_dir.mkdir(parents=True, exist_ok=True)
    ensure_active_principle_program(state, round_number)

    mechanism_context_json = mechanism_context(state, pool_summary, None)
    mechanism_debate = None
    mechanism_proposal = None
    mechanism_critique = None
    mech_artifacts: list[dict[str, Any]] = []
    mechanism_proposer = _client("MECHANISM_A", args, root, log_dir=mechanism_llm_dir / "a")
    mechanism_critic = _client("MECHANISM_B", args, root, log_dir=mechanism_llm_dir / "b")
    mechanism_proposal = call_json(
        mechanism_proposer,
        system=MECHANISM_AGENT_A_SYSTEM,
        user=prompt_mechanism_proposal(context_with_debate_history(mechanism_context_json, mech_artifacts)),
        role="mechanism_agent_a",
        metadata={"role": "mechanism_agent_a", "round": round_number, "cycle": 1},
        json_repair_attempts=args.json_repair_attempts,
    )
    mech_artifacts.append({"role": "A", "cycle": 1, "payload": mechanism_proposal})
    mechanism_rejection_streak = 0
    for cycle in range(1, max(1, args.max_dialogue_rounds) + 1):
        mechanism_critique = call_json(
            mechanism_critic,
            system=MECHANISM_AGENT_B_SYSTEM,
            user=prompt_mechanism_critique(context_with_debate_history(mechanism_context_json, mech_artifacts), mechanism_proposal, cycle),
            role="mechanism_agent_b",
            metadata={"role": "mechanism_agent_b", "round": round_number, "cycle": cycle},
            json_repair_attempts=args.json_repair_attempts,
        )
        mech_artifacts.append({"role": "B", "cycle": cycle, "payload": mechanism_critique})
        if agreement_reached(mechanism_proposal, mechanism_critique):
            mechanism_debate = mechanism_consensus_from_accepted_proposal(
                mechanism_proposal,
                round_number=round_number,
                dialogue=mech_artifacts,
                consensus_summary="Mechanism consensus reconstructed directly from the accepted Agent A proposal after Agent B agreement.",
            )
            break
        if critique_requires_counterproposal(
            mechanism_critique,
            rejection_streak=mechanism_rejection_streak,
            threshold=args.critic_counterproposal_after,
        ):
            mechanism_rejection_streak = 0
            previous_proposal = mechanism_proposal
            mechanism_proposal = call_json(
                mechanism_critic,
                system=MECHANISM_AGENT_B_SYSTEM,
                user=prompt_mechanism_counterproposal(context_with_debate_history(mechanism_context_json, mech_artifacts), previous_proposal, mechanism_critique, cycle),
                role="mechanism_agent_b_counterproposal",
                metadata={"role": "mechanism_agent_b_counterproposal", "round": round_number, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            mech_artifacts.append({"role": "B", "cycle": cycle, "mode": "counterproposal", "payload": mechanism_proposal})
            mechanism_critique = call_json(
                mechanism_proposer,
                system=MECHANISM_AGENT_A_SYSTEM,
                user=prompt_mechanism_reverse_critique(context_with_debate_history(mechanism_context_json, mech_artifacts), previous_proposal, mechanism_proposal, cycle),
                role="mechanism_agent_a_reverse_critique",
                metadata={"role": "mechanism_agent_a_reverse_critique", "round": round_number, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            mech_artifacts.append({"role": "A", "cycle": cycle, "mode": "reverse_critique", "payload": mechanism_critique})
            if agreement_reached(mechanism_proposal, mechanism_critique):
                mechanism_debate = mechanism_consensus_from_accepted_proposal(
                    mechanism_proposal,
                    round_number=round_number,
                    dialogue=mech_artifacts,
                    consensus_summary="Mechanism consensus reconstructed directly from the accepted Agent B counterproposal after Agent A agreement.",
                )
                break
        else:
            mechanism_rejection_streak += 1
        if cycle >= args.max_dialogue_rounds:
            break
        mechanism_proposal = call_json(
            mechanism_proposer,
            system=MECHANISM_AGENT_A_SYSTEM,
            user=prompt_mechanism_revision(context_with_debate_history(mechanism_context_json, mech_artifacts), mechanism_proposal, mechanism_critique, cycle + 1),
            role="mechanism_agent_a",
            metadata={"role": "mechanism_agent_a", "round": round_number, "cycle": cycle + 1},
            json_repair_attempts=args.json_repair_attempts,
        )
        mech_artifacts.append({"role": "A", "cycle": cycle + 1, "payload": mechanism_proposal})
    if mechanism_debate is None:
        mechanism_debate = {"status": "unresolved", "round": round_number, "dialogue": mech_artifacts}
    else:
        mechanism_debate.setdefault("round", round_number)
        mechanism_debate.setdefault("dialogue", mech_artifacts)

    if mechanism_debate.get("status") != "consensus":
        write_json(mechanism_path, mechanism_debate)
        round_summary = summarize_skipped_round(
            round_number=round_number,
            status="unresolved_mechanism",
            artifact=mechanism_debate,
            mechanism_payload=mechanism_debate,
        )
        record_skipped_round(state, round_summary)
        write_round_report(round_dir, round_summary)
        return state, {
            "round": round_number,
            "status": "unresolved_mechanism",
            "artifact": mechanism_debate,
            "round_summary": round_summary,
        }
    accepted_mechanisms = mechanism_debate.get("accepted_mechanisms")
    if (not isinstance(accepted_mechanisms, list) or not accepted_mechanisms) and isinstance(mechanism_proposal.get("mechanisms"), list):
        mechanism_debate["accepted_mechanisms"] = [
            _normalize_mechanism_item(item, index)
            for index, item in enumerate(mechanism_proposal.get("mechanisms", []), start=1)
        ]
        mechanism_debate.setdefault(
            "consensus_summary",
            "Mechanism consensus reconstructed from the accepted Agent A proposal after Agent A/B agreement.",
        )
    mechanism_errors = validate_mechanism_payload(mechanism_debate)
    if mechanism_errors:
        mechanism_debate = {
            "status": "unresolved",
            "round": round_number,
            "reason": "mechanism consensus failed validation",
            "validation_errors": mechanism_errors,
            "dialogue": mech_artifacts,
        }
        write_json(mechanism_path, mechanism_debate)
        return state, {
            "round": round_number,
            "status": "invalid_mechanism",
            "artifact": mechanism_debate,
        }
    write_json(mechanism_path, mechanism_debate)
    update_principle_program_from_mechanism(state, mechanism_debate)

    prediction_context_json = prediction_context(
        state,
        pool_summary,
        None,
        current_inputs={
            "accepted_mechanisms": mechanism_debate.get("accepted_mechanisms", []),
            "target_count": args.target_count,
            "materialization_backend": args.materialization_backend,
            "mattergen_defaults": mattergen_prompt_defaults(args),
        },
    )
    prediction_debate = None
    prediction_artifacts: list[dict[str, Any]] = []
    prediction_proposer = _client("PREDICTION_C", args, root, log_dir=prediction_llm_dir / "c")
    prediction_critic = _client("PREDICTION_D", args, root, log_dir=prediction_llm_dir / "d")
    prediction_proposal = call_json(
        prediction_proposer,
        system=PREDICTION_AGENT_C_SYSTEM,
        user=prompt_prediction_proposal(context_with_debate_history(prediction_context_json, prediction_artifacts)),
        role="prediction_agent_c",
        metadata={"role": "prediction_agent_c", "round": round_number, "cycle": 1},
        json_repair_attempts=args.json_repair_attempts,
    )
    prediction_artifacts.append({"role": "C", "cycle": 1, "payload": prediction_proposal})
    prediction_rejection_streak = 0
    for cycle in range(1, max(1, args.max_dialogue_rounds) + 1):
        prediction_critique = call_json(
            prediction_critic,
            system=PREDICTION_AGENT_D_SYSTEM,
            user=prompt_prediction_critique(context_with_debate_history(prediction_context_json, prediction_artifacts), prediction_proposal, cycle),
            role="prediction_agent_d",
            metadata={"role": "prediction_agent_d", "round": round_number, "cycle": cycle},
            json_repair_attempts=args.json_repair_attempts,
        )
        prediction_artifacts.append({"role": "D", "cycle": cycle, "payload": prediction_critique})
        if agreement_reached(prediction_proposal, prediction_critique):
            prediction_debate = call_json(
                prediction_critic,
                system=PREDICTION_AGENT_D_SYSTEM,
                user=prompt_prediction_final(context_with_debate_history(prediction_context_json, prediction_artifacts), prediction_proposal, prediction_critique),
                role="prediction_consensus",
                metadata={"role": "prediction_consensus", "round": round_number, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            break
        if critique_requires_counterproposal(
            prediction_critique,
            rejection_streak=prediction_rejection_streak,
            threshold=args.critic_counterproposal_after,
        ):
            prediction_rejection_streak = 0
            previous_proposal = prediction_proposal
            prediction_proposal = call_json(
                prediction_critic,
                system=PREDICTION_AGENT_D_SYSTEM,
                user=prompt_prediction_counterproposal(context_with_debate_history(prediction_context_json, prediction_artifacts), previous_proposal, prediction_critique, cycle),
                role="prediction_agent_d_counterproposal",
                metadata={"role": "prediction_agent_d_counterproposal", "round": round_number, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            prediction_artifacts.append({"role": "D", "cycle": cycle, "mode": "counterproposal", "payload": prediction_proposal})
            prediction_critique = call_json(
                prediction_proposer,
                system=PREDICTION_AGENT_C_SYSTEM,
                user=prompt_prediction_reverse_critique(context_with_debate_history(prediction_context_json, prediction_artifacts), previous_proposal, prediction_proposal, cycle),
                role="prediction_agent_c_reverse_critique",
                metadata={"role": "prediction_agent_c_reverse_critique", "round": round_number, "cycle": cycle},
                json_repair_attempts=args.json_repair_attempts,
            )
            prediction_artifacts.append({"role": "C", "cycle": cycle, "mode": "reverse_critique", "payload": prediction_critique})
            if agreement_reached(prediction_proposal, prediction_critique):
                prediction_debate = call_json(
                    prediction_proposer,
                    system=PREDICTION_AGENT_C_SYSTEM,
                    user=prompt_prediction_reverse_final(context_with_debate_history(prediction_context_json, prediction_artifacts), prediction_proposal, prediction_critique),
                    role="prediction_consensus",
                    metadata={"role": "prediction_reverse_consensus", "round": round_number, "cycle": cycle},
                    json_repair_attempts=args.json_repair_attempts,
                )
                break
        else:
            prediction_rejection_streak += 1
        if cycle >= args.max_dialogue_rounds:
            break
        prediction_proposal = call_json(
            prediction_proposer,
            system=PREDICTION_AGENT_C_SYSTEM,
            user=prompt_prediction_revision(context_with_debate_history(prediction_context_json, prediction_artifacts), prediction_proposal, prediction_critique, cycle + 1),
            role="prediction_agent_c",
            metadata={"role": "prediction_agent_c", "round": round_number, "cycle": cycle + 1},
            json_repair_attempts=args.json_repair_attempts,
        )
        prediction_artifacts.append({"role": "C", "cycle": cycle + 1, "payload": prediction_proposal})
    if prediction_debate is None:
        prediction_debate = {"status": "unresolved", "round": round_number, "dialogue": prediction_artifacts}
    else:
        prediction_debate.setdefault("round", round_number)
        prediction_debate.setdefault("dialogue", prediction_artifacts)

    if prediction_debate.get("status") != "consensus":
        write_json(prediction_path, prediction_debate)
        round_summary = summarize_skipped_round(
            round_number=round_number,
            status="unresolved_prediction",
            artifact=prediction_debate,
            mechanism_payload=mechanism_debate,
            prediction_payload=prediction_debate,
        )
        record_skipped_round(state, round_summary)
        write_round_report(round_dir, round_summary)
        return state, {
            "round": round_number,
            "status": "unresolved_prediction",
            "artifact": prediction_debate,
            "round_summary": round_summary,
        }
    accepted_predictions = prediction_debate.get("accepted_predictions")
    if (not isinstance(accepted_predictions, list) or not accepted_predictions) and isinstance(prediction_proposal.get("predictions"), list):
        prediction_debate["accepted_predictions"] = [
            _normalize_prediction_item(item, index)
            for index, item in enumerate(prediction_proposal.get("predictions", []), start=1)
        ]
        prediction_debate.setdefault(
            "consensus_summary",
            "Prediction consensus reconstructed from the accepted Agent C proposal after Agent C/D agreement.",
        )
    prediction_errors = validate_prediction_payload(prediction_debate)
    if prediction_errors:
        prediction_debate = {
            "status": "unresolved",
            "round": round_number,
            "reason": "prediction consensus failed validation",
            "validation_errors": prediction_errors,
            "dialogue": prediction_artifacts,
        }
        write_json(prediction_path, prediction_debate)
        return state, {
            "round": round_number,
            "status": "invalid_prediction",
            "artifact": prediction_debate,
        }
    write_json(prediction_path, prediction_debate)

    execution_debate, selected_records, materialization_errors = materialize_with_repair(
        round_number=round_number,
        root=root,
        args=args,
        state=state,
        pool_records=pool_records,
        pool_summary=pool_summary,
        llm_log_dir=execution_llm_dir,
        prediction_consensus=prediction_debate,
        known_formulas=known_formulas,
    )
    write_json(execution_path, execution_debate)
    prediction_execution_feedback_round = 0
    while (
        is_prediction_design_infeasible(execution_debate)
        and prediction_execution_feedback_round < max(0, int(args.max_prediction_execution_feedback_rounds))
    ):
        prediction_execution_feedback_round += 1
        execution_feedback = prediction_feedback_from_execution(
            execution_debate=execution_debate,
            prediction_debate=prediction_debate,
            feedback_round=prediction_execution_feedback_round,
        )
        log_event(
            round_dir,
            (
                f"ROUND {label} execution feedback round {prediction_execution_feedback_round}: "
                "E/F marked prediction_design_infeasible; returning to C/D"
            ),
        )
        prediction_debate = run_prediction_feedback_debate(
            round_number=round_number,
            root=root,
            args=args,
            state=state,
            pool_summary=pool_summary,
            prediction_llm_dir=prediction_llm_dir,
            mechanism_debate=mechanism_debate,
            execution_feedback=execution_feedback,
            feedback_round=prediction_execution_feedback_round,
        )
        write_json(round_dir / f"prediction_debate_feedback_{prediction_execution_feedback_round:02d}.json", prediction_debate)
        write_json(prediction_path, prediction_debate)
        if prediction_debate.get("status") != "consensus":
            round_summary = summarize_skipped_round(
                round_number=round_number,
                status="unresolved_prediction",
                artifact=prediction_debate,
                mechanism_payload=mechanism_debate,
                prediction_payload=prediction_debate,
            )
            record_skipped_round(state, round_summary)
            write_round_report(round_dir, round_summary)
            return state, {
                "round": round_number,
                "status": "unresolved_prediction",
                "artifact": prediction_debate,
                "round_summary": round_summary,
            }
        prediction_errors = validate_prediction_payload(prediction_debate)
        if prediction_errors:
            prediction_debate = {
                "status": "unresolved",
                "round": round_number,
                "reason": "prediction feedback consensus failed validation",
                "validation_errors": prediction_errors,
                "dialogue": prediction_debate.get("dialogue", []),
                "execution_feedback_round": prediction_execution_feedback_round,
            }
            write_json(prediction_path, prediction_debate)
            return state, {
                "round": round_number,
                "status": "invalid_prediction",
                "artifact": prediction_debate,
            }
        execution_debate, selected_records, materialization_errors = materialize_with_repair(
            round_number=round_number,
            root=root,
            args=args,
            state=state,
            pool_records=pool_records,
            pool_summary=pool_summary,
            llm_log_dir=execution_llm_dir,
            prediction_consensus=prediction_debate,
            known_formulas=known_formulas,
        )
        write_json(round_dir / f"execution_debate_feedback_{prediction_execution_feedback_round:02d}.json", execution_debate)
        write_json(execution_path, execution_debate)
    if execution_debate.get("status") != "consensus":
        unresolved_status = "unresolved_prediction" if is_prediction_design_infeasible(execution_debate) else "unresolved_execution"
        round_summary = summarize_skipped_round(
            round_number=round_number,
            status=unresolved_status,
            artifact=execution_debate,
            mechanism_payload=mechanism_debate,
            prediction_payload=prediction_debate,
            execution_payload=execution_debate,
        )
        record_skipped_round(state, round_summary)
        write_round_report(round_dir, round_summary)
        return state, {
            "round": round_number,
            "status": unresolved_status,
            "artifact": execution_debate,
            "round_summary": round_summary,
        }

    input_structures = []
    for item_index, record in enumerate(selected_records, start=1):
        if isinstance(record.get("structure_dict"), Mapping):
            structure = Structure.from_dict(dict(record["structure_dict"]))
        else:
            path = Path(str(record.get("cif_path") or ""))
            structure = Structure.from_file(str(path))
        payload = structure.as_dict()
        payload["properties"] = dict(payload.get("properties") or {})
        payload["properties"]["crystal_llm_material_id"] = record.get("material_id")
        payload["properties"]["crystal_llm_formula"] = record.get("formula")
        payload["properties"]["crystal_llm_source"] = record.get("crystal_llm_source", "mp_pool")
        payload["properties"]["crystal_llm_pool_metadata"] = {
            "bundle_id": record.get("physics_bundle_id"),
            "prediction_ids": record.get("physics_prediction_ids", []),
            "role": record.get("physics_role"),
            "expected_relation": record.get("physics_expected_relation"),
            "selection_order": record.get("physics_selection_order"),
            "candidate_pool_index": record.get("material_id"),
        }
        if record.get("crystal_llm_generated_from_formula_probes"):
            payload["properties"]["crystal_llm_generator_formula_probes"] = record.get("crystal_llm_generated_from_formula_probes")
        if record.get("crystal_llm_generated_from_structure_dicts"):
            payload["properties"]["crystal_llm_generator_structure_dicts"] = record.get("crystal_llm_generated_from_structure_dicts")
        if record.get("crystal_llm_generator_preflight"):
            payload["properties"]["crystal_llm_generator_preflight"] = record.get("crystal_llm_generator_preflight")
        if record.get("crystal_llm_generator_backend"):
            payload["properties"]["crystal_llm_generator_backend"] = record.get("crystal_llm_generator_backend")
        if record.get("crystal_llm_generated_from_mattergen_requests"):
            payload["properties"]["crystal_llm_generator_mattergen_requests"] = record.get("crystal_llm_generated_from_mattergen_requests")
        if record.get("crystal_llm_mattergen_report"):
            payload["properties"]["crystal_llm_mattergen_report"] = record.get("crystal_llm_mattergen_report")
        if record.get("crystal_llm_mattergen_controller_motif_checks"):
            payload["properties"]["crystal_llm_mattergen_controller_motif_checks"] = record.get("crystal_llm_mattergen_controller_motif_checks")
        input_structures.append(payload)
    write_json(input_path, input_structures)

    if args.force or not json_ok(results_path) or not eval_log.exists():
        run_evaluator(
            root=root,
            round_dir=round_dir,
            input_path=input_path,
            results_path=results_path,
            training_data=training_data,
            ppd_path=ppd_path,
            eval_log=eval_log,
            args=args,
        )

    run_analysis(
        root=root,
        round_dir=round_dir,
        input_path=input_path,
        results_path=results_path,
        eval_log=eval_log,
    )

    analysis_summary = round_dir / "analysis" / "summary.json"
    if not json_ok(analysis_summary):
        raise FileNotFoundError(f"analysis summary missing: {analysis_summary}")
    summary_data = read_json(analysis_summary, {})
    if not isinstance(summary_data, Mapping):
        raise ValueError("analysis summary must be a JSON object")
    rows = load_e_hull_rows(round_dir / "analysis" / "e_hull_ranked.csv")
    bundle_results = bundle_results_from_analysis(selected_records, rows, execution_debate.get("accepted_bundles", []))
    round_summary = summarize_round(
        round_number=round_number,
        mechanism_payload=mechanism_debate,
        prediction_payload=prediction_debate,
        execution_payload=execution_debate,
        analysis_summary=summary_data,
        selected_records=selected_records,
        bundle_results=bundle_results,
    )
    active_program = _compact_principle_program_for_context(state.get("current_principle_program"))
    if active_program:
        round_summary["principle_program"] = active_program
    principle_postmortem = run_principle_postmortem(
        args=args,
        root=root,
        round_dir=round_dir,
        llm_log_dir=postmortem_llm_dir,
        state=state,
        pool_summary=pool_summary,
        round_summary=round_summary,
    )
    round_summary["principle_postmortem"] = principle_postmortem
    update_principle_program_after_postmortem(state, round_summary=round_summary, postmortem=principle_postmortem)
    write_round_report(round_dir, round_summary)

    state.setdefault("history", [])
    if not isinstance(state["history"], list):
        state["history"] = []
    state["history"].append(round_summary)
    state["updated_at_utc"] = utc_now()
    state["status"] = "running"
    support_rate = round_summary["evaluation_summary"]["support_rate"]
    state["latest_support_rate"] = support_rate
    best_support = state.get("best_support_rate")
    if best_support is None or (support_rate is not None and (best_support is None or support_rate > float(best_support))):
        state["best_support_rate"] = support_rate
        state["best_round"] = round_number
        state["best_round_summary"] = round_summary
    return state, {
        "round": round_number,
        "status": "complete",
        "round_summary": round_summary,
        "analysis_summary": summary_data,
        "bundle_results": bundle_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.json_repair_attempts = max(0, int(args.json_repair_attempts))
    root = Path(args.root).resolve()
    work_dir = (root / args.work_dir).resolve()
    memory_dir = (root / args.memory_dir).resolve()
    candidate_pool_path = (root / args.candidate_pool).resolve()
    pool_records = load_candidate_pool(candidate_pool_path)
    pool_summary = pool_digest(pool_records)
    work_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / "state.json"
    if state_path.exists():
        state = load_state(state_path)
    else:
        state = initial_state(work_dir, args.seed_base)
        write_json(state_path, state)
    print(
        f"[{utc_now()}] controller_start root={root} work_dir={work_dir} pool_count={len(pool_records)} "
        f"max_rounds={args.max_rounds} target_count={args.target_count}",
        flush=True,
    )

    current_round = next_round_to_run(state)
    rounds_completed = 0
    while args.max_rounds == 0 or rounds_completed < args.max_rounds:
        round_dir = round_output_dir(work_dir, current_round)
        try:
            state, result = run_round(
                args=args,
                root=root,
                work_dir=work_dir,
                memory_dir=memory_dir,
                pool_records=pool_records,
                pool_summary=pool_summary,
                state=state,
                round_number=current_round,
            )
            ensure_round_completed(result)
            write_json(state_path, state)
            if result.get("status") == "complete":
                log_event(round_dir, f"ROUND {round_label(current_round)} complete support_rate={state.get('latest_support_rate')}")
            else:
                log_event(round_dir, f"ROUND {round_label(current_round)} skipped status={result.get('status')}")
            rounds_completed += 1
            current_round += 1
            if args.sleep_between_rounds > 0:
                time.sleep(args.sleep_between_rounds)
        except KeyboardInterrupt:
            print(f"[{utc_now()}] controller_interrupted", flush=True)
            write_json(state_path, state)
            return 130
        except RecoverableLLMFailure as exc:
            pause = {
                "schema_version": "af_recoverable_llm_pause.v1",
                "created_at_utc": utc_now(),
                "status": "recoverable_llm_failure",
                "round": current_round,
                "role": exc.role,
                "metadata": exc.metadata,
                "error": exc.error,
                "attempts": exc.attempts,
                "work_dir": str(work_dir),
                "state_path": str(state_path),
                "recommended_action": "Restart the A-F controller from the same work_dir after the LLM relay is healthy.",
            }
            write_json(round_dir / "recoverable_pause.json", pause)
            state["status"] = "recoverable_llm_failure"
            state["updated_at_utc"] = utc_now()
            write_json(state_path, state)
            log_event(
                round_dir,
                f"recoverable_llm_pause round={current_round} role={exc.role} "
                f"exit_code={RECOVERABLE_LLM_EXIT_CODE} error={exc.error}",
            )
            return RECOVERABLE_LLM_EXIT_CODE
        except Exception as exc:
            log_event(round_dir, f"controller_error round={current_round}: {exc}")
            state["status"] = "error"
            state["updated_at_utc"] = utc_now()
            write_json(state_path, state)
            if not args.continue_on_error:
                return 1
            current_round += 1
            rounds_completed += 1
    finalize_controller_state(state)
    write_json(state_path, state)
    print(f"[{utc_now()}] controller_done rounds_completed={rounds_completed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
