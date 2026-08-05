from __future__ import annotations

import itertools
import json
from pathlib import Path
import warnings

import pytest
from pymatgen.core import Lattice, Structure

import crystal_llm.run_xy_experience_debate as xy_debate
from crystal_llm.llm_client import LLMError
from crystal_llm.run_xy_experience_debate import (
    JSONOutputRepairFailure,
    RecoverableLLMFailure,
    DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR,
    DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM,
    DEFAULT_MATTERGEN_HARD_TARGET_MAX_EXACT_ELEMENTS,
    DEFAULT_MATTERGEN_HARD_TARGET_MIN_NUM_BATCHES,
    DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM,
    DEFAULT_MATTERGEN_MIN_VOLUME_PER_ATOM,
    _compact_strategy_cooldowns_for_prompt,
    _mattergen_config_with_xy_density_cap,
    _normalise_mattergen_request_for_run,
    _xy_density_volume_cap_from_material_description,
    abstract_formula_tokens_from_text,
    call_json_object,
    compact_dialogue,
    compact_candidate_payload_for_prompt,
    compact_material_description_for_prompt,
    compact_material_description_for_review_prompt,
    compact_review_for_prompt,
    compact_state_context,
    build_sequential_report,
    candidate_payload_declares_no_faithful_generator_candidate,
    design_book_agrees,
    design_rules_from_payload,
    enforce_exhausted_strategy_route_audit_guard,
    enforce_postmortem_next_strategy_guard,
    evaluator_null_elements_from_formula,
    evaluator_null_elements_from_memory,
    executable_generator_rule_from_locked_spec,
    failed_or_used_formulas_from_memory,
    failed_volume_boundaries_from_memory,
    freeze_materialized_candidate_specs,
    formula_mentions_from_text,
    generator_executable_schema_rules,
    is_recoverable_llm_error_message,
    json_output_repair_feedback,
    latest_strategy_search_policy_supersedes,
    latest_strategy_selection_order_errors,
    enforce_no_valid_strategy_audit_guard,
    enforce_latest_strategy_selection_order_guard,
    locked_candidate_specs_from_feedback,
    memory_with_transient_materialization_errors,
    materialization_dry_run_feedback,
    materialize_candidate_specs,
    material_description_from_payload,
    material_payload_declares_no_valid_description,
    mattergen_excluded_target_preflight_errors,
    mattergen_feedback_requires_xy_strategy_revision,
    mattergen_saturation_analysis_from_errors,
    mattergen_strategy_cooldown_preflight_errors,
    mattergen_strategy_cooldowns_from_memory,
    missing_no_valid_strategy_formula_audits,
    no_valid_material_consensus_from_payload,
    normalize_candidate_spec,
    parse_args,
    parse_formula_probe_string,
    populate_sequential_controller_constraints,
    prompt_x_sequential_material_proposal,
    prompt_x_sequential_postmortem,
    prompt_w_sequential_candidate_counterproposal,
    prompt_x_sequential_material_reverse_review,
    prompt_y_sequential_material_review,
    prompt_y_sequential_material_final,
    prompt_y_sequential_material_counterproposal,
    prompt_y_sequential_postmortem_review,
    prompt_z_sequential_candidate_reverse_review,
    prompt_w_sequential_candidate_review,
    prompt_z_sequential_candidate,
    postmortem_blocked_next_strategy_formulas,
    read_sequential_memory,
    review_requires_counterproposal,
    sequential_candidate_agrees,
    sequential_context_payload,
    sequential_description_agrees,
    sequential_generator_repair_feedback,
    sequential_iteration_plan,
    sequential_fallback_postmortem,
    sequential_no_valid_description_agrees,
    sequential_no_valid_description_agrees_with_context,
    seed_sequential_memory_if_needed,
    sanitize_json_like_model_output,
    selected_reduced_formula_from_payload,
    split_shards,
    strategy_formula_candidates_from_text,
    strategy_formula_candidate_tokens_from_text,
    summarize_single_evaluation,
    template_formula_from_material_description,
    template_volume_boundary_key,
    template_volume_boundary_keys_from_formula,
    used_formulas_from_memory,
    validate_xy_strategy_constraints,
    validate_xy_sun_candidate_queue,
    validate_template_only_material_description,
    write_input_structures,
    xy_strategy_constraints_from_context,
    xy_search_policy_from_memory,
)
from crystal_llm.xy_sequential_supervisor import strip_managed_controller_args


def template_audit(template: str = "rocksalt") -> dict[str, object]:
    return {
        "mechanism_local_motif": "explicit generator template motif",
        "required_coordination_or_polyhedra": ["template coordination is the intended local motif"],
        "chosen_template": template,
        "template_realizes_motif": True,
        "unsupported_motif_substitution": False,
        "structure_dict_required": False,
        "why_template_is_faithful": "the generator template is the local motif being tested",
        "generator_limitations": [],
    }


def rocksalt_material_payload(a_element: str, x_element: str = "Se") -> dict[str, object]:
    return {
        "status": "material_description_proposal",
        "agent": "X",
        "iteration": 1,
        "material_description": {
            "natural_language_description": f"rocksalt {a_element}{x_element}",
            "generator_template": "rocksalt",
            "generator_role_mapping": {
                "A": {"element": a_element, "oxidation_state": 2},
                "X": {"element": x_element, "oxidation_state": -2},
            },
            "template_formula_family": "AX rocksalt chalcogenide",
            "expected_local_motif": "sixfold octahedral rocksalt coordination",
            "why_template_is_faithful": "rocksalt is the intended octahedral motif",
        },
    }


def xy_eval_record(
    iteration: int,
    formula: str,
    template: str,
    x_element: str,
    e_hull: float,
    *,
    is_sun: bool = False,
) -> dict[str, object]:
    role_mapping = {
        "rocksalt": {
            "A": {"element": formula.rstrip(x_element) or "La", "oxidation_state": 2},
            "X": {"element": x_element, "oxidation_state": -2},
        },
        "fluorite": {
            "A": {"element": formula.rstrip("H2") or "La", "oxidation_state": 2},
            "X": {"element": x_element, "oxidation_state": -1},
        },
        "double_perovskite": {
            "A": {"element": "Ba", "oxidation_state": 2},
            "B": {"element": "Ca", "oxidation_state": 2},
            "B2": {"element": "Os", "oxidation_state": 6},
            "X": {"element": x_element, "oxidation_state": -2},
        },
        "spinel": {
            "A": {"element": "Zn", "oxidation_state": 2},
            "B": {"element": "Ga", "oxidation_state": 3},
            "X": {"element": x_element, "oxidation_state": -2},
        },
    }.get(
        template,
        {
            "A": {"element": "La", "oxidation_state": 3},
            "X": {"element": x_element, "oxidation_state": -3},
        },
    )
    return {
        "iteration": iteration,
        "status": "evaluated",
        "material_description": {
            "generator_template": template,
            "generator_role_mapping": role_mapping,
            "template_formula_family": f"{template}/{x_element}",
        },
        "executable_generator_rule": {"formula": formula, "generator_template": template},
        "selected_record": {"formula": formula},
        "evaluation_result": {"formula": formula, "e_hull": e_hull, "is_sun": is_sun},
        "xy_postmortem": {"outcome_class": "sun" if is_sun else ("near_miss" if e_hull < 0.03 else "high_e_hull")},
    }


def test_split_shards_oversamples_without_losing_target() -> None:
    shards = split_shards(candidate_count=100, oversample=1.25, shards=20)

    assert len(shards) == 20
    assert sum(shards) == 125
    assert max(shards) - min(shards) <= 1


def test_sequential_candidate_count_omitted_means_unbounded() -> None:
    args = parse_args(["--generation-protocol", "sequential_single"])
    iterations, limit = sequential_iteration_plan(args.candidate_count)

    assert args.candidate_count is None
    assert limit is None
    assert list(itertools.islice(iterations, 5)) == [1, 2, 3, 4, 5]


def test_sequential_candidate_count_positive_limits_iterations() -> None:
    iterations, limit = sequential_iteration_plan(3)

    assert limit == 3
    assert list(iterations) == [1, 2, 3]


def test_supervisor_strips_managed_controller_args() -> None:
    assert strip_managed_controller_args(
        [
            "--",
            "--state",
            "state.json",
            "--work-dir",
            "old_work",
            "--seed-sequential-memory=old_memory.json",
            "--candidate-count",
            "100",
        ]
    ) == ["--state", "state.json", "--candidate-count", "100"]


def test_summarize_single_evaluation_marks_missing_e_hull(tmp_path) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "e_hull_ranked.csv").write_text(
        "index,formula,e_hull,nsites,template_guess,volume_per_atom\n",
        encoding="utf-8",
    )
    (analysis_dir / "summary.json").write_text(
        json.dumps(
            {
                "count": 0,
                "input_structure_count": 1,
                "missing_e_hull_count": 1,
                "missing_e_hull_rows": [
                    {
                        "index": 1,
                        "formula": "RaF2",
                        "reason": "e_hull_not_found_in_evaluator_log",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = summarize_single_evaluation(tmp_path, {"formula": "RaF2"})

    assert result["formula"] == "RaF2"
    assert result["e_hull"] is None
    assert result["is_sun"] is False
    assert result["evaluation_error"] == "missing_e_hull"
    assert result["missing_e_hull_count"] == 1


class _ScriptedTextClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.user_prompts: list[str] = []

    def complete_text(self, *, system: str, user: str, metadata: dict[str, object]) -> str:
        self.user_prompts.append(user)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return str(response)


def test_call_json_object_retries_transport_errors_without_json_repair_prompt() -> None:
    client = _ScriptedTextClient([LLMError("<urlopen error timed out>"), '{"ok": true}'])

    result = call_json_object(
        client,  # type: ignore[arg-type]
        system="system",
        user="ORIGINAL_PROMPT",
        role="agent_z_sequential",
        metadata={"role": "agent_z_sequential"},
        json_repair_attempts=1,
    )

    assert result == {"ok": True}
    assert client.user_prompts == ["ORIGINAL_PROMPT", "ORIGINAL_PROMPT"]


def test_call_json_object_role_retries_recoverable_without_json_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_ROLE_RECOVERABLE_RETRIES", "2")
    monkeypatch.setenv("LLM_ROLE_RECOVERABLE_RETRY_SLEEP", "0")
    client = _ScriptedTextClient(
        [
            LLMError("LLM HTTP 503: server_is_overloaded"),
            LLMError("LLM HTTP 503: server_is_overloaded"),
            '{"ok": true}',
        ]
    )

    result = call_json_object(
        client,  # type: ignore[arg-type]
        system="system",
        user="ORIGINAL_PROMPT",
        role="agent_z_sequential",
        metadata={"role": "agent_z_sequential"},
        json_repair_attempts=0,
    )

    assert result == {"ok": True}
    assert client.user_prompts == ["ORIGINAL_PROMPT", "ORIGINAL_PROMPT", "ORIGINAL_PROMPT"]


def test_call_json_object_raises_recoverable_failure_after_transport_budget() -> None:
    client = _ScriptedTextClient([LLMError("<urlopen error timed out>"), LLMError("<urlopen error [Errno 111] Connection refused>")])

    try:
        call_json_object(
            client,  # type: ignore[arg-type]
            system="system",
            user="ORIGINAL_PROMPT",
            role="agent_y_sequential",
            metadata={"role": "agent_y_sequential", "stage": "xy_review", "iteration": 25},
            json_repair_attempts=1,
        )
    except RecoverableLLMFailure as exc:
        assert exc.role == "agent_y_sequential"
        assert exc.metadata["iteration"] == 25
        assert "Connection refused" in exc.error
    else:
        raise AssertionError("expected RecoverableLLMFailure")

    assert client.user_prompts == ["ORIGINAL_PROMPT", "ORIGINAL_PROMPT"]


def test_call_json_object_uses_json_repair_prompt_for_invalid_text() -> None:
    client = _ScriptedTextClient(["not json", '{"ok": true}'])

    result = call_json_object(
        client,  # type: ignore[arg-type]
        system="system",
        user="ORIGINAL_PROMPT",
        role="agent_z_sequential",
        metadata={"role": "agent_z_sequential"},
        json_repair_attempts=1,
    )

    assert result == {"ok": True}
    assert client.user_prompts[0] == "ORIGINAL_PROMPT"
    assert "JSON_REPAIR_REQUEST" in client.user_prompts[1]
    assert "Previous response preview:\nnot json" in client.user_prompts[1]
    assert "Do not request tools in this JSON repair response" in client.user_prompts[1]
    assert "do not concatenate a tool_request JSON object" in client.user_prompts[1]


def test_sanitize_json_like_model_output_removes_replace_expression() -> None:
    text = '{"request_id": "iter37 q1".replace(" ", ""), "ok": true}'

    assert sanitize_json_like_model_output(text) == '{"request_id": "iter37q1", "ok": true}'


def test_call_json_object_accepts_simple_replace_expression_after_sanitize() -> None:
    client = _ScriptedTextClient(['{"request_id": "iter37 q1".replace(" ", ""), "ok": true}'])

    result = call_json_object(
        client,  # type: ignore[arg-type]
        system="system",
        user="ORIGINAL_PROMPT",
        role="agent_z_sequential",
        metadata={"role": "agent_z_sequential"},
        json_repair_attempts=0,
    )

    assert result == {"request_id": "iter37q1", "ok": True}


def test_call_json_object_raises_structured_json_failure_after_repair_budget() -> None:
    client = _ScriptedTextClient(["not json"])

    with pytest.raises(JSONOutputRepairFailure) as exc_info:
        call_json_object(
            client,  # type: ignore[arg-type]
            system="system",
            user="ORIGINAL_PROMPT",
            role="agent_z_sequential",
            metadata={"role": "agent_z_sequential", "iteration": 37},
            json_repair_attempts=0,
        )

    exc = exc_info.value
    assert exc.role == "agent_z_sequential"
    assert exc.metadata["iteration"] == 37
    assert "JSON" in exc.error
    assert exc.last_text == "not json"


def test_json_output_repair_feedback_is_actionable_for_z() -> None:
    exc = JSONOutputRepairFailure(
        role="agent_z_sequential",
        metadata={"role": "agent_z_sequential"},
        error="JSONDecodeError: Expecting ',' delimiter",
        attempts=3,
        last_text='{"request_id": "abc".replace(" ", "")}',
    )

    feedback = json_output_repair_feedback(
        exc=exc,
        iteration=37,
        description_attempt=1,
        repair_round=2,
        backend="mattergen",
    )

    assert feedback["source"] == "controller_json_parse_feedback"
    assert feedback["controller_error"] == "JSONDecodeError: Expecting ',' delimiter"
    assert ".replace(...)" in feedback["json_rules"]["forbidden_in_json_values"]
    assert "previous_response_preview" in feedback
    assert feedback["executable_generator_rules"]["generator_backend"] == "mattergen"


def test_recoverable_llm_error_message_classification() -> None:
    assert is_recoverable_llm_error_message("LLM HTTP 502: bad gateway")
    assert is_recoverable_llm_error_message("<urlopen error [Errno 104] Connection reset by peer>")
    assert is_recoverable_llm_error_message("<urlopen error [Errno 111] Connection refused>")
    assert is_recoverable_llm_error_message("error={'code':'server_is_overloaded','message':'Our servers are currently overloaded. Please try again later.'}")
    assert is_recoverable_llm_error_message("Codex CLI timed out after 1800.0s: failed to refresh available models: timeout waiting for child process to exit")
    assert is_recoverable_llm_error_message("Codex CLI exited with code 1: stream disconnected before completion: error sending request for url (https://llm.example.invalid/v1/responses)")
    assert is_recoverable_llm_error_message("worker quit with fatal: Transport channel closed, when Client(HttpRequest(...))")
    assert not is_recoverable_llm_error_message("LLM HTTP 403: negative balance")
    assert not is_recoverable_llm_error_message("JSONDecodeError: bad json")


def test_experience_context_exposes_full_principle_book() -> None:
    state = {
        "current_round": 12,
        "history": [{"round": index} for index in range(5)],
        "principle_book": [
            {
                "program_id": f"principle_program_{index:03d}",
                "status": "validated_principle" if index % 2 else "rejected_principle",
                "principle_statement": f"principle {index}",
            }
            for index in range(1, 9)
        ],
    }

    experience_context = compact_state_context(state, include_experience=True)
    baseline_context = compact_state_context(state, include_experience=False)

    assert len(experience_context["principle_book"]) == 8
    assert experience_context["principle_book"][0]["program_id"] == "principle_program_001"
    assert baseline_context["principle_book"] == []


def test_design_rules_from_payload_accepts_design_experience_book() -> None:
    payload = {
        "design_experience_book": [
            {
                "design_rule_id": "xy_s001_r001",
                "source_principle_ids": ["principle_program_001"],
            }
        ]
    }

    assert design_rules_from_payload(payload)[0]["design_rule_id"] == "xy_s001_r001"


def test_design_book_agreement_requires_rule_payload() -> None:
    review = {"agree": True, "approved_design_rule_ids": ["xy_s001_r001"]}

    assert not design_book_agrees(review, {"design_experience_book": []})
    assert design_book_agrees(
        review,
        {"design_experience_book": [{"design_rule_id": "xy_s001_r001"}]},
    )


def test_sequential_context_points_to_generation_history(tmp_path) -> None:
    context = sequential_context_payload(
        state={"current_round": 5, "history": [], "principle_book": []},
        mode="experience_xy",
        iteration=3,
        memory_path=tmp_path / "memory.json",
        candidate_source="generator",
        seed=99,
    )

    assert context["generation_protocol"] == "sequential_single"
    assert context["iteration"] == 3
    assert context["candidate_count_this_iteration"] == 1
    assert "xy_generation_history_path" not in context


def test_sequential_context_exposes_template_only_constraints(tmp_path) -> None:
    context = sequential_context_payload(
        state={"current_round": 5, "history": [], "principle_book": []},
        mode="experience_xy",
        iteration=3,
        memory_path=tmp_path / "memory.json",
        candidate_source="generator",
        seed=99,
        generator_template_only=True,
    )

    constraints = context["controller_constraints"]
    assert constraints["generator_template_only"] is True
    assert constraints["structure_dicts_allowed"] is False
    assert "perovskite" in constraints["allowed_generator_templates"]
    assert constraints["generator_template_role_requirements"]["perovskite"] == ["A", "B", "X"]


def test_sequential_context_mattergen_disables_template_hard_gates(tmp_path) -> None:
    context = sequential_context_payload(
        state={"current_round": 5, "history": [], "principle_book": []},
        mode="experience_xy",
        iteration=3,
        memory_path=tmp_path / "memory.json",
        candidate_source="generator",
        seed=99,
        generator_backend="mattergen",
    )

    constraints = context["controller_constraints"]
    policy = constraints["search_policy"]
    assert context["generator_backend"] == "mattergen"
    assert constraints["mattergen_backend_enabled"] is True
    assert policy["schema_version"] == "xy_search_policy.mattergen.v1"
    assert policy["template_policy_enforcement"] == "disabled_for_mattergen"
    assert "exploit_template_allowlist" not in policy
    assert "preferred_jump_templates" not in policy
    x_prompt = prompt_x_sequential_material_proposal(context, iteration=3)
    y_prompt = prompt_y_sequential_material_review(
        context,
        {"status": "material_description_proposal", "material_description": {}},
        iteration=3,
        cycle=1,
    )
    forbidden_hard_gate = "candidate 1 and candidate 2 must come from preferred_exploitation_basins or exploit_template_allowlist"
    assert "MatterGen chemical-system acquisition budget" in y_prompt
    assert "chemical-system acquisition budget" in x_prompt
    assert forbidden_hard_gate not in x_prompt
    assert forbidden_hard_gate not in y_prompt


def test_mattergen_success_supersedes_stale_backend_repair_gate(tmp_path) -> None:
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "iteration": 10,
                        "status": "evaluated",
                        "candidate_spec": {
                            "mattergen_requests": [
                                {
                                    "backend": "mattergen",
                                    "properties_to_condition_on": {
                                        "chemical_system": "Rb-Cd-Br",
                                        "energy_above_hull": 0.0,
                                    },
                                }
                            ]
                        },
                        "selected_record": {"crystal_llm_generator_backend": "mattergen"},
                        "evaluation_result": {"formula": "Rb2CdBr4", "e_hull": 0.007},
                    },
                    {
                        "iteration": 11,
                        "status": "strategy_blocked",
                        "xy_postmortem": {
                            "outcome_class": "strategy_blocked",
                            "next_strategy": (
                                "Hard gate first: if explicit MatterGen backend repair confirmation is still absent, "
                                "return no_valid_material_description and dispatch nothing. If repaired, use MatterGen "
                                "chemical-system conditioning only on Na-Zn-Br: 1) reduced_formula=Na2ZnBr4, "
                                "2) reduced_formula=NaZn2Br5, 3) reduced_formula=Na3Zn2Br7."
                            ),
                            "failure_boundaries": [
                                "Do not dispatch any MatterGen request until backend repair is explicitly confirmed.",
                                "Do not repeat Rb2CdBr4.",
                            ],
                        },
                    },
                ]
            }
        )
    )

    context = sequential_context_payload(
        state={"current_round": 5, "history": [], "principle_book": []},
        mode="experience_xy",
        iteration=12,
        memory_path=memory_path,
        candidate_source="generator",
        seed=99,
        generator_backend="mattergen",
    )

    status = context["controller_constraints"]["mattergen_operational_status"]
    latest = context["latest_xy_strategy_constraints"]
    latest_text = json.dumps(latest).lower()
    assert status["verified"] is True
    assert latest["mattergen_backend_gate_superseded"] is True
    assert "repair confirmation" not in latest_text
    assert "do not dispatch" not in latest_text
    assert latest["next_strategy_candidate_formulas"] == ["Na2ZnBr4", "NaZn2Br5", "Na3Zn2Br7"]
    assert "Na2ZnBr4" in latest["next_strategy"]
    assert "Rb2CdBr4" in str(latest["failure_boundaries"])

    proposal_prompt = prompt_x_sequential_material_proposal(context, iteration=12)
    assert "Stale backend/CUDA/no-kernel repair gates in historical postmortems are superseded" in proposal_prompt


def test_mattergen_gate_sanitizer_does_not_promote_ranked_header_as_formula(tmp_path) -> None:
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "iteration": 20,
                        "status": "evaluated",
                        "candidate_spec": {"mattergen_requests": [{"backend": "mattergen"}]},
                        "evaluation_result": {"formula": "Rb3Cd4Br11", "e_hull": 0.02},
                    },
                    {
                        "iteration": 21,
                        "status": "strategy_blocked",
                        "xy_postmortem": {
                            "next_strategy": (
                                "Backend-gated ranked strict-SUN acquisition queue. "
                                "Stage 1 only after repair: MatterGen chemical-system conditioning only. "
                                "Ordered queue after audit: q1 Na2CuBr4, q2 Li2CuBr4, "
                                "q3 NaCu2Br5, q4 LiCu2Br5."
                            ),
                            "failure_boundaries": [
                                "Do not dispatch any MatterGen request until backend repair is explicitly confirmed.",
                                "Reject fixed-template proxy structures; after repair use MatterGen conditioning only.",
                            ],
                        },
                    },
                ]
            }
        )
    )

    context = sequential_context_payload(
        state={"current_round": 5, "history": [], "principle_book": []},
        mode="experience_xy",
        iteration=22,
        memory_path=memory_path,
        candidate_source="generator",
        seed=99,
        generator_backend="mattergen",
    )

    latest = context["latest_xy_strategy_constraints"]
    latest_text = json.dumps(latest).lower()
    assert latest["next_strategy_candidate_formulas"] == [
        "Na2CuBr4",
        "Li2CuBr4",
        "NaCu2Br5",
        "LiCu2Br5",
    ]
    assert "ranked" not in latest["next_strategy_candidate_formulas"]
    assert "after repair" not in latest_text
    assert "do not dispatch" not in latest_text


def test_sequential_context_exposes_latest_xy_strategy_constraints(tmp_path) -> None:
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "iteration": 7,
                        "evaluation_result": {"formula": "CaZrO3", "e_hull": 0.2077},
                        "xy_postmortem": {
                            "outcome_class": "high_e_hull",
                            "next_strategy": "Use a faithful Rb-Cd-Br mapping first; otherwise use Ba-based ABO3 with a new B-site.",
                            "failure_boundaries": [
                                {
                                    "boundary": "monotonic A-site contraction in alkaline-earth zirconates",
                                    "reason": "CaZrO3 failed after SrZrO3.",
                                }
                            ],
                        },
                    }
                ]
            }
        )
    )

    context = sequential_context_payload(
        state={"current_round": 5, "history": [], "principle_book": []},
        mode="experience_xy",
        iteration=8,
        memory_path=memory_path,
        candidate_source="generator",
        seed=101,
        generator_template_only=True,
    )

    latest = context["latest_xy_strategy_constraints"]
    assert latest["source_iteration"] == 7
    assert latest["source_formula"] == "CaZrO3"
    assert "Ba-based ABO3" in latest["next_strategy"]
    assert "failure_boundaries are reject conditions" in " ".join(latest["binding_policy"])
    assert "monotonic A-site contraction" in str(latest["failure_boundaries"])


def test_latest_strategy_keeps_fallback_formulas_for_no_valid_audit(tmp_path) -> None:
    next_strategy = (
        "Use this ordered duplicate-gated route. Candidate order: "
        "1. generator_template=perovskite; reduced_formula=RbCdBr3. "
        "2. generator_template=double_perovskite; reduced_formula=Rb2CdMgBr6. "
        "3. generator_template=double_perovskite; reduced_formula=Rb2CdZnBr6. "
        + "Keep these bromide audit details concise. " * 25
        + "4. If all Rb-Cd-Br candidates are blocked, pivot to compact oxide ternaries: "
        "generator_template=spinel; reduced_formula=MgCd2O4, then "
        "generator_template=spinel; reduced_formula=ZnCd2O4, then "
        "generator_template=delafossite; reduced_formula=NaCdO2."
    )
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "iteration": 7,
                        "xy_postmortem": {
                            "outcome_class": "weak_near_miss",
                            "next_strategy": next_strategy,
                        },
                    }
                ]
            }
        )
    )

    context = sequential_context_payload(
        state={"current_round": 5, "history": [], "principle_book": []},
        mode="experience_xy",
        iteration=8,
        memory_path=memory_path,
        candidate_source="generator",
        seed=101,
        generator_template_only=True,
    )

    latest = context["latest_xy_strategy_constraints"]
    assert latest["next_strategy_candidate_formulas"] == [
        "RbCdBr3",
        "Rb2MgCdBr6",
        "Rb2ZnCdBr6",
        "MgCd2O4",
        "ZnCd2O4",
        "NaCdO2",
    ]
    assert strategy_formula_candidates_from_text(next_strategy)[-3:] == [
        "MgCd2O4",
        "ZnCd2O4",
        "NaCdO2",
    ]


def test_latest_strategy_order_guard_rejects_skipping_first_candidate() -> None:
    context = {
        "latest_xy_strategy_constraints": {
            "next_strategy_candidate_formulas": ["SrSe", "BaSe"],
        },
        "controller_constraints": {"failed_or_used_reduced_formulas": []},
    }
    review = {"agree": True, "approved": True, "overall_reasoning_summary": "approve"}

    guarded = enforce_latest_strategy_selection_order_guard(
        review,
        rocksalt_material_payload("Ba"),
        context,
    )

    assert guarded["agree"] is False
    assert guarded["approved"] is False
    assert "select SrSe first" in guarded["required_revision"]
    assert guarded["controller_strategy_order_errors"]


def test_latest_strategy_order_guard_accepts_first_candidate() -> None:
    context = {
        "latest_xy_strategy_constraints": {
            "next_strategy_candidate_formulas": ["SrSe", "BaSe"],
        },
        "controller_constraints": {"failed_or_used_reduced_formulas": []},
    }
    review = {"agree": True, "approved": True}

    guarded = enforce_latest_strategy_selection_order_guard(
        review,
        rocksalt_material_payload("Sr"),
        context,
    )

    assert guarded["agree"] is True
    assert guarded["approved"] is True


def test_latest_strategy_order_guard_uses_mattergen_queue_formula() -> None:
    context = {
        "generator_backend": "mattergen",
        "latest_xy_strategy_constraints": {
            "next_strategy_candidate_formulas": ["Rb3Mn4Br11", "Rb3Mn2Br7"],
        },
        "controller_constraints": {"failed_or_used_reduced_formulas": []},
    }
    proposal = {
        "status": "material_description_proposal",
        "sun_candidate_queue": [
            {
                "candidate_id": "q1",
                "rank": 1,
                "reduced_formula": "Rb3Mn4Br11",
                "acquisition_mode": "exploit",
                "material_description": {"generator_template": "mattergen"},
            },
            {
                "candidate_id": "q2",
                "rank": 2,
                "reduced_formula": "Rb3Mn2Br7",
                "acquisition_mode": "exploit",
                "material_description": {"generator_template": "mattergen"},
            },
        ],
        "selected_candidate_id": "q2",
        "selection_rationale": "incorrectly skip q1",
        "material_description": {
            "generator_template": "mattergen",
            "preferred_reduced_formula": "Rb3Mn2Br7",
        },
    }

    assert selected_reduced_formula_from_payload(proposal) == "Rb3Mn2Br7"
    errors = latest_strategy_selection_order_errors(proposal, context)

    assert errors
    assert "select Rb3Mn4Br11 first" in errors[0]


def test_latest_strategy_order_guard_rejects_singleton_after_generator_feedback() -> None:
    context = {
        "latest_xy_strategy_constraints": {
            "next_strategy_candidate_formulas": ["SrSe", "BaSe"],
        },
        "controller_constraints": {"failed_or_used_reduced_formulas": []},
    }
    feedback = {
        "materialization_errors": [
            "xy_candidate_001: formula_probes[0] generated BaSe with template rocksalt "
            "but failed structure validation: volume_per_atom_too_large"
        ]
    }

    errors = latest_strategy_selection_order_errors(
        rocksalt_material_payload("Sr"),
        context,
        feedback,
    )

    assert errors
    assert "fewer than two currently viable formulas" in errors[0]


def test_latest_strategy_sanitizes_seeded_duplicate_postmortem_formulas(tmp_path) -> None:
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "records": [
                    {"iteration": 1, "evaluation_result": {"formula": "BaTiO3"}},
                    {"iteration": 2, "evaluation_result": {"formula": "KNbO3"}},
                    {
                        "iteration": 3,
                        "evaluation_result": {"formula": "YAlO3", "e_hull": 0.13},
                        "xy_postmortem": {
                            "outcome_class": "high_e_hull",
                            "next_strategy": (
                                "Use d0 perovskites: "
                                "1. reduced_formula=BaTiO3. "
                                "2. reduced_formula=PbTiO3. "
                                "3. reduced_formula=PbZrO3. "
                                "4. reduced_formula=KNbO3."
                            ),
                            "failure_boundaries": [],
                        },
                    },
                ]
            }
        )
    )

    context = sequential_context_payload(
        state={"current_round": 5, "history": [], "principle_book": []},
        mode="experience_xy",
        iteration=4,
        memory_path=memory_path,
        candidate_source="generator",
        seed=101,
        generator_template_only=True,
    )

    latest = context["latest_xy_strategy_constraints"]
    assert "BaTiO3" not in latest["next_strategy"]
    assert "KNbO3" not in latest["next_strategy"]
    assert "PbTiO3" in latest["next_strategy"]
    assert "PbZrO3" in latest["next_strategy"]
    assert latest["next_strategy_candidate_formulas"] == ["TiPbO3", "ZrPbO3"]
    assert any("BaTiO3" in item for item in latest["controller_postmortem_audit_errors"])
    assert latest["controller_postmortem_audit_error_total"] >= 2


def test_sequential_context_compacts_failed_or_used_formula_constraints(tmp_path) -> None:
    memory_path = tmp_path / "memory.json"
    records = [
        {"iteration": index, "evaluation_result": {"formula": f"Li{index}O"}}
        for index in range(1, 211)
    ]
    records.append({"iteration": 211, "evaluation_result": {"formula": "YAlO3"}})
    memory_path.write_text(json.dumps({"records": records}))

    context = sequential_context_payload(
        state={"current_round": 5, "history": [], "principle_book": []},
        mode="experience_xy",
        iteration=212,
        memory_path=memory_path,
        candidate_source="generator",
        seed=101,
        generator_template_only=True,
    )
    populate_sequential_controller_constraints(context, read_sequential_memory(memory_path))

    constraints = context["controller_constraints"]
    formulas = constraints["failed_or_used_reduced_formulas"]
    assert constraints["failed_or_used_reduced_formula_total"] > len(formulas)
    assert constraints["failed_or_used_reduced_formula_omitted_count"] > 0
    assert len(formulas) <= 160
    assert "YAlO3" in formulas
    assert "full hidden failed/used formula set" in constraints["failed_or_used_reduced_formula_context_policy"]


def test_failed_formula_hints_include_active_mattergen_system_history(tmp_path) -> None:
    memory_path = tmp_path / "memory.json"
    records = [
        {"iteration": index, "evaluation_result": {"formula": f"Li{index}O"}}
        for index in range(1, 80)
    ]
    records.extend(
        [
            {"iteration": 80, "evaluation_result": {"formula": "Rb2MnBr4", "e_hull": 0.024}},
            {"iteration": 81, "evaluation_result": {"formula": "RbMnBr3", "e_hull": 0.011}},
            {"iteration": 82, "evaluation_result": {"formula": "Rb2MnBr3", "e_hull": 0.095}},
        ]
    )
    memory_path.write_text(json.dumps({"records": records}))
    context = sequential_context_payload(
        state={"current_round": 5, "history": [], "principle_book": []},
        mode="experience_xy",
        iteration=83,
        memory_path=memory_path,
        candidate_source="generator",
        seed=101,
        generator_backend="mattergen",
        xy_sun_candidate_queue_size=4,
    )

    populate_sequential_controller_constraints(context, read_sequential_memory(memory_path))

    hints = context["controller_constraints"]["failed_or_used_reduced_formula_hints"]
    assert "Rb2MnBr4" in hints
    assert "RbMnBr3" in hints
    assert len(hints) <= 32


def test_sequential_context_omits_full_principle_book_in_favor_of_rag(tmp_path) -> None:
    state = {
        "current_round": 5,
        "history": [],
        "principle_book": [
            {
                "program_id": f"principle_program_{index:03d}",
                "status": "validated_principle",
                "topic_key": f"topic-{index}",
                "principle_statement": "long principle " * 200,
                "micro_mechanism": "long mechanism " * 200,
                "reasoning_chain": ["long reasoning " * 100],
                "boundaries": ["long boundary " * 100],
                "residual_risks": ["long risk " * 100],
            }
            for index in range(30)
        ],
    }

    context = sequential_context_payload(
        state=state,
        mode="experience_xy",
        iteration=3,
        memory_path=tmp_path / "memory.json",
        candidate_source="generator",
        seed=99,
        generator_template_only=True,
    )

    assert "principle_book" not in context
    assert context["principle_book_index_tail"] == []
    assert "long principle" not in str(context)


def test_xy_search_policy_scores_exploit_basins_and_cools_recent_failures() -> None:
    records = [
        xy_eval_record(1, "LaH2", "fluorite", "H", -0.001, is_sun=True),
        xy_eval_record(2, "GdH2", "fluorite", "H", 0.025),
        xy_eval_record(3, "YH2", "fluorite", "H", 0.026),
        xy_eval_record(4, "GdS", "rocksalt", "S", 0.011),
        xy_eval_record(5, "Ba2CaOsO6", "double_perovskite", "O", 0.278),
        xy_eval_record(6, "Ba2CoMoO6", "double_perovskite", "O", 0.102),
        xy_eval_record(7, "Ca2ZnWO6", "double_perovskite", "O", 0.194),
        xy_eval_record(8, "Sr2LaTaO6", "double_perovskite", "O", 0.331),
    ]

    policy = xy_search_policy_from_memory({"records": records}, iteration=11)

    assert policy["current_search_mode"] == "exploit"
    assert "fluorite" in policy["exploit_template_allowlist"]
    assert "rocksalt" in policy["exploit_template_allowlist"]
    assert "double_perovskite" in policy["cooled_templates"]
    assert any(item["basin_key"] == "template=fluorite;X=H" for item in policy["preferred_exploitation_basins"])
    assert policy["budget"] == {"exploit": 0.70, "explore_adjacent": 0.20, "orthogonal_jump": 0.10}


def test_sequential_context_injects_xy_search_policy(tmp_path) -> None:
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "records": [
                    xy_eval_record(1, "LaH2", "fluorite", "H", -0.001, is_sun=True),
                    xy_eval_record(2, "GdH2", "fluorite", "H", 0.025),
                ]
            }
        )
    )

    context = sequential_context_payload(
        state={"current_round": 5, "history": [], "principle_book": []},
        mode="experience_xy",
        iteration=11,
        memory_path=memory_path,
        candidate_source="generator",
        seed=101,
        generator_template_only=True,
    )

    policy = context["controller_constraints"]["search_policy"]
    assert policy["schema_version"] == "xy_search_policy.v1"
    assert policy["current_search_mode"] == "exploit"
    assert "preferred_exploitation_basins" in policy


def test_populate_constraints_injects_machine_strategy_constraints(tmp_path) -> None:
    memory = {
        "records": [
            {"iteration": 1, "evaluation_result": {"formula": "Rb2MnBr5", "e_hull": -0.01}},
            {
                "iteration": 2,
                "evaluation_result": {"formula": "RbMnBr4", "e_hull": -0.14},
                "xy_postmortem": {
                    "outcome_class": "sun",
                    "next_strategy": (
                        "Use this ordered queue: 1) Rb2MnBr5, 2) Rb3Mn4Br11, "
                        "3) Rb3Mn2Br7."
                    ),
                    "failure_boundaries": ["Do not repeat RbMnBr4."],
                },
            },
        ]
    }
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(json.dumps(memory))
    context = sequential_context_payload(
        state={"current_round": 5, "history": [], "principle_book": []},
        mode="experience_xy",
        iteration=3,
        memory_path=memory_path,
        candidate_source="generator",
        seed=101,
        generator_backend="mattergen",
        xy_sun_candidate_queue_size=4,
    )

    populate_sequential_controller_constraints(context, read_sequential_memory(memory_path))

    strategy_constraints = context["controller_constraints"]["strategy_constraints"]
    assert strategy_constraints["schema_version"] == "xy_strategy_constraints.v1"
    assert strategy_constraints["binding"] is True
    assert strategy_constraints["first_required_formula"] == "Rb3Mn4Br11"
    assert strategy_constraints["ordered_candidate_formulas"] == ["Rb3Mn4Br11", "Rb3Mn2Br7"]
    assert strategy_constraints["legal_ordered_candidate_formulas"] == ["Rb3Mn4Br11", "Rb3Mn2Br7"]


def test_populate_strategy_constraints_excludes_failed_formula_from_legal_order() -> None:
    memory = {
        "records": [
            {"iteration": 1, "evaluation_result": {"formula": "Rb2MnBr5", "e_hull": 0.04}},
        ]
    }
    context = {
        "generator_backend": "mattergen",
        "latest_xy_strategy_constraints": {
            "next_strategy_candidate_formulas": [
                "RbMn4Br9",
                "Rb2Mn5Br12",
                "Rb3Mn2Br7",
                "RbMn2Br5",
            ],
            "failure_boundaries": ["Do not propose future candidates with blocked formula RbMn2Br5."],
        },
        "controller_constraints": {
            "xy_sun_candidate_queue_size": 4,
            "search_policy": {
                "backend": "mattergen",
                "current_search_mode": "exploit",
            },
        },
    }

    populate_sequential_controller_constraints(context, memory)

    strategy_constraints = context["controller_constraints"]["strategy_constraints"]
    assert strategy_constraints["first_required_formula"] == "RbMn4Br9"
    assert strategy_constraints["ordered_candidate_formulas"] == [
        "RbMn4Br9",
        "Rb2Mn5Br12",
        "Rb3Mn2Br7",
        "RbMn2Br5",
    ]
    assert strategy_constraints["legal_ordered_candidate_formulas"] == [
        "RbMn4Br9",
        "Rb2Mn5Br12",
        "Rb3Mn2Br7",
    ]
    assert strategy_constraints["blocked_candidate_formula_reasons"]["RbMn2Br5"] == [
        "blocked_by_latest_failure_boundaries"
    ]


def test_latest_strategy_constraints_preserve_and_block_later_failure_boundaries() -> None:
    memory = {
        "records": [
            {
                "iteration": 1,
                "evaluation_result": {"formula": "Rb3Mn2Br7", "e_hull": 0.0108},
                "xy_postmortem": {
                    "outcome_class": "near_miss",
                    "next_strategy": "Use candidates: Rb5Mn2Br9, Rb4MnBr6, Rb7Mn2Br11.",
                    "failure_boundaries": [
                        "Do not repeat evaluated Rb3Mn2Br7.",
                        "Do not repeat used soft target Rb4MnBr6.",
                    ],
                },
            }
        ]
    }
    context = {
        "generator_backend": "mattergen",
        "latest_xy_strategy_constraints": {},
        "controller_constraints": {
            "xy_sun_candidate_queue_size": 4,
            "search_policy": {
                "backend": "mattergen",
                "current_search_mode": "exploit",
            },
        },
    }
    from crystal_llm.run_xy_experience_debate import latest_xy_strategy_constraints_from_memory

    context["latest_xy_strategy_constraints"] = latest_xy_strategy_constraints_from_memory(memory)
    populate_sequential_controller_constraints(context, memory)

    latest = context["latest_xy_strategy_constraints"]
    assert any("Rb4MnBr6" in item for item in latest["failure_boundaries"])
    strategy_constraints = context["controller_constraints"]["strategy_constraints"]
    assert "Rb4MnBr6" not in strategy_constraints["ordered_candidate_formulas"]
    assert "Rb4MnBr6" not in strategy_constraints["legal_ordered_candidate_formulas"]
    assert strategy_constraints["legal_ordered_candidate_formulas"] == ["Rb5Mn2Br9", "Rb7Mn2Br11"]


def test_sequential_xy_prompts_make_latest_strategy_boundaries_binding(tmp_path) -> None:
    context = {
        "generation_protocol": "sequential_single",
        "latest_xy_strategy_constraints": {
            "next_strategy": "First try faithful Rb-Cd-Br; fallback only to Ba-based ABO3 with a new B-site.",
            "failure_boundaries": ["avoid monotonic A-site contraction to Sr/Ca alkaline-earth perovskites"],
        },
        "controller_constraints": {
            "control_candidate_requested": False,
            "generator_template_only": True,
            "allowed_generator_templates": ["perovskite"],
        },
    }
    proposal = {
        "status": "material_description_proposal",
        "agent": "X",
        "material_description": {
            "natural_language_description": "BaSnO3 oxide perovskite",
            "generator_template": "perovskite",
        },
    }
    review = {"agree": False, "approved": False, "required_revision": "follow latest strategy"}

    prompts = [
        prompt_x_sequential_material_proposal(context, iteration=8, template_only=True),
        prompt_y_sequential_material_review(context, proposal, iteration=8, cycle=1, template_only=True),
        prompt_y_sequential_material_counterproposal(context, proposal, review, iteration=8, cycle=1, template_only=True),
        prompt_x_sequential_material_reverse_review(context, proposal, proposal, iteration=8, cycle=1, template_only=True),
    ]

    for prompt in prompts:
        assert "BINDING_XY_STRATEGY_BOUNDARY_POLICY" in prompt
        assert "search_policy" in prompt
        assert "70/20/10" in prompt
        assert "supersedes" in prompt
        assert "latest_xy_strategy_constraints" in prompt
        assert "control_candidate_requested" in prompt
        assert "no_valid_material" in prompt
    assert "You must reject" in prompts[1]
    assert "Your counterproposal must obey" in prompts[2]
    assert "impossibility_certificate" in prompts[2]


def test_y_counterproposal_prompt_omits_review_embedded_counterproposal() -> None:
    context = {
        "generation_protocol": "sequential_single",
        "latest_xy_strategy_constraints": {
            "next_strategy": (
                "Controller-sanitized next_strategy: named candidate list is exhausted; "
                "same validated basin may continue with two new formulas."
            ),
            "next_strategy_candidate_formulas": [],
            "failure_boundaries": ["avoid Cs-Hg-Br repairs"],
        },
        "controller_constraints": {
            "control_candidate_requested": False,
            "search_policy": {"current_search_mode": "exploit"},
            "failed_or_used_reduced_formulas": ["RbCdBr3"],
        },
    }
    proposal = {
        "status": "material_description_proposal",
        "agent": "X",
        "sun_candidate_queue": [
            {"candidate_id": "q1", "reduced_formula": "Rb3Cd2Br7"},
            {"candidate_id": "q2", "reduced_formula": "Rb4Cd3Br10"},
        ],
        "selected_candidate_id": "q1",
        "material_description": {
            "natural_language_description": "Rb-Cd-Br dense bromide",
            "generator_template": "mattergen",
        },
    }
    review = {
        "agree": False,
        "approved": False,
        "required_revision": "audit two non-duplicate formulas in the validated basin",
        "counterproposal_material_description": {
            "status": "material_description_proposal",
            "sun_candidate_queue": [
                {"candidate_id": "q1", "reduced_formula": "ZrAs"},
                {"candidate_id": "q2", "reduced_formula": "HfAs"},
            ],
            "selected_candidate_id": "q1",
            "material_description": {"natural_language_description": "ZrAs rocksalt pnictide"},
        },
    }

    prompt = prompt_y_sequential_material_counterproposal(
        context,
        proposal,
        review,
        iteration=3,
        cycle=1,
    )

    assert "audit two non-duplicate formulas" in prompt
    assert "Rb3Cd2Br7" in prompt
    assert "ZrAs" not in prompt
    assert "HfAs" not in prompt


def test_search_policy_supersedes_stale_latest_strategy_order() -> None:
    context = {
        "latest_xy_strategy_constraints": {
            "next_strategy": (
                "Continue the bromide perovskite route: first CsMnBr3, then KMnBr3, "
                "then CsFeBr3, then KFeBr3."
            ),
            "next_strategy_candidate_formulas": ["CsMnBr3", "KMnBr3", "CsFeBr3", "KFeBr3"],
            "failure_boundaries": ["perovskite A=Cs/K with X=Br is volume-blocked"],
        },
        "controller_constraints": {
            "search_policy": {
                "current_search_mode": "exploit",
                "exploit_template_allowlist": ["fluorite", "rocksalt"],
                "cooled_templates": ["perovskite"],
            }
        },
    }
    proposal = {
        "material_description": {
            "generator_template": "fluorite",
            "generator_role_mapping": {
                "A": {"element": "Ce", "oxidation_state": 2},
                "X": {"element": "H", "oxidation_state": -1},
            },
            "expected_local_motif": "native CaF2-type hydride coordination",
            "why_template_is_faithful": "fluorite directly realizes the intended hydride motif",
        }
    }

    assert latest_strategy_search_policy_supersedes(context) is True
    assert latest_strategy_selection_order_errors(proposal, context) == []


def test_no_valid_cannot_terminate_when_search_policy_supersedes_stale_route() -> None:
    context = {
        "latest_xy_strategy_constraints": {
            "next_strategy": "Try perovskite CsMnBr3, KMnBr3, CsFeBr3, KFeBr3.",
            "next_strategy_candidate_formulas": ["CsMnBr3", "KMnBr3", "CsFeBr3", "KFeBr3"],
            "failure_boundaries": ["perovskite bromides are blocked"],
        },
        "controller_constraints": {
            "search_policy": {
                "current_search_mode": "exploit",
                "exploit_template_allowlist": ["fluorite", "rocksalt"],
                "cooled_templates": ["perovskite"],
            }
        },
    }
    proposal = {
        "status": "no_valid_material_description",
        "material_description": {},
        "impossibility_certificate": {
            "audited_formulas": ["CsMnBr3", "KMnBr3", "CsFeBr3", "KFeBr3"],
            "reason": "All bromide perovskite candidates are blocked.",
        },
    }
    review = {"agree": True, "approved": True, "overall_reasoning_summary": "old route blocked"}

    guarded = enforce_no_valid_strategy_audit_guard(review, proposal, context)

    assert sequential_no_valid_description_agrees_with_context(review, proposal, context) is False
    assert guarded["agree"] is False
    assert "search_policy supersedes" in guarded["required_revision"]


def test_no_valid_guard_does_not_reject_normal_search_policy_proposal() -> None:
    context = {
        "latest_xy_strategy_constraints": {
            "next_strategy": "Try perovskite CsMnBr3, KMnBr3, CsFeBr3, KFeBr3.",
            "next_strategy_candidate_formulas": ["CsMnBr3", "KMnBr3", "CsFeBr3", "KFeBr3"],
            "failure_boundaries": ["perovskite bromides are blocked"],
        },
        "controller_constraints": {
            "search_policy": {
                "current_search_mode": "exploit",
                "exploit_template_allowlist": ["fluorite", "rocksalt"],
                "cooled_templates": ["perovskite"],
            }
        },
    }
    proposal = {
        "status": "material_description_proposal",
        "material_description": {
            "generator_template": "fluorite",
            "generator_role_mapping": {
                "A": {"element": "Er", "oxidation_state": 2},
                "X": {"element": "H", "oxidation_state": -1},
            },
            "expected_local_motif": "fluorite hydride coordination",
            "why_template_is_faithful": "fluorite directly realizes the hydride motif",
        },
    }
    review = {"agree": True, "approved": True, "overall_reasoning_summary": "current-policy proposal is valid"}

    guarded = enforce_no_valid_strategy_audit_guard(review, proposal, context)

    assert guarded["agree"] is True
    assert guarded["approved"] is True
    assert "required_revision" not in guarded


def test_sequential_xy_prompts_require_ranked_sun_candidate_queue(tmp_path) -> None:
    context = sequential_context_payload(
        state={"current_round": 1, "principle_book": []},
        mode="experience_xy",
        iteration=3,
        memory_path=tmp_path / "memory.json",
        candidate_source="generator",
        seed=7,
        generator_template_only=True,
        xy_sun_candidate_queue_size=4,
    )
    proposal = {
        "status": "material_description_proposal",
        "agent": "X",
        "iteration": 3,
        "sun_candidate_queue": [
            {
                "candidate_id": "q1",
                "rank": 1,
                "reduced_formula": "EuO",
                "material_description": {
                    "natural_language_description": "rocksalt EuO",
                    "target_family": "rocksalt monoxide",
                    "expected_local_motif": "EuO6 octahedra",
                    "generator_template": "rocksalt",
                    "generator_role_mapping": {
                        "A": {"element": "Eu", "oxidation_state": 2},
                        "X": {"element": "O", "oxidation_state": -2},
                    },
                    "template_formula_family": "AX rocksalt",
                    "why_template_is_faithful": "rocksalt directly realizes six-coordinate EuO",
                },
            },
            {
                "candidate_id": "q2",
                "rank": 2,
                "reduced_formula": "YbO",
                "material_description": {
                    "natural_language_description": "rocksalt YbO",
                    "target_family": "rocksalt monoxide",
                    "expected_local_motif": "YbO6 octahedra",
                    "generator_template": "rocksalt",
                    "generator_role_mapping": {
                        "A": {"element": "Yb", "oxidation_state": 2},
                        "X": {"element": "O", "oxidation_state": -2},
                    },
                    "template_formula_family": "AX rocksalt",
                    "why_template_is_faithful": "rocksalt directly realizes six-coordinate YbO",
                },
            },
        ],
        "selected_candidate_id": "q1",
        "selection_rationale": "EuO is the best legal strict-SUN bet.",
        "material_description": {
            "natural_language_description": "rocksalt EuO",
            "target_family": "rocksalt monoxide",
            "expected_local_motif": "EuO6 octahedra",
            "generator_template": "rocksalt",
            "generator_role_mapping": {
                "A": {"element": "Eu", "oxidation_state": 2},
                "X": {"element": "O", "oxidation_state": -2},
            },
            "template_formula_family": "AX rocksalt",
            "why_template_is_faithful": "rocksalt directly realizes six-coordinate EuO",
        },
    }

    x_prompt = prompt_x_sequential_material_proposal(context, iteration=3, template_only=True)
    y_prompt = prompt_y_sequential_material_review(context, proposal, iteration=3, cycle=1, template_only=True)
    compact = compact_material_description_for_prompt(proposal)

    assert "ranked queue" in x_prompt
    assert "strict SUN probability" in x_prompt
    assert "selected_candidate_id" in x_prompt
    assert "ranked SUN candidate queue" in y_prompt
    assert compact["sun_candidate_queue"][0]["candidate_id"] == "q1"
    assert compact["selected_candidate_id"] == "q1"


def test_review_prompt_compaction_preserves_full_visible_queue() -> None:
    payload = {
        "status": "material_description_proposal",
        "agent": "Y",
        "iteration": 2,
        "selected_candidate_id": "q1",
        "sun_candidate_queue": [
            {
                "candidate_id": "q1",
                "rank": 1,
                "reduced_formula": "Rb3Mn4Br11",
                "acquisition_mode": "exploit",
                "expected_e_hull_band": "-0.04 to +0.02",
                "material_description": {
                    "natural_language_description": "Rb-Mn-Br candidate 1",
                    "generator_template": "mattergen",
                },
            },
            {
                "candidate_id": "q2",
                "rank": 2,
                "reduced_formula": "Rb2Mn3Br8",
                "acquisition_mode": "exploit",
                "expected_e_hull_band": "-0.02 to +0.04",
                "material_description": {
                    "natural_language_description": "Rb-Mn-Br candidate 2",
                    "generator_template": "mattergen",
                },
            },
            {
                "candidate_id": "q3",
                "rank": 3,
                "reduced_formula": "Rb3Mn2Br7",
                "acquisition_mode": "exploit",
                "expected_e_hull_band": "-0.01 to +0.06",
                "material_description": {
                    "natural_language_description": "Rb-Mn-Br candidate 3",
                    "generator_template": "mattergen",
                },
            },
            {
                "candidate_id": "q4",
                "rank": 4,
                "reduced_formula": "Rb4MnBr6",
                "acquisition_mode": "exploit",
                "expected_e_hull_band": "0.00 to +0.08",
                "material_description": {
                    "natural_language_description": "Rb-Mn-Br candidate 4",
                    "generator_template": "mattergen",
                },
            },
        ],
        "material_description": {
            "natural_language_description": "selected Rb3Mn4Br11",
            "generator_template": "mattergen",
        },
    }

    compact = compact_material_description_for_review_prompt(payload)

    assert compact["sun_candidate_queue_count"] == 4
    assert [item["reduced_formula"] for item in compact["sun_candidate_queue"]] == [
        "Rb3Mn4Br11",
        "Rb2Mn3Br8",
        "Rb3Mn2Br7",
        "Rb4MnBr6",
    ]
    assert [item["reduced_formula"] for item in compact["sun_candidate_queue_formula_order"]] == [
        "Rb3Mn4Br11",
        "Rb2Mn3Br8",
        "Rb3Mn2Br7",
        "Rb4MnBr6",
    ]


def test_validate_xy_sun_candidate_queue_rejects_missing_queue() -> None:
    context = {"controller_constraints": {"xy_sun_candidate_queue_size": 4}}
    payload = {
        "status": "material_description_consensus",
        "selected_candidate_id": "q1",
        "selection_rationale": "single candidate",
        "material_description": {
            "generator_template": "rocksalt",
            "generator_role_mapping": {
                "A": {"element": "Eu", "oxidation_state": 2},
                "X": {"element": "O", "oxidation_state": -2},
            },
        },
    }
    errors = validate_xy_sun_candidate_queue(payload, context=context)
    assert any("sun_candidate_queue" in error for error in errors)


def test_material_consensus_inherits_queue_from_approved_proposal() -> None:
    context = {"controller_constraints": {"xy_sun_candidate_queue_size": 4}}
    q1_material = {
        "generator_template": "rocksalt",
        "generator_role_mapping": {
            "A": {"element": "Eu", "oxidation_state": 2},
            "X": {"element": "O", "oxidation_state": -2},
        },
    }
    q2_material = {
        "generator_template": "rocksalt",
        "generator_role_mapping": {
            "A": {"element": "Sr", "oxidation_state": 2},
            "X": {"element": "O", "oxidation_state": -2},
        },
    }
    proposal = {
        "status": "material_description_proposal",
        "sun_candidate_queue": [
            {"candidate_id": "q1", "reduced_formula": "EuO", "material_description": q1_material},
            {"candidate_id": "q2", "reduced_formula": "SrO", "material_description": q2_material},
        ],
        "selected_candidate_id": "q1",
        "selection_rationale": "q1 is the first legal candidate in the approved queue.",
        "material_description": q1_material,
    }
    compact_consensus = {
        "status": "material_description_consensus",
        "material_description": q1_material,
    }

    carried = xy_debate.material_consensus_with_source_queue(compact_consensus, proposal)

    assert carried["selected_candidate_id"] == "q1"
    assert [item["reduced_formula"] for item in carried["sun_candidate_queue"]] == ["EuO", "SrO"]
    assert validate_xy_sun_candidate_queue(carried, context=context) == []


def test_material_consensus_fills_selected_queue_item_material_description() -> None:
    context = {"controller_constraints": {"xy_sun_candidate_queue_size": 4}}
    selected_material = {
        "generator_template": "mattergen",
        "preferred_reduced_formula": "Na2CuBr4",
        "generator_role_mapping": {
            "A": {"element": "Na", "oxidation_state": 1},
            "M": {"element": "Cu", "oxidation_state": 2},
            "X": {"element": "Br", "oxidation_state": -1},
        },
        "natural_language_description": "MatterGen Na-Cu-Br bromocuprate soft target.",
    }
    proposal = {
        "status": "material_description_proposal",
        "sun_candidate_queue": [
            {"candidate_id": "q1", "reduced_formula": "Na2CuBr4"},
            {"candidate_id": "q2", "reduced_formula": "Li2CuBr4"},
        ],
        "selected_candidate_id": "q1",
        "selection_rationale": "q1 is the first legal MatterGen candidate.",
        "material_description": selected_material,
    }
    compact_consensus = {
        "status": "material_description_consensus",
        "sun_candidate_queue": [
            {"candidate_id": "q1", "reduced_formula": "Na2CuBr4"},
            {"candidate_id": "q2", "reduced_formula": "Li2CuBr4"},
        ],
        "selected_candidate_id": "q1",
        "selection_rationale": "q1 is the first legal MatterGen candidate.",
        "material_description": selected_material,
    }

    carried = xy_debate.material_consensus_with_source_queue(compact_consensus, proposal)

    assert carried["sun_candidate_queue"][0]["material_description"]["preferred_reduced_formula"] == "Na2CuBr4"
    assert validate_xy_sun_candidate_queue(carried, context=context) == []


def test_payload_with_controller_iteration_overrides_model_iteration() -> None:
    payload = {
        "status": "material_description_review",
        "agent": "Y",
        "iteration": 99,
        "counterproposal_material_description": {
            "status": "material_description_proposal",
            "agent": "X",
            "iteration": 100,
        },
    }

    normalized = xy_debate.payload_with_controller_iteration(payload, iteration=4)

    assert normalized["iteration"] == 4
    assert normalized["counterproposal_material_description"]["iteration"] == 4


def test_approved_material_consensus_from_payload_finalizes_source_queue_without_llm() -> None:
    context = {"controller_constraints": {"xy_sun_candidate_queue_size": 4}}
    q1_material = {
        "generator_template": "mattergen",
        "preferred_reduced_formula": "Rb4Cd3Br10",
        "generator_role_mapping": {
            "A": {"element": "Rb", "oxidation_state": 1},
            "M": {"element": "Cd", "oxidation_state": 2},
            "X": {"element": "Br", "oxidation_state": -1},
        },
        "natural_language_description": "MatterGen Rb-Cd-Br soft bromide target.",
    }
    proposal = {
        "status": "material_description_proposal",
        "agent": "X",
        "iteration": 99,
        "sun_candidate_queue": [
            {"candidate_id": "q1", "rank": 1, "reduced_formula": "Rb4Cd3Br10", "material_description": q1_material},
            {"candidate_id": "q2", "rank": 2, "reduced_formula": "Rb3Cd2Br7"},
        ],
        "selected_candidate_id": "q1",
        "selection_rationale": "q1 is the best strict-SUN bet.",
        "material_description": q1_material,
        "proposal_summary": "Rb-Cd-Br queue.",
    }
    review = {
        "status": "material_description_review",
        "agent": "Y",
        "iteration": 99,
        "agree": True,
        "approved": True,
        "overall_reasoning_summary": "Approve q1 after queue audit.",
        "risk_audit": {"ranked_queue_present": True},
    }

    consensus = xy_debate.approved_material_consensus_from_payload(proposal, review, iteration=5)

    assert consensus["status"] == "material_description_consensus"
    assert consensus["agent"] == "Y"
    assert consensus["iteration"] == 5
    assert consensus["selected_candidate_id"] == "q1"
    assert consensus["debate_summary"] == "Approve q1 after queue audit."
    assert consensus["final_review_risk_audit"] == {"ranked_queue_present": True}
    assert [item["reduced_formula"] for item in consensus["sun_candidate_queue"]] == ["Rb4Cd3Br10", "Rb3Cd2Br7"]
    assert validate_xy_sun_candidate_queue(consensus, context=context) == []


def test_compact_review_treats_null_sun_candidate_queue_as_missing() -> None:
    review = {
        "status": "reject",
        "counterproposal_material_description": {
            "status": "proposal",
            "agent": "X",
            "sun_candidate_queue": None,
            "selected_candidate_id": "q1",
            "material_description": {
                "generator_template": "rocksalt",
                "generator_role_mapping": {
                    "A": {"element": "Sr", "oxidation_state": 2},
                    "X": {"element": "F", "oxidation_state": -1},
                },
            },
        },
    }

    compact = compact_review_for_prompt(review)

    assert compact["counterproposal_material_description"]["selected_candidate_id"] == "q1"
    assert "sun_candidate_queue" not in compact["counterproposal_material_description"]


def test_validate_xy_sun_candidate_queue_accepts_selected_queue_item() -> None:
    context = {"controller_constraints": {"xy_sun_candidate_queue_size": 4}}
    payload = {
        "status": "material_description_consensus",
        "sun_candidate_queue": [
            {
                "candidate_id": "q1",
                "material_description": {
                    "generator_template": "rocksalt",
                    "generator_role_mapping": {
                        "A": {"element": "Eu", "oxidation_state": 2},
                        "X": {"element": "O", "oxidation_state": -2},
                    },
                },
            },
            {
                "candidate_id": "q2",
                "material_description": {
                    "generator_template": "rocksalt",
                    "generator_role_mapping": {
                        "A": {"element": "Yb", "oxidation_state": 2},
                        "X": {"element": "O", "oxidation_state": -2},
                    },
                },
            },
        ],
        "selected_candidate_id": "q1",
        "selection_rationale": "q1 is the strongest legal SUN bet.",
        "material_description": {
            "generator_template": "rocksalt",
            "generator_role_mapping": {
                "A": {"element": "Eu", "oxidation_state": 2},
                "X": {"element": "O", "oxidation_state": -2},
            },
        },
    }
    assert validate_xy_sun_candidate_queue(payload, context=context) == []


def test_validate_xy_sun_candidate_queue_accepts_selected_formula_with_top_level_description() -> None:
    context = {"controller_constraints": {"xy_sun_candidate_queue_size": 4}}
    payload = {
        "status": "material_description_consensus",
        "sun_candidate_queue": [
            {"candidate_id": "q1", "reduced_formula": "TbGaO3"},
            {"candidate_id": "q2", "reduced_formula": "YbAlO3"},
        ],
        "selected_candidate_id": "q1",
        "selection_rationale": "q1 is the strongest legal SUN bet.",
        "material_description": {
            "generator_template": "perovskite",
            "generator_role_mapping": {
                "A": {"element": "Tb", "oxidation_state": 3},
                "B": {"element": "Ga", "oxidation_state": 3},
                "X": {"element": "O", "oxidation_state": -2},
            },
        },
    }

    assert validate_xy_sun_candidate_queue(payload, context=context) == []


def test_validate_xy_sun_candidate_queue_requires_two_legal_after_sanitation() -> None:
    context = {
        "controller_constraints": {
            "xy_sun_candidate_queue_size": 4,
            "failed_or_used_reduced_formulas": ["YbO"],
        }
    }
    payload = {
        "status": "material_description_consensus",
        "sun_candidate_queue": [
            {
                "candidate_id": "q1",
                "material_description": {
                    "generator_template": "rocksalt",
                    "generator_role_mapping": {
                        "A": {"element": "Eu", "oxidation_state": 2},
                        "X": {"element": "O", "oxidation_state": -2},
                    },
                },
            },
            {
                "candidate_id": "q2",
                "reduced_formula": "YbO",
                "material_description": {
                    "generator_template": "rocksalt",
                    "generator_role_mapping": {
                        "A": {"element": "Yb", "oxidation_state": 2},
                        "X": {"element": "O", "oxidation_state": -2},
                    },
                },
            },
        ],
        "selected_candidate_id": "q1",
        "selection_rationale": "q1 is the strongest legal SUN bet.",
        "material_description": {
            "generator_template": "rocksalt",
            "generator_role_mapping": {
                "A": {"element": "Eu", "oxidation_state": 2},
                "X": {"element": "O", "oxidation_state": -2},
            },
        },
    }

    errors = validate_xy_sun_candidate_queue(payload, context=context)

    assert any("YbO is already failed/used" in error for error in errors)
    assert any("fewer than two controller-legal candidates" in error for error in errors)


def test_validate_xy_sun_candidate_queue_rejects_exploit_mode_drift_to_cooled_template() -> None:
    context = {
        "controller_constraints": {
            "xy_sun_candidate_queue_size": 4,
            "failed_or_used_reduced_formulas": [],
            "search_policy": {
                "current_search_mode": "exploit",
                "exploit_template_allowlist": ["fluorite", "rocksalt"],
                "cooled_templates": ["double_perovskite"],
                "cooled_basin_keys": ["template=double_perovskite;X=O"],
            },
        }
    }
    dp_material = {
        "natural_language_description": "Ba2CaOsO6 double perovskite",
        "target_family": "oxide double perovskite",
        "generator_template": "double_perovskite",
        "generator_role_mapping": {
            "A": {"element": "Ba", "oxidation_state": 2},
            "B": {"element": "Ca", "oxidation_state": 2},
            "B2": {"element": "Os", "oxidation_state": 6},
            "X": {"element": "O", "oxidation_state": -2},
        },
        "template_formula_family": "A2BB2O6 double perovskite",
        "expected_local_motif": "ordered BO6 octahedra",
        "why_template_is_faithful": "double_perovskite directly realizes ordered octahedra",
    }
    spinel_material = {
        "natural_language_description": "ZnGa2O4 spinel",
        "target_family": "oxide spinel",
        "generator_template": "spinel",
        "generator_role_mapping": {
            "A": {"element": "Zn", "oxidation_state": 2},
            "B": {"element": "Ga", "oxidation_state": 3},
            "X": {"element": "O", "oxidation_state": -2},
        },
        "template_formula_family": "AB2O4 spinel",
        "expected_local_motif": "spinel cation sublattice",
        "why_template_is_faithful": "spinel directly realizes the intended motif",
    }
    payload = {
        "status": "material_description_consensus",
        "sun_candidate_queue": [
            {"candidate_id": "q1", "reduced_formula": "Ba2CaOsO6", "material_description": dp_material},
            {"candidate_id": "q2", "reduced_formula": "ZnGa2O4", "material_description": spinel_material},
        ],
        "selected_candidate_id": "q1",
        "selection_rationale": "wrongly drifts away from exploit allowlist",
        "material_description": dp_material,
    }

    errors = validate_xy_sun_candidate_queue(payload, context=context)

    assert any("cooled template" in error for error in errors)
    assert any("exploit mode requires" in error for error in errors)
    assert any("fewer than two controller-legal" in error for error in errors)


def test_validate_xy_strategy_constraints_rejects_mattergen_queue_drift() -> None:
    context = {
        "generator_backend": "mattergen",
        "controller_constraints": {
            "xy_sun_candidate_queue_size": 4,
            "strategy_constraints": {
                "schema_version": "xy_strategy_constraints.v1",
                "binding": True,
                "queue_min_legal_items": 2,
                "queue_max_items": 4,
                "required_acquisition_mode": "exploit",
                "latest_strategy_order_enforced": True,
                "legal_ordered_candidate_formulas": [
                    "Rb3Mn4Br11",
                    "Rb3Mn2Br7",
                    "Rb4MnBr6",
                ],
                "first_required_formula": "Rb3Mn4Br11",
            },
        },
    }
    payload = {
        "status": "material_description_consensus",
        "selected_candidate_id": "q1",
        "selection_rationale": "wrongly starts from lower priority formula.",
        "sun_candidate_queue": [
            {
                "candidate_id": "q1",
                "rank": 1,
                "reduced_formula": "Rb3Mn2Br7",
                "acquisition_mode": "explore_adjacent",
                "material_description": {
                    "generator_template": "mattergen",
                    "preferred_reduced_formula": "Rb3Mn2Br7",
                },
            },
            {
                "candidate_id": "q2",
                "rank": 2,
                "reduced_formula": "Rb4MnBr6",
                "acquisition_mode": "exploit",
                "material_description": {
                    "generator_template": "mattergen",
                    "preferred_reduced_formula": "Rb4MnBr6",
                },
            },
        ],
        "material_description": {
            "generator_template": "mattergen",
            "preferred_reduced_formula": "Rb3Mn2Br7",
        },
    }

    errors = validate_xy_strategy_constraints(payload, context=context)

    assert any("acquisition_mode='explore_adjacent'" in error for error in errors)
    assert any("must start with first legal strategy formula Rb3Mn4Br11" in error for error in errors)
    assert any("omitted legal strategy formula" in error and "Rb3Mn4Br11" in error for error in errors)
    assert any("select first_required_formula Rb3Mn4Br11" in error for error in errors)


def test_validate_xy_queue_rejects_hidden_failed_formula() -> None:
    context = {
        "generator_backend": "mattergen",
        "controller_constraints": {
            "xy_sun_candidate_queue_size": 4,
            "failed_or_used_reduced_formulas": ["Rb2MnBr3"],
            "search_policy": {"backend": "mattergen", "current_search_mode": "exploit"},
            "strategy_constraints": {
                "schema_version": "xy_strategy_constraints.v1",
                "binding": True,
                "queue_min_legal_items": 2,
                "queue_max_items": 4,
                "required_acquisition_mode": "exploit",
                "latest_strategy_order_enforced": False,
            },
        },
    }
    payload = {
        "status": "material_description_consensus",
        "selected_candidate_id": "q1",
        "selection_rationale": "hidden duplicate should be rejected.",
        "sun_candidate_queue": [
            {
                "candidate_id": "q1",
                "rank": 1,
                "reduced_formula": "Rb2MnBr4",
                "acquisition_mode": "exploit",
                "material_description": {
                    "generator_template": "mattergen",
                    "preferred_reduced_formula": "Rb2MnBr4",
                },
            },
            {
                "candidate_id": "q2",
                "rank": 2,
                "reduced_formula": "Rb3Mn2Br7",
                "acquisition_mode": "exploit",
                "material_description": {
                    "generator_template": "mattergen",
                    "preferred_reduced_formula": "Rb3Mn2Br7",
                },
            },
        ],
        "material_description": {
            "generator_template": "mattergen",
            "preferred_reduced_formula": "Rb2MnBr4",
        },
    }

    errors = validate_xy_sun_candidate_queue(
        payload,
        context=context,
        forbidden_reduced_formulas={"Rb2MnBr4"},
    )

    assert any("Rb2MnBr4" in error and "hidden full failed/used set" in error for error in errors)
    assert any("selected material reduced_formula Rb2MnBr4 is already failed/used" in error for error in errors)


def test_mattergen_preflight_rejects_target_formula_in_exclusions() -> None:
    specs = [
        {
            "candidate_id": "Z5-1",
            "source": "generator",
            "reduced_formula": "Rb2MnBr4",
            "mattergen_requests": [
                {
                    "backend": "mattergen",
                    "properties_to_condition_on": {"chemical_system": "Br-Mn-Rb", "energy_above_hull": 0.0},
                    "filters": {
                        "chemical_system": "Br-Mn-Rb",
                        "target_reduced_formula": "Rb2MnBr4",
                        "require_target_reduced_formula": False,
                    },
                }
            ],
        }
    ]

    errors = mattergen_excluded_target_preflight_errors(specs, {"Rb2MnBr4"})

    assert errors == [
        "mattergen_target_excluded: candidate=Z5-1 target_reduced_formula=Rb2MnBr4 is already failed/used or known/training; "
        "do not submit a MatterGen request whose soft strategy target is also excluded. "
        "Return to X/Y for a non-duplicate queue item or choose a reviewed non-excluded queue candidate."
    ]


def test_failed_or_used_formulas_include_mattergen_seen_and_boundary_formulas() -> None:
    formulas = failed_or_used_formulas_from_memory(
        {
            "records": [
                {
                    "selected_record": {
                        "formula": "Rb4MnBr5",
                        "crystal_llm_mattergen_report": {
                            "accepted_formulas": [
                                "Rb4MnBr5",
                                "Rb3Mn2Br6",
                                "RbMnBr6",
                                "Rb2Mn2Br7",
                            ]
                        },
                    },
                    "xy_postmortem": {
                        "failure_boundaries": [
                            "Do not repeat the just-used soft target RbMn2Br5 in the immediate next proposal.",
                            "Do not target nonselected/materialized batch products Rb3Mn2Br6 or Rb2Mn2Br7.",
                        ]
                    },
                }
            ]
        }
    )

    assert {"Rb4MnBr5", "Rb3(MnBr3)2", "RbMnBr6", "Rb2Mn2Br7", "RbMn2Br5"} <= formulas


def test_abstract_formula_tokens_ignore_english_words_with_evaluator_null_prefix() -> None:
    tokens = abstract_formula_tokens_from_text("Ranked exploit queue: RbMn3Br7, then Rb2Mn3Br8.")

    assert "Ranked" not in tokens


def test_evaluator_null_formula_detection_avoids_pymatgen_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert evaluator_null_elements_from_formula("RfO2") == {"Rf"}
        assert "RfO2" in strategy_formula_candidate_tokens_from_text("Use ordered candidates: RfO2, then YbO.")

    assert not any("No Pauling electronegativity" in str(item.message) for item in caught)


def test_template_only_material_description_rejects_evaluator_null_elements() -> None:
    payload = {
        "material_description": {
            "generator_template": "rocksalt",
            "generator_role_mapping": {
                "A": {"element": "No", "oxidation_state": 2},
                "X": {"element": "P", "oxidation_state": -2},
            },
            "expected_local_motif": "rocksalt No-P octahedral coordination",
            "why_template_is_faithful": "rocksalt topology is the intended motif",
        }
    }

    errors = validate_template_only_material_description(payload)

    assert any("evaluator-null/high-risk elements" in error and "No" in error for error in errors)


def test_xy_return_feedback_is_visible_to_y_debate_prompts() -> None:
    context = {
        "latest_xy_strategy_constraints": {
            "next_strategy": "Propose exactly AgCuBr2 delafossite; if blocked, no fallback.",
            "failure_boundaries": [],
        },
        "controller_constraints": {
            "generator_template_only": True,
            "allowed_generator_templates": ["delafossite"],
        },
    }
    proposal = {
        "status": "no_valid_material_description",
        "agent": "X",
        "material_description": {},
        "impossibility_certificate": {
            "reason": "AgCuBr2 delafossite failed generator materialization."
        },
    }
    review = {"agree": False, "approved": False}
    return_feedback = {
        "reason": "Z/W returned the material description to X/Y.",
        "materialization_errors": [
            "xy_candidate_001: formula_probes[0] generated CuAgBr2 with template delafossite but failed structure validation: volume_per_atom_too_large"
        ],
        "w_review": {"return_to_xy": True},
    }

    prompts = [
        prompt_y_sequential_material_review(
            context,
            proposal,
            iteration=1,
            cycle=1,
            return_feedback=return_feedback,
            template_only=True,
        ),
        prompt_y_sequential_material_counterproposal(
            context,
            proposal,
            review,
            iteration=1,
            cycle=1,
            return_feedback=return_feedback,
            template_only=True,
        ),
        prompt_x_sequential_material_reverse_review(
            context,
            proposal,
            proposal,
            iteration=1,
            cycle=1,
            return_feedback=return_feedback,
            template_only=True,
        ),
        prompt_y_sequential_material_final(
            context,
            proposal,
            {"agree": True, "approved": True},
            iteration=1,
            return_feedback=return_feedback,
            template_only=True,
        ),
    ]

    for prompt in prompts:
        assert "Z/W_RETURN_FEEDBACK_FROM_PREVIOUS_DESCRIPTION_ATTEMPT" in prompt
        assert "CuAgBr2" in prompt
        assert "that exact mapping is blocked" in prompt
        assert "do not" in prompt.lower()
    assert "overrides stale next_strategy" in prompts[0]
    assert "do not counterpropose that candidate again" in prompts[1]
    assert "do not finalize that mapping" in prompts[3]


def test_sequential_material_final_prompt_uses_compact_context_and_payloads(tmp_path) -> None:
    state = {
        "current_round": 5,
        "history": [],
        "principle_book": [
            {
                "program_id": f"principle_program_{index:03d}",
                "status": "validated_principle",
                "topic_key": f"topic-{index}",
                "principle_statement": "long principle " * 200,
            }
            for index in range(30)
        ],
    }
    context = sequential_context_payload(
        state=state,
        mode="experience_xy",
        iteration=4,
        memory_path=tmp_path / "memory.json",
        candidate_source="generator",
        seed=100,
        generator_template_only=True,
    )
    proposal = {
        "status": "material_description_proposal",
        "agent": "X",
        "material_description": {
            "natural_language_description": "Ba-Hf-O perovskite " + ("irrelevant detail " * 300) + "TAIL_SENTINEL",
            "generator_template": "perovskite",
            "generator_role_mapping": {
                "A": {"element": "Ba", "oxidation_state": 2},
                "B": {"element": "Hf", "oxidation_state": 4},
                "X": {"element": "O", "oxidation_state": -2},
            },
            "expected_local_motif": "corner-sharing BO6 octahedra",
            "template_formula_family": "oxide perovskite",
            "why_template_is_faithful": "perovskite realizes the intended motif",
        },
    }
    review = {
        "status": "material_description_review",
        "agent": "Y",
        "agree": True,
        "approved": True,
        "required_revision": ("not needed " * 300) + "REVIEW_TAIL_SENTINEL",
        "overall_reasoning_summary": ("summary " * 300) + "SUMMARY_TAIL_SENTINEL",
    }

    prompt = prompt_y_sequential_material_final(context, proposal, review, iteration=4, template_only=True)

    assert "GENERATOR_TEMPLATE_ONLY_CONSTRAINT" in prompt
    assert "perovskite(A,B,X)" in prompt
    assert "long principle" not in prompt
    assert "TAIL_SENTINEL" not in prompt
    assert "REVIEW_TAIL_SENTINEL" not in prompt
    assert "SUMMARY_TAIL_SENTINEL" not in prompt
    assert len(prompt) < 16000


def test_compact_material_prompt_preserves_no_valid_certificate() -> None:
    payload = {
        "status": "no_valid_material_description",
        "agent": "X",
        "iteration": 7,
        "material_description": {},
        "impossibility_certificate": {
            "template_audit": {
                "perovskite": "RbCdBr3 duplicate and volume-boundary blocked",
                "double_perovskite": "Rb2CdMBr6 family volume-boundary blocked",
            },
            "conclusion": "no allowed branch remains",
        },
        "proposal_summary": "strategy blocked",
    }

    compact = compact_material_description_for_prompt(payload)
    assert compact["status"] == "no_valid_material_description"
    assert compact["material_description"] == {}
    assert compact["impossibility_certificate"]["template_audit"]["perovskite"].startswith("RbCdBr3")

    review = {
        "counterproposal_material_description": payload,
        "overall_reasoning_summary": "Y supplies the same no-valid certificate",
    }
    compact_review = compact_review_for_prompt(review)
    assert compact_review["counterproposal_material_description"]["status"] == "no_valid_material_description"
    assert "impossibility_certificate" in compact_review["counterproposal_material_description"]


def test_sequential_postmortem_prompts_require_operational_boundaries(tmp_path) -> None:
    context = sequential_context_payload(
        state={"current_round": 5, "history": [], "principle_book": []},
        mode="experience_xy",
        iteration=7,
        memory_path=tmp_path / "memory.json",
        candidate_source="generator",
        seed=100,
        generator_template_only=True,
    )
    record = {
        "iteration": 7,
        "evaluation_result": {"formula": "CaZrO3", "e_hull": 0.2077},
    }
    postmortem = {
        "next_strategy": "avoid A-site contraction",
        "failure_boundaries": ["Ca/Sr zirconates"],
    }

    x_prompt = prompt_x_sequential_postmortem(context, record, iteration=7)
    y_prompt = prompt_y_sequential_postmortem_review(context, record, postmortem, iteration=7)

    assert "operational instructions" in x_prompt
    assert "formulas, families, templates, roles, and variation axes" in x_prompt
    assert "strategy_blocked" in x_prompt
    assert "failed_or_used_reduced_formulas" in x_prompt
    assert "Do not name a listed formula even as a conditional fallback" in x_prompt
    assert "Used/failed formulas may appear only as" in x_prompt
    assert "failed_volume_template_boundaries" in x_prompt
    assert "pre-audited as absent from visible failed_or_used_reduced_formulas" in x_prompt
    assert "duplicate-gated ordered set" in x_prompt
    assert '"weak_near_miss" means 0.03 <= e_hull < 0.10' in x_prompt
    assert "operational enough" in y_prompt
    assert "exact fallback order" in y_prompt
    assert "strategy_blocked" in y_prompt
    assert "failed_or_used_reduced_formulas" in y_prompt
    assert "do not preserve the formula as a conditional fallback" in y_prompt
    assert "Repeated formulas are allowed only inside failure_boundaries" in y_prompt
    assert "failed_volume_template_boundaries" in y_prompt
    assert "single unaudited formula" in y_prompt
    assert "pre-audited, duplicate-gated ordered set" in y_prompt
    assert "If X labels a weak_near_miss as near_miss" in y_prompt


def test_postmortem_guard_removes_duplicate_next_strategy_formulas() -> None:
    postmortem = {
        "status": "postmortem_consensus",
        "next_strategy": (
            "Use d0 perovskites in this order: "
            "1. generator_template=perovskite; A=Ba:+2; B=Ti:+4; X=O:-2; reduced_formula=BaTiO3. "
            "2. generator_template=perovskite; A=Pb:+2; B=Ti:+4; X=O:-2; reduced_formula=PbTiO3; "
            "defend TiO6 octahedra. "
            "3. generator_template=perovskite; A=Pb:+2; B=Zr:+4; X=O:-2; reduced_formula=PbZrO3. "
            "4. generator_template=perovskite; A=K:+1; B=Nb:+5; X=O:-2; reduced_formula=KNbO3. "
            "If fewer than two clear, pivot to spinels ordered as: 1. ZnAl2O4, 2. CoAl2O4."
        ),
        "failure_boundaries": ["Do not repeat YAlO3."],
    }
    forbidden = {"BaTiO3", "KNbO3", "YAlO3"}

    assert postmortem_blocked_next_strategy_formulas(postmortem, forbidden) == ["BaTiO3", "KNbO3"]

    guarded = enforce_postmortem_next_strategy_guard(
        postmortem,
        forbidden_reduced_formulas=forbidden,
    )
    next_strategy = guarded["next_strategy"]
    next_formulas = formula_mentions_from_text(next_strategy)

    assert "BaTiO3" not in next_formulas
    assert "KNbO3" not in next_formulas
    assert "PbTiO3" in next_strategy
    assert "PbZrO3" in next_strategy
    assert "ZnAl2O4" in next_strategy
    assert "CoAl2O4" in next_strategy
    assert "TiO6" not in next_formulas
    assert guarded["controller_original_next_strategy"] == postmortem["next_strategy"]
    assert guarded["controller_postmortem_audit_errors"] == [
        "next_strategy named prior failed/used or failure-boundary reduced_formula BaTiO3; removed from binding strategy",
        "next_strategy named prior failed/used or failure-boundary reduced_formula KNbO3; removed from binding strategy",
    ]
    assert any("BaTiO3, KNbO3" in str(item) for item in guarded["failure_boundaries"])


def test_postmortem_guard_marks_fully_duplicate_route_exhausted() -> None:
    postmortem = {
        "status": "postmortem_consensus",
        "next_strategy": (
            "Continue route: 1. reduced_formula=BaTiO3; 2. reduced_formula=KNbO3."
        ),
        "failure_boundaries": [],
    }

    guarded = enforce_postmortem_next_strategy_guard(
        postmortem,
        forbidden_reduced_formulas={"BaTiO3", "KNbO3"},
    )

    assert formula_mentions_from_text(guarded["next_strategy"]) == []
    assert "named candidate list is exhausted" in guarded["next_strategy"]
    assert "does not by itself prove" in guarded["next_strategy"]
    assert len(guarded["controller_postmortem_audit_errors"]) == 2


def test_postmortem_guard_exhausts_route_when_only_one_candidate_survives() -> None:
    postmortem = {
        "status": "postmortem_consensus",
        "next_strategy": (
            "Propose ordered fluorite dioxides: CeO2 first, ThO2 second, UO2 third. "
            "Use native AO2 fluorite coordination."
        ),
        "failure_boundaries": [],
    }

    guarded = enforce_postmortem_next_strategy_guard(
        postmortem,
        forbidden_reduced_formulas={"CeO2", "ThO2"},
    )

    assert "named candidate list is exhausted" in guarded["next_strategy"]
    assert "UO2" not in formula_mentions_from_text(guarded["next_strategy"])
    assert any("fewer than two" in item for item in guarded["controller_postmortem_audit_errors"])


def test_strategy_formula_extraction_prefers_candidate_set_over_abstract_templates() -> None:
    strategy = (
        "For the next X/Y proposal, use this ordered non-duplicate candidate set and stop at the first "
        "formula that clears controller audits at proposal time: MnAl2O4, then FeAl2O4, then CoAl2O4. "
        "Use generator_template=spinel matching AB2O4 charge balance. Do not substitute PbBO3 or AX2 routes."
    )

    assert strategy_formula_candidates_from_text(strategy) == ["MnAl2O4", "Al2FeO4", "Al2CoO4"]

    sanitized_strategy = (
        "Continue only with this ordered non-duplicate formula set from the same route/fallbacks: "
        "MnAl2O4, FeAl2O4, AB2O4, PbBO3, AX2, BO6. Reconstruct a complete generator_template."
    )
    assert strategy_formula_candidates_from_text(sanitized_strategy) == ["MnAl2O4", "Al2FeO4"]


def test_strategy_formula_extraction_keeps_aln_and_skips_coordination_tokens() -> None:
    strategy = (
        "Use a duplicate-gated ordered AX wurtzite III-nitride set: first propose wurtzite AlN "
        "with Al3+ on A and N3- on X, targeting native tetrahedral AlN4/NAl4 coordination and "
        "stronger shorter Al-N bonding; if and only if AlN is blocked by duplicate or "
        "failed-volume/template audit, propose wurtzite InN with In3+ on A and N3- on X as "
        "the second pre-audited route member."
    )

    assert strategy_formula_candidates_from_text(strategy) == ["AlN", "InN"]
    assert abstract_formula_tokens_from_text(strategy) == []

    guarded = enforce_postmortem_next_strategy_guard(
        {"status": "postmortem_consensus", "next_strategy": strategy, "failure_boundaries": []},
        forbidden_reduced_formulas={"GaN"},
    )
    assert guarded["next_strategy"] == strategy


def test_strategy_formula_extraction_keeps_all_caps_binary_candidate_formulas() -> None:
    strategy = (
        "Use a duplicate-gated ordered native tetrahedral AX III-V set outside used formulas: "
        "first propose wurtzite BN with B3+ on A and N3- on X, requiring genuine tetrahedral "
        "BN4/NB4 wurtzite connectivity; if BN is duplicate- or boundary-blocked, propose "
        "zincblende BP with B3+ on A and P3- on X, requiring genuine tetrahedral BP4/PB4 "
        "zincblende connectivity. If no non-duplicate faithful tetrahedral candidate remains, "
        "mark the route exhausted rather than falling back to AlN/GaN/InN repeats."
    )

    assert strategy_formula_candidates_from_text(strategy) == ["BN", "BP"]
    assert "SUN" not in formula_mentions_from_text("Strict SUN requires negative e_hull.")

    guarded = enforce_postmortem_next_strategy_guard(
        {"status": "postmortem_consensus", "next_strategy": strategy, "failure_boundaries": []},
        forbidden_reduced_formulas={"AlN", "GaN"},
    )
    assert "BN, BP" in guarded["next_strategy"]
    assert "InN" not in guarded["next_strategy"]


def test_postmortem_guard_removes_abstract_formula_tokens_without_duplicates() -> None:
    postmortem = {
        "status": "postmortem_consensus",
        "next_strategy": (
            "Continue only with this ordered non-duplicate formula set from the same route/fallbacks: "
            "MnAl2O4, FeAl2O4, AB2O4, PbBO3, AX2, BO6."
        ),
        "failure_boundaries": [],
    }

    guarded = enforce_postmortem_next_strategy_guard(postmortem, forbidden_reduced_formulas=set())

    assert "MnAl2O4" in guarded["next_strategy"]
    assert "FeAl2O4" in guarded["next_strategy"]
    assert "AB2O4" not in guarded["next_strategy"]
    assert "PbBO3" not in guarded["next_strategy"]
    assert "AX2" not in guarded["next_strategy"]
    assert "BO6" not in guarded["next_strategy"]
    assert guarded["controller_postmortem_audit_errors"] == [
        "next_strategy named abstract/non-concrete formula token AB2O4; removed from binding strategy",
        "next_strategy named abstract/non-concrete formula token PbBO3; removed from binding strategy",
        "next_strategy named abstract/non-concrete formula token AX2; removed from binding strategy",
        "next_strategy named abstract/non-concrete formula token BO6; removed from binding strategy",
    ]


def test_postmortem_guard_removes_coordination_fragment_and_keeps_fallbacks() -> None:
    strategy = (
        "Use this ordered route: CrO6 first, then GdCrO3, then TbCrO3, then DyCrO3. "
        "CrO6 is the intended octahedral motif; the chromites are concrete fallbacks."
    )

    assert "CrO6" in abstract_formula_tokens_from_text(strategy)
    assert strategy_formula_candidates_from_text(strategy) == ["GdCrO3", "TbCrO3", "DyCrO3"]

    guarded = enforce_postmortem_next_strategy_guard(
        {"status": "postmortem_consensus", "next_strategy": strategy, "failure_boundaries": []},
        forbidden_reduced_formulas=set(),
    )

    assert "CrO6" not in strategy_formula_candidates_from_text(guarded["next_strategy"])
    assert strategy_formula_candidates_from_text(guarded["next_strategy"]) == ["GdCrO3", "TbCrO3", "DyCrO3"]
    assert any("CrO6" in item and "abstract/non-concrete" in item for item in guarded["controller_postmortem_audit_errors"])


def test_postmortem_guard_exhausts_candidate_set_with_only_abstract_tokens() -> None:
    postmortem = {
        "status": "postmortem_consensus",
        "next_strategy": (
            "Continue only with this ordered non-duplicate formula set from the same route/fallbacks: BO6."
        ),
        "failure_boundaries": [],
    }

    guarded = enforce_postmortem_next_strategy_guard(postmortem, forbidden_reduced_formulas=set())

    assert "BO6" not in guarded["next_strategy"]
    assert "named candidate list is exhausted" in guarded["next_strategy"]
    assert guarded["controller_postmortem_audit_errors"] == [
        "next_strategy named abstract/non-concrete formula token BO6; removed from binding strategy"
    ]


def test_postmortem_guard_removes_evaluator_null_element_formulas() -> None:
    postmortem = {
        "status": "postmortem_consensus",
        "next_strategy": "Use ordered rocksalt pnictides: NoP first, then YbP, then TmP.",
        "failure_boundaries": [],
    }

    guarded = enforce_postmortem_next_strategy_guard(postmortem, forbidden_reduced_formulas=set())

    assert strategy_formula_candidates_from_text(guarded["next_strategy"]) == ["YbP", "TmP"]
    assert any("NoP" in item and "evaluator-null/high-risk" in item for item in guarded["controller_postmortem_audit_errors"])


def test_sequential_fallback_postmortem_preserves_stable_not_sun_signal() -> None:
    record = {
        "iteration": 7,
        "status": "evaluated",
        "material_description": {
            "generator_template": "perovskite",
            "generator_role_mapping": {
                "A": {"element": "Na", "oxidation_state": 1},
                "B": {"element": "Ta", "oxidation_state": 5},
                "X": {"element": "O", "oxidation_state": -2},
            },
        },
        "evaluation_result": {
            "formula": "NaTaO3",
            "e_hull": 0.02949183437402425,
            "is_sun": False,
            "near_stable_0_03": True,
        },
    }

    postmortem = sequential_fallback_postmortem(record, iteration=7, reason="LLMError: connection refused")

    assert postmortem["status"] == "postmortem_consensus"
    assert postmortem["agent"] == "controller_fallback"
    assert postmortem["outcome_class"] == "near_miss"
    assert "below 0.03 eV/atom" in postmortem["next_strategy"]
    assert "Do not repeat NaTaO3" in postmortem["next_strategy"]
    assert "template=perovskite" in postmortem["failure_boundaries"][1]


def test_sequential_fallback_postmortem_separates_weak_near_miss() -> None:
    record = {
        "iteration": 8,
        "status": "evaluated",
        "material_description": {
            "generator_template": "rutile",
            "generator_role_mapping": {
                "A": {"element": "Ti", "oxidation_state": 4},
                "X": {"element": "O", "oxidation_state": -2},
            },
        },
        "evaluation_result": {
            "formula": "TiO2",
            "e_hull": 0.04319719573893188,
            "is_sun": False,
            "near_stable_0_03": False,
            "near_stable_0_10": True,
        },
    }

    postmortem = sequential_fallback_postmortem(record, iteration=8, reason="LLMError: timeout")

    assert postmortem["outcome_class"] == "weak_near_miss"
    assert "not below 0.03 eV/atom" in postmortem["next_strategy"]
    assert "weak support only" in postmortem["next_strategy"]
    assert "Do not repeat TiO2" in postmortem["next_strategy"]


def test_template_only_material_description_requires_allowed_template_and_roles() -> None:
    errors = validate_template_only_material_description(
        {
            "material_description": {
                "natural_language_description": "template-controlled oxide perovskite",
                "generator_template": "perovskite",
                "generator_role_mapping": {
                    "A": {"element": "Ba", "oxidation_state": 2},
                    "B": {"element": "Hf", "oxidation_state": 4},
                    "X": {"element": "O", "oxidation_state": -2},
                },
                "expected_local_motif": "corner-sharing BO6 octahedra",
                "why_template_is_faithful": "perovskite template explicitly realizes the BO6 network",
            }
        }
    )

    assert errors == []


def test_template_only_material_description_rejects_stoichiometry_proxy() -> None:
    errors = validate_template_only_material_description(
        {
            "material_description": {
                "natural_language_description": "CaSiN2 forced into an ABX2 template as a generator vehicle",
                "generator_template": "delafossite",
                "generator_role_mapping": {
                    "A": {"element": "Ca", "oxidation_state": 2},
                    "B": {"element": "Si", "oxidation_state": 4},
                    "X": {"element": "N", "oxidation_state": -3},
                },
                "expected_local_motif": "compact Si-N covalent nitridosilicate network",
                "why_template_is_faithful": "delafossite is only a stoichiometry vehicle for ABN2 charge balance",
            }
        }
    )

    assert any("stoichiometry vehicle" in error for error in errors)
    assert any("intended local motif" in error for error in errors)


def test_template_only_material_description_allows_negated_proxy_wording() -> None:
    errors = validate_template_only_material_description(
        {
            "material_description": {
                "natural_language_description": "AgCuBr2 layered delafossite bromide",
                "generator_template": "delafossite",
                "generator_role_mapping": {
                    "A": {"element": "Ag", "oxidation_state": 1},
                    "B": {"element": "Cu", "oxidation_state": 1},
                    "X": {"element": "Br", "oxidation_state": -1},
                },
                "expected_local_motif": "layered delafossite A-B-X2 topology",
                "why_template_is_faithful": (
                    "The delafossite template natively realizes the intended layered A-B-X2 motif; "
                    "it is not being used as a generic formula container or stoichiometry proxy."
                ),
            }
        }
    )

    assert not any("unfaithful proxy pattern" in error for error in errors)


def test_template_proxy_rejection_allows_rather_than_proxy_wording() -> None:
    reason = xy_debate.template_proxy_rejection_reason(
        "MatterGen samples full periodic structures rather than using a fixed-template proxy or manual cell."
    )

    assert reason == ""


def test_template_only_material_description_rejects_custom_structure() -> None:
    errors = validate_template_only_material_description(
        {
            "material_description": {
                "generator_template": "custom_structure",
                "generator_role_mapping": {},
                "expected_local_motif": "isolated tetrahedra",
                "why_template_is_faithful": "requires a hand-built cell",
            }
        }
    )

    assert any("forbids custom_structure" in error for error in errors)


def test_template_only_material_description_rejects_charge_imbalance() -> None:
    errors = validate_template_only_material_description(
        {
            "material_description": {
                "generator_template": "double_perovskite",
                "generator_role_mapping": {
                    "A": {"element": "Rb", "oxidation_state": 1},
                    "B": {"element": "Cd", "oxidation_state": 2},
                    "B2": {"element": "Sn", "oxidation_state": 4},
                    "X": {"element": "Br", "oxidation_state": -1},
                },
                "expected_local_motif": "ordered bromide octahedra",
                "why_template_is_faithful": "double perovskite template realizes ordered B/B2 octahedra",
            }
        }
    )

    assert any("not charge-neutral" in error for error in errors)
    assert any("net_charge=2" in error for error in errors)


def test_template_only_material_description_rejects_history_failed_formula() -> None:
    payload = {
        "material_description": {
            "generator_template": "double_perovskite",
            "generator_role_mapping": {
                "A": {"element": "Rb", "oxidation_state": 1},
                "B": {"element": "Cd", "oxidation_state": 2},
                "B2": {"element": "Zn", "oxidation_state": 2},
                "X": {"element": "Br", "oxidation_state": -1},
            },
            "expected_local_motif": "ordered bromide octahedra",
            "why_template_is_faithful": "double perovskite template realizes ordered B/B2 octahedra",
        }
    }

    assert template_formula_from_material_description(payload) == "Rb2ZnCdBr6"

    errors = validate_template_only_material_description(
        payload,
        forbidden_reduced_formulas={"Rb2CdZnBr6"},
    )

    assert any("repeats a prior failed or used" in error for error in errors)


def test_template_only_material_description_rejects_failed_volume_boundary() -> None:
    payload = {
        "material_description": {
            "generator_template": "double_perovskite",
            "generator_role_mapping": {
                "A": {"element": "Rb", "oxidation_state": 1},
                "B": {"element": "Mg", "oxidation_state": 2},
                "B2": {"element": "Cd", "oxidation_state": 2},
                "X": {"element": "Br", "oxidation_state": -1},
            },
            "expected_local_motif": "ordered bromide octahedra",
            "why_template_is_faithful": "double perovskite template realizes ordered B/B2 octahedra",
        }
    }
    memory = {
        "records": [
            {
                "materialization_errors": [
                    "xy_candidate_001: formula_probes[0] generated Rb2SrCdBr6 with template double_perovskite but failed structure validation: volume_per_atom_too_large"
                ]
            }
        ]
    }

    boundaries = failed_volume_boundaries_from_memory(memory)
    errors = validate_template_only_material_description(
        payload,
        forbidden_volume_boundaries=boundaries,
    )

    assert template_volume_boundary_key(payload) in boundaries
    assert any("volume_per_atom_too_large generator boundary" in error for error in errors)


def test_template_only_material_description_rejects_transient_spinel_volume_boundary() -> None:
    payload = {
        "material_description": {
            "generator_template": "spinel",
            "generator_role_mapping": {
                "A": {"element": "Hg", "oxidation_state": 2},
                "B": {"element": "Rb", "oxidation_state": 1},
                "X": {"element": "Br", "oxidation_state": -1},
            },
            "expected_local_motif": "Rb-rich bromide spinel",
            "why_template_is_faithful": "spinel template realizes AB2X4 bromide stoichiometry",
        }
    }
    memory = memory_with_transient_materialization_errors(
        {"records": []},
        [
            "xy_candidate_001: formula_probes[0] generated Rb2CdBr4 with template spinel but failed structure validation: volume_per_atom_too_large"
        ],
    )

    boundaries = failed_volume_boundaries_from_memory(memory)
    errors = validate_template_only_material_description(
        payload,
        forbidden_volume_boundaries=boundaries,
    )

    assert "template=spinel;contains=Rb;X=Br;failure=volume_per_atom_too_large" in boundaries
    assert any("volume_per_atom_too_large generator boundary" in error for error in errors)


def test_template_only_material_description_rejects_transient_same_formula_after_return() -> None:
    payload = {
        "material_description": {
            "generator_template": "delafossite",
            "generator_role_mapping": {
                "A": {"element": "Ag", "oxidation_state": 1},
                "B": {"element": "Cu", "oxidation_state": 1},
                "X": {"element": "Br", "oxidation_state": -1},
            },
            "expected_local_motif": "layered delafossite bromide",
            "why_template_is_faithful": "delafossite directly realizes the layered ABX2 motif",
        }
    }
    memory = memory_with_transient_materialization_errors(
        {"records": []},
        [
            "xy_candidate_001: formula_probes[0] generated CuAgBr2 with template delafossite but failed structure validation: volume_per_atom_too_large"
        ],
    )

    errors = validate_template_only_material_description(
        payload,
        forbidden_reduced_formulas=failed_or_used_formulas_from_memory(memory),
        forbidden_volume_boundaries=failed_volume_boundaries_from_memory(memory),
    )

    assert template_formula_from_material_description(payload) in failed_or_used_formulas_from_memory(memory)
    assert any("repeats a prior failed or used" in error for error in errors)


def test_failed_volume_boundaries_generalize_double_perovskite_a_site_from_formula() -> None:
    payload = {
        "material_description": {
            "generator_template": "double_perovskite",
            "generator_role_mapping": {
                "A": {"element": "K", "oxidation_state": 1},
                "B": {"element": "Ca", "oxidation_state": 2},
                "B2": {"element": "Cd", "oxidation_state": 2},
                "X": {"element": "Br", "oxidation_state": -1},
            },
            "expected_local_motif": "ordered bromide octahedra",
            "why_template_is_faithful": "double perovskite template realizes ordered B/B2 octahedra",
        }
    }
    memory = {
        "records": [
            {
                "materialization_errors": [
                    "xy_candidate_001: formula_probes[0] generated K2MgCdBr6 with template double_perovskite but failed structure validation: volume_per_atom_too_large"
                ]
            }
        ]
    }

    boundaries = failed_volume_boundaries_from_memory(memory)
    errors = validate_template_only_material_description(
        payload,
        forbidden_volume_boundaries=boundaries,
    )

    assert "template=double_perovskite;A=K;X=Br;failure=volume_per_atom_too_large" in boundaries
    assert any("volume_per_atom_too_large generator boundary" in error for error in errors)


def test_template_volume_boundary_keys_from_formula_infers_alkali_a_site() -> None:
    assert "template=perovskite;A=Na;X=Br;failure=volume_per_atom_too_large" in (
        template_volume_boundary_keys_from_formula("perovskite", "NaCdBr3")
    )
    assert "template=rocksalt;A=K;X=Br;failure=volume_per_atom_too_large" in (
        template_volume_boundary_keys_from_formula("rocksalt", "KBr")
    )


def test_sequential_description_and_candidate_agreement() -> None:
    proposal = {"material_description": {"natural_language_description": "Rb-Cd-Br soft cage candidate"}}
    assert sequential_description_agrees({"agree": True, "approved": True}, proposal)
    assert not sequential_description_agrees({"agree": False, "approved": False}, proposal)

    no_valid = {
        "status": "no_valid_material_description",
        "material_description": {},
        "impossibility_certificate": {"decision": "all allowed Rb-Cd-Br branches are blocked"},
        "proposal_summary": "return to strategy",
    }
    review = {"agree": True, "approved": True, "overall_reasoning_summary": "audit accepted"}
    assert material_payload_declares_no_valid_description(no_valid)
    assert sequential_no_valid_description_agrees(review, no_valid)
    consensus = no_valid_material_consensus_from_payload(no_valid, review, iteration=7)
    assert consensus["status"] == "no_valid_material_consensus"
    assert consensus["material_description"] == {}
    assert not sequential_description_agrees(review, no_valid)

    candidate_proposal = {"candidate_specs": [{"id": "seq_001"}]}
    assert sequential_candidate_agrees({"agree": True, "approved_candidate_ids": ["seq_001"]}, candidate_proposal)
    assert not sequential_candidate_agrees({"agree": True, "return_to_xy": True}, candidate_proposal)


def test_no_valid_consensus_requires_all_strategy_formula_audits() -> None:
    context = {
        "latest_xy_strategy_constraints": {
            "next_strategy": (
                "First audit RbCdBr3; if blocked, use MgCd2O4 spinel then ZnCd2O4 spinel."
            ),
            "next_strategy_candidate_formulas": ["RbCdBr3", "MgCd2O4", "ZnCd2O4"],
        }
    }
    no_valid = {
        "status": "no_valid_material_description",
        "material_description": {},
        "impossibility_certificate": {
            "audit": [
                {"formula": "RbCdBr3", "blocked": True},
            ],
            "conclusion": "all allowed branches are blocked",
        },
    }
    review = {"agree": True, "approved": True, "overall_reasoning_summary": "audit accepted"}

    assert missing_no_valid_strategy_formula_audits(no_valid, context) == ["MgCd2O4", "ZnCd2O4"]
    assert not sequential_no_valid_description_agrees_with_context(review, no_valid, context)
    guarded = enforce_no_valid_strategy_audit_guard(review, no_valid, context)
    assert guarded["agree"] is False
    assert guarded["approved"] is False
    assert "MgCd2O4, ZnCd2O4" in guarded["required_revision"]


def test_order_guard_uses_first_remaining_formula_after_removed_motif() -> None:
    context = {
        "latest_xy_strategy_constraints": {
            "next_strategy": (
                "Controller-sanitized next_strategy: removed CrO6. "
                "Continue only with this ordered non-duplicate formula set from the same route/fallbacks: "
                "GdCrO3, TbCrO3, DyCrO3."
            ),
            "next_strategy_candidate_formulas": ["GdCrO3", "TbCrO3", "DyCrO3"],
            "controller_postmortem_audit_errors": [
                "next_strategy named abstract/non-concrete formula token CrO6; removed from binding strategy"
            ],
        },
        "controller_constraints": {"failed_or_used_reduced_formulas": []},
    }
    proposal = {
        "material_description": {
            "generator_template": "perovskite",
            "generator_role_mapping": {
                "A": {"element": "Gd", "oxidation_state": 3},
                "B": {"element": "Cr", "oxidation_state": 3},
                "X": {"element": "O", "oxidation_state": -2},
            },
            "expected_local_motif": "corner-sharing CrO6 octahedra",
            "why_template_is_faithful": "perovskite realizes the chromite octahedral framework",
        }
    }

    assert latest_strategy_selection_order_errors(proposal, context) == []


def test_exhausted_latest_strategy_rejects_single_candidate_without_two_formula_audit() -> None:
    context = {
        "latest_xy_strategy_constraints": {
            "next_strategy": (
                "Controller-sanitized next_strategy: Treat the named route as exhausted for the next turn. "
                "X/Y must either name a materially different route with at least two pre-audited non-duplicate "
                "formulas or return no_valid_material_description."
            ),
            "next_strategy_candidate_formulas": [],
        },
        "controller_constraints": {
            "failed_or_used_reduced_formulas": ["MnAl2O4"],
        },
    }
    proposal = {
        "status": "material_description_proposal",
        "proposal_summary": "Propose FeAl2O4 as a faithful normal spinel.",
        "material_description": {
            "generator_template": "spinel",
            "generator_role_mapping": {
                "A": {"element": "Fe", "oxidation_state": 2},
                "B": {"element": "Al", "oxidation_state": 3},
                "X": {"element": "O", "oxidation_state": -2},
            },
            "expected_local_motif": "normal spinel",
            "why_template_is_faithful": "spinel topology is the intended motif",
        },
    }
    review = {"agree": True, "approved": True}

    guarded = enforce_exhausted_strategy_route_audit_guard(review, proposal, context)

    assert guarded["agree"] is False
    assert guarded["approved"] is False
    assert "at least two concrete" in guarded["required_revision"]


def test_named_candidate_list_exhaustion_still_requires_two_formula_audit() -> None:
    context = {
        "latest_xy_strategy_constraints": {
            "next_strategy": (
                "Controller-sanitized next_strategy: the named candidate list is exhausted. "
                "The validated mechanism basin may continue only with two new formulas."
            ),
            "next_strategy_candidate_formulas": [],
        },
        "controller_constraints": {"failed_or_used_reduced_formulas": []},
    }
    proposal = {
        "status": "material_description_proposal",
        "proposal_summary": "Only FeAl2O4 was audited.",
        "material_description": {
            "generator_template": "spinel",
            "generator_role_mapping": {
                "A": {"element": "Fe", "oxidation_state": 2},
                "B": {"element": "Al", "oxidation_state": 3},
                "X": {"element": "O", "oxidation_state": -2},
            },
            "expected_local_motif": "normal spinel",
            "why_template_is_faithful": "spinel topology is the intended motif",
        },
    }
    review = {"agree": True, "approved": True}

    guarded = enforce_exhausted_strategy_route_audit_guard(review, proposal, context)

    assert guarded["agree"] is False
    assert "at least two concrete" in guarded["required_revision"]


def test_exhausted_latest_strategy_allows_two_formula_replacement_audit() -> None:
    context = {
        "latest_xy_strategy_constraints": {
            "next_strategy": "Treat the named route as exhausted for the next turn.",
            "next_strategy_candidate_formulas": [],
        },
        "controller_constraints": {"failed_or_used_reduced_formulas": []},
    }
    proposal = {
        "status": "material_description_proposal",
        "proposal_summary": (
            "New route audit: FeAl2O4 and MgFe2O4 are concrete non-duplicate spinel candidates; "
            "select FeAl2O4 first."
        ),
        "material_description": {
            "generator_template": "spinel",
            "generator_role_mapping": {
                "A": {"element": "Fe", "oxidation_state": 2},
                "B": {"element": "Al", "oxidation_state": 3},
                "X": {"element": "O", "oxidation_state": -2},
            },
            "expected_local_motif": "normal spinel",
            "why_template_is_faithful": "spinel topology is the intended motif",
        },
    }
    review = {"agree": True, "approved": True}

    guarded = enforce_exhausted_strategy_route_audit_guard(review, proposal, context)

    assert guarded == review


def test_exhausted_latest_strategy_counts_sun_candidate_queue_formulas() -> None:
    context = {
        "latest_xy_strategy_constraints": {
            "next_strategy": "Treat the named route as exhausted for the next turn.",
            "next_strategy_candidate_formulas": [],
        },
        "controller_constraints": {"failed_or_used_reduced_formulas": ["Ba2MgWO6"]},
    }
    proposal = {
        "status": "material_description_proposal",
        "proposal_summary": "Select q1 from the audited replacement queue.",
        "sun_candidate_queue": [
            {"candidate_id": "q1", "reduced_formula": "Ba2MgMoO6"},
            {"candidate_id": "q2", "reduced_formula": "Sr2MgMoO6"},
            {"candidate_id": "q3", "reduced_formula": "Ba2MgWO6"},
        ],
        "selected_candidate_id": "q1",
        "material_description": {
            "generator_template": "double_perovskite",
            "generator_role_mapping": {
                "A": {"element": "Ba", "oxidation_state": 2},
                "B": {"element": "Mg", "oxidation_state": 2},
                "B2": {"element": "Mo", "oxidation_state": 6},
                "X": {"element": "O", "oxidation_state": -2},
            },
            "expected_local_motif": "ordered oxide double perovskite",
            "why_template_is_faithful": "double_perovskite topology is the intended motif",
        },
    }
    review = {"agree": True, "approved": True}

    guarded = enforce_exhausted_strategy_route_audit_guard(review, proposal, context)

    assert guarded == review


def test_reviewer_rejection_requires_counterproposal_by_default() -> None:
    assert review_requires_counterproposal({"agree": False, "required_revision": "fix template"}, rejection_streak=0)
    assert not review_requires_counterproposal({"agree": True}, rejection_streak=0)
    assert not review_requires_counterproposal({"agree": False}, rejection_streak=0, threshold=0)


def test_sequential_memory_reads_missing_file(tmp_path) -> None:
    memory = read_sequential_memory(tmp_path / "missing.json")

    assert memory["records"] == []
    assert material_description_from_payload({"material_description": {"target_family": "halide"}})["target_family"] == "halide"


def test_seed_sequential_memory_if_needed_copies_prior_records(tmp_path) -> None:
    seed_path = tmp_path / "prior_memory.json"
    target_path = tmp_path / "new_memory.json"
    seed_path.write_text(
        '{"schema_version":"xy_sequential_single_memory.v1","records":[{"iteration":1,"materialization_errors":["RbCdBr3 failed"]}]}'
    )

    report = seed_sequential_memory_if_needed(target_path, seed_path)
    memory = read_sequential_memory(target_path)

    assert report and report["seeded"] is True
    assert report["record_count"] == 1
    assert memory["seeded_from"] == str(seed_path)
    assert memory["records"][0]["materialization_errors"] == ["RbCdBr3 failed"]


def test_seed_sequential_memory_if_needed_preserves_existing_records(tmp_path) -> None:
    seed_path = tmp_path / "prior_memory.json"
    target_path = tmp_path / "new_memory.json"
    seed_path.write_text('{"records":[{"iteration":1}]}')
    target_path.write_text('{"records":[{"iteration":99}]}')

    report = seed_sequential_memory_if_needed(target_path, seed_path)
    memory = read_sequential_memory(target_path)

    assert report and report["seeded"] is False
    assert memory["records"] == [{"iteration": 99}]


def test_used_formulas_from_sequential_memory_reduces_formulas() -> None:
    memory = {
        "records": [
            {"evaluation_result": {"formula": "Rb2Cd2Br6"}},
            {"selected_record": {"formula": "MgAl2O4"}},
            {
                "materialization_errors": [
                    "xy_candidate_001: formula_probes[0] generated Rb2ZnCdBr6 with template double_perovskite but failed structure validation: volume_per_atom_too_large",
                    "xy_candidate_001: formula_probes[0] generated Mg(FeO2)2 but it is in the known/training formula set",
                ]
            },
        ]
    }

    formulas = used_formulas_from_memory(memory)
    failed_or_used = failed_or_used_formulas_from_memory(memory)

    assert "RbCdBr3" in formulas
    assert "MgAl2O4" in formulas
    assert "Rb2ZnCdBr6" not in formulas
    assert "Mg(FeO2)2" not in formulas
    assert "Rb2ZnCdBr6" in failed_or_used
    assert "Mg(FeO2)2" in failed_or_used


def test_populate_constraints_exposes_evaluator_null_elements() -> None:
    memory = {
        "records": [
            {
                "status": "evaluated",
                "evaluation_result": {
                    "formula": "NoP",
                    "e_hull": None,
                    "evaluation_error": "missing_e_hull",
                },
            }
        ]
    }
    context = {"controller_constraints": {}}

    assert "No" in evaluator_null_elements_from_memory(memory)
    populate_sequential_controller_constraints(context, memory)

    constraints = context["controller_constraints"]
    assert "No" in constraints["forbidden_evaluator_null_elements"]
    assert "forbidden_evaluator_null_element_policy" in constraints


def test_build_sequential_report_counts_sun() -> None:
    class Args:
        mode = "experience_xy"
        generation_protocol = "sequential_single"
        candidate_count = 2

    report = build_sequential_report(
        args=Args(),
        state={"current_round": 10, "principle_book": [1, 2]},
        memory={
            "records": [
                {"evaluation_result": {"is_sun": True, "near_stable_0_03": True}},
                {"evaluation_result": {"is_sun": False, "near_stable_0_03": True}},
            ]
        },
        materialization_errors=[],
    )

    assert report["evaluated_count"] == 2
    assert report["sun_count"] == 1
    assert report["sun_ratio"] == 0.5


def test_normalize_candidate_rejects_energy_prefilter_fields() -> None:
    spec = {
        "id": "xy_s001_c001",
        "source": "mp_pool",
        "count": 1,
        "query": {
            "elements_all": ["Rb", "Cd", "Br"],
            "formation_energy_per_atom_max": -1.0,
            "preferred_order": ["formation_energy_per_atom asc"],
        },
        "cited_principle_ids": ["principle_program_015"],
    }

    normalized, errors = normalize_candidate_spec(spec, default_id="fallback")

    assert normalized is None
    assert any("forbidden" in error for error in errors)
    assert any("preferred_order" in error for error in errors)


def test_normalize_candidate_lifts_formula_exclusion_out_of_query() -> None:
    spec = {
        "id": "xy_s001_c001",
        "source": "mp_pool",
        "query": {
            "elements_all": ["Rb", "Cd", "Br"],
            "preferred_order": "random",
            "exclude_formula_probes": ["RbCdBr3"],
        },
        "cited_principle_ids": ["principle_program_015"],
    }

    normalized, errors = normalize_candidate_spec(spec, default_id="fallback")

    assert not errors
    assert normalized is not None
    assert normalized["exclude_formulas"] == ["RbCdBr3"]
    assert "exclude_formula_probes" not in normalized["query"]


def test_normalize_candidate_repairs_common_query_aliases() -> None:
    spec = {
        "id": "xy_s001_c001",
        "source": "mp_pool",
        "query": {
            "elements_all": ["Na", "P", "S"],
            "formula": "Na3PS4",
            "num_sites_max": 20,
            "fields": ["material_id"],
            "random_seed": 7,
            "deduplicate_against_prior_material_ids": ["mp-a"],
        },
        "cited_principle_ids": ["principle_program_004"],
    }

    normalized, errors = normalize_candidate_spec(spec, default_id="fallback")

    assert not errors
    assert normalized is not None
    assert normalized["query"]["formula_in"] == ["Na3PS4"]
    assert normalized["query"]["nsites_max"] == 20
    assert "formula" not in normalized["query"]
    assert "num_sites_max" not in normalized["query"]
    assert "fields" not in normalized["query"]
    assert "random_seed" not in normalized["query"]
    assert "deduplicate_against_prior_material_ids" not in normalized["query"]


def test_parse_formula_probe_string_to_generator_probe() -> None:
    parsed, error = parse_formula_probe_string(
        "template=perovskite;A=Ba:+2;B=Hf:+4;X=O:-2;family=oxide_perovskite",
        default_id="probe_1",
    )

    assert error is None
    assert parsed is not None
    assert parsed["template"] == "perovskite"
    assert parsed["roles"]["A"] == {"element": "Ba", "oxidation_state": 2}
    assert parsed["roles"]["B"] == {"element": "Hf", "oxidation_state": 4}
    assert parsed["roles"]["X"] == {"element": "O", "oxidation_state": -2}


def test_generator_only_normalize_rejects_mp_pool_material_id() -> None:
    normalized, errors = normalize_candidate_spec(
        {
            "id": "xy_s001_c001",
            "source": "mp_pool",
            "material_ids": ["mp-1"],
            "cited_principle_ids": ["principle_program_001"],
        },
        default_id="fallback",
        allowed_sources={"generator"},
    )

    assert normalized is None
    assert any("not allowed" in error for error in errors)


def test_generator_only_normalize_accepts_formula_probe_string() -> None:
    normalized, errors = normalize_candidate_spec(
        {
            "id": "xy_s001_c001",
            "formula_probe_strings": ["template=rocksalt;A=Li:+1;X=F:-1;family=halide"],
            "cited_principle_ids": ["principle_program_001"],
            "template_consistency_audit": template_audit("rocksalt"),
        },
        default_id="fallback",
        allowed_sources={"generator"},
    )

    assert errors == []
    assert normalized is not None
    assert normalized["source"] == "generator"
    assert normalized["formula_probes"][0]["template"] == "rocksalt"
    assert normalized["formula_probes"][0]["roles"]["A"] == {"element": "Li", "oxidation_state": 1}
    assert "formula_probe_strings" not in normalized


def test_generator_only_normalize_repairs_common_generator_aliases() -> None:
    normalized, errors = normalize_candidate_spec(
        {
            "candidate_id": "seq_001",
            "source": "generator",
            "formula_probe_string": "template=rocksalt;A=Li:+1;X=F:-1;family=halide",
            "template_consistency_audit": {
                "chosen_generator_template": "rocksalt",
                "template_faithful": True,
                "unsupported_proxy": False,
                "local_motif": "rocksalt octahedral binary halide coordination",
                "required_coordination": "octahedral AX coordination",
                "audit_reasoning": "rocksalt realizes the intended binary octahedral motif",
            },
        },
        default_id="fallback",
        allowed_sources={"generator"},
    )

    assert errors == []
    assert normalized is not None
    assert normalized["id"] == "seq_001"
    assert normalized["formula_probes"][0]["template"] == "rocksalt"
    assert "formula_probe_string" not in normalized
    assert "candidate_id" not in normalized
    assert normalized["template_consistency_audit"]["chosen_template"] == "rocksalt"
    assert normalized["template_consistency_audit"]["template_realizes_motif"] is True


def test_generator_only_normalize_repairs_custom_structure_template_suffix() -> None:
    normalized, errors = normalize_candidate_spec(
        {
            "id": "seq_001",
            "source": "generator",
            "structure_dicts": [{"@module": "pymatgen.core.structure", "@class": "Structure"}],
            "template_consistency_audit": {
                **template_audit("custom_structure_Rb4CdBr6_P1"),
                "why_template_is_faithful": "custom Structure.as_dict realizes isolated CdBr6 cages",
            },
        },
        default_id="fallback",
        allowed_sources={"generator"},
    )

    assert errors == []
    assert normalized is not None
    assert normalized["template_consistency_audit"]["chosen_template"] == "custom_structure"
    assert any("custom_structure_Rb4CdBr6_P1" in note for note in normalized["normalization_notes"])


def test_sequential_prompts_include_generator_error_feedback() -> None:
    feedback = {
        "generator_errors": [
            "seq_001: formula_probe_string template must be one of allowed templates, got 'A2BX4_halometallate'",
            "seq_001: formula_probes[0] generated RbCdBr3 with template perovskite but failed structure validation: volume_per_atom_too_large (volume_per_atom=38.785, min_distance=2.894, required volume_per_atom=4.0..29.5, min_distance>=0.75, max_sites=80)",
        ]
    }
    context = {"iteration": 3}
    material_consensus = {"material_description": {"target_family": "halide"}}

    z_prompt = prompt_z_sequential_candidate(context, material_consensus, iteration=3, feedback=feedback)
    w_prompt = prompt_w_sequential_candidate_review(
        context,
        material_consensus,
        {"candidate_specs": [{"id": "seq_001"}]},
        iteration=3,
        cycle=2,
        feedback=feedback,
    )

    assert "formula_probe_strings" in z_prompt
    assert "structure_dicts" in z_prompt
    assert "A2BX4_halometallate" in z_prompt
    assert "Z/W's responsibility" in z_prompt
    assert "formula_probe_strings" in w_prompt
    assert "A2BX4_halometallate" in w_prompt
    assert "Set return_to_xy=true when" in w_prompt
    assert "every faithful generator representation is barred" in w_prompt
    assert "volume_per_atom_too_large" in z_prompt
    assert "must not repeat any failed formula_probe_string" in z_prompt
    assert "Every listed generator error is binding" in z_prompt
    assert "every faithful allowed representation has been barred" in z_prompt
    assert "must reject any Z repair or W counterproposal that repeats" in w_prompt


def test_mattergen_high_exclusion_no_accepted_returns_to_xy_strategy() -> None:
    first_errors = [
        "W_counter_Z1: MatterGen materialized 0 records but requested 1; "
        "chemical_system=Br-Mn-Rb; target_reduced_formula=Rb7Mn2Br11; "
        "report_status=no_accepted_structures; accepted_count=0; "
        "reject_reasons={'chemical_system_not_exact': 1, 'excluded_reduced_formula': 15}"
    ]
    candidate_consensus = {
        "agreed_candidate_specs": [
            {
                "candidate_id": "W_counter_Z1",
                "source": "generator",
                "mattergen_requests": [
                    {
                        "backend": "mattergen",
                        "properties_to_condition_on": {"chemical_system": "Br-Mn-Rb"},
                        "filters": {
                            "chemical_system": ["Br", "Mn", "Rb"],
                            "target_reduced_formula": "Rb7Mn2Br11",
                        },
                    }
                ],
            }
        ]
    }

    saturation = mattergen_saturation_analysis_from_errors(
        first_errors,
        request_payload=candidate_consensus,
    )
    assert saturation["status"] == "mattergen_duplicate_pressure"
    assert saturation["requires_xy_strategy_revision"] is False
    assert saturation["controlled_rescue_allowed"] is True
    assert saturation["excluded_ratio"] == 0.9375
    assert "Br-Mn-Rb" in saturation["blocked_chemical_systems"]

    repeated_errors = first_errors + [
        "W_counter_Z1: MatterGen materialized 0 records but requested 1; "
        "chemical_system=Br-Mn-Rb; target_reduced_formula=Rb7Mn2Br11; "
        "report_status=no_accepted_structures; accepted_count=0; "
        "reject_reasons={'chemical_system_not_exact': 2, 'excluded_reduced_formula': 46}"
    ]
    feedback = sequential_generator_repair_feedback(
        iteration=1,
        description_attempt=2,
        repair_round=1,
        errors=repeated_errors,
        z_proposal={"candidate_specs": []},
        w_review={"agree": False},
        candidate_consensus=candidate_consensus,
        backend="mattergen",
    )
    assert mattergen_feedback_requires_xy_strategy_revision(feedback)
    assert feedback["mattergen_saturation"]["blocked_chemical_systems"] == ["Br-Mn-Rb"]
    assert feedback["mattergen_saturation"]["status"] == "saturated_mattergen_basin"
    assert "saturated_mattergen_basin" in feedback["controller_instruction"]
    assert "increasing sampling budget" in feedback["controller_instruction"]


def test_mattergen_halogen_stoichiometry_guard_rejects_low_halogen_drift() -> None:
    request = {"filters": {"target_reduced_formula": "Rb6MnBr8"}}

    rejection = xy_debate._mattergen_halogen_stoichiometry_rejection("Rb3Mn2Br", request)

    assert "halogen_stoichiometry_guard" in rejection
    assert "Rb3Mn2Br" in rejection
    assert xy_debate._mattergen_halogen_stoichiometry_rejection("Rb3MnBr8", request) == ""


def test_sequential_counterproposal_prompts_use_reverse_review_protocol() -> None:
    context = {"iteration": 3}
    material_consensus = {"material_description": {"target_family": "halide", "generator_template": "perovskite"}}
    proposal = {"candidate_specs": [{"id": "seq_001", "source": "generator", "count": 1}]}
    review = {"agree": False, "rejected_candidates": [{"candidate_id": "seq_001", "required_revision": "fix roles"}]}

    y_counter = prompt_y_sequential_material_counterproposal(
        context,
        {"material_description": {"target_family": "halide"}},
        {"agree": False, "required_revision": "choose allowed template"},
        iteration=3,
        cycle=1,
        template_only=True,
    )
    x_reverse = prompt_x_sequential_material_reverse_review(
        context,
        {"material_description": {"target_family": "halide"}},
        {"material_description": {"target_family": "oxide"}},
        iteration=3,
        cycle=1,
        template_only=True,
    )
    w_counter = prompt_w_sequential_candidate_counterproposal(
        context,
        material_consensus,
        proposal,
        review,
        iteration=3,
        cycle=1,
        feedback={"generator_errors": ["volume_per_atom_too_large"]},
        template_only=True,
    )
    z_reverse = prompt_z_sequential_candidate_reverse_review(
        context,
        material_consensus,
        proposal,
        {"candidate_specs": [{"id": "seq_002"}]},
        iteration=3,
        cycle=1,
        feedback={"generator_errors": ["volume_per_atom_too_large"]},
        template_only=True,
    )

    assert "STRICT_COUNTERPROPOSAL_PROTOCOL" in y_counter
    assert 'agent to "X"' in y_counter
    assert 'agent to "Y"' in x_reverse
    assert 'agent to "Z"' in w_counter
    assert "volume_per_atom_too_large" in w_counter
    assert "every faithful allowed formula-probe representation is barred" in w_counter
    assert 'agent to "W"' in z_reverse
    assert "every faithful allowed generator representation is barred" in z_reverse


def test_candidate_payload_declares_no_faithful_generator_candidate() -> None:
    assert candidate_payload_declares_no_faithful_generator_candidate(
        {"candidate_specs": [], "feasibility_assessment": {"can_generate_faithfully": False}}
    )
    assert candidate_payload_declares_no_faithful_generator_candidate(
        {"candidate_specs": [], "feasibility_assessment": {"can_generate_faithfully": "false"}}
    )
    assert not candidate_payload_declares_no_faithful_generator_candidate(
        {
            "candidate_specs": [{"source": "generator"}],
            "feasibility_assessment": {"can_generate_faithfully": False},
        }
    )
    assert not candidate_payload_declares_no_faithful_generator_candidate(
        {"candidate_specs": [], "feasibility_assessment": {"can_generate_faithfully": True}}
    )


def test_sequential_template_only_prompts_forbid_structure_dicts() -> None:
    context = {"iteration": 3}
    material_consensus = {
        "material_description": {
            "generator_template": "perovskite",
            "target_family": "oxide perovskite",
        }
    }

    z_prompt = prompt_z_sequential_candidate(context, material_consensus, iteration=3, template_only=True)
    w_prompt = prompt_w_sequential_candidate_review(
        context,
        material_consensus,
        {"candidate_specs": [{"id": "seq_001"}]},
        iteration=3,
        cycle=2,
        template_only=True,
    )

    assert "GENERATOR_TEMPLATE_ONLY_CONSTRAINT" in z_prompt
    assert "structure_dicts and structure_dict aliases are invalid" in w_prompt
    assert '"formula_probe_strings"' in z_prompt
    assert '"formula_probes"' in z_prompt


def test_w_candidate_review_prompt_preserves_required_generator_fields() -> None:
    context = {"iteration": 3}
    material_consensus = {"material_description": {"generator_template": "perovskite"}}
    proposal = {
        "candidate_specs": [
            {
                "id": "seq_001",
                "source": "generator",
                "count": 1,
                "formula_probe_strings": [
                    "template=perovskite;A=Rb:+1;B=Cd:+2;X=Br:-1;family=RbCdBr3_halide_perovskite"
                ],
                "generator_template": "perovskite",
                "generator_role_mapping": {
                    "A": {"element": "Rb", "oxidation_state": 1},
                    "B": {"element": "Cd", "oxidation_state": 2},
                    "X": {"element": "Br", "oxidation_state": -1},
                },
                "template_consistency_audit": template_audit("perovskite"),
            }
        ]
    }

    w_prompt = prompt_w_sequential_candidate_review(
        context,
        material_consensus,
        proposal,
        iteration=3,
        cycle=1,
        template_only=True,
    )

    assert '"count":1' in w_prompt
    assert '"generator_template":"perovskite"' in w_prompt
    assert '"generator_role_mapping"' in w_prompt


def test_executable_generator_rule_records_success_contract() -> None:
    rule = executable_generator_rule_from_locked_spec(
        {
            "id": "seq_001",
            "source": "generator",
            "count": 1,
            "formula_probes": [
                {
                    "id": "seq_001_probe",
                    "template": "rocksalt",
                    "roles": {
                        "A": {"element": "Li", "oxidation_state": 1},
                        "X": {"element": "F", "oxidation_state": -1},
                    },
                }
            ],
            "template_consistency_audit": template_audit("rocksalt"),
        },
        {"formula": "LiF", "material_id": "generated::seq_001::LiF"},
        repair_round=2,
        preceding_errors=["previous schema error"],
    )

    assert rule["input_type"] == "formula_probe"
    assert rule["generator_template"] == "rocksalt"
    assert rule["repair_round"] == 2
    assert "formula_probe_strings" in generator_executable_schema_rules()["candidate_contract"]["exactly_one_input_field"]
    assert "validation_window" in generator_executable_schema_rules()["structure_dicts"]


def test_generator_template_only_schema_excludes_structure_dicts() -> None:
    rules = generator_executable_schema_rules(template_only=True)

    assert rules["candidate_contract"]["exactly_one_input_field"] == ["formula_probe_strings", "formula_probes"]
    assert rules["structure_dicts"]["accepted"] is False
    assert "structure_dicts" in rules["template_only"]["z_w_generator_inputs_forbidden"]


def test_mattergen_schema_uses_request_contract() -> None:
    rules = generator_executable_schema_rules(backend="mattergen")

    assert rules["candidate_contract"]["exactly_one_input_field"] == ["mattergen_requests"]
    assert rules["mattergen_requests"]["required_shape"]["backend"] == "mattergen"
    assert rules["mattergen_requests"]["required_shape"]["properties_to_condition_on"]["energy_above_hull"] == 0.0
    assert rules["mattergen_requests"]["required_shape"]["diffusion_guidance_factor"] == DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR
    assert rules["mattergen_requests"]["required_shape"]["filters"]["min_volume_per_atom"] == DEFAULT_MATTERGEN_MIN_VOLUME_PER_ATOM
    assert rules["mattergen_requests"]["required_shape"]["filters"]["max_volume_per_atom"] == DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM
    assert rules["mattergen_requests"]["required_shape"]["filters"]["require_chemical_system_exact"] is True
    assert rules["mattergen_requests"]["required_shape"]["filters"]["allowed_elements"] == [
        "same elements as chemical_system"
    ]
    assert rules["mattergen_requests"]["required_shape"]["filters"]["required_elements"] == [
        "same elements as chemical_system"
    ]
    assert "formula_probes" in rules["forbidden_when_mattergen"]


def test_normalize_accepts_mattergen_request_candidate() -> None:
    normalized, errors = normalize_candidate_spec(
        {
            "id": "mg_001",
            "source": "generator",
            "mattergen_requests": [
                {
                    "backend": "mattergen",
                    "properties_to_condition_on": {"chemical_system": "Na-Cl"},
                    "filters": {"chemical_system": ["Na", "Cl"], "max_sites": 20},
                }
            ],
            "template_consistency_audit": template_audit("mattergen"),
        },
        default_id="fallback",
        allowed_sources={"generator"},
    )

    assert errors == []
    assert normalized is not None
    request = normalized["mattergen_requests"][0]
    assert request["properties_to_condition_on"]["energy_above_hull"] == 0.0
    assert request["filters"]["chemical_system"] == ["Na", "Cl"]
    assert request["diffusion_guidance_factor"] == DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR


def test_mattergen_strategy_cooldowns_group_repeated_halide_basin_failures() -> None:
    memory = {
        "records": [
            {
                "iteration": 11,
                "status": "not_materialized",
                "candidate_spec": {
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {"chemical_system": "Rb-Cd-Br"},
                            "filters": {"chemical_system": ["Rb", "Cd", "Br"]},
                        }
                    ]
                },
                "materialization_errors": [
                    "mattergen materialized 0 records: generated RbCdBr3 is in known/training formula set"
                ],
            },
            {
                "iteration": 12,
                "status": "not_materialized",
                "candidate_spec": {
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {"chemical_system": "Cs-Cd-Br"},
                            "filters": {"chemical_system": ["Cs", "Cd", "Br"]},
                        }
                    ]
                },
                "materialization_errors": [
                    "mattergen materialized 0 records: generated CsCdBr3 is in known/training formula set"
                ],
            },
        ]
    }

    cooldowns = mattergen_strategy_cooldowns_from_memory(memory)

    blocked_systems = {item["chemical_system"] for item in cooldowns["blocked_chemical_systems"]}
    blocked_patterns = {item["family_pattern"] for item in cooldowns["blocked_family_patterns"]}
    assert {"Br-Cd-Rb", "Br-Cd-Cs"} <= blocked_systems
    assert "alkali-Cd-halide" in blocked_patterns
    assert cooldowns["policy"] == "hard_preflight"


def test_mattergen_strategy_cooldowns_ignore_successful_rescue_errors() -> None:
    memory = {
        "records": [
            {
                "iteration": 21,
                "status": "evaluated",
                "candidate_spec": {
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {"chemical_system": "Br-Mn-Rb"},
                            "filters": {"chemical_system": ["Br", "Mn", "Rb"]},
                        }
                    ]
                },
                "selected_record": {
                    "formula": "Rb4Mn3Br11",
                    "generator_properties_to_condition_on": {"chemical_system": "Br-Mn-Rb"},
                },
                "materialization_errors": [
                    "Z1: MatterGen materialized 0 records but requested 1; "
                    "chemical_system=Br-Mn-Rb; target_reduced_formula=Rb7Mn2Br11; "
                    "report_status=no_accepted_structures; accepted_count=0; "
                    "reject_reasons={'chemical_system_not_exact': 1, 'excluded_reduced_formula': 15}"
                ],
            }
        ]
    }

    assert mattergen_strategy_cooldowns_from_memory(memory) == {}


def test_mattergen_strategy_cooldowns_ignore_transient_current_iteration() -> None:
    memory = memory_with_transient_materialization_errors(
        {"records": []},
        [
            "Z1: MatterGen materialized 0 records but requested 1; "
            "chemical_system=Br-Mn-Rb; target_reduced_formula=Rb5MnBr7; "
            "report_status=no_accepted_structures; accepted_count=0; "
            "reject_reasons={'chemical_system_not_exact': 5, 'excluded_reduced_formula': 11}"
        ],
    )

    assert mattergen_strategy_cooldowns_from_memory(memory) == {}


def test_compact_strategy_cooldowns_prioritizes_high_failure_systems() -> None:
    cooldowns = {
        "policy": "hard_preflight",
        "blocked_chemical_systems": [
            {"chemical_system": f"dummy-{index}", "failure_count": 1, "source_iterations": [index]}
            for index in range(8)
        ]
        + [{"chemical_system": "Br-Rb-Zn", "failure_count": 8, "source_iterations": [40]}],
        "blocked_family_patterns": [],
    }

    compact = _compact_strategy_cooldowns_for_prompt(cooldowns)

    visible = [item["chemical_system"] for item in compact["blocked_chemical_systems"]]
    assert "Br-Rb-Zn" in visible
    assert visible[0] == "Br-Rb-Zn"


def test_mattergen_strategy_preflight_blocks_same_family_pattern() -> None:
    cooldowns = mattergen_strategy_cooldowns_from_memory(
        {
            "records": [
                {
                    "iteration": 11,
                    "status": "not_materialized",
                    "candidate_spec": {
                        "mattergen_requests": [
                            {
                                "backend": "mattergen",
                                "properties_to_condition_on": {"chemical_system": "Rb-Cd-Br"},
                                "filters": {"chemical_system": ["Rb", "Cd", "Br"]},
                            }
                        ]
                    },
                    "materialization_errors": [
                        "strategy_cooldown_block: candidate=old chemical_system=Br-Cd-Rb family_pattern=alkali-Cd-halide"
                    ],
                },
                {
                    "iteration": 12,
                    "status": "not_materialized",
                    "candidate_spec": {
                        "mattergen_requests": [
                            {
                                "backend": "mattergen",
                                "properties_to_condition_on": {"chemical_system": "Cs-Cd-Br"},
                                "filters": {"chemical_system": ["Cs", "Cd", "Br"]},
                            }
                        ]
                    },
                    "materialization_errors": [
                        "strategy_cooldown_block: candidate=old chemical_system=Br-Cd-Cs family_pattern=alkali-Cd-halide"
                    ],
                },
            ]
        }
    )

    errors = mattergen_strategy_cooldown_preflight_errors(
        [
            {
                "id": "mg_new",
                "source": "generator",
                "mattergen_requests": [
                    {
                        "backend": "mattergen",
                        "properties_to_condition_on": {"chemical_system": "K-Cd-Br"},
                        "filters": {"chemical_system": ["K", "Cd", "Br"]},
                    }
                ],
            }
        ],
        cooldowns,
    )

    assert errors
    assert any("strategy_cooldown_block" in item for item in errors)
    assert any("family_pattern=alkali-Cd-halide" in item for item in errors)


def test_xy_queue_validation_rejects_strategy_cooldown_formula() -> None:
    context = {
        "generator_backend": "mattergen",
        "candidate_count_this_iteration": 1,
        "controller_constraints": {
            "strategy_cooldowns": {
                "policy": "hard_preflight",
                "blocked_chemical_systems": [
                    {"chemical_system": "Br-Cd-Rb", "family_pattern": "alkali-Cd-halide"}
                ],
                "blocked_family_patterns": [],
            }
        },
    }
    payload = {
        "status": "material_description_consensus",
        "selected_candidate_id": "q1",
        "sun_candidate_queue": [
            {
                "candidate_id": "q1",
                "rank": 1,
                "reduced_formula": "Rb2CdBr4",
                "acquisition_mode": "exploit",
                "material_description": {"generator_template": "mattergen"},
            },
            {
                "candidate_id": "q2",
                "rank": 2,
                "reduced_formula": "NaCl",
                "acquisition_mode": "explore_adjacent",
                "material_description": {"generator_template": "mattergen"},
            },
        ],
        "material_description": {"generator_template": "mattergen"},
    }

    errors = validate_xy_sun_candidate_queue(payload, context=context)

    assert any("strategy_cooldowns.blocked_chemical_systems" in error for error in errors)
    assert any("chemical_system=Br-Cd-Rb" in error for error in errors)


def test_populate_sequential_controller_constraints_exposes_mattergen_strategy_cooldowns() -> None:
    context = {"generator_backend": "mattergen", "controller_constraints": {}}
    memory = {
        "records": [
            {
                "iteration": 11,
                "status": "not_materialized",
                "candidate_spec": {
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {"chemical_system": "Rb-Cd-Br"},
                            "filters": {"chemical_system": ["Rb", "Cd", "Br"]},
                        }
                    ]
                },
                "materialization_errors": [
                    "mattergen materialized 0 records: generated RbCdBr3 is in known/training formula set"
                ],
            }
        ]
    }

    populate_sequential_controller_constraints(context, memory)

    cooldowns = context["controller_constraints"]["strategy_cooldowns"]
    assert cooldowns["policy"] == "hard_preflight"
    assert cooldowns["blocked_chemical_systems"][0]["chemical_system"] == "Br-Cd-Rb"


def test_mattergen_run_request_defaults_guidance_factor() -> None:
    request = _normalise_mattergen_request_for_run(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Ce-H", "energy_above_hull": 0.0},
            "filters": {"chemical_system": ["Ce", "H"], "max_sites": 20},
        },
        candidate_id="mg_001",
        mattergen_config={
            "checkpoint": "chemical_system_energy_above_hull",
            "model_path": "/tmp/fake_mattergen_model",
            "target_count": 4,
            "batch_size": 8,
            "num_batches": 1,
            "max_sites": 20,
            "diffusion_guidance_factor": DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR,
        },
        excluded_formulas=set(),
        max_sites=20,
        count=1,
    )

    assert request["diffusion_guidance_factor"] == DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR
    assert request["filters"]["require_chemical_system_exact"] is True
    assert request["filters"]["allowed_elements"] == ["Ce", "H"]
    assert request["filters"]["required_elements"] == ["Ce", "H"]
    assert request["filters"]["min_volume_per_atom"] == DEFAULT_MATTERGEN_MIN_VOLUME_PER_ATOM
    assert request["filters"]["max_volume_per_atom"] == DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM


def test_mattergen_run_request_normalizes_dict_chemical_system_filter() -> None:
    request = _normalise_mattergen_request_for_run(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Ag-Br-Rb", "energy_above_hull": 0.0},
            "filters": {
                "chemical_system": {"allowed": ["Ag", "Br", "Rb"], "required": ["Ag", "Br", "Rb"]},
                "max_sites": 20,
            },
        },
        candidate_id="mg_001",
        mattergen_config={
            "checkpoint": "chemical_system_energy_above_hull",
            "model_path": "/tmp/fake_mattergen_model",
            "target_count": 4,
            "batch_size": 8,
            "num_batches": 2,
            "max_sites": 20,
            "min_volume_per_atom": 4.0,
            "max_volume_per_atom": 45.0,
            "diffusion_guidance_factor": 1.0,
        },
        excluded_formulas=set(),
        max_sites=20,
        count=1,
    )

    assert request["filters"]["chemical_system"] == ["Ag", "Br", "Rb"]
    assert request["filters"]["allowed_elements"] == ["Ag", "Br", "Rb"]
    assert request["filters"]["required_elements"] == ["Ag", "Br", "Rb"]
    assert request["filters"]["require_chemical_system_exact"] is True


def test_mattergen_candidate_level_target_is_copied_into_request() -> None:
    candidate, errors = normalize_candidate_spec(
        {
            "candidate_id": "Z2-1",
            "source": "generator",
            "count": 1,
            "target_reduced_formula": "RbMn2Br5",
            "require_target_reduced_formula": False,
            "mattergen_requests": [
                {
                    "backend": "mattergen",
                    "properties_to_condition_on": {"chemical_system": "Br-Mn-Rb", "energy_above_hull": 0.0},
                    "filters": {
                        "chemical_system": "Br-Mn-Rb",
                        "allowed_elements": ["Br", "Mn", "Rb"],
                        "required_elements": ["Br", "Mn", "Rb"],
                        "max_sites": 20,
                    },
                }
            ],
            "template_consistency_audit": {
                "chosen_template": "mattergen",
                "template_realizes_motif": True,
                "unsupported_motif_substitution": False,
                "mechanism_local_motif": "Rb-Mn-Br full-structure generation",
                "required_coordination_or_polyhedra": "Mn-Br coordination",
                "why_template_is_faithful": "MatterGen directly samples the requested chemical system.",
            },
        },
        default_id="Z2-1",
        allowed_sources={"generator"},
        allow_structure_dicts=False,
    )

    assert errors == []
    assert candidate is not None
    request = candidate["mattergen_requests"][0]
    assert request["target_reduced_formula"] == "RbMn2Br5"
    assert request["filters"]["target_reduced_formula"] == "RbMn2Br5"
    assert request["filters"]["require_target_reduced_formula"] is False


def test_mattergen_run_request_falls_back_to_candidate_level_target() -> None:
    request = _normalise_mattergen_request_for_run(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Br-Mn-Rb", "energy_above_hull": 0.0},
            "filters": {"chemical_system": ["Br", "Mn", "Rb"], "max_sites": 20},
        },
        candidate_id="Z2-1",
        mattergen_config={
            "checkpoint": "chemical_system_energy_above_hull",
            "model_path": "/tmp/fake_mattergen_model",
            "target_count": 4,
            "batch_size": 8,
            "num_batches": 2,
            "max_sites": 20,
            "min_volume_per_atom": 4.0,
            "max_volume_per_atom": 45.0,
            "diffusion_guidance_factor": 1.0,
            "force_soft_target_reduced_formula": True,
        },
        excluded_formulas=set(),
        max_sites=20,
        count=1,
        candidate_target_reduced_formula="RbMn2Br5",
        candidate_require_target_reduced_formula=False,
    )

    assert request["target_reduced_formula"] == "RbMn2Br5"
    assert request["filters"]["target_reduced_formula"] == "RbMn2Br5"
    assert request["filters"]["require_target_reduced_formula"] is False


def test_materialize_mattergen_request_includes_hidden_excluded_formulas(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_mattergen_request(request: dict[str, object], *, candidate_id: str, mattergen_config: dict[str, object]):
        captured["request"] = request
        return [], [], {"status": "ok", "accepted_count": 0}

    monkeypatch.setattr(xy_debate, "_run_mattergen_request", fake_run_mattergen_request)

    materialize_candidate_specs(
        [],
        [
            {
                "candidate_id": "Z1",
                "source": "generator",
                "count": 1,
                "target_reduced_formula": "RbMn2Br5",
                "mattergen_requests": [
                    {
                        "backend": "mattergen",
                        "properties_to_condition_on": {"chemical_system": "Br-Mn-Rb", "energy_above_hull": 0.0},
                        "filters": {
                            "chemical_system": ["Br", "Mn", "Rb"],
                            "target_reduced_formula": "RbMn2Br5",
                            "exclude_reduced_formulas": ["Rb2MnBr5"],
                            "max_sites": 20,
                        },
                    }
                ],
                "template_consistency_audit": {
                    "chosen_template": "mattergen",
                    "template_realizes_motif": True,
                    "unsupported_motif_substitution": False,
                    "mechanism_local_motif": "Rb-Mn-Br full-structure generation",
                    "required_coordination_or_polyhedra": "Mn-Br coordination",
                    "why_template_is_faithful": "MatterGen directly samples the requested chemical system.",
                },
            }
        ],
        target_count=1,
        seed=1,
        max_sites=20,
        known_formulas=None,
        allowed_sources={"generator"},
        mattergen_config={
            "checkpoint": "chemical_system_energy_above_hull",
            "model_path": "/tmp/fake_mattergen_model",
            "runner": "local",
            "run_dir": "/tmp/fake_mattergen_run",
            "target_count": 4,
            "batch_size": 8,
            "num_batches": 2,
            "max_sites": 20,
            "min_volume_per_atom": 4.0,
            "max_volume_per_atom": 45.0,
            "diffusion_guidance_factor": 1.0,
            "force_soft_target_reduced_formula": True,
        },
        additional_excluded_formulas={"Rb3Mn4Br13", "RbMnBr3"},
    )

    request = captured["request"]
    assert isinstance(request, dict)
    excluded = set(request["filters"]["exclude_reduced_formulas"])
    assert {"Rb2MnBr5", "Rb3Mn4Br13", "RbMnBr3"}.issubset(excluded)
    assert request["filters"]["target_reduced_formula"] == "RbMn2Br5"


def test_mattergen_run_request_forces_configured_guidance_factor() -> None:
    request = _normalise_mattergen_request_for_run(
        {
            "backend": "mattergen",
            "diffusion_guidance_factor": 2.0,
            "properties_to_condition_on": {"chemical_system": "Ag-Br-Rb", "energy_above_hull": 0.0},
            "filters": {"chemical_system": ["Ag", "Br", "Rb"], "max_sites": 20},
        },
        candidate_id="mg_001",
        mattergen_config={
            "checkpoint": "chemical_system_energy_above_hull",
            "model_path": "/tmp/fake_mattergen_model",
            "target_count": 4,
            "batch_size": 8,
            "num_batches": 2,
            "max_sites": 20,
            "min_volume_per_atom": 4.0,
            "max_volume_per_atom": 45.0,
            "diffusion_guidance_factor": 1.0,
        },
        excluded_formulas=set(),
        max_sites=20,
        count=1,
    )

    assert request["diffusion_guidance_factor"] == 1.0
    assert "forced diffusion_guidance_factor" in " ".join(request["controller_normalization_notes"])


def test_xy_density_volume_cap_detects_high_volume_near_miss() -> None:
    cap = _xy_density_volume_cap_from_material_description(
        {
            "material_description": {
                "known_risks": ["Prior near_miss had high-volume packing near the volume_per_atom ceiling."],
                "natural_language_description": "dense compact Cs-Hg-Br bromide",
            }
        }
    )

    assert cap == DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM


def test_xy_density_volume_cap_ignores_plain_mattergen_description() -> None:
    assert (
        _xy_density_volume_cap_from_material_description(
            {"material_description": {"natural_language_description": "exact Na-Cl MatterGen search"}}
        )
        is None
    )


def test_mattergen_run_request_applies_xy_density_cap() -> None:
    base_config = {
        "checkpoint": "chemical_system_energy_above_hull",
        "model_path": "/tmp/fake_mattergen_model",
        "target_count": 4,
        "batch_size": 8,
        "num_batches": 1,
        "max_sites": 20,
        "max_volume_per_atom": DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM,
        "diffusion_guidance_factor": DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR,
    }
    capped_config = _mattergen_config_with_xy_density_cap(
        base_config,
        {
            "material_description": {
                "known_risks": ["high-volume Cs-rich polymorphs"],
                "history_lessons_used": ["prior near_miss had volume_per_atom near 45"],
            }
        },
    )

    request = _normalise_mattergen_request_for_run(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Br-Cs-Hg", "energy_above_hull": 0.0},
            "filters": {
                "chemical_system": ["Br", "Cs", "Hg"],
                "max_sites": 20,
                "max_volume_per_atom": DEFAULT_MATTERGEN_MAX_VOLUME_PER_ATOM,
            },
        },
        candidate_id="mg_001",
        mattergen_config=capped_config,
        excluded_formulas=set(),
        max_sites=20,
        count=1,
    )

    assert request["filters"]["max_volume_per_atom"] == DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM
    assert request["max_volume_per_atom"] == DEFAULT_MATTERGEN_DENSITY_EDGE_MAX_VOLUME_PER_ATOM
    assert any("controller_density_volume_cap" in note for note in request["controller_normalization_notes"])


def test_mattergen_run_request_preserves_excluded_reduced_formulas() -> None:
    request = _normalise_mattergen_request_for_run(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Br-Hg-Rb", "energy_above_hull": 0.0},
            "filters": {
                "chemical_system": ["Br", "Hg", "Rb"],
                "exclude_reduced_formulas": ["RbHg2Br"],
                "max_sites": 20,
            },
        },
        candidate_id="mg_001",
        mattergen_config={
            "checkpoint": "chemical_system_energy_above_hull",
            "model_path": "/tmp/fake_mattergen_model",
            "target_count": 4,
            "batch_size": 8,
            "num_batches": 1,
            "max_sites": 20,
            "diffusion_guidance_factor": DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR,
        },
        excluded_formulas={"RbHgBr2"},
        max_sites=20,
        count=1,
    )

    excluded = set(request["filters"]["exclude_reduced_formulas"])
    assert {"RbHg2Br", "RbHgBr2"}.issubset(excluded)
    assert not {"Rb", "Hg", "Br"}.intersection(excluded)


def test_mattergen_run_request_allows_explicit_subset_fallback_only() -> None:
    request = _normalise_mattergen_request_for_run(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Rb-Cd-Zn-Br", "energy_above_hull": 0.0},
            "filters": {
                "chemical_system": ["Rb", "Cd", "Zn", "Br"],
                "allow_subset_fallback": True,
                "max_sites": 20,
            },
        },
        candidate_id="mg_001",
        mattergen_config={
            "checkpoint": "chemical_system_energy_above_hull",
            "model_path": "/tmp/fake_mattergen_model",
            "target_count": 4,
            "batch_size": 8,
            "num_batches": 1,
            "max_sites": 20,
            "diffusion_guidance_factor": DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR,
        },
        excluded_formulas=set(),
        max_sites=20,
        count=1,
    )

    assert request["filters"]["require_chemical_system_exact"] is False
    assert request["filters"]["allowed_elements"] == ["Rb", "Cd", "Zn", "Br"]
    assert "required_elements" not in request["filters"]


def test_mattergen_run_request_treats_target_formula_as_soft_by_default() -> None:
    request = _normalise_mattergen_request_for_run(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Yb-Si", "energy_above_hull": 0.0},
            "filters": {
                "chemical_system": ["Yb", "Si"],
                "target_reduced_formula": "Yb2Si",
                "max_sites": 20,
            },
        },
        candidate_id="mg_001",
        mattergen_config={
            "checkpoint": "chemical_system_energy_above_hull",
            "model_path": "/tmp/fake_mattergen_model",
            "target_count": 4,
            "batch_size": 8,
            "num_batches": 2,
            "max_sites": 20,
            "diffusion_guidance_factor": DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR,
            "allow_hard_target_formula": False,
        },
        excluded_formulas=set(),
        max_sites=20,
        count=1,
    )

    assert request["filters"]["target_reduced_formula"] == "Yb2Si"
    assert request["filters"]["require_target_reduced_formula"] is False


def test_mattergen_run_request_preserves_explicit_hard_target_formula() -> None:
    request = _normalise_mattergen_request_for_run(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Yb-Si", "energy_above_hull": 0.0},
            "filters": {
                "chemical_system": ["Yb", "Si"],
                "target_reduced_formula": "Yb2Si",
                "require_target_reduced_formula": True,
                "max_sites": 20,
            },
        },
        candidate_id="mg_001",
        mattergen_config={
            "checkpoint": "chemical_system_energy_above_hull",
            "model_path": "/tmp/fake_mattergen_model",
            "target_count": 4,
            "batch_size": 8,
            "num_batches": 2,
            "max_sites": 20,
            "diffusion_guidance_factor": DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR,
            "allow_hard_target_formula": False,
        },
        excluded_formulas=set(),
        max_sites=20,
        count=1,
    )

    assert request["filters"]["require_target_reduced_formula"] is True
    assert request["num_batches"] == DEFAULT_MATTERGEN_HARD_TARGET_MIN_NUM_BATCHES


def test_sequential_xy_mattergen_config_forces_soft_target_formula() -> None:
    base_config = {
        "checkpoint": "chemical_system_energy_above_hull",
        "model_path": "/tmp/fake_mattergen_model",
        "target_count": 4,
        "batch_size": 8,
        "num_batches": 2,
        "max_sites": 20,
        "diffusion_guidance_factor": DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR,
    }
    sequential_config = _mattergen_config_with_xy_density_cap(base_config, {"material_description": {}})

    request = _normalise_mattergen_request_for_run(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Rb-Cd-Br", "energy_above_hull": 0.0},
            "filters": {
                "chemical_system": ["Rb", "Cd", "Br"],
                "target_reduced_formula": "Rb4Cd3Br10",
                "require_target_reduced_formula": True,
                "max_sites": 20,
            },
        },
        candidate_id="mg_001",
        mattergen_config=sequential_config,
        excluded_formulas=set(),
        max_sites=20,
        count=1,
    )

    assert request["filters"]["target_reduced_formula"] == "Rb4Cd3Br10"
    assert request["filters"]["require_target_reduced_formula"] is False
    assert "controller_forced_soft_target_formula" in " ".join(request["controller_normalization_notes"])


def test_mattergen_run_request_softens_four_element_hard_target_formula() -> None:
    request = _normalise_mattergen_request_for_run(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Br-Cd-K-Rb", "energy_above_hull": 0.0},
            "filters": {
                "chemical_system": ["Br", "Cd", "K", "Rb"],
                "target_reduced_formula": "K2RbCd3Br9",
                "require_target_reduced_formula": True,
                "max_sites": 20,
            },
        },
        candidate_id="mg_001",
        mattergen_config={
            "checkpoint": "chemical_system_energy_above_hull",
            "model_path": "/tmp/fake_mattergen_model",
            "target_count": 4,
            "batch_size": 8,
            "num_batches": 2,
            "max_sites": 20,
            "diffusion_guidance_factor": DEFAULT_MATTERGEN_DIFFUSION_GUIDANCE_FACTOR,
            "allow_hard_target_formula": False,
        },
        excluded_formulas=set(),
        max_sites=20,
        count=1,
    )

    assert len(request["filters"]["chemical_system"]) > DEFAULT_MATTERGEN_HARD_TARGET_MAX_EXACT_ELEMENTS
    assert request["filters"]["target_reduced_formula"] == "K2RbCd3Br9"
    assert request["filters"]["require_target_reduced_formula"] is False
    assert request["num_batches"] == DEFAULT_MATTERGEN_HARD_TARGET_MIN_NUM_BATCHES
    assert "controller_softened_hard_target_formula" in " ".join(request["controller_normalization_notes"])


def test_mattergen_slurm_held_pending_reason_fails_fast(tmp_path, monkeypatch) -> None:
    class Completed:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command, **kwargs):
        if command[0] == "sbatch":
            return Completed(0, "2185207\n")
        if command[0] == "squeue":
            return Completed(0, "PENDING|launch_failed_requeued_held\n")
        raise AssertionError(command)

    monkeypatch.setattr(xy_debate.subprocess, "run", fake_run)
    monkeypatch.setattr(xy_debate.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="launch_failed_requeued_held"):
        xy_debate._submit_and_wait_mattergen(
            tmp_path / "run_mattergen.sbatch",
            mattergen_config={"root": str(tmp_path), "job_timeout": 0, "poll_sec": 1},
            run_dir=tmp_path,
        )


def test_normalize_promotes_nested_mattergen_template_audit() -> None:
    normalized, errors = normalize_candidate_spec(
        {
            "id": "mg_nested_audit",
            "source": "generator",
            "mattergen_requests": [
                {
                    "backend": "mattergen",
                    "properties_to_condition_on": {"chemical_system": "Eu-Si", "energy_above_hull": 0.0},
                    "filters": {"chemical_system": "Eu-Si", "target_reduced_formula": "Eu2Si", "max_sites": 20},
                    "diffusion_guidance_factor": 1.0,
                    "template_consistency_audit": {
                        "chosen_template": "mattergen",
                        "is_faithful": True,
                        "audit": "MatterGen samples full Eu-Si periodic structures for the Eu2Si Zintl motif.",
                    },
                }
            ],
        },
        default_id="fallback",
        allowed_sources={"generator"},
    )

    assert errors == []
    assert normalized is not None
    audit = normalized["template_consistency_audit"]
    assert audit["chosen_template"] == "mattergen"
    assert audit["template_realizes_motif"] is True
    assert audit["unsupported_motif_substitution"] is False
    assert audit["why_template_is_faithful"].startswith("MatterGen samples full Eu-Si")


def test_normalize_completes_partial_mattergen_template_audit() -> None:
    normalized, errors = normalize_candidate_spec(
        {
            "id": "mg_partial_audit",
            "source": "generator",
            "mattergen_requests": [
                {
                    "backend": "mattergen",
                    "properties_to_condition_on": {"chemical_system": "Rb-Cd-Br", "energy_above_hull": 0.0},
                    "filters": {
                        "chemical_system": ["Rb", "Cd", "Br"],
                        "target_reduced_formula": "Rb4Cd3Br10",
                        "require_target_reduced_formula": True,
                        "max_sites": 20,
                    },
                }
            ],
            "template_consistency_audit": {
                "chosen_template": "mattergen",
                "why_template_is_faithful": "MatterGen samples full Rb-Cd-Br periodic bromocadmate structures.",
            },
        },
        default_id="fallback",
        allowed_sources={"generator"},
    )

    assert errors == []
    assert normalized is not None
    audit = normalized["template_consistency_audit"]
    assert audit["chosen_template"] == "mattergen"
    assert audit["template_realizes_motif"] is True
    assert audit["unsupported_motif_substitution"] is False
    assert audit["mechanism_local_motif"]
    assert audit["required_coordination_or_polyhedra"]
    assert normalized["mattergen_requests"][0]["filters"]["require_target_reduced_formula"] is True
    assert "completed MatterGen template_consistency_audit defaults" in normalized["normalization_notes"]


def test_normalize_maps_mattergen_audit_note_aliases() -> None:
    normalized, errors = normalize_candidate_spec(
        {
            "candidate_id": "X2-1",
            "source": "generator",
            "mattergen_requests": [
                {
                    "backend": "mattergen",
                    "properties_to_condition_on": {"chemical_system": "Ag-Br-Rb", "energy_above_hull": 0.0},
                    "filters": {"chemical_system": "Ag-Br-Rb", "max_sites": 20},
                }
            ],
            "template_consistency_audit": {
                "chosen_template": "mattergen",
                "template_realizes_motif": True,
                "unsupported_motif_substitution": False,
                "motif_notes": "MatterGen samples full Ag-Br/Rb cavity structures.",
                "polyhedra_notes": "Ag-Br coordination polyhedra and Rb ionic cavities.",
                "faithfulness_notes": "The request is a full-structure MatterGen sample, not a hand-built proxy.",
            },
        },
        default_id="fallback",
        allowed_sources={"generator"},
    )

    assert errors == []
    assert normalized is not None
    assert normalized["id"] == "X2-1"
    audit = normalized["template_consistency_audit"]
    assert audit["mechanism_local_motif"].startswith("MatterGen samples")
    assert audit["required_coordination_or_polyhedra"].startswith("Ag-Br coordination")
    assert audit["why_template_is_faithful"].startswith("The request is")
    assert "motif_notes" not in audit
    assert "polyhedra_notes" not in audit
    assert "faithfulness_notes" not in audit


def test_compact_candidate_payload_preserves_mattergen_soft_target_and_audit_aliases() -> None:
    compact = compact_candidate_payload_for_prompt(
        {
            "status": "candidate_proposal",
            "agent": "Z",
            "iteration": 2,
            "candidate_specs": [
                {
                    "candidate_id": "X2-1",
                    "source": "generator",
                    "count": 1,
                    "target_reduced_formula": "Rb2AgBr3",
                    "require_target_reduced_formula": False,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Ag-Br-Rb",
                                "energy_above_hull": 0.0,
                            },
                            "target_reduced_formula": "Rb2AgBr3",
                            "require_target_reduced_formula": False,
                            "filters": {
                                "chemical_system": "Ag-Br-Rb",
                                "allowed_elements": ["Ag", "Br", "Rb"],
                                "required_elements": ["Ag", "Br", "Rb"],
                                "exclude_reduced_formulas": [
                                    "RbAgBr3",
                                    "RbCu3Br4",
                                    "RbCu2Br3",
                                    "RbHg2Br3",
                                    "Rb(CdBr2)3",
                                    "Rb3(CdBr5)2",
                                    "RbZrBr3",
                                    "CsZrBr3",
                                    "RbZnBr3",
                                ],
                            },
                        }
                    ],
                    "template_consistency_audit": {
                        "chosen_template": "mattergen",
                        "template_realizes_motif": True,
                        "unsupported_motif_substitution": False,
                        "motif_notes": "MatterGen full-structure Ag-Br/Rb cavity generation.",
                        "polyhedra_notes": "Ag-Br coordination environments.",
                        "faithfulness_notes": "Preserves the selected Ag-Br-Rb chemical system as a soft target.",
                    },
                }
            ],
        }
    )

    spec = compact["candidate_specs"][0]
    assert spec["id"] == "X2-1"
    assert spec["candidate_id"] == "X2-1"
    assert spec["require_target_reduced_formula"] is False
    audit = spec["template_consistency_audit"]
    assert audit["mechanism_local_motif"].startswith("MatterGen full-structure")
    assert audit["required_coordination_or_polyhedra"].startswith("Ag-Br coordination")
    assert audit["why_template_is_faithful"].startswith("Preserves the selected")
    request = spec["mattergen_requests"][0]
    assert request["require_target_reduced_formula"] is False
    assert "exclude_reduced_formulas_tail" not in request["filters"]
    assert "exclude_reduced_formulas_omitted_count" not in request["filters"]
    assert request["filters"]["exclude_reduced_formulas"] == [
        "RbCu3Br4",
        "RbCu2Br3",
        "RbHg2Br3",
        "Rb(CdBr2)3",
        "Rb3(CdBr5)2",
        "RbZrBr3",
        "CsZrBr3",
        "RbZnBr3",
    ]


def test_two_stage_normalize_requires_design_rule_ids() -> None:
    normalized, errors = normalize_candidate_spec(
        {
            "id": "zw_s001_c001",
            "formula_probe_strings": ["template=rocksalt;A=Li:+1;X=F:-1;family=halide"],
            "cited_principle_ids": ["principle_program_001"],
            "template_consistency_audit": template_audit("rocksalt"),
        },
        default_id="fallback",
        allowed_sources={"generator"},
        require_design_rule_ids=True,
    )

    assert normalized is None
    assert any("design_rule_ids" in error for error in errors)


def test_two_stage_materialize_accepts_design_rule_grounded_candidate() -> None:
    selected, locked_specs, errors = materialize_candidate_specs(
        [],
        [
            {
                "id": "zw_s001_c001",
                "formula_probe_strings": ["template=rocksalt;A=Li:+1;X=F:-1;family=halide"],
                "design_rule_ids": ["xy_s001_r001"],
                "cited_principle_ids": ["principle_program_001"],
                "template_consistency_audit": template_audit("rocksalt"),
            }
        ],
        target_count=1,
        seed=123,
        max_sites=80,
        known_formulas=None,
        allowed_sources={"generator"},
        require_design_rule_ids=True,
    )

    assert errors == []
    assert len(selected) == 1
    assert locked_specs[0]["design_rule_ids"] == ["xy_s001_r001"]


def test_generator_only_normalize_requires_template_consistency_audit() -> None:
    normalized, errors = normalize_candidate_spec(
        {
            "id": "xy_s001_c001",
            "formula_probe_strings": ["template=rocksalt;A=Li:+1;X=F:-1;family=halide"],
            "cited_principle_ids": ["principle_program_001"],
        },
        default_id="fallback",
        allowed_sources={"generator"},
    )

    assert normalized is None
    assert any("template_consistency_audit" in error for error in errors)


def test_generator_only_normalize_rejects_template_audit_mismatch() -> None:
    bad_audit = template_audit("perovskite")
    normalized, errors = normalize_candidate_spec(
        {
            "id": "xy_s001_c001",
            "formula_probe_strings": ["template=rocksalt;A=Li:+1;X=F:-1;family=halide"],
            "cited_principle_ids": ["principle_program_001"],
            "template_consistency_audit": bad_audit,
        },
        default_id="fallback",
        allowed_sources={"generator"},
    )

    assert normalized is None
    assert any("does not match generator template" in error for error in errors)


def test_generator_only_normalize_rejects_template_proxy_audit() -> None:
    proxy_audit = {
        **template_audit("delafossite"),
        "mechanism_local_motif": "compact Si-N nitridosilicate network",
        "why_template_is_faithful": "delafossite is a stoichiometry proxy for ABN2 role balance",
    }
    normalized, errors = normalize_candidate_spec(
        {
            "id": "xy_s001_c001",
            "formula_probe_strings": ["template=delafossite;A=Ca:+2;B=Si:+4;X=N:-3;family=nitridosilicate"],
            "cited_principle_ids": ["principle_program_001"],
            "template_consistency_audit": proxy_audit,
        },
        default_id="fallback",
        allowed_sources={"generator"},
    )

    assert normalized is None
    assert any("stoichiometry proxy" in error for error in errors)


def test_generator_template_only_normalize_rejects_structure_dicts() -> None:
    normalized, errors = normalize_candidate_spec(
        {
            "id": "xy_s001_c001",
            "source": "generator",
            "structure_dicts": [{"@module": "pymatgen.core.structure", "@class": "Structure"}],
            "cited_principle_ids": ["principle_program_001"],
            "template_consistency_audit": template_audit("custom_structure"),
        },
        default_id="fallback",
        allowed_sources={"generator"},
        allow_structure_dicts=False,
    )

    assert normalized is None
    assert any("template-only run forbids structure_dicts" in error for error in errors)


def test_materialize_candidate_specs_generator_only_formula_probe_string() -> None:
    selected, locked_specs, errors = materialize_candidate_specs(
        [],
        [
            {
                "id": "xy_s001_c001",
                "formula_probe_strings": ["template=rocksalt;A=Li:+1;X=F:-1;family=halide"],
                "cited_principle_ids": ["principle_program_001"],
                "template_consistency_audit": template_audit("rocksalt"),
            }
        ],
        target_count=1,
        seed=123,
        max_sites=80,
        known_formulas=None,
        allowed_sources={"generator"},
    )

    assert errors == []
    assert len(selected) == 1
    assert selected[0]["crystal_llm_source"] == "generator"
    assert selected[0]["formula"] == "LiF"
    assert locked_specs[0]["source"] == "generator"


def test_materialize_candidate_specs_mattergen_from_existing_cif(tmp_path) -> None:
    root = Path.cwd()
    source_structure = Structure(
        Lattice.cubic(3.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    cif_path = tmp_path / "nacl.cif"
    source_structure.to(filename=str(cif_path))
    mattergen_config = {
        "root": str(root),
        "work_dir": str(tmp_path / "mattergen_work"),
        "adapter_path": str(root / "mattergen_backend_prototype" / "mattergen_adapter.py"),
        "mattergen_root": str(root),
        "mattergen_bin": "/bin/true",
        "model_path": str(tmp_path / "fake_model"),
        "checkpoint": "chemical_system_energy_above_hull",
        "target_count": 1,
        "batch_size": 1,
        "num_batches": 1,
        "max_sites": 20,
        "runner": "local",
        "from_existing": [str(cif_path)],
    }

    selected, locked_specs, errors = materialize_candidate_specs(
        [],
        [
            {
                "id": "mg_001",
                "source": "generator",
                "count": 1,
                "mattergen_requests": [
                    {
                        "backend": "mattergen",
                        "properties_to_condition_on": {"chemical_system": "Na-Cl", "energy_above_hull": 0.0},
                        "filters": {"chemical_system": ["Na", "Cl"], "max_sites": 20},
                    }
                ],
                "template_consistency_audit": template_audit("mattergen"),
            }
        ],
        target_count=1,
        seed=123,
        max_sites=20,
        known_formulas=None,
        allowed_sources={"generator"},
        mattergen_config=mattergen_config,
    )

    assert errors == []
    assert len(selected) == 1
    assert selected[0]["formula"] == "NaCl"
    assert selected[0]["crystal_llm_generator_backend"] == "mattergen"
    assert locked_specs[0]["mattergen_requests"][0]["backend"] == "mattergen"

    input_path = tmp_path / "input.json"
    write_input_structures(selected, input_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    properties = payload[0]["properties"]
    assert properties["crystal_llm_generator_backend"] == "mattergen"
    assert properties["crystal_llm_generator_mattergen_requests"][0]["backend"] == "mattergen"


def test_materialize_candidate_specs_mattergen_batch_target_overrides_count_one(monkeypatch, tmp_path) -> None:
    structures = [
        Structure(
            Lattice.cubic(3.2),
            ["Na", "Cl"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        ),
        Structure(
            Lattice.cubic(3.4),
            ["Na", "Na", "Cl"],
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.5]],
        ),
    ]
    captured: dict[str, object] = {}

    def fake_run_mattergen_request(request, *, candidate_id, mattergen_config):
        captured["request"] = dict(request)
        return (
            [structure.as_dict() for structure in structures],
            [
                {"material_id": "adapter::nacl", "formula": "NaCl"},
                {"material_id": "adapter::na2cl", "formula": "Na2Cl"},
            ],
            {"status": "success", "accepted_count": 2, "accepted_formulas": ["NaCl", "Na2Cl"]},
        )

    monkeypatch.setattr(xy_debate, "_run_mattergen_request", fake_run_mattergen_request)
    mattergen_config = {
        "root": str(Path.cwd()),
        "work_dir": str(tmp_path / "mattergen_work"),
        "model_path": str(tmp_path / "fake_model"),
        "checkpoint": "chemical_system_energy_above_hull",
        "target_count": 2,
        "batch_size": 4,
        "num_batches": 3,
        "max_sites": 20,
        "runner": "local",
    }

    selected, locked_specs, errors = materialize_candidate_specs(
        [],
        [
            {
                "id": "mg_batch",
                "source": "generator",
                "count": 1,
                "mattergen_requests": [
                    {
                        "backend": "mattergen",
                        "properties_to_condition_on": {"chemical_system": "Na-Cl", "energy_above_hull": 0.0},
                        "batch_size": 1,
                        "num_batches": 1,
                        "filters": {"chemical_system": ["Na", "Cl"], "max_sites": 20},
                    }
                ],
                "template_consistency_audit": template_audit("mattergen"),
            }
        ],
        target_count=2,
        seed=123,
        max_sites=20,
        known_formulas=None,
        allowed_sources={"generator"},
        mattergen_config=mattergen_config,
    )

    assert errors == []
    assert [item["formula"] for item in selected] == ["NaCl", "Na2Cl"]
    assert locked_specs[0]["count"] == 2
    assert captured["request"]["target_count"] == 2
    assert captured["request"]["batch_size"] == 4
    assert captured["request"]["num_batches"] == 3
    assert "raised MatterGen batch_size/num_batches" in captured["request"]["controller_normalization_notes"][0]


def test_materialize_candidate_specs_rejects_mattergen_subset_element_drift(monkeypatch, tmp_path) -> None:
    drift_structure = Structure(
        Lattice.cubic(5.0),
        ["Zn", "Cd", "Br", "Br"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
    )

    def fake_run_mattergen_request(request, *, candidate_id, mattergen_config):
        return (
            [drift_structure.as_dict()],
            [{"material_id": "adapter::subset_drift", "formula": "ZnCdBr2"}],
            {"status": "success", "accepted_count": 1, "accepted_formulas": ["ZnCdBr2"]},
        )

    monkeypatch.setattr(xy_debate, "_run_mattergen_request", fake_run_mattergen_request)
    mattergen_config = {
        "root": str(Path.cwd()),
        "work_dir": str(tmp_path / "mattergen_work"),
        "model_path": str(tmp_path / "fake_model"),
        "checkpoint": "chemical_system_energy_above_hull",
        "target_count": 1,
        "batch_size": 1,
        "num_batches": 1,
        "max_sites": 20,
        "runner": "local",
    }

    selected, locked_specs, errors = materialize_candidate_specs(
        [],
        [
            {
                "id": "mg_001",
                "source": "generator",
                "count": 1,
                "mattergen_requests": [
                    {
                        "backend": "mattergen",
                        "properties_to_condition_on": {
                            "chemical_system": "Rb-Cd-Zn-Br",
                            "energy_above_hull": 0.0,
                        },
                        "filters": {
                            "chemical_system": ["Rb", "Cd", "Zn", "Br"],
                            "require_chemical_system_exact": False,
                            "required_elements": ["Rb", "Cd", "Zn", "Br"],
                            "max_sites": 20,
                        },
                    }
                ],
                "template_consistency_audit": template_audit("mattergen"),
            }
        ],
        target_count=1,
        seed=123,
        max_sites=20,
        known_formulas=None,
        allowed_sources={"generator"},
        mattergen_config=mattergen_config,
    )

    assert selected == []
    assert locked_specs == []
    assert any(
        "rejected by element guard" in error
        and ("missing_required_elements" in error or "chemical_system_not_exact" in error)
        for error in errors
    )


def test_materialize_candidate_specs_skips_known_mattergen_formula_before_truncation(monkeypatch, tmp_path) -> None:
    duplicate_structure = Structure(
        Lattice.cubic(3.2),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    novel_structure = Structure(
        Lattice.cubic(3.2),
        ["Na", "Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.5]],
    )

    def fake_run_mattergen_request(request, *, candidate_id, mattergen_config):
        return (
            [duplicate_structure.as_dict(), novel_structure.as_dict()],
            [{"material_id": "adapter::duplicate", "formula": "NaCl"}, {"material_id": "adapter::novel", "formula": "Na2Cl"}],
            {"status": "success", "accepted_count": 2, "accepted_formulas": ["NaCl", "Na2Cl"]},
        )

    monkeypatch.setattr(xy_debate, "_run_mattergen_request", fake_run_mattergen_request)
    mattergen_config = {
        "root": str(Path.cwd()),
        "work_dir": str(tmp_path / "mattergen_work"),
        "model_path": str(tmp_path / "fake_model"),
        "checkpoint": "chemical_system_energy_above_hull",
        "target_count": 2,
        "batch_size": 2,
        "num_batches": 1,
        "max_sites": 20,
        "runner": "local",
    }

    selected, locked_specs, errors = materialize_candidate_specs(
        [],
        [
            {
                "id": "mg_001",
                "source": "generator",
                "count": 1,
                "mattergen_requests": [
                    {
                        "backend": "mattergen",
                        "properties_to_condition_on": {
                            "chemical_system": "Na-Cl",
                            "energy_above_hull": 0.0,
                        },
                        "filters": {"chemical_system": ["Na", "Cl"], "max_sites": 20},
                    }
                ],
                "template_consistency_audit": template_audit("mattergen"),
            }
        ],
        target_count=1,
        seed=123,
        max_sites=20,
        known_formulas={"NaCl"},
        allowed_sources={"generator"},
        mattergen_config=mattergen_config,
    )

    assert len(selected) == 1
    assert selected[0]["formula"] == "Na2Cl"
    assert [spec["id"] for spec in locked_specs] == ["mg_001"]
    assert any("MatterGen generated formula NaCl" in error and "known/training formula set" in error for error in errors)


def test_materialize_candidate_specs_reports_formula_probe_validation_reason() -> None:
    selected, _locked_specs, errors = materialize_candidate_specs(
        [],
        [
            {
                "id": "seq_001",
                "source": "generator",
                "count": 1,
                "formula_probe_strings": [
                    "template=perovskite;A=Rb:+1;B=Cd:+2;X=Br:-1;family=RbCdBr3_halide_perovskite"
                ],
                "template_consistency_audit": template_audit("perovskite"),
            }
        ],
        target_count=1,
        seed=20260525,
        max_sites=80,
        known_formulas=None,
        allowed_sources={"generator"},
        allow_structure_dicts=False,
    )

    assert selected == []
    assert any("volume_per_atom_too_large" in error for error in errors)
    assert any("required volume_per_atom=4.0..29.5" in error for error in errors)


def test_materialize_candidate_specs_hard_deduplicates_reduced_formula_across_generator_specs() -> None:
    selected, locked_specs, errors = materialize_candidate_specs(
        [],
        [
            {
                "id": "xy_s001_c001",
                "formula_probe_strings": ["template=rocksalt;A=Li:+1;X=F:-1;family=halide"],
                "cited_principle_ids": ["principle_program_001"],
                "template_consistency_audit": template_audit("rocksalt"),
            },
            {
                "id": "xy_s002_c001",
                "formula_probe_strings": ["template=rocksalt;A=Li:+1;X=F:-1;family=halide"],
                "cited_principle_ids": ["principle_program_001"],
                "template_consistency_audit": template_audit("rocksalt"),
            },
        ],
        target_count=2,
        seed=123,
        max_sites=80,
        known_formulas=None,
        allowed_sources={"generator"},
    )

    assert len(selected) == 1
    assert selected[0]["formula"] == "LiF"
    assert [spec["id"] for spec in locked_specs] == ["xy_s001_c001"]
    assert any("duplicate reduced_formula LiF" in error for error in errors)


def test_materialize_candidate_specs_can_disable_reduced_formula_dedup() -> None:
    selected, locked_specs, errors = materialize_candidate_specs(
        [],
        [
            {
                "id": "xy_s001_c001",
                "formula_probe_strings": ["template=rocksalt;A=Li:+1;X=F:-1;family=halide"],
                "cited_principle_ids": ["principle_program_001"],
                "template_consistency_audit": template_audit("rocksalt"),
            },
            {
                "id": "xy_s002_c001",
                "formula_probe_strings": ["template=rocksalt;A=Li:+1;X=F:-1;family=halide"],
                "cited_principle_ids": ["principle_program_001"],
                "template_consistency_audit": template_audit("rocksalt"),
            },
        ],
        target_count=2,
        seed=123,
        max_sites=80,
        known_formulas=None,
        allowed_sources={"generator"},
        max_per_reduced_formula=0,
    )

    assert len(selected) == 2
    assert len(locked_specs) == 2
    assert not any("duplicate reduced_formula" in error for error in errors)


def test_materialize_candidate_specs_deduplicates_query_matches() -> None:
    pool = [
        {
            "material_id": "mp-a",
            "formula": "RbCdBr3",
            "elements": ["Rb", "Cd", "Br"],
            "nelements": 3,
            "nsites": 5,
            "cif_path": "/tmp/a.cif",
        },
        {
            "material_id": "mp-b",
            "formula": "Rb2CdBr4",
            "elements": ["Rb", "Cd", "Br"],
            "nelements": 3,
            "nsites": 7,
            "cif_path": "/tmp/b.cif",
        },
    ]
    specs = [
        {
            "id": "xy_s001_c001",
            "source": "mp_pool",
            "query": {"elements_all": ["Rb", "Cd", "Br"], "preferred_order": "random"},
            "cited_principle_ids": ["principle_program_015"],
        },
        {
            "id": "xy_s001_c002",
            "source": "mp_pool",
            "query": {"elements_all": ["Rb", "Cd", "Br"], "preferred_order": "random"},
            "cited_principle_ids": ["principle_program_015"],
        },
    ]

    selected, locked_specs, errors = materialize_candidate_specs(
        pool,
        specs,
        target_count=2,
        seed=123,
        max_sites=80,
        known_formulas=None,
    )

    assert len(selected) == 2
    assert len(locked_specs) == 2
    assert {record["material_id"] for record in selected} == {"mp-a", "mp-b"}
    assert not errors


def test_materialize_candidate_specs_honors_excluded_formulas() -> None:
    pool = [
        {
            "material_id": "mp-a",
            "formula": "RbCdBr3",
            "elements": ["Rb", "Cd", "Br"],
            "nelements": 3,
            "nsites": 5,
            "cif_path": "/tmp/a.cif",
        },
        {
            "material_id": "mp-b",
            "formula": "Rb2CdBr4",
            "elements": ["Rb", "Cd", "Br"],
            "nelements": 3,
            "nsites": 7,
            "cif_path": "/tmp/b.cif",
        },
    ]

    selected, locked_specs, errors = materialize_candidate_specs(
        pool,
        [
            {
                "id": "xy_s001_c001",
                "source": "mp_pool",
                "query": {
                    "elements_all": ["Rb", "Cd", "Br"],
                    "preferred_order": "material_id",
                    "exclude_formula_probes": ["RbCdBr3"],
                },
                "cited_principle_ids": ["principle_program_015"],
            }
        ],
        target_count=1,
        seed=123,
        max_sites=80,
        known_formulas=None,
    )

    assert len(selected) == 1
    assert selected[0]["material_id"] == "mp-b"
    assert locked_specs[0]["exclude_formulas"] == ["RbCdBr3"]
    assert not errors


def test_materialization_dry_run_feedback_freezes_exact_material_ids() -> None:
    pool = [
        {
            "material_id": "mp-a",
            "formula": "RbCdBr3",
            "elements": ["Rb", "Cd", "Br"],
            "nelements": 3,
            "nsites": 5,
            "cif_path": "/tmp/a.cif",
        },
        {
            "material_id": "mp-b",
            "formula": "Rb2CdBr4",
            "elements": ["Rb", "Cd", "Br"],
            "nelements": 3,
            "nsites": 7,
            "cif_path": "/tmp/b.cif",
        },
    ]
    specs = [
        {
            "id": "xy_s001_c001",
            "source": "mp_pool",
            "query": {"elements_all": ["Rb", "Cd", "Br"], "preferred_order": "material_id"},
            "cited_principle_ids": ["principle_program_015"],
        }
    ]

    selected, frozen, feedback = materialization_dry_run_feedback(
        pool_records=pool,
        specs=specs,
        target_count=1,
        seed=123,
        max_sites=80,
        known_formulas=None,
    )

    assert feedback["status"] == "passed"
    assert selected[0]["material_id"] == "mp-a"
    assert frozen[0]["material_ids"] == ["mp-a"]
    assert frozen[0]["query"]["preferred_order"] == "material_id"


def test_locked_candidate_specs_from_feedback_preserves_partial_successes() -> None:
    feedback = {
        "status": "failed",
        "locked_candidate_specs_to_preserve": [
            {"id": "xy_s001_c001", "source": "mp_pool", "material_ids": ["mp-a"]},
            "not-a-spec",
            {"id": "xy_s001_c002", "source": "generator", "structure_dicts": [{"sites": []}]},
        ],
    }

    preserved = locked_candidate_specs_from_feedback(feedback)

    assert [item["id"] for item in preserved] == ["xy_s001_c001", "xy_s001_c002"]
    assert preserved[0]["material_ids"] == ["mp-a"]


def test_freeze_materialized_candidate_specs_preserves_generator_structure() -> None:
    structure = {
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "charge": 0,
        "lattice": {
            "matrix": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            "pbc": [True, True, True],
            "a": 3,
            "b": 3,
            "c": 3,
            "alpha": 90,
            "beta": 90,
            "gamma": 90,
            "volume": 27,
        },
        "sites": [
            {
                "species": [{"element": "Na", "occu": 1}],
                "abc": [0, 0, 0],
                "xyz": [0, 0, 0],
                "label": "Na",
                "properties": {},
            }
        ],
    }
    frozen = freeze_materialized_candidate_specs(
        [
            {
                "material_id": "generated::xy::Na",
                "formula": "Na",
                "structure_dict": structure,
                "crystal_llm_source": "generator",
                "xy_candidate_spec": {"id": "xy_s001_c001", "source": "generator"},
            }
        ]
    )

    assert frozen[0]["source"] == "generator"
    assert frozen[0]["material_ids"] == []
    assert frozen[0]["structure_dicts"] == [structure]


def test_compact_dialogue_breaks_final_payload_cycle() -> None:
    consensus = {
        "status": "consensus",
        "agent": "Y",
        "shard_id": 1,
        "agreed_candidate_specs": [{"id": "xy_s001_c001"}],
    }
    artifacts = [{"role": "Y", "cycle": 3, "mode": "final", "payload": consensus}]
    consensus["dialogue"] = artifacts

    compact = compact_dialogue(artifacts)

    assert compact[0]["payload_summary"]["candidate_count"] == 1
    assert "dialogue" not in compact[0]["payload_summary"]
