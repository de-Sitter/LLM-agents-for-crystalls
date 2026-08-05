from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pymatgen.core import Lattice, Structure

import crystal_llm.run_material_physics_mvp as mvp
from crystal_llm.llm_client import LLMError
from crystal_llm.local_agent_runtime import LocalAgentRuntime, _limit_tool_output, _requires_mandatory_candidate_pool
from crystal_llm.material_physics_schema import select_matches, validate_execution_payload, validate_mechanism_payload, validate_query
from crystal_llm.run_material_physics_mvp import (
    _client,
    agreement_reached,
    build_context_payload,
    bundle_results_from_analysis,
    call_json,
    context_with_debate_history,
    critique_requires_counterproposal,
    ensure_active_principle_program,
    ensure_round_completed,
    finalize_controller_state,
    is_prediction_design_infeasible,
    is_recoverable_llm_error_message,
    materialize_plan,
    mattergen_config_from_args,
    mattergen_prompt_defaults,
    mechanism_consensus_from_accepted_proposal,
    next_round_to_run,
    normalize_role_payload,
    parse_args,
    prompt_mechanism_counterproposal,
    prompt_mechanism_critique,
    prompt_mechanism_final,
    prediction_feedback_from_execution,
    prompt_mechanism_proposal,
    prompt_mechanism_reverse_critique,
    prompt_mechanism_reverse_final,
    prompt_mechanism_revision,
    prompt_prediction_critique,
    prompt_prediction_counterproposal,
    prompt_prediction_final,
    prompt_prediction_proposal,
    prompt_prediction_reverse_critique,
    prompt_prediction_reverse_final,
    prompt_prediction_revision,
    prompt_execution_final,
    prompt_execution_proposal,
    prompt_execution_critique,
    prompt_execution_counterproposal,
    prompt_execution_repair_proposal,
    prompt_execution_reverse_critique,
    prompt_execution_reverse_final,
    prompt_execution_revision,
    reconcile_execution_consensus,
    retry_prompt,
    run_analysis,
    RecoverableLLMFailure,
    summarize_materialization_feedback,
    summarize_round,
    update_principle_program_after_postmortem,
    valid_shape,
)


def test_normalizes_mechanism_agent_a_schema_variant() -> None:
    payload = normalize_role_payload(
        {
            "role": "mechanism_agent_a",
            "stage": "mechanism",
            "mechanism_hypotheses": [
                {
                    "name": "charge_balance",
                    "hypothesis": "Charge balanced compounds should usually have lower e_hull.",
                    "rationale": "Simple oxidation-state assignments reduce electrostatic frustration.",
                    "physical_basis": "ionic bonding",
                    "observable_features": ["integer oxidation states"],
                    "expected_trend": "lower e_hull",
                }
            ],
        },
        "mechanism_agent_a",
    )

    assert valid_shape(payload, "mechanism_agent_a")
    assert payload["agent"] == "A"
    assert payload["mechanisms"][0]["claim"].startswith("Charge balanced")
    assert payload["mechanisms"][0]["evidence_chain"]


def test_mechanism_agent_a_accepts_premature_consensus_shape() -> None:
    payload = normalize_role_payload(
        {
            "status": "consensus",
            "accepted_mechanisms": [
                {
                    "id": "m001",
                    "claim": "Simple charge balance improves stability.",
                    "rationale_summary": "Common oxidation states reduce redox frustration.",
                    "evidence_chain": ["closed shell anions"],
                    "scope": "ionic compounds",
                    "confidence": "medium",
                }
            ],
        },
        "mechanism_agent_a",
    )

    assert valid_shape(payload, "mechanism_agent_a")
    assert payload["agent"] == "A"
    assert payload["agree"] is True
    assert payload["mechanisms"][0]["id"] == "m001"


def test_mechanism_agent_a_normalizes_verbose_agent_label() -> None:
    payload = normalize_role_payload(
        {
            "status": "proposal",
            "agent": "Agent A",
            "mechanisms": [
                {
                    "id": "m001",
                    "claim": "Oxyfluorides can lower e_hull versus matched oxides.",
                    "rationale_summary": "Mixed hard anions improve charge balance and local coordination.",
                    "evidence_chain": ["Prior rounds supported matched O/F tests."],
                    "scope": "matched hard-anion ionic chemistries",
                    "confidence": "medium",
                }
            ],
            "agree": None,
            "concede": None,
        },
        "mechanism_agent_a",
    )

    assert valid_shape(payload, "mechanism_agent_a")
    assert payload["agent"] == "A"
    assert payload["agree"] is False
    assert payload["concede"] is False


def test_mechanism_agent_a_normalizes_role_name_success_status() -> None:
    payload = normalize_role_payload(
        {
            "status": "success",
            "agent": "mechanism_agent_a",
            "mechanisms": [
                {
                    "id": "m001",
                    "claim": "Mixed anions can reduce bond-valence mismatch.",
                    "rationale_summary": "Different anions satisfy distinct cation coordination preferences.",
                    "evidence_chain": ["RAG showed repeated support for mixed O/F tests."],
                    "scope": "ionic mixed-anion structures",
                    "confidence": "medium",
                }
            ],
            "agree": False,
            "concede": False,
        },
        "mechanism_agent_a",
    )

    assert valid_shape(payload, "mechanism_agent_a")
    assert payload["agent"] == "A"
    assert payload["status"] == "proposal"


def test_normalizes_wrapped_stage_payloads() -> None:
    mechanism = normalize_role_payload(
        {
            "material_physics_mechanism_consensus": {
                "status": "consensus",
                "accepted_mechanisms": [
                    {
                        "id": "m001",
                        "claim": "Simple charge balance improves stability.",
                        "rationale_summary": "Common oxidation states reduce redox frustration.",
                        "evidence_chain": ["closed shell anions"],
                        "scope": "ionic compounds",
                        "confidence": "medium",
                    }
                ],
            },
            "material_physics_prediction_consensus": {
                "status": "consensus",
                "accepted_predictions": [
                    {
                        "id": "p001",
                        "claim": "Charge-balanced primaries should outperform charge-frustrated controls.",
                    }
                ],
            },
            "material_physics_execution_plan": {
                "status": "consensus",
                "accepted_bundles": [{"id": "b001"}],
            }
        },
        "mechanism_agent_a",
    )
    prediction = normalize_role_payload(
        {
            "material_physics_mechanism_consensus": {
                "status": "consensus",
                "accepted_mechanisms": [{"id": "m001", "claim": "x", "rationale_summary": "y", "evidence_chain": [], "scope": "z", "confidence": "medium"}],
            },
            "material_physics_prediction_consensus": {
                "status": "consensus",
                "accepted_predictions": [
                    {
                        "id": "p001",
                        "claim": "Charge-balanced primaries should outperform charge-frustrated controls.",
                    }
                ],
            },
            "material_physics_execution_plan": {
                "status": "consensus",
                "accepted_bundles": [{"id": "b001"}],
            },
        },
        "prediction_agent_c",
    )

    assert valid_shape(mechanism, "mechanism_agent_a")
    assert mechanism["mechanisms"][0]["id"] == "m001"
    assert valid_shape(prediction, "prediction_agent_c")
    assert prediction["predictions"][0]["id"] == "p001"


def test_mechanism_consensus_fills_missing_rationale_summary() -> None:
    payload = normalize_role_payload(
        {
            "status": "consensus",
            "accepted_mechanisms": [
                {
                    "id": "m001",
                    "claim": "Common-valence ionic salts are often stable.",
                    "evidence_chain": ["charge balance"],
                    "scope": "ionic salts",
                    "confidence": "medium",
                }
            ],
            "rejected_mechanisms": [{"id": "m002", "claim": "bad"}],
        },
        "mechanism_consensus",
    )

    assert valid_shape(payload, "mechanism_consensus")
    assert payload["accepted_mechanisms"][0]["rationale_summary"] == payload["accepted_mechanisms"][0]["claim"]
    assert validate_mechanism_payload(payload) == []


def test_mechanism_agent_b_accepts_premature_consensus_shape() -> None:
    payload = normalize_role_payload(
        {
            "status": "consensus",
            "accepted_mechanisms": [
                {
                    "id": "m001",
                    "claim": "Simple charge balance improves stability.",
                    "rationale_summary": "Common oxidation states reduce redox frustration.",
                    "evidence_chain": ["closed shell anions"],
                    "scope": "ionic compounds",
                    "confidence": "medium",
                }
            ],
            "rejected_mechanisms": [
                {
                    "id": "m002",
                    "claim": "Density alone determines stability.",
                    "rejection_reason": "Too proxy-like.",
                }
            ],
        },
        "mechanism_agent_b",
    )

    assert valid_shape(payload, "mechanism_agent_b")
    assert payload["agent"] == "B"
    assert payload["agree"] is True
    assert payload["required_revisions"]


def test_mechanism_agent_b_accepts_no_consensus_shape() -> None:
    payload = normalize_role_payload(
        {
            "status": "no_consensus",
            "accepted_mechanisms": [],
            "rejected_mechanisms": [
                {
                    "id": "m001",
                    "claim": "A broad mechanism.",
                    "rejection_reason": "Too broad.",
                }
            ],
            "consensus_summary": "No mechanism survives.",
        },
        "mechanism_agent_b",
    )

    assert valid_shape(payload, "mechanism_agent_b")
    assert payload["agent"] == "B"
    assert payload["required_revisions"]
    assert payload["status"] == "no_consensus"


def test_normalizes_prediction_and_execution_proposer_variants() -> None:
    prediction = normalize_role_payload(
        {
            "role": "prediction_agent_c",
            "testable_predictions": [{"id": "p1", "claim": "Primary should be lower e_hull."}],
        },
        "prediction_agent_c",
    )
    execution = normalize_role_payload(
        {
            "role": "execution_agent_e",
            "test_bundles": [{"id": "b1", "primary": {"count": 5}, "control": {"count": 5}}],
        },
        "execution_agent_e",
    )

    assert valid_shape(prediction, "prediction_agent_c")
    assert prediction["agent"] == "C"
    assert prediction["predictions"][0]["id"] == "p1"
    assert valid_shape(execution, "execution_agent_e")
    assert execution["agent"] == "E"
    assert execution["bundles"][0]["id"] == "b1"


def test_execution_agent_e_consensus_variant_marks_agreement() -> None:
    execution = normalize_role_payload(
        {
            "status": "consensus",
            "accepted_bundles": [{"id": "b1", "primary": {"count": 5}, "control": {"count": 5}}],
        },
        "execution_agent_e",
    )

    assert valid_shape(execution, "execution_agent_e")
    assert execution["agent"] == "E"
    assert execution["agree"] is True


def test_normalizes_critic_status_payload_without_agent_letter() -> None:
    critique = normalize_role_payload(
        {
            "status": "critique",
            "agree": False,
            "accepted_mechanisms": [{"id": "m1", "critique": "Valid but too broad."}],
            "rejected_mechanisms": [{"id": "m2", "rejection_reason": "Unsupported."}],
        },
        "mechanism_agent_b",
    )

    assert valid_shape(critique, "mechanism_agent_b")
    assert critique["agent"] == "B"
    assert critique["required_revisions"]


def test_normalizes_auditor_needs_repair_payload() -> None:
    audit = normalize_role_payload(
        {
            "status": "needs_repair",
            "agent": "F",
            "agree_with_agent_E": False,
            "audit_summary": "Unsupported query fields are present.",
            "bundle_audits": [
                {
                    "id": "b1",
                    "decision": "reject_until_repaired",
                    "reason": "elements_any_second_set and formula_constraint are not supported.",
                    "required_repairs": ["Use only supported query keys."],
                }
            ],
        },
        "execution_agent_f",
    )

    assert valid_shape(audit, "execution_agent_f")
    assert audit["agree"] is False
    assert audit["required_revisions"]


def test_logical_counterproposal_roles_validate_parser_compatible_shapes() -> None:
    mechanism = normalize_role_payload(
        {
            "status": "proposal",
            "agent": "A",
            "mechanisms": [{"id": "m001", "claim": "Dense charge-balanced halides should reduce e_hull."}],
            "agree": False,
            "concede": False,
        },
        "mechanism_agent_b_counterproposal",
    )
    prediction = normalize_role_payload(
        {
            "status": "proposal",
            "agent": "C",
            "predictions": [{"id": "p001", "claim": "The primary branch should have lower e_hull."}],
            "agree": False,
            "concede": False,
        },
        "prediction_agent_d_counterproposal",
    )
    execution = normalize_role_payload(
        {
            "status": "proposal",
            "agent": "E",
            "bundles": [{"id": "b001", "prediction_ids": ["p001"]}],
            "agree": False,
            "concede": False,
        },
        "execution_agent_f_counterproposal",
    )

    assert valid_shape(mechanism, "mechanism_agent_b_counterproposal")
    assert valid_shape(prediction, "prediction_agent_d_counterproposal")
    assert valid_shape(execution, "execution_agent_f_counterproposal")
    prompt = retry_prompt(
        "prediction_agent_d_counterproposal",
        "Agent D: produce a counterproposal.",
        "{}",
        "output did not match required shape",
    )
    assert "role `prediction_agent_d_counterproposal`" in prompt
    assert "predictions" in prompt


def test_prediction_agent_d_accepts_audit_complete_variant() -> None:
    audit = normalize_role_payload(
        {
            "status": "audit_complete",
            "verdict": "reject_partial",
            "accepted_predictions": [],
            "rejected_predictions": [
                {
                    "id": "p001",
                    "reason": "Primary/control split is under-specified and not executable.",
                }
            ],
            "summary": "Prediction needs tighter matching before it can be tested.",
        },
        "prediction_agent_d",
    )

    assert valid_shape(audit, "prediction_agent_d")
    assert audit["agent"] == "D"
    assert audit["agree"] is False
    assert audit["required_revisions"] == [
        "p001: Primary/control split is under-specified and not executable."
    ]


def test_prediction_agent_d_accepts_premature_consensus_shape() -> None:
    payload = normalize_role_payload(
        {
            "status": "consensus",
            "accepted_predictions": [
                {
                    "id": "p001",
                    "claim": "Charge-balanced primaries should outperform charge-frustrated controls.",
                    "rejection_reason": "placeholder",
                }
            ],
            "rejected_predictions": [],
            "consensus_summary": "Audit accepted the prediction with minor caveats.",
        },
        "prediction_agent_d",
    )

    assert valid_shape(payload, "prediction_agent_d")
    assert payload["agent"] == "D"
    assert payload["required_revisions"]


def test_prediction_agent_d_accepts_partial_rejection_shape() -> None:
    payload = normalize_role_payload(
        {
            "status": "partial_rejection",
            "overall_summary": "p001 is usable, p002 is too confounded.",
            "accepted_predictions": [
                {
                    "id": "p001",
                    "verdict": "accept_with_caveat",
                    "audit_summary": "Weak but executable.",
                }
            ],
            "rejected_predictions": [
                {
                    "id": "p002",
                    "verdict": "reject",
                    "rejection_reason": "Confounded comparison.",
                }
            ],
        },
        "prediction_agent_d",
    )

    assert valid_shape(payload, "prediction_agent_d")
    assert payload["agent"] == "D"
    assert payload["required_revisions"]


def test_prediction_agent_d_accepts_verdict_issues_shape() -> None:
    payload = normalize_role_payload(
        {
            "agree": False,
            "overall_verdict": "reject",
            "summary": "C's plan is executable but overstates the accepted mechanism.",
            "issues": [
                {
                    "type": "mechanistic_overclaim",
                    "detail": "The accepted mechanism is conditional, but the prediction treats higher e_hull as a default outcome.",
                },
                {
                    "type": "budget_violation",
                    "detail": "Two 3+3 comparisons require 12 materials, exceeding the target count of 10.",
                },
            ],
            "decision": "reject",
        },
        "prediction_agent_d",
    )

    assert valid_shape(payload, "prediction_agent_d")
    assert payload["agent"] == "D"
    assert payload["agree"] is False
    assert any("conditional" in item for item in payload["required_revisions"])
    assert any("12 materials" in item for item in payload["required_revisions"])


def test_prediction_agent_c_list_concede_is_not_dialogue_concession() -> None:
    proposal = normalize_role_payload(
        {
            "status": "consensus",
            "agent": "prediction_agent_c",
            "predictions": [{"id": "p001", "claim": "Primary should be lower e_hull."}],
            "agree": ["m001"],
            "concede": [
                "This mechanism is narrow and should not be generalized to all mixed-anion materials."
            ],
        },
        "prediction_agent_c",
    )
    critique = normalize_role_payload(
        {
            "status": "rejected",
            "agent": "D",
            "issues": [{"detail": "The claim is stronger than the mechanism."}],
        },
        "prediction_agent_d",
    )

    assert valid_shape(proposal, "prediction_agent_c")
    assert proposal["agree"] is False
    assert proposal["concede"] is False
    assert valid_shape(critique, "prediction_agent_d")
    assert agreement_reached(proposal, critique) is False


def test_agreement_reached_ignores_proposal_side_concede_when_reviewer_rejects() -> None:
    proposal = {
        "status": "consensus",
        "agent": "C",
        "agree": False,
        "concede": True,
        "predictions": [{"id": "p001", "claim": "Counterproposal"}],
    }
    critique = {
        "status": "rejected",
        "agent": "D",
        "agree": False,
        "concede": False,
        "required_revisions": ["Counterproposal is underfilled."],
    }

    assert agreement_reached(proposal, critique) is False


def test_agreement_reached_accepts_reviewer_explicit_acceptance() -> None:
    proposal = {
        "status": "consensus",
        "agent": "C",
        "agree": False,
        "concede": True,
        "predictions": [{"id": "p001", "claim": "Counterproposal"}],
    }
    critique = {
        "status": "consensus",
        "agent": "D",
        "agree": True,
        "concede": False,
        "required_revisions": [],
    }

    assert agreement_reached(proposal, critique) is True


def test_mechanism_consensus_reconstructed_from_accepted_counterproposal() -> None:
    proposal = {
        "status": "proposal",
        "agent": "A",
        "mechanisms": [
            {
                "id": "b001",
                "claim": "Compact Li/Na transition-metal oxophosphates are stabilized by phosphate buffering.",
                "rationale_summary": "P-O covalency and Li/Na charge compensation reduce bond-valence frustration.",
                "evidence_chain": ["A/B used empty-history RAG and selected the underexplored O+P cluster."],
                "scope": "Compact Li/Na transition-metal P-O frameworks.",
                "confidence": "low",
            }
        ],
    }
    critique = {"status": "review", "agent": "B", "agree": True, "concede": True, "required_revisions": []}

    assert agreement_reached(proposal, critique) is True
    consensus = mechanism_consensus_from_accepted_proposal(
        proposal,
        round_number=1,
        dialogue=[{"role": "B", "mode": "counterproposal", "payload": proposal}],
        consensus_summary="accepted counterproposal",
    )

    assert consensus["status"] == "consensus"
    assert consensus["accepted_mechanisms"][0]["id"] == "b001"
    assert validate_mechanism_payload(consensus) == []


def test_execution_agent_f_accepts_verdict_issues_shape() -> None:
    payload = normalize_role_payload(
        {
            "overall_verdict": "reject",
            "summary": "The executable bundle uses unsupported selectors.",
            "issues": [
                {
                    "type": "unsupported_query_key",
                    "detail": "Remove invented query keys before materialization.",
                }
            ],
            "decision": "reject",
        },
        "execution_agent_f",
    )

    assert valid_shape(payload, "execution_agent_f")
    assert payload["agent"] == "F"
    assert payload["agree"] is False
    assert payload["required_revisions"] == [
        "unsupported_query_key: Remove invented query keys before materialization."
    ]


def test_execution_agent_e_can_report_materialization_conflict() -> None:
    conflict = normalize_role_payload(
        {
            "role": "execution_agent_e",
            "status": "materialization_conflict",
            "conflicting_bundle_ids": ["b1"],
            "reason": "Accepted predictions require mutually exclusive element filters.",
            "minimal_fix_needed": "Relax one prediction.",
        },
        "execution_agent_e",
    )

    assert valid_shape(conflict, "execution_agent_e")
    assert conflict["agent"] == "E"
    assert conflict["status"] == "materialization_conflict"


def test_execution_agent_e_can_report_hypothesis_conflict() -> None:
    conflict = normalize_role_payload(
        {
            "role": "execution_agent_e",
            "status": "hypothesis_conflict",
            "feasible": False,
            "prediction_ids": ["p001", "p002"],
            "reason": "The matched-pair design requires 12 materials but the cap is 10.",
            "minimal_fix_needed": ["Relax one accepted prediction or raise the cap."],
            "intended_bundles": [],
        },
        "execution_agent_e",
    )

    assert valid_shape(conflict, "execution_agent_e")
    assert conflict["agent"] == "E"
    assert conflict["status"] == "hypothesis_conflict"


def test_execution_agent_f_accepts_premature_consensus_shape() -> None:
    payload = normalize_role_payload(
        {
            "status": "consensus",
            "accepted_bundles": [
                {
                    "id": "b001",
                    "prediction_ids": ["p001"],
                    "primary": {"count": 5, "query": {"elements_all": ["La", "O"]}},
                    "control": {"count": 5, "query": {"elements_all": ["La", "O"]}},
                }
            ],
            "rejected_bundles": [],
            "consensus_summary": "Audit accepted the execution plan.",
        },
        "execution_agent_f",
    )

    assert valid_shape(payload, "execution_agent_f")
    assert payload["agent"] == "F"
    assert payload["required_revisions"]


def test_normalizes_consensus_field_variants() -> None:
    mechanism = normalize_role_payload(
        {"status": "consensus", "mechanisms": [{"id": "m1", "claim": "x"}]},
        "mechanism_consensus",
    )
    prediction = normalize_role_payload(
        {"status": "consensus", "predictions": [{"id": "p1", "claim": "y"}]},
        "prediction_consensus",
    )
    execution = normalize_role_payload(
        {"status": "consensus", "bundles": [{"id": "b1"}]},
        "execution_consensus",
    )

    assert valid_shape(mechanism, "mechanism_consensus")
    assert mechanism["accepted_mechanisms"][0]["id"] == "m1"
    assert valid_shape(prediction, "prediction_consensus")
    assert prediction["accepted_predictions"][0]["id"] == "p1"
    assert valid_shape(execution, "execution_consensus")
    assert execution["accepted_bundles"][0]["id"] == "b1"


def test_prediction_consensus_fills_required_defaults() -> None:
    payload = normalize_role_payload(
        {
            "status": "consensus",
            "accepted_predictions": [
                {
                    "id": "p001",
                    "mechanism_ids": ["m001"],
                    "claim": "Large A-site ABO3 perovskites should have lower e_hull than small A-site controls.",
                    "comparison_design": {"primary": "large A-site", "control": "small A-site"},
                }
            ],
        },
        "prediction_consensus",
    )

    assert valid_shape(payload, "prediction_consensus")
    prediction = payload["accepted_predictions"][0]
    assert prediction["predicted_relation"] == "primary_lower_e_hull_than_control"
    assert prediction["falsification_criteria"]


def test_prediction_consensus_normalizes_ok_status_with_predictions() -> None:
    payload = normalize_role_payload(
        {
            "status": "ok",
            "accepted_predictions": [{"id": "p001", "claim": "Li branch should beat Na control."}],
            "consensus_summary": "Both agents accepted the prediction.",
        },
        "prediction_consensus",
    )

    assert payload["status"] == "consensus"
    assert valid_shape(payload, "prediction_consensus")


def test_prediction_consensus_accepts_explicit_no_consensus() -> None:
    payload = normalize_role_payload(
        {
            "status": "rejected",
            "accepted_predictions": [],
            "consensus_summary": "No prediction achieved joint acceptance.",
        },
        "prediction_consensus",
    )

    assert valid_shape(payload, "prediction_consensus")
    assert payload["status"] == "rejected"
    assert payload["accepted_predictions"] == []


def test_execution_consensus_accepts_no_materialized_consensus_for_repair_loop() -> None:
    payload = normalize_role_payload(
        {
            "status": "no_materialized_consensus",
            "accepted_bundles": [],
            "rejected_bundles": [{"id": "b1", "reason": "MP pool has no matching control group."}],
            "consensus_summary": "No MP-pool-only plan is executable; generator fallback is needed.",
        },
        "execution_consensus",
    )

    assert valid_shape(payload, "execution_consensus")
    assert payload["status"] == "no_materialized_consensus"


def test_execution_consensus_accepts_compact_agent_f_acceptance() -> None:
    payload = normalize_role_payload(
        {
            "agent": "F",
            "status": "consensus",
            "overall_verdict": "accept",
            "accepted_bundle_ids": ["b001", "b002"],
            "bundle_reviews": [
                {"bundle_id": "b001", "verdict": "accept"},
                {"bundle_id": "b002", "verdict": "accept"},
            ],
            "required_revisions": [],
        },
        "execution_consensus",
    )

    assert valid_shape(payload, "execution_consensus")
    assert payload["accepted_bundles"] == []


def test_execution_consensus_normalizes_accepted_status_with_bundle_ids() -> None:
    payload = normalize_role_payload(
        {
            "status": "accepted",
            "accepted_bundle_ids": [
                "E_repair_p001_LiCoPO4_vs_NaCoPO4_exact_mattergen",
                "E_repair_p002_LiFePO4_vs_NaFePO4_exact_mattergen",
            ],
            "rejected_bundles": [],
            "consensus_summary": "Approve Agent E's repaired MatterGen-native execution plan.",
        },
        "execution_consensus",
    )

    assert payload["status"] == "consensus"
    assert payload["accepted_bundles"] == []
    assert valid_shape(payload, "execution_consensus")


def test_execution_agent_e_normalizes_bundle_aliases() -> None:
    payload = normalize_role_payload(
        {
            "agent": "E",
            "status": "consensus",
            "accepted_bundles": [
                {
                    "bundle_id": "E_p001_mattergen_LiMnPO_vs_LiFePO",
                    "prediction_id": "p001",
                    "expected_relation": "primary_lower_e_hull_than_control",
                    "branches": {
                        "primary": {
                            "source": "mattergen",
                            "count": 1,
                            "mattergen_requests": [
                                {
                                    "backend": "mattergen",
                                    "properties_to_condition_on": {"chemical_system": "Li-Mn-P-O", "energy_above_hull": 0.0},
                                }
                            ],
                        },
                        "control": {
                            "source": "mattergen",
                            "count": 1,
                            "mattergen_requests": [
                                {
                                    "backend": "mattergen",
                                    "properties_to_condition_on": {"chemical_system": "Li-Fe-P-O", "energy_above_hull": 0.0},
                                }
                            ],
                        },
                    },
                }
            ],
        },
        "execution_agent_e",
    )

    bundle = payload["bundles"][0]
    assert bundle["id"] == "E_p001_mattergen_LiMnPO_vs_LiFePO"
    assert bundle["prediction_ids"] == ["p001"]
    assert bundle["primary"]["mattergen_requests"][0]["properties_to_condition_on"]["chemical_system"] == "Li-Mn-P-O"
    assert bundle["control"]["mattergen_requests"][0]["properties_to_condition_on"]["chemical_system"] == "Li-Fe-P-O"
    assert valid_shape(payload, "execution_agent_e")


def test_execution_agent_e_normalizes_branch_aliases() -> None:
    payload = normalize_role_payload(
        {
            "agent": "E",
            "status": "materialization_blueprints_ready",
            "bundles": [
                {
                    "id": "F_repair_p001_KFeAsO4_vs_KFePO4_mattergen_exact_formula",
                    "prediction_ids": ["p001"],
                    "expected_relation": "primary_lower_e_hull_than_control",
                    "primary_branch": {
                        "source": "mattergen",
                        "count": 2,
                        "mattergen_requests": [
                            {
                                "backend": "mattergen",
                                "properties_to_condition_on": {
                                    "chemical_system": "K-Fe-As-O",
                                    "energy_above_hull": 0.0,
                                },
                                "filters": {"chemical_system": ["K", "Fe", "As", "O"]},
                            }
                        ],
                    },
                    "control_branch": {
                        "source": "mattergen",
                        "count": 2,
                        "mattergen_requests": [
                            {
                                "backend": "mattergen",
                                "properties_to_condition_on": {
                                    "chemical_system": "K-Fe-P-O",
                                    "energy_above_hull": 0.0,
                                },
                                "filters": {"chemical_system": ["K", "Fe", "P", "O"]},
                            }
                        ],
                    },
                }
            ],
        },
        "execution_agent_e",
    )

    bundle = payload["bundles"][0]
    assert bundle["primary"]["count"] == 2
    assert bundle["control"]["count"] == 2
    assert bundle["primary"]["mattergen_requests"][0]["properties_to_condition_on"]["chemical_system"] == "K-Fe-As-O"
    assert bundle["control"]["mattergen_requests"][0]["properties_to_condition_on"]["chemical_system"] == "K-Fe-P-O"
    assert valid_shape(payload, "execution_agent_e")


def test_reconcile_execution_consensus_rebuilds_accepted_bundles_from_ids() -> None:
    proposal = {
        "bundles": [
            {
                "id": "b001",
                "prediction_ids": ["p001"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "formula_probes": [
                        {
                            "id": "p1",
                            "template": "perovskite",
                            "roles": {
                                "A": {"element": "La", "oxidation_state": 3},
                                "B": {"element": "Al", "oxidation_state": 3},
                                "X": {"element": "O", "oxidation_state": -2},
                            },
                        }
                    ],
                },
                "control": {
                    "formula_probes": [
                        {
                            "id": "c1",
                            "template": "perovskite",
                            "roles": {
                                "A": {"element": "La", "oxidation_state": 3},
                                "B": {"element": "Co", "oxidation_state": 3},
                                "X": {"element": "O", "oxidation_state": -2},
                            },
                        }
                    ],
                },
                "rationale_summary": "ok",
                "selection_notes": "ok",
            },
            {
                "id": "b002",
                "prediction_ids": ["p002"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "formula_probes": [
                        {
                            "id": "p2",
                            "template": "spinel",
                            "roles": {
                                "A": {"element": "Mg", "oxidation_state": 2},
                                "B": {"element": "Al", "oxidation_state": 3},
                                "X": {"element": "O", "oxidation_state": -2},
                            },
                        }
                    ],
                },
                "control": {
                    "formula_probes": [
                        {
                            "id": "c2",
                            "template": "spinel",
                            "roles": {
                                "A": {"element": "Ni", "oxidation_state": 2},
                                "B": {"element": "Fe", "oxidation_state": 3},
                                "X": {"element": "O", "oxidation_state": -2},
                            },
                        }
                    ],
                },
                "rationale_summary": "ok",
                "selection_notes": "ok",
            },
        ]
    }
    consensus = {
        "status": "consensus",
        "agent": "F",
        "agree": True,
        "concede": False,
        "accepted_bundle_ids": ["b002"],
        "rejected_bundle_ids": [],
        "required_revisions": [],
        "bundle_reviews": [],
        "consensus_summary": "ok",
        "risk_flags": [],
    }

    resolved = reconcile_execution_consensus(consensus, proposal)

    assert valid_shape(resolved, "execution_consensus")
    assert [bundle["id"] for bundle in resolved["accepted_bundles"]] == ["b002"]
    assert validate_execution_payload(resolved, target_count=2) == []


def test_reconcile_execution_consensus_matches_ordinal_id_to_bundle_alias() -> None:
    proposal = normalize_role_payload(
        {
            "agent": "E",
            "status": "consensus",
            "accepted_bundles": [
                {
                    "bundle_id": "E_p001_mattergen_LiMnPO_vs_LiFePO",
                    "prediction_id": "p001",
                    "expected_relation": "primary_lower_e_hull_than_control",
                    "branches": {
                        "primary": {
                            "source": "mattergen",
                            "count": 1,
                            "mattergen_requests": [
                                {
                                    "backend": "mattergen",
                                    "properties_to_condition_on": {"chemical_system": "Li-Mn-P-O", "energy_above_hull": 0.0},
                                    "filters": {"chemical_system": "Li-Mn-P-O"},
                                    "target_count": 1,
                                }
                            ],
                        },
                        "control": {
                            "source": "mattergen",
                            "count": 1,
                            "mattergen_requests": [
                                {
                                    "backend": "mattergen",
                                    "properties_to_condition_on": {"chemical_system": "Li-Fe-P-O", "energy_above_hull": 0.0},
                                    "filters": {"chemical_system": "Li-Fe-P-O"},
                                    "target_count": 1,
                                }
                            ],
                        },
                    },
                    "rationale_summary": "proposal rationale",
                    "selection_notes": "proposal notes",
                }
            ],
        },
        "execution_agent_e",
    )
    consensus = {
        "status": "consensus",
        "accepted_bundle_ids": ["b001"],
        "rejected_bundle_ids": [],
        "required_revisions": [],
        "bundle_reviews": [],
        "consensus_summary": "accept the first bundle",
    }

    resolved = reconcile_execution_consensus(consensus, proposal)

    bundle = resolved["accepted_bundles"][0]
    assert bundle["id"] == "E_p001_mattergen_LiMnPO_vs_LiFePO"
    assert bundle["prediction_ids"] == ["p001"]
    assert bundle["primary"]["mattergen_requests"][0]["properties_to_condition_on"]["chemical_system"] == "Li-Mn-P-O"
    assert bundle["control"]["mattergen_requests"][0]["properties_to_condition_on"]["chemical_system"] == "Li-Fe-P-O"
    assert validate_execution_payload(resolved, target_count=2) == []


def test_reconcile_execution_consensus_restores_compact_bundle_branches() -> None:
    proposal = {
        "bundles": [
            {
                "id": "b001",
                "prediction_ids": ["p001"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "source": "generator",
                    "count": 1,
                    "structure_dicts": [{"@module": "pymatgen.core.structure", "@class": "Structure"}],
                },
                "control": {
                    "source": "generator",
                    "count": 1,
                    "structure_dicts": [{"@module": "pymatgen.core.structure", "@class": "Structure"}],
                },
                "rationale_summary": "proposal rationale",
                "selection_notes": "proposal notes",
            }
        ]
    }
    consensus = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b001",
                "prediction_ids": ["p001"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {"source": "generator", "count": 1, "motif": "compact primary summary"},
                "control": {"source": "generator", "count": 1, "motif": "compact control summary"},
                "rationale_summary": "accepted rationale",
                "selection_notes": "accepted notes",
            }
        ],
        "consensus_summary": "ok",
    }

    resolved = reconcile_execution_consensus(consensus, proposal)

    bundle = resolved["accepted_bundles"][0]
    assert bundle["rationale_summary"] == "accepted rationale"
    assert bundle["primary"]["structure_dicts"] == proposal["bundles"][0]["primary"]["structure_dicts"]
    assert bundle["control"]["structure_dicts"] == proposal["bundles"][0]["control"]["structure_dicts"]
    assert validate_execution_payload(resolved, target_count=2) == []


def test_reconcile_execution_consensus_restores_generator_branch_with_query_only() -> None:
    structure = Structure(Lattice.cubic(4.0), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    proposal = {
        "bundles": [
            {
                "id": "b001",
                "prediction_ids": ["p001"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "source": "generator",
                    "count": 1,
                    "query": {"elements_all": ["Li", "F"]},
                    "structure_dicts": [structure.as_dict()],
                },
                "control": {"source": "mp_pool", "count": 1, "query": {"elements_all": ["Na", "F"]}},
                "rationale_summary": "proposal rationale",
                "selection_notes": "proposal notes",
            }
        ]
    }
    consensus = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b001",
                "prediction_ids": ["p001"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {"source": "generator", "count": 1, "query": {"elements_all": ["Li", "F"]}},
                "control": {"source": "mp_pool", "count": 1, "query": {"elements_all": ["Na", "F"]}},
                "rationale_summary": "accepted rationale",
                "selection_notes": "accepted notes",
            }
        ],
        "consensus_summary": "ok",
    }

    resolved = reconcile_execution_consensus(consensus, proposal)

    primary = resolved["accepted_bundles"][0]["primary"]
    assert primary["structure_dicts"] == [structure.as_dict()]
    assert validate_execution_payload(resolved, target_count=2) == []


def test_ensure_round_completed_rejects_unresolved_round() -> None:
    with pytest.raises(RuntimeError, match="did not reach evaluator completion"):
        ensure_round_completed(
            {
                "round": 7,
                "status": "invalid_prediction",
                "artifact": {"validation_errors": ["accepted_predictions must be a non-empty list"]},
            }
        )


def test_ensure_round_completed_accepts_nonfatal_unresolved_stage() -> None:
    ensure_round_completed(
        {
            "round": 7,
            "status": "unresolved_prediction",
            "artifact": {
                "status": "rejected",
                "accepted_predictions": [],
                "consensus_summary": "No prediction achieved joint acceptance.",
            },
        }
    )


def test_ensure_round_completed_accepts_complete_round() -> None:
    ensure_round_completed({"round": 5, "status": "complete", "round_summary": {"evaluation_summary": {"support_rate": 1.0}}})


def test_finalize_controller_state_marks_normal_exit_completed() -> None:
    state = {"status": "running", "updated_at_utc": "old"}

    finalize_controller_state(state)

    assert state["status"] == "completed"
    assert state["updated_at_utc"] != "old"


def test_finalize_controller_state_preserves_error_state() -> None:
    state = {"status": "error", "updated_at_utc": "old"}

    finalize_controller_state(state)

    assert state["status"] == "error"
    assert state["updated_at_utc"] != "old"


def test_principle_postmortem_normalization_accepts_core_shape() -> None:
    payload = normalize_role_payload(
        {
            "status": "finalize",
            "program_id": "principle_program_001",
            "round": 4,
            "hypothesis_status": "supported",
            "principle_update_action": "finalize",
            "current_principle_statement": "Compact charge-balanced nitrides are stabilized.",
            "micro_mechanism": "Short M-N bonds reduce electrostatic mismatch.",
            "causal_interpretation": "Primary branch wins and no control-only SUN remains unexplained.",
            "failure_boundaries": ["Do not generalize to loose multinary nitrides."],
            "unresolved_contradictions": [],
            "next_test_focus": "Start a new principle program.",
            "experience_book_entry": {
                "principle_statement": "Compact charge-balanced nitrides are stabilized.",
                "reasoning_chain": ["Primary branch repeatedly beats matched controls."],
                "evidence_rounds": [1, 2, 3, 4],
                "boundaries": ["Loose multinary nitrides excluded."],
                "residual_risks": [],
            },
        },
        "principle_postmortem",
    )

    assert valid_shape(payload, "principle_postmortem")
    assert payload["status"] == "finalize"
    assert payload["principle_update_action"] == "finalize"


def test_principle_program_finalize_writes_book_entry() -> None:
    state = {"history": [], "principle_book": []}
    ensure_active_principle_program(state, 1)
    round_summary = {
        "round": 1,
        "evaluation_summary": {
            "support_rate": 1.0,
            "primary_sun_count": 2,
            "control_sun_count": 0,
            "mechanism_validated_sun_count": 2,
        },
        "bundle_results": [{"delta": -0.2}],
    }
    postmortem = {
        "status": "finalize",
        "program_id": "principle_program_001",
        "round": 1,
        "hypothesis_status": "supported",
        "principle_update_action": "finalize",
        "current_principle_statement": "Compact charge-balanced nitrides are stabilized.",
        "micro_mechanism": "Short M-N bonds reduce electrostatic mismatch.",
        "causal_interpretation": "Primary branch wins.",
        "failure_boundaries": ["Large loose cells excluded."],
        "unresolved_contradictions": [],
        "next_test_focus": "Start a new principle.",
        "experience_book_entry": {
            "principle_statement": "Compact charge-balanced nitrides are stabilized.",
            "reasoning_chain": ["Matched primary beats control."],
            "evidence_rounds": [1],
            "boundaries": ["Large loose cells excluded."],
            "residual_risks": [],
        },
    }

    update_principle_program_after_postmortem(state, round_summary=round_summary, postmortem=postmortem)

    assert state["current_principle_program"] is None
    assert state["principle_book"][0]["status"] == "validated_principle"
    assert state["principle_book"][0]["evidence_rounds"] == [1]


def test_principle_program_finalize_updates_explicit_existing_book_entry() -> None:
    state = {
        "history": [],
        "principle_book": [
            {
                "program_id": "principle_program_001",
                "status": "validated_principle",
                "closed_at_round": 1,
                "principle_statement": "Compact Mg-Fe oxyfluorides are destabilized relative to Mg-Fe oxides.",
                "micro_mechanism": "F disrupts Fe-O-Fe oxide-network cohesion.",
                "reasoning_chain": ["Round 1 supported the penalty."],
                "evidence_rounds": [1],
                "boundaries": ["Compact Mg-Fe-O-F versus Mg-Fe-O only."],
                "residual_risks": [],
            }
        ],
    }
    ensure_active_principle_program(state, 2)
    round_summary = {
        "round": 2,
        "evaluation_summary": {"support_rate": 1.0},
        "bundle_results": [{"delta": 0.04}],
    }
    postmortem = {
        "status": "finalize",
        "program_id": "principle_program_002",
        "round": 2,
        "hypothesis_status": "supported",
        "principle_update_action": "finalize",
        "current_principle_statement": "Compact Mg-Fe oxyfluorides retain an oxide-relative hull penalty.",
        "micro_mechanism": "F disrupts dense Fe-O-Fe network cohesion.",
        "causal_interpretation": "The same Mg-Fe-O-F causal chain was refined.",
        "failure_boundaries": ["Lower compactness may attenuate the penalty."],
        "unresolved_contradictions": [],
        "next_test_focus": "Start a distinct principle.",
        "experience_book_entry": {
            "principle_identity": "update_existing_principle",
            "updates_principle_id": "principle_program_001",
            "principle_statement": "Compact Mg-Fe oxyfluorides retain an oxide-relative hull penalty.",
            "reasoning_chain": ["Round 2 refined the same penalty boundary."],
            "evidence_rounds": [2],
            "boundaries": ["Lower compactness may attenuate the penalty."],
            "residual_risks": [],
        },
    }

    update_principle_program_after_postmortem(state, round_summary=round_summary, postmortem=postmortem)

    assert state["current_principle_program"] is None
    assert len(state["principle_book"]) == 1
    entry = state["principle_book"][0]
    assert entry["program_id"] == "principle_program_001"
    assert entry["evidence_rounds"] == [1, 2]
    assert entry["last_updated_round"] == 2
    assert "principle_program_002" in entry["merged_program_ids"]
    assert any("attenuate" in item for item in entry["boundaries"])


def test_principle_program_finalize_heuristically_merges_same_causal_theme() -> None:
    state = {
        "history": [],
        "principle_book": [
            {
                "program_id": "principle_program_001",
                "status": "validated_principle",
                "closed_at_round": 4,
                "principle_statement": "Finite-gap Mg-Fe oxyfluorides are destabilized relative to matched Mg-Fe oxides when F disrupts Fe-O-Fe network cohesion.",
                "micro_mechanism": "F reduces Fe-O-Fe bridge continuity and introduces weaker mixed Fe-O/F coordination.",
                "reasoning_chain": ["Rounds 1-4 supported primary higher e_hull."],
                "evidence_rounds": [1, 2, 3, 4],
                "boundaries": [],
                "residual_risks": [],
            }
        ],
    }
    ensure_active_principle_program(state, 5)
    round_summary = {"round": 5, "evaluation_summary": {"support_rate": 1.0}, "bundle_results": []}
    postmortem = {
        "status": "finalize",
        "program_id": "principle_program_002",
        "round": 5,
        "hypothesis_status": "supported",
        "principle_update_action": "finalize",
        "current_principle_statement": "Compact Mg-Fe oxyfluorides show the same oxide-relative hull penalty because F disrupts Fe-O-Fe network cohesion.",
        "micro_mechanism": "F breaks dense Fe-O-Fe connectivity in Mg-Fe-O-F relative to Mg-Fe-O.",
        "causal_interpretation": "Same causal theme, narrower compact boundary.",
        "failure_boundaries": ["Compactness increases but does not create the penalty."],
        "unresolved_contradictions": [],
        "next_test_focus": "Move to a different cation family.",
        "experience_book_entry": {
            "principle_statement": "Compact Mg-Fe oxyfluorides show the same oxide-relative hull penalty because F disrupts Fe-O-Fe network cohesion.",
            "reasoning_chain": ["Round 5 added a compactness refinement."],
            "evidence_rounds": [5],
            "boundaries": ["Compactness increases but does not create the penalty."],
            "residual_risks": [],
        },
    }

    update_principle_program_after_postmortem(state, round_summary=round_summary, postmortem=postmortem)

    assert len(state["principle_book"]) == 1
    assert state["principle_book"][0]["evidence_rounds"] == [1, 2, 3, 4, 5]


def test_principle_program_finalize_keeps_distinct_element_scope_separate() -> None:
    state = {
        "history": [],
        "principle_book": [
            {
                "program_id": "principle_program_001",
                "status": "validated_principle",
                "closed_at_round": 4,
                "principle_statement": "Finite-gap Mg-Fe oxyfluorides are destabilized relative to matched Mg-Fe oxides when F disrupts Fe-O-Fe network cohesion.",
                "micro_mechanism": "F reduces Fe-O-Fe bridge continuity.",
                "reasoning_chain": [],
                "evidence_rounds": [4],
                "boundaries": [],
                "residual_risks": [],
            }
        ],
    }
    ensure_active_principle_program(state, 5)
    round_summary = {"round": 5, "evaluation_summary": {"support_rate": 1.0}, "bundle_results": []}
    postmortem = {
        "status": "finalize",
        "program_id": "principle_program_002",
        "round": 5,
        "hypothesis_status": "supported",
        "principle_update_action": "finalize",
        "current_principle_statement": "Compact Mg-Mn oxyfluorides are destabilized relative to Mg-Mn oxides when F disrupts Mn-O-Mn cohesion.",
        "micro_mechanism": "F reduces Mn-O-Mn bridge continuity.",
        "causal_interpretation": "This is a different cation scope.",
        "failure_boundaries": [],
        "unresolved_contradictions": [],
        "next_test_focus": "Refine Mg-Mn.",
        "experience_book_entry": {
            "principle_statement": "Compact Mg-Mn oxyfluorides are destabilized relative to Mg-Mn oxides when F disrupts Mn-O-Mn cohesion.",
            "reasoning_chain": ["Round 5 supported Mg-Mn."],
            "evidence_rounds": [5],
            "boundaries": [],
            "residual_risks": [],
        },
    }

    update_principle_program_after_postmortem(state, round_summary=round_summary, postmortem=postmortem)

    assert len(state["principle_book"]) == 2
    assert state["principle_book"][1]["program_id"] == "principle_program_002"


def test_round_summary_splits_primary_and_control_sun_counts() -> None:
    selected_records = [
        {"material_id": "p1"},
        {"material_id": "p2"},
        {"material_id": "c1"},
        {"material_id": "c2"},
    ]
    analysis_rows = [
        {"index": 1, "e_hull": -0.01},
        {"index": 2, "e_hull": 0.02},
        {"index": 3, "e_hull": -0.03},
        {"index": 4, "e_hull": 0.10},
    ]
    bundles = [
        {
            "id": "b001",
            "prediction_ids": ["p001"],
            "expected_relation": "primary_lower_e_hull_than_control",
            "primary": {"count": 2},
            "control": {"count": 2},
        }
    ]

    bundle_results = bundle_results_from_analysis(selected_records, analysis_rows, bundles)
    summary = summarize_round(
        round_number=1,
        mechanism_payload={"accepted_mechanisms": [{"id": "m001"}]},
        prediction_payload={"accepted_predictions": [{"id": "p001"}]},
        execution_payload={"accepted_bundles": bundles},
        analysis_summary={"count": 4, "mean_e_hull": 0.02, "min_e_hull": -0.03, "max_e_hull": 0.10, "sun_strict_e_hull_lt_0": 2},
        selected_records=selected_records,
        bundle_results=bundle_results,
    )

    assert bundle_results[0]["primary_sun_count"] == 1
    assert bundle_results[0]["control_sun_count"] == 1
    assert summary["evaluation_summary"]["primary_sun_count"] == 1
    assert summary["evaluation_summary"]["control_sun_count"] == 1
    assert summary["evaluation_summary"]["mechanism_validated_sun_count"] == 1


def test_context_can_include_current_stage_inputs() -> None:
    context = build_context_payload(
        state={"schema_version": "x", "history": []},
        pool_summary={"count": 3},
        stage="prediction",
        current_inputs={
            "accepted_mechanisms": [{"id": "m1", "claim": "charge balance"}],
            "target_count": 10,
        },
    )

    assert context["current_inputs"]["accepted_mechanisms"][0]["id"] == "m1"
    assert context["current_inputs"]["target_count"] == 10
    assert "generator_formula_probe_schema_reference" not in context
    assert "material_physics_prediction_consensus" in context["required_output_schema_reference"]
    assert "material_physics_execution_plan" not in context["required_output_schema_reference"]
    assert any("deterministic matched-pair" in item for item in context["instructions"])
    assert any("validation-design agents" in item for item in context["instructions"])
    assert any("discriminating faithful tests" in item for item in context["instructions"])
    assert context["mechanism_search_policy"]["current_mode"] == "exploitation"
    assert any("recent repetition" in item for item in context["instructions"])
    assert context["current_inputs"]["accepted_mechanism_ids"] == ["m1"]
    assert any("target_count=10" in item for item in context["instructions"])


def test_mechanism_context_includes_search_policy() -> None:
    context = build_context_payload(
        state={"schema_version": "x", "current_round": 8, "history": []},
        pool_summary={"count": 3},
        stage="mechanism",
    )

    assert context["mechanism_search_policy"]["current_mode"] == "neighbor_exploration"
    assert context["mechanism_search_policy"]["cycle_fraction"]["exploitation"] == 0.7
    assert any("underexplored_clusters" in item for item in context["instructions"])


def test_context_includes_active_principle_program_and_book() -> None:
    state = {
        "schema_version": "x",
        "current_round": 2,
        "history": [],
        "current_principle_program": {
            "program_id": "principle_program_001",
            "status": "active",
            "started_round": 1,
            "inner_iteration": 2,
            "current_principle_statement": "Compact charge-balanced nitrides are stabilized.",
            "micro_mechanism": "Short M-N bonds lower electrostatic frustration.",
            "evidence_rounds": [{"round": 1, "hypothesis_status": "supported", "support_rate": 1.0}],
        },
        "principle_book": [
            {
                "program_id": "principle_program_000",
                "status": "validated_principle",
                "principle_statement": "Oxide phosphates need narrow boundaries.",
            }
        ],
    }

    context = build_context_payload(state=state, pool_summary={"count": 3}, stage="mechanism")

    assert context["current_principle_program"]["program_id"] == "principle_program_001"
    assert context["principle_book_tail"][0]["program_id"] == "principle_program_000"
    assert any("Primary objective" in item for item in context["instructions"])
    assert any("microscopic causal logic" in item for item in context["instructions"])


def test_execution_context_includes_generator_schema_reference() -> None:
    context = build_context_payload(
        state={"schema_version": "x", "history": []},
        pool_summary={"count": 3},
        stage="execution",
        current_inputs={"accepted_predictions": [{"id": "p1", "claim": "charge balance"}]},
    )

    assert "generator_formula_probe_schema_reference" in context
    assert "material_physics_execution_plan" in context["required_output_schema_reference"]
    assert "material_physics_prediction_consensus" not in context["required_output_schema_reference"]
    assert context["mechanism_search_policy"]["current_mode"] == "exploitation"
    assert any("material IDs" in item for item in context["instructions"])


def test_context_compacts_execution_stage_current_inputs() -> None:
    long_text = "mechanistic explanation " * 120
    context = build_context_payload(
        state={"schema_version": "x", "history": []},
        pool_summary={"count": 3},
        stage="execution",
        current_inputs={
            "accepted_predictions": [
                {
                    "id": "p1",
                    "mechanism_ids": ["m1"],
                    "claim": long_text,
                    "predicted_relation": "primary_lower_e_hull_than_control",
                    "comparison_design": {
                        "primary_count": 5,
                        "control_count": 5,
                        "primary_query": {
                            "material_ids": [f"mp-{idx}" for idx in range(40)],
                            "elements_all": ["Li", "P", "O", "F"],
                            "unsupported_pool_dump": long_text,
                        },
                        "control_query": {"elements_all": ["Li", "V", "O", "F"]},
                    },
                    "falsification_criteria": [long_text, "second criterion", "third criterion"],
                    "full_reasoning_trace": long_text,
                }
            ],
            "target_count": 10,
        },
    )

    prediction = context["current_inputs"]["accepted_predictions"][0]
    assert len(prediction["claim"]) < 400
    assert "full_reasoning_trace" not in prediction
    assert len(prediction["falsification_criteria"]) == 2
    assert len(prediction["falsification_criteria"][0]) < 260
    assert prediction["falsification_criteria"][1] == "second criterion"
    primary_query = prediction["comparison_design"]["primary_query"]
    assert "unsupported_pool_dump" not in primary_query
    assert len(primary_query["material_ids"]) == 40
    assert "material_ids_omitted_count" not in primary_query
    assert "generator_formula_probe_schema_reference" in context


def test_compact_prediction_context_keeps_query_schema_keys_executable() -> None:
    many_exclusions = [
        "H",
        "He",
        "Be",
        "B",
        "C",
        "N",
        "Ne",
        "S",
        "Cl",
        "Ar",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
    ]
    context = build_context_payload(
        state={"status": "running", "current_round": 1, "history": []},
        pool_summary={"count": 3},
        stage="prediction",
        current_inputs={
            "accepted_predictions": [
                {
                    "id": "p001",
                    "mechanism_ids": ["b001"],
                    "claim": "O-F primaries should outperform O-S controls.",
                    "predicted_relation": "primary_lower_e_hull_than_control",
                    "comparison_design": {
                        "primary_count": 5,
                        "control_count": 5,
                        "primary_query": {
                            "elements_all": ["O", "F"],
                            "elements_none": many_exclusions,
                            "preferred_order": ["nelements asc", "formula asc"],
                        },
                        "control_query": {
                            "elements_all": ["O", "S"],
                            "elements_none": ["F"],
                            "preferred_order": ["nelements asc", "formula asc"],
                        },
                    },
                }
            ]
        },
    )

    primary_query = context["current_inputs"]["accepted_predictions"][0]["comparison_design"]["primary_query"]
    assert primary_query["elements_none"] == many_exclusions
    assert "elements_none_omitted_count" not in primary_query


def test_execution_repair_feedback_summarizes_underfilled_mp_pool_branch() -> None:
    plan = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b001",
                "prediction_ids": ["p001"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "source": "mp_pool",
                    "count": 5,
                    "query": {"elements_all": ["Li", "P", "O", "F"], "nelements_min": 4, "nelements_max": 4},
                },
                "control": {"source": "mp_pool", "count": 5, "query": {"elements_all": ["Li", "V", "O", "F"]}},
            }
        ],
    }

    feedback = summarize_materialization_feedback(
        plan=plan,
        materialization_errors=["b001.primary requested 5 records but only 0 materialized from the mp_pool"],
        proposal={"agent": "E", "bundles": plan["accepted_bundles"], "verbose": "x" * 2000},
        critique={"agent": "F", "agree": False, "required_revisions": ["primary branch underfilled"]},
    )

    failed = feedback["failed_branches"][0]
    assert failed["bundle_id"] == "b001"
    assert failed["role"] == "primary"
    assert failed["branch"]["query"]["elements_all"] == ["Li", "P", "O", "F"]
    assert "Do not repeat the same MP-pool query" in failed["required_action"]
    assert "Do not replay the same underfilled MP-pool query" in feedback["required_repair"]


def test_execution_repair_feedback_summarizes_failed_generator_formula_probes() -> None:
    plan = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b001",
                "prediction_ids": ["p001"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "source": "generator",
                    "count": 2,
                    "query": {"elements_all": ["O", "F"]},
                    "formula_probes": ["TiOF2", "VOF3"],
                },
                "control": {"source": "mp_pool", "count": 2, "query": {"elements_all": ["O"]}},
            }
        ],
    }

    feedback = summarize_materialization_feedback(
        plan=plan,
        materialization_errors=["b001.primary requested 2 generator structures but only 0 formula_probes materialized"],
        proposal={"agent": "E", "bundles": plan["accepted_bundles"]},
        critique={"agent": "F", "agree": True},
    )

    failed = feedback["failed_branches"][0]
    assert failed["bundle_id"] == "b001"
    assert failed["role"] == "primary"
    assert failed["source"] == "generator_formula_probes"
    assert failed["failed_formula_probes"] == ["TiOF2", "VOF3"]
    assert "Do not repeat the same generator formula_probes" in failed["required_action"]
    assert "prediction_design_infeasible" in failed["required_action"]
    assert "failed generator formula_probes" in feedback["required_repair"]


def test_execution_repair_prompt_requires_generator_or_new_query() -> None:
    prompt = prompt_execution_repair_proposal("{}")

    assert "do not repeat the same source/query" in prompt
    assert 'source="generator"' in prompt
    assert "formula_probes or structure_dicts" in prompt
    assert "formula_probes materialized zero structures" in prompt


def test_prediction_stage_context_carries_execution_feedback() -> None:
    context = build_context_payload(
        state={"schema_version": "x", "history": []},
        pool_summary={"count": 3},
        stage="prediction",
        current_inputs={
            "accepted_mechanisms": [{"id": "m1", "claim": "charge balance"}],
            "target_count": 10,
        },
        repair_feedback={
            "status": "prediction_design_infeasible",
            "prediction_ids": ["p001"],
            "blocking_issue": "Exact alkali monoxides cannot be faithfully materialized.",
        },
    )

    assert context["repair_feedback"]["status"] == "prediction_design_infeasible"
    assert any("being revisited after E/F execution review" in item for item in context["instructions"])
    assert any("must not repeat the same infeasible" in item for item in context["instructions"])


def test_prediction_feedback_context_preserves_repair_constraints_without_dialogue_bloat() -> None:
    prediction = {
        "id": "p001_revised",
        "mechanism_ids": ["m001_refined"],
        "claim": "Charge-balanced difluorides should beat monofluoride controls.",
        "predicted_relation": "primary_lower_e_hull_than_control",
        "comparison_design": {
            "primary_count": 5,
            "control_count": 5,
            "primary_query": {"formula_regex": "^(MgF2|CaF2)$", "preferred_order": ["formation_energy_per_atom asc"]},
            "control_query": {"formula_regex": "^(MgF|CaF)$", "preferred_order": ["formation_energy_per_atom asc"]},
        },
    }
    context = build_context_payload(
        state={"schema_version": "x", "history": []},
        pool_summary={"count": 140588, "common_elements": list("abcdefghijklmnopqrst")},
        stage="prediction",
        current_inputs={"accepted_mechanisms": [{"id": "m001_refined", "claim": "charge balance"}], "target_count": 10},
        repair_feedback={
            "status": "prediction_design_infeasible",
            "feedback_round": 1,
            "prediction_ids": ["p001_revised"],
            "blocking_issue": "control branch has only 2 MP-pool monofluoride rows and generator controls are charge-imbalanced",
            "why_execution_cannot_fix_it": "E/F cannot change stoichiometry, branch counts, or chemistry without changing C/D's prediction.",
            "required_cd_reconsideration": ["revise control chemistry", "keep planned_material_count >= target_count"],
            "previous_prediction_summary": {
                "status": "consensus",
                "accepted_predictions": [prediction],
                "predictions": [prediction],
                "dialogue": [{"payload": "verbose"} for _ in range(20)],
                "consensus_summary": "x" * 2000,
            },
        },
    )

    feedback = context["repair_feedback"]
    assert "charge-imbalanced" in feedback["blocking_issue"]
    assert "revise control chemistry" in feedback["required_cd_reconsideration"]
    previous = feedback["previous_prediction_summary"]
    assert "dialogue" not in previous
    assert "accepted_predictions" not in previous
    kept = previous["predictions"][0]
    assert kept["id"] == "p001_revised"
    assert kept["comparison_design"]["primary_count"] == 5
    assert kept["comparison_design"]["control_query"]["formula_regex"] == "^(MgF|CaF)$"
    assert context["candidate_pool_summary"]["common_elements_omitted_count"] == 4


def test_execution_prompts_support_prediction_design_feedback() -> None:
    proposal_prompt = prompt_execution_proposal("{}")
    repair_prompt = prompt_execution_repair_proposal("{}")
    final_prompt = prompt_execution_final(
        "{}",
        {"agent": "E", "status": "prediction_design_infeasible", "accepted_bundles": []},
        {"agent": "F", "agree": True, "required_revisions": []},
    )

    for prompt in (proposal_prompt, repair_prompt, final_prompt):
        assert "prediction_design_infeasible" in prompt
        assert "prediction_design_feedback" in prompt


def test_prediction_feedback_from_execution_compacts_ef_issue() -> None:
    execution_debate = {
        "status": "prediction_design_infeasible",
        "consensus_summary": "Exact LiO controls cannot be faithfully materialized.",
        "prediction_design_feedback": {
            "prediction_ids": ["p001"],
            "blocking_issue": "LiO/NaO controls fail formal charge balance.",
            "why_execution_cannot_fix_it": ["Generator would violate preflight."],
            "required_cd_reconsideration": ["Choose a different control family."],
        },
    }
    prediction_debate = {"accepted_predictions": [{"id": "p001", "claim": "charge balance"}]}

    feedback = prediction_feedback_from_execution(
        execution_debate=execution_debate,
        prediction_debate=prediction_debate,
        feedback_round=1,
    )

    assert feedback["status"] == "prediction_design_infeasible"
    assert feedback["prediction_ids"] == ["p001"]
    assert "LiO/NaO" in feedback["blocking_issue"]
    assert feedback["required_cd_reconsideration"] == ["Choose a different control family."]
    assert is_prediction_design_infeasible(execution_debate)


def test_execution_normalizer_accepts_prediction_design_infeasible_consensus() -> None:
    payload = normalize_role_payload(
        {
            "status": "prediction_design_infeasible",
            "accepted_bundles": [],
            "rejected_bundles": [{"id": "b001", "reason": "control impossible"}],
            "prediction_design_feedback": {
                "prediction_ids": ["p001"],
                "blocking_issue": "control branch impossible",
            },
            "consensus_summary": "send back to C/D",
        },
        "execution_consensus",
    )

    assert valid_shape(payload, "execution_consensus")
    assert is_prediction_design_infeasible(payload)


def test_parse_args_has_prediction_execution_feedback_limit() -> None:
    args = parse_args([])
    assert args.max_prediction_execution_feedback_rounds == 2
    assert args.critic_counterproposal_after == 1


def test_context_with_debate_history_adds_strict_alternating_protocol() -> None:
    context = context_with_debate_history(
        '{"stage":"prediction","current_inputs":{"target_count":10}}',
        [
            {
                "role": "C",
                "cycle": 1,
                "payload": {
                    "agent": "C",
                    "predictions": [
                        {
                            "id": "p001",
                            "claim": "primary lower",
                            "comparison_design": {
                                "primary_count": 5,
                                "control_count": 5,
                                "primary_query": {"elements_all": ["Li"]},
                                "control_query": {"elements_all": ["Na"]},
                            },
                        }
                    ],
                },
            },
            {"role": "D", "cycle": 1, "payload": {"agent": "D", "required_revisions": ["under-supported"]}},
        ],
    )
    payload = json.loads(context)

    assert payload["debate_protocol"]["mode"] == "strict_alternating_proposal_review"
    assert "RAG" in payload["debate_protocol"]["rag_policy"]
    assert [turn["role"] for turn in payload["debate_history"]] == ["C", "D"]
    assert payload["debate_history"][0]["payload"]["predictions"][0]["comparison_design"]["primary_count"] == 5


def test_execution_prompts_preserve_reviewable_structure_payloads() -> None:
    structure = Structure(Lattice.cubic(4.0), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    proposal = {
        "status": "ok",
        "agent": "E",
        "bundles": [
            {
                "id": "b001",
                "prediction_ids": ["p001"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "source": "generator",
                    "count": 1,
                    "query": {"elements_all": ["Li", "F"]},
                    "structure_dicts": [structure.as_dict()],
                },
                "control": {"source": "mp_pool", "count": 1, "query": {"elements_all": ["Na", "F"]}},
                "rationale_summary": "long rationale " * 200,
            }
        ],
    }
    prompt = prompt_execution_critique("{}", proposal, cycle=1)

    assert '"structure_dicts"' in prompt
    assert "structure_dict_count" not in prompt
    assert '"sites"' in prompt
    assert '"_prompt_summary"' in prompt
    assert '"formal_charge_imbalance"' in prompt
    assert "long rationale " * 50 not in prompt


def test_execution_final_prompt_omits_executable_structure_payloads() -> None:
    structure = Structure(Lattice.cubic(4.0), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    proposal = {
        "status": "ok",
        "agent": "E",
        "agree": True,
        "bundles": [
            {
                "id": "b001",
                "prediction_ids": ["p001"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "source": "generator",
                    "count": 1,
                    "query": {"elements_all": ["Li", "F"]},
                    "structure_dicts": [structure.as_dict()],
                },
                "control": {
                    "source": "generator",
                    "count": 1,
                    "formula_probes": ["NaF"],
                },
                "rationale_summary": "accepted",
                "selection_notes": "accepted",
            }
        ],
    }
    critique = {
        "status": "accepted",
        "agent": "F",
        "agree": True,
        "concede": False,
        "required_revisions": [],
        "accepted_bundle_ids": ["b001"],
    }

    prompt = prompt_execution_final("{}", proposal, critique)

    assert "accepted_bundle_ids" in prompt
    assert '"structure_dicts":' not in prompt
    assert '"formula_probes":' not in prompt
    assert '"sites":' not in prompt
    assert "executable_structure_dicts_stored_by_controller" in prompt
    assert "executable_formula_probes_stored_by_controller" in prompt


def test_execution_final_payload_stores_long_notes_without_truncation_text() -> None:
    long_rationale = "charge balance rationale " * 80
    long_selection = "selection rule details " * 80
    payload = {
        "status": "ok",
        "agent": "E",
        "bundles": [
            {
                "id": "b001",
                "prediction_ids": ["p001"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "rationale_summary": long_rationale,
                "selection_notes": long_selection,
                "primary": {"source": "mattergen", "count": 1},
                "control": {"source": "mattergen", "count": 1},
            }
        ],
    }

    compact = mvp.compact_execution_payload_for_final_prompt(payload)
    text = json.dumps(compact)
    bundle = compact["bundles"][0]

    assert "rationale_summary" not in bundle
    assert "selection_notes" not in bundle
    assert bundle["compact_payload_is_not_executable_original"] is True
    assert bundle["rationale_summary_stored_by_controller"] is True
    assert bundle["rationale_summary_chars"] == len(long_rationale.strip())
    assert bundle["selection_notes_stored_by_controller"] is True
    assert bundle["selection_notes_chars"] == len(long_selection.strip())
    assert "..." not in text


def test_execution_debate_history_marks_compact_payload_non_authoritative() -> None:
    context = context_with_debate_history(
        '{"stage":"execution","current_inputs":{"target_count":2}}',
        [
            {
                "role": "E",
                "cycle": 1,
                "payload": {
                    "agent": "E",
                    "bundles": [
                        {
                            "id": "b001",
                            "prediction_ids": ["p001"],
                            "expected_relation": "primary_lower_e_hull_than_control",
                            "rationale_summary": "rationale text " * 120,
                            "selection_notes": "selection text " * 120,
                            "primary": {"source": "mattergen", "count": 1},
                            "control": {"source": "mattergen", "count": 1},
                        }
                    ],
                },
            }
        ],
    )

    payload = json.loads(context)
    bundle = payload["debate_history"][0]["payload"]["bundles"][0]

    assert bundle["compact_payload_is_not_executable_original"] is True
    assert "rationale_summary" not in bundle
    assert "selection_notes" not in bundle
    assert bundle["rationale_summary_stored_by_controller"] is True
    assert "compact_payload_policy" in payload["debate_protocol"]


def test_execution_consensus_retry_prompt_omits_invalid_structure_payload() -> None:
    invalid = '{"status":"consensus","accepted_bundles":[{"id":"b001","primary":{"structure_dicts":[{"bad copied structure": true}]}}'

    prompt = retry_prompt(
        "execution_consensus",
        "Agent F final task with compact accepted bundle summaries.",
        invalid,
        "JSONDecodeError: Expecting ',' delimiter",
    )

    assert "accepted_bundle_ids" in prompt
    assert "Do not output structure_dicts" in prompt
    assert "bad copied structure" not in prompt
    assert "Previous invalid execution_consensus output omitted" in prompt


def test_execution_prompts_preserve_short_formula_probe_lists() -> None:
    proposal = {
        "status": "ok",
        "agent": "E",
        "bundles": [
            {
                "id": "b002",
                "prediction_ids": ["p002"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {"source": "mp_pool", "count": 2, "query": {"elements_all": ["Cs", "Pb", "Br"]}},
                "control": {
                    "source": "generator",
                    "count": 2,
                    "formula_probes": ["LiPbBr3", "Li2PbBr4"],
                },
            }
        ],
    }

    prompt = prompt_execution_critique("{}", proposal, cycle=1)

    assert '"formula_probes"' in prompt
    assert "LiPbBr3" in prompt
    assert "formula_probe_count" not in prompt


def test_execution_prompts_preserve_structured_formula_probe_lists() -> None:
    probe = {
        "template": "ABO3",
        "roles": {
            "A": {"element": "Ca", "oxidation_state": 2},
            "B": {"element": "Ti", "oxidation_state": 4},
            "O": {"element": "O", "oxidation_state": -2},
        },
    }
    proposal = {
        "status": "ok",
        "agent": "E",
        "bundles": [
            {
                "id": "b001",
                "prediction_ids": ["p001"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {"source": "generator", "count": 1, "formula_probes": [probe]},
                "control": {"source": "mp_pool", "count": 1, "material_ids": ["mp-1"]},
            }
        ],
    }

    prompt = prompt_execution_critique("{}", proposal, cycle=1)

    assert '"formula_probes"' in prompt
    assert '"template": "ABO3"' in prompt
    assert '"oxidation_state": 4' in prompt
    assert '"material_ids"' in prompt
    assert "mp-1" in prompt
    assert "formula_probe_count" not in prompt


def test_execution_repair_context_uses_structure_summaries_not_fake_structure_dicts() -> None:
    structure = Structure(Lattice.cubic(4.0), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    context = build_context_payload(
        state={"schema_version": "x", "history": []},
        pool_summary={"count": 3},
        stage="execution",
        current_inputs={"accepted_predictions": [{"id": "p001", "claim": "x"}]},
        repair_feedback={
            "proposal": {
                "agent": "E",
                "bundles": [
                    {
                        "id": "b001",
                        "primary": {"source": "generator", "count": 1, "structure_dicts": [structure.as_dict()]},
                    }
                ],
            }
        },
    )

    primary = context["repair_feedback"]["proposal"]["bundles"][0]["primary"]
    assert "structure_dict_summaries" in primary
    assert "structure_dicts" not in primary
    assert primary["structure_dicts_available_in_prior_payload"] is True
    assert primary["structure_dict_summaries"][0]["formula"] == "LiF"


def test_prediction_prompts_preserve_target_budget_prediction_lists() -> None:
    predictions = [
        {
            "id": f"p{index:03d}",
            "mechanism_ids": ["m001"],
            "claim": f"prediction {index}",
            "predicted_relation": "primary_lower_e_hull_than_control",
            "comparison_design": {
                "primary_count": 1,
                "control_count": 1,
                "primary_query": {"formula_regex": f"^A{index}O$"},
                "control_query": {"formula_regex": f"^B{index}O$"},
            },
        }
        for index in range(1, 6)
    ]
    proposal = {
        "status": "consensus",
        "agent": "C",
        "planned_material_count": 10,
        "target_count": 10,
        "accepted_predictions": predictions,
        "predictions": predictions,
    }

    prompt = prompt_prediction_critique("{}", proposal, cycle=1)

    assert '"p005"' in prompt
    assert "accepted_predictions_omitted_count" not in prompt
    assert "predictions_omitted_count" not in prompt


def test_prediction_prompt_deduplicates_agent_c_prediction_lists() -> None:
    predictions = [
        {
            "id": f"p{index:03d}",
            "claim": f"prediction {index}",
            "comparison_design": {
                "primary_count": 1,
                "control_count": 1,
                "primary_query": {"formula_regex": f"^A{index}O$"},
                "control_query": {"formula_regex": f"^B{index}O$"},
            },
        }
        for index in range(1, 6)
    ]
    proposal = {
        "status": "consensus",
        "agent": "C",
        "planned_material_count": 10,
        "accepted_predictions": predictions,
        "predictions": predictions,
    }

    prompt = prompt_prediction_critique("{}", proposal, cycle=1)

    assert '"p005"' in prompt
    assert '"accepted_predictions"' not in prompt
    assert prompt.count('"p005"') == 1


def test_execution_context_preserves_target_budget_accepted_predictions() -> None:
    predictions = [
        {
            "id": f"p{index:03d}",
            "claim": f"prediction {index}",
            "comparison_design": {"primary_count": 1, "control_count": 1},
        }
        for index in range(1, 6)
    ]

    context = build_context_payload(
        state={"schema_version": "x", "history": []},
        pool_summary={"count": 3},
        stage="execution",
        current_inputs={"accepted_predictions": predictions, "target_count": 10},
    )

    accepted = context["current_inputs"]["accepted_predictions"]
    assert [item["id"] for item in accepted] == ["p001", "p002", "p003", "p004", "p005"]
    assert "accepted_predictions_omitted_count" not in context["current_inputs"]


def test_prediction_retry_prompt_rejects_materialization_shape() -> None:
    prompt = retry_prompt(
        "prediction_agent_c",
        "Agent C: revise after critique.",
        '{"status":"ok","formula_probes":[]}',
        "output did not match required shape for prediction_agent_c",
    )

    assert "predictions" in prompt
    assert "Do not output formula_probes" in prompt
    assert "non-identical primary_query" in prompt
    assert "Do not request tools in this JSON repair response" in prompt
    assert "Do not concatenate a tool_request JSON object" in prompt
    assert "material_id" not in prompt


def test_call_json_uses_transient_retry_prompt_for_llm_error() -> None:
    class FlakyClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_text(self, *, system: str, user: str, metadata: dict) -> str:
            self.prompts.append(user)
            if len(self.prompts) == 1:
                raise LLMError("The read operation timed out")
            return '{"agent":"A","mechanisms":[],"agree":false,"concede":false}'

    client = FlakyClient()

    result = call_json(
        client,  # type: ignore[arg-type]
        system="system",
        user="Agent A: propose mechanisms.",
        role="mechanism_agent_a",
        metadata={"role": "mechanism_agent_a"},
        json_repair_attempts=1,
    )

    assert result["agent"] == "A"
    assert len(client.prompts) == 2
    assert "failed before usable text was extracted" in client.prompts[1]
    assert "previous output for role" not in client.prompts[1]


def test_call_json_raises_recoverable_failure_for_server_overload() -> None:
    class OverloadedClient:
        def complete_text(self, *, system: str, user: str, metadata: dict) -> str:
            raise LLMError(
                "LLM stream ended with error: {'error': {'code': 'server_is_overloaded', "
                "'message': 'Our servers are currently overloaded. Please try again later.'}}"
            )

    with pytest.raises(RecoverableLLMFailure) as exc_info:
        call_json(
            OverloadedClient(),  # type: ignore[arg-type]
            system="system",
            user="Agent B: review mechanisms.",
            role="mechanism_agent_b",
            metadata={"role": "mechanism_agent_b", "round": 88},
            json_repair_attempts=1,
        )

    assert exc_info.value.role == "mechanism_agent_b"
    assert exc_info.value.metadata["round"] == 88
    assert exc_info.value.attempts == 2
    assert is_recoverable_llm_error_message(exc_info.value.error)


def test_call_json_role_retries_recoverable_llm_without_json_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_ROLE_RECOVERABLE_RETRIES", "2")
    monkeypatch.setenv("LLM_ROLE_RECOVERABLE_RETRY_SLEEP", "0")

    class EventuallyAvailableClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.metadata: list[dict] = []

        def complete_text(self, *, system: str, user: str, metadata: dict) -> str:
            self.prompts.append(user)
            self.metadata.append(dict(metadata))
            if len(self.prompts) < 3:
                raise LLMError(
                    "LLM stream ended with error: {'error': {'code': 'server_is_overloaded', "
                    "'message': 'Our servers are currently overloaded. Please try again later.'}}"
                )
            return '{"agent":"B","mechanisms":[],"agree":false,"concede":false}'

    client = EventuallyAvailableClient()

    result = call_json(
        client,  # type: ignore[arg-type]
        system="system",
        user="Agent B: review mechanisms.",
        role="mechanism_agent_b",
        metadata={"role": "mechanism_agent_b", "round": 90},
        json_repair_attempts=0,
    )

    assert result["agent"] == "B"
    assert len(client.prompts) == 3
    assert client.prompts == ["Agent B: review mechanisms."] * 3
    assert [entry["json_retry_attempt"] for entry in client.metadata] == [0, 0, 0]


def test_next_round_to_run_retries_incomplete_recoverable_round() -> None:
    state = {"status": "recoverable_llm_failure", "current_round": 88, "history": [{"round": 87}]}
    assert next_round_to_run(state) == 88


def test_next_round_to_run_advances_after_completed_round() -> None:
    state = {"status": "running", "current_round": 88, "history": [{"round": 88}]}
    assert next_round_to_run(state) == 89


def test_call_json_runs_local_agent_tool_loop(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("agent runtime can read this file\n", encoding="utf-8")

    class ToolClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_text(self, *, system: str, user: str, metadata: dict) -> str:
            self.prompts.append(user)
            if len(self.prompts) == 1:
                return json.dumps(
                    {
                        "status": "tool_request",
                        "tool_calls": [
                            {
                                "id": "readme",
                                "name": "read_project_file",
                                "arguments": {"path": "README.md", "max_chars": 40},
                            },
                            {
                                "id": "artifact",
                                "name": "write_agent_artifact",
                                "arguments": {"path": "note.txt", "content": "tool loop worked"},
                            },
                        ],
                    }
                )
            assert "LOCAL_AGENT_TOOL_RESULTS" in user
            assert "agent runtime can read this file" in user
            return '{"agent":"A","mechanisms":[],"agree":false,"concede":false}'

    client = ToolClient()
    client.local_agent_runtime = LocalAgentRuntime(  # type: ignore[attr-defined]
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "agent_artifacts",
        max_steps=2,
    )

    result = call_json(
        client,  # type: ignore[arg-type]
        system="system",
        user="Agent A: propose mechanisms.",
        role="mechanism_agent_a",
        metadata={"role": "mechanism_agent_a"},
        json_repair_attempts=0,
    )

    assert result["agent"] == "A"
    assert len(client.prompts) == 2
    assert (tmp_path / "agent_artifacts" / "note.txt").read_text(encoding="utf-8") == "tool loop worked"
    assert list((tmp_path / "traces").glob("agent_trace_*.json"))


def test_local_agent_query_candidate_pool_tool(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "material_id": "mp-1",
                        "formula": "LiFeO2",
                        "elements": ["Li", "Fe", "O"],
                        "nelements": 3,
                        "nsites": 4,
                        "formation_energy_per_atom": -1.0,
                    }
                ),
                json.dumps(
                    {
                        "material_id": "mp-2",
                        "formula": "NaCl",
                        "elements": ["Na", "Cl"],
                        "nelements": 2,
                        "nsites": 2,
                        "formation_energy_per_atom": -0.5,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        candidate_pool_path=pool,
    )

    result = runtime.execute_tool_call(
        {
            "id": "q1",
            "name": "query_candidate_pool",
            "arguments": {
                "query": {"elements_all": ["Li", "O"], "preferred_order": ["formation_energy_per_atom asc"]},
                "count": 3,
            },
        }
    )

    assert result.ok
    assert result.output["valid_query"] is True
    assert result.output["match_count"] == 1
    assert result.output["distinct_formula_count"] == 1
    assert result.output["first_distinct_formulas"] == ["LiFeO2"]
    assert result.output["examples"][0]["material_id"] == "mp-1"


def test_candidate_pool_batch_keeps_primary_and_control_summaries(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    records = []
    for index in range(12):
        records.append(
            {
                "material_id": f"mp-li-{index}",
                "formula": "LiTi(PO3)4" if index < 6 else f"Li{index}TiP{index}O{index + 4}",
                "elements": ["Li", "Ti", "P", "O"],
                "nelements": 4,
                "nsites": 34 + index,
                "band_gap": 0.1 * index,
                "volume_per_atom": 10.0 + index,
            }
        )
    for index in range(12):
        records.append(
            {
                "material_id": f"mp-ti-{index}",
                "formula": "Ti(PO3)4" if index < 5 else f"Ti{index}P{index}O{index + 4}",
                "elements": ["Ti", "P", "O"],
                "nelements": 3,
                "nsites": 34 + index,
                "band_gap": 0.1 * index,
                "volume_per_atom": 9.0 + index,
            }
        )
    pool.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        candidate_pool_path=pool,
    )

    primary = runtime.execute_tool_call(
        {
            "id": "primary",
            "name": "query_candidate_pool",
            "arguments": {
                "query": {"elements_all": ["Li", "Ti", "P", "O"], "preferred_order": ["nsites asc"]},
                "count": 30,
            },
        }
    )
    control = runtime.execute_tool_call(
        {
            "id": "control",
            "name": "query_candidate_pool",
            "arguments": {
                "query": {"elements_all": ["Ti", "P", "O"], "elements_none": ["Li"], "preferred_order": ["nsites asc"]},
                "count": 30,
            },
        }
    )

    assert primary.output["distinct_formula_count"] >= 5
    assert control.output["distinct_formula_count"] >= 5
    assert len(primary.output["examples"]) <= 4
    transcript = runtime._append_tool_results(
        "",
        [
            {"id": primary.id, "name": primary.name, "ok": primary.ok, "output": dict(primary.output)},
            {"id": control.id, "name": control.name, "ok": control.ok, "output": dict(control.output)},
        ],
    )
    assert '"id": "primary"' in transcript
    assert '"id": "control"' in transcript
    assert '"distinct_formula_count"' in transcript
    assert "omitted_items" not in transcript


def test_local_agent_requires_candidate_pool_for_prediction_roles(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps(
            {
                "material_id": "mp-1",
                "formula": "LiFeO2",
                "elements": ["Li", "Fe", "O"],
                "nelements": 3,
                "nsites": 4,
                "formation_energy_per_atom": -1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class InitiallySkippingCandidatePoolClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_text(self, *, system: str, user: str, metadata: dict) -> str:
            self.prompts.append(user)
            if len(self.prompts) == 1:
                return '{"status":"consensus","agent":"C","predictions":[],"agree":true,"concede":false}'
            if len(self.prompts) == 2:
                assert "MANDATORY_CANDIDATE_POOL_CHECK_MISSING" in user
                return json.dumps(
                    {
                        "status": "tool_request",
                        "tool_calls": [
                            {
                                "id": "pool",
                                "name": "query_candidate_pool",
                                "arguments": {"query": {"elements_all": ["Li", "O"]}, "count": 2},
                            }
                        ],
                    }
                )
            assert "LOCAL_AGENT_TOOL_RESULTS" in user
            return '{"status":"consensus","agent":"C","predictions":[],"agree":true,"concede":false}'

    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        candidate_pool_path=pool,
        max_steps=4,
    )

    text = runtime.complete_text(
        InitiallySkippingCandidatePoolClient(),
        system="system",
        user="Agent C must use query_candidate_pool before final JSON.",
        metadata={"role": "prediction_agent_c"},
    )

    assert json.loads(text)["agent"] == "C"
    trace = json.loads(next((tmp_path / "traces").glob("agent_trace_*.json")).read_text(encoding="utf-8"))
    assert trace["mandatory_candidate_pool_retry"] is True
    assert trace["tool_steps"][0]["results"][0]["name"] == "query_candidate_pool"
    assert trace["tool_steps"][0]["results"][0]["ok"] is True


def test_local_agent_requires_candidate_pool_from_system_prompt_for_execution_roles(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps(
            {
                "material_id": "mp-1",
                "formula": "LiVOF2",
                "elements": ["Li", "V", "O", "F"],
                "nelements": 4,
                "nsites": 5,
                "formation_energy_per_atom": -2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class InitiallySkippingExecutionPoolClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_text(self, *, system: str, user: str, metadata: dict) -> str:
            self.prompts.append(user)
            if len(self.prompts) == 1:
                return '{"status":"consensus","accepted_bundles":[]}'
            if len(self.prompts) == 2:
                assert "MANDATORY_CANDIDATE_POOL_CHECK_MISSING" in user
                return json.dumps(
                    {
                        "status": "tool_request",
                        "tool_calls": [
                            {
                                "id": "branch",
                                "name": "query_candidate_pool",
                                "arguments": {"query": {"elements_all": ["Li", "V", "O", "F"]}, "count": 1},
                            }
                        ],
                    }
                )
            assert "LOCAL_AGENT_TOOL_RESULTS" in user
            return '{"status":"consensus","accepted_bundles":[]}'

    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        candidate_pool_path=pool,
        max_steps=4,
    )

    text = runtime.complete_text(
        InitiallySkippingExecutionPoolClient(),
        system="Before final execution JSON, request query_candidate_pool for every MP-pool branch.",
        user="Agent E: translate the accepted prediction into a bundle.",
        metadata={"role": "execution_agent_e"},
    )

    assert json.loads(text)["status"] == "consensus"
    trace = json.loads(next((tmp_path / "traces").glob("agent_trace_*.json")).read_text(encoding="utf-8"))
    assert trace["mandatory_candidate_pool_retry"] is True
    assert trace["tool_steps"][0]["results"][0]["name"] == "query_candidate_pool"
    assert trace["tool_steps"][0]["results"][0]["ok"] is True


def test_local_agent_requires_candidate_pool_for_execution_counterproposal_roles(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps(
            {
                "material_id": "mp-1",
                "formula": "LiVOF2",
                "elements": ["Li", "V", "O", "F"],
                "nelements": 4,
                "formation_energy_per_atom": -2.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class CounterproposalClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_text(self, *, system: str, user: str, metadata: dict) -> str:
            self.prompts.append(user)
            if len(self.prompts) == 1:
                return '{"status":"consensus","accepted_bundles":[]}'
            if len(self.prompts) == 2:
                assert "MANDATORY_CANDIDATE_POOL_CHECK_MISSING" in user
                return json.dumps(
                    {
                        "status": "tool_request",
                        "tool_calls": [
                            {
                                "id": "branch",
                                "name": "query_candidate_pool",
                                "arguments": {"query": {"elements_all": ["Li", "V", "O", "F"]}, "count": 1},
                            }
                        ],
                    }
                )
            return '{"status":"consensus","accepted_bundles":[]}'

    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        candidate_pool_path=pool,
        max_steps=4,
    )

    runtime.complete_text(
        CounterproposalClient(),
        system="Before final audit JSON, request query_candidate_pool for every MP-pool branch.",
        user="Agent F: produce a counterproposal.",
        metadata={"role": "execution_agent_f_counterproposal"},
    )

    trace = json.loads(next((tmp_path / "traces").glob("agent_trace_*.json")).read_text(encoding="utf-8"))
    assert trace["mandatory_candidate_pool_retry"] is True
    assert trace["tool_steps"][0]["results"][0]["name"] == "query_candidate_pool"


def test_local_agent_forces_candidate_pool_from_prompt_queries_after_retry(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "material_id": "mp-1",
                        "formula": "LiVOF2",
                        "elements": ["Li", "V", "O", "F"],
                        "nelements": 4,
                        "formation_energy_per_atom": -2.0,
                    }
                ),
                json.dumps(
                    {
                        "material_id": "mp-2",
                        "formula": "LiVO2",
                        "elements": ["Li", "V", "O"],
                        "nelements": 3,
                        "formation_energy_per_atom": -1.5,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class NeverRequestsCandidatePoolClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_text(self, *, system: str, user: str, metadata: dict) -> str:
            self.prompts.append(user)
            if len(self.prompts) < 3:
                return '{"status":"consensus","agent":"D","required_revisions":[],"agree":true,"concede":true}'
            assert "LOCAL_AGENT_TOOL_RESULTS" in user
            assert "MANDATORY_CANDIDATE_POOL_FORCED_BY_CONTROLLER" in user
            return '{"status":"consensus","agent":"D","required_revisions":[],"agree":true,"concede":true}'

    prompt_context = {
        "accepted_predictions": [
            {
                "id": "p001",
                "comparison_design": {
                    "primary_count": 1,
                    "control_count": 1,
                    "primary_query": {"elements_all": ["Li", "V", "O", "F"]},
                    "control_query": {"elements_all": ["Li", "V", "O"], "elements_none": ["F"]},
                },
            }
        ]
    }
    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        candidate_pool_path=pool,
        max_steps=4,
    )

    text = runtime.complete_text(
        NeverRequestsCandidatePoolClient(),
        system="Before final critique JSON, request query_candidate_pool for the proposal pools.",
        user="Agent C: critique counterproposal.\n```json\n" + json.dumps(prompt_context) + "\n```",
        metadata={"role": "prediction_agent_c_reverse_critique"},
    )

    assert json.loads(text)["agree"] is True
    trace = json.loads(next((tmp_path / "traces").glob("agent_trace_*.json")).read_text(encoding="utf-8"))
    assert trace["mandatory_candidate_pool_retry"] is True
    assert trace["forced_mandatory_candidate_pool"] is True
    assert trace["tool_steps"][0]["forced"] is True
    assert [item["name"] for item in trace["tool_steps"][0]["results"]] == [
        "query_candidate_pool",
        "query_candidate_pool",
    ]


def test_local_agent_summarize_mechanism_evidence_tool(tmp_path: Path) -> None:
    work_dir = tmp_path / "physics_mvp_runs" / "current"
    round_dir = work_dir / "round_001" / "analysis"
    round_dir.mkdir(parents=True)
    (round_dir / "summary.json").write_text(
        json.dumps(
            {
                "merged_evaluator_novelty": {"sun_score": 0.2, "sun_both_novel_count": 2},
                "sun_strict_e_hull_lt_0": 2,
            }
        ),
        encoding="utf-8",
    )
    state = {
        "status": "running",
        "current_round": 2,
        "history": [
            {
                "round": 1,
                "accepted_mechanisms": [
                    {
                        "id": "m001",
                        "claim": "La-O-F-S hard/soft anion complementarity stabilizes matched compounds.",
                        "rationale_summary": "O/F hard bonding and S polarizability can relieve local mismatch.",
                    }
                ],
                "accepted_predictions": [
                    {
                        "id": "p001",
                        "mechanism_ids": ["m001"],
                        "claim": "La-O-F-S should beat La-O-F controls.",
                    },
                    {
                        "id": "p002",
                        "mechanism_ids": ["m001"],
                        "claim": "Strict La-F-S should beat La-S controls.",
                    },
                ],
                "bundle_results": [
                    {
                        "bundle_id": "b001",
                        "prediction_ids": ["p001"],
                        "supported": True,
                        "delta": -0.04,
                        "expected_relation": "primary_lower_e_hull_than_control",
                    },
                    {
                        "bundle_id": "b002",
                        "prediction_ids": ["p002"],
                        "supported": False,
                        "delta": 0.9,
                        "expected_relation": "primary_lower_e_hull_than_control",
                    },
                ],
                "evaluation_summary": {
                    "support_rate": 0.5,
                    "supported_bundle_count": 1,
                    "bundle_count": 2,
                    "min_e_hull": -0.01,
                    "mean_e_hull": 0.03,
                    "stable_count": 1,
                },
            }
        ],
    }
    (work_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        state_path=work_dir / "state.json",
    )

    result = runtime.execute_tool_call(
        {
            "id": "evidence",
            "name": "summarize_mechanism_evidence",
            "arguments": {"limit": 4, "include_failed": True},
        }
    )

    assert result.ok
    assert result.output["tool_status"] == "ok"
    assert "status" not in result.output
    assert result.output["controller_state_status"] == "running"
    assert result.output["history_len"] == 1
    assert result.output["offset"] == 0
    assert result.output["next_offset"] is None
    assert result.output["has_more"] is False
    evidence_round = result.output["rounds"][0]
    assert evidence_round["sun_score"] == 0.2
    assert evidence_round["supported_predictions"][0]["prediction_id"] == "p001"
    assert evidence_round["failed_predictions"][0]["prediction_id"] == "p002"
    assert result.output["search_policy"]["current_mode"] == "exploitation"
    views = result.output["evidence_views"]
    assert views["top_successes"][0]["round"] == 1
    assert "near_misses" in views
    assert "failure_boundaries" in views
    assert "underexplored_clusters" in views


def test_local_agent_summarize_mechanism_evidence_renames_error_state_status(tmp_path: Path) -> None:
    work_dir = tmp_path / "physics_mvp_runs" / "current"
    work_dir.mkdir(parents=True)
    (work_dir / "state.json").write_text(
        json.dumps({"status": "error", "current_round": 7, "history": []}),
        encoding="utf-8",
    )
    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        state_path=work_dir / "state.json",
    )

    result = runtime.execute_tool_call(
        {
            "id": "evidence",
            "name": "summarize_mechanism_evidence",
            "arguments": {"brief": True},
        }
    )

    assert result.ok
    assert result.output["tool_status"] == "ok"
    assert result.output["controller_state_status"] == "error"
    assert "status" not in result.output


def test_local_agent_summarize_mechanism_evidence_falls_back_on_internal_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work_dir = tmp_path / "physics_mvp_runs" / "current"
    work_dir.mkdir(parents=True)
    (work_dir / "state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "current_round": 8,
                "history": [{"round": 1, "evaluation_summary": {"support_rate": 1.0}}],
                "principle_book": [{"program_id": "p001", "principle_statement": "compact oxide rule"}],
            }
        ),
        encoding="utf-8",
    )

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic evidence view failure")

    monkeypatch.setattr("crystal_llm.local_agent_runtime._mechanism_evidence_views", boom)
    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        state_path=work_dir / "state.json",
    )

    result = runtime.execute_tool_call(
        {
            "id": "evidence",
            "name": "summarize_mechanism_evidence",
            "arguments": {"brief": True},
        }
    )

    assert result.ok
    assert result.output["tool_status"] == "fallback"
    assert result.output["rag_fallback_used"] is True
    assert result.output["controller_state_status"] == "running"
    assert result.output["principle_book_tail"]


def test_local_agent_summarize_mechanism_evidence_paginates(tmp_path: Path) -> None:
    work_dir = tmp_path / "physics_mvp_runs" / "current"
    work_dir.mkdir(parents=True)
    state = {
        "history": [
            {"round": 1, "evaluation_summary": {"support_rate": 0.1}},
            {"round": 2, "evaluation_summary": {"support_rate": 0.2}},
            {"round": 3, "evaluation_summary": {"support_rate": 0.3}},
        ]
    }
    (work_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        state_path=work_dir / "state.json",
    )

    result = runtime.execute_tool_call(
        {
            "id": "evidence-page",
            "name": "summarize_mechanism_evidence",
            "arguments": {"limit": 1, "offset": 1, "include_failed": True},
        }
    )

    assert result.ok
    assert result.output["offset"] == 1
    assert result.output["limit"] == 1
    assert result.output["next_offset"] == 2
    assert result.output["has_more"] is True
    assert [item["round"] for item in result.output["rounds"]] == [2]


def test_local_agent_mechanism_evidence_uses_active_round_policy(tmp_path: Path) -> None:
    work_dir = tmp_path / "physics_mvp_runs" / "current"
    (work_dir / "round_008").mkdir(parents=True)
    state = {
        "status": "running",
        "current_round": 7,
        "history": [
            {"round": number, "evaluation_summary": {"support_rate": 0.1 * number}}
            for number in range(1, 8)
        ],
    }
    (work_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        state_path=work_dir / "state.json",
    )

    result = runtime.execute_tool_call(
        {
            "id": "evidence",
            "name": "summarize_mechanism_evidence",
            "arguments": {"limit": 4, "include_failed": True},
        }
    )

    assert result.ok
    assert result.output["current_round"] == 7
    assert result.output["active_round"] == 8
    assert result.output["search_policy"]["round"] == 8
    assert result.output["search_policy"]["current_mode"] == "neighbor_exploration"


def test_local_agent_brief_mechanism_evidence_marks_search_policy_scope(tmp_path: Path) -> None:
    work_dir = tmp_path / "physics_mvp_runs" / "current"
    work_dir.mkdir(parents=True)
    (work_dir / "round_008").mkdir(parents=True)
    state = {
        "status": "running",
        "current_round": 7,
        "history": [
            {"round": number, "evaluation_summary": {"support_rate": 0.1 * number}}
            for number in range(1, 8)
        ],
    }
    (work_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        state_path=work_dir / "state.json",
    )

    result = runtime.execute_tool_call(
        {
            "id": "evidence",
            "name": "summarize_mechanism_evidence",
            "arguments": {"limit": 6, "include_failed": True, "brief": True},
        }
    )

    assert result.ok
    assert result.output["brief"] is True
    assert result.output["search_policy"]["current_mode"] == "neighbor_exploration"
    assert "not a binding X/Y" in result.output["search_policy_scope"]
    assert "CONTEXT_JSON.controller_constraints.search_policy.current_search_mode" in result.output["usage_note"]
    assert "CONTEXT_JSON.controller_constraints.search_policy.current_search_mode" in result.output["xy_sequential_policy_warning"]


def test_local_agent_mechanism_evidence_reports_underexplored_clusters(tmp_path: Path) -> None:
    work_dir = tmp_path / "physics_mvp_runs" / "current"
    work_dir.mkdir(parents=True)
    pool_path = tmp_path / "data" / "pool.jsonl"
    pool_path.parent.mkdir(parents=True)
    records = [
        {"material_id": "mp-of1", "formula": "LaOF", "elements": ["La", "O", "F"], "nelements": 3},
        {"material_id": "mp-ofs1", "formula": "La6S3(OF4)2", "elements": ["La", "O", "F", "S"], "nelements": 4},
        {"material_id": "mp-ofs2", "formula": "Pr6S3(OF4)2", "elements": ["Pr", "O", "F", "S"], "nelements": 4},
        {"material_id": "mp-ofs3", "formula": "Nd3SOF5", "elements": ["Nd", "O", "F", "S"], "nelements": 4},
        {"material_id": "mp-ofs4", "formula": "Nd6S3(OF4)2", "elements": ["Nd", "O", "F", "S"], "nelements": 4},
        {"material_id": "mp-ofs5", "formula": "La4Bi4S8O3F", "elements": ["La", "Bi", "O", "F", "S"], "nelements": 5},
    ]
    pool_path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    (work_dir / "state.json").write_text(
        json.dumps(
            {
                "current_round": 10,
                "history": [
                    {
                        "round": 1,
                        "selected_material_ids": ["mp-of1"],
                        "evaluation_summary": {"support_rate": 1.0, "min_e_hull": 0.01, "stable_count": 0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        candidate_pool_path=pool_path,
        state_path=work_dir / "state.json",
    )

    result = runtime.execute_tool_call(
        {
            "id": "evidence",
            "name": "summarize_mechanism_evidence",
            "arguments": {"limit": 4, "include_failed": True},
        }
    )

    assert result.ok
    assert result.output["search_policy"]["current_mode"] == "far_exploration"
    underexplored = result.output["evidence_views"]["underexplored_clusters"]
    assert any(item["anion_motif"] == "F+O+S" for item in underexplored)
    assert underexplored[0]["recent_or_total_selected_count"] == 0


def test_local_agent_round_history_treats_missing_status_as_complete(tmp_path: Path) -> None:
    work_dir = tmp_path / "physics_mvp_runs" / "current"
    work_dir.mkdir(parents=True)
    state = {
        "history": [
            {"round": 1, "evaluation_summary": {"support_rate": 0.25}},
            {"round": 2, "status": "skipped"},
        ]
    }
    (work_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        state_path=work_dir / "state.json",
    )

    result = runtime.execute_tool_call(
        {
            "id": "history",
            "name": "summarize_round_history",
            "arguments": {"limit": 12, "status": "complete"},
        }
    )

    assert result.ok
    assert result.output["history_len"] == 1
    assert result.output["history_tail"][0]["round"] == 1


def test_local_agent_summarize_xy_generation_history_tool(tmp_path: Path) -> None:
    history_path = tmp_path / "xy_runs" / "seq" / "sequential_memory.json"
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "iteration": 1,
                        "status": "evaluated",
                        "material_description": {
                            "natural_language_description": "Rb-Cd-Br soft cage candidate",
                            "target_family": "halocadmate",
                        },
                        "candidate_spec": {"id": "seq_001", "formula_probes": [{"template": "perovskite"}]},
                        "selected_record": {"formula": "RbCdBr3"},
                        "evaluation_result": {"formula": "RbCdBr3", "e_hull": -0.002, "is_sun": True},
                        "xy_postmortem": {"next_strategy": "try adjacent alkali cavity with same Cd-Br motif"},
                    },
                    {
                        "iteration": 2,
                        "status": "evaluated",
                        "selected_record": {"formula": "NaCl"},
                        "evaluation_result": {"formula": "NaCl", "e_hull": 0.12, "is_sun": False},
                        "xy_postmortem": {"causal_interpretation": "template proxy was too generic"},
                    },
                    {
                        "iteration": 3,
                        "status": "not_materialized",
                        "candidate_spec": {
                            "formula_probe_strings": [
                                "template=perovskite;A=Rb:+1;B=Cd:+2;X=Br:-1;family=failed_rubidium_cadmium_bromide"
                            ]
                        },
                        "materialization_errors": [
                            "xy_candidate_001: formula_probes[0] generated RbCdBr3 with template perovskite but failed structure validation: volume_per_atom_too_large (volume_per_atom=38.785)",
                            "xy_candidate_001: formula_probes[0] generated Rb2ZnCdBr6 with template double_perovskite but failed structure validation: volume_per_atom_too_large (volume_per_atom=34.736)",
                            "xy_candidate_001: formula_probes[0] generated Mg(FeO2)2 but it is in the known/training formula set",
                        ],
                        "xy_postmortem": {
                            "causal_interpretation": "generator materialization failure; do not repeat Rb-Cd-Br bromide templates"
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        xy_history_path=history_path,
        max_tool_result_chars=10000,
    )

    result = runtime.execute_tool_call(
        {
            "id": "xy-history",
            "name": "summarize_xy_generation_history",
            "arguments": {"limit": 10, "include_failed": True},
        }
    )

    assert result.ok
    assert result.output["record_count"] == 3
    assert result.output["summary"]["sun_count"] == 1
    assert result.output["summary"]["best"]["formula"] == "RbCdBr3"
    assert result.output["records"][0]["material_description"]["target_family"] == "halocadmate"
    assert "Rb2ZnCdBr6" in result.output["summary"]["recent_formulas"]
    assert "Mg(FeO2)2" in result.output["summary"]["recent_formulas"]
    assert result.output["records"][2]["failed_generated_formulas"][0]["formula"] == "RbCdBr3"
    assert result.output["records"][2]["failed_generated_formulas"][2]["formula"] == "Mg(FeO2)2"
    assert "formula_probe_strings" in result.output["records"][2]["generator"]


def test_local_agent_sequential_rag_forces_mechanism_and_xy_history(tmp_path: Path) -> None:
    work_dir = tmp_path / "physics_mvp_runs" / "current"
    work_dir.mkdir(parents=True)
    (work_dir / "state.json").write_text(json.dumps({"history": []}), encoding="utf-8")
    history_path = tmp_path / "xy_runs" / "seq" / "sequential_memory.json"
    history_path.parent.mkdir(parents=True)
    history_path.write_text(json.dumps({"records": []}), encoding="utf-8")

    class NeverRequestsSequentialRagClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_text(self, *, system: str, user: str, metadata: dict) -> str:
            self.prompts.append(user)
            assert "MANDATORY_XY_SEQUENTIAL_RAG_PREFILLED_BY_CONTROLLER" in user
            assert "LOCAL_AGENT_TOOL_RESULTS" in user
            return '{"status":"material_description_proposal","material_description":{"target_family":"template-only test"}}'

    client = NeverRequestsSequentialRagClient()
    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        state_path=work_dir / "state.json",
        xy_history_path=history_path,
        max_steps=6,
    )

    text = runtime.complete_text(
        client,
        system="system",
        user="MANDATORY_XY_SEQUENTIAL_RAG:\nAgent X sequential single-material proposal.",
        metadata={"role": "agent_x_sequential"},
    )

    assert json.loads(text)["material_description"]["target_family"] == "template-only test"
    assert len(client.prompts) == 1
    trace = json.loads(next((tmp_path / "traces").glob("agent_trace_*.json")).read_text(encoding="utf-8"))
    assert trace["prefilled_mandatory_mechanism_rag"] is True
    assert trace["prefilled_mandatory_xy_generation_rag"] is True
    result_names = [result["name"] for step in trace["tool_steps"] for result in step["results"]]
    assert result_names == [
        "summarize_mechanism_evidence",
        "summarize_xy_generation_history",
    ]


def test_local_agent_enforces_mandatory_mechanism_rag(tmp_path: Path) -> None:
    work_dir = tmp_path / "physics_mvp_runs" / "current"
    work_dir.mkdir(parents=True)
    (work_dir / "state.json").write_text(json.dumps({"history": []}), encoding="utf-8")

    class InitiallySkippingClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_text(self, *, system: str, user: str, metadata: dict) -> str:
            self.prompts.append(user)
            if len(self.prompts) == 1:
                return '{"agent":"A","mechanisms":[],"agree":false,"concede":false}'
            if len(self.prompts) == 2:
                assert "MANDATORY_MECHANISM_RAG_MISSING" in user
                return json.dumps(
                    {
                        "status": "tool_request",
                        "tool_calls": [
                            {
                                "id": "history",
                                "name": "summarize_mechanism_evidence",
                                "arguments": {"limit": 12, "include_failed": True},
                            }
                        ],
                    }
                )
            assert "LOCAL_AGENT_TOOL_RESULTS" in user
            return '{"agent":"A","mechanisms":[],"agree":false,"concede":false}'

    client = InitiallySkippingClient()
    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        state_path=work_dir / "state.json",
        max_steps=3,
    )

    text = runtime.complete_text(
        client,
        system="system",
        user="MANDATORY_MECHANISM_RAG:\nAgent A: propose mechanisms.",
        metadata={"role": "mechanism_agent_a"},
    )

    assert json.loads(text)["agent"] == "A"
    assert len(client.prompts) == 3
    trace = json.loads(next((tmp_path / "traces").glob("agent_trace_*.json")).read_text(encoding="utf-8"))
    assert trace["mandatory_mechanism_rag_retry"] is True
    assert trace["tool_steps"][0]["results"][0]["name"] == "summarize_mechanism_evidence"


def test_local_agent_compacts_duplicate_mechanism_evidence_page(tmp_path: Path) -> None:
    work_dir = tmp_path / "physics_mvp_runs" / "current"
    work_dir.mkdir(parents=True)
    (work_dir / "state.json").write_text(json.dumps({"history": [{"round": 1}]}), encoding="utf-8")

    class RepeatsFirstPageClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_text(self, *, system: str, user: str, metadata: dict) -> str:
            self.prompts.append(user)
            if len(self.prompts) == 1:
                return json.dumps(
                    {
                        "status": "tool_request",
                        "tool_calls": [
                            {
                                "id": "first",
                                "name": "summarize_mechanism_evidence",
                                "arguments": {"limit": 12, "include_failed": True},
                            }
                        ],
                    }
                )
            if len(self.prompts) == 2:
                return json.dumps(
                    {
                        "status": "tool_request",
                        "tool_calls": [
                            {
                                "id": "repeat",
                                "name": "summarize_mechanism_evidence",
                                "arguments": {"limit": 12, "offset": 0, "include_failed": True},
                            }
                        ],
                    }
                )
            assert "duplicate_request" in user
            return '{"agent":"A","mechanisms":[],"agree":false,"concede":false}'

    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        state_path=work_dir / "state.json",
        max_steps=4,
    )

    text = runtime.complete_text(
        RepeatsFirstPageClient(),
        system="system",
        user="MANDATORY_MECHANISM_RAG:\nAgent A: propose mechanisms.",
        metadata={"role": "mechanism_agent_a"},
    )

    assert json.loads(text)["agent"] == "A"
    trace = json.loads(next((tmp_path / "traces").glob("agent_trace_*.json")).read_text(encoding="utf-8"))
    assert trace["tool_steps"][1]["results"][0]["output"]["duplicate_request"] is True


def test_local_agent_forces_mandatory_mechanism_rag_after_retry(tmp_path: Path) -> None:
    work_dir = tmp_path / "physics_mvp_runs" / "current"
    work_dir.mkdir(parents=True)
    (work_dir / "state.json").write_text(json.dumps({"history": []}), encoding="utf-8")

    class NeverRequestsToolClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_text(self, *, system: str, user: str, metadata: dict) -> str:
            self.prompts.append(user)
            if len(self.prompts) < 3:
                return '{"agent":"A","mechanisms":[{"id":"m001","claim":"too early"}],"agree":false,"concede":false}'
            assert "LOCAL_AGENT_TOOL_RESULTS" in user
            assert "MANDATORY_MECHANISM_RAG_FORCED_BY_CONTROLLER" in user
            return '{"agent":"A","mechanisms":[],"agree":false,"concede":false}'

    runtime = LocalAgentRuntime(
        root=tmp_path,
        trace_dir=tmp_path / "traces",
        writable_dir=tmp_path / "artifacts",
        state_path=work_dir / "state.json",
        max_steps=4,
    )

    text = runtime.complete_text(
        NeverRequestsToolClient(),
        system="system",
        user="MANDATORY_MECHANISM_RAG:\nAgent A: propose mechanisms.",
        metadata={"role": "mechanism_agent_a"},
    )

    assert json.loads(text)["agent"] == "A"
    trace = json.loads(next((tmp_path / "traces").glob("agent_trace_*.json")).read_text(encoding="utf-8"))
    assert trace["forced_mandatory_mechanism_rag"] is True
    assert trace["tool_steps"][0]["forced"] is True
    assert trace["tool_steps"][0]["results"][0]["name"] == "summarize_mechanism_evidence"


def test_tool_output_limit_preserves_oversized_first_result_identity() -> None:
    value = [
        {
            "id": "mandatory-evidence-0",
            "name": "summarize_mechanism_evidence",
            "ok": True,
            "output": {
                "state_path": "physics_mvp_runs/current/state.json",
                "history_len": 2,
                "evidence_views": {
                    "top_successes": [{"round": 2, "mechanism": "Na phosphate " * 200}],
                    "near_misses": [{"round": 1, "mechanism": "Li phosphate " * 200}],
                },
                "rounds": [{"round": 2, "accepted_mechanisms": [{"claim": "Na " * 500}]}],
            },
        }
    ]

    limited = _limit_tool_output(value, 700)

    assert isinstance(limited, list)
    assert limited[0]["id"] == "mandatory-evidence-0"
    assert limited[0]["name"] == "summarize_mechanism_evidence"
    assert limited[0]["ok"] is True
    assert limited[0]["output"]["history_len"] == 2
    assert limited[0]["output"]["_truncated"] is True
    assert limited[-1]["_truncated"] is True


def test_local_agent_parses_tool_request_before_extra_final_json(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(root=tmp_path, trace_dir=tmp_path / "traces", writable_dir=tmp_path / "artifacts")
    text = (
        '{"status":"tool_request","tool_calls":[{"id":"q","name":"list_project_files","arguments":{"max_files":1}}]}'
        '{"status":"consensus","agent":"C","predictions":[]}'
    )

    request = runtime.parse_tool_request(text)

    assert request is not None
    assert request["status"] == "tool_request"
    assert request["tool_calls"][0]["name"] == "list_project_files"


def test_local_agent_coerces_wrapped_single_tool_request(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(root=tmp_path, trace_dir=tmp_path / "traces", writable_dir=tmp_path / "artifacts")
    text = json.dumps(
        {
            "tool_request": {
                "tool": "query_candidate_pool",
                "arguments": {
                    "queries": [
                        {
                            "id": "primary",
                            "elements_all": ["O", "F", "S"],
                            "elements_any": ["La", "Nd", "Pr"],
                            "nelements_min": 4,
                            "nelements_max": 4,
                            "count": 4,
                        },
                        {
                            "id": "control",
                            "elements_all": ["O", "F"],
                            "elements_any": ["La", "Nd", "Pr"],
                            "elements_none": ["S"],
                            "count": 6,
                        },
                    ]
                },
            }
        }
    )

    request = runtime.parse_tool_request(text)

    assert request is not None
    assert request["status"] == "tool_request"
    assert [call["id"] for call in request["tool_calls"]] == ["primary", "control"]
    assert request["tool_calls"][0]["name"] == "query_candidate_pool"
    assert request["tool_calls"][0]["arguments"]["query"]["elements_all"] == ["O", "F", "S"]
    assert request["tool_calls"][1]["arguments"]["count"] == 6


def test_local_agent_coerces_type_tool_request_branches(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(root=tmp_path, trace_dir=tmp_path / "traces", writable_dir=tmp_path / "artifacts")
    text = json.dumps(
        {
            "type": "tool_request",
            "tool": "query_candidate_pool",
            "arguments": {
                "branches": {
                    "primary": {"query": {"elements_all": ["O", "F", "S"]}, "limit": 4},
                    "control": {"query": {"elements_all": ["O", "F"], "elements_none": ["S"]}, "limit": 6},
                }
            },
        }
    )

    request = runtime.parse_tool_request(text)

    assert request is not None
    assert [call["id"] for call in request["tool_calls"]] == ["primary", "control"]
    assert request["tool_calls"][0]["arguments"] == {"query": {"elements_all": ["O", "F", "S"]}, "count": 4}
    assert request["tool_calls"][1]["arguments"] == {"query": {"elements_all": ["O", "F"], "elements_none": ["S"]}, "count": 6}


def test_local_agent_uses_trailing_final_after_tool_results(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("ok\n", encoding="utf-8")
    tool_and_final = (
        '{"status":"tool_request","tool_calls":[{"id":"read","name":"read_project_file","arguments":{"path":"README.md","max_chars":20}}]}'
        '{"agent":"A","mechanisms":[],"agree":false,"concede":false}'
    )

    class RepeatingClient:
        def __init__(self) -> None:
            self.calls = 0
            self.user_prompts: list[str] = []

        def complete_text(self, *, system: str, user: str, metadata: dict) -> str:
            self.calls += 1
            self.user_prompts.append(user)
            return tool_and_final

    runtime = LocalAgentRuntime(root=tmp_path, trace_dir=tmp_path / "traces", writable_dir=tmp_path / "artifacts", max_steps=3)
    client = RepeatingClient()

    text = runtime.complete_text(client, system="system", user="user", metadata={})

    assert json.loads(text)["agent"] == "A"
    assert client.calls == 2
    assert "LOCAL_AGENT_JSON_BOUNDARY_REPAIR" in client.user_prompts[1]
    assert "ignored the trailing final JSON" in client.user_prompts[1]
    assert "Do not concatenate two top-level JSON objects" in client.user_prompts[1]
    trace = json.loads(next((tmp_path / "traces").glob("agent_trace_*.json")).read_text(encoding="utf-8"))
    assert trace["final_status"] == "trailing_final_after_tool_results"
    assert trace["ignored_trailing_final_after_tool_request"] is True
    assert len(trace["tool_steps"]) == 1


def test_prediction_prompt_requires_deterministic_matched_pair_design() -> None:
    prompt = prompt_prediction_proposal("{}")

    assert "deterministic matched-pair design" in prompt
    assert "non-identical primary_query and control_query" in prompt
    assert "mixed-anion versus single-anion split" in prompt
    assert "accepted_mechanism_ids" in prompt
    assert "planned_material_count" in prompt
    assert "at least current_inputs.target_count" in prompt
    assert "stay close to the immediately prior proposal" in prompt
    assert "distilled evidence" in prompt


def test_mechanism_prompt_requires_compact_complete_json() -> None:
    prompt = prompt_mechanism_proposal("{}")

    assert "COMPACT_OUTPUT_RULES" in prompt
    assert "at most 3 mechanisms" in prompt
    assert "claim <= 30 words" in prompt
    assert "Never truncate JSON" in prompt


def test_mechanism_client_uses_stage_token_cap(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "LLM_BASE_URL=https://example.invalid/v1",
                "LLM_API_KEY=test",
                "LLM_MODEL=test-model",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        dotenv=".env",
        mechanism_model=None,
        prediction_model=None,
        execution_model=None,
        temperature=None,
        max_tokens=None,
        mechanism_max_tokens=3072,
        prediction_max_tokens=4096,
        execution_max_tokens=4096,
    )

    client = _client("MECHANISM_A", args, tmp_path)
    assert client.config.max_tokens == 3072
    assert isinstance(client.local_agent_runtime, LocalAgentRuntime)  # type: ignore[attr-defined]
    assert client.local_agent_runtime.allow_project_writes is False  # type: ignore[attr-defined]
    assert client.local_agent_runtime.max_steps >= 12  # type: ignore[attr-defined]
    assert client.local_agent_runtime.max_tool_calls_per_step >= 8  # type: ignore[attr-defined]
    client = _client("PREDICTION_C", args, tmp_path)
    assert client.config.max_tokens == 4096
    client = _client("EXECUTION_E", args, tmp_path)
    assert client.config.max_tokens == 4096

    args.max_tokens = 1234
    client = _client("MECHANISM_A", args, tmp_path)
    assert client.config.max_tokens == 1234
    client = _client("PREDICTION_C", args, tmp_path)
    assert client.config.max_tokens == 1234
    client = _client("EXECUTION_E", args, tmp_path)
    assert client.config.max_tokens == 1234


def test_prediction_prompts_require_target_count_materialization_budget() -> None:
    context = '{"current_inputs":{"target_count":10}}'
    proposal = {"agent": "C", "predictions": [{"id": "p001", "claim": "primary lower"}]}
    critique = {"agent": "D", "required_revisions": ["under budget"]}
    counterproposal = {"agent": "C", "predictions": [{"id": "p002", "claim": "counterproposal"}]}

    prompts = [
        prompt_prediction_proposal(context),
        prompt_prediction_counterproposal(context, proposal, critique, cycle=2),
        prompt_prediction_reverse_critique(context, proposal, counterproposal, cycle=2),
        prompt_prediction_reverse_final(context, counterproposal, critique),
        prompt_prediction_critique(context, proposal, cycle=2),
        prompt_prediction_revision(context, proposal, critique, cycle=2),
        prompt_prediction_final(context, proposal, critique),
    ]

    for prompt in prompts:
        assert "planned_material_count" in prompt
        assert "current_inputs.target_count" in prompt


def test_prediction_prompts_force_history_rag_in_exploration_modes() -> None:
    context = json.dumps(
        {
            "mechanism_search_policy": {"current_mode": "far_exploration"},
            "current_inputs": {"target_count": 10},
        }
    )
    proposal = {"agent": "C", "predictions": [{"id": "p001", "claim": "primary lower"}]}
    critique = {"agent": "D", "required_revisions": ["too repetitive"]}
    counterproposal = {"agent": "C", "predictions": [{"id": "p002", "claim": "counterproposal"}]}

    prompts = [
        prompt_prediction_proposal(context),
        prompt_prediction_counterproposal(context, proposal, critique, cycle=2),
        prompt_prediction_reverse_critique(context, proposal, counterproposal, cycle=2),
        prompt_prediction_reverse_final(context, counterproposal, critique),
        prompt_prediction_critique(context, proposal, cycle=2),
        prompt_prediction_revision(context, proposal, critique, cycle=2),
        prompt_prediction_final(context, proposal, critique),
    ]

    for prompt in prompts:
        assert "MANDATORY_HISTORY_RAG" in prompt
        assert "recent_repetition" in prompt
        assert "underexplored_clusters" in prompt
        assert "rematerialize the same recent chemistry" in prompt


def test_prediction_prompts_anchor_cd_to_ab_validation_design() -> None:
    context = json.dumps(
        {
            "mechanism_search_policy": {"current_mode": "neighbor_exploration"},
            "current_inputs": {
                "target_count": 10,
                "accepted_mechanism_ids": ["m001"],
            },
        }
    )
    proposal = {"agent": "C", "predictions": [{"id": "p001", "mechanism_ids": ["m001"]}]}
    critique = {"agent": "D", "required_revisions": ["drifted from the mechanism"]}
    counterproposal = {"agent": "C", "predictions": [{"id": "p002", "mechanism_ids": ["m001"]}]}

    prompts = [
        prompt_prediction_proposal(context),
        prompt_prediction_counterproposal(context, proposal, critique, cycle=2),
        prompt_prediction_reverse_critique(context, proposal, counterproposal, cycle=2),
        prompt_prediction_reverse_final(context, counterproposal, critique),
        prompt_prediction_critique(context, proposal, cycle=2),
        prompt_prediction_revision(context, proposal, critique, cycle=2),
        prompt_prediction_final(context, proposal, critique),
    ]

    for prompt in prompts:
        assert "C/D_HYPOTHESIS_VALIDATION_POLICY" in prompt
        assert "A/B own scientific exploration and mechanism invention" in prompt
        assert "Do not introduce a new materials principle" in prompt
        assert "accepted mechanism_id" in prompt
        assert "historically successful" in prompt or "stable basin" in prompt


def test_execution_prompts_force_history_rag_in_exploration_modes() -> None:
    context = json.dumps(
        {
            "mechanism_search_policy": {"current_mode": "neighbor_exploration"},
            "current_inputs": {"target_count": 10},
        }
    )
    proposal = {"agent": "E", "bundles": [{"id": "b001", "prediction_ids": ["p001"]}]}
    critique = {"agent": "F", "required_revisions": ["too repetitive"]}
    counterproposal = {"agent": "E", "bundles": [{"id": "b002", "prediction_ids": ["p001"]}]}

    prompts = [
        prompt_execution_proposal(context),
        prompt_execution_repair_proposal(context),
        prompt_execution_counterproposal(context, proposal, critique, cycle=1),
        prompt_execution_reverse_critique(context, proposal, counterproposal, cycle=1),
        prompt_execution_reverse_final(context, counterproposal, critique),
        prompt_execution_critique(context, proposal, cycle=1),
        prompt_execution_revision(context, proposal, critique, cycle=1),
        prompt_execution_final(context, proposal, critique),
    ]

    for prompt in prompts:
        assert "MANDATORY_HISTORY_RAG" in prompt
        assert "material_id, formula" in prompt
        assert "same top low-formation-energy rows" in prompt
        assert "prediction_design_infeasible" in prompt
        assert "E/F_COMPACT_PAYLOAD_POLICY" in prompt


def test_cdef_prompts_do_not_force_history_rag_in_exploitation_mode() -> None:
    context = json.dumps(
        {
            "mechanism_search_policy": {"current_mode": "exploitation"},
            "current_inputs": {"target_count": 10},
        }
    )

    assert "C/D/E/F_EXPLORATION_ANTI_REPETITION_POLICY" in prompt_prediction_proposal(context)
    assert "MANDATORY_HISTORY_RAG" not in prompt_prediction_proposal(context)
    assert "C/D/E/F_EXPLORATION_ANTI_REPETITION_POLICY" in prompt_execution_proposal(context)
    assert "MANDATORY_HISTORY_RAG" not in prompt_execution_proposal(context)


def test_mattergen_execution_prompts_bound_sampling_and_current_json_audits() -> None:
    context = json.dumps(
        {
            "materialization_backend": "mattergen",
            "mechanism_search_policy": {"current_mode": "exploitation"},
            "current_inputs": {"target_count": 4},
        }
    )
    proposal = {"agent": "E", "bundles": [{"id": "b001", "prediction_ids": ["p001"]}]}
    critique = {"agent": "F", "required_revisions": ["audit current JSON"]}
    counterproposal = {"agent": "E", "bundles": [{"id": "b001", "prediction_ids": ["p001"]}]}

    assert f"target_count <= {mvp.DEFAULT_MATTERGEN_MAX_TARGET_COUNT}" in prompt_execution_proposal(context)
    assert f"batch_size <= {mvp.DEFAULT_MATTERGEN_MAX_BATCH_SIZE}" in prompt_execution_repair_proposal(context)
    assert f"num_batches <= {mvp.DEFAULT_MATTERGEN_MAX_NUM_BATCHES}" in prompt_execution_counterproposal(
        context, proposal, critique, cycle=1
    )
    reverse_prompt = prompt_execution_reverse_critique(context, proposal, counterproposal, cycle=1)
    critique_prompt = prompt_execution_critique(context, proposal, cycle=1)
    assert "inspect only the current AGENT_F_COUNTERPROPOSAL_JSON executable filters" in reverse_prompt
    assert "inspect only the current AGENT_E_JSON executable filters" in critique_prompt
    assert "exclude_reduced_formulas_count is invalid only if it literally appears" in reverse_prompt
    assert "exclude_reduced_formulas_count is invalid only if it literally appears" in critique_prompt


def test_mechanism_prompts_require_historical_rag() -> None:
    context = "{}"
    proposal = {"agent": "A", "mechanisms": [{"id": "m001", "claim": "broad claim"}]}
    critique = {"agent": "B", "required_revisions": ["too broad"], "agree": False}
    counterproposal = {"agent": "A", "mechanisms": [{"id": "m002", "claim": "narrow claim"}]}

    prompts = [
        prompt_mechanism_proposal(context),
        prompt_mechanism_counterproposal(context, proposal, critique, cycle=1),
        prompt_mechanism_reverse_critique(context, proposal, counterproposal, cycle=1),
        prompt_mechanism_reverse_final(context, counterproposal, critique),
        prompt_mechanism_critique(context, proposal, cycle=1),
        prompt_mechanism_revision(context, proposal, critique, cycle=1),
        prompt_mechanism_final(context, proposal, critique),
    ]

    for prompt in prompts:
        assert "MANDATORY_MECHANISM_RAG" in prompt
        assert "summarize_mechanism_evidence" in prompt
        assert "offset" in prompt
        assert "supported_predictions" in prompt
        assert "failed_predictions" in prompt
        assert "MECHANISM_SEARCH_POLICY" in prompt
        assert "underexplored_clusters" in prompt


def test_runtime_instructions_do_not_seed_legacy_la_ofs_mechanism() -> None:
    context = build_context_payload(
        state={"schema_version": "material_physics_mvp.v1", "current_round": 3, "history": []},
        pool_summary={},
        stage="mechanism",
    )
    prompt = "\n".join(context["instructions"])
    prediction_context = build_context_payload(
        state={"schema_version": "material_physics_mvp.v1", "current_round": 3, "history": []},
        pool_summary={},
        stage="prediction",
    )
    prediction_prompt = prompt_prediction_proposal("{}") + prompt_prediction_revision(
        "{}",
        {"agent": "C", "predictions": []},
        {"agent": "D", "required_revisions": []},
        cycle=1,
    )

    assert "La-O-F" not in prompt
    assert "La-O-F" not in "\n".join(prediction_context["instructions"])
    assert "La-O-F" not in prediction_prompt
    assert "F_and_S" not in prediction_prompt


def test_constructive_critic_triggers_after_repeated_rejections() -> None:
    assert critique_requires_counterproposal(
        {"agent": "D", "agree": False, "required_revisions": ["not matched"]},
        rejection_streak=0,
    )
    assert critique_requires_counterproposal(
        {"agent": "D", "agree": False, "required_revisions": ["not matched"]},
        rejection_streak=1,
        threshold=2,
    )
    assert not critique_requires_counterproposal(
        {"agent": "D", "agree": False, "required_revisions": ["not matched"]},
        rejection_streak=0,
        threshold=2,
    )
    assert not critique_requires_counterproposal(
        {"agent": "D", "agree": True, "required_revisions": []},
        rejection_streak=3,
        threshold=2,
    )


def test_constructive_critic_does_not_treat_empty_consensus_as_agreement() -> None:
    mechanism_critique = normalize_role_payload(
        {
            "status": "consensus",
            "accepted_mechanisms": [],
            "rejected_mechanisms": [{"id": "m001", "rejection_reason": "Too broad."}],
        },
        "mechanism_agent_b",
    )
    prediction_critique = normalize_role_payload(
        {
            "status": "consensus",
            "accepted_predictions": [],
            "rejected_predictions": [{"id": "p001", "rejection_reason": "Too confounded."}],
        },
        "prediction_agent_d",
    )

    assert mechanism_critique["agree"] is False
    assert prediction_critique["agree"] is False
    assert critique_requires_counterproposal(mechanism_critique, rejection_streak=0, threshold=1)
    assert critique_requires_counterproposal(prediction_critique, rejection_streak=0, threshold=1)


def test_prediction_counterproposal_prompt_makes_d_construct_a_c_shaped_prediction() -> None:
    prompt = prompt_prediction_counterproposal(
        "{}",
        {"agent": "C", "predictions": [{"id": "p001", "claim": "too broad"}]},
        {"agent": "D", "required_revisions": ["split F-only and S-only controls"]},
        cycle=3,
    )

    assert "Agent D" in prompt
    assert "must now propose" in prompt
    assert "Return only Agent C JSON" in prompt
    assert "predictions" in prompt
    assert "AGENT_D_JSON" in prompt
    assert "impossibility_certificate" in prompt
    assert "nearest faithful analogue" in prompt
    assert "distilled evidence" in prompt


def test_prediction_reverse_critique_prompt_makes_c_audit_d_counterproposal() -> None:
    prompt = prompt_prediction_reverse_critique(
        "{}",
        {"agent": "C", "predictions": [{"id": "p001", "claim": "old"}]},
        {"agent": "D", "predictions": [{"id": "p002", "claim": "counterproposal"}]},
        cycle=3,
    )

    assert "Agent C" in prompt
    assert "critique Agent D" in prompt
    assert "Return only Agent D JSON" in prompt
    assert "AGENT_D_COUNTERPROPOSAL_JSON" in prompt


def test_execution_normalizer_accepts_status_ok_bundle_payload() -> None:
    payload = normalize_role_payload(
        {
            "status": "ok",
            "accepted_bundles": [{"id": "b001", "primary": {"count": 1}, "control": {"count": 1}}],
            "consensus_summary": "ok",
        },
        "execution_agent_e",
    )

    assert valid_shape(payload, "execution_agent_e")
    assert payload["agent"] == "E"


def test_execution_prompt_forbids_custom_query_dsl() -> None:
    prompt = prompt_execution_proposal("{}")

    assert "elements_exact" in prompt
    assert "branch_predicate" in prompt
    assert "structure_dicts" in prompt


def test_validate_execution_payload_rejects_list_selection_order_without_crashing() -> None:
    payload = {
        "status": "consensus",
        "consensus_summary": "ok",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "rationale_summary": "ok",
                "selection_notes": "ok",
                "primary": {
                    "count": 5,
                    "query": {"material_ids": ["m1"]},
                    "selection_order": ["exact_element_set", "nsites_ascending"],
                },
                "control": {
                    "count": 5,
                    "query": {"material_ids": ["m2"]},
                    "selection_order": "material_id",
                },
            }
        ],
    }

    errors = validate_execution_payload(payload, target_count=10)
    assert any("selection_order" in err for err in errors)


def test_validate_execution_payload_accepts_generator_source_branch() -> None:
    payload = {
        "status": "consensus",
        "consensus_summary": "ok",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "rationale_summary": "ok",
                "selection_notes": "ok",
                "primary": {
                    "count": 1,
                    "source": "generator",
                    "formula_probes": [
                        {
                            "id": "g_primary_1",
                            "template": "rocksalt",
                            "roles": {
                                "A": {"element": "Li", "oxidation_state": 1},
                                "X": {"element": "F", "oxidation_state": -1},
                            },
                        }
                    ],
                },
                "control": {
                    "count": 1,
                    "source": "mp_pool",
                    "query": {"material_ids": ["m2"]},
                },
            }
        ],
    }

    errors = validate_execution_payload(payload, target_count=2)
    assert errors == []


def test_validate_execution_payload_defaults_formula_probes_to_generator() -> None:
    payload = {
        "status": "consensus",
        "consensus_summary": "ok",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "rationale_summary": "ok",
                "selection_notes": "ok",
                "primary": {
                    "formula_probes": [
                        {
                            "id": "primary_1",
                            "template": "perovskite",
                            "roles": {
                                "A": {"element": "Ba", "oxidation_state": 2},
                                "B": {"element": "Hf", "oxidation_state": 4},
                                "X": {"element": "O", "oxidation_state": -2},
                            },
                        }
                    ],
                },
                "control": {
                    "formula_probes": [
                        {
                            "id": "control_1",
                            "template": "perovskite",
                            "roles": {
                                "A": {"element": "Li", "oxidation_state": 1},
                                "B": {"element": "Hf", "oxidation_state": 4},
                                "X": {"element": "O", "oxidation_state": -2},
                            },
                        }
                    ],
                },
            }
        ],
    }

    errors = validate_execution_payload(payload, target_count=2)
    assert errors == []


def test_materialize_plan_defaults_formula_probes_to_generator() -> None:
    plan = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "formula_probes": [
                        {
                            "id": "primary_1",
                            "template": "perovskite",
                            "roles": {
                                "A": {"element": "Ba", "oxidation_state": 2},
                                "B": {"element": "Hf", "oxidation_state": 4},
                                "X": {"element": "O", "oxidation_state": -2},
                            },
                        }
                    ],
                },
                "control": {
                    "formula_probes": [
                        {
                            "id": "control_1",
                            "template": "perovskite",
                            "roles": {
                                "A": {"element": "Li", "oxidation_state": 1},
                                "B": {"element": "Hf", "oxidation_state": 4},
                                "X": {"element": "O", "oxidation_state": -2},
                            },
                        }
                    ],
                },
            }
        ],
    }

    selected, errors = materialize_plan([], plan, seed=1)

    assert errors == []
    assert len(selected) == 2
    assert {item["physics_role"] for item in selected} == {"primary", "control"}
    assert all(item["crystal_llm_source"] == "generator" for item in selected)


def test_materialize_plan_accepts_generator_structure_dicts() -> None:
    primary_structure_dict = Structure(
        Lattice.cubic(5.2),
        ["La", "La", "O", "F", "F", "S"],
        [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
            [0.25, 0.25, 0.25],
        ],
    ).as_dict()
    control_structure_dict = Structure(
        Lattice.cubic(4.8),
        ["La", "O", "F"],
        [
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
        ],
    ).as_dict()
    plan = {
        "status": "consensus",
        "consensus_summary": "ok",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "rationale_summary": "ok",
                "selection_notes": "ok",
                "primary": {
                    "count": 1,
                    "source": "generator",
                    "structure_dicts": [primary_structure_dict],
                },
                "control": {
                    "count": 1,
                    "source": "generator",
                    "structure_dicts": [control_structure_dict],
                },
            }
        ],
    }

    errors = validate_execution_payload(plan, target_count=2)
    assert errors == []

    selected, materialization_errors = materialize_plan([], plan, seed=1)

    assert materialization_errors == []
    assert len(selected) == 2
    assert {item["physics_role"] for item in selected} == {"primary", "control"}
    assert all(item["crystal_llm_source"] == "generator" for item in selected)
    assert {item["structure_dict"]["sites"][0]["species"][0]["element"] for item in selected} == {"La"}


def test_materialize_plan_rescales_generator_structure_dicts_to_query_window() -> None:
    large_structure = Structure(
        Lattice.cubic(12.0),
        ["La", "O", "F"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    ).as_dict()
    branch = {
        "count": 1,
        "source": "generator",
        "query": {
            "density_min": 3.5,
            "density_max": 9.5,
            "volume_per_atom_min": 8.0,
            "volume_per_atom_max": 28.0,
        },
        "structure_dicts": [large_structure],
    }
    plan = {
        "status": "consensus",
        "consensus_summary": "ok",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "rationale_summary": "ok",
                "selection_notes": "ok",
                "primary": branch,
                "control": branch,
            }
        ],
    }

    selected, materialization_errors = materialize_plan([], plan, seed=1)

    assert materialization_errors == []
    assert len(selected) == 2
    for item in selected:
        structure = Structure.from_dict(item["structure_dict"])
        assert 8.0 <= structure.volume / len(structure) <= 28.0
        assert 3.5 <= float(structure.density) <= 9.5
        assert item["crystal_llm_generator_preflight"]["notes"]


def test_materialize_plan_rejects_formally_unbalanced_generator_structure_dict() -> None:
    bad_structure = Structure(
        Lattice.cubic(5.0),
        ["La", "O", "F", "S"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5]],
    ).as_dict()
    good_structure = Structure(
        Lattice.cubic(4.8),
        ["La", "O", "F"],
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]],
    ).as_dict()
    plan = {
        "status": "consensus",
        "consensus_summary": "ok",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "rationale_summary": "ok",
                "selection_notes": "ok",
                "primary": {"count": 1, "source": "generator", "structure_dicts": [bad_structure]},
                "control": {"count": 1, "source": "generator", "structure_dicts": [good_structure]},
            }
        ],
    }

    selected, materialization_errors = materialize_plan([], plan, seed=1)

    assert len(selected) == 1
    assert any("formal charge imbalance" in error for error in materialization_errors)


def test_validate_execution_payload_rejects_multi_prediction_bundle() -> None:
    payload = {
        "status": "consensus",
        "consensus_summary": "ok",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1", "p2"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "rationale_summary": "ok",
                "selection_notes": "ok",
                "primary": {"count": 5, "source": "mp_pool", "query": {"material_ids": ["m1"]}},
                "control": {"count": 5, "source": "mp_pool", "query": {"material_ids": ["m2"]}},
            }
        ],
    }

    errors = validate_execution_payload(payload, target_count=10)
    assert any("one prediction" in err or "prediction_ids" in err for err in errors)


def test_run_analysis_creates_summary_when_missing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    round_dir = tmp_path / "round_001"
    round_dir.mkdir()
    input_path = round_dir / "input.json"
    results_path = round_dir / "results.json"
    eval_log = round_dir / "eval" / "full_cpu_0.out"
    eval_log.parent.mkdir(parents=True)

    structure = Structure(Lattice.cubic(3.0), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    input_path.write_text(
        json.dumps([structure.as_dict()], ensure_ascii=False),
        encoding="utf-8",
    )
    results_path.write_text(json.dumps({"success_rate": {"success_rate": 1.0}}, ensure_ascii=False), encoding="utf-8")
    eval_log.write_text("E-hull distance: 0.123\n", encoding="utf-8")

    run_analysis(
        root=root,
        round_dir=round_dir,
        input_path=input_path,
        results_path=results_path,
        eval_log=eval_log,
    )

    assert (round_dir / "analysis" / "summary.json").exists()
    assert (round_dir / "analysis" / "e_hull_ranked.csv").exists()


def test_run_analysis_records_missing_e_hull_without_crashing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    round_dir = tmp_path / "round_001"
    round_dir.mkdir()
    input_path = round_dir / "input.json"
    results_path = round_dir / "results.json"
    eval_log = round_dir / "eval" / "full_cpu_0.out"
    eval_log.parent.mkdir(parents=True)

    structure = Structure(Lattice.cubic(6.0), ["Ra", "F", "F"], [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    input_path.write_text(json.dumps([structure.as_dict()], ensure_ascii=False), encoding="utf-8")
    results_path.write_text(
        json.dumps({"success_rate": {"success_rate": 0.0}, "novelty": {"sun_score": 0.0}}, ensure_ascii=False),
        encoding="utf-8",
    )
    eval_log.write_text(
        "E-hull calculation error: Unable to get decomposition for None ComputedStructureEntry - RaF2\n",
        encoding="utf-8",
    )

    run_analysis(
        root=root,
        round_dir=round_dir,
        input_path=input_path,
        results_path=results_path,
        eval_log=eval_log,
    )

    summary = json.loads((round_dir / "analysis" / "summary.json").read_text(encoding="utf-8"))
    ranked = (round_dir / "analysis" / "e_hull_ranked.csv").read_text(encoding="utf-8")
    assert summary["count"] == 0
    assert summary["input_structure_count"] == 1
    assert summary["missing_e_hull_count"] == 1
    assert summary["missing_e_hull_rows"][0]["formula"] == "RaF2"
    assert ranked.strip() == "index,formula,e_hull,nsites,template_guess,volume_per_atom"


def test_validate_query_rejects_list_preferred_order_without_crashing() -> None:
    from crystal_llm.material_physics_schema import validate_query

    errors = validate_query({"material_ids": ["m1"], "preferred_order": ["material_id", "random"]})
    assert any("preferred_order" in err for err in errors)


def test_select_matches_honors_ordered_preferred_order_spec() -> None:
    records = [
        {
            "material_id": "m1",
            "formula": "A1",
            "formation_energy_per_atom": -0.20,
            "band_gap": 1.5,
            "density": 5.0,
            "volume_per_atom": 10.0,
        },
        {
            "material_id": "m2",
            "formula": "A2",
            "formation_energy_per_atom": -0.30,
            "band_gap": 0.5,
            "density": 4.0,
            "volume_per_atom": 9.0,
        },
        {
            "material_id": "m3",
            "formula": "A3",
            "formation_energy_per_atom": -0.30,
            "band_gap": 2.5,
            "density": 6.0,
            "volume_per_atom": 8.0,
        },
    ]
    query = {
        "preferred_order": [
            "formation_energy_per_atom asc",
            "band_gap desc",
            "density asc",
            "volume_per_atom asc",
            "formula asc",
        ]
    }

    assert validate_query(query) == []
    selected = select_matches(records, query, count=3, seed=0)

    assert [item["material_id"] for item in selected] == ["m3", "m2", "m1"]


def test_materialize_plan_supports_generator_branch() -> None:
    plan = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "count": 1,
                    "source": "generator",
                    "formula_probes": [
                        {
                            "id": "g_primary_1",
                            "template": "rocksalt",
                            "roles": {
                                "A": {"element": "Li", "oxidation_state": 1},
                                "X": {"element": "F", "oxidation_state": -1},
                            },
                        }
                    ],
                },
                "control": {
                    "count": 1,
                    "source": "generator",
                    "formula_probes": [
                        {
                            "id": "g_control_1",
                            "template": "cesium_chloride",
                            "roles": {
                                "A": {"element": "Na", "oxidation_state": 1},
                                "X": {"element": "Cl", "oxidation_state": -1},
                            },
                        }
                    ],
                },
            }
        ],
    }

    selected, errors = materialize_plan([], plan, seed=1)

    assert errors == []
    assert len(selected) == 2
    assert selected[0]["crystal_llm_source"] == "generator"
    assert selected[0]["structure_dict"]["@class"] == "Structure"


def test_execution_payload_accepts_mattergen_branch() -> None:
    payload = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "rationale_summary": "Compare matched MatterGen chemical systems.",
                "selection_notes": "Take accepted generated structures after adapter filtering.",
                "primary": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Li-F",
                                "energy_above_hull": 0.0,
                            },
                            "diffusion_guidance_factor": 1.0,
                            "filters": {"chemical_system": ["Li", "F"], "max_sites": 20},
                        }
                    ],
                },
                "control": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Na-Cl",
                                "energy_above_hull": 0.0,
                            },
                            "diffusion_guidance_factor": 1.0,
                            "filters": {"chemical_system": ["Na", "Cl"], "max_sites": 20},
                        }
                    ],
                },
            }
        ],
        "consensus_summary": "MatterGen branches are executable.",
    }

    assert validate_execution_payload(payload, target_count=2) == []


def test_execution_normalizer_flattens_mattergen_sampling_alias() -> None:
    payload = normalize_role_payload(
        {
            "agent": "E",
            "status": "consensus",
            "bundles": [
                {
                    "id": "b1",
                    "prediction_ids": ["p1"],
                    "expected_relation": "primary_lower_e_hull_than_control",
                    "primary": {
                        "source": "mattergen",
                        "count": 1,
                        "mattergen_requests": [
                            {
                                "backend": "mattergen",
                                "properties_to_condition_on": {"chemical_system": "Li-F", "energy_above_hull": 0.0},
                                "filters": {"chemical_system": ["Li", "F"], "max_sites": 20},
                                "sampling": {"target_count": 3, "batch_size": 8, "num_batches": 6},
                            }
                        ],
                    },
                    "control": {
                        "source": "mattergen",
                        "count": 1,
                        "mattergen_requests": [
                            {
                                "backend": "mattergen",
                                "properties_to_condition_on": {"chemical_system": "Na-Cl", "energy_above_hull": 0.0},
                                "filters": {"chemical_system": ["Na", "Cl"], "max_sites": 20},
                                "sampling": {"batch_size": 8, "num_batches": 6},
                            }
                        ],
                    },
                }
            ],
        },
        "execution_agent_e",
    )

    assert valid_shape(payload, "execution_agent_e")
    primary_request = payload["bundles"][0]["primary"]["mattergen_requests"][0]
    control_request = payload["bundles"][0]["control"]["mattergen_requests"][0]
    assert primary_request["target_count"] == 3
    assert primary_request["batch_size"] == 8
    assert primary_request["num_batches"] == 6
    assert "sampling" not in primary_request
    assert control_request["batch_size"] == 8
    assert control_request["num_batches"] == 6


def test_compact_mattergen_request_reports_sampling_alias() -> None:
    compact = mvp.compact_mattergen_request_for_feedback(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Li-F", "energy_above_hull": 0.0},
            "filters": {"chemical_system": ["Li", "F"], "max_sites": 20},
            "sampling": {"target_count": 3, "batch_size": 8, "num_batches": 6},
        }
    )

    assert compact["target_count"] == 3
    assert compact["batch_size"] == 8
    assert compact["num_batches"] == 6


def test_compact_mattergen_request_keeps_exclude_list_executable() -> None:
    compact = mvp.compact_mattergen_request_for_feedback(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Cu-S", "energy_above_hull": 0.0},
            "filters": {
                "chemical_system": ["Cu", "S"],
                "max_sites": 20,
                "exclude_reduced_formulas": ["Cu2S", "CuS", "Cu7S4", "Cu9S5", "CuS2", "Cu31S16", "Cu39S28"],
            },
        }
    )

    filters = compact["filters"]
    assert filters["exclude_reduced_formulas"] == ["Cu2S", "CuS", "Cu7S4", "Cu9S5", "CuS2", "Cu31S16"]
    assert "exclude_reduced_formulas_count" not in filters


def test_mattergen_native_source_policy_rejects_generator_fallback() -> None:
    payload = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "rationale_summary": "Fallback should be rejected in MatterGen-native execution.",
                "selection_notes": "Bad generator fallback.",
                "primary": {
                    "source": "generator",
                    "count": 1,
                    "formula_probes": ["LiFePO4"],
                },
                "control": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Fe-P-O",
                                "energy_above_hull": 0.0,
                            },
                            "filters": {"chemical_system": ["Fe", "P", "O"], "max_sites": 20},
                        }
                    ],
                },
            }
        ],
        "consensus_summary": "Mixed fallback is not allowed in MatterGen-native mode.",
    }

    errors = mvp.validate_mattergen_native_execution_sources(payload)

    assert any("source='mattergen'" in error and "generator" in error for error in errors)
    assert any("non-MatterGen materialization payloads" in error for error in errors)


def test_mattergen_native_source_policy_rejects_unsupported_filter_gate() -> None:
    payload = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "rationale_summary": "Invalid filter gate should be rejected before MatterGen.",
                "selection_notes": "The controller does not implement stoichiometry gates.",
                "primary": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Li-Mn-P-O",
                                "energy_above_hull": 0.0,
                            },
                            "filters": {
                                "chemical_system": ["Li", "Mn", "P", "O"],
                                "require_chemical_system_exact": True,
                                "max_sites": 20,
                                "stoichiometry_gate": {"P": 1, "O": 4},
                            },
                        }
                    ],
                },
                "control": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Mn-P-O",
                                "energy_above_hull": 0.0,
                            },
                            "filters": {
                                "chemical_system": ["Mn", "P", "O"],
                                "require_chemical_system_exact": True,
                                "max_sites": 20,
                            },
                        }
                    ],
                },
            }
        ],
        "consensus_summary": "Unsupported filter gate must not be treated as executable.",
    }

    errors = mvp.validate_mattergen_native_execution_sources(payload)

    assert any("unsupported MatterGen filter keys" in error for error in errors)
    assert any("stoichiometry_gate" in error for error in errors)
    assert any("prediction_design_infeasible" in error for error in errors)


def test_mattergen_underfill_feedback_reports_reject_reasons(monkeypatch, tmp_path) -> None:
    primary_structure = Structure(Lattice.cubic(4.2), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    control_structure = Structure(Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    by_branch = {"b1_primary": primary_structure, "b1_control": control_structure}

    def fake_run_mattergen_request(request, *, branch_id, mattergen_config):
        structure = by_branch[branch_id]
        return (
            [structure.as_dict()],
            [{"material_id": f"adapter::{branch_id}", "formula": structure.composition.reduced_formula}],
            {
                "status": "success",
                "target_count": request.get("target_count"),
                "accepted_count": 1,
                "accepted_formulas": [structure.composition.reduced_formula],
                "reject_reasons": {"too_few_sites": 7},
                "reject_examples": [{"index": 1, "formula": "LiF", "reason": "too_few_sites"}],
            },
        )

    monkeypatch.setattr(mvp, "_run_mattergen_request", fake_run_mattergen_request)
    args = parse_args(
        [
            "--materialization-backend",
            "mattergen",
            "--mattergen-runner",
            "local",
            "--mattergen-root",
            str(tmp_path),
        ]
    )
    config = mattergen_config_from_args(args, Path.cwd(), tmp_path / "mattergen")
    plan = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "rationale_summary": "MatterGen underfill feedback should be specific.",
                "selection_notes": "Use exact systems.",
                "primary": {
                    "source": "mattergen",
                    "count": 2,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Li-F",
                                "energy_above_hull": 0.0,
                            },
                            "filters": {"chemical_system": ["Li", "F"], "min_sites": 11, "max_sites": 16},
                        }
                    ],
                },
                "control": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Na-Cl",
                                "energy_above_hull": 0.0,
                            },
                            "filters": {"chemical_system": ["Na", "Cl"], "min_sites": 11, "max_sites": 16},
                        }
                    ],
                },
            }
        ],
        "consensus_summary": "Underfill is reported.",
    }

    _, errors = materialize_plan([], plan, seed=1, mattergen_config=config)
    feedback = summarize_materialization_feedback(plan=plan, materialization_errors=errors, proposal=plan, critique={"agent": "F"})

    assert any("too_few_sites" in error and "lower/remove filters.min_sites" in error for error in errors)
    assert feedback["failed_branches"][0]["failed_mattergen_request"]["filters"]["min_sites"] == 11
    assert "Keep source='mattergen'" in feedback["failed_branches"][0]["required_action"]
    assert "underfilled MatterGen filters" in feedback["required_repair"]


def test_materialize_plan_supports_mattergen_branch(monkeypatch, tmp_path) -> None:
    primary_structure = Structure(Lattice.cubic(4.2), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    control_structure = Structure(Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    by_branch = {"b1_primary": primary_structure, "b1_control": control_structure}
    captured_requests = {}

    def fake_run_mattergen_request(request, *, branch_id, mattergen_config):
        captured_requests[branch_id] = request
        structure = by_branch[branch_id]
        formula = structure.composition.reduced_formula
        return (
            [structure.as_dict()],
            [{"material_id": f"adapter::{branch_id}", "formula": formula}],
            {"status": "success", "accepted_count": 1, "request_id": request["request_id"]},
        )

    monkeypatch.setattr(mvp, "_run_mattergen_request", fake_run_mattergen_request)
    args = parse_args(
        [
            "--materialization-backend",
            "mattergen",
            "--mattergen-runner",
            "local",
            "--mattergen-root",
            str(tmp_path),
        ]
    )
    config = mattergen_config_from_args(args, Path.cwd(), tmp_path / "mattergen")
    plan = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Li-F",
                                "energy_above_hull": 0.0,
                            },
                            "filters": {"chemical_system": ["Li", "F"], "max_sites": 20},
                        }
                    ],
                },
                "control": {
                    "source": "mattergen",
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
                },
            }
        ],
    }

    selected, errors = materialize_plan(
        [],
        plan,
        seed=1,
        mattergen_config=config,
        known_formulas={"Fe2O3", "Li2F", "LiF2", "Na2Cl", "NaCl2", "RbBr"},
    )

    assert errors == []
    assert [item["physics_role"] for item in selected] == ["primary", "control"]
    assert all(item["crystal_llm_source"] == "mattergen" for item in selected)
    assert all(item["crystal_llm_generator_backend"] == "mattergen" for item in selected)
    assert selected[0]["crystal_llm_generated_from_mattergen_requests"][0]["diffusion_guidance_factor"] == 1.0
    assert captured_requests["b1_primary"]["filters"]["require_chemical_system_exact"] is True
    assert captured_requests["b1_control"]["filters"]["require_chemical_system_exact"] is True
    assert captured_requests["b1_primary"]["filters"]["exclude_reduced_formulas"] == ["Li2F", "LiF2"]
    assert captured_requests["b1_control"]["filters"]["exclude_reduced_formulas"] == ["Na2Cl", "NaCl2"]


def test_mattergen_hard_target_increases_sampling_budget(monkeypatch, tmp_path) -> None:
    primary_structure = Structure(Lattice.cubic(4.2), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    control_structure = Structure(Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    by_branch = {"b1_primary": primary_structure, "b1_control": control_structure}
    captured_requests = {}

    def fake_run_mattergen_request(request, *, branch_id, mattergen_config):
        captured_requests[branch_id] = request
        structure = by_branch[branch_id]
        formula = structure.composition.reduced_formula
        return (
            [structure.as_dict()],
            [{"material_id": f"adapter::{branch_id}", "formula": formula}],
            {"status": "success", "accepted_count": 1, "request_id": request["request_id"]},
        )

    monkeypatch.setattr(mvp, "_run_mattergen_request", fake_run_mattergen_request)
    args = parse_args(
        [
            "--materialization-backend",
            "mattergen",
            "--mattergen-runner",
            "local",
            "--mattergen-root",
            str(tmp_path),
            "--mattergen-num-batches",
            "2",
        ]
    )
    config = mattergen_config_from_args(args, Path.cwd(), tmp_path / "mattergen")
    plan = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Li-F",
                                "energy_above_hull": 0.0,
                            },
                            "filters": {
                                "chemical_system": ["Li", "F"],
                                "target_reduced_formula": "LiF",
                                "require_target_reduced_formula": True,
                                "max_sites": 20,
                            },
                        }
                    ],
                },
                "control": {
                    "source": "mattergen",
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
                },
            }
        ],
    }

    selected, errors = materialize_plan([], plan, seed=1, mattergen_config=config)

    assert errors == []
    assert len(selected) == 2
    assert captured_requests["b1_primary"]["filters"]["require_target_reduced_formula"] is True
    assert captured_requests["b1_primary"]["num_batches"] == mvp.DEFAULT_MATTERGEN_HARD_TARGET_MIN_NUM_BATCHES
    assert captured_requests["b1_control"]["num_batches"] == 2


def test_mattergen_hard_target_sampling_budget_is_capped(monkeypatch, tmp_path) -> None:
    primary_structure = Structure(Lattice.cubic(4.2), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    control_structure = Structure(Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    by_branch = {"b1_primary": primary_structure, "b1_control": control_structure}
    captured_requests = {}

    def fake_run_mattergen_request(request, *, branch_id, mattergen_config):
        captured_requests[branch_id] = request
        structure = by_branch[branch_id]
        return (
            [structure.as_dict()],
            [{"material_id": f"adapter::{branch_id}", "formula": structure.composition.reduced_formula}],
            {"status": "success", "accepted_count": 1, "request_id": request["request_id"]},
        )

    monkeypatch.setattr(mvp, "_run_mattergen_request", fake_run_mattergen_request)
    args = parse_args(
        [
            "--materialization-backend",
            "mattergen",
            "--mattergen-runner",
            "local",
            "--mattergen-root",
            str(tmp_path),
        ]
    )
    config = mattergen_config_from_args(args, Path.cwd(), tmp_path / "mattergen")
    plan = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "target_count": 16,
                            "batch_size": 64,
                            "num_batches": 128,
                            "properties_to_condition_on": {
                                "chemical_system": "Li-F",
                                "energy_above_hull": 0.0,
                            },
                            "filters": {
                                "chemical_system": ["Li", "F"],
                                "target_reduced_formula": "LiF",
                                "require_target_reduced_formula": True,
                                "max_sites": 20,
                            },
                        }
                    ],
                },
                "control": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {"chemical_system": "Na-Cl", "energy_above_hull": 0.0},
                            "filters": {"chemical_system": ["Na", "Cl"], "max_sites": 20},
                        }
                    ],
                },
            }
        ],
    }

    selected, errors = materialize_plan([], plan, seed=1, mattergen_config=config)

    assert errors == []
    assert len(selected) == 2
    request = captured_requests["b1_primary"]
    assert request["target_count"] == mvp.DEFAULT_MATTERGEN_MAX_TARGET_COUNT
    assert request["batch_size"] == mvp.DEFAULT_MATTERGEN_MAX_BATCH_SIZE
    assert request["num_batches"] == mvp.DEFAULT_MATTERGEN_MAX_NUM_BATCHES
    assert request["batch_size"] * request["num_batches"] <= mvp.DEFAULT_MATTERGEN_MAX_RAW_SAMPLES
    assert any("capped target_count" in note for note in request["controller_normalization_notes"])
    assert any("capped batch_size" in note for note in request["controller_normalization_notes"])
    assert any("capped num_batches" in note for note in request["controller_normalization_notes"])


def test_mattergen_nested_sampling_alias_controls_request(monkeypatch, tmp_path) -> None:
    primary_structure = Structure(Lattice.cubic(4.2), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    control_structure = Structure(Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    by_branch = {"b1_primary": primary_structure, "b1_control": control_structure}
    captured_requests = {}

    def fake_run_mattergen_request(request, *, branch_id, mattergen_config):
        captured_requests[branch_id] = request
        structure = by_branch[branch_id]
        formula = structure.composition.reduced_formula
        return (
            [structure.as_dict()],
            [{"material_id": f"adapter::{branch_id}", "formula": formula}],
            {"status": "success", "accepted_count": 1, "request_id": request["request_id"]},
        )

    monkeypatch.setattr(mvp, "_run_mattergen_request", fake_run_mattergen_request)
    args = parse_args(
        [
            "--materialization-backend",
            "mattergen",
            "--mattergen-runner",
            "local",
            "--mattergen-root",
            str(tmp_path),
            "--mattergen-num-batches",
            "2",
        ]
    )
    config = mattergen_config_from_args(args, Path.cwd(), tmp_path / "mattergen")
    plan = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {"chemical_system": "Li-F", "energy_above_hull": 0.0},
                            "filters": {"chemical_system": ["Li", "F"], "max_sites": 20},
                            "sampling": {"target_count": 5, "batch_size": 8, "num_batches": 7},
                        }
                    ],
                },
                "control": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {"chemical_system": "Na-Cl", "energy_above_hull": 0.0},
                            "filters": {"chemical_system": ["Na", "Cl"], "max_sites": 20},
                            "sampling": {"batch_size": 8, "num_batches": 6},
                        }
                    ],
                },
            }
        ],
    }

    selected, errors = materialize_plan([], plan, seed=1, mattergen_config=config)

    assert errors == []
    assert len(selected) == 2
    assert captured_requests["b1_primary"]["target_count"] == 5
    assert captured_requests["b1_primary"]["batch_size"] == 8
    assert captured_requests["b1_primary"]["num_batches"] == 7
    assert captured_requests["b1_control"]["batch_size"] == 8
    assert captured_requests["b1_control"]["num_batches"] == 6
    assert "sampling" not in captured_requests["b1_primary"]


def test_materialize_plan_reuses_locked_successful_mattergen_branch(monkeypatch, tmp_path) -> None:
    primary_structure = Structure(Lattice.cubic(4.2), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    control_structures = [
        Structure(Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        Structure(Lattice.cubic(4.4), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    ]
    calls: list[str] = []

    def fake_run_mattergen_request(request, *, branch_id, mattergen_config):
        calls.append(branch_id)
        if branch_id == "b1_primary":
            structures = [primary_structure]
        elif int(request["filters"].get("min_sites") or 1) >= 9:
            structures = control_structures[:1]
        else:
            structures = control_structures
        return (
            [structure.as_dict() for structure in structures],
            [{"material_id": f"adapter::{branch_id}::{index}", "formula": structure.composition.reduced_formula} for index, structure in enumerate(structures)],
            {
                "status": "success",
                "target_count": request.get("target_count"),
                "accepted_count": len(structures),
                "accepted_formulas": [structure.composition.reduced_formula for structure in structures],
                "reject_reasons": {"too_few_sites": 3} if len(structures) == 1 else {},
            },
        )

    monkeypatch.setattr(mvp, "_run_mattergen_request", fake_run_mattergen_request)
    args = parse_args(
        [
            "--materialization-backend",
            "mattergen",
            "--mattergen-runner",
            "local",
            "--mattergen-root",
            str(tmp_path),
        ]
    )
    config = mattergen_config_from_args(args, Path.cwd(), tmp_path / "mattergen")
    plan = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Li-F",
                                "energy_above_hull": 0.0,
                            },
                            "filters": {"chemical_system": ["Li", "F"], "max_sites": 20},
                        }
                    ],
                },
                "control": {
                    "source": "mattergen",
                    "count": 2,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Na-Cl",
                                "energy_above_hull": 0.0,
                            },
                            "filters": {"chemical_system": ["Na", "Cl"], "min_sites": 9, "max_sites": 20},
                        }
                    ],
                },
            }
        ],
    }
    branch_cache_records: dict[str, list[dict[str, object]]] = {}
    branch_cache_fingerprints: dict[str, str] = {}

    selected, errors = materialize_plan(
        [],
        plan,
        seed=1,
        mattergen_config=config,
        branch_cache_records=branch_cache_records,
        branch_cache_fingerprints=branch_cache_fingerprints,
    )

    assert [item["physics_role"] for item in selected] == ["primary"]
    assert any("b1.control requested 2 records but only 1 materialized" in error for error in errors)
    assert calls == ["b1_primary", "b1_control"]
    assert "b1.primary" in branch_cache_fingerprints

    repaired_plan = json.loads(json.dumps(plan))
    repaired_plan["accepted_bundles"][0]["control"]["mattergen_requests"][0]["filters"]["min_sites"] = 1
    calls.clear()

    selected, errors = materialize_plan(
        [],
        repaired_plan,
        seed=1,
        mattergen_config=config,
        branch_cache_records=branch_cache_records,
        branch_cache_fingerprints=branch_cache_fingerprints,
    )

    assert errors == []
    assert calls == ["b1_control"]
    assert [item["physics_role"] for item in selected] == ["primary", "control", "control"]
    assert selected[0]["crystal_llm_materialization_cache_reused"] is True

    mutated_successful_branch_plan = json.loads(json.dumps(repaired_plan))
    mutated_successful_branch_plan["accepted_bundles"][0]["primary"]["mattergen_requests"][0]["filters"]["max_sites"] = 24
    calls.clear()

    _selected, errors = materialize_plan(
        [],
        mutated_successful_branch_plan,
        seed=1,
        mattergen_config=config,
        branch_cache_records=branch_cache_records,
        branch_cache_fingerprints=branch_cache_fingerprints,
    )

    assert calls == []
    assert any("already materialized successfully" in error and "Preserve successful branches exactly" in error for error in errors)


def test_materialize_plan_applies_mattergen_phosphate_motif_constraint(monkeypatch, tmp_path) -> None:
    lattice = Lattice.cubic(6.0)
    invalid_primary = Structure(
        lattice,
        ["Li", "Li", "Mn", "P", "O"],
        [
            [0, 0, 0],
            [0, 0.5, 0],
            [0.5, 0, 0],
            [0.5, 0.5, 0.5],
            [0.741667, 0.5, 0.5],
        ],
    )
    phosphate_primary = Structure(
        lattice,
        ["Li", "Mn", "P", "O", "O", "O", "O"],
        [
            [0, 0, 0],
            [0.5, 0, 0],
            [0.5, 0.5, 0.5],
            [0.741667, 0.5, 0.5],
            [0.258333, 0.5, 0.5],
            [0.5, 0.741667, 0.5],
            [0.5, 0.5, 0.741667],
        ],
    )
    control_structure = Structure(Lattice.cubic(4.2), ["Li", "Mn", "O"], [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]])
    by_branch = {"b1_primary": [invalid_primary, phosphate_primary], "b1_control": [control_structure]}

    def fake_run_mattergen_request(request, *, branch_id, mattergen_config):
        structures = by_branch[branch_id]
        return (
            [structure.as_dict() for structure in structures],
            [
                {"material_id": f"adapter::{branch_id}::{index}", "formula": structure.composition.reduced_formula}
                for index, structure in enumerate(structures)
            ],
            {"status": "success", "accepted_count": len(structures), "request_id": request["request_id"]},
        )

    monkeypatch.setattr(mvp, "_run_mattergen_request", fake_run_mattergen_request)
    args = parse_args(
        [
            "--materialization-backend",
            "mattergen",
            "--mattergen-runner",
            "local",
            "--mattergen-root",
            str(tmp_path),
        ]
    )
    config = mattergen_config_from_args(args, Path.cwd(), tmp_path / "mattergen")
    plan = {
        "status": "consensus",
        "materialization_constraints": {
            "primary_acceptance_constraint": "Li-Mn-P-O primary structures must contain phosphate/pyrophosphate-like P-O coordination.",
        },
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Li-Mn-P-O",
                                "energy_above_hull": 0.0,
                            },
                            "filters": {"chemical_system": ["Li", "Mn", "P", "O"], "max_sites": 20},
                        }
                    ],
                },
                "control": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Li-Mn-O",
                                "energy_above_hull": 0.0,
                            },
                            "filters": {"chemical_system": ["Li", "Mn", "O"], "max_sites": 20},
                        }
                    ],
                },
            }
        ],
    }

    selected, errors = materialize_plan([], plan, seed=1, mattergen_config=config)

    assert errors == []
    primary = next(item for item in selected if item["physics_role"] == "primary")
    assert primary["formula"] == "LiMnPO4"
    motif_rejects = primary["crystal_llm_mattergen_report"]["controller_motif_rejects"]
    assert motif_rejects[0]["formula"] == "Li2MnPO"
    assert motif_rejects[0]["reason"].startswith("phosphate_like_p_o_low_o_to_p_ratio")


def test_prediction_mattergen_drift_constraints_are_applied_to_execution_plan(monkeypatch, tmp_path) -> None:
    lattice = Lattice.cubic(6.0)
    invalid_primary = Structure(
        lattice,
        ["Li", "Fe", "Fe", "P", "O"],
        [
            [0, 0, 0],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0.5, 0.5, 0.5],
            [0.741667, 0.5, 0.5],
        ],
    )
    phosphate_primary = Structure(
        lattice,
        ["Li", "Fe", "P", "O", "O", "O", "O"],
        [
            [0, 0, 0],
            [0.5, 0, 0],
            [0.5, 0.5, 0.5],
            [0.741667, 0.5, 0.5],
            [0.258333, 0.5, 0.5],
            [0.5, 0.741667, 0.5],
            [0.5, 0.5, 0.741667],
        ],
    )
    control_structure = Structure(
        lattice,
        ["Fe", "P", "O", "O", "O"],
        [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0.741667, 0.5, 0.5],
            [0.258333, 0.5, 0.5],
            [0.5, 0.741667, 0.5],
        ],
    )
    by_branch = {"b1_primary": [invalid_primary, phosphate_primary], "b1_control": [control_structure]}

    def fake_run_mattergen_request(request, *, branch_id, mattergen_config):
        structures = by_branch[branch_id]
        return (
            [structure.as_dict() for structure in structures],
            [
                {"material_id": f"adapter::{branch_id}::{index}", "formula": structure.composition.reduced_formula}
                for index, structure in enumerate(structures)
            ],
            {"status": "success", "accepted_count": len(structures), "request_id": request["request_id"]},
        )

    monkeypatch.setattr(mvp, "_run_mattergen_request", fake_run_mattergen_request)
    args = parse_args(
        [
            "--materialization-backend",
            "mattergen",
            "--mattergen-runner",
            "local",
            "--mattergen-root",
            str(tmp_path),
        ]
    )
    config = mattergen_config_from_args(args, Path.cwd(), tmp_path / "mattergen")
    prediction_consensus = {
        "accepted_predictions": [
            {
                "id": "p1",
                "comparison_design": {
                    "primary_mattergen": {
                        "chemical_system": "Li-Fe-P-O",
                        "drift_rejection_criteria": ["Reject structures lacking short P-O polyhedra."],
                    },
                    "control_mattergen": {
                        "chemical_system": "Fe-P-O",
                        "drift_rejection_criteria": ["Reject non-phosphate oxides."],
                    },
                },
            }
        ]
    }
    plan = {
        "status": "consensus",
        "accepted_bundles": [
            {
                "id": "b1",
                "prediction_ids": ["p1"],
                "expected_relation": "primary_lower_e_hull_than_control",
                "primary": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Li-Fe-P-O",
                                "energy_above_hull": 0.0,
                            },
                            "filters": {"chemical_system": ["Li", "Fe", "P", "O"], "max_sites": 20},
                        }
                    ],
                },
                "control": {
                    "source": "mattergen",
                    "count": 1,
                    "mattergen_requests": [
                        {
                            "backend": "mattergen",
                            "properties_to_condition_on": {
                                "chemical_system": "Fe-P-O",
                                "energy_above_hull": 0.0,
                            },
                            "filters": {"chemical_system": ["Fe", "P", "O"], "max_sites": 20},
                        }
                    ],
                },
            }
        ],
    }

    augmented = mvp.execution_plan_with_prediction_drift_constraints(plan, prediction_consensus)
    assert "short P-O polyhedra" in " ".join(augmented["accepted_bundles"][0]["primary"]["drift_rejection_criteria"])
    selected, errors = materialize_plan([], augmented, seed=1, mattergen_config=config)

    assert errors == []
    primary = next(item for item in selected if item["physics_role"] == "primary")
    assert primary["formula"] == "LiFePO4"
    assert primary["crystal_llm_mattergen_report"]["controller_motif_rejects"][0]["formula"] == "LiFe2PO"


def test_mattergen_native_prompt_disables_mandatory_candidate_pool_gate() -> None:
    context = build_context_payload(
        state={"current_round": 1},
        pool_summary={},
        stage="prediction",
        current_inputs={
            "accepted_mechanisms": [],
            "target_count": 2,
            "materialization_backend": "mattergen",
            "mattergen_defaults": mattergen_prompt_defaults(parse_args([])),
        },
    )
    prompt = prompt_prediction_proposal(json.dumps(context))

    assert "MATTERGEN_NATIVE_NO_MANDATORY_MP_POOL" in prompt
    assert "require_chemical_system_exact" in prompt
    assert not _requires_mandatory_candidate_pool("prediction_agent_c", prompt)


def test_execution_repair_prompt_locks_successful_mattergen_branches() -> None:
    context = {
        "current_inputs": {
            "accepted_predictions": [],
            "target_count": 2,
            "materialization_backend": "mattergen",
        },
        "repair_feedback": {
            "successful_branches_to_preserve": ["b1.primary"],
            "successful_branch_policy": "locked",
        },
    }

    prompt = prompt_execution_repair_proposal(json.dumps(context))

    assert "successful_branches_to_preserve" in prompt
    assert "controller-locked" in prompt
    assert "Preserve their source, count, query, MatterGen requests, filters, sampling knobs" in prompt
    assert "You may increase target_count, batch_size, or num_batches only on failed MatterGen branches" in prompt
