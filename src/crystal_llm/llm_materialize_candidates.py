"""Agent C materialization of consensus hypotheses into formula probes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from crystal_llm.filters import load_known_formulas
from crystal_llm.hypothesis_schema import (
    MaterializerValidation,
    _string_list,
    schema_reference_json,
    validate_materializer_payload,
)
from crystal_llm.llm_client import LLMConfig, LLMError, ResponsesClient, extract_json_object
from crystal_llm.memory import STRICT_SUN_NOTE, read_json, write_json


MAX_JSON_REPAIR_ATTEMPTS = 2


AGENT_C_SYSTEM = """You are Agent C in a hypothesis-first crystal search MVP.
Your task is to turn an A/B consensus into exactly 10 generator-readable
formula_probes.

You must obey the local generator interface:
- Use only the listed templates.
- Use exactly the required roles for each template.
- Use positive oxidation states for cation roles and negative oxidation states
  for X.
- Use only elements available in the generator pools for the chosen oxidation
  states.
- Ensure role oxidation states are charge-neutral under the template
  stoichiometry.
- Use distinct role elements within a probe.
- Use exactly 10 probes and at least the requested template diversity.

If the consensus hypotheses are mutually contradictory or impossible to satisfy
with the generator interface, return status "hypothesis_conflict" and explain
the smallest fix needed. Otherwise return status "ok".

Do not reveal hidden chain-of-thought. Provide concise materialization notes and
reasoning summaries only.

Return JSON only:
{
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
        "X": {"element": "O", "oxidation_state": -2}
      },
      "hypothesis_ids": ["h001"],
      "rationale_summary": "..."
    }
  ],
  "materialization_notes": ["..."]
}

or:
{
  "status": "hypothesis_conflict",
  "conflicting_hypothesis_ids": ["h001", "h002"],
  "reason": "...",
  "minimal_fix_needed": "..."
}
"""


AGENT_D_SYSTEM = """You are Agent D, the strict consistency auditor for Agent C.
Your KPI is zero hypothesis-inconsistent materials entering the evaluator.

Audit Agent C's output against:
- every accepted A/B hypothesis,
- the local generator/schema constraints,
- compact-memory hard repeat rules,
- the strict SUN definition.

Be adversarial about consistency, but do not reject for vague preference,
speculation, or because a material is low-confidence. A rejection must cite a
specific violated hypothesis id, schema/generator rule, repeat rule, or
unresolved contradiction.

If Agent C returns materials, audit every formula_probe individually. If Agent C
returns hypothesis_conflict, audit whether that conflict is real and whether the
minimal fix is specific enough to send back to A/B.

Do not reveal hidden chain-of-thought. Provide concise audit summaries only.

Return JSON only:
{
  "status": "approved",
  "reviewed_agent_c_status": "ok",
  "approved_formula_probe_ids": ["r0001_probe_001"],
  "rejected_items": [],
  "conflict_review": null,
  "overall_reasoning_summary": "All probes satisfy the accepted hypotheses and hard rules."
}

or:
{
  "status": "rejected",
  "reviewed_agent_c_status": "ok",
  "approved_formula_probe_ids": ["r0001_probe_001"],
  "rejected_items": [
    {
      "probe_id": "r0001_probe_002",
      "formula_if_known": "K2ZnFeF6",
      "violated_hypothesis_ids": ["h001"],
      "violation_type": "hypothesis_mismatch",
      "reason": "The probe uses Rb where h001 requires K.",
      "required_fix": "Replace the A-site with K or cite a different accepted hypothesis that permits Rb."
    }
  ],
  "conflict_review": null,
  "overall_reasoning_summary": "One probe violates h001."
}

or:
{
  "status": "approved",
  "reviewed_agent_c_status": "hypothesis_conflict",
  "approved_formula_probe_ids": [],
  "rejected_items": [],
  "conflict_review": {
    "cd_agree_conflict": true,
    "conflicting_hypothesis_ids": ["h001", "h002"],
    "reason": "The conflict is real and blocks generator-readable probes.",
    "minimal_fix_needed": "Remove the incompatible exact formula requirement."
  },
  "overall_reasoning_summary": "Agent C's conflict report is valid."
}
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize consensus hypotheses into generator formula_probes.")
    parser.add_argument("--consensus", required=True, help="Consensus artifact from llm_debate_hypotheses.")
    parser.add_argument("--output", required=True, help="Materializer artifact JSON.")
    parser.add_argument("--strategy-output", required=True, help="Generator strategy JSON output.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--training-data", default=None)
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--min-template-diversity", type=int, default=2)
    parser.add_argument("--max-sites", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compact-memory", default=None)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument(
        "--auditor-max-rounds",
        type=int,
        default=4,
        help="Max Agent D reject/Agent C repair cycles inside one materializer call.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--llm-log-dir", default=None)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_consensus(path: Path) -> dict[str, Any]:
    raw = read_json(path, {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    consensus = raw.get("consensus")
    if isinstance(consensus, Mapping):
        return dict(consensus)
    if raw.get("status") == "consensus":
        return dict(raw)
    raise ValueError(f"{path} does not contain consensus payload")


def load_compact_memory(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    raw = read_json(Path(path), {})
    return dict(raw) if isinstance(raw, Mapping) else {}


def materializer_memory_view(compact_memory: Mapping[str, Any]) -> dict[str, Any]:
    constraints = compact_memory.get("negative_constraints")
    if not isinstance(constraints, Mapping):
        constraints = {}
    formula_memory = compact_memory.get("formula_memory")
    formulas = formula_memory if isinstance(formula_memory, list) else []
    return {
        "global_summary": compact_memory.get("global_summary"),
        "negative_constraints": dict(constraints),
        "high_priority_formula_memory": formulas[:80],
        "instructions": [
            "Do not exactly repeat formulas in forbidden_exact_repeat_formulas.",
            "Use near_miss_non_sun_formulas only as mutation neighborhoods.",
            "At most one exact confirmed_sun anchor replay is allowed in the 10 probes.",
        ],
    }


def repeat_memory_errors(validation: MaterializerValidation, compact_memory: Mapping[str, Any]) -> list[str]:
    if validation.status != "ok" or not validation.formulas:
        return []
    constraints = compact_memory.get("negative_constraints")
    if not isinstance(constraints, Mapping):
        return []
    forbidden = {str(item) for item in constraints.get("forbidden_exact_repeat_formulas", [])}
    anchors = {str(item) for item in constraints.get("confirmed_sun_anchors", [])}
    errors: list[str] = []
    repeated_forbidden = sorted({formula for formula in validation.formulas if formula in forbidden})
    if repeated_forbidden:
        errors.append(
            "formula_probes exactly repeat previously evaluated non-SUN formulas: "
            + ", ".join(repeated_forbidden)
            + ". Replace them with new formulas or justified mutations."
        )
    anchor_replays = [formula for formula in validation.formulas if formula in anchors]
    if len(anchor_replays) > 1:
        errors.append(
            "formula_probes include more than one exact confirmed SUN anchor replay: "
            + ", ".join(anchor_replays)
            + ". Keep at most one anchor and use the remaining slots for mutations/new probes."
        )
    return errors


def augment_validation_with_memory(
    validation: MaterializerValidation,
    compact_memory: Mapping[str, Any],
) -> MaterializerValidation:
    extra_errors = repeat_memory_errors(validation, compact_memory)
    if not extra_errors:
        return validation
    return MaterializerValidation(
        ok=False,
        status=validation.status,
        errors=list(validation.errors) + extra_errors,
        warnings=list(validation.warnings),
        strategy=validation.strategy,
        generated_count=validation.generated_count,
        formulas=validation.formulas,
        template_counts=validation.template_counts,
    )


def materialization_prompt(
    consensus: Mapping[str, Any],
    target_count: int,
    min_template_diversity: int,
    compact_memory: Mapping[str, Any],
) -> str:
    context = {
        "strict_sun_definition": STRICT_SUN_NOTE,
        "target_count": target_count,
        "minimum_template_diversity": min_template_diversity,
        "schema_reference": json.loads(schema_reference_json(target_count)),
        "consensus": dict(consensus),
        "compact_memory": materializer_memory_view(compact_memory),
        "instructions": [
            "Generate exactly target_count formula_probes.",
            "Every probe must satisfy all accepted hypotheses unless you return hypothesis_conflict.",
            "Prefer candidates that are chemically plausible and falsify different parts of the consensus.",
            "Use at least minimum_template_diversity distinct templates unless consensus truly forbids it; if forbidden, return conflict.",
            "Hard rule: do not exactly repeat any formula in compact_memory.negative_constraints.forbidden_exact_repeat_formulas.",
            "Hard rule: include at most one exact formula from compact_memory.negative_constraints.confirmed_sun_anchors.",
            "Do not include prose outside JSON.",
        ],
    }
    return f"""Materialize the A/B consensus into formula_probes for the generator.

CONTEXT_JSON:
```json
{json.dumps(context, ensure_ascii=False, indent=2)}
```

Return only Agent C JSON."""


def debug_prompt(
    consensus: Mapping[str, Any],
    original_user: str,
    previous_payload: Mapping[str, Any] | None,
    validation: MaterializerValidation,
    target_count: int,
    min_template_diversity: int,
    compact_memory: Mapping[str, Any],
) -> str:
    return f"""Your previous Agent C output could not be accepted by the local
schema/generator validation. Debug it and return a corrected JSON object.

If the error proves the A/B hypotheses are impossible or contradictory under
the generator interface, return status "hypothesis_conflict". Otherwise return
status "ok" with exactly {target_count} valid formula_probes and at least
{min_template_diversity} distinct templates.

Strict metric guard: {STRICT_SUN_NOTE}

LOCAL_SCHEMA_REFERENCE:
```json
{schema_reference_json(target_count)}
```

CONSENSUS_JSON:
```json
{json.dumps(dict(consensus), ensure_ascii=False, indent=2)}
```

COMPACT_MEMORY_REPEAT_RULES:
```json
{json.dumps(materializer_memory_view(compact_memory), ensure_ascii=False, indent=2)}
```

VALIDATION_STATUS:
```json
{json.dumps({
    "ok": validation.ok,
    "status": validation.status,
    "errors": validation.errors,
    "warnings": validation.warnings,
    "generated_count": validation.generated_count,
    "formulas": validation.formulas,
    "template_counts": validation.template_counts,
}, ensure_ascii=False, indent=2)}
```

PREVIOUS_AGENT_C_JSON:
```json
{json.dumps(dict(previous_payload or {}), ensure_ascii=False, indent=2)}
```

ORIGINAL_TASK_PROMPT:
```text
{original_user}
```

Return only corrected Agent C JSON."""


def auditor_prompt(
    consensus: Mapping[str, Any],
    agent_c_payload: Mapping[str, Any],
    validation: MaterializerValidation,
    target_count: int,
    min_template_diversity: int,
    compact_memory: Mapping[str, Any],
) -> str:
    return f"""Audit Agent C's materialization output before it can reach the
generator/evaluator.

Audit policy:
- If Agent C status is "ok", review every formula_probe individually.
- Approve only if all probes satisfy the accepted hypotheses and hard rules.
- Reject with concrete, localized fixes for each bad probe.
- If Agent C status is "hypothesis_conflict", approve only if the conflict is
  real, blocks all valid 10-probe materialization, and the minimal fix is
  specific enough to return to A/B.
- Do not reject merely because a material is low-confidence or likely non-SUN.

Strict metric guard: {STRICT_SUN_NOTE}

LOCAL_SCHEMA_REFERENCE:
```json
{schema_reference_json(target_count)}
```

CONSENSUS_JSON:
```json
{json.dumps(dict(consensus), ensure_ascii=False, indent=2)}
```

COMPACT_MEMORY_REPEAT_RULES:
```json
{json.dumps(materializer_memory_view(compact_memory), ensure_ascii=False, indent=2)}
```

LOCAL_VALIDATION_STATUS:
```json
{json.dumps({
    "ok": validation.ok,
    "status": validation.status,
    "errors": validation.errors,
    "warnings": validation.warnings,
    "generated_count": validation.generated_count,
    "formulas": validation.formulas,
    "template_counts": validation.template_counts,
    "target_count": target_count,
    "minimum_template_diversity": min_template_diversity,
}, ensure_ascii=False, indent=2)}
```

AGENT_C_JSON:
```json
{json.dumps(dict(agent_c_payload), ensure_ascii=False, indent=2)}
```

Return only Agent D audit JSON."""


def cd_repair_prompt(
    consensus: Mapping[str, Any],
    original_user: str,
    previous_payload: Mapping[str, Any],
    validation: MaterializerValidation,
    audit: Mapping[str, Any],
    target_count: int,
    min_template_diversity: int,
    compact_memory: Mapping[str, Any],
) -> str:
    return f"""Agent D rejected your previous Agent C output. Revise it.

If Agent D rejected specific probes, repair only those failure modes while
preserving all valid materialization intent. If Agent D rejected your
hypothesis_conflict report, either generate valid probes or produce a sharper
conflict report that Agent D can audit.

Return status "hypothesis_conflict" only if the accepted hypotheses are truly
impossible or contradictory under the generator interface. Otherwise return
status "ok" with exactly {target_count} valid formula_probes and at least
{min_template_diversity} distinct templates.

Strict metric guard: {STRICT_SUN_NOTE}

LOCAL_SCHEMA_REFERENCE:
```json
{schema_reference_json(target_count)}
```

CONSENSUS_JSON:
```json
{json.dumps(dict(consensus), ensure_ascii=False, indent=2)}
```

COMPACT_MEMORY_REPEAT_RULES:
```json
{json.dumps(materializer_memory_view(compact_memory), ensure_ascii=False, indent=2)}
```

LOCAL_VALIDATION_STATUS:
```json
{json.dumps({
    "ok": validation.ok,
    "status": validation.status,
    "errors": validation.errors,
    "warnings": validation.warnings,
    "generated_count": validation.generated_count,
    "formulas": validation.formulas,
    "template_counts": validation.template_counts,
}, ensure_ascii=False, indent=2)}
```

AGENT_D_AUDIT_JSON:
```json
{json.dumps(dict(audit), ensure_ascii=False, indent=2)}
```

PREVIOUS_AGENT_C_JSON:
```json
{json.dumps(dict(previous_payload), ensure_ascii=False, indent=2)}
```

ORIGINAL_TASK_PROMPT:
```text
{original_user}
```

Return only corrected Agent C JSON."""


def parse_retry_prompt(role: str, original_user: str, invalid_output: str, error: str, target_count: int) -> str:
    return f"""Your previous output for role `{role}` could not be parsed as one
valid JSON object. Revise it into exactly one valid JSON object and preserve the
same materialization intent unless the error requires correction.

Strict metric guard: {STRICT_SUN_NOTE}

LOCAL_SCHEMA_REFERENCE:
```json
{schema_reference_json(target_count)}
```

PARSER_ERROR:
```text
{error}
```

ORIGINAL_TASK_PROMPT:
```text
{original_user}
```

INVALID_PREVIOUS_OUTPUT:
```text
{invalid_output}
```
"""


def transient_llm_retry_prompt(role: str, original_user: str, error: str) -> str:
    return f"""The previous LLM call for role `{role}` failed before usable text
could be extracted. Retry the original task now and return exactly one valid
JSON object.

TRANSIENT_LLM_ERROR:
```text
{error}
```

ORIGINAL_TASK_PROMPT:
```text
{original_user}
```
"""


def call_json(
    client: ResponsesClient,
    *,
    system: str,
    user: str,
    metadata: Mapping[str, Any],
    target_count: int,
) -> dict[str, Any]:
    role = str(metadata.get("role", "agent_c"))
    prompt = user
    last_text = ""
    last_error = ""
    for attempt in range(MAX_JSON_REPAIR_ATTEMPTS + 1):
        try:
            text = client.complete_text(
                system=system,
                user=prompt,
                metadata={**dict(metadata), "json_retry_attempt": attempt},
            )
            last_text = text
        except LLMError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_JSON_REPAIR_ATTEMPTS:
                prompt = transient_llm_retry_prompt(role, user, last_error)
                continue
            raise ValueError(f"LLM call for {role} failed after retry: {last_error}") from exc
        try:
            parsed = extract_json_object(text)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if isinstance(parsed, Mapping):
                return dict(parsed)
            last_error = f"Agent C output must be a JSON object, got {type(parsed).__name__}"
        if attempt < MAX_JSON_REPAIR_ATTEMPTS:
            prompt = parse_retry_prompt(role, user, last_text, last_error, target_count)
    raise ValueError(f"Agent C output could not be parsed after retry: {last_error}")


def probe_ids(payload: Mapping[str, Any]) -> list[str]:
    probes = payload.get("formula_probes")
    if not isinstance(probes, list):
        return []
    ids: list[str] = []
    for index, probe in enumerate(probes):
        if isinstance(probe, Mapping):
            probe_id = str(probe.get("id") or "").strip()
            ids.append(probe_id or f"formula_probes[{index}]")
    return ids


def validate_agent_d_audit(
    audit: Any,
    *,
    agent_c_payload: Mapping[str, Any],
    validation: MaterializerValidation,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(audit, Mapping):
        return ["Agent D audit must be a JSON object"]
    status = str(audit.get("status") or "").strip()
    if status not in {"approved", "rejected"}:
        errors.append('Agent D audit.status must be "approved" or "rejected"')
    reviewed = str(audit.get("reviewed_agent_c_status") or "").strip()
    agent_c_status = str(agent_c_payload.get("status") or "").strip()
    if reviewed != agent_c_status:
        errors.append(
            f"Agent D audit.reviewed_agent_c_status must match Agent C status "
            f"{agent_c_status!r}, got {reviewed!r}"
        )
    rejected = audit.get("rejected_items")
    if not isinstance(rejected, list):
        errors.append("Agent D audit.rejected_items must be a list")
        rejected = []
    approved_ids = audit.get("approved_formula_probe_ids")
    if not isinstance(approved_ids, list):
        errors.append("Agent D audit.approved_formula_probe_ids must be a list")
        approved_ids = []
    if status == "rejected" and not rejected:
        errors.append("Agent D rejected the output but did not provide rejected_items")
    for index, item in enumerate(rejected):
        if not isinstance(item, Mapping):
            errors.append(f"Agent D rejected_items[{index}] must be an object")
            continue
        if not str(item.get("reason") or "").strip():
            errors.append(f"Agent D rejected_items[{index}].reason is required")
        if not str(item.get("required_fix") or "").strip():
            errors.append(f"Agent D rejected_items[{index}].required_fix is required")
    if agent_c_status == "ok":
        expected_ids = set(probe_ids(agent_c_payload))
        approved_set = {str(item) for item in approved_ids}
        rejected_ids = {
            str(item.get("probe_id"))
            for item in rejected
            if isinstance(item, Mapping) and item.get("probe_id") is not None
        }
        unknown = sorted((approved_set | rejected_ids) - expected_ids)
        if unknown:
            errors.append(f"Agent D audit references unknown probe ids: {unknown}")
        if status == "approved":
            if not validation.ok:
                errors.append("Agent D cannot approve locally invalid Agent C materialization")
            missing = sorted(expected_ids - approved_set)
            if missing:
                errors.append(f"Agent D approval did not approve every probe id: {missing}")
            if rejected:
                errors.append("Agent D approval must not include rejected_items")
    elif agent_c_status == "hypothesis_conflict":
        conflict_review = audit.get("conflict_review")
        if status == "approved":
            if not isinstance(conflict_review, Mapping):
                errors.append("Agent D approved a conflict but conflict_review is missing")
            else:
                if conflict_review.get("cd_agree_conflict") is not True:
                    errors.append("Agent D approved conflict requires conflict_review.cd_agree_conflict=true")
                if not _string_list(conflict_review.get("conflicting_hypothesis_ids")):
                    errors.append("Agent D conflict_review.conflicting_hypothesis_ids must be non-empty")
                if not str(conflict_review.get("reason") or "").strip():
                    errors.append("Agent D conflict_review.reason is required")
                if not str(conflict_review.get("minimal_fix_needed") or "").strip():
                    errors.append("Agent D conflict_review.minimal_fix_needed is required")
    else:
        errors.append(f"Agent D cannot audit unsupported Agent C status {agent_c_status!r}")
    return errors


def run_materializer(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    consensus_path = Path(args.consensus)
    consensus = load_consensus(consensus_path)
    compact_memory = load_compact_memory(args.compact_memory)
    training_data = Path(args.training_data).resolve() if args.training_data else (
        root / "archive" / "matllmsearch_evaluator" / "data" / "a_training.json"
    )
    known_formulas = load_known_formulas(str(training_data))
    client = ResponsesClient(
        LLMConfig.from_env(
            dotenv=args.dotenv,
            role="MATERIALIZER",
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        log_dir=args.llm_log_dir,
    )
    auditor_client = ResponsesClient(
        LLMConfig.from_env(
            dotenv=args.dotenv,
            role="MATERIALIZER_AUDITOR",
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        log_dir=args.llm_log_dir,
    )

    artifacts: list[dict[str, Any]] = []
    prompt = materialization_prompt(consensus, args.target_count, args.min_template_diversity, compact_memory)
    previous_payload: dict[str, Any] | None = None
    validation = MaterializerValidation(False, "not_started")
    attempts = max(1, args.max_attempts)
    auditor_max_rounds = max(1, args.auditor_max_rounds)

    for attempt in range(1, attempts + 1):
        payload = call_json(
            client,
            system=AGENT_C_SYSTEM,
            user=prompt,
            metadata={"role": "agent_c", "round": args.round, "attempt": attempt},
            target_count=args.target_count,
        )
        validation = validate_materializer_payload(
            payload,
            target_count=args.target_count,
            max_sites=args.max_sites,
            seed=args.seed,
            known_formulas=known_formulas,
            min_template_diversity=args.min_template_diversity,
        )
        validation = augment_validation_with_memory(validation, compact_memory)
        artifacts.append(
            {
                "attempt": attempt,
                "payload": payload,
                "validation": {
                    "ok": validation.ok,
                    "status": validation.status,
                    "errors": validation.errors,
                    "warnings": validation.warnings,
                    "generated_count": validation.generated_count,
                    "formulas": validation.formulas,
                    "template_counts": validation.template_counts,
                },
            }
        )
        previous_payload = payload
        if validation.ok and validation.status in {"ok", "hypothesis_conflict"}:
            audit_prompt = auditor_prompt(
                consensus,
                payload,
                validation,
                args.target_count,
                args.min_template_diversity,
                compact_memory,
            )
            audit: dict[str, Any] | None = None
            audit_errors: list[str] = []
            for audit_attempt in range(1, 3):
                audit = call_json(
                    auditor_client,
                    system=AGENT_D_SYSTEM,
                    user=audit_prompt,
                    metadata={
                        "role": "agent_d",
                        "round": args.round,
                        "agent_c_attempt": attempt,
                        "audit_attempt": audit_attempt,
                    },
                    target_count=args.target_count,
                )
                audit_errors = validate_agent_d_audit(audit, agent_c_payload=payload, validation=validation)
                if not audit_errors:
                    break
                audit_prompt = f"""Your previous Agent D audit JSON failed local audit-schema validation.
Return a corrected Agent D audit JSON for the same Agent C output.

AUDIT_SCHEMA_ERRORS:
```json
{json.dumps(audit_errors, ensure_ascii=False, indent=2)}
```

ORIGINAL_AUDIT_TASK:
```text
{auditor_prompt(consensus, payload, validation, args.target_count, args.min_template_diversity, compact_memory)}
```"""
            artifacts[-1]["agent_d_audit"] = audit
            artifacts[-1]["agent_d_audit_errors"] = audit_errors
            if audit is not None and not audit_errors and audit.get("status") == "approved":
                if validation.status == "ok":
                    strategy = validation.strategy
                    write_json(Path(args.strategy_output), strategy)
                    return {
                        "created_at_utc": utc_now(),
                        "status": "ok",
                        "round": args.round,
                        "consensus_path": str(consensus_path),
                        "strategy_output": str(Path(args.strategy_output)),
                        "attempts": artifacts,
                        "final_payload": payload,
                        "agent_d_final_audit": audit,
                        "validation": artifacts[-1]["validation"],
                    }
                return {
                    "created_at_utc": utc_now(),
                    "status": "hypothesis_conflict",
                    "round": args.round,
                    "consensus_path": str(consensus_path),
                    "attempts": artifacts,
                    "conflict_report": payload,
                    "agent_d_final_audit": audit,
                }
            if attempt < min(attempts, auditor_max_rounds):
                prompt = cd_repair_prompt(
                    consensus,
                    materialization_prompt(consensus, args.target_count, args.min_template_diversity, compact_memory),
                    payload,
                    validation,
                    audit
                    if audit is not None
                    else {
                        "status": "rejected",
                        "reviewed_agent_c_status": validation.status,
                        "rejected_items": [
                            {
                                "probe_id": "agent_d_audit",
                                "violation_type": "audit_schema_invalid",
                                "reason": "Agent D audit could not be validated locally.",
                                "required_fix": "Return a more explicit Agent C materialization or conflict report that Agent D can audit.",
                            }
                        ],
                        "audit_schema_errors": audit_errors,
                    },
                    args.target_count,
                    args.min_template_diversity,
                    compact_memory,
                )
                continue
            return {
                "created_at_utc": utc_now(),
                "status": "hypothesis_conflict",
                "round": args.round,
                "consensus_path": str(consensus_path),
                "attempts": artifacts,
                "conflict_report": {
                    "status": "hypothesis_conflict",
                    "conflicting_hypothesis_ids": _string_list(payload.get("hypothesis_ids"))
                    or _string_list(payload.get("conflicting_hypothesis_ids")),
                    "reason": "Agent C and Agent D did not reach agreement within "
                    f"{auditor_max_rounds} C/D audit-repair rounds.",
                    "minimal_fix_needed": "A/B should make the accepted hypotheses more deterministic, remove "
                    "ambiguous exact formula requirements, or relax constraints that Agent D rejected.",
                    "last_agent_c_payload": payload,
                    "last_agent_d_audit": audit,
                    "last_agent_d_audit_errors": audit_errors,
                },
                "agent_d_final_audit": audit,
            }
        prompt = debug_prompt(
            consensus,
            materialization_prompt(consensus, args.target_count, args.min_template_diversity, compact_memory),
            previous_payload,
            validation,
            args.target_count,
            args.min_template_diversity,
            compact_memory,
        )

    return {
        "created_at_utc": utc_now(),
        "status": "invalid",
        "round": args.round,
        "consensus_path": str(consensus_path),
        "attempts": artifacts,
        "last_validation": {
            "ok": validation.ok,
            "status": validation.status,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "generated_count": validation.generated_count,
            "formulas": validation.formulas,
            "template_counts": validation.template_counts,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_materializer(args)
    write_json(Path(args.output), payload)
    print(f"wrote {args.output}")
    print(f"status={payload.get('status')}")
    return 0 if payload.get("status") in {"ok", "hypothesis_conflict"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
