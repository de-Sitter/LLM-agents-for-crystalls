"""Agent C/D compiler and audit loop for restricted candidate scorer DSL."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import orjson

from crystal_llm.llm_client import LLMConfig, LLMError, ResponsesClient, extract_json_object
from crystal_llm.memory import read_json, write_json
from crystal_llm.scorer_dsl import (
    ALLOWED_FEATURES,
    ALLOWED_OPS,
    FORBIDDEN_FEATURES,
    diagnostic_distribution,
    validate_scorer_behavior,
    validate_scorer_payload,
)


MAX_JSON_REPAIR_ATTEMPTS = 2


AGENT_C_SYSTEM = """You are Agent C, the compiler in a candidate-pool scoring MVP.
Your job is to convert A/B consensus rules into one restricted JSON scorer DSL.
You must not output Python code. You must not use forbidden target/leakage fields.

Return JSON only:
{
  "status": "ok",
  "scorer": {
    "schema_version": "candidate_scorer.v1",
    "scorer_id": "round_0001_scorer",
    "hypothesis_ids": ["r001"],
    "description": "...",
    "terms": [
      {"op": "contains_any_element", "feature": "elements", "values": ["F"], "weight": 2.0, "reason": "..."}
    ]
  },
  "compilation_notes": ["..."],
  "assumptions": ["..."]
}

If the A/B rules are contradictory or impossible to express in the DSL, return:
{
  "status": "rule_conflict",
  "conflicting_rule_ids": ["r001", "r002"],
  "reason": "...",
  "needed_ab_revision": ["..."]
}
"""


AGENT_D_SYSTEM = """You are Agent D, the adversarial auditor for Agent C's scorer
DSL. Your KPI is to reject flawed scorers whenever justified.

Check:
- fidelity to A/B consensus;
- no leakage fields such as is_stable, e_hull, SUN, novelty, material_id, cif_path;
- no executable code;
- no hidden use of evaluated outcomes for unevaluated candidates;
- terms are neither absurdly narrow nor too broad;
- every term has a materials-rule reason.

Do not reveal hidden chain-of-thought. Provide concise reasoning summaries.

Return JSON only:
{
  "agent": "D",
  "approve": true,
  "audit_summary": "...",
  "required_revisions": [],
  "risk_flags": [],
  "fidelity_notes": ["..."]
}
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile A/B consensus rules into scorer DSL with Agent D audit.")
    parser.add_argument("--consensus", required=True)
    parser.add_argument("--candidate-pool", default="data/mp_candidate_pool/mp_candidates_filtered.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scorer-output", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--auditor-max-rounds", type=int, default=4)
    parser.add_argument("--validation-sample-size", type=int, default=20000)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--llm-log-dir", default=None)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_candidate_sample(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = orjson.loads(line)
            if isinstance(item, dict):
                records.append(item)
                if len(records) >= limit:
                    break
    return records


def dsl_reference() -> dict[str, Any]:
    return {
        "schema_version": "candidate_scorer.v1",
        "allowed_features": sorted(ALLOWED_FEATURES),
        "forbidden_features": sorted(FORBIDDEN_FEATURES | {"is_stable"}),
        "allowed_ops": sorted(ALLOWED_OPS),
        "examples": [
            {"op": "contains_any_element", "feature": "elements", "values": ["F", "O"], "weight": 1.5},
            {"op": "numeric_range", "feature": "band_gap", "min": 0.5, "max": 2.5, "weight": 1.0},
            {"op": "category_in", "feature": "crystal_system", "values": ["Cubic", "Tetragonal"], "weight": 0.8},
            {"op": "numeric_prefer_low", "feature": "nsites", "scale": 40.0, "weight": 0.5},
            {
                "op": "all_of",
                "weight": -0.8,
                "conditions": [
                    {"op": "contains_any_element", "feature": "elements", "values": ["V", "Nb", "Ta", "W", "Re"]},
                    {"op": "numeric_range", "feature": "nelements", "min": 1, "max": 3},
                    {"op": "contains_no_elements", "feature": "elements", "values": ["O", "F", "Cl", "Br", "I", "N", "P", "As", "S", "Se"]},
                    {"op": "numeric_range", "feature": "band_gap", "min": 0.0, "max": 0.05}
                ],
                "reason": "Conjunctive gated penalty: only fires if all conditions match."
            },
        ],
    }


def compile_prompt(
    consensus: Mapping[str, Any],
    sample_records: Sequence[Mapping[str, Any]],
    repair_feedback: Mapping[str, Any] | None,
    round_number: int,
) -> str:
    payload = {
        "round": round_number,
        "ab_consensus": dict(consensus),
        "dsl_reference": dsl_reference(),
        "candidate_pool_sample_records": list(sample_records[:12]),
        "repair_feedback": repair_feedback,
        "instructions": [
            "Compile the accepted A/B rules into scorer.terms.",
            "Use only allowed_features.",
            "Do not use is_stable, e_hull, SUN, novelty, material_id, cif_path, properties_path, or any evaluator result as a scorer feature.",
            "Each term weight should be modest; use multiple interpretable terms rather than one huge weight.",
            "If the rules are contradictory or impossible under the DSL, return status=rule_conflict.",
        ],
    }
    return f"""Agent C: compile A/B consensus into the restricted scorer DSL.

TASK_JSON:
```json
{json.dumps(payload, indent=2, ensure_ascii=False)}
```

Return only Agent C JSON."""


def audit_prompt(
    consensus: Mapping[str, Any],
    compiler_payload: Mapping[str, Any],
    local_validation: Mapping[str, Any],
    sample_records: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "ab_consensus": dict(consensus),
        "agent_c_payload": dict(compiler_payload),
        "local_program_validation": dict(local_validation),
        "candidate_pool_sample_records": list(sample_records[:8]),
        "audit_policy": dsl_reference(),
    }
    return f"""Agent D: audit Agent C's scorer DSL.

AUDIT_JSON:
```json
{json.dumps(payload, indent=2, ensure_ascii=False)}
```

Return only Agent D JSON."""


def retry_prompt(role: str, original_user: str, invalid_output: str, error: str) -> str:
    return f"""Your previous output for role `{role}` was invalid. Return exactly
one valid JSON object and preserve the intended content when possible.

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
{invalid_output}
```
"""


def valid_shape(parsed: Any, role: str) -> bool:
    if not isinstance(parsed, Mapping):
        return False
    if role == "scorer_compiler":
        if parsed.get("status") == "rule_conflict":
            return isinstance(parsed.get("reason"), str)
        return parsed.get("status") == "ok" and isinstance(parsed.get("scorer"), Mapping)
    if role == "scorer_auditor":
        return parsed.get("agent") == "D" and isinstance(parsed.get("approve"), bool)
    return True


def call_json(client: ResponsesClient, *, system: str, user: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    role = str(metadata.get("role", ""))
    prompt = user
    last_text = ""
    last_error = ""
    for attempt in range(MAX_JSON_REPAIR_ATTEMPTS + 1):
        try:
            text = client.complete_text(system=system, user=prompt, metadata={**dict(metadata), "json_retry_attempt": attempt})
            last_text = text
        except LLMError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_JSON_REPAIR_ATTEMPTS:
                prompt = retry_prompt(role, user, "", last_error)
                continue
            raise
        try:
            parsed = extract_json_object(text)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if valid_shape(parsed, role):
                return dict(parsed)
            last_error = f"output did not match required shape for {role}"
        if attempt < MAX_JSON_REPAIR_ATTEMPTS:
            prompt = retry_prompt(role, user, last_text, last_error)
    raise ValueError(f"LLM output for {role} could not be repaired: {last_error}")


def local_validate(payload: Mapping[str, Any], sample_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if payload.get("status") == "rule_conflict":
        return {"ok": True, "kind": "rule_conflict"}
    scorer = payload.get("scorer")
    if not isinstance(scorer, Mapping):
        return {"ok": False, "errors": ["missing scorer object"]}
    errors = validate_scorer_behavior(scorer, sample_records)
    distribution = diagnostic_distribution(scorer, sample_records) if not validate_scorer_payload(scorer) else {}
    return {"ok": not errors, "errors": errors, "score_distribution": distribution}


def run_compile(args: argparse.Namespace) -> dict[str, Any]:
    consensus = read_json(Path(args.consensus), {})
    if not isinstance(consensus, Mapping):
        raise ValueError("consensus must be a JSON object")
    sample_records = read_candidate_sample(Path(args.candidate_pool), args.validation_sample_size)
    if not sample_records:
        raise ValueError("candidate pool sample is empty")

    compiler = ResponsesClient(
        LLMConfig.from_env(dotenv=args.dotenv, role="SCORER_COMPILER", model=args.model, temperature=args.temperature, max_tokens=args.max_tokens),
        log_dir=args.llm_log_dir,
    )
    auditor = ResponsesClient(
        LLMConfig.from_env(dotenv=args.dotenv, role="SCORER_AUDITOR", model=args.model, temperature=args.temperature, max_tokens=args.max_tokens),
        log_dir=args.llm_log_dir,
    )

    dialogue: list[dict[str, Any]] = []
    repair_feedback: Mapping[str, Any] | None = None
    for attempt in range(1, max(1, args.max_attempts) + 1):
        c_payload = call_json(
            compiler,
            system=AGENT_C_SYSTEM,
            user=compile_prompt(consensus, sample_records, repair_feedback, args.round),
            metadata={"role": "scorer_compiler", "round": args.round, "attempt": attempt},
        )
        local = local_validate(c_payload, sample_records)
        dialogue.append({"role": "C", "attempt": attempt, "payload": c_payload, "local_validation": local})

        if c_payload.get("status") == "rule_conflict":
            d_payload = call_json(
                auditor,
                system=AGENT_D_SYSTEM,
                user=audit_prompt(consensus, c_payload, local, sample_records),
                metadata={"role": "scorer_auditor", "round": args.round, "attempt": attempt},
            )
            dialogue.append({"role": "D", "attempt": attempt, "payload": d_payload})
            if bool(d_payload.get("approve")):
                return {
                    "status": "rule_conflict",
                    "created_at_utc": utc_now(),
                    "round": args.round,
                    "conflict": c_payload,
                    "agent_d_final_audit": d_payload,
                    "dialogue": dialogue,
                }
            repair_feedback = {"agent_d_rejection": d_payload, "local_validation": local}
            continue

        if not local.get("ok"):
            repair_feedback = {"local_validation_errors": local.get("errors", []), "local_validation": local}
            continue

        d_payload = call_json(
            auditor,
            system=AGENT_D_SYSTEM,
            user=audit_prompt(consensus, c_payload, local, sample_records),
            metadata={"role": "scorer_auditor", "round": args.round, "attempt": attempt},
        )
        dialogue.append({"role": "D", "attempt": attempt, "payload": d_payload})
        if bool(d_payload.get("approve")):
            scorer = c_payload["scorer"]
            return {
                "status": "ok",
                "created_at_utc": utc_now(),
                "round": args.round,
                "scorer": scorer,
                "compiler_payload": c_payload,
                "local_validation": local,
                "agent_d_final_audit": d_payload,
                "dialogue": dialogue,
            }
        repair_feedback = {"agent_d_rejection": d_payload, "local_validation": local}

    return {
        "status": "failed",
        "created_at_utc": utc_now(),
        "round": args.round,
        "reason": "Agent C/D did not produce an approved valid scorer within max attempts.",
        "dialogue": dialogue,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_compile(args)
    write_json(Path(args.output), result)
    if result.get("status") == "ok":
        write_json(Path(args.scorer_output), result["scorer"])
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") in {"ok", "rule_conflict"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
