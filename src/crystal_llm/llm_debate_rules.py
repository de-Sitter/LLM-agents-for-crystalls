"""A/B debate for general candidate-pool scoring rules."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from crystal_llm.compact_hypothesis_memory import build_compact_memory
from crystal_llm.llm_client import LLMConfig, LLMError, ResponsesClient, extract_json_object
from crystal_llm.memory import STRICT_SUN_NOTE, read_json, write_json
from crystal_llm.scorer_dsl import ALLOWED_FEATURES, FORBIDDEN_FEATURES


MAX_JSON_REPAIR_ATTEMPTS = 2


AGENT_A_SYSTEM = """You are Agent A in a candidate-pool rule-learning MVP.
Your job is to propose general, falsifiable materials rules that may enrich
strict SUN candidates when applied to a real Materials Project candidate pool.

Use the compact history: prior scorer rules, top/random/bottom evaluator
outcomes, positive experiences, negative experiences, and known counterexamples.
Your rules must be expressible using observable candidate-pool fields only.

Do not reveal hidden chain-of-thought. Provide concise evidence chains,
premises, counterevidence considered, assumptions, and reasoning summaries.
Return JSON only:
{
  "agent": "A",
  "agree": false,
  "concede": false,
  "rules": [
    {
      "id": "r001",
      "claim": "...",
      "rationale_summary": "...",
      "expected_positive_features": ["..."],
      "expected_negative_features": ["..."],
      "known_counterexamples": ["..."],
      "allowed_observable_fields": ["elements", "band_gap"],
      "falsifiable_predictions": ["top-scored candidates should ..."],
      "scope": "...",
      "confidence": "low|medium|high"
    }
  ],
  "response_to_b": "...",
  "required_b_revisions": []
}
"""


AGENT_B_SYSTEM = """You are Agent B, the rigorous critic in a candidate-pool
rule-learning MVP.

Attack Agent A's rules using materials knowledge, prior outcomes, metric
discipline, and leakage controls. Do not object blindly: each objection must
cite a premise, implication, counterexample, or evaluator/candidate-pool
constraint. Do not reveal hidden chain-of-thought; provide evidence chains and
reasoning summaries only.

Reject any rule that depends on target leakage fields such as is_stable, e_hull,
SUN, novelty, material_id identity, or direct evaluator results for unevaluated
candidates.

Return JSON only during debate:
{
  "agent": "B",
  "agree": true,
  "concede": false,
  "critique_summary": "...",
  "evidence_chain": [
    {"premise": "...", "implication": "...", "confidence": "low|medium|high"}
  ],
  "counterexamples": ["..."],
  "required_revisions": ["..."],
  "accepted_rule_ids": ["r001"],
  "rejected_rule_ids": ["r002"],
  "risk_flags": ["leakage_risk"]
}

When asked for final consensus, return JSON only:
{
  "status": "consensus",
  "conceded_by": "A|B|none",
  "accepted_rules": [
    {
      "id": "r001",
      "claim": "...",
      "rationale_summary": "...",
      "expected_positive_features": ["..."],
      "expected_negative_features": ["..."],
      "known_counterexamples": ["..."],
      "allowed_observable_fields": ["elements", "band_gap"],
      "falsifiable_predictions": ["..."],
      "scope": "...",
      "confidence": "low|medium|high"
    }
  ],
  "rejected_rules": [
    {"id": "r002", "claim": "...", "rejection_reason": "..."}
  ],
  "consensus_summary": "...",
  "scorer_requirements": {
    "must_use_fields": ["elements"],
    "must_not_use_fields": ["is_stable", "e_hull", "sun"],
    "desired_behavior": ["top-ranked candidates should be enriched over random"]
  }
}
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A/B debate for candidate-pool scoring rules.")
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--memory-dir", default="memory")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--max-dialogue-rounds", type=int, default=100)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--llm-log-dir", default=None)
    parser.add_argument("--compact-memory", default=None)
    parser.add_argument("--scorer-conflict-report", default=None)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slim_state(state: Mapping[str, Any], *, recent_rounds: int = 5) -> dict[str, Any]:
    history = state.get("history", [])
    if not isinstance(history, list):
        history = []
    return {
        "schema_version": state.get("schema_version"),
        "status": state.get("status"),
        "current_round": state.get("current_round"),
        "best_top_bucket_strict_sun_rate": state.get("best_top_bucket_strict_sun_rate"),
        "best_round": state.get("best_round"),
        "history": history[-recent_rounds:],
        "unresolved_debates": (state.get("unresolved_debates") or [])[-5:] if isinstance(state.get("unresolved_debates"), list) else [],
    }


def prompt_context(
    *,
    state: Mapping[str, Any],
    compact_memory: Mapping[str, Any],
    scorer_conflict_report: Mapping[str, Any] | None,
) -> str:
    payload = {
        "strict_sun_definition": STRICT_SUN_NOTE,
        "candidate_pool": {
            "source": "data/mp_candidate_pool/mp_candidates_filtered.jsonl",
            "description": "Real MP structures outside evaluator training formulas, with CIF paths and non-target metadata.",
            "allowed_observable_fields": sorted(ALLOWED_FEATURES),
            "forbidden_leakage_fields": sorted(FORBIDDEN_FEATURES | {"is_stable"}),
            "important_note": "MP is_stable is deliberately forbidden as a scorer input because it leaks target-like stability information.",
        },
        "state": slim_state(state),
        "compact_memory": dict(compact_memory),
        "scorer_conflict_report": scorer_conflict_report,
        "instructions": [
            "Propose general rules, not specific material identities.",
            "Rules must be compilable into the restricted scorer DSL.",
            "Use only observable allowed fields.",
            "Do not use material_id, cif_path, properties_path, is_stable, e_hull, novelty, SUN, or any evaluator result for unevaluated candidates as scoring inputs.",
            "Prefer rules with falsifiable top-vs-random predictions.",
            "Use evidence chains and reasoning summaries; do not expose hidden chain-of-thought.",
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def proposal_prompt(context_json: str) -> str:
    return f"""Agent A: propose or revise candidate-pool scoring rules.

CONTEXT_JSON:
```json
{context_json}
```

Return only Agent A JSON."""


def critique_prompt(context_json: str, proposal: Mapping[str, Any], cycle: int) -> str:
    return f"""Agent B: critique Agent A's candidate-pool rules for cycle {cycle}.

CONTEXT_JSON:
```json
{context_json}
```

AGENT_A_JSON:
```json
{json.dumps(dict(proposal), indent=2, ensure_ascii=False)}
```

Return only Agent B JSON."""


def revision_prompt(context_json: str, proposal: Mapping[str, Any], critique: Mapping[str, Any], cycle: int) -> str:
    return f"""Agent A: revise after Agent B critique in cycle {cycle}.

CONTEXT_JSON:
```json
{context_json}
```

PREVIOUS_AGENT_A_JSON:
```json
{json.dumps(dict(proposal), indent=2, ensure_ascii=False)}
```

AGENT_B_JSON:
```json
{json.dumps(dict(critique), indent=2, ensure_ascii=False)}
```

Return only Agent A JSON."""


def final_prompt(context_json: str, proposal: Mapping[str, Any], critique: Mapping[str, Any]) -> str:
    return f"""Agent B: write final consensus rules for Agent C to compile.

Only include rules that both agents accept. Ensure every accepted rule is
expressible using allowed observable fields from the candidate pool.

CONTEXT_JSON:
```json
{context_json}
```

FINAL_AGENT_A_JSON:
```json
{json.dumps(dict(proposal), indent=2, ensure_ascii=False)}
```

FINAL_AGENT_B_JSON:
```json
{json.dumps(dict(critique), indent=2, ensure_ascii=False)}
```

Return only final consensus JSON."""


def valid_shape(parsed: Any, role: str) -> bool:
    if not isinstance(parsed, Mapping):
        return False
    if role == "rule_agent_a":
        return parsed.get("agent") == "A" and isinstance(parsed.get("rules"), list)
    if role == "rule_agent_b":
        return parsed.get("agent") == "B" and isinstance(parsed.get("required_revisions"), list)
    if role == "rule_consensus":
        return parsed.get("status") == "consensus" and isinstance(parsed.get("accepted_rules"), list)
    return True


def retry_prompt(role: str, original_user: str, invalid_output: str, error: str) -> str:
    return f"""Your previous output for role `{role}` was not valid JSON or did
not match the required shape. Return exactly one valid JSON object.

STRICT_METRIC_GUARD:
{STRICT_SUN_NOTE}

PARSER_OR_SHAPE_ERROR:
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


def agreement_reached(proposal: Mapping[str, Any], critique: Mapping[str, Any]) -> bool:
    if bool(proposal.get("concede")) or bool(critique.get("concede")):
        return True
    return bool(proposal.get("agree")) and bool(critique.get("agree"))


def run_debate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    memory_dir = (root / args.memory_dir).resolve()
    state = read_json(Path(args.state), {})
    if not isinstance(state, Mapping):
        state = {}
    compact: Mapping[str, Any]
    compact_path = Path(args.compact_memory) if args.compact_memory else None
    if compact_path and compact_path.exists():
        raw = read_json(compact_path, {})
        compact = raw if isinstance(raw, Mapping) else {}
    else:
        compact = build_compact_memory(state, memory_dir=memory_dir)

    conflict = None
    if args.scorer_conflict_report:
        raw = read_json(Path(args.scorer_conflict_report), None)
        if isinstance(raw, Mapping):
            conflict = dict(raw)
    context_json = prompt_context(state=state, compact_memory=compact, scorer_conflict_report=conflict)

    proposer = ResponsesClient(
        LLMConfig.from_env(dotenv=args.dotenv, role="RULE_PROPOSER", model=args.model, temperature=args.temperature, max_tokens=args.max_tokens),
        log_dir=args.llm_log_dir,
    )
    critic = ResponsesClient(
        LLMConfig.from_env(dotenv=args.dotenv, role="RULE_CRITIC", model=args.model, temperature=args.temperature, max_tokens=args.max_tokens),
        log_dir=args.llm_log_dir,
    )

    artifacts: list[dict[str, Any]] = []
    proposal = call_json(
        proposer,
        system=AGENT_A_SYSTEM,
        user=proposal_prompt(context_json),
        metadata={"role": "rule_agent_a", "round": args.round, "cycle": 1},
    )
    artifacts.append({"role": "A", "cycle": 1, "payload": proposal})
    critique: dict[str, Any] = {}
    consensus: dict[str, Any] | None = None

    for cycle in range(1, max(1, args.max_dialogue_rounds) + 1):
        critique = call_json(
            critic,
            system=AGENT_B_SYSTEM,
            user=critique_prompt(context_json, proposal, cycle),
            metadata={"role": "rule_agent_b", "round": args.round, "cycle": cycle},
        )
        artifacts.append({"role": "B", "cycle": cycle, "payload": critique})
        if agreement_reached(proposal, critique):
            consensus = call_json(
                critic,
                system=AGENT_B_SYSTEM,
                user=final_prompt(context_json, proposal, critique),
                metadata={"role": "rule_consensus", "round": args.round, "cycle": cycle},
            )
            break
        if cycle >= args.max_dialogue_rounds:
            break
        proposal = call_json(
            proposer,
            system=AGENT_A_SYSTEM,
            user=revision_prompt(context_json, proposal, critique, cycle + 1),
            metadata={"role": "rule_agent_a", "round": args.round, "cycle": cycle + 1},
        )
        artifacts.append({"role": "A", "cycle": cycle + 1, "payload": proposal})

    if consensus is None:
        return {
            "status": "unresolved",
            "created_at_utc": utc_now(),
            "round": args.round,
            "dialogue": artifacts,
            "reason": "A/B reached max_dialogue_rounds without explicit consensus.",
        }
    consensus["created_at_utc"] = utc_now()
    consensus["round"] = args.round
    consensus["dialogue"] = artifacts
    return consensus


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_debate(args)
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
