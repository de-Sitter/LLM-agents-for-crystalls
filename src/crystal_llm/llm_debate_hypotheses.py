"""A/B hypothesis debate for the hypothesis-first MVP."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from crystal_llm.compact_hypothesis_memory import build_compact_memory
from crystal_llm.hypothesis_schema import (
    generator_element_pools,
    schema_reference_json,
    template_interface,
    validate_consensus_payload,
)
from crystal_llm.llm_client import LLMConfig, LLMError, ResponsesClient, extract_json_object
from crystal_llm.memory import STRICT_SUN_NOTE, read_json, write_json


MAX_JSON_REPAIR_ATTEMPTS = 2


AGENT_A_SYSTEM = """You are Agent A in a hypothesis-first crystal search MVP.
Your job is to propose general, falsifiable hypotheses about what kinds of
prototype-constrained materials have a higher chance of being strict SUN.

Use the full historical context: prior hypotheses, generated formula probes,
all evaluator metrics, positive experiences, and negative experiences. Make
claims narrow enough that Agent C can instantiate them as generator-readable
formula probes.

Do not reveal hidden chain-of-thought. Provide concise evidence chains,
premises, counterevidence considered, assumptions, and a reasoning summary.

Return JSON only with:
{
  "agent": "A",
  "agree": false,
  "concede": false,
  "hypotheses": [
    {
      "id": "h001",
      "claim": "...",
      "rationale_summary": "...",
      "evidence_chain": [
        {"premise": "...", "observation": "...", "implication": "...", "confidence": "low|medium|high"}
      ],
      "target_templates": ["perovskite"],
      "chemical_constraints": {
        "preferred_anions": ["O", "F"],
        "role_preferences": {"A": ["K", "Rb", "Cs"], "B": ["Fe", "Co"]},
        "oxidation_state_preferences": {"A": [1], "B": [2, 4], "X": [-1, -2]},
        "avoid_elements": []
      },
      "falsifiable_predictions": ["..."],
      "scope": "...",
      "risk_notes": ["..."]
    }
  ],
  "response_to_b": "...",
  "required_b_revisions": []
}
"""


AGENT_B_SYSTEM = """You are Agent B, the rigorous critic in a hypothesis-first
crystal search MVP.

Your job is to attack Agent A's hypotheses using materials knowledge, generator
constraints, prior evaluator outcomes, and metric discipline. You should not
object blindly: every objection must cite a premise, implication, relevant
counterexample, or generator/evaluator constraint. Do not reveal hidden
chain-of-thought. Provide evidence chains and reasoning summaries only.

If the hypotheses are acceptable for Agent C to materialize, explicitly agree.
If they are not acceptable, list required revisions. You may concede if Agent A
has answered the critique.

Return JSON only during debate:
{
  "agent": "B",
  "agree": true | false,
  "concede": false,
  "critique_summary": "...",
  "evidence_chain": [
    {"premise": "...", "implication": "...", "confidence": "low|medium|high"}
  ],
  "counterexamples": ["..."],
  "required_revisions": ["..."],
  "accepted_hypothesis_ids": ["h001"],
  "rejected_hypothesis_ids": ["h002"],
  "risk_flags": ["..."]
}

When asked for final consensus, return JSON only:
{
  "status": "consensus",
  "conceded_by": "A|B|none",
  "accepted_hypotheses": [
    {
      "id": "h001",
      "claim": "...",
      "rationale_summary": "...",
      "evidence_chain": [...],
      "target_templates": ["perovskite"],
      "chemical_constraints": {...},
      "falsifiable_predictions": ["..."],
      "scope": "...",
      "risk_notes": ["..."]
    }
  ],
  "rejected_hypotheses": [
    {"id": "h002", "claim": "...", "rejection_reason": "..."}
  ],
  "consensus_summary": "...",
  "materialization_constraints": {
    "candidate_count": 10,
    "allowed_templates": ["perovskite", "spinel"],
    "minimum_template_diversity": 2
  }
}
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agent A/B debate for general crystal-generation hypotheses.")
    parser.add_argument("--state", required=True, help="Hypothesis MVP state JSON.")
    parser.add_argument("--output", required=True, help="Consensus or unresolved debate artifact JSON.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--memory-dir", default="memory")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--max-dialogue-rounds", type=int, default=100)
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--min-template-diversity", type=int, default=2)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--llm-log-dir", default=None)
    parser.add_argument("--conflict-report", default=None, help="Optional Agent C conflict report JSON to resume debate.")
    parser.add_argument(
        "--compact-memory",
        default=None,
        help="Deterministic compact memory JSON. If omitted or missing, it is built from state.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slim_state(state: Mapping[str, Any], *, max_rounds: int = 0) -> dict[str, Any]:
    history = state.get("history", [])
    if not isinstance(history, list):
        history = []
    history_payload = history[-max_rounds:] if max_rounds > 0 else []
    unresolved = state.get("unresolved_debates", [])
    if not isinstance(unresolved, list):
        unresolved = []
    return {
        "schema_version": state.get("schema_version"),
        "status": state.get("status"),
        "success": state.get("success"),
        "current_round": state.get("current_round"),
        "best_round": state.get("best_round"),
        "best_e_hull": state.get("best_e_hull"),
        "best_formula": state.get("best_formula"),
        "best_template": state.get("best_template"),
        "history": history_payload,
        "unresolved_debates": unresolved[-5:],
    }


def prompt_context(
    *,
    state: Mapping[str, Any],
    compact_memory: Mapping[str, Any],
    target_count: int,
    min_template_diversity: int,
    conflict_report: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "strict_sun_definition": STRICT_SUN_NOTE,
        "mvp_constraints": {
            "candidate_count": target_count,
            "allowed_templates": list(template_interface()),
            "minimum_template_diversity": min_template_diversity,
            "agent_c_output_contract": "Agent C will instantiate consensus as formula_probes only.",
            "generator_interface": schema_reference_json(target_count),
        },
        "template_interface": template_interface(),
        "generator_element_pools": generator_element_pools(),
        "state": slim_state(state),
        "compact_memory": dict(compact_memory),
        "agent_c_conflict_report": conflict_report,
        "instructions": [
            "Propose general hypotheses first; do not output concrete formula_probes here.",
            "Every accepted hypothesis must be specific enough for Agent C to instantiate.",
            "Respect generator role names and oxidation-state pools.",
            "Use compact_memory as the authoritative historical context; do not ask for full raw logs.",
            "Do not exactly replay formulas in compact_memory.negative_constraints.forbidden_exact_repeat_formulas.",
            "Never count e_hull >= 0 as SUN.",
            "Use evidence chains and reasoning summaries; do not expose hidden chain-of-thought.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def proposal_prompt(context_json: str) -> str:
    return f"""Agent A: propose or revise the next general hypotheses.

CONTEXT_JSON:
```json
{context_json}
```

Return only Agent A JSON."""


def critique_prompt(context_json: str, proposal: Mapping[str, Any], cycle: int) -> str:
    return f"""Agent B: critique Agent A's hypotheses for debate cycle {cycle}.

CONTEXT_JSON:
```json
{context_json}
```

AGENT_A_JSON:
```json
{json.dumps(dict(proposal), ensure_ascii=False, indent=2)}
```

Return only Agent B JSON."""


def revision_prompt(context_json: str, proposal: Mapping[str, Any], critique: Mapping[str, Any], cycle: int) -> str:
    return f"""Agent A: revise after Agent B critique in cycle {cycle}.

If Agent B is correct, narrow or remove the affected hypotheses. If Agent B is
wrong, answer with evidence summaries and keep the hypothesis only if it remains
falsifiable and materializable.

CONTEXT_JSON:
```json
{context_json}
```

PREVIOUS_AGENT_A_JSON:
```json
{json.dumps(dict(proposal), ensure_ascii=False, indent=2)}
```

AGENT_B_JSON:
```json
{json.dumps(dict(critique), ensure_ascii=False, indent=2)}
```

Return only Agent A JSON."""


def final_prompt(
    context_json: str,
    proposal: Mapping[str, Any],
    critique: Mapping[str, Any],
    validation_errors: Sequence[str] | None = None,
) -> str:
    errors_block = ""
    if validation_errors:
        errors_block = f"""
LOCAL_CONSENSUS_SCHEMA_ERRORS:
```json
{json.dumps(list(validation_errors), ensure_ascii=False, indent=2)}
```
"""
    return f"""Agent B: write the final consensus for Agent C.

Only include hypotheses that both agents can accept after the debate. If a
hypothesis was narrowed, write the narrowed version. The final consensus must be
valid under the schema and usable by Agent C to generate exactly 10 probes.

CONTEXT_JSON:
```json
{context_json}
```

FINAL_AGENT_A_JSON:
```json
{json.dumps(dict(proposal), ensure_ascii=False, indent=2)}
```

FINAL_AGENT_B_JSON:
```json
{json.dumps(dict(critique), ensure_ascii=False, indent=2)}
```
{errors_block}
Return only final consensus JSON."""


def required_shape_for_role(role: str) -> str:
    if role == "hypothesis_agent_a":
        return (
            '{"agent":"A","agree":false,"concede":false,"hypotheses":[{"id":"h001",'
            '"claim":"...","rationale_summary":"...","evidence_chain":[],'
            '"target_templates":["perovskite"],"chemical_constraints":{},'
            '"falsifiable_predictions":[],"scope":"...","risk_notes":[]}],'
            '"response_to_b":"...","required_b_revisions":[]}'
        )
    if role == "hypothesis_agent_b":
        return (
            '{"agent":"B","agree":true,"concede":false,"critique_summary":"...",'
            '"evidence_chain":[],"counterexamples":[],"required_revisions":[],'
            '"accepted_hypothesis_ids":[],"rejected_hypothesis_ids":[],"risk_flags":[]}'
        )
    if role == "hypothesis_consensus":
        return (
            '{"status":"consensus","conceded_by":"A|B|none","accepted_hypotheses":[...],'
            '"rejected_hypotheses":[],"consensus_summary":"...",'
            '"materialization_constraints":{"candidate_count":10,'
            '"allowed_templates":["perovskite"],"minimum_template_diversity":2}}'
        )
    return "{}"


def valid_shape_for_role(parsed: Any, role: str, target_count: int) -> bool:
    if not isinstance(parsed, Mapping):
        return False
    if role == "hypothesis_agent_a":
        return parsed.get("agent") == "A" and isinstance(parsed.get("hypotheses"), list)
    if role == "hypothesis_agent_b":
        return (
            parsed.get("agent") == "B"
            and isinstance(parsed.get("required_revisions"), list)
            and isinstance(parsed.get("counterexamples"), list)
        )
    if role == "hypothesis_consensus":
        return not validate_consensus_payload(parsed, target_count=target_count)
    return True


def retry_prompt(role: str, original_user: str, invalid_output: str, error: str, target_count: int) -> str:
    return f"""Your previous output for role `{role}` could not be parsed or did
not match the required JSON shape. Revise your own previous answer into exactly
one valid JSON object. Preserve substantive hypotheses and critiques unless the
error identifies a schema/generator incompatibility.

Strict metric guard: {STRICT_SUN_NOTE}

REQUIRED_JSON_SHAPE:
```json
{required_shape_for_role(role)}
```

LOCAL_SCHEMA_REFERENCE:
```json
{schema_reference_json(target_count)}
```

PARSER_OR_VALIDATION_ERROR:
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

Strict metric guard: {STRICT_SUN_NOTE}

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
    role = str(metadata.get("role", ""))
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
            parsed = None
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if not isinstance(parsed, Mapping):
                last_error = f"LLM output for {role} must be a JSON object, got {type(parsed).__name__}"
            elif not valid_shape_for_role(parsed, role, target_count):
                if role == "hypothesis_consensus":
                    errors = validate_consensus_payload(parsed, target_count=target_count)
                    last_error = "consensus schema errors: " + "; ".join(errors)
                else:
                    last_error = f"LLM output for {role} did not match required shape"
            else:
                return dict(parsed)
        if attempt < MAX_JSON_REPAIR_ATTEMPTS:
            prompt = retry_prompt(role, user, last_text, last_error, target_count)
    raise ValueError(f"LLM output for {role} could not be repaired after retry: {last_error}")


def agreement_reached(proposal: Mapping[str, Any], critique: Mapping[str, Any]) -> bool:
    if bool(proposal.get("concede")) or bool(critique.get("concede")):
        return True
    return bool(proposal.get("agree")) and bool(critique.get("agree"))


def run_debate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    memory_dir = (root / args.memory_dir).resolve()
    state_path = Path(args.state)
    state = read_json(state_path, {})
    if not isinstance(state, Mapping):
        state = {}
    compact_memory: Mapping[str, Any]
    compact_path = Path(args.compact_memory) if args.compact_memory else None
    if compact_path and compact_path.exists():
        raw_compact = read_json(compact_path, {})
        compact_memory = raw_compact if isinstance(raw_compact, Mapping) else {}
    else:
        compact_memory = build_compact_memory(state, memory_dir=memory_dir)
    conflict_report = None
    if args.conflict_report:
        raw = read_json(Path(args.conflict_report), None)
        if isinstance(raw, Mapping):
            conflict_report = dict(raw)
    context_json = prompt_context(
        state=state,
        compact_memory=compact_memory,
        target_count=args.target_count,
        min_template_diversity=args.min_template_diversity,
        conflict_report=conflict_report,
    )

    proposer = ResponsesClient(
        LLMConfig.from_env(
            dotenv=args.dotenv,
            role="HYPOTHESIS_PROPOSER",
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        log_dir=args.llm_log_dir,
    )
    critic = ResponsesClient(
        LLMConfig.from_env(
            dotenv=args.dotenv,
            role="HYPOTHESIS_CRITIC",
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        log_dir=args.llm_log_dir,
    )

    artifacts: list[dict[str, Any]] = []
    proposal = call_json(
        proposer,
        system=AGENT_A_SYSTEM,
        user=proposal_prompt(context_json),
        metadata={"role": "hypothesis_agent_a", "round": args.round, "cycle": 1},
        target_count=args.target_count,
    )
    artifacts.append({"role": "A", "cycle": 1, "payload": proposal})
    critique: dict[str, Any] = {}
    consensus: dict[str, Any] | None = None
    max_rounds = max(1, args.max_dialogue_rounds)

    for cycle in range(1, max_rounds + 1):
        critique = call_json(
            critic,
            system=AGENT_B_SYSTEM,
            user=critique_prompt(context_json, proposal, cycle),
            metadata={"role": "hypothesis_agent_b", "round": args.round, "cycle": cycle},
            target_count=args.target_count,
        )
        artifacts.append({"role": "B", "cycle": cycle, "payload": critique})

        if agreement_reached(proposal, critique):
            consensus = call_json(
                critic,
                system=AGENT_B_SYSTEM,
                user=final_prompt(context_json, proposal, critique),
                metadata={"role": "hypothesis_consensus", "round": args.round, "cycle": cycle},
                target_count=args.target_count,
            )
            artifacts.append({"role": "B", "cycle": cycle, "payload": consensus, "final": True})
            break

        if cycle < max_rounds:
            proposal = call_json(
                proposer,
                system=AGENT_A_SYSTEM,
                user=revision_prompt(context_json, proposal, critique, cycle),
                metadata={"role": "hypothesis_agent_a", "round": args.round, "cycle": cycle + 1},
                target_count=args.target_count,
            )
            artifacts.append({"role": "A", "cycle": cycle + 1, "payload": proposal})

    if consensus is None:
        return {
            "created_at_utc": utc_now(),
            "status": "unresolved",
            "round": args.round,
            "max_dialogue_rounds": max_rounds,
            "strict_sun_definition": STRICT_SUN_NOTE,
            "reason": "A/B did not reach explicit agreement or concession before max_dialogue_rounds.",
            "context": json.loads(context_json),
            "artifacts": artifacts,
        }

    errors = validate_consensus_payload(consensus, target_count=args.target_count)
    if errors:
        consensus = call_json(
            critic,
            system=AGENT_B_SYSTEM,
            user=final_prompt(context_json, proposal, critique, errors),
            metadata={"role": "hypothesis_consensus", "round": args.round, "cycle": "repair_final"},
            target_count=args.target_count,
        )
        artifacts.append({"role": "B", "cycle": "repair_final", "payload": consensus, "final": True})

    return {
        "created_at_utc": utc_now(),
        "status": "consensus",
        "round": args.round,
        "max_dialogue_rounds": max_rounds,
        "strict_sun_definition": STRICT_SUN_NOTE,
        "context": json.loads(context_json),
        "artifacts": artifacts,
        "consensus": consensus,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output)
    payload = run_debate(args)
    write_json(output_path, payload)
    print(f"wrote {output_path}")
    print(f"status={payload.get('status')}")
    return 0 if payload.get("status") in {"consensus", "unresolved"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
