"""LLM A/B reflection for a single-structure evolution round."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from crystal_llm.experience import merge_new_experiences
from crystal_llm.llm_client import LLMConfig, ResponsesClient, extract_json_object
from crystal_llm.memory import STRICT_SUN_NOTE, read_json, write_json

MAX_JSON_REPAIR_ATTEMPTS = 2
ALLOWED_CONFIDENCE = {"low", "medium-low", "medium", "medium-high", "high"}


AGENT_A_REFLECT_SYSTEM = """You are Agent A, the proposer for experience updates
in a single-structure crystal evolution loop.

Read the full evolution history, the executed edit, local validation, evaluator
results, and per-structure e_hull. Propose narrow positive and negative
experiences for future edits. Do not call a result SUN unless e_hull < 0 and
the evaluator reports both novelty conditions.

Return JSON only:
{
  "agent": "A",
  "positive_experiences": [ ... ],
  "negative_experiences": [ ... ],
  "open_questions": [ ... ]
}
"""


AGENT_B_REFLECT_SYSTEM = """You are Agent B, the critic and final referee for
single-structure evolution experience.

Critique Agent A for overgeneralization, exact-formula leakage, metric confusion,
and unsupported chemistry. Then write only experiences both agents can agree on.

Return JSON only:
{
  "agent": "B",
  "agree": true,
  "positive_experiences": [
    {
      "experience_key": "stable_snake_case_key",
      "claim": "...",
      "detailed_reasoning": "...",
      "evidence": [{"source":"...","summary":"..."}],
      "counterevidence_considered": [{"source":"...","summary":"..."}],
      "scope": "...",
      "confidence": "low | medium | high",
      "actionability": "structure_edit | candidate_review | generator_prior | do_not_use_yet",
      "recommended_action": "...",
      "do_not_generalize_to": ["..."],
      "tags": ["single_structure_evolution"]
    }
  ],
  "negative_experiences": [ ... same schema ... ],
  "rejected_claims": [{"claim":"...","reason":"..."}],
  "consensus_summary": "..."
}
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reflect on one single-structure evolution round.")
    parser.add_argument("--state", required=True, help="Evolution state JSON after evaluator analysis.")
    parser.add_argument("--round-dir", required=True, help="Evolution round directory.")
    parser.add_argument("--output", required=True, help="Reflection report JSON.")
    parser.add_argument("--memory-dir", default="memory")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--llm-log-dir", default=None)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slim_history(state: Mapping[str, Any], limit: int = 40) -> list[Any]:
    history = state.get("history", [])
    if not isinstance(history, list):
        return []
    return history[-limit:]


def build_context(state: Mapping[str, Any], round_dir: Path) -> dict[str, Any]:
    paths = {
        "edit_decision": round_dir / "edit_decision.json",
        "evaluation_result": round_dir / "results.json",
        "analysis_summary": round_dir / "analysis" / "summary.json",
        "ranked_e_hull": round_dir / "analysis" / "e_hull_ranked.csv",
    }
    context = {
        "strict_sun_definition": STRICT_SUN_NOTE,
        "state_summary": {
            "status": state.get("status"),
            "success": state.get("success"),
            "current_round": state.get("current_round"),
            "best_round": state.get("best_round"),
            "best_e_hull": state.get("best_e_hull"),
            "best_structure_path": state.get("best_structure_path"),
        },
        "latest_history": slim_history(state),
        "round_artifacts": {},
    }
    for key, path in paths.items():
        if path.exists() and path.suffix == ".json":
            context["round_artifacts"][key] = read_json(path, {})
        elif path.exists():
            context["round_artifacts"][key] = path.read_text(encoding="utf-8", errors="replace")[:20000]
    return context


def proposal_prompt(context: Mapping[str, Any]) -> str:
    return f"""Propose single-structure evolution experiences from this round.

CONTEXT_JSON:
```json
{json.dumps(dict(context), ensure_ascii=False, indent=2)}
```

Return only Agent A JSON."""


def final_prompt(context: Mapping[str, Any], proposal: Mapping[str, Any]) -> str:
    return f"""Critique Agent A and write final agreed experiences.

Strict metric guard: {STRICT_SUN_NOTE}

CONTEXT_JSON:
```json
{json.dumps(dict(context), ensure_ascii=False, indent=2)}
```

AGENT_A_JSON:
```json
{json.dumps(dict(proposal), ensure_ascii=False, indent=2)}
```

Return only Agent B final JSON."""


def required_shape_for_role(role: str) -> str:
    if role == "evolution_reflect_agent_a":
        return (
            '{"agent":"A","positive_experiences":[],"negative_experiences":[],'
            '"open_questions":[]}'
        )
    if role == "evolution_reflect_agent_b_final":
        return (
            '{"agent":"B","agree":true,"positive_experiences":[],"negative_experiences":[],'
            '"rejected_claims":[],"consensus_summary":"..."} where every item in '
            'positive_experiences and negative_experiences has non-empty '
            'experience_key, claim, detailed_reasoning, evidence, scope, confidence, '
            'actionability, recommended_action, and do_not_generalize_to fields'
        )
    return "{}"


def valid_experience_record(item: Any) -> bool:
    if not isinstance(item, Mapping):
        return False
    for key in ("experience_key", "claim", "detailed_reasoning", "recommended_action"):
        if not str(item.get(key) or "").strip():
            return False
    if not isinstance(item.get("evidence"), list) or not item["evidence"]:
        return False
    if not str(item.get("scope") or "").strip():
        return False
    confidence = str(item.get("confidence") or "").strip().lower().replace("_", "-").replace(" ", "-")
    if confidence not in ALLOWED_CONFIDENCE:
        return False
    if not str(item.get("actionability") or "").strip():
        return False
    return True


def valid_shape_for_role(parsed: Any, role: str) -> bool:
    if not isinstance(parsed, Mapping):
        return False
    if role == "evolution_reflect_agent_a":
        return (
            isinstance(parsed.get("positive_experiences"), list)
            and isinstance(parsed.get("negative_experiences"), list)
            and isinstance(parsed.get("open_questions"), list)
        )
    if role == "evolution_reflect_agent_b_final":
        positive = parsed.get("positive_experiences")
        negative = parsed.get("negative_experiences")
        return (
            parsed.get("agent") == "B"
            and parsed.get("agree") is True
            and isinstance(positive, list)
            and isinstance(negative, list)
            and isinstance(parsed.get("rejected_claims"), list)
            and str(parsed.get("consensus_summary") or "").strip()
            and all(valid_experience_record(item) for item in positive)
            and all(valid_experience_record(item) for item in negative)
        )
    return isinstance(parsed, Mapping)


def retry_prompt(role: str, original_user: str, invalid_output: str, error: str) -> str:
    return f"""Your previous output for role `{role}` could not be parsed or did not match the required JSON shape.
Revise your own previous answer into exactly one valid JSON object. Preserve the
substantive consensus, evidence, counterevidence, and metric cautions. Do not
add markdown, prose, comments, or trailing commas.

Strict metric guard: {STRICT_SUN_NOTE}

PARSER_OR_VALIDATION_ERROR:
```text
{error}
```

REQUIRED_JSON_SHAPE:
```json
{required_shape_for_role(role)}
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
        text = client.complete_text(
            system=system,
            user=prompt,
            metadata={**dict(metadata), "json_retry_attempt": attempt},
        )
        last_text = text
        try:
            parsed = extract_json_object(text)
        except Exception as exc:
            parsed = None
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if not isinstance(parsed, Mapping):
                last_error = f"LLM output for {role} must be a JSON object, got {type(parsed).__name__}"
            elif not valid_shape_for_role(parsed, role):
                last_error = f"LLM output for {role} did not match required shape"
            else:
                return dict(parsed)
        if attempt < MAX_JSON_REPAIR_ATTEMPTS:
            prompt = retry_prompt(role, user, last_text, last_error)
    raise ValueError(f"LLM output for {role} could not be repaired after Agent retry: {last_error}")


def _as_records(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    state_path = Path(args.state)
    round_dir = Path(args.round_dir)
    output_path = Path(args.output)
    memory_dir = Path(args.memory_dir)
    state = read_json(state_path, {})
    if not isinstance(state, Mapping):
        state = {}
    context = build_context(state, round_dir)

    proposer = ResponsesClient(
        LLMConfig.from_env(
            dotenv=args.dotenv,
            role="PROPOSER",
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        log_dir=args.llm_log_dir,
    )
    critic = ResponsesClient(
        LLMConfig.from_env(
            dotenv=args.dotenv,
            role="CRITIC",
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        log_dir=args.llm_log_dir,
    )

    proposal = call_json(
        proposer,
        system=AGENT_A_REFLECT_SYSTEM,
        user=proposal_prompt(context),
        metadata={"role": "evolution_reflect_agent_a"},
    )
    final = call_json(
        critic,
        system=AGENT_B_REFLECT_SYSTEM,
        user=final_prompt(context, proposal),
        metadata={"role": "evolution_reflect_agent_b_final"},
    )

    round_number = int(state.get("current_round") or 0)
    created_by = {
        "workflow": "single_structure_evolution_mvp",
        "proposer_model": proposer.config.model,
        "critic_model": critic.config.model,
    }
    positive_raw = []
    for record in _as_records(final.get("positive_experiences")):
        payload = dict(record)
        payload["created_by"] = created_by
        payload.setdefault("tags", [])
        positive_raw.append(payload)
    negative_raw = []
    for record in _as_records(final.get("negative_experiences")):
        payload = dict(record)
        payload["created_by"] = created_by
        payload.setdefault("tags", [])
        negative_raw.append(payload)

    positive_created = merge_new_experiences(
        memory_dir / "positive_experience.jsonl",
        positive_raw,
        polarity="positive",
        round_number=round_number,
    )
    negative_created = merge_new_experiences(
        memory_dir / "negative_experience.jsonl",
        negative_raw,
        polarity="negative",
        round_number=round_number,
    )

    report = {
        "created_at_utc": utc_now(),
        "state": str(state_path),
        "round_dir": str(round_dir),
        "agent_a": proposal,
        "agent_b_final": final,
        "positive_created": [record["id"] for record in positive_created],
        "negative_created": [record["id"] for record in negative_created],
        "consensus_summary": final.get("consensus_summary"),
    }
    write_json(output_path, report)
    print(f"wrote {output_path}")
    print(f"positive_experiences_added={len(positive_created)}")
    print(f"negative_experiences_added={len(negative_created)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
