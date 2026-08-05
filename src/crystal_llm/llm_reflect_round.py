"""Run two-agent LLM reflection and write versioned experience JSONL files."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from crystal_llm.build_round_context import build_context_payload, context_markdown, latest_completed_round
from crystal_llm.experience import experience_path, merge_new_experiences
from crystal_llm.llm_client import LLMConfig, ResponsesClient, extract_json_object
from crystal_llm.memory import STRICT_SUN_NOTE, round_label, write_json


ALLOWED_CONFIDENCE = {"low", "medium-low", "medium", "medium-high", "high"}


AGENT_A_SYSTEM = """You are Agent A, the proposer in a crystal-generation research loop.
Your job is to read the complete historical context and propose detailed positive
and negative experiences for future rounds.

Rules:
- The strict SUN metric is e_hull < 0 only. e_hull < 0.03 and e_hull < 0.10 are diagnostics, not success.
- Preserve natural-language reasoning. Do not collapse insights into broad element boosts.
- Every proposed experience must explain the evidence, reasoning, counterevidence, scope, confidence, and future action.
- Separate replay preservation from genuine discovery/generalization.
- Prefer narrow scope when evidence is exact-formula, template-, role-, or branch-specific.

Return JSON only with:
{
  "agent": "A",
  "positive_experiences": [ ... ],
  "negative_experiences": [ ... ],
  "open_questions": [ ... ]
}
"""


AGENT_B_SYSTEM = """You are Agent B, the critic. Your job is to attack Agent A's
claims using the same complete historical context.

For every proposed experience, look for counterexamples, leakage from replay,
metric confusion, overgeneralization, seed noise, and unsupported chemistry. You
may accept a claim, reject it, or narrow its scope.

Return JSON only with:
{
  "agent": "B",
  "critiques": [
    {
      "target_claim_or_key": "...",
      "verdict": "accept | reject | narrow | needs_more_evidence",
      "counterevidence": [...],
      "reasoning": "...",
      "proposed_revision": "..."
    }
  ],
  "global_concerns": [ ... ]
}
"""


REFEREE_SYSTEM = """You are the referee. You read the full historical context,
Agent A's proposals, and Agent B's critiques. Write only the experiences that
survive the debate.

Every final experience must include a non-empty detailed_reasoning field. The
detailed_reasoning must explain what specific round records, e_hull values,
accepted/rejected counterexamples, and metric cautions support the claim. Do not
leave it blank and do not merely repeat the claim.

Return JSON only with:
{
  "positive_experiences": [
    {
      "experience_key": "stable_snake_case_key",
      "claim": "detailed natural-language claim",
      "detailed_reasoning": "why this follows from the evidence",
      "evidence": [{"source": "...", "summary": "..."}],
      "counterevidence_considered": [{"source": "...", "summary": "..."}],
      "scope": "where this should and should not apply",
      "confidence": "low | medium | high",
      "actionability": "generator_prior | candidate_review | formula_probe | replay_only | do_not_use_yet",
      "recommended_action": "...",
      "do_not_generalize_to": ["..."],
      "supersedes": ["optional previous record ids"],
      "tags": ["..."]
    }
  ],
  "negative_experiences": [
    {
      "experience_key": "stable_snake_case_key",
      "claim": "pitfall to avoid",
      "detailed_reasoning": "why this is a pitfall",
      "evidence": [{"source": "...", "summary": "..."}],
      "counterevidence_considered": [{"source": "...", "summary": "..."}],
      "scope": "where this warning applies",
      "confidence": "low | medium | high",
      "actionability": "candidate_review | formula_probe | replay_only | do_not_use_yet",
      "recommended_action": "...",
      "do_not_generalize_to": ["..."],
      "supersedes": ["optional previous record ids"],
      "tags": ["..."]
    }
  ],
  "rejected_claims": [
    {"claim": "...", "reason": "..."}
  ],
  "consensus_summary": "brief summary"
}

Do not output unsupported experiences. Do not count e_hull >= 0 as SUN.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM A/B debate and write positive/negative experience JSONL.")
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument("--memory-dir", default="memory", help="Memory directory.")
    parser.add_argument("--round", type=int, default=None, help="Round being reflected on. Defaults to latest completed round.")
    parser.add_argument("--latest-round", type=int, default=None, help="Latest completed round to include in full history.")
    parser.add_argument("--start-round", type=int, default=1, help="First historical round to include.")
    parser.add_argument("--dotenv", default=".env", help="Environment file with LLM_BASE_URL/API key/model.")
    parser.add_argument("--model", default=None, help="Override model for all agents.")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max output tokens.")
    parser.add_argument("--debate-rounds", type=int, default=2, help="Number of A/B critique cycles before referee.")
    parser.add_argument("--include-structures", action="store_true", help="Include full structure dictionaries in context.")
    parser.add_argument("--context-output", default=None, help="Context JSON path.")
    parser.add_argument("--context-markdown-output", default=None, help="Context markdown path.")
    parser.add_argument("--debate-dir", default=None, help="Directory for debate artifacts.")
    parser.add_argument("--positive-output", default=None, help="Positive experience JSONL path.")
    parser.add_argument("--negative-output", default=None, help="Negative experience JSONL path.")
    parser.add_argument("--llm-log-dir", default=None, help="Directory for raw LLM request/response logs.")
    return parser.parse_args(argv)


def write_artifact(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def required_shape_for_role(role: str) -> str:
    if role == "agent_a":
        return (
            '{"agent":"A","positive_experiences":[],"negative_experiences":[],'
            '"open_questions":[]}'
        )
    if role == "agent_b":
        return (
            '{"agent":"B","critiques":[{"target_claim_or_key":"...",'
            '"verdict":"accept | reject | narrow | needs_more_evidence",'
            '"counterevidence":[],"reasoning":"...","proposed_revision":"..."}],'
            '"global_concerns":[]}'
        )
    if role == "referee":
        return (
            '{"positive_experiences":[],"negative_experiences":[],'
            '"rejected_claims":[],"consensus_summary":"..."} where every item in '
            'positive_experiences and negative_experiences has non-empty '
            'experience_key, claim, detailed_reasoning, evidence, scope, confidence, '
            'actionability, recommended_action, and do_not_generalize_to fields'
        )
    return "{}"


def valid_experience_record(item: Any) -> bool:
    if not isinstance(item, Mapping):
        return False
    required_text = ("experience_key", "claim", "detailed_reasoning", "recommended_action")
    for key in required_text:
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
    if role == "agent_a":
        return (
            isinstance(parsed.get("positive_experiences"), list)
            and isinstance(parsed.get("negative_experiences"), list)
            and isinstance(parsed.get("open_questions"), list)
        )
    if role == "agent_b":
        return isinstance(parsed.get("critiques"), list) and isinstance(parsed.get("global_concerns"), list)
    if role == "referee":
        positive = parsed.get("positive_experiences")
        negative = parsed.get("negative_experiences")
        return (
            isinstance(positive, list)
            and isinstance(negative, list)
            and isinstance(parsed.get("rejected_claims"), list)
            and all(valid_experience_record(item) for item in positive)
            and all(valid_experience_record(item) for item in negative)
        )
    return True


def call_json(
    client: ResponsesClient,
    *,
    system: str,
    user: str,
    artifact_path: Path,
    metadata: Mapping[str, Any],
) -> Any:
    text = client.complete_text(system=system, user=user, metadata=metadata)
    role = str(metadata.get("role", ""))
    repair_text: str | None = None
    try:
        parsed = extract_json_object(text)
    except Exception:
        parsed = None

    if not valid_shape_for_role(parsed, role):
        repair_prompt = f"""The previous LLM output is not valid executable JSON for role `{role}`.
Convert it into exactly one JSON object with the required shape. Preserve
substantive claims, critiques, evidence, counterevidence, and metric cautions.
Do not add prose or markdown.

Strict metric guard: {STRICT_SUN_NOTE}

REQUIRED_JSON_SHAPE:
```json
{required_shape_for_role(role)}
```

ORIGINAL_TASK_PROMPT:
```text
{user}
```

INVALID_OUTPUT:
```text
{text}
```
"""
        repair_text = client.complete_text(
            system=system,
            user=repair_prompt,
            metadata={**dict(metadata), "repair": True},
        )
        parsed = extract_json_object(repair_text)
        if not valid_shape_for_role(parsed, role):
            raise ValueError(f"LLM output for role {role} could not be repaired into required JSON shape")

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata),
        "raw_text": text,
        "repair_text": repair_text,
        "parsed": parsed,
    }
    write_artifact(artifact_path, payload)
    return payload["parsed"]


def context_for_prompt(context: Mapping[str, Any]) -> str:
    return json.dumps(context, ensure_ascii=False, indent=2)


def proposal_prompt(context_json: str) -> str:
    return f"""Read the complete historical context below and propose experiences.

Strict metric guard: {STRICT_SUN_NOTE}

COMPLETE_CONTEXT_JSON:
```json
{context_json}
```
"""


def critic_prompt(context_json: str, proposal: Mapping[str, Any], cycle: int) -> str:
    return f"""Critique Agent A cycle {cycle} proposals against the complete historical context.

Strict metric guard: {STRICT_SUN_NOTE}

AGENT_A_PROPOSAL_JSON:
```json
{json.dumps(proposal, ensure_ascii=False, indent=2)}
```

COMPLETE_CONTEXT_JSON:
```json
{context_json}
```
"""


def revision_prompt(context_json: str, proposal: Mapping[str, Any], critique: Mapping[str, Any], cycle: int) -> str:
    return f"""Revise Agent A proposals after Agent B cycle {cycle} critique.

Keep only claims that survive the critique, narrow overbroad claims, and make
evidence/counterevidence explicit.

PREVIOUS_AGENT_A_JSON:
```json
{json.dumps(proposal, ensure_ascii=False, indent=2)}
```

AGENT_B_CRITIQUE_JSON:
```json
{json.dumps(critique, ensure_ascii=False, indent=2)}
```

COMPLETE_CONTEXT_JSON:
```json
{context_json}
```
"""


def referee_prompt(context_json: str, proposal: Mapping[str, Any], critique: Mapping[str, Any]) -> str:
    return f"""Produce final consensus experiences from the debate.

Strict metric guard: {STRICT_SUN_NOTE}

For every final positive or negative experience:
- detailed_reasoning is mandatory and must be non-empty.
- It must explain the exact evidence path from historical records to the claim.
- It must state why counterevidence did not defeat the narrowed claim.
- It must not label e_hull >= 0 candidates as SUN.

FINAL_AGENT_A_JSON:
```json
{json.dumps(proposal, ensure_ascii=False, indent=2)}
```

FINAL_AGENT_B_JSON:
```json
{json.dumps(critique, ensure_ascii=False, indent=2)}
```

COMPLETE_CONTEXT_JSON:
```json
{context_json}
```
"""


def _as_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    memory_dir = (root / args.memory_dir).resolve()
    latest_round = latest_completed_round(root, memory_dir, args.latest_round)
    reflection_round = args.round if args.round is not None else latest_round
    label = round_label(reflection_round)
    debate_dir = Path(args.debate_dir) if args.debate_dir else memory_dir / "debates" / label
    llm_log_dir = Path(args.llm_log_dir) if args.llm_log_dir else debate_dir / "llm_calls"
    positive_path = Path(args.positive_output) if args.positive_output else experience_path(memory_dir, "positive")
    negative_path = Path(args.negative_output) if args.negative_output else experience_path(memory_dir, "negative")
    context_output = Path(args.context_output) if args.context_output else memory_dir / "round_contexts" / f"{label}.json"
    context_md_output = (
        Path(args.context_markdown_output)
        if args.context_markdown_output
        else memory_dir / "round_contexts" / f"{label}.md"
    )

    context = build_context_payload(
        root,
        memory_dir,
        reflection_round=reflection_round,
        latest_round=latest_round,
        start_round=args.start_round,
        include_structures=args.include_structures,
    )
    write_json(context_output, context)
    context_md_output.parent.mkdir(parents=True, exist_ok=True)
    context_md_output.write_text(context_markdown(context), encoding="utf-8")
    context_json = context_for_prompt(context)

    proposer = ResponsesClient(
        LLMConfig.from_env(
            dotenv=args.dotenv,
            role="PROPOSER",
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        log_dir=llm_log_dir,
    )
    critic = ResponsesClient(
        LLMConfig.from_env(
            dotenv=args.dotenv,
            role="CRITIC",
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        log_dir=llm_log_dir,
    )
    referee = ResponsesClient(
        LLMConfig.from_env(
            dotenv=args.dotenv,
            role="REFEREE",
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        log_dir=llm_log_dir,
    )

    proposal = call_json(
        proposer,
        system=AGENT_A_SYSTEM,
        user=proposal_prompt(context_json),
        artifact_path=debate_dir / "agent_a_cycle_01.json",
        metadata={"round": label, "role": "agent_a", "cycle": 1},
    )
    critique: Mapping[str, Any] = {}
    cycles = max(1, args.debate_rounds)
    for cycle in range(1, cycles + 1):
        critique = call_json(
            critic,
            system=AGENT_B_SYSTEM,
            user=critic_prompt(context_json, proposal, cycle),
            artifact_path=debate_dir / f"agent_b_cycle_{cycle:02d}.json",
            metadata={"round": label, "role": "agent_b", "cycle": cycle},
        )
        if cycle < cycles:
            proposal = call_json(
                proposer,
                system=AGENT_A_SYSTEM,
                user=revision_prompt(context_json, proposal, critique, cycle),
                artifact_path=debate_dir / f"agent_a_cycle_{cycle + 1:02d}.json",
                metadata={"round": label, "role": "agent_a", "cycle": cycle + 1},
            )

    consensus = call_json(
        referee,
        system=REFEREE_SYSTEM,
        user=referee_prompt(context_json, proposal, critique),
        artifact_path=debate_dir / "consensus.json",
        metadata={"round": label, "role": "referee"},
    )

    created_by = {
        "proposer_model": proposer.config.model,
        "critic_model": critic.config.model,
        "referee_model": referee.config.model,
        "debate_rounds": cycles,
    }
    positive_raw = []
    for item in _as_list(consensus.get("positive_experiences") if isinstance(consensus, Mapping) else None):
        payload = dict(item)
        payload["created_by"] = created_by
        positive_raw.append(payload)
    negative_raw = []
    for item in _as_list(consensus.get("negative_experiences") if isinstance(consensus, Mapping) else None):
        payload = dict(item)
        payload["created_by"] = created_by
        negative_raw.append(payload)

    positive_created = merge_new_experiences(
        positive_path,
        positive_raw,
        polarity="positive",
        round_number=reflection_round,
    )
    negative_created = merge_new_experiences(
        negative_path,
        negative_raw,
        polarity="negative",
        round_number=reflection_round,
    )

    report = {
        "round": label,
        "context": str(context_output),
        "debate_dir": str(debate_dir),
        "positive_output": str(positive_path),
        "negative_output": str(negative_path),
        "positive_created": [record["id"] for record in positive_created],
        "negative_created": [record["id"] for record in negative_created],
        "consensus_summary": consensus.get("consensus_summary") if isinstance(consensus, Mapping) else None,
    }
    write_artifact(debate_dir / "reflection_report.json", report)
    print(f"wrote context {context_output}")
    print(f"wrote debate artifacts {debate_dir}")
    print(f"positive_experiences_added={len(positive_created)}")
    print(f"negative_experiences_added={len(negative_created)}")
    print(f"positive_output={positive_path}")
    print(f"negative_output={negative_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
