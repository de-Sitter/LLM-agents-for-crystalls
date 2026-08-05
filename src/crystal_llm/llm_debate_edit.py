"""LLM A/B debate for one executable structure edit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pymatgen.core import Structure

from crystal_llm.experience import compact_experience, load_experiences
from crystal_llm.llm_client import LLMConfig, LLMError, ResponsesClient, extract_json_object
from crystal_llm.memory import STRICT_SUN_NOTE, read_json, write_json
from crystal_llm.structure_edit import EDIT_SCHEMA_DESCRIPTION, apply_edit, edit_to_json_string, structure_summary

MAX_JSON_REPAIR_ATTEMPTS = 2


AGENT_A_EDIT_SYSTEM = """You are Agent A in a single-structure crystal evolution loop.
Your job is to propose one small executable edit to the current crystal.

Use all historical information: prior structures, edits, validation failures,
evaluator results, positive experiences, and negative experiences. Your proposal
must be narrow and justified. Do not claim success unless e_hull < 0 and novelty
conditions are met.

Return JSON only:
{
  "agent": "A",
  "agree": false,
  "analysis": "...",
  "proposed_edit": { ... executable edit JSON ... },
  "expected_effect": "...",
  "risks": ["..."],
  "experience_used": ["..."]
}
"""


AGENT_B_EDIT_SYSTEM = """You are Agent B, the critic and final editor in a
single-structure crystal evolution loop.

You must challenge Agent A's proposal for metric confusion, overgeneralization,
invalid chemistry, invalid geometry, excessive edits, and conflicts with
negative experience. If the edit is acceptable, explicitly agree. If not, revise
it into a safer executable edit or reject it with noop.

Return JSON only. During critique cycles:
{
  "agent": "B",
  "agree": true | false,
  "critique": "...",
  "revised_edit": { ... executable edit JSON ... },
  "required_changes": ["..."],
  "risk_flags": ["..."]
}

For the final answer, return JSON only:
{
  "agent": "B",
  "agree": true,
  "consensus_summary": "...",
  "final_edit": { ... executable edit JSON ... },
  "final_edit_json": "{\\"op\\":\\"...\\", ...}",
  "safety_notes": ["..."]
}
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM A/B debate for one structure edit.")
    parser.add_argument("--state", required=True, help="Single-structure evolution state JSON.")
    parser.add_argument("--structure", required=True, help="Current single-structure JSON file.")
    parser.add_argument("--output", required=True, help="Edit decision artifact JSON.")
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--memory-dir", default="memory", help="Memory directory.")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--debate-rounds", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--llm-log-dir", default=None)
    parser.add_argument("--max-sites", type=int, default=80)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_structure(path: Path) -> Structure:
    data = read_json(path, None)
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"{path} must contain exactly one structure")
        data = data[0]
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a pymatgen Structure dictionary")
    return Structure.from_dict(dict(data))


def slim_state(state: Mapping[str, Any], *, max_rounds: int = 30) -> dict[str, Any]:
    history = state.get("history", [])
    if not isinstance(history, list):
        history = []
    return {
        "schema_version": state.get("schema_version"),
        "status": state.get("status"),
        "success": state.get("success"),
        "current_round": state.get("current_round"),
        "best_round": state.get("best_round"),
        "best_e_hull": state.get("best_e_hull"),
        "best_structure_path": state.get("best_structure_path"),
        "history": history[-max_rounds:],
    }


def prompt_context(
    *,
    state: Mapping[str, Any],
    structure: Structure,
    experiences: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    context = {
        "strict_sun_definition": STRICT_SUN_NOTE,
        "edit_schema": EDIT_SCHEMA_DESCRIPTION,
        "current_structure_summary": structure_summary(structure),
        "evolution_state": slim_state(state),
        "active_positive_experience": [
            compact_experience(record)
            for record in experiences.get("positive", [])
        ],
        "active_negative_experience": [
            compact_experience(record)
            for record in experiences.get("negative", [])
        ],
        "instructions": [
            "Use the complete evolution history, not only the current structure.",
            "Prefer one small edit per round.",
            "The program will run the full evaluator after the edit.",
            "If all non-noop edits look unsafe, output noop with a reason.",
            "Never count e_hull >= 0 as SUN.",
        ],
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def proposal_prompt(context_json: str) -> str:
    return f"""Propose the next executable structure edit.

CONTEXT_JSON:
```json
{context_json}
```

Return only Agent A JSON."""


def critique_prompt(context_json: str, proposal: Mapping[str, Any], cycle: int) -> str:
    return f"""Critique or revise Agent A's edit proposal for cycle {cycle}.

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
    return f"""Revise Agent A's proposal after Agent B cycle {cycle}.

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

Return only Agent A JSON. Set agree=true only if you explicitly accept Agent B's revised edit."""


def final_prompt(
    context_json: str,
    proposal: Mapping[str, Any],
    critique: Mapping[str, Any],
    executable_check: Mapping[str, Any],
) -> str:
    return f"""Produce the final agreed edit. Agent B is responsible for the final
machine-readable edit JSON string.

The final edit must be executable under this schema:
```text
{EDIT_SCHEMA_DESCRIPTION}
```

CONTEXT_JSON:
```json
{context_json}
```

FINAL_AGENT_A_JSON:
```json
{json.dumps(dict(proposal), ensure_ascii=False, indent=2)}
```

FINAL_AGENT_B_CRITIQUE_JSON:
```json
{json.dumps(dict(critique), ensure_ascii=False, indent=2)}
```

LOCAL_EXECUTABILITY_CHECK_FOR_CURRENT_B_EDIT:
```json
{json.dumps(dict(executable_check), ensure_ascii=False, indent=2)}
```

Return only final Agent B JSON."""


def required_shape_for_role(role: str) -> str:
    if role == "edit_agent_a":
        return (
            '{"agent":"A","agree":false,"analysis":"...","proposed_edit":{"op":"..."},'
            '"expected_effect":"...","risks":[],"experience_used":[]}'
        )
    if role == "edit_agent_b":
        return (
            '{"agent":"B","agree":true,"critique":"...","revised_edit":{"op":"..."},'
            '"required_changes":[],"risk_flags":[]}'
        )
    if role == "edit_agent_b_final":
        return (
            '{"agent":"B","agree":true,"consensus_summary":"...",'
            '"final_edit":{"op":"..."},"final_edit_json":"{\\"op\\":\\"...\\"}",'
            '"safety_notes":[]}'
        )
    return "{}"


def valid_shape_for_role(parsed: Any, role: str) -> bool:
    if not isinstance(parsed, Mapping):
        return False
    if role == "edit_agent_a":
        return isinstance(parsed.get("proposed_edit"), Mapping)
    if role == "edit_agent_b":
        return isinstance(candidate_edit_from_b(parsed), dict)
    if role == "edit_agent_b_final":
        try:
            final_edit_from_payload(parsed)
        except Exception:
            return False
        return parsed.get("agent") == "B" and parsed.get("agree") is True
    return True


def retry_prompt(role: str, original_user: str, invalid_output: str, error: str) -> str:
    return f"""Your previous output for role `{role}` could not be parsed or did not match the required JSON shape.
Revise your own previous answer into exactly one valid JSON object. Preserve the
same intended edit unless the error shows it is not executable. Do not add
markdown, prose, comments, or trailing commas.

Strict metric guard: {STRICT_SUN_NOTE}

Executable edit schema:
```text
{EDIT_SCHEMA_DESCRIPTION}
```

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


def transient_llm_retry_prompt(role: str, original_user: str, error: str) -> str:
    return f"""The previous LLM call for role `{role}` failed before the program could extract assistant text.
Retry the original task now and return exactly one valid JSON object. Do not add
markdown, prose, comments, or trailing commas.

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


def call_json(client: ResponsesClient, *, system: str, user: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
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
            raise ValueError(f"LLM call for {role} failed after Agent retry: {last_error}") from exc
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


def call_final_json(
    client: ResponsesClient,
    *,
    system: str,
    user: str,
    metadata: Mapping[str, Any],
    structure: Structure,
    max_sites: int,
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
            raise ValueError(f"LLM call for {role} failed after Agent retry: {last_error}") from exc
        try:
            parsed = extract_json_object(text)
            if not isinstance(parsed, Mapping):
                raise ValueError(f"LLM output for {role} must be a JSON object, got {type(parsed).__name__}")
            if not valid_shape_for_role(parsed, role):
                raise ValueError(f"LLM output for {role} did not match required shape")
            final_edit = final_edit_from_payload(parsed)
            final_check = apply_edit(structure, final_edit, max_sites=max_sites)
            if not final_check.ok:
                raise ValueError(
                    "final_edit is not executable: "
                    f"error={final_check.error}; validation_reasons={final_check.validation_reasons}"
                )
            return dict(parsed)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < MAX_JSON_REPAIR_ATTEMPTS:
            prompt = retry_prompt(role, user, last_text, last_error)
    raise ValueError(f"LLM output for {role} could not be repaired after Agent retry: {last_error}")



def candidate_edit_from_b(payload: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("revised_edit", "final_edit", "proposed_edit"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    final_edit_json = payload.get("final_edit_json")
    if isinstance(final_edit_json, str) and final_edit_json.strip():
        parsed = extract_json_object(final_edit_json)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {"op": "noop", "reason": "Agent B did not provide an executable edit object."}


def final_edit_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("final_edit"), Mapping):
        return dict(payload["final_edit"])
    final_edit_json = payload.get("final_edit_json")
    if isinstance(final_edit_json, str) and final_edit_json.strip():
        parsed = extract_json_object(final_edit_json)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise ValueError("final Agent B payload did not include final_edit or final_edit_json")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    memory_dir = (root / args.memory_dir).resolve()
    state_path = Path(args.state)
    structure_path = Path(args.structure)
    output_path = Path(args.output)
    state = read_json(state_path, {})
    if not isinstance(state, Mapping):
        state = {}
    structure = load_structure(structure_path)
    experiences = load_experiences(memory_dir, include_inactive=False)
    context_json = prompt_context(state=state, structure=structure, experiences=experiences)

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

    artifacts: list[dict[str, Any]] = []
    proposal = call_json(
        proposer,
        system=AGENT_A_EDIT_SYSTEM,
        user=proposal_prompt(context_json),
        metadata={"role": "edit_agent_a", "cycle": 1},
    )
    artifacts.append({"role": "A", "cycle": 1, "payload": proposal})
    critique: dict[str, Any] = {}

    cycles = max(1, args.debate_rounds)
    for cycle in range(1, cycles + 1):
        critique = call_json(
            critic,
            system=AGENT_B_EDIT_SYSTEM,
            user=critique_prompt(context_json, proposal, cycle),
            metadata={"role": "edit_agent_b", "cycle": cycle},
        )
        artifacts.append({"role": "B", "cycle": cycle, "payload": critique})
        if bool(proposal.get("agree")) and bool(critique.get("agree")):
            break
        if cycle < cycles:
            proposal = call_json(
                proposer,
                system=AGENT_A_EDIT_SYSTEM,
                user=revision_prompt(context_json, proposal, critique, cycle),
                metadata={"role": "edit_agent_a", "cycle": cycle + 1},
            )
            artifacts.append({"role": "A", "cycle": cycle + 1, "payload": proposal})

    candidate_edit = candidate_edit_from_b(critique)
    check = apply_edit(structure, candidate_edit, max_sites=args.max_sites)
    executable_check = {
        "ok": check.ok,
        "error": check.error,
        "validation_reasons": check.validation_reasons,
        "candidate_edit": candidate_edit,
    }
    final = call_final_json(
        critic,
        system=AGENT_B_EDIT_SYSTEM,
        user=final_prompt(context_json, proposal, critique, executable_check),
        metadata={"role": "edit_agent_b_final"},
        structure=structure,
        max_sites=args.max_sites,
    )
    final_edit = final_edit_from_payload(final)
    final_check = apply_edit(structure, final_edit, max_sites=args.max_sites)
    if not final_check.ok:
        fallback_edit = {
            "op": "noop",
            "reason": f"Final edit rejected by local executor: {final_check.error}; {final_check.validation_reasons}",
        }
        final_edit = fallback_edit
        final_check = apply_edit(structure, final_edit, max_sites=args.max_sites)

    payload = {
        "created_at_utc": utc_now(),
        "state": str(state_path),
        "structure": str(structure_path),
        "debate_rounds_requested": cycles,
        "strict_sun_definition": STRICT_SUN_NOTE,
        "artifacts": artifacts,
        "final_agent_b": final,
        "final_edit": final_edit,
        "final_edit_json": edit_to_json_string(final_edit),
        "local_executable": final_check.ok,
        "local_error": final_check.error,
        "local_validation_reasons": final_check.validation_reasons,
    }
    write_json(output_path, payload)
    print(f"wrote {output_path}")
    print(f"final_edit_json={payload['final_edit_json']}")
    print(f"local_executable={final_check.ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
