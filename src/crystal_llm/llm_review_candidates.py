"""Review generated candidate structures with LLM natural-language experience."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pymatgen.core import Structure

from crystal_llm.experience import compact_experience, load_experiences
from crystal_llm.filters import (
    StructureDeduplicator,
    load_known_formulas,
    min_interatomic_distance,
    reduced_formula,
    validate_structure,
)
from crystal_llm.llm_client import LLMConfig, ResponsesClient, extract_json_object


REVIEWER_SYSTEM = """You are the LLM reviewer for candidate crystal structures.
You must review candidates one by one using the positive and negative experience
records. Your output controls which structures are evaluated.

Rules:
- You may decide pass, reject, or modify.
- If you modify a structure, only request a small micro-adjustment. Do not invent
  arbitrary coordinates, new sites, new species, or a new prototype.
- The only executable modification currently allowed is:
  {"operation": "scale_lattice", "scale_factor": number between 0.97 and 1.03}
- If you believe a larger change, substitution, or new prototype is needed,
  reject the candidate and record it as a suggestion in reasoning.
- Strict SUN means e_hull < 0 only; near-zero positive e_hull is not success.
- Prefer precise scope. Do not globally generalize exact formula experience.
- Keep the executable decision consistent with the reasoning. If your reasoning
  says the candidate directly matches a prior negative formula/prototype/family,
  is a known pitfall, should not receive ordinary priority, should not be
  promoted, or should be kept "if at all", the decision must be "reject" unless
  the prompt explicitly grants a diversity-probe override. No such override is
  granted by default.
- Direct negative exact-formula evidence outranks broad positive family evidence.
- Do not use "modify" for a chemical/prototype-level negative; lattice scaling
  is only for small geometry concerns.

Return JSON only with this exact top-level shape. The "decision" value must be
one of the three strings "pass", "reject", or "modify"; do not nest another
object under "decision".
{
  "candidate_id": "...",
  "decision": "pass | reject | modify",
  "reasoning": "detailed reason grounded in experience records",
  "positive_experience_used": ["experience ids"],
  "negative_experience_used": ["experience ids"],
  "risk_flags": ["..."],
  "modification": {
    "operation": "none | scale_lattice",
    "scale_factor": 1.0,
    "reason": "..."
  }
}
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-review generated candidate structures before evaluator input.")
    parser.add_argument("--candidate-pool", required=True, help="Generated candidate pool JSON.")
    parser.add_argument("--output", default="input.json", help="Reviewed final pymatgen Structure JSON.")
    parser.add_argument("--report", default=None, help="Review report JSON path.")
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument("--memory-dir", default="memory", help="Memory directory.")
    parser.add_argument("--positive-memory", default=None, help="Positive experience JSONL path; defaults to memory/positive_experience.jsonl.")
    parser.add_argument("--negative-memory", default=None, help="Negative experience JSONL path; defaults to memory/negative_experience.jsonl.")
    parser.add_argument("--target-count", type=int, default=10, help="Number of accepted structures to output.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Maximum candidates to ask LLM to review.")
    parser.add_argument("--dotenv", default=".env", help="Environment file with LLM config.")
    parser.add_argument("--model", default=None, help="Override LLM model.")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max output tokens.")
    parser.add_argument("--llm-log-dir", default=None, help="Directory for raw LLM call logs.")
    parser.add_argument("--decision-dir", default=None, help="Directory for per-candidate decision artifacts.")
    parser.add_argument("--max-sites", type=int, default=80, help="Maximum sites allowed after review.")
    parser.add_argument("--training-data", default=None, help="Optional training data path for known-composition filtering.")
    parser.add_argument("--allow-known-compositions", action="store_true", help="Allow formulas present in training data.")
    parser.add_argument("--min-scale", type=float, default=0.97, help="Minimum accepted lattice scale factor.")
    parser.add_argument("--max-scale", type=float, default=1.03, help="Maximum accepted lattice scale factor.")
    parser.add_argument("--allow-short-output", action="store_true", help="Write fewer than target-count if LLM rejects too many candidates.")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
    return records


def load_active_experience(
    memory_dir: Path,
    *,
    positive_path: Path | None = None,
    negative_path: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if positive_path or negative_path:
        positive = read_jsonl(positive_path) if positive_path else read_jsonl(memory_dir / "positive_experience.jsonl")
        negative = read_jsonl(negative_path) if negative_path else read_jsonl(memory_dir / "negative_experience.jsonl")
        positive = [record for record in positive if str(record.get("status", "active")).lower() in {"active", "supported", "tentative"}]
        negative = [record for record in negative if str(record.get("status", "active")).lower() in {"active", "supported", "tentative"}]
        return {"positive": positive, "negative": negative}
    return load_experiences(memory_dir, include_inactive=False)


def structure_summary(structure: Structure, index: int) -> dict[str, Any]:
    properties = structure.properties if isinstance(structure.properties, Mapping) else {}
    metadata = properties.get("crystal_llm_metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "candidate_id": str(properties.get("crystal_llm_candidate_pool_index", index)),
        "index": index,
        "formula": reduced_formula(structure),
        "template": properties.get("crystal_llm_template"),
        "score": properties.get("crystal_llm_score"),
        "nsites": len(structure),
        "volume": structure.volume,
        "volume_per_atom": structure.volume / max(1, len(structure)),
        "min_interatomic_distance": min_interatomic_distance(structure),
        "lattice_abc": list(structure.lattice.abc),
        "lattice_angles": list(structure.lattice.angles),
        "metadata": dict(metadata),
    }


def review_prompt(
    *,
    candidate: Mapping[str, Any],
    positive: Sequence[Mapping[str, Any]],
    negative: Sequence[Mapping[str, Any]],
) -> str:
    return f"""Review this candidate structure using the experience records.

Positive experiences encourage future choices. Negative experiences are pitfalls
to avoid. Explain exactly which records matter and why.

CANDIDATE_JSON:
```json
{json.dumps(candidate, ensure_ascii=False, indent=2)}
```

ACTIVE_POSITIVE_EXPERIENCE_JSON:
```json
{json.dumps([compact_experience(record) for record in positive], ensure_ascii=False, indent=2)}
```

ACTIVE_NEGATIVE_EXPERIENCE_JSON:
```json
{json.dumps([compact_experience(record) for record in negative], ensure_ascii=False, indent=2)}
```

MANDATORY FINAL OUTPUT:
Return exactly one valid JSON object matching this shape, with real values.
Do not copy a default decision from this template; choose the decision that
matches your reasoning and the experience records:
{{"candidate_id":"{candidate.get('candidate_id', '')}","decision":"CHOOSE_ONE_OF_pass_reject_modify","reasoning":"...","positive_experience_used":[],"negative_experience_used":[],"risk_flags":[],"modification":{{"operation":"none","scale_factor":1.0,"reason":"..."}}}}
No markdown, no bullet list, no prose before or after the JSON.
"""


def valid_decision_shape(parsed: Any) -> bool:
    if not isinstance(parsed, Mapping):
        return False
    decision = parsed.get("decision")
    if not isinstance(decision, str) or decision.strip().lower() not in {"pass", "reject", "modify"}:
        return False
    modification = parsed.get("modification")
    if not isinstance(modification, Mapping):
        return False
    operation = modification.get("operation")
    if not isinstance(operation, str) or operation.strip().lower() not in {"none", "scale_lattice"}:
        return False
    return True


def decision_consistency_issues(decision: Mapping[str, Any]) -> list[str]:
    """Detect contradictions between executable decision and the LLM's own text."""
    verdict = str(decision.get("decision", "")).strip().lower()
    if verdict not in {"pass", "modify"}:
        return []

    reasoning = str(decision.get("reasoning", "")).lower()
    risk_flags = [
        str(flag).lower()
        for flag in decision.get("risk_flags", [])
        if isinstance(flag, (str, int, float))
    ]
    negative_used = [
        str(item).lower()
        for item in decision.get("negative_experience_used", [])
        if isinstance(item, (str, int, float))
    ]

    explicit_override = any("explicit_diversity_override" in flag for flag in risk_flags) or (
        "explicit_diversity_override" in reasoning
    )
    if explicit_override:
        return []

    issues: list[str] = []
    direct_negative_markers = [
        "direct_prior_negative_formula",
        "exact_formula_prior_failure",
        "direct formula-level negative",
        "direct historical",
        "directly sampled",
        "directly applies",
        "known pitfall",
    ]
    negative_reasoning_markers = [
        "should not receive ordinary priority",
        "should not be promoted",
        "not a promising lead",
        "noncompetitive",
        "deprioritized",
        "deprioritization",
        "far from sun",
        "if kept",
        "only as an explicit",
        "not enough to overcome",
    ]

    if any(marker in reasoning for marker in direct_negative_markers):
        issues.append("reasoning cites direct negative formula/prototype/family evidence")
    if any(marker in reasoning for marker in negative_reasoning_markers):
        issues.append("reasoning says the candidate is low-priority or should not be promoted")
    if any("direct_prior_negative_formula" in flag or "deprioritized" in flag for flag in risk_flags):
        issues.append("risk_flags contain direct negative or deprioritization markers")
    if any("directly applies" in item for item in negative_used):
        issues.append("negative_experience_used says a negative record directly applies")
    if verdict == "modify" and issues:
        issues.append("modify is inconsistent with chemical/prototype-level negative evidence")
    return issues


def repair_inconsistent_decision(
    *,
    client: ResponsesClient,
    decision: Mapping[str, Any],
    candidate: Mapping[str, Any],
    positive: Sequence[Mapping[str, Any]],
    negative: Sequence[Mapping[str, Any]],
    issues: Sequence[str],
) -> tuple[dict[str, Any], str]:
    repair_prompt = f"""The reviewer output below is valid JSON, but its executable
decision may contradict its own reasoning and risk flags.

You must preserve the substantive material judgment, but fix the top-level
"decision" if needed. The controller has not granted any explicit diversity-probe
override for this candidate.

Consistency rules:
- If reasoning cites direct prior negative formula/prototype/family evidence,
  known pitfall behavior, or says the candidate should not receive ordinary
  priority / should not be promoted / should be kept only "if at all", choose
  "reject" unless an explicit diversity override is granted in the prompt.
- Do not change a negative review into "pass" merely because geometry is not
  pathological.
- Do not use "modify" for chemical/prototype-level negative evidence; lattice
  scaling is only for small geometry concerns.
- If you keep "pass", you must explain why the listed consistency issues do not
  actually apply. Do not invent an override.

CONSISTENCY_ISSUES:
```json
{json.dumps(list(issues), ensure_ascii=False, indent=2)}
```

CANDIDATE_JSON:
```json
{json.dumps(candidate, ensure_ascii=False, indent=2)}
```

ACTIVE_POSITIVE_EXPERIENCE_JSON:
```json
{json.dumps([compact_experience(record) for record in positive], ensure_ascii=False, indent=2)}
```

ACTIVE_NEGATIVE_EXPERIENCE_JSON:
```json
{json.dumps([compact_experience(record) for record in negative], ensure_ascii=False, indent=2)}
```

CURRENT_REVIEW_JSON:
```json
{json.dumps(dict(decision), ensure_ascii=False, indent=2)}
```

Return exactly one valid JSON object matching the original schema. No markdown,
no prose before or after the JSON.
"""
    repaired_text = client.complete_text(
        system=REVIEWER_SYSTEM,
        user=repair_prompt,
        metadata={"role": "candidate_reviewer_consistency_repair", "candidate_id": candidate.get("candidate_id")},
    )
    repaired = extract_json_object(repaired_text)
    if not valid_decision_shape(repaired):
        raise ValueError(f"LLM consistency repair for candidate {candidate.get('candidate_id')} did not return executable JSON")
    return dict(repaired), repaired_text


def parse_or_repair_decision(
    *,
    client: ResponsesClient,
    raw_text: str,
    candidate: Mapping[str, Any],
    positive: Sequence[Mapping[str, Any]],
    negative: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    try:
        parsed = extract_json_object(raw_text)
        if valid_decision_shape(parsed):
            parsed_dict = dict(parsed)
            issues = decision_consistency_issues(parsed_dict)
            if issues:
                return repair_inconsistent_decision(
                    client=client,
                    decision=parsed_dict,
                    candidate=candidate,
                    positive=positive,
                    negative=negative,
                    issues=issues,
                )
            return parsed_dict, raw_text
    except Exception:
        parsed = None

    repair_prompt = f"""The reviewer output below is not in the required executable JSON shape.
Convert it into exactly one valid JSON object. Preserve the substantive reasoning,
but force the top-level "decision" field to be one string: pass, reject, or modify.
If the analysis says there is no experience-based reason to reject or modify and
the intrinsic geometry is not pathological, choose "pass".
If the analysis cites direct prior negative formula/prototype/family evidence,
says a negative record directly applies, says the candidate should not receive
ordinary priority, says it should not be promoted, or says it should be kept only
"if at all", choose "reject" unless the prompt explicitly grants a diversity
override. No diversity override is granted here.

CANDIDATE_JSON:
```json
{json.dumps(candidate, ensure_ascii=False, indent=2)}
```

ACTIVE_POSITIVE_EXPERIENCE_JSON:
```json
{json.dumps([compact_experience(record) for record in positive], ensure_ascii=False, indent=2)}
```

ACTIVE_NEGATIVE_EXPERIENCE_JSON:
```json
{json.dumps([compact_experience(record) for record in negative], ensure_ascii=False, indent=2)}
```

INVALID_REVIEWER_OUTPUT:
```text
{raw_text}
```

Return exactly:
{{"candidate_id":"{candidate.get('candidate_id', '')}","decision":"CHOOSE_ONE_OF_pass_reject_modify","reasoning":"...","positive_experience_used":[],"negative_experience_used":[],"risk_flags":[],"modification":{{"operation":"none","scale_factor":1.0,"reason":"..."}}}}
"""
    repaired_text = client.complete_text(
        system=REVIEWER_SYSTEM,
        user=repair_prompt,
        metadata={"role": "candidate_reviewer_repair", "candidate_id": candidate.get("candidate_id")},
    )
    repaired = extract_json_object(repaired_text)
    if not valid_decision_shape(repaired):
        raise ValueError(f"LLM review for candidate {candidate.get('candidate_id')} did not return executable JSON")
    repaired_dict = dict(repaired)
    issues = decision_consistency_issues(repaired_dict)
    if issues:
        return repair_inconsistent_decision(
            client=client,
            decision=repaired_dict,
            candidate=candidate,
            positive=positive,
            negative=negative,
            issues=issues,
        )
    return repaired_dict, repaired_text


def apply_decision(
    structure: Structure,
    decision: Mapping[str, Any],
    *,
    min_scale: float,
    max_scale: float,
    max_sites: int,
) -> tuple[Structure | None, str | None]:
    verdict = str(decision.get("decision", "")).strip().lower()
    if verdict == "pass":
        reviewed = structure.copy()
    elif verdict == "reject":
        return None, "llm_rejected"
    elif verdict == "modify":
        modification = decision.get("modification")
        if not isinstance(modification, Mapping):
            return None, "missing_modification"
        operation = str(modification.get("operation", "")).strip().lower()
        if operation != "scale_lattice":
            return None, f"unsupported_modification:{operation or 'none'}"
        try:
            scale_factor = float(modification.get("scale_factor"))
        except (TypeError, ValueError):
            return None, "invalid_scale_factor"
        if not (min_scale <= scale_factor <= max_scale):
            return None, "scale_factor_out_of_bounds"
        reviewed = structure.copy()
        reviewed.scale_lattice(reviewed.volume * scale_factor**3)
    else:
        return None, f"unknown_decision:{verdict or 'empty'}"

    validation = validate_structure(reviewed, max_sites=max_sites)
    if not validation.ok:
        return None, "invalid_after_review:" + ",".join(validation.reasons)

    properties = dict(reviewed.properties or {})
    metadata = dict(properties.get("crystal_llm_metadata") or {})
    metadata.update(
        {
            "llm_review": "true",
            "llm_review_decision": verdict,
            "llm_review_reasoning": str(decision.get("reasoning", ""))[:4000],
            "llm_review_positive_experience_used": json.dumps(decision.get("positive_experience_used", []), ensure_ascii=False),
            "llm_review_negative_experience_used": json.dumps(decision.get("negative_experience_used", []), ensure_ascii=False),
        }
    )
    modification = decision.get("modification")
    if isinstance(modification, Mapping):
        metadata["llm_review_modification"] = json.dumps(dict(modification), ensure_ascii=False)
    properties["crystal_llm_metadata"] = metadata
    properties["crystal_llm_reviewed"] = True
    reviewed.properties = properties
    return reviewed, None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    memory_dir = (root / args.memory_dir).resolve()
    candidate_path = Path(args.candidate_pool)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_suffix(".review.report.json")
    decision_dir = Path(args.decision_dir) if args.decision_dir else report_path.parent / "llm_review_decisions"
    llm_log_dir = Path(args.llm_log_dir) if args.llm_log_dir else decision_dir / "llm_calls"
    positive_path = Path(args.positive_memory) if args.positive_memory else None
    negative_path = Path(args.negative_memory) if args.negative_memory else None

    raw = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("candidate pool must be a list of pymatgen Structure dictionaries")
    structures = [Structure.from_dict(item) for item in raw]
    if args.max_candidates is not None:
        structures = structures[: max(0, args.max_candidates)]

    experience = load_active_experience(memory_dir, positive_path=positive_path, negative_path=negative_path)
    known_formulas = load_known_formulas(args.training_data)
    client = ResponsesClient(
        LLMConfig.from_env(
            dotenv=args.dotenv,
            role="REVIEWER",
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        log_dir=llm_log_dir,
    )

    accepted: list[Structure] = []
    decisions: list[dict[str, Any]] = []
    reject_reasons: Counter = Counter()
    dedup = StructureDeduplicator()

    for index, structure in enumerate(structures):
        if len(accepted) >= args.target_count:
            break
        candidate = structure_summary(structure, index)
        text = client.complete_text(
            system=REVIEWER_SYSTEM,
            user=review_prompt(
                candidate=candidate,
                positive=experience["positive"],
                negative=experience["negative"],
            ),
            metadata={"role": "candidate_reviewer", "candidate_id": candidate["candidate_id"]},
        )
        decision, final_text = parse_or_repair_decision(
            client=client,
            raw_text=text,
            candidate=candidate,
            positive=experience["positive"],
            negative=experience["negative"],
        )
        decision.setdefault("candidate_id", candidate["candidate_id"])
        decision_artifact = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate": candidate,
            "raw_text": text,
            "final_text": final_text,
            "decision": decision,
        }
        write_json(decision_dir / f"candidate_{index:04d}.json", decision_artifact)

        reviewed, reject_reason = apply_decision(
            structure,
            decision,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
            max_sites=args.max_sites,
        )
        if reviewed is None:
            reject_reasons[reject_reason or "rejected"] += 1
            decisions.append({**decision_artifact, "accepted": False, "reject_reason": reject_reason})
            continue
        formula = reduced_formula(reviewed)
        if known_formulas and formula in known_formulas and not args.allow_known_compositions:
            reject_reasons["known_formula_after_review"] += 1
            decisions.append({**decision_artifact, "accepted": False, "reject_reason": "known_formula_after_review"})
            continue
        if not dedup.add(reviewed):
            reject_reasons["duplicate_after_review"] += 1
            decisions.append({**decision_artifact, "accepted": False, "reject_reason": "duplicate_after_review"})
            continue
        accepted.append(reviewed)
        decisions.append({**decision_artifact, "accepted": True, "output_index": len(accepted) - 1})

    if len(accepted) < args.target_count and not args.allow_short_output:
        write_json(
            report_path,
            {
                "status": "failed_short_output",
                "accepted_count": len(accepted),
                "target_count": args.target_count,
                "reject_reasons": dict(reject_reasons),
                "decision_dir": str(decision_dir),
            },
        )
        raise RuntimeError(
            f"LLM review accepted only {len(accepted)} structures; "
            "increase candidate pool/max-candidates or pass --allow-short-output"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, [structure.as_dict() for structure in accepted])
    report = {
        "status": "ok",
        "candidate_pool": str(candidate_path),
        "output": str(output_path),
        "target_count": args.target_count,
        "accepted_count": len(accepted),
        "reviewed_count": len(decisions),
        "positive_experience_count": len(experience["positive"]),
        "negative_experience_count": len(experience["negative"]),
        "reject_reasons": dict(reject_reasons),
        "decision_dir": str(decision_dir),
        "llm_log_dir": str(llm_log_dir),
        "accepted_formulas": [reduced_formula(structure) for structure in accepted],
        "decisions": [
            {
                "candidate_id": item["candidate"]["candidate_id"],
                "formula": item["candidate"]["formula"],
                "decision": item["decision"].get("decision"),
                "accepted": item["accepted"],
                "reject_reason": item.get("reject_reason"),
                "reasoning": item["decision"].get("reasoning"),
            }
            for item in decisions
        ],
    }
    write_json(report_path, report)
    print(f"reviewed_count={len(decisions)}")
    print(f"accepted_count={len(accepted)}")
    print(f"wrote {output_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
