"""Versioned positive/negative natural-language experience records."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


POSITIVE_EXPERIENCE_FILE = "positive_experience.jsonl"
NEGATIVE_EXPERIENCE_FILE = "negative_experience.jsonl"
ACTIVE_STATUSES = {"active", "supported", "tentative"}
FINAL_STATUSES = {"superseded", "rejected", "retired"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str, *, fallback: str = "experience") -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return text[:80] or fallback


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if isinstance(item, dict):
                records.append(item)
    return records


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(dict(record), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            json.dump(dict(record), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def experience_path(memory_dir: Path, polarity: str) -> Path:
    if polarity == "positive":
        return memory_dir / POSITIVE_EXPERIENCE_FILE
    if polarity == "negative":
        return memory_dir / NEGATIVE_EXPERIENCE_FILE
    raise ValueError(f"unknown experience polarity: {polarity}")


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if item]
    return [str(value)]


def normalize_experience(
    raw: Mapping[str, Any],
    *,
    polarity: str,
    round_number: int,
    index: int,
    existing_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a schema-normalized experience record.

    The LLM may provide either ``experience_key`` or only a claim.  The key is
    the version identity; ``id`` is the immutable record id.
    """

    claim = str(
        raw.get("claim") or raw.get("natural_language") or raw.get("experience") or raw.get("title") or ""
    ).strip()
    if not claim:
        raise ValueError("experience record is missing claim")

    experience_key = str(raw.get("experience_key") or raw.get("key") or slugify(claim)).strip()
    if not experience_key:
        experience_key = f"{polarity}_round_{round_number:03d}_{index:03d}"

    supersedes = _as_string_list(raw.get("supersedes"))
    same_key_records = [record for record in existing_records if record.get("experience_key") == experience_key]
    if not supersedes and same_key_records:
        supersedes = [str(record.get("id")) for record in same_key_records if record.get("status", "active") in ACTIVE_STATUSES]
    previous_versions = [
        int(record.get("version", 0) or 0)
        for record in same_key_records
        if isinstance(record.get("version", 0), int) or str(record.get("version", "")).isdigit()
    ]
    version = max(previous_versions or [0]) + 1
    if raw.get("version") is not None and not same_key_records:
        try:
            version = max(1, int(raw["version"]))
        except (TypeError, ValueError):
            version = 1

    record_id = str(raw.get("id") or f"{polarity}.{experience_key}.v{version}").strip()
    status = str(raw.get("status") or "active").strip().lower()
    if status in FINAL_STATUSES and not raw.get("allow_final_status"):
        status = "active"

    normalized = {
        "id": record_id,
        "schema_version": "experience.v1",
        "polarity": polarity,
        "experience_key": experience_key,
        "version": version,
        "status": status,
        "created_at_utc": str(raw.get("created_at_utc") or utc_now()),
        "valid_from_round": int(raw.get("valid_from_round") or round_number),
        "valid_until_round": raw.get("valid_until_round"),
        "supersedes": supersedes,
        "superseded_by": [],
        "claim": claim,
        "detailed_reasoning": str(
            raw.get("detailed_reasoning") or raw.get("reasoning") or raw.get("experience") or ""
        ).strip(),
        "evidence": raw.get("evidence") if isinstance(raw.get("evidence"), list) else [],
        "counterevidence_considered": (
            raw.get("counterevidence_considered")
            if isinstance(raw.get("counterevidence_considered"), list)
            else raw.get("counterevidence")
            if isinstance(raw.get("counterevidence"), list)
            else []
        ),
        "scope": str(raw.get("scope") or "").strip(),
        "confidence": str(raw.get("confidence") or "low").strip().lower(),
        "actionability": str(raw.get("actionability") or "candidate_review").strip(),
        "recommended_action": str(raw.get("recommended_action") or raw.get("action") or "").strip(),
        "do_not_generalize_to": _as_string_list(raw.get("do_not_generalize_to")),
        "source_round": int(raw.get("source_round") or round_number),
        "created_by": raw.get("created_by") if isinstance(raw.get("created_by"), Mapping) else {},
        "tags": _as_string_list(raw.get("tags")),
    }
    extra = raw.get("extra")
    if isinstance(extra, Mapping):
        normalized["extra"] = dict(extra)
    return normalized


def merge_new_experiences(
    path: Path,
    new_records: Sequence[Mapping[str, Any]],
    *,
    polarity: str,
    round_number: int,
) -> list[dict[str, Any]]:
    """Merge new records into a JSONL file and mark superseded records."""

    existing = read_jsonl(path)
    merged = [dict(record) for record in existing]
    id_to_record = {str(record.get("id")): record for record in merged if record.get("id")}
    normalized_records: list[dict[str, Any]] = []
    for index, raw in enumerate(new_records, start=1):
        normalized = normalize_experience(
            raw,
            polarity=polarity,
            round_number=round_number,
            index=index,
            existing_records=merged + normalized_records,
        )
        for old_id in normalized.get("supersedes", []):
            old = id_to_record.get(str(old_id))
            if old is None:
                continue
            old["status"] = "superseded"
            old["superseded_by"] = list(dict.fromkeys(_as_string_list(old.get("superseded_by")) + [normalized["id"]]))
            old["updated_at_utc"] = utc_now()
        normalized_records.append(normalized)
        id_to_record[normalized["id"]] = normalized

    merged.extend(normalized_records)
    write_jsonl(path, merged)
    return normalized_records


def load_experiences(memory_dir: Path, *, include_inactive: bool = False) -> dict[str, list[dict[str, Any]]]:
    positive = read_jsonl(experience_path(memory_dir, "positive"))
    negative = read_jsonl(experience_path(memory_dir, "negative"))
    if not include_inactive:
        positive = [record for record in positive if str(record.get("status", "active")).lower() in ACTIVE_STATUSES]
        negative = [record for record in negative if str(record.get("status", "active")).lower() in ACTIVE_STATUSES]
    return {"positive": positive, "negative": negative}


def compact_experience(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "polarity",
        "experience_key",
        "version",
        "status",
        "claim",
        "detailed_reasoning",
        "scope",
        "confidence",
        "actionability",
        "recommended_action",
        "do_not_generalize_to",
        "tags",
    )
    return {key: record.get(key) for key in keys if key in record}
