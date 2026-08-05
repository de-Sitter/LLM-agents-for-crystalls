"""Restricted scorer DSL for MP candidate-pool ranking.

The LLM is allowed to propose this JSON DSL, but not executable code. This
module validates and interprets the DSL deterministically over candidate-pool
records.
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Mapping, Sequence


ALLOWED_FEATURES = {
    "elements",
    "formula",
    "nelements",
    "nsites",
    "band_gap",
    "formation_energy_per_atom",
    "energy_per_atom",
    "volume",
    "density",
    "crystal_system",
    "spacegroup_symbol",
    "spacegroup_number",
}

FORBIDDEN_FEATURES = {
    "is_stable",
    "e_hull",
    "e_hull_distance",
    "sun",
    "sun_score",
    "novelty",
    "strict_sun",
    "material_id",
    "cif_path",
    "properties_path",
}

ALLOWED_OPS = {
    "all_of",
    "any_of",
    "contains_any_element",
    "contains_all_elements",
    "contains_no_elements",
    "formula_regex",
    "numeric_range",
    "numeric_prefer_low",
    "numeric_prefer_high",
    "numeric_prefer_target",
    "category_in",
    "category_not_in",
}

NUMERIC_FEATURES = {
    "nelements",
    "nsites",
    "band_gap",
    "formation_energy_per_atom",
    "energy_per_atom",
    "volume",
    "density",
    "spacegroup_number",
}

CATEGORY_FEATURES = {"crystal_system", "spacegroup_symbol", "spacegroup_number"}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _as_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _string_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(item) for item in values if isinstance(item, (str, int, float))}


def _validate_term(term: Any, prefix: str, errors: list[str], *, nested: bool = False) -> float:
    local_abs_weight = 0.0
    if not isinstance(term, Mapping):
        errors.append(f"{prefix} must be an object")
        return local_abs_weight

    op = term.get("op")
    feature = term.get("feature")
    if op not in ALLOWED_OPS:
        errors.append(f"{prefix}.op is not allowed: {op}")

    weight = _as_float(term.get("weight"))
    if nested:
        if "weight" in term and weight is None:
            errors.append(f"{prefix}.weight must be a finite number if provided")
    else:
        if weight is None:
            errors.append(f"{prefix}.weight must be a finite number")
        else:
            if abs(weight) > 10:
                errors.append(f"{prefix}.weight absolute value must be <= 10")
            local_abs_weight += abs(weight)

    if op in {"all_of", "any_of"}:
        conditions = term.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            errors.append(f"{prefix}.conditions must be a non-empty list")
        elif len(conditions) > 8:
            errors.append(f"{prefix}.conditions length must be <= 8")
        else:
            for index, condition in enumerate(conditions):
                _validate_term(condition, f"{prefix}.conditions[{index}]", errors, nested=True)
        return local_abs_weight

    if feature not in ALLOWED_FEATURES:
        errors.append(f"{prefix}.feature is not allowed: {feature}")
    if feature in FORBIDDEN_FEATURES:
        errors.append(f"{prefix}.feature is forbidden target/leakage field: {feature}")

    if op in {"contains_any_element", "contains_all_elements", "contains_no_elements"}:
        if feature != "elements":
            errors.append(f"{prefix}.{op} requires feature=elements")
        if not _string_set(term.get("values")):
            errors.append(f"{prefix}.values must contain at least one element symbol")
    elif op == "formula_regex":
        if feature != "formula":
            errors.append(f"{prefix}.formula_regex requires feature=formula")
        pattern = term.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            errors.append(f"{prefix}.pattern must be a non-empty regex string")
        else:
            try:
                re.compile(pattern)
            except re.error as exc:
                errors.append(f"{prefix}.pattern is invalid regex: {exc}")
    elif op == "numeric_range":
        if feature not in NUMERIC_FEATURES:
            errors.append(f"{prefix}.numeric_range uses non-numeric feature")
        minimum = term.get("min")
        maximum = term.get("max")
        if minimum is not None and _as_float(minimum) is None:
            errors.append(f"{prefix}.min must be null or finite number")
        if maximum is not None and _as_float(maximum) is None:
            errors.append(f"{prefix}.max must be null or finite number")
        if _as_float(minimum) is not None and _as_float(maximum) is not None and float(minimum) > float(maximum):
            errors.append(f"{prefix}.min must be <= max")
    elif op in {"numeric_prefer_low", "numeric_prefer_high"}:
        if feature not in NUMERIC_FEATURES:
            errors.append(f"{prefix}.{op} uses non-numeric feature")
        scale = _as_float(term.get("scale"))
        if scale is None or scale <= 0:
            errors.append(f"{prefix}.scale must be a positive finite number")
    elif op == "numeric_prefer_target":
        if feature not in NUMERIC_FEATURES:
            errors.append(f"{prefix}.numeric_prefer_target uses non-numeric feature")
        target = _as_float(term.get("target"))
        scale = _as_float(term.get("scale"))
        if target is None:
            errors.append(f"{prefix}.target must be a finite number")
        if scale is None or scale <= 0:
            errors.append(f"{prefix}.scale must be a positive finite number")
    elif op in {"category_in", "category_not_in"}:
        if feature not in CATEGORY_FEATURES:
            errors.append(f"{prefix}.{op} uses non-category feature")
        if not _string_set(term.get("values")):
            errors.append(f"{prefix}.values must be a non-empty list")
    return local_abs_weight


def validate_scorer_payload(payload: Mapping[str, Any], *, max_terms: int = 24) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != "candidate_scorer.v1":
        errors.append("schema_version must be candidate_scorer.v1")
    if not isinstance(payload.get("scorer_id"), str) or not payload.get("scorer_id"):
        errors.append("scorer_id must be a non-empty string")
    if not isinstance(payload.get("hypothesis_ids"), list) or not payload.get("hypothesis_ids"):
        errors.append("hypothesis_ids must be a non-empty list")
    terms = payload.get("terms")
    if not isinstance(terms, list) or not terms:
        errors.append("terms must be a non-empty list")
        return errors
    if len(terms) > max_terms:
        errors.append(f"terms length must be <= {max_terms}")

    total_abs_weight = 0.0
    for index, term in enumerate(terms):
        total_abs_weight += _validate_term(term, f"terms[{index}]", errors)

    if total_abs_weight <= 0:
        errors.append("sum(abs(weight)) must be positive")
    return errors


def term_score(term: Mapping[str, Any], record: Mapping[str, Any]) -> float:
    op = str(term.get("op"))
    feature = str(term.get("feature"))
    weight = float(term.get("weight") or 0.0)
    value = record.get(feature)

    if op == "all_of":
        conditions = term.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            return 0.0
        if not all(isinstance(condition, Mapping) and term_matches(condition, record) for condition in conditions):
            return 0.0
        return weight
    if op == "any_of":
        conditions = term.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            return 0.0
        if not any(isinstance(condition, Mapping) and term_matches(condition, record) for condition in conditions):
            return 0.0
        return weight
    if op == "contains_any_element":
        elements = set(str(item) for item in _as_list(record.get("elements")))
        return weight if elements.intersection(_string_set(term.get("values"))) else 0.0
    if op == "contains_all_elements":
        elements = set(str(item) for item in _as_list(record.get("elements")))
        return weight if _string_set(term.get("values")).issubset(elements) else 0.0
    if op == "contains_no_elements":
        elements = set(str(item) for item in _as_list(record.get("elements")))
        return weight if not elements.intersection(_string_set(term.get("values"))) else 0.0
    if op == "formula_regex":
        pattern = str(term.get("pattern") or "")
        try:
            return weight if re.search(pattern, str(record.get("formula") or "")) else 0.0
        except re.error:
            return 0.0
    if op == "numeric_range":
        number = _as_float(value)
        if number is None:
            return 0.0
        minimum = _as_float(term.get("min"))
        maximum = _as_float(term.get("max"))
        if minimum is not None and number < minimum:
            return 0.0
        if maximum is not None and number > maximum:
            return 0.0
        return weight
    if op == "numeric_prefer_low":
        number = _as_float(value)
        scale = _as_float(term.get("scale"))
        if number is None or scale is None or scale <= 0:
            return 0.0
        return weight / (1.0 + max(0.0, number) / scale)
    if op == "numeric_prefer_high":
        number = _as_float(value)
        scale = _as_float(term.get("scale"))
        if number is None or scale is None or scale <= 0:
            return 0.0
        return weight * (number / (abs(number) + scale))
    if op == "numeric_prefer_target":
        number = _as_float(value)
        target = _as_float(term.get("target"))
        scale = _as_float(term.get("scale"))
        if number is None or target is None or scale is None or scale <= 0:
            return 0.0
        return weight / (1.0 + abs(number - target) / scale)
    if op == "category_in":
        return weight if str(value) in _string_set(term.get("values")) else 0.0
    if op == "category_not_in":
        return weight if str(value) not in _string_set(term.get("values")) else 0.0
    return 0.0


def term_matches(term: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    probe = dict(term)
    probe["weight"] = 1.0
    return bool(term_score(probe, record))


def score_record(payload: Mapping[str, Any], record: Mapping[str, Any]) -> float:
    return float(sum(term_score(term, record) for term in payload.get("terms", []) if isinstance(term, Mapping)))


def _rank_normalized_contributions(term: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[float]:
    """Return rank-normalized contributions for a numeric term.

    The LLM-facing contract names this as ``candidate_pool_percentile_rank``:
    for ``numeric_prefer_low``, lower raw values get larger contributions.
    Ties receive the average rank, and records with missing values get zero.
    """

    feature = str(term.get("feature"))
    op = str(term.get("op"))
    weight = float(term.get("weight") or 0.0)
    values: list[tuple[float, int]] = []
    for index, record in enumerate(records):
        number = _as_float(record.get(feature))
        if number is not None:
            values.append((number, index))
    contributions = [0.0 for _ in records]
    if not values:
        return contributions
    values.sort(key=lambda item: item[0])
    denominator = max(1, len(values) - 1)
    pos = 0
    while pos < len(values):
        end = pos + 1
        while end < len(values) and values[end][0] == values[pos][0]:
            end += 1
        average_rank = (pos + end - 1) / 2.0
        percentile = average_rank / denominator
        if op == "numeric_prefer_low":
            preference = 1.0 - percentile
        elif op == "numeric_prefer_high":
            preference = percentile
        else:
            preference = max(0.0, 1.0 - abs(percentile - 0.5) * 2.0)
        for _, original_index in values[pos:end]:
            contributions[original_index] = weight * preference
        pos = end
    return contributions


def score_records(payload: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[float]:
    """Score records with support for pool-level normalization contracts."""

    scores = [0.0 for _ in records]
    for term in payload.get("terms", []):
        if not isinstance(term, Mapping):
            continue
        if term.get("normalization") == "candidate_pool_percentile_rank":
            for index, contribution in enumerate(_rank_normalized_contributions(term, records)):
                scores[index] += contribution
            continue
        for index, record in enumerate(records):
            scores[index] += term_score(term, record)
    return [float(value) for value in scores]


def diagnostic_distribution(payload: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = score_records(payload, records)
    if not scores:
        return {"count": 0}
    unique_scores = len(set(round(value, 8) for value in scores))
    sorted_scores = sorted(scores)
    bins = Counter()
    for value in scores:
        if value <= 0:
            bins["<=0"] += 1
        elif value < 1:
            bins["0-1"] += 1
        elif value < 3:
            bins["1-3"] += 1
        elif value < 6:
            bins["3-6"] += 1
        else:
            bins[">=6"] += 1
    return {
        "count": len(scores),
        "unique_scores": unique_scores,
        "min": sorted_scores[0],
        "p10": sorted_scores[int(0.10 * (len(sorted_scores) - 1))],
        "median": sorted_scores[int(0.50 * (len(sorted_scores) - 1))],
        "p90": sorted_scores[int(0.90 * (len(sorted_scores) - 1))],
        "max": sorted_scores[-1],
        "bins": dict(bins),
    }


def validate_scorer_behavior(
    payload: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    min_unique_scores: int = 5,
    min_positive_fraction: float = 0.01,
    max_positive_fraction: float = 1.0,
) -> list[str]:
    errors = validate_scorer_payload(payload)
    if errors:
        return errors
    if not records:
        return ["candidate pool records are empty"]
    scores = score_records(payload, records)
    unique_scores = len(set(round(value, 8) for value in scores))
    if unique_scores < min_unique_scores:
        errors.append(f"scorer is too flat: unique_scores={unique_scores} < {min_unique_scores}")
    positive_fraction = sum(value > 0 for value in scores) / len(scores)
    if positive_fraction < min_positive_fraction:
        errors.append(f"scorer is too narrow: positive_fraction={positive_fraction:.4f}")
    if positive_fraction > max_positive_fraction:
        errors.append(f"scorer is too broad: positive_fraction={positive_fraction:.4f}")
    sorted_scores = sorted(scores)
    if len(sorted_scores) >= 2:
        spread = sorted_scores[-1] - sorted_scores[0]
        if spread <= 1e-9:
            errors.append("scorer is too flat: score spread is zero")
    return errors
