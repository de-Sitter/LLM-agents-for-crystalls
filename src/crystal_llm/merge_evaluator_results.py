"""Merge independent_evaluator JSON outputs from structure shards.

This is intended for fast iteration on the current generator, whose output has
unique reduced formulas.  Under that condition cross-shard StructureMatcher
duplicates are not expected because pymatgen's default matcher requires
composition compatibility.
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import math
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge shard evaluator results.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Result JSON paths or glob patterns.")
    parser.add_argument("--output", required=True, help="Merged output JSON path.")
    return parser.parse_args(argv)


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted(set(paths))


def weighted_mean(values: list[tuple[float, int]]) -> float | None:
    total = sum(n for _, n in values)
    if total == 0:
        return None
    return sum(value * n for value, n in values) / total


def merge_stability_stats(results: list[dict[str, Any]]) -> dict[str, float]:
    valid_parts: list[tuple[dict[str, Any], int]] = []
    for result in results:
        stats = result.get("stability_stats")
        count = result.get("success_rate", {}).get("valid_structures", 0)
        if stats and count:
            valid_parts.append((stats, int(count)))

    if not valid_parts:
        return {}

    count_total = sum(count for _, count in valid_parts)
    mean = sum(stats["mean_e_hull"] * count for stats, count in valid_parts) / count_total
    second_moment = sum(
        count * (stats.get("std_e_hull", 0.0) ** 2 + stats["mean_e_hull"] ** 2)
        for stats, count in valid_parts
    ) / count_total
    variance = max(0.0, second_moment - mean**2)

    return {
        "min_e_hull": min(stats["min_e_hull"] for stats, _ in valid_parts),
        "max_e_hull": max(stats["max_e_hull"] for stats, _ in valid_parts),
        "mean_e_hull": mean,
        "median_e_hull": valid_parts[0][0].get("median_e_hull") if len(valid_parts) == 1 else None,
        "std_e_hull": math.sqrt(variance),
        "note": "median_e_hull is exact only when merging a single shard; evaluator summaries do not include per-structure e_hull values",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = expand_inputs(args.inputs)
    if not paths:
        raise ValueError("no input files matched")

    results: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            results.append(json.load(handle))

    total = sum(result.get("total_structures", 0) for result in results)
    if total == 0:
        raise ValueError("merged total_structures is zero")

    composition_counts: Counter[str] = Counter()
    for result in results:
        composition_counts.update(result.get("composition_diversity_details", {}).get("composition_counts", {}))

    valid_structures = sum(result.get("validity", {}).get("valid_structures", 0) for result in results)
    valid_compositions = sum(result.get("validity", {}).get("valid_compositions", 0) for result in results)

    success_counts = {
        "valid_structures": sum(result.get("success_rate", {}).get("valid_structures", 0) for result in results),
        "stable_structures_0": sum(result.get("success_rate", {}).get("stable_structures_0", 0) for result in results),
        "stable_structures_0.03": sum(result.get("success_rate", {}).get("stable_structures_0.03", 0) for result in results),
        "stable_structures_0.10": sum(result.get("success_rate", {}).get("stable_structures_0.10", 0) for result in results),
        "success_structures": sum(result.get("success_rate", {}).get("success_structures", 0) for result in results),
    }

    both_novel_count = sum(result.get("novelty", {}).get("both_novel_count", 0) for result in results)
    sun_both_novel_count = sum(result.get("novelty", {}).get("sun_both_novel_count", 0) for result in results)
    novel_compositions = sum(
        result.get("novelty", {}).get("composition_novelty", {}).get("novel_compositions", 0)
        for result in results
    )
    novel_structures = sum(
        result.get("novelty", {}).get("structural_novelty", {}).get("novel_structures", 0)
        for result in results
    )
    unique_structures = sum(
        result.get("structural_diversity_details", {}).get("unique_structures", 0)
        for result in results
    )

    merged = {
        "total_structures": total,
        "validity": {
            "structural_validity": valid_structures / total,
            "composition_validity": valid_compositions / total,
            "valid_structures": valid_structures,
            "valid_compositions": valid_compositions,
            "total_structures": total,
        },
        "structural_validity": valid_structures / total,
        "composition_validity": valid_compositions / total,
        "composition_diversity": len(composition_counts) / total,
        "composition_diversity_details": {
            "composition_diversity": len(composition_counts) / total,
            "unique_compositions": len(composition_counts),
            "total_structures": total,
            "composition_ratio": len(composition_counts) / total,
            "composition_counts": dict(composition_counts),
        },
        "structural_diversity": unique_structures / total,
        "structural_diversity_details": {
            "structural_diversity": unique_structures / total,
            "unique_structures": unique_structures,
            "total_structures": total,
            "structural_ratio": unique_structures / total,
            "note": "merged from shard unique counts; exact when formulas are globally unique",
        },
        "success_rate": {
            "validity_rate": success_counts["valid_structures"] / total,
            "valid_structures": success_counts["valid_structures"],
            "metastability_0": success_counts["stable_structures_0"] / total,
            "metastability_0.03": success_counts["stable_structures_0.03"] / total,
            "metastability_0.10": success_counts["stable_structures_0.10"] / total,
            "m3gnet_metastability": success_counts["stable_structures_0.10"] / total,
            "stable_structures_0": success_counts["stable_structures_0"],
            "stable_structures_0.03": success_counts["stable_structures_0.03"],
            "stable_structures_0.10": success_counts["stable_structures_0.10"],
            "stability_rate_0.03": success_counts["stable_structures_0.03"] / total,
            "stability_rate_0.10": success_counts["stable_structures_0.10"] / total,
            "success_rate": success_counts["success_structures"] / total,
            "success_structures": success_counts["success_structures"],
            "total_structures": total,
        },
        "stability_stats": merge_stability_stats(results),
        "m3gnet_metastability": success_counts["stable_structures_0.10"] / total,
        "overall_novelty": both_novel_count / total,
        "novelty": {
            "sun_score": sun_both_novel_count / total,
            "both_novel_count": both_novel_count,
            "sun_both_novel_count": sun_both_novel_count,
            "total_structures": total,
            "composition_novelty": {
                "composition_novelty": novel_compositions / max(1, len(composition_counts)),
                "novel_compositions": novel_compositions,
                "total_compositions": len(composition_counts),
            },
            "structural_novelty": {
                "structural_novelty": novel_structures / total,
                "novel_structures": novel_structures,
                "total_structures": total,
            },
        },
        "composition_novelty": novel_compositions / max(1, len(composition_counts)),
        "structural_novelty": novel_structures / total,
        "merged_from": [str(path) for path in paths],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)
    print(f"wrote merged result to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
