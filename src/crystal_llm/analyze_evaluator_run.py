"""Build per-structure e-hull analysis from evaluator shard logs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any, Iterable, Sequence

from pymatgen.core import Element, Structure


E_HULL_RE = re.compile(
    r"E-hull distance:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
SHARD_RE = re.compile(r"_(\d+)\.out$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-structure e-hull values from shard evaluator logs.")
    parser.add_argument("--round-dir", default=None, help="Round directory containing input.json.")
    parser.add_argument("--input", default=None, help="Input structure JSON. Defaults to ROUND_DIR/input.json.")
    parser.add_argument("--logs", nargs="+", default=None, help="Evaluator log paths or glob patterns.")
    parser.add_argument("--job-id", default=None, help="Slurm array job id, used to find crystal_eval_shard logs.")
    parser.add_argument("--shards", type=int, default=None, help="Shard count. Defaults to shard manifest or log count.")
    parser.add_argument("--analysis-dir", default=None, help="Output directory. Defaults to ROUND_DIR/analysis.")
    parser.add_argument("--result-json", default=None, help="Optional merged evaluator JSON to include in summary.")
    return parser.parse_args(argv)


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern) if not Path(pattern).is_absolute() else Path("/").glob(pattern[1:]))
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pattern))
    return sorted(set(paths))


def infer_defaults(args: argparse.Namespace) -> tuple[Path, list[Path], int | None, Path]:
    round_dir = Path(args.round_dir) if args.round_dir else None
    input_path = Path(args.input) if args.input else (round_dir / "input.json" if round_dir else None)
    if input_path is None:
        raise ValueError("provide --input or --round-dir")

    patterns: list[str] = []
    if args.logs:
        patterns.extend(args.logs)
    elif args.job_id:
        patterns.append(f"slurm_logs/crystal_eval_shard-{args.job_id}_*.out")
    else:
        raise ValueError("provide --logs or --job-id")

    log_paths = expand_paths(patterns)
    if not log_paths:
        raise FileNotFoundError(f"no evaluator logs matched: {patterns}")

    shards = args.shards
    if shards is None and round_dir:
        manifest = round_dir / "eval" / "shards" / "manifest.json"
        if manifest.exists():
            with manifest.open("r", encoding="utf-8") as handle:
                shards = int(json.load(handle)["shards"])

    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else (round_dir / "analysis" if round_dir else Path("analysis"))
    return input_path, log_paths, shards, analysis_dir


def parse_e_hulls(log_paths: list[Path], shards: int | None) -> dict[int, float]:
    by_index: dict[int, float] = {}
    if shards is None:
        shard_ids = []
        for path in log_paths:
            match = SHARD_RE.search(path.name)
            if match:
                shard_ids.append(int(match.group(1)))
        shards = max(shard_ids) + 1 if shard_ids else len(log_paths)

    for path in log_paths:
        match = SHARD_RE.search(path.name)
        if not match:
            if shards == 1 and len(log_paths) == 1:
                shard = 0
            else:
                raise ValueError(f"cannot infer shard index from log path: {path}")
        else:
            shard = int(match.group(1))
        values = [float(value) for value in E_HULL_RE.findall(path.read_text(encoding="utf-8", errors="replace"))]
        for local_pos, e_hull in enumerate(values):
            by_index[shard + local_pos * shards] = e_hull
    return by_index


def likely_anion(elements: list[str]) -> str | None:
    scored: list[tuple[float, str]] = []
    for symbol in elements:
        try:
            x = Element(symbol).X
        except Exception:
            x = float("nan")
        if x is None or math.isnan(x):
            continue
        scored.append((float(x), symbol))
    if not scored:
        return None
    return max(scored)[1]


def guess_template(structure: Structure) -> str:
    props = structure.properties or {}
    template = props.get("crystal_llm_template")
    if isinstance(template, str) and template:
        return template

    nsites = len(structure)
    composition = structure.composition
    reduced = composition.reduced_composition
    amounts = {el.symbol: float(amount) for el, amount in reduced.items()}
    total = int(round(sum(amounts.values())))
    elements = list(amounts)

    if nsites == 5 and total == 5 and len(elements) == 3:
        return "perovskite"
    if nsites == 10 and total == 10 and len(elements) == 4:
        return "double_perovskite"
    if nsites == 56 and total == 7 and len(elements) == 3:
        return "spinel"
    if nsites == 30 and total == 5 and len(elements) == 2:
        return "corundum"
    if nsites == 8 and total == 2 and len(elements) == 2:
        return "rocksalt"
    if nsites == 6 and total == 3 and len(elements) == 2:
        return "rutile"
    if nsites == 2 and total == 2 and len(elements) == 2:
        return "cesium_chloride"
    if nsites == 4 and total == 2 and len(elements) == 2:
        return "wurtzite"
    if nsites == 12 and total == 3 and len(elements) == 2:
        anion = likely_anion(elements)
        if anion and int(round(amounts[anion])) == 2:
            return "fluorite"
        if anion and int(round(amounts[anion])) == 1:
            return "antifluorite"
        return "fluorite_or_antifluorite"
    if total == 4 and len(elements) == 3:
        return "delafossite"
    return "unknown"


def load_structures(path: Path) -> list[Structure]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("input structures must be a JSON list")
    return [Structure.from_dict(item) for item in raw]


def row_for(index: int, structure: Structure, e_hull: float) -> dict[str, Any]:
    return {
        "index": index + 1,
        "formula": structure.composition.reduced_formula,
        "e_hull": e_hull,
        "nsites": len(structure),
        "template_guess": guess_template(structure),
        "volume_per_atom": structure.volume / len(structure),
    }


def missing_row_for(index: int, structure: Structure) -> dict[str, Any]:
    return {
        "index": index + 1,
        "formula": structure.composition.reduced_formula,
        "nsites": len(structure),
        "template_guess": guess_template(structure),
        "volume_per_atom": structure.volume / len(structure),
        "reason": "e_hull_not_found_in_evaluator_log",
    }


def template_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["template_guess"])].append(float(row["e_hull"]))

    stats: dict[str, dict[str, Any]] = {}
    for template in sorted(grouped):
        values = grouped[template]
        stats[template] = {
            "count": len(values),
            "e_hull_lt_0": sum(value < 0.0 for value in values),
            "e_hull_lt_0_03": sum(value < 0.03 for value in values),
            "e_hull_lt_0_10": sum(value < 0.10 for value in values),
            "mean_e_hull": mean(values),
            "min_e_hull": min(values),
        }
    return stats


def load_optional_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def write_outputs(
    analysis_dir: Path,
    rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    input_structure_count: int,
    stats: dict[str, dict[str, Any]],
    merged_result: dict[str, Any] | None,
    log_paths: list[Path],
) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = analysis_dir / "e_hull_ranked.csv"
    with ranked_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "formula", "e_hull", "nsites", "template_guess", "volume_per_atom"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (analysis_dir / "template_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    values = [float(row["e_hull"]) for row in rows]
    summary = {
        "count": len(rows),
        "input_structure_count": input_structure_count,
        "missing_e_hull_count": len(missing_rows),
        "missing_e_hull_rows": missing_rows,
        "sun_strict_e_hull_lt_0": sum(value < 0.0 for value in values),
        "e_hull_lt_0_03": sum(value < 0.03 for value in values),
        "e_hull_lt_0_10": sum(value < 0.10 for value in values),
        "mean_e_hull": mean(values) if values else None,
        "min_e_hull": min(values) if values else None,
        "max_e_hull": max(values) if values else None,
        "top_25": rows[:25],
        "template_stats": stats,
        "log_paths": [str(path) for path in log_paths],
    }
    if merged_result:
        summary["merged_evaluator_success_rate"] = merged_result.get("success_rate")
        summary["merged_evaluator_stability_stats"] = merged_result.get("stability_stats")
        summary["merged_evaluator_novelty"] = merged_result.get("novelty")

    with (analysis_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"wrote {ranked_path}")
    print(f"wrote {analysis_dir / 'template_stats.json'}")
    print(f"wrote {analysis_dir / 'summary.json'}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path, log_paths, shards, analysis_dir = infer_defaults(args)
    structures = load_structures(input_path)
    e_hulls = parse_e_hulls(log_paths, shards)
    missing = sorted(set(range(len(structures))) - set(e_hulls))
    extra = sorted(set(e_hulls) - set(range(len(structures))))
    if extra:
        raise ValueError(f"found e-hull values for {len(extra)} out-of-range structures, first extra index {extra[0]}")

    rows = [row_for(index, structures[index], e_hulls[index]) for index in range(len(structures)) if index in e_hulls]
    rows.sort(key=lambda row: float(row["e_hull"]))
    missing_rows = [missing_row_for(index, structures[index]) for index in missing]
    stats = template_stats(rows)
    merged_result = load_optional_json(args.result_json)
    write_outputs(analysis_dir, rows, missing_rows, len(structures), stats, merged_result, log_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
