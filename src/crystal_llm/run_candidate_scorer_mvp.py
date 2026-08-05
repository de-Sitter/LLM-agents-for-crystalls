"""Controller for the candidate-pool scorer MVP."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import orjson

from crystal_llm.memory import STRICT_SUN_NOTE, read_json, write_json
from crystal_llm.run_hypothesis_mvp import (
    analyze_round,
    default_ppd_path,
    default_training_data,
    json_ok,
    log_event,
    run_command,
    submit_or_run_evaluator,
)


SCHEMA_VERSION = "candidate_scorer_mvp.v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run candidate-pool scorer MVP.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--work-dir", default="candidate_scorer_runs/current")
    parser.add_argument("--memory-dir", default="memory")
    parser.add_argument("--candidate-pool", default="data/mp_candidate_pool/mp_candidates_filtered.jsonl")
    parser.add_argument("--training-data", default=None)
    parser.add_argument("--ppd-path", default=None)
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--seed-base", type=int, default=20260507)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--max-dialogue-rounds", type=int, default=100)
    parser.add_argument("--compiler-max-attempts", type=int, default=6)
    parser.add_argument("--auditor-max-rounds", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--random-k", type=int, default=20)
    parser.add_argument("--bottom-k", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--evaluator-backend", choices=("slurm", "local"), default="slurm")
    parser.add_argument("--slurm-evaluator-script", default="slurm_evaluate_round_4090.sbatch")
    parser.add_argument("--slurm-partition", default="8-4090")
    parser.add_argument("--slurm-gres", default="gpu:rtx4090:1")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=16)
    parser.add_argument("--force-new", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--sleep-between-rounds", type=float, default=0.0)
    parser.add_argument("--scoring-max-records", type=int, default=0)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initial_state(work_dir: Path, seed_base: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status": "initialized",
        "strict_sun_definition": STRICT_SUN_NOTE,
        "seed_base": seed_base,
        "current_round": 0,
        "best_round": None,
        "best_top_bucket_strict_sun_rate": None,
        "best_top_bucket_strict_sun_count": None,
        "history": [],
        "unresolved_debates": [],
        "work_dir": str(work_dir),
    }


def command_text(cmd: Sequence[str]) -> str:
    return " ".join(str(part) for part in cmd)


def run_rule_debate(
    *,
    args: argparse.Namespace,
    root: Path,
    state_path: Path,
    round_dir: Path,
    round_number: int,
    conflict_report: Path | None = None,
) -> Path:
    output = round_dir / "rule_debate.json"
    cmd = [
        sys.executable,
        "-m",
        "crystal_llm.llm_debate_rules",
        "--state",
        str(state_path),
        "--output",
        str(output),
        "--root",
        str(root),
        "--memory-dir",
        str(args.memory_dir),
        "--dotenv",
        str(root / args.dotenv),
        "--round",
        str(round_number),
        "--max-dialogue-rounds",
        str(args.max_dialogue_rounds),
        "--llm-log-dir",
        str(round_dir / "rule_llm_calls"),
    ]
    if conflict_report:
        cmd.extend(["--scorer-conflict-report", str(conflict_report)])
    run_command(cmd, cwd=root, log_path=round_dir / "rule_debate.log", round_dir=round_dir, step_name="rule_debate")
    return output


def run_compile_scorer(
    *,
    args: argparse.Namespace,
    root: Path,
    consensus_path: Path,
    round_dir: Path,
    round_number: int,
) -> tuple[Path, Path]:
    output = round_dir / "scorer_compile.json"
    scorer = round_dir / "scorer.json"
    run_command(
        [
            sys.executable,
            "-m",
            "crystal_llm.llm_compile_scorer",
            "--consensus",
            str(consensus_path),
            "--candidate-pool",
            str(root / args.candidate_pool),
            "--output",
            str(output),
            "--scorer-output",
            str(scorer),
            "--root",
            str(root),
            "--dotenv",
            str(root / args.dotenv),
            "--round",
            str(round_number),
            "--max-attempts",
            str(args.compiler_max_attempts),
            "--auditor-max-rounds",
            str(args.auditor_max_rounds),
            "--llm-log-dir",
            str(round_dir / "scorer_llm_calls"),
        ],
        cwd=root,
        log_path=round_dir / "scorer_compile.log",
        round_dir=round_dir,
        step_name="compile_scorer",
    )
    return output, scorer


def run_scoring(
    *,
    args: argparse.Namespace,
    root: Path,
    round_dir: Path,
    scorer_path: Path,
    seed: int,
) -> Path:
    scored_dir = round_dir / "scored"
    cmd = [
        sys.executable,
        "-m",
        "crystal_llm.score_candidate_pool",
        "--candidate-pool",
        str(root / args.candidate_pool),
        "--scorer",
        str(scorer_path),
        "--output-dir",
        str(scored_dir),
        "--top-k",
        str(args.top_k),
        "--random-k",
        str(args.random_k),
        "--bottom-k",
        str(args.bottom_k),
        "--seed",
        str(seed),
    ]
    if args.scoring_max_records > 0:
        cmd.extend(["--max-records", str(args.scoring_max_records)])
    run_command(cmd, cwd=root, log_path=round_dir / "score_candidate_pool.log", round_dir=round_dir, step_name="score_candidate_pool")
    return scored_dir / "input.json"


def load_selected_records(round_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = round_dir / "scored" / "selected_records.jsonl"
    if not path.exists():
        return records
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                item = orjson.loads(line)
                if isinstance(item, dict):
                    records.append(item)
    return records


def load_ranked_rows(round_dir: Path) -> list[dict[str, str]]:
    path = round_dir / "analysis" / "e_hull_ranked.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bucket_analysis(round_dir: Path) -> dict[str, Any]:
    selected = load_selected_records(round_dir)
    rows = load_ranked_rows(round_dir)
    by_index = {}
    for row in rows:
        try:
            by_index[int(row["index"]) - 1] = row
        except Exception:
            continue
    buckets: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(selected):
        bucket = str(record.get("bucket") or "unknown")
        entry = buckets.setdefault(
            bucket,
            {"count": 0, "strict_sun_count": 0, "e_hull_lt_0_03": 0, "e_hull_lt_0_10": 0, "min_e_hull": None, "mean_e_hull": None, "formulas": []},
        )
        entry["count"] += 1
        row = by_index.get(index)
        if not row:
            continue
        try:
            e_hull = float(row["e_hull"])
        except Exception:
            continue
        entry.setdefault("_e_hulls", []).append(e_hull)
        entry["formulas"].append(
            {
                "material_id": record.get("material_id"),
                "formula": record.get("formula"),
                "score": record.get("score"),
                "e_hull": e_hull,
            }
        )
        if e_hull < 0.0:
            entry["strict_sun_count"] += 1
        if e_hull < 0.03:
            entry["e_hull_lt_0_03"] += 1
        if e_hull < 0.10:
            entry["e_hull_lt_0_10"] += 1
    for entry in buckets.values():
        values = entry.pop("_e_hulls", [])
        if values:
            entry["min_e_hull"] = min(values)
            entry["mean_e_hull"] = sum(values) / len(values)
        count = max(1, int(entry["count"]))
        entry["strict_sun_rate"] = entry["strict_sun_count"] / count
        entry["e_hull_lt_0_03_rate"] = entry["e_hull_lt_0_03"] / count
        entry["e_hull_lt_0_10_rate"] = entry["e_hull_lt_0_10"] / count
        entry["formulas"] = sorted(entry["formulas"], key=lambda item: float(item["e_hull"]))[:20]
    return buckets


def append_unresolved(state: dict[str, Any], round_dir: Path, debate_path: Path, reason: str) -> None:
    state.setdefault("unresolved_debates", []).append(
        {"round": state.get("current_round"), "created_at_utc": utc_now(), "round_dir": str(round_dir), "debate_path": str(debate_path), "reason": reason}
    )


def run_one_round(args: argparse.Namespace, root: Path, work_dir: Path, state_path: Path) -> bool:
    state = read_json(state_path, {})
    if not isinstance(state, dict):
        state = initial_state(work_dir, args.seed_base)
    state["current_round"] = int(state.get("current_round") or 0) + 1
    state["status"] = "running"
    state["updated_at_utc"] = utc_now()
    round_number = int(state["current_round"])
    round_dir = work_dir / f"round_{round_number:04d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    write_json(state_path, state)
    seed = args.seed_base + round_number

    log_event(round_dir, f"ROUND {round_number:04d} begin seed={seed}")
    debate_path = run_rule_debate(args=args, root=root, state_path=state_path, round_dir=round_dir, round_number=round_number)
    debate = read_json(debate_path, {})
    if not isinstance(debate, Mapping) or debate.get("status") != "consensus":
        append_unresolved(state, round_dir, debate_path, "A/B rule debate did not produce consensus.")
        write_json(state_path, state)
        return False

    compile_path, scorer_path = run_compile_scorer(args=args, root=root, consensus_path=debate_path, round_dir=round_dir, round_number=round_number)
    compiled = read_json(compile_path, {})
    if not isinstance(compiled, Mapping):
        raise ValueError("invalid scorer compile artifact")
    if compiled.get("status") == "rule_conflict":
        append_unresolved(state, round_dir, compile_path, "C/D agreed A/B rules are contradictory or not expressible.")
        write_json(state_path, state)
        return False
    if compiled.get("status") != "ok":
        raise ValueError(f"scorer compile failed: {compile_path}")

    input_path = run_scoring(args=args, root=root, round_dir=round_dir, scorer_path=scorer_path, seed=seed)
    results_path = round_dir / "results.json"
    eval_log = round_dir / "eval" / "full_cpu_0.out"
    training_data = Path(args.training_data).resolve() if args.training_data else default_training_data(root)
    ppd_path = Path(args.ppd_path).resolve() if args.ppd_path else default_ppd_path(root)
    submit_or_run_evaluator(
        args=args,
        root=root,
        round_dir=round_dir,
        input_path=input_path,
        results_path=results_path,
        training_data=training_data,
        ppd_path=ppd_path,
        eval_log=eval_log,
    )
    analyze_round(root, round_dir, input_path, results_path, eval_log)
    buckets = bucket_analysis(round_dir)
    write_json(round_dir / "analysis" / "bucket_summary.json", buckets)

    top = buckets.get("top", {})
    top_rate = float(top.get("strict_sun_rate") or 0.0)
    history_record = {
        "round": round_number,
        "created_at_utc": utc_now(),
        "round_dir": str(round_dir),
        "rule_debate_path": str(debate_path),
        "scorer_path": str(scorer_path),
        "score_summary": read_json(round_dir / "scored" / "score_summary.json", {}),
        "analysis_summary": read_json(round_dir / "analysis" / "summary.json", {}),
        "bucket_summary": buckets,
    }
    state.setdefault("history", []).append(history_record)
    best_rate = state.get("best_top_bucket_strict_sun_rate")
    if best_rate is None or top_rate > float(best_rate):
        state["best_top_bucket_strict_sun_rate"] = top_rate
        state["best_top_bucket_strict_sun_count"] = int(top.get("strict_sun_count") or 0)
        state["best_round"] = round_number
    state["latest_top_bucket_strict_sun_rate"] = top_rate
    state["latest_bucket_summary"] = buckets
    state["status"] = "running"
    state["updated_at_utc"] = utc_now()
    write_json(state_path, state)
    log_event(round_dir, f"ROUND {round_number:04d} complete top_strict_sun_rate={top_rate:.4f} buckets={json.dumps(buckets, ensure_ascii=False)[:1000]}")
    return top_rate > 0.0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    work_dir = (root / args.work_dir).resolve()
    state_path = work_dir / "state.json"

    if args.force_new and work_dir.exists():
        archive = work_dir.with_name(f"{work_dir.name}_archive_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        shutil.move(str(work_dir), str(archive))
    work_dir.mkdir(parents=True, exist_ok=True)
    if not state_path.exists():
        write_json(state_path, initial_state(work_dir, args.seed_base))
        print(f"initialized candidate scorer MVP at {work_dir}", flush=True)

    completed = 0
    print(
        f"[{utc_now()}] candidate_scorer_mvp_start root={root} work_dir={work_dir} "
        f"max_rounds={args.max_rounds} top/random/bottom={args.top_k}/{args.random_k}/{args.bottom_k} "
        f"backend={args.evaluator_backend}",
        flush=True,
    )
    while args.max_rounds == 0 or completed < args.max_rounds:
        try:
            run_one_round(args, root, work_dir, state_path)
            completed += 1
            if args.sleep_between_rounds > 0:
                time.sleep(args.sleep_between_rounds)
        except KeyboardInterrupt:
            print(f"[{utc_now()}] candidate_scorer_mvp_interrupted", flush=True)
            return 130
        except Exception as exc:
            print(f"[{utc_now()}] candidate_scorer_mvp_error: {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                return 1
            completed += 1
    print(f"[{utc_now()}] candidate_scorer_mvp_done rounds_completed={completed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
