"""Resumable controller for the real-LLM crystal search loop."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from crystal_llm.experience import NEGATIVE_EXPERIENCE_FILE, POSITIVE_EXPERIENCE_FILE
from crystal_llm.memory import read_json, round_label, write_json


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the iterative LLM-controlled crystal generation workflow.")
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--memory-dir", default="memory", help="Memory directory relative to root.")
    parser.add_argument("--start-round", type=int, default=None, help="Explicit first round. Defaults to latest reflected + 1.")
    parser.add_argument("--max-rounds", type=int, default=0, help="Number of rounds to run. 0 means keep looping.")
    parser.add_argument("--target-count", type=int, default=10, help="Reviewed structures evaluated per round.")
    parser.add_argument("--candidate-pool-target-count", type=int, default=20, help="Generator target before multiplier.")
    parser.add_argument("--candidate-multiplier", type=int, default=4, help="Candidate pool multiplier.")
    parser.add_argument("--max-candidates", type=int, default=80, help="Maximum generated candidates sent to LLM reviewer.")
    parser.add_argument("--generator-jobs", type=int, default=1, help="Parallel jobs for raw candidate generation.")
    parser.add_argument("--seed-base", type=int, default=20260504, help="Base seed; round number is added.")
    parser.add_argument("--training-data", default=None, help="Training JSON path. Defaults to evaluator archive data.")
    parser.add_argument("--ppd-path", default=None, help="Phase diagram pickle path. Defaults to evaluator archive data.")
    parser.add_argument("--device", default="cpu", help="Evaluator device, e.g. cpu or cuda.")
    parser.add_argument(
        "--evaluator-backend",
        choices=("local", "slurm"),
        default="local",
        help="Run evaluator in the controller process or submit it to Slurm and wait.",
    )
    parser.add_argument("--slurm-evaluator-script", default="slurm_evaluate_round_4090.sbatch")
    parser.add_argument("--slurm-partition", default="8-4090")
    parser.add_argument("--slurm-gres", default="gpu:rtx4090:1")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=16)
    parser.add_argument("--eval-timeout", type=int, default=1200, help="Evaluator timeout in seconds per round.")
    parser.add_argument("--debate-rounds", type=int, default=2, help="A/B critique cycles before referee.")
    parser.add_argument("--dotenv", default=".env", help="LLM environment file.")
    parser.add_argument("--sleep-between-rounds", type=float, default=0.0, help="Delay after each completed round.")
    parser.add_argument("--force", action="store_true", help="Re-run steps even if expected outputs already exist.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue to next round after a failure.")
    parser.add_argument(
        "--no-include-structures",
        action="store_true",
        help="Do not include full structure dictionaries in reflection context.",
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def command_text(cmd: Sequence[str]) -> str:
    return " ".join(str(part) for part in cmd)


def log_event(round_dir: Path, message: str) -> None:
    round_dir.mkdir(parents=True, exist_ok=True)
    line = f"[{utc_now()}] {message}\n"
    with (round_dir / "controller.log").open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
    round_dir: Path,
    step_name: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process_env = os.environ.copy()
    if env:
        process_env.update(dict(env))
    process_env["PYTHONPATH"] = f"{cwd / 'src'}:{process_env.get('PYTHONPATH', '')}"

    log_event(round_dir, f"START {step_name}: {command_text(cmd)}")
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# started_at_utc={utc_now()}\n")
        log.write(f"# cwd={cwd}\n")
        log.write(f"# command={command_text(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = proc.wait()
        log.write(f"# finished_at_utc={utc_now()}\n")
        log.write(f"# return_code={return_code}\n")
    elapsed = time.monotonic() - start
    if return_code != 0:
        log_event(round_dir, f"FAIL {step_name}: return_code={return_code} elapsed_sec={elapsed:.1f}")
        raise subprocess.CalledProcessError(return_code, list(cmd))
    log_event(round_dir, f"DONE {step_name}: elapsed_sec={elapsed:.1f}")


def json_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return True


def review_report_ok(path: Path) -> bool:
    payload = read_json(path, {})
    return isinstance(payload, Mapping) and payload.get("status") == "ok"


def latest_reflected_round(memory_dir: Path) -> int:
    state = read_json(memory_dir / "state.json", {})
    if isinstance(state, Mapping) and isinstance(state.get("latest_reflected_round"), int):
        return int(state["latest_reflected_round"])
    debate_dir = memory_dir / "debates"
    latest = 0
    if debate_dir.exists():
        for path in debate_dir.glob("round_*/reflection_report.json"):
            try:
                latest = max(latest, int(path.parent.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return latest


def update_state(memory_dir: Path, round_number: int) -> None:
    state_path = memory_dir / "state.json"
    state = read_json(state_path, {})
    if not isinstance(state, dict):
        state = {}
    imported_rounds = {
        int(item)
        for item in state.get("imported_rounds", [])
        if isinstance(item, int) or str(item).isdigit()
    }
    imported_rounds.add(round_number)
    state.setdefault("schema_version", "memory_state.v2_fresh_llm_flow")
    state["latest_imported_round"] = max(int(state.get("latest_imported_round") or 0), round_number)
    state["latest_reflected_round"] = max(int(state.get("latest_reflected_round") or 0), round_number)
    state["imported_rounds"] = sorted(imported_rounds)
    state["updated_at_utc"] = utc_now()
    state["files"] = {
        "positive_experience": POSITIVE_EXPERIENCE_FILE,
        "negative_experience": NEGATIVE_EXPERIENCE_FILE,
        "round_contexts": "round_contexts/",
        "debates": "debates/",
    }
    state["counts"] = {
        "positive_experience": count_jsonl(memory_dir / POSITIVE_EXPERIENCE_FILE),
        "negative_experience": count_jsonl(memory_dir / NEGATIVE_EXPERIENCE_FILE),
    }
    write_json(state_path, state)


def default_training_data(root: Path) -> Path:
    return root / "archive" / "matllmsearch_evaluator" / "data" / "a_training.json"


def default_ppd_path(root: Path) -> Path:
    return root / "archive" / "matllmsearch_evaluator" / "data" / "2024-08-07-ppd-mp.pkl"


def round_summary_line(summary_path: Path) -> str:
    payload = read_json(summary_path, {})
    if not isinstance(payload, Mapping):
        return "summary unavailable"
    top = payload.get("top_25")
    best = top[0] if isinstance(top, list) and top and isinstance(top[0], Mapping) else {}
    return (
        f"count={payload.get('count')} "
        f"strict_sun={payload.get('sun_strict_e_hull_lt_0')} "
        f"lt_0.10={payload.get('e_hull_lt_0_10')} "
        f"min_e_hull={payload.get('min_e_hull')} "
        f"best={best.get('formula')}:{best.get('e_hull')}"
    )


def run_round(args: argparse.Namespace, root: Path, memory_dir: Path, round_number: int) -> None:
    label = round_label(round_number)
    round_dir = root / "rounds" / label
    eval_dir = round_dir / "eval"
    training_data = Path(args.training_data).resolve() if args.training_data else default_training_data(root)
    ppd_path = Path(args.ppd_path).resolve() if args.ppd_path else default_ppd_path(root)
    seed = args.seed_base + round_number

    candidate_pool = round_dir / "candidate_pool.json"
    candidate_report = round_dir / "candidate_pool.report.json"
    reviewed_input = round_dir / "input.json"
    review_report = round_dir / "llm_review.report.json"
    results_json = round_dir / "results.json"
    eval_log = eval_dir / "full_cpu_0.out"
    analysis_summary = round_dir / "analysis" / "summary.json"
    reflection_report = memory_dir / "debates" / label / "reflection_report.json"

    log_event(round_dir, f"ROUND {label} begin seed={seed}")

    if args.force or not json_ok(candidate_pool):
        run_command(
            [
                sys.executable,
                "-m",
                "crystal_llm.generate",
                "--target-count",
                str(args.candidate_pool_target_count),
                "--candidate-multiplier",
                str(args.candidate_multiplier),
                "--candidate-pool-only",
                "--output",
                str(candidate_pool),
                "--report",
                str(candidate_report),
                "--training-data",
                str(training_data),
                "--seed",
                str(seed),
                "--jobs",
                str(args.generator_jobs),
            ],
            cwd=root,
            log_path=round_dir / "generate.log",
            round_dir=round_dir,
            step_name="generate_candidate_pool",
        )
    else:
        log_event(round_dir, "SKIP generate_candidate_pool: candidate pool exists")

    if args.force or not (json_ok(reviewed_input) and review_report_ok(review_report)):
        run_command(
            [
                sys.executable,
                "-m",
                "crystal_llm.llm_review_candidates",
                "--root",
                str(root),
                "--memory-dir",
                str(memory_dir.relative_to(root) if memory_dir.is_relative_to(root) else memory_dir),
                "--candidate-pool",
                str(candidate_pool),
                "--target-count",
                str(args.target_count),
                "--max-candidates",
                str(args.max_candidates),
                "--training-data",
                str(training_data),
                "--output",
                str(reviewed_input),
                "--report",
                str(review_report),
                "--decision-dir",
                str(round_dir / "llm_review_decisions"),
                "--llm-log-dir",
                str(round_dir / "llm_review_llm_calls"),
                "--dotenv",
                str(root / args.dotenv),
            ],
            cwd=root,
            log_path=round_dir / "llm_review.log",
            round_dir=round_dir,
            step_name="llm_review_candidates",
        )
    else:
        log_event(round_dir, "SKIP llm_review_candidates: reviewed input exists")

    run_command(
        [
            sys.executable,
            "-m",
            "crystal_llm.validate_output",
            "--input",
            str(reviewed_input),
        ],
        cwd=root,
        log_path=round_dir / "validate.log",
        round_dir=round_dir,
        step_name="validate_reviewed_input",
    )

    if args.force or not json_ok(results_json) or not eval_log.exists():
        eval_dir.mkdir(parents=True, exist_ok=True)
        if args.evaluator_backend == "slurm":
            export_values = {
                "ALL": None,
                "ROOT_DIR": str(root),
                "INPUT": str(reviewed_input),
                "OUTPUT": str(results_json),
                "TRAINING_DATA": str(training_data),
                "PPD_PATH": str(ppd_path),
                "DEVICE": str(args.device),
                "EVAL_LOG": str(eval_log),
                "EVAL_TIMEOUT": str(args.eval_timeout),
            }
            export_arg = ",".join(
                key if value is None else f"{key}={value}" for key, value in export_values.items()
            )
            sbatch_cmd = [
                "sbatch",
                "--wait",
                "--parsable",
                f"--partition={args.slurm_partition}",
                f"--cpus-per-task={args.slurm_cpus_per_task}",
                f"--export={export_arg}",
            ]
            if args.slurm_gres:
                sbatch_cmd.append(f"--gres={args.slurm_gres}")
            sbatch_cmd.append(str(root / args.slurm_evaluator_script))
            run_command(
                sbatch_cmd,
                cwd=root,
                log_path=round_dir / "evaluate_submit.log",
                round_dir=round_dir,
                step_name="evaluate_full_slurm",
            )
            if not eval_log.exists():
                raise FileNotFoundError(f"Slurm evaluator completed but log was not written: {eval_log}")
            if not json_ok(results_json):
                raise FileNotFoundError(f"Slurm evaluator completed but result JSON is missing/invalid: {results_json}")
        else:
            run_command(
                ["timeout", str(args.eval_timeout), str(root / "evaluate_full.sh")],
                cwd=root,
                log_path=eval_log,
                env={
                    "INPUT": str(reviewed_input),
                    "OUTPUT": str(results_json),
                    "TRAINING_DATA": str(training_data),
                    "PPD_PATH": str(ppd_path),
                    "DEVICE": str(args.device),
                },
                round_dir=round_dir,
                step_name="evaluate_full",
            )
    else:
        log_event(round_dir, "SKIP evaluate_full: results and evaluator log exist")

    if args.force or not json_ok(analysis_summary):
        run_command(
            [
                sys.executable,
                "-m",
                "crystal_llm.analyze_evaluator_run",
                "--round-dir",
                str(round_dir),
                "--logs",
                str(eval_log),
                "--shards",
                "1",
                "--result-json",
                str(results_json),
            ],
            cwd=root,
            log_path=round_dir / "analyze.log",
            round_dir=round_dir,
            step_name="analyze_evaluator_run",
        )
    else:
        log_event(round_dir, "SKIP analyze_evaluator_run: summary exists")

    if args.force or not json_ok(reflection_report):
        reflect_cmd = [
            sys.executable,
            "-m",
            "crystal_llm.llm_reflect_round",
            "--root",
            str(root),
            "--memory-dir",
            str(memory_dir.relative_to(root) if memory_dir.is_relative_to(root) else memory_dir),
            "--round",
            str(round_number),
            "--latest-round",
            str(round_number),
            "--debate-rounds",
            str(args.debate_rounds),
            "--context-output",
            str(memory_dir / "round_contexts" / f"{label}.json"),
            "--context-markdown-output",
            str(memory_dir / "round_contexts" / f"{label}.md"),
            "--debate-dir",
            str(memory_dir / "debates" / label),
            "--llm-log-dir",
            str(memory_dir / "debates" / label / "llm_calls"),
            "--dotenv",
            str(root / args.dotenv),
        ]
        if not args.no_include_structures:
            reflect_cmd.append("--include-structures")
        run_command(
            reflect_cmd,
            cwd=root,
            log_path=round_dir / "reflect.log",
            round_dir=round_dir,
            step_name="llm_reflect_round",
        )
    else:
        log_event(round_dir, "SKIP llm_reflect_round: reflection report exists")

    update_state(memory_dir, round_number)
    log_event(round_dir, f"ROUND {label} complete: {round_summary_line(analysis_summary)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    memory_dir = (root / args.memory_dir).resolve()
    root.joinpath("rounds").mkdir(exist_ok=True)
    memory_dir.mkdir(exist_ok=True)

    current_round = args.start_round if args.start_round is not None else latest_reflected_round(memory_dir) + 1
    rounds_completed = 0
    print(
        f"[{utc_now()}] controller_start root={root} start_round={current_round} "
        f"max_rounds={args.max_rounds} target_count={args.target_count} "
        f"device={args.device} evaluator_backend={args.evaluator_backend}",
        flush=True,
    )

    while args.max_rounds == 0 or rounds_completed < args.max_rounds:
        try:
            run_round(args, root, memory_dir, current_round)
            rounds_completed += 1
            current_round += 1
            if args.sleep_between_rounds > 0:
                time.sleep(args.sleep_between_rounds)
        except KeyboardInterrupt:
            print(f"[{utc_now()}] controller_interrupted", flush=True)
            return 130
        except Exception as exc:
            print(f"[{utc_now()}] controller_error round={current_round}: {exc}", flush=True)
            if not args.continue_on_error:
                return 1
            current_round += 1
    print(f"[{utc_now()}] controller_done rounds_completed={rounds_completed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
