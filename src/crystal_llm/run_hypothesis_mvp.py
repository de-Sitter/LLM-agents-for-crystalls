"""Controller for the hypothesis-first LLM crystal generation MVP."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from pymatgen.core import Structure

from crystal_llm.compact_hypothesis_memory import write_compact_memory
from crystal_llm.memory import STRICT_SUN_NOTE, read_json, write_json


SCHEMA_VERSION = "hypothesis_first_mvp.v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hypothesis-first LLM crystal generation MVP.")
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--work-dir", default="hypothesis_runs/current", help="MVP working directory.")
    parser.add_argument("--memory-dir", default="memory", help="Memory directory relative to root.")
    parser.add_argument("--training-data", default=None)
    parser.add_argument("--ppd-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=20260506)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="0 means keep looping until SUN success or interruption unless --keep-going-after-success is set.",
    )
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--min-template-diversity", type=int, default=2)
    parser.add_argument("--max-dialogue-rounds", type=int, default=100)
    parser.add_argument("--materializer-max-attempts", type=int, default=8)
    parser.add_argument(
        "--auditor-max-rounds",
        type=int,
        default=4,
        help="Max Agent D reject/Agent C repair cycles inside one materializer call.",
    )
    parser.add_argument(
        "--max-conflict-revisions",
        type=int,
        default=0,
        help="Max Agent C conflict returns to A/B per round. 0 means unlimited.",
    )
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--max-sites", type=int, default=80)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--evaluator-backend", choices=("slurm", "local"), default="slurm")
    parser.add_argument("--slurm-evaluator-script", default="slurm_evaluate_round_4090.sbatch")
    parser.add_argument("--slurm-partition", default="8-4090")
    parser.add_argument("--slurm-gres", default="gpu:rtx4090:1")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=16)
    parser.add_argument("--force-new", action="store_true", help="Archive any existing work-dir and start a new run.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with the next round after a failure.")
    parser.add_argument(
        "--keep-going-after-success",
        action="store_true",
        help="Keep iterating after rounds that contain one or more strict SUN structures.",
    )
    parser.add_argument("--sleep-between-rounds", type=float, default=0.0)
    parser.add_argument("--compact-memory-recent-rounds", type=int, default=3)
    parser.add_argument("--compact-memory-max-formulas", type=int, default=80)
    parser.add_argument("--compact-memory-max-hypotheses", type=int, default=30)
    parser.add_argument("--compact-memory-max-experiences-per-polarity", type=int, default=16)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_training_data(root: Path) -> Path:
    return root / "archive" / "matllmsearch_evaluator" / "data" / "a_training.json"


def default_ppd_path(root: Path) -> Path:
    return root / "archive" / "matllmsearch_evaluator" / "data" / "2024-08-07-ppd-mp.pkl"


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
    round_dir: Path,
    step_name: str,
    env: Mapping[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process_env = os.environ.copy()
    process_env["PYTHONPATH"] = f"{cwd / 'src'}:{process_env.get('PYTHONPATH', '')}"
    if env:
        process_env.update(dict(env))

    log_event(round_dir, f"START {step_name}: {command_text(cmd)}")
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# started_at_utc={utc_now()}\n")
        handle.write(f"# cwd={cwd}\n")
        handle.write(f"# command={command_text(cmd)}\n")
        handle.flush()
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
            handle.write(line)
            handle.flush()
        return_code = proc.wait()
        handle.write(f"# finished_at_utc={utc_now()}\n")
        handle.write(f"# elapsed_sec={time.monotonic() - started:.1f}\n")
        handle.write(f"# return_code={return_code}\n")
    elapsed = time.monotonic() - started
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


def initial_state(work_dir: Path, seed_base: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status": "initialized",
        "success": False,
        "strict_sun_definition": STRICT_SUN_NOTE,
        "seed_base": seed_base,
        "current_round": 0,
        "best_round": None,
        "best_e_hull": None,
        "best_formula": None,
        "best_template": None,
        "best_structure_path": None,
        "history": [],
        "unresolved_debates": [],
        "work_dir": str(work_dir),
    }


def load_state(state_path: Path) -> dict[str, Any]:
    state = read_json(state_path, {})
    if not isinstance(state, dict):
        raise ValueError(f"{state_path} must contain a JSON object")
    return state


def refresh_compact_memory(
    *,
    args: argparse.Namespace,
    root: Path,
    work_dir: Path,
    memory_dir: Path,
    state_path: Path,
    round_dir: Path | None = None,
) -> Path:
    output_path = work_dir / "compact_memory.json"
    payload = write_compact_memory(
        state_path,
        output_path,
        memory_dir=memory_dir,
        recent_rounds=args.compact_memory_recent_rounds,
        max_formulas=args.compact_memory_max_formulas,
        max_hypotheses=args.compact_memory_max_hypotheses,
        max_experiences_per_polarity=args.compact_memory_max_experiences_per_polarity,
    )
    if round_dir is not None:
        log_event(
            round_dir,
            "compact_memory refreshed: "
            f"rounds={payload['global_summary']['completed_successful_round_records']} "
            f"formulas={len(payload['formula_memory'])} "
            f"forbidden={len(payload['negative_constraints']['forbidden_exact_repeat_formulas'])}",
        )
    return output_path


def submit_or_run_evaluator(
    *,
    args: argparse.Namespace,
    root: Path,
    round_dir: Path,
    input_path: Path,
    results_path: Path,
    training_data: Path,
    ppd_path: Path,
    eval_log: Path,
) -> None:
    if args.evaluator_backend == "local":
        run_command(
            ["timeout", str(args.eval_timeout), str(root / "evaluate_full.sh")],
            cwd=root,
            log_path=eval_log,
            round_dir=round_dir,
            step_name="evaluate_full_local",
            env={
                "INPUT": str(input_path),
                "OUTPUT": str(results_path),
                "TRAINING_DATA": str(training_data),
                "PPD_PATH": str(ppd_path),
                "DEVICE": str(args.device),
            },
        )
        return

    export_values = {
        "ALL": None,
        "ROOT_DIR": str(root),
        "INPUT": str(input_path),
        "OUTPUT": str(results_path),
        "TRAINING_DATA": str(training_data),
        "PPD_PATH": str(ppd_path),
        "DEVICE": str(args.device),
        "EVAL_LOG": str(eval_log),
        "EVAL_TIMEOUT": str(args.eval_timeout),
    }
    export_arg = ",".join(key if value is None else f"{key}={value}" for key, value in export_values.items())
    cmd = [
        "sbatch",
        "--wait",
        "--parsable",
        f"--partition={args.slurm_partition}",
        f"--cpus-per-task={args.slurm_cpus_per_task}",
        f"--export={export_arg}",
    ]
    if args.slurm_gres:
        cmd.append(f"--gres={args.slurm_gres}")
    cmd.append(str(root / args.slurm_evaluator_script))
    run_command(
        cmd,
        cwd=root,
        log_path=round_dir / "evaluate_submit.log",
        round_dir=round_dir,
        step_name="evaluate_full_slurm",
    )
    if not eval_log.exists():
        raise FileNotFoundError(f"evaluator did not write log: {eval_log}")
    if not json_ok(results_path):
        raise FileNotFoundError(f"evaluator did not write valid result JSON: {results_path}")


def analyze_round(root: Path, round_dir: Path, input_path: Path, results_path: Path, eval_log: Path) -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "crystal_llm.analyze_evaluator_run",
            "--round-dir",
            str(round_dir),
            "--input",
            str(input_path),
            "--logs",
            str(eval_log),
            "--shards",
            "1",
            "--result-json",
            str(results_path),
        ],
        cwd=root,
        log_path=round_dir / "analyze.log",
        round_dir=round_dir,
        step_name="analyze_evaluator_run",
    )


def read_e_hull_rows(round_dir: Path) -> list[dict[str, str]]:
    ranked = round_dir / "analysis" / "e_hull_ranked.csv"
    if not ranked.exists():
        return []
    with ranked.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_input_structures(input_path: Path) -> list[dict[str, Any]]:
    raw = read_json(input_path, [])
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def strict_sun_success(summary: Mapping[str, Any]) -> bool:
    try:
        strict_count = int(summary.get("sun_strict_e_hull_lt_0") or 0)
    except (TypeError, ValueError):
        strict_count = 0
    return strict_count > 0


def strict_sun_stats(summary: Mapping[str, Any]) -> tuple[int, int, float]:
    try:
        strict_count = int(summary.get("sun_strict_e_hull_lt_0") or 0)
    except (TypeError, ValueError):
        strict_count = 0
    try:
        total_count = int(summary.get("count") or 0)
    except (TypeError, ValueError):
        total_count = 0
    strict_rate = strict_count / total_count if total_count > 0 else 0.0
    return strict_count, total_count, strict_rate


def update_best_state(state: dict[str, Any], round_dir: Path, input_path: Path) -> None:
    rows = read_e_hull_rows(round_dir)
    if not rows:
        return
    best = rows[0]
    try:
        e_hull = float(best["e_hull"])
    except (KeyError, TypeError, ValueError):
        return
    current_best = state.get("best_e_hull")
    if current_best is not None and e_hull >= float(current_best):
        return

    structures = read_input_structures(input_path)
    try:
        index = int(best["index"]) - 1
    except (KeyError, TypeError, ValueError):
        index = -1
    best_path = Path(state["work_dir"]) / "best_structure.json"
    if 0 <= index < len(structures):
        Structure.from_dict(structures[index])
        write_json(best_path, [structures[index]])
        state["best_structure_path"] = str(best_path)
    state["best_e_hull"] = e_hull
    state["best_round"] = state["current_round"]
    state["best_formula"] = best.get("formula")
    state["best_template"] = best.get("template_guess")


def round_summary_line(summary: Mapping[str, Any]) -> str:
    top = summary.get("top_25")
    best = top[0] if isinstance(top, list) and top and isinstance(top[0], Mapping) else {}
    strict_count, total_count, strict_rate = strict_sun_stats(summary)
    return (
        f"count={summary.get('count')} "
        f"strict_sun={strict_count}/{total_count} "
        f"strict_sun_rate={strict_rate:.3f} "
        f"lt_0.03={summary.get('e_hull_lt_0_03')} "
        f"lt_0.10={summary.get('e_hull_lt_0_10')} "
        f"min_e_hull={summary.get('min_e_hull')} "
        f"best={best.get('formula')}:{best.get('e_hull')}"
    )


def build_history_record(
    *,
    state: Mapping[str, Any],
    round_dir: Path,
    consensus_artifact: Mapping[str, Any],
    materializer_artifact: Mapping[str, Any],
    generation_report: Mapping[str, Any],
    evaluator_result: Mapping[str, Any],
    analysis_summary: Mapping[str, Any],
    success: bool,
) -> dict[str, Any]:
    consensus = consensus_artifact.get("consensus")
    if not isinstance(consensus, Mapping):
        consensus = {}
    final_payload = materializer_artifact.get("final_payload")
    if not isinstance(final_payload, Mapping):
        final_payload = {}
    probes = final_payload.get("formula_probes")
    if not isinstance(probes, list):
        probes = []
    strict_count, total_count, strict_rate = strict_sun_stats(analysis_summary)
    return {
        "round": state.get("current_round"),
        "created_at_utc": utc_now(),
        "round_dir": str(round_dir),
        "success": success,
        "strict_sun_count": strict_count,
        "total_count": total_count,
        "strict_sun_rate": strict_rate,
        "consensus_summary": consensus.get("consensus_summary"),
        "accepted_hypotheses": consensus.get("accepted_hypotheses", []),
        "materialized_formula_probes": probes,
        "materializer_validation": materializer_artifact.get("validation"),
        "agent_d_final_audit": materializer_artifact.get("agent_d_final_audit"),
        "generation_report": generation_report,
        "evaluator_result": {
            "validity": evaluator_result.get("validity"),
            "success_rate": evaluator_result.get("success_rate"),
            "stability_stats": evaluator_result.get("stability_stats"),
            "novelty": evaluator_result.get("novelty"),
        },
        "analysis_summary": analysis_summary,
    }


def run_hypothesis_debate(
    *,
    args: argparse.Namespace,
    root: Path,
    memory_dir: Path,
    state_path: Path,
    round_dir: Path,
    round_number: int,
    revision: int,
    conflict_report: Path | None,
    compact_memory_path: Path,
) -> Path:
    suffix = f"_rev{revision:02d}" if revision else ""
    output = round_dir / f"hypothesis_debate{suffix}.json"
    cmd = [
        sys.executable,
        "-m",
        "crystal_llm.llm_debate_hypotheses",
        "--state",
        str(state_path),
        "--output",
        str(output),
        "--root",
        str(root),
        "--memory-dir",
        str(memory_dir.relative_to(root) if memory_dir.is_relative_to(root) else memory_dir),
        "--dotenv",
        str(root / args.dotenv),
        "--round",
        str(round_number),
        "--max-dialogue-rounds",
        str(args.max_dialogue_rounds),
        "--target-count",
        str(args.target_count),
        "--min-template-diversity",
        str(args.min_template_diversity),
        "--compact-memory",
        str(compact_memory_path),
        "--llm-log-dir",
        str(round_dir / f"hypothesis_llm_calls{suffix}"),
    ]
    if conflict_report:
        cmd.extend(["--conflict-report", str(conflict_report)])
    run_command(
        cmd,
        cwd=root,
        log_path=round_dir / f"hypothesis_debate{suffix}.log",
        round_dir=round_dir,
        step_name=f"hypothesis_debate{suffix or '_initial'}",
    )
    return output


def run_materializer(
    *,
    args: argparse.Namespace,
    root: Path,
    consensus_path: Path,
    round_dir: Path,
    round_number: int,
    revision: int,
    training_data: Path,
    seed: int,
    compact_memory_path: Path,
) -> tuple[Path, Path]:
    suffix = f"_rev{revision:02d}" if revision else ""
    output = round_dir / f"materializer{suffix}.json"
    strategy = round_dir / f"strategy{suffix}.json"
    run_command(
        [
            sys.executable,
            "-m",
            "crystal_llm.llm_materialize_candidates",
            "--consensus",
            str(consensus_path),
            "--output",
            str(output),
            "--strategy-output",
            str(strategy),
            "--root",
            str(root),
            "--dotenv",
            str(root / args.dotenv),
            "--round",
            str(round_number),
            "--training-data",
            str(training_data),
            "--target-count",
            str(args.target_count),
            "--min-template-diversity",
            str(args.min_template_diversity),
            "--max-sites",
            str(args.max_sites),
            "--seed",
            str(seed),
            "--compact-memory",
            str(compact_memory_path),
            "--max-attempts",
            str(args.materializer_max_attempts),
            "--auditor-max-rounds",
            str(args.auditor_max_rounds),
            "--llm-log-dir",
            str(round_dir / f"materializer_llm_calls{suffix}"),
        ],
        cwd=root,
        log_path=round_dir / f"materializer{suffix}.log",
        round_dir=round_dir,
        step_name=f"materializer{suffix or '_initial'}",
    )
    return output, strategy


def resolve_strategy(
    *,
    args: argparse.Namespace,
    root: Path,
    memory_dir: Path,
    state_path: Path,
    round_dir: Path,
    round_number: int,
    training_data: Path,
    seed: int,
    compact_memory_path: Path,
) -> tuple[Path | None, Path | None, Path | None]:
    conflict_report: Path | None = None
    revision = 0
    while args.max_conflict_revisions == 0 or revision <= args.max_conflict_revisions:
        debate_path = run_hypothesis_debate(
            args=args,
            root=root,
            memory_dir=memory_dir,
            state_path=state_path,
            round_dir=round_dir,
            round_number=round_number,
            revision=revision,
            conflict_report=conflict_report,
            compact_memory_path=compact_memory_path,
        )
        debate = read_json(debate_path, {})
        if not isinstance(debate, Mapping):
            raise ValueError(f"debate artifact is invalid: {debate_path}")
        if debate.get("status") == "unresolved":
            return debate_path, None, None
        if debate.get("status") != "consensus":
            raise ValueError(f"unexpected debate status in {debate_path}: {debate.get('status')}")

        materializer_path, strategy_path = run_materializer(
            args=args,
            root=root,
            consensus_path=debate_path,
            round_dir=round_dir,
            round_number=round_number,
            revision=revision,
            training_data=training_data,
            seed=seed,
            compact_memory_path=compact_memory_path,
        )
        materializer = read_json(materializer_path, {})
        if not isinstance(materializer, Mapping):
            raise ValueError(f"materializer artifact is invalid: {materializer_path}")
        if materializer.get("status") == "ok":
            return debate_path, materializer_path, strategy_path
        if materializer.get("status") == "hypothesis_conflict":
            conflict_report = materializer_path
            revision += 1
            log_event(round_dir, f"Agent C reported hypothesis conflict; returning to A/B revision={revision}")
            continue
        raise ValueError(f"Agent C failed to materialize valid probes: {materializer_path}")
    raise RuntimeError(f"Agent C conflict loop exceeded max_conflict_revisions={args.max_conflict_revisions}")


def run_generation(
    *,
    args: argparse.Namespace,
    root: Path,
    round_dir: Path,
    strategy_path: Path,
    training_data: Path,
    seed: int,
) -> Path:
    input_path = round_dir / "input.json"
    report_path = round_dir / "input.report.json"
    run_command(
        [
            sys.executable,
            "-m",
            "crystal_llm.generate",
            "--target-count",
            str(args.target_count),
            "--candidate-multiplier",
            "1",
            "--candidate-pool-only",
            "--formula-probes-only",
            "--output",
            str(input_path),
            "--report",
            str(report_path),
            "--strategy",
            str(strategy_path),
            "--training-data",
            str(training_data),
            "--seed",
            str(seed),
            "--max-sites",
            str(args.max_sites),
        ],
        cwd=root,
        log_path=round_dir / "generate.log",
        round_dir=round_dir,
        step_name="generate_formula_probes",
    )
    run_command(
        [
            sys.executable,
            "-m",
            "crystal_llm.validate_output",
            "--input",
            str(input_path),
        ],
        cwd=root,
        log_path=round_dir / "validate.log",
        round_dir=round_dir,
        step_name="validate_generated_input",
    )
    generated = read_json(input_path, [])
    if not isinstance(generated, list) or len(generated) != args.target_count:
        raise RuntimeError(f"generator wrote {len(generated) if isinstance(generated, list) else 'invalid'} structures")
    return input_path


def append_unresolved(state: dict[str, Any], round_dir: Path, debate_path: Path) -> None:
    state.setdefault("unresolved_debates", []).append(
        {
            "round": state.get("current_round"),
            "created_at_utc": utc_now(),
            "round_dir": str(round_dir),
            "debate_path": str(debate_path),
            "reason": "A/B reached max_dialogue_rounds without explicit consensus.",
        }
    )


def run_one_round(args: argparse.Namespace, root: Path, work_dir: Path, memory_dir: Path, state_path: Path) -> bool:
    state = load_state(state_path)
    state["current_round"] = int(state.get("current_round") or 0) + 1
    state["status"] = "running"
    state["updated_at_utc"] = utc_now()
    round_number = int(state["current_round"])
    round_dir = work_dir / f"round_{round_number:04d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    write_json(state_path, state)

    training_data = Path(args.training_data).resolve() if args.training_data else default_training_data(root)
    ppd_path = Path(args.ppd_path).resolve() if args.ppd_path else default_ppd_path(root)
    seed = args.seed_base + round_number
    log_event(round_dir, f"ROUND {round_number:04d} begin seed={seed}")
    compact_memory_path = refresh_compact_memory(
        args=args,
        root=root,
        work_dir=work_dir,
        memory_dir=memory_dir,
        state_path=state_path,
        round_dir=round_dir,
    )

    debate_path, materializer_path, strategy_path = resolve_strategy(
        args=args,
        root=root,
        memory_dir=memory_dir,
        state_path=state_path,
        round_dir=round_dir,
        round_number=round_number,
        training_data=training_data,
        seed=seed,
        compact_memory_path=compact_memory_path,
    )
    if strategy_path is None or materializer_path is None:
        append_unresolved(state, round_dir, debate_path if debate_path else round_dir / "hypothesis_debate.json")
        state["status"] = "running"
        state["updated_at_utc"] = utc_now()
        write_json(state_path, state)
        log_event(round_dir, "ROUND unresolved: A/B did not produce usable consensus")
        return False

    input_path = run_generation(
        args=args,
        root=root,
        round_dir=round_dir,
        strategy_path=strategy_path,
        training_data=training_data,
        seed=seed,
    )

    results_path = round_dir / "results.json"
    eval_log = round_dir / "eval" / "full_cpu_0.out"
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

    generation_report = read_json(round_dir / "input.report.json", {})
    evaluator_result = read_json(results_path, {})
    analysis_summary = read_json(round_dir / "analysis" / "summary.json", {})
    if not isinstance(generation_report, Mapping):
        generation_report = {}
    if not isinstance(evaluator_result, Mapping):
        evaluator_result = {}
    if not isinstance(analysis_summary, Mapping):
        analysis_summary = {}

    success = strict_sun_success(analysis_summary)
    debate_artifact = read_json(debate_path, {})
    materializer_artifact = read_json(materializer_path, {})
    if not isinstance(debate_artifact, Mapping):
        debate_artifact = {}
    if not isinstance(materializer_artifact, Mapping):
        materializer_artifact = {}
    history_record = build_history_record(
        state=state,
        round_dir=round_dir,
        consensus_artifact=debate_artifact,
        materializer_artifact=materializer_artifact,
        generation_report=generation_report,
        evaluator_result=evaluator_result,
        analysis_summary=analysis_summary,
        success=success,
    )
    state.setdefault("history", []).append(history_record)
    update_best_state(state, round_dir, input_path)
    strict_count, total_count, strict_rate = strict_sun_stats(analysis_summary)
    state["latest_round_success"] = success
    state["latest_round_strict_sun_count"] = strict_count
    state["latest_round_total_count"] = total_count
    state["latest_round_strict_sun_rate"] = strict_rate
    state["success"] = bool(state.get("success")) or success
    state["status"] = "running" if args.keep_going_after_success else ("success" if success else "running")
    state["updated_at_utc"] = utc_now()
    write_json(state_path, state)
    refresh_compact_memory(
        args=args,
        root=root,
        work_dir=work_dir,
        memory_dir=memory_dir,
        state_path=state_path,
        round_dir=round_dir,
    )
    log_event(round_dir, f"ROUND {round_number:04d} complete: {round_summary_line(analysis_summary)}")
    return success


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    work_dir = (root / args.work_dir).resolve()
    memory_dir = (root / args.memory_dir).resolve()
    state_path = work_dir / "state.json"

    if args.force_new and work_dir.exists():
        archive = work_dir.with_name(f"{work_dir.name}_archive_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        shutil.move(str(work_dir), str(archive))

    work_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    if not state_path.exists():
        write_json(state_path, initial_state(work_dir, args.seed_base))
        print(f"initialized hypothesis-first MVP at {work_dir}", flush=True)

    completed = 0
    print(
        f"[{utc_now()}] hypothesis_mvp_start root={root} work_dir={work_dir} "
        f"max_rounds={args.max_rounds} target_count={args.target_count} "
        f"evaluator_backend={args.evaluator_backend} "
        f"keep_going_after_success={args.keep_going_after_success}",
        flush=True,
    )
    while args.max_rounds == 0 or completed < args.max_rounds:
        try:
            state = load_state(state_path)
            if bool(state.get("success")) and not args.keep_going_after_success:
                print(f"[{utc_now()}] already successful: {state_path}", flush=True)
                return 0
            success = run_one_round(args, root, work_dir, memory_dir, state_path)
            completed += 1
            if success and not args.keep_going_after_success:
                print(f"[{utc_now()}] strict SUN success reached after {completed} new round(s)", flush=True)
                return 0
            if args.sleep_between_rounds > 0:
                time.sleep(args.sleep_between_rounds)
        except KeyboardInterrupt:
            print(f"[{utc_now()}] hypothesis_mvp_interrupted", flush=True)
            return 130
        except Exception as exc:
            print(f"[{utc_now()}] hypothesis_mvp_error: {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                return 1
            completed += 1
    print(f"[{utc_now()}] hypothesis_mvp_done rounds_completed={completed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
