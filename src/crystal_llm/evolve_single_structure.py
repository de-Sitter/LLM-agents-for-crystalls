"""MVP controller for single-structure LLM-guided evolution."""

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

from crystal_llm.filters import reduced_formula, validate_structure
from crystal_llm.generate import parse_args as generate_parse_args
from crystal_llm.generate import load_strategy_file, load_known_formulas, collect_candidates, select_final
from crystal_llm.memory import read_json, write_json
from crystal_llm.structure_edit import apply_edit, structure_summary


SCHEMA_VERSION = "single_structure_evolution.v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-structure LLM-guided crystal evolution MVP.")
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--work-dir", default="single_evolution/current", help="Evolution working directory.")
    parser.add_argument("--memory-dir", default="memory", help="Memory directory relative to root.")
    parser.add_argument("--training-data", default=None)
    parser.add_argument("--ppd-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--max-rounds", type=int, default=0, help="0 means keep looping until SUN success or interruption.")
    parser.add_argument("--debate-rounds", type=int, default=3)
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--max-sites", type=int, default=80)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--evaluator-backend", choices=("slurm", "local"), default="slurm")
    parser.add_argument("--slurm-evaluator-script", default="slurm_evaluate_round_4090.sbatch")
    parser.add_argument("--slurm-partition", default="8-4090")
    parser.add_argument("--slurm-gres", default="gpu:rtx4090:1")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=16)
    parser.add_argument("--initialize-only", action="store_true", help="Create initial structure and stop.")
    parser.add_argument("--force-new", action="store_true", help="Overwrite existing evolution state.")
    parser.add_argument("--skip-reflection", action="store_true", help="Do not run LLM reflection after evaluator.")
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_training_data(root: Path) -> Path:
    return root / "archive" / "matllmsearch_evaluator" / "data" / "a_training.json"


def default_ppd_path(root: Path) -> Path:
    return root / "archive" / "matllmsearch_evaluator" / "data" / "2024-08-07-ppd-mp.pkl"


def command_text(cmd: Sequence[str]) -> str:
    return " ".join(str(part) for part in cmd)


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
    step_name: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process_env = os.environ.copy()
    process_env["PYTHONPATH"] = f"{cwd / 'src'}:{process_env.get('PYTHONPATH', '')}"
    if env:
        process_env.update(dict(env))
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
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(cmd))


def load_structure(path: Path) -> Structure:
    raw = read_json(path, None)
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError(f"{path} must contain exactly one structure")
        raw = raw[0]
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must contain a pymatgen Structure dictionary")
    return Structure.from_dict(dict(raw))


def write_single_structure(path: Path, structure: Structure) -> None:
    write_json(path, [structure.as_dict()])


def generate_initial_structure(root: Path, training_data: Path, seed: int, max_sites: int) -> Structure:
    gen_args = generate_parse_args(
        [
            "--target-count",
            "1",
            "--candidate-multiplier",
            "30",
            "--seed",
            str(seed),
            "--training-data",
            str(training_data),
            "--max-sites",
            str(max_sites),
        ]
    )
    strategy = load_strategy_file(None)
    known = load_known_formulas(str(training_data))
    candidates, _ = collect_candidates(gen_args, strategy, known)
    selected = select_final(candidates, 1, 1, strategy)
    if not selected:
        raise RuntimeError("generator did not produce an initial structure")
    return selected[0].structure


def initial_state(work_dir: Path, structure_path: Path, seed: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status": "initialized",
        "success": False,
        "seed": seed,
        "current_round": 0,
        "current_structure_path": str(structure_path),
        "best_round": None,
        "best_e_hull": None,
        "best_structure_path": None,
        "history": [],
        "work_dir": str(work_dir),
    }


def read_first_e_hull(round_dir: Path) -> float | None:
    ranked = round_dir / "analysis" / "e_hull_ranked.csv"
    if not ranked.exists():
        return None
    with ranked.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    try:
        return float(rows[0]["e_hull"])
    except (KeyError, TypeError, ValueError):
        return None


def evaluator_success(result: Mapping[str, Any], e_hull: float | None) -> bool:
    validity = result.get("validity", {})
    novelty = result.get("novelty", {})
    if not isinstance(validity, Mapping) or not isinstance(novelty, Mapping):
        return False
    structural_valid = float(validity.get("structural_validity", 0.0) or 0.0) >= 1.0
    composition_valid = float(validity.get("composition_validity", 0.0) or 0.0) >= 1.0
    sun_score = float(novelty.get("sun_score", 0.0) or 0.0)
    return structural_valid and composition_valid and sun_score >= 1.0 and e_hull is not None and e_hull < 0.0


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
            env={
                "INPUT": str(input_path),
                "OUTPUT": str(results_path),
                "TRAINING_DATA": str(training_data),
                "PPD_PATH": str(ppd_path),
                "DEVICE": str(args.device),
            },
            step_name="evaluate_full_local",
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
    export_arg = ",".join(
        key if value is None else f"{key}={value}"
        for key, value in export_values.items()
    )
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
        step_name="evaluate_full_slurm",
    )
    if not results_path.exists():
        raise FileNotFoundError(f"evaluator did not write {results_path}")


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
        step_name="analyze_evaluator_run",
    )


def update_best_state(state: dict[str, Any], round_dir: Path, edited_structure: Structure, e_hull: float | None) -> None:
    if e_hull is None:
        return
    current_best = state.get("best_e_hull")
    if current_best is None or e_hull < float(current_best):
        best_path = Path(state["work_dir"]) / "best_structure.json"
        write_single_structure(best_path, edited_structure)
        state["best_e_hull"] = e_hull
        state["best_round"] = state["current_round"]
        state["best_structure_path"] = str(best_path)


def append_history(
    state: dict[str, Any],
    *,
    round_dir: Path,
    before: Structure,
    after: Structure,
    edit: Mapping[str, Any],
    result: Mapping[str, Any],
    e_hull: float | None,
    success: bool,
) -> None:
    novelty = result.get("novelty", {}) if isinstance(result, Mapping) else {}
    validity = result.get("validity", {}) if isinstance(result, Mapping) else {}
    success_rate = result.get("success_rate", {}) if isinstance(result, Mapping) else {}
    state.setdefault("history", []).append(
        {
            "round": state["current_round"],
            "created_at_utc": utc_now(),
            "round_dir": str(round_dir),
            "edit": dict(edit),
            "before": structure_summary(before),
            "after": structure_summary(after),
            "formula": reduced_formula(after),
            "e_hull": e_hull,
            "success": success,
            "validity": validity,
            "success_rate": success_rate,
            "novelty": novelty,
        }
    )


def run_reflection(args: argparse.Namespace, root: Path, memory_dir: Path, state_path: Path, round_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "crystal_llm.llm_reflect_evolution",
        "--state",
        str(state_path),
        "--round-dir",
        str(round_dir),
        "--output",
        str(round_dir / "reflection_report.json"),
        "--memory-dir",
        str(memory_dir),
        "--dotenv",
        str(root / args.dotenv),
        "--llm-log-dir",
        str(round_dir / "reflect_llm_calls"),
    ]
    try:
        run_command(
            cmd,
            cwd=root,
            log_path=round_dir / "reflect.log",
            step_name="llm_reflect_evolution",
        )
    except Exception as exc:
        failure = {
            "created_at_utc": utc_now(),
            "step": "llm_reflect_evolution",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "command": command_text(cmd),
            "log_path": str(round_dir / "reflect.log"),
            "continued": True,
        }
        write_json(round_dir / "reflection_failure.json", failure)
        print(
            "reflection_failed_but_continuing="
            f"{type(exc).__name__}: {exc}; wrote {round_dir / 'reflection_failure.json'}",
            flush=True,
        )


def run_one_round(args: argparse.Namespace, root: Path, work_dir: Path, memory_dir: Path, state_path: Path) -> bool:
    state = read_json(state_path, {})
    if not isinstance(state, dict):
        raise ValueError("state JSON must be an object")
    current_structure_path = Path(state["current_structure_path"])
    current = load_structure(current_structure_path)
    state["current_round"] = int(state.get("current_round") or 0) + 1
    state["updated_at_utc"] = utc_now()
    round_dir = work_dir / f"round_{state['current_round']:04d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    before_path = round_dir / "before.json"
    write_single_structure(before_path, current)
    write_json(state_path, state)

    run_command(
        [
            sys.executable,
            "-m",
            "crystal_llm.llm_debate_edit",
            "--state",
            str(state_path),
            "--structure",
            str(before_path),
            "--output",
            str(round_dir / "edit_decision.json"),
            "--root",
            str(root),
            "--memory-dir",
            str(memory_dir.relative_to(root) if memory_dir.is_relative_to(root) else memory_dir),
            "--dotenv",
            str(root / args.dotenv),
            "--debate-rounds",
            str(args.debate_rounds),
            "--llm-log-dir",
            str(round_dir / "edit_llm_calls"),
            "--max-sites",
            str(args.max_sites),
        ],
        cwd=root,
        log_path=round_dir / "edit_debate.log",
        step_name="llm_debate_edit",
    )

    edit_decision = read_json(round_dir / "edit_decision.json", {})
    if not isinstance(edit_decision, Mapping):
        raise ValueError("edit decision must be a JSON object")
    final_edit = edit_decision.get("final_edit")
    if not isinstance(final_edit, Mapping):
        raise ValueError("edit decision missing final_edit")
    applied = apply_edit(current, final_edit, max_sites=args.max_sites)
    if not applied.ok or applied.structure is None:
        raise RuntimeError(f"local edit failed despite debate check: {applied.error} {applied.validation_reasons}")
    edited = applied.structure
    validation = validate_structure(edited, max_sites=args.max_sites)
    if not validation.ok:
        raise RuntimeError(f"edited structure invalid: {validation.reasons}")

    input_path = round_dir / "input.json"
    write_single_structure(input_path, edited)
    shutil.copyfile(input_path, work_dir / "current_structure.json")
    state["current_structure_path"] = str(work_dir / "current_structure.json")

    training_data = Path(args.training_data).resolve() if args.training_data else default_training_data(root)
    ppd_path = Path(args.ppd_path).resolve() if args.ppd_path else default_ppd_path(root)
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
    result = read_json(results_path, {})
    if not isinstance(result, Mapping):
        result = {}
    e_hull = read_first_e_hull(round_dir)
    success = evaluator_success(result, e_hull)
    append_history(
        state,
        round_dir=round_dir,
        before=current,
        after=edited,
        edit=final_edit,
        result=result,
        e_hull=e_hull,
        success=success,
    )
    update_best_state(state, round_dir, edited, e_hull)
    state["success"] = success
    state["status"] = "success" if success else "running"
    state["updated_at_utc"] = utc_now()
    write_json(state_path, state)

    if not args.skip_reflection:
        run_reflection(args, root, memory_dir, state_path, round_dir)
    return success


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    work_dir = (root / args.work_dir).resolve()
    memory_dir = (root / args.memory_dir).resolve()
    state_path = work_dir / "state.json"
    work_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    if args.force_new and state_path.exists():
        archive = work_dir.with_name(f"{work_dir.name}_archive_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        shutil.move(str(work_dir), str(archive))
        work_dir.mkdir(parents=True, exist_ok=True)

    if not state_path.exists():
        training_data = Path(args.training_data).resolve() if args.training_data else default_training_data(root)
        initial = generate_initial_structure(root, training_data, args.seed, args.max_sites)
        current_path = work_dir / "current_structure.json"
        write_single_structure(current_path, initial)
        state = initial_state(work_dir, current_path, args.seed)
        state["initial_structure_summary"] = structure_summary(initial)
        write_json(state_path, state)
        print(f"initialized single-structure evolution at {work_dir}")
        print(f"initial_formula={reduced_formula(initial)}")

    if args.initialize_only:
        print(f"state={state_path}")
        return 0

    completed = 0
    while args.max_rounds == 0 or completed < args.max_rounds:
        state = read_json(state_path, {})
        if isinstance(state, Mapping) and bool(state.get("success")):
            print(f"already successful: {state_path}")
            return 0
        success = run_one_round(args, root, work_dir, memory_dir, state_path)
        completed += 1
        if success:
            print(f"SUN success reached after {completed} new round(s)")
            return 0

    print(f"completed {completed} round(s); success not reached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
