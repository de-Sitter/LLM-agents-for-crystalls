"""Supervisor for recoverable sequential X/Y optimizer relay failures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

from crystal_llm.llm_client import LLMConfig, extract_json_object, make_llm_client
from crystal_llm.memory import write_json
from crystal_llm.run_xy_experience_debate import RECOVERABLE_LLM_EXIT_CODE, command_text


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed-memory", required=True)
    parser.add_argument("--work-prefix", default="xy_runs/experience_xy_sequential_supervised")
    parser.add_argument("--log-prefix", default="controller_logs/experience_xy_sequential_supervised")
    parser.add_argument("--supervisor-log", default=None)
    parser.add_argument("--last-paths-file", default=".log/last_xy_resume_paths.txt")
    parser.add_argument("--supervisor-state", default=".log/xy_sequential_supervisor_state.json")
    parser.add_argument("--max-restarts", type=int, default=0, help="0 means unlimited recoverable restarts.")
    parser.add_argument("--restart-sleep-sec", type=float, default=300.0)
    parser.add_argument("--smoke-timeout", type=int, default=60)
    parser.add_argument("--smoke-retries", type=int, default=1)
    parser.add_argument("--smoke-retry-sleep", type=float, default=2.0)
    parser.add_argument("--smoke-max-attempts", type=int, default=0, help="0 means keep trying until healthy.")
    parser.add_argument("controller_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def strip_managed_controller_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        args = args[1:]
    managed_with_value = {"--work-dir", "--seed-sequential-memory"}
    stripped: list[str] = []
    skip_next = False
    for index, item in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if item in managed_with_value:
            skip_next = True
            continue
        if any(item.startswith(f"{option}=") for option in managed_with_value):
            continue
        stripped.append(item)
    return stripped


def write_last_paths(path: Path, *, work: Path, log: Path, seed: Path, supervisor_log: Path, attempt: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"WORK={work}",
                f"LOG={log}",
                f"SEED={seed}",
                f"SUPERVISOR_LOG={supervisor_log}",
                f"SUPERVISOR_ATTEMPT={attempt}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def smoke_relay(root: Path, *, timeout: int, retries: int, retry_sleep: float, log_dir: Path) -> None:
    config = LLMConfig.from_env(dotenv=root / ".env", role="supervisor_smoke", timeout=timeout, max_tokens=128)
    client = make_llm_client(config, log_dir=log_dir, retries=retries, retry_sleep=retry_sleep)
    text = client.complete_text(
        system="Return exactly one compact JSON object and no prose.",
        user='Return {"ok":true,"test":"xy_supervisor_smoke"}.',
        metadata={"role": "xy_supervisor_smoke"},
        max_tokens=128,
    )
    parsed = extract_json_object(text)
    if not isinstance(parsed, dict) or parsed.get("ok") is not True:
        raise RuntimeError(f"relay smoke returned unexpected payload: {parsed!r}")


def wait_for_healthy_relay(args: argparse.Namespace, root: Path, attempt: int) -> None:
    max_attempts = max(0, int(args.smoke_max_attempts))
    tries = 0
    while True:
        tries += 1
        smoke_log_dir = root / ".log" / "xy_supervisor_smoke" / f"restart_{attempt:03d}_try_{tries:03d}"
        try:
            started = time.time()
            smoke_relay(
                root,
                timeout=args.smoke_timeout,
                retries=args.smoke_retries,
                retry_sleep=args.smoke_retry_sleep,
                log_dir=smoke_log_dir,
            )
            print(
                f"[{utc_now()}] supervisor_smoke_ok attempt={attempt} try={tries} "
                f"elapsed_sec={time.time() - started:.1f}",
                flush=True,
            )
            return
        except Exception as exc:
            print(
                f"[{utc_now()}] supervisor_smoke_failed attempt={attempt} try={tries} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            if max_attempts and tries >= max_attempts:
                raise
            time.sleep(max(1.0, float(args.restart_sleep_sec)))


def run_controller_once(
    *,
    root: Path,
    controller_args: list[str],
    work_dir: Path,
    log_path: Path,
    seed_memory: Path,
) -> int:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "crystal_llm.run_xy_experience_debate",
        "--work-dir",
        str(work_dir),
        "--seed-sequential-memory",
        str(seed_memory),
        *controller_args,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{utc_now()}] supervisor_start_controller command={command_text(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=root,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return process.wait()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    controller_args = strip_managed_controller_args(list(args.controller_args))
    seed_memory = Path(args.seed_memory)
    if not seed_memory.is_absolute():
        seed_memory = root / seed_memory
    supervisor_log = root / args.supervisor_log if args.supervisor_log else root / f"{args.log_prefix}_supervisor_{utc_stamp()}.log"
    state_path = root / args.supervisor_state
    last_paths_file = root / args.last_paths_file
    restart_count = 0

    while True:
        attempt = restart_count + 1
        stamp = utc_stamp()
        work_dir = root / f"{args.work_prefix}_{stamp}_attempt{attempt:03d}"
        log_path = root / f"{args.log_prefix}_{stamp}_attempt{attempt:03d}.log"
        write_last_paths(
            last_paths_file,
            work=work_dir.relative_to(root),
            log=log_path.relative_to(root),
            seed=seed_memory.relative_to(root) if seed_memory.is_relative_to(root) else seed_memory,
            supervisor_log=supervisor_log.relative_to(root),
            attempt=attempt,
        )
        write_json(
            state_path,
            {
                "schema_version": "xy_sequential_supervisor_state.v1",
                "updated_at_utc": utc_now(),
                "status": "running_controller",
                "attempt": attempt,
                "work_dir": str(work_dir),
                "log_path": str(log_path),
                "seed_memory": str(seed_memory),
                "controller_args": controller_args,
            },
        )
        code = run_controller_once(
            root=root,
            controller_args=controller_args,
            work_dir=work_dir,
            log_path=log_path,
            seed_memory=seed_memory,
        )
        if code == 0:
            write_json(
                state_path,
                {
                    "schema_version": "xy_sequential_supervisor_state.v1",
                    "updated_at_utc": utc_now(),
                    "status": "completed",
                    "attempt": attempt,
                    "work_dir": str(work_dir),
                    "log_path": str(log_path),
                    "exit_code": code,
                },
            )
            return 0
        if code != RECOVERABLE_LLM_EXIT_CODE:
            write_json(
                state_path,
                {
                    "schema_version": "xy_sequential_supervisor_state.v1",
                    "updated_at_utc": utc_now(),
                    "status": "stopped_nonrecoverable",
                    "attempt": attempt,
                    "work_dir": str(work_dir),
                    "log_path": str(log_path),
                    "exit_code": code,
                },
            )
            return code

        restart_count += 1
        seed_memory = work_dir / "sequential_memory.json"
        write_json(
            state_path,
            {
                "schema_version": "xy_sequential_supervisor_state.v1",
                "updated_at_utc": utc_now(),
                "status": "waiting_after_recoverable_llm_failure",
                "attempt": attempt,
                "work_dir": str(work_dir),
                "log_path": str(log_path),
                "exit_code": code,
                "next_seed_memory": str(seed_memory),
                "recoverable_pause": str(work_dir / "recoverable_pause.json"),
            },
        )
        if args.max_restarts and restart_count > int(args.max_restarts):
            print(f"[{utc_now()}] supervisor_max_restarts_reached restarts={restart_count}", flush=True)
            return code
        print(
            f"[{utc_now()}] supervisor_recoverable_restart_scheduled "
            f"restart={restart_count} sleep_sec={args.restart_sleep_sec} next_seed={seed_memory}",
            flush=True,
        )
        time.sleep(max(0.0, float(args.restart_sleep_sec)))
        wait_for_healthy_relay(args, root, attempt)


if __name__ == "__main__":
    raise SystemExit(main())
