"""Smoke test for a controller-managed local LLM agent loop.

This module intentionally does not modify the production A/B/C/D/E/F workflow.
It proves the minimal runtime pattern needed for those roles to become local
agents: the model emits structured tool requests, Python executes local tools,
and the tool results are fed back to the model for a final JSON answer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from crystal_llm.llm_client import LLMConfig, LLMError, ResponsesClient, extract_json_object


ARTIFACT_ROOT = Path("agent_artifacts/local_agent_smoke")
ALLOWED_SHELL_COMMANDS = {
    "pwd",
}


@dataclass(frozen=True)
class ToolResult:
    id: str
    name: str
    ok: bool
    output: Mapping[str, Any]


def _safe_relative_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be project-relative and cannot contain '..': {raw_path}")
    return path


def read_file_tool(arguments: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    rel_path = _safe_relative_path(str(arguments.get("path", "")))
    max_chars = int(arguments.get("max_chars", 500))
    max_chars = max(1, min(max_chars, 2000))
    path = root / rel_path
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(rel_path),
        "exists": path.exists(),
        "chars_returned": min(len(text), max_chars),
        "truncated": len(text) > max_chars,
        "text": text[:max_chars],
    }


def write_file_tool(arguments: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    rel_path = _safe_relative_path(str(arguments.get("path", "")))
    artifact_root = ARTIFACT_ROOT
    if not rel_path.parts[: len(artifact_root.parts)] == artifact_root.parts:
        raise ValueError(f"write_file is restricted to {artifact_root}/")
    content = str(arguments.get("content", ""))
    if len(content) > 2000:
        raise ValueError("write_file content exceeds 2000 characters")
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "path": str(rel_path),
        "bytes_written": len(content.encode("utf-8")),
        "exists": path.exists(),
    }


def run_shell_tool(arguments: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    command = str(arguments.get("command", ""))
    if command not in ALLOWED_SHELL_COMMANDS:
        raise ValueError(f"command is not allowed in this smoke test: {command}")
    completed = subprocess.run(
        command,
        cwd=root,
        shell=True,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def execute_tool_call(call: Mapping[str, Any], *, root: Path) -> ToolResult:
    call_id = str(call.get("id") or f"call_{call.get('name', 'unknown')}")
    name = str(call.get("name", ""))
    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {}
    try:
        if name == "read_file":
            output = read_file_tool(arguments, root=root)
        elif name == "write_file":
            output = write_file_tool(arguments, root=root)
        elif name == "run_shell":
            output = run_shell_tool(arguments, root=root)
        else:
            raise ValueError(f"unknown tool: {name}")
    except Exception as exc:
        return ToolResult(id=call_id, name=name, ok=False, output={"error": f"{type(exc).__name__}: {exc}"})
    return ToolResult(id=call_id, name=name, ok=True, output=output)


def coerce_protocol_payload(parsed: Any) -> dict[str, Any]:
    """Accept common wrapper variants from LLMs while preserving the protocol."""

    if not isinstance(parsed, Mapping):
        raise ValueError("LLM response is not a JSON object")
    payload = dict(parsed)
    if isinstance(payload.get("tool_request"), Mapping):
        payload = dict(payload["tool_request"])
    elif isinstance(payload.get("final"), Mapping):
        payload = dict(payload["final"])
    return payload


def _tool_schema_text() -> str:
    return json.dumps(
        {
            "tool_request": {
                "status": "tool_request",
                "tool_calls": [
                    {"id": "read-readme", "name": "read_file", "arguments": {"path": "README.md", "max_chars": 300}},
                    {
                        "id": "write-artifact",
                        "name": "write_file",
                        "arguments": {
                            "path": "agent_artifacts/local_agent_smoke/agent_wrote.txt",
                            "content": "local-agent-smoke-ok",
                        },
                    },
                    {"id": "pwd", "name": "run_shell", "arguments": {"command": "pwd"}},
                ],
            },
            "final": {
                "status": "done",
                "can_act_locally": True,
                "read_file_observed": True,
                "write_file_confirmed": True,
                "shell_observed": True,
                "summary": "one concise sentence",
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def run_smoke_agent(
    *,
    root: Path,
    dotenv: Path,
    model: str | None,
    timeout: int,
    max_tokens: int,
    log_dir: Path,
    max_steps: int,
) -> dict[str, Any]:
    client = ResponsesClient(
        LLMConfig.from_env(dotenv=dotenv, role="smoke", model=model, timeout=timeout, max_tokens=max_tokens),
        log_dir=log_dir,
        retries=1,
        retry_sleep=2,
    )
    system = (
        "You are a local-agent smoke-test role. Return JSON only. "
        "You cannot access files or run commands yourself; instead request tools with the exact JSON protocol. "
        "After tool results are supplied, return the final JSON object."
    )
    transcript = (
        "Goal: prove that an LLM role can act through a local Python agent runtime.\n"
        "First response requirement: request exactly these three tools: read README.md, write the smoke artifact, and run pwd.\n"
        "Second response requirement: after tool_results are provided, return final JSON with status='done' and booleans confirming the three local actions.\n"
        "Protocol examples and exact tool arguments:\n"
        f"{_tool_schema_text()}\n"
    )
    tool_history: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    for step in range(max_steps):
        text = client.complete_text(
            system=system,
            user=transcript,
            temperature=0,
            max_tokens=max_tokens,
            metadata={"role": "local_agent_smoke", "step": step},
        )
        parsed = extract_json_object(text)
        response_obj = coerce_protocol_payload(parsed)
        responses.append(response_obj)
        if response_obj.get("status") == "done":
            return {
                "ok": True,
                "final": response_obj,
                "tool_history": tool_history,
                "responses": responses,
            }
        if response_obj.get("status") != "tool_request":
            raise ValueError(f"step {step} returned unsupported status: {response_obj.get('status')!r}")
        calls = response_obj.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError(f"step {step} did not include tool_calls")
        results = [execute_tool_call(call, root=root) for call in calls if isinstance(call, Mapping)]
        result_payload = [
            {"id": result.id, "name": result.name, "ok": result.ok, "output": dict(result.output)}
            for result in results
        ]
        tool_history.extend(result_payload)
        transcript += "\nTool results from local Python runtime:\n"
        transcript += json.dumps(result_payload, ensure_ascii=False, indent=2)
        transcript += "\nNow return the final JSON object; do not request more tools unless a required action failed.\n"
    raise ValueError(f"agent did not finish within {max_steps} steps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal local-agent tool-use smoke test.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--log-dir", type=Path, default=Path("agent_artifacts/local_agent_smoke/llm_calls"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    log_dir = (root / args.log_dir).resolve() if not args.log_dir.is_absolute() else args.log_dir
    try:
        result = run_smoke_agent(
            root=root,
            dotenv=args.dotenv if args.dotenv.is_absolute() else root / args.dotenv,
            model=args.model,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            log_dir=log_dir,
            max_steps=args.max_steps,
        )
    except (LLMError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
