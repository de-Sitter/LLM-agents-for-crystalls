"""Controller-managed local agent runtime for LLM roles.

The project LLM calls are text-only Responses API calls.  This runtime gives a
role local agency without requiring provider-native tool calling: the model can
return a JSON tool request, Python executes a small set of local tools, and the
tool results are appended to the next model call.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any, Mapping, Sequence

from crystal_llm.llm_client import LLMError, extract_json_object
from crystal_llm.material_physics_schema import query_matches, select_matches, validate_query


DEFAULT_ALLOWED_COMMANDS = {"pwd", "rg", "ls", "sed", "head", "tail", "wc"}
SENSITIVE_PATH_MARKERS = {
    ".env",
    ".log",
    "authorized_keys",
    "credentials",
    "credential",
    "passwd",
    "password",
    "secret",
    "token",
}
COMPACT_RECORD_KEYS = (
    "material_id",
    "formula",
    "elements",
    "nelements",
    "nsites",
    "crystal_system",
    "spacegroup_number",
    "band_gap",
    "formation_energy_per_atom",
    "density",
    "volume_per_atom",
)
CANDIDATE_TOOL_EXAMPLE_LIMIT = 4
CANDIDATE_TOOL_DISTINCT_EXAMPLE_LIMIT = 6
CANDIDATE_TOOL_DISTINCT_SCAN_LIMIT = 80
PROMPT_TOOL_RESULT_CHAR_CAP = 2400
GENERATED_FORMULA_ERROR_RE = re.compile(
    r"\bgenerated\s+([A-Z][A-Za-z0-9()]+)(?:\s+with template\s+([A-Za-z0-9_]+))?\b"
)
GENERATED_FORMULA_EXCLUDED_RE = re.compile(r"\bgenerated\s+formula\s+([A-Z][A-Za-z0-9()]+)\b")
QUERY_ARGUMENT_KEYS = {
    "material_ids",
    "formulas",
    "elements_all",
    "elements_any",
    "elements_none",
    "nelements_min",
    "nelements_max",
    "nsites_min",
    "nsites_max",
    "density_min",
    "density_max",
    "volume_per_atom_min",
    "volume_per_atom_max",
    "band_gap_min",
    "band_gap_max",
    "formation_energy_per_atom_min",
    "formation_energy_per_atom_max",
    "crystal_system",
    "spacegroup_number",
    "preferred_order",
}
EXPLORATION_ANIONS = {"O", "F", "S", "N", "P", "Se", "Te", "Cl", "Br", "I"}
RARE_EARTH_ELEMENTS = {
    "Sc",
    "Y",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
}
ALKALI_ELEMENTS = {"Li", "Na", "K", "Rb", "Cs"}
ALKALINE_EARTH_ELEMENTS = {"Be", "Mg", "Ca", "Sr", "Ba"}
TRANSITION_METAL_ELEMENTS = {
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
}


@dataclass
class ToolResult:
    id: str
    name: str
    ok: bool
    output: Mapping[str, Any]


def _tool_output_declares_error(output: Mapping[str, Any]) -> bool:
    status = str(output.get("tool_status") or "").strip().lower()
    if status == "error":
        return True
    if "error" in output and len(output) <= 2 and "tool_status" not in output:
        return True
    return False


def _mandatory_tool_result_ok(result: ToolResult, expected_name: str) -> bool:
    return result.name == expected_name and result.ok and not _tool_output_declares_error(result.output)


def _coerce_tool_request_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Accept common single-tool wrappers emitted by models.

    The runtime advertises a strict ``status/tool_calls`` protocol, but some
    models naturally return a compact wrapper such as
    ``{"tool_request": {"tool": "...", "arguments": {...}}}``. Coercing that
    shape keeps the controller-managed tool loop provider-agnostic while still
    executing only known tools through ``execute_tool_call``.
    """

    if isinstance(payload.get("tool_calls"), list):
        return {"status": "tool_request", "tool_calls": payload.get("tool_calls")}

    tool_name = str(payload.get("name") or payload.get("tool") or "").strip()
    if not tool_name:
        return dict(payload)
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {}
    return {"status": "tool_request", "tool_calls": _expand_single_tool_call(tool_name, arguments)}


def _expand_single_tool_call(tool_name: str, arguments: Mapping[str, Any]) -> list[dict[str, Any]]:
    if tool_name != "query_candidate_pool":
        return [{"id": str(arguments.get("id") or tool_name), "name": tool_name, "arguments": dict(arguments)}]

    calls: list[dict[str, Any]] = []
    raw_queries = arguments.get("queries")
    if isinstance(raw_queries, list):
        for index, item in enumerate(raw_queries, start=1):
            if not isinstance(item, Mapping):
                continue
            calls.append(_candidate_pool_call_from_item(item, index))
        if calls:
            return calls

    branches = arguments.get("branches")
    if isinstance(branches, Mapping):
        for index, (branch_name, item) in enumerate(branches.items(), start=1):
            if not isinstance(item, Mapping):
                continue
            call = _candidate_pool_call_from_item(item, index)
            call["id"] = str(item.get("id") or branch_name or call["id"])
            calls.append(call)
        if calls:
            return calls

    return [{"id": str(arguments.get("id") or tool_name), "name": tool_name, "arguments": dict(arguments)}]


def _candidate_pool_call_from_item(item: Mapping[str, Any], index: int) -> dict[str, Any]:
    query = item.get("query")
    if not isinstance(query, Mapping):
        query = {key: item[key] for key in QUERY_ARGUMENT_KEYS if key in item}
    if not isinstance(query, Mapping) or not query:
        predicate = item.get("predicate")
        query = dict(predicate) if isinstance(predicate, Mapping) else {}
    count = item.get("count", item.get("limit", item.get("requested_examples", 10)))
    seed = item.get("seed")
    arguments: dict[str, Any] = {"query": dict(query), "count": count}
    if seed is not None:
        arguments["seed"] = seed
    return {
        "id": str(item.get("id") or item.get("name") or f"query-{index}"),
        "name": "query_candidate_pool",
        "arguments": arguments,
    }


def _requires_mandatory_candidate_pool(role_name: str, prompt_text: str) -> bool:
    gated_prefixes = ("prediction_agent_c", "prediction_agent_d", "execution_agent_e", "execution_agent_f")
    if not any(role_name == prefix or role_name.startswith(f"{prefix}_") for prefix in gated_prefixes):
        return False
    if "MATTERGEN_NATIVE_NO_MANDATORY_MP_POOL" in prompt_text:
        return False
    return "query_candidate_pool" in prompt_text


def _candidate_pool_calls_from_prompt(prompt_text: str, *, max_calls: int) -> list[dict[str, Any]]:
    parsed_blocks = _json_values_from_fenced_blocks(prompt_text)
    calls: list[dict[str, Any]] = []
    for block in parsed_blocks:
        _collect_candidate_pool_calls(block, calls)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in calls:
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            continue
        query = arguments.get("query")
        if not isinstance(query, Mapping) or not query:
            continue
        key = json.dumps({"query": dict(query), "count": arguments.get("count")}, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(call)
    if len(deduped) > max_calls:
        return deduped[-max_calls:]
    return deduped


def _json_values_from_fenced_blocks(text: str) -> list[Any]:
    values: list[Any] = []
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        raw = match.group(1).strip()
        if not raw:
            continue
        try:
            values.append(json.loads(raw))
        except Exception:
            continue
    return values


def _collect_candidate_pool_calls(value: Any, calls: list[dict[str, Any]], *, prefix: str = "context") -> None:
    if isinstance(value, list):
        for index, item in enumerate(value, start=1):
            _collect_candidate_pool_calls(item, calls, prefix=f"{prefix}-{index}")
        return
    if not isinstance(value, Mapping):
        return

    primary_query = value.get("primary_query")
    if isinstance(primary_query, Mapping):
        calls.append(_candidate_pool_query_call(f"{prefix}-primary", primary_query, value.get("primary_count", 5)))
    control_query = value.get("control_query")
    if isinstance(control_query, Mapping):
        calls.append(_candidate_pool_query_call(f"{prefix}-control", control_query, value.get("control_count", 5)))

    for branch_name in ("primary", "control"):
        branch = value.get(branch_name)
        if not isinstance(branch, Mapping):
            continue
        query = branch.get("query")
        if isinstance(query, Mapping) and str(branch.get("source") or "mp_pool").strip().lower() == "mp_pool":
            calls.append(_candidate_pool_query_call(f"{prefix}-{branch_name}", query, branch.get("count", 5)))

    for key, item in value.items():
        _collect_candidate_pool_calls(item, calls, prefix=f"{prefix}-{key}")


def _candidate_pool_query_call(call_id: str, query: Mapping[str, Any], count: Any) -> dict[str, Any]:
    try:
        parsed_count = int(count)
    except Exception:
        parsed_count = 5
    return {
        "id": call_id,
        "name": "query_candidate_pool",
        "arguments": {"query": dict(query), "count": max(1, parsed_count)},
    }


@dataclass
class LocalAgentRuntime:
    """Run local tool-use loops around a text-only LLM client."""

    root: Path
    trace_dir: Path
    writable_dir: Path
    candidate_pool_path: Path | None = None
    state_path: Path | None = None
    xy_history_path: Path | None = None
    max_steps: int = 12
    max_tool_calls_per_step: int = 8
    max_tool_result_chars: int = 6000
    command_timeout: int = 15
    allowed_commands: set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED_COMMANDS))
    allow_project_writes: bool = False
    allow_candidate_pool_tools: bool = True

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.trace_dir = self.trace_dir.resolve()
        self.writable_dir = self.writable_dir.resolve()
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.writable_dir.mkdir(parents=True, exist_ok=True)
        self._candidate_pool_cache: list[dict[str, Any]] | None = None

    def complete_text(
        self,
        client: Any,
        *,
        system: str,
        user: str,
        metadata: Mapping[str, Any],
    ) -> str:
        """Return final model text after satisfying any local tool requests."""

        augmented_system = self._augment_system(system)
        transcript = user
        trace: dict[str, Any] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "metadata": _safe_json(metadata),
            "tool_steps": [],
        }
        last_text = ""
        role_name = str(metadata.get("role") or "").strip().lower()
        mandatory_mechanism_rag = _requires_mandatory_mechanism_rag(user)
        mechanism_rag_satisfied = not mandatory_mechanism_rag
        missing_rag_retry_sent = False
        forced_mandatory_rag_sent = False
        mandatory_xy_generation_rag = _requires_mandatory_xy_generation_rag(user)
        xy_generation_rag_satisfied = not mandatory_xy_generation_rag
        missing_xy_generation_rag_retry_sent = False
        forced_xy_generation_rag_sent = False
        mandatory_candidate_pool = self.allow_candidate_pool_tools and _requires_mandatory_candidate_pool(role_name, f"{system}\n{user}")
        candidate_pool_satisfied = not mandatory_candidate_pool
        missing_candidate_pool_retry_sent = False
        forced_candidate_pool_sent = False
        seen_mechanism_evidence_calls: set[str] = set()
        started = time.time()
        if mandatory_xy_generation_rag:
            prefill_calls: list[dict[str, Any]] = []
            if mandatory_mechanism_rag:
                prefill_calls.append(
                    {
                        "id": "mandatory-evidence-0",
                        "name": "summarize_mechanism_evidence",
                        "arguments": {"limit": 6, "offset": 0, "include_failed": True, "brief": True},
                    }
                )
            prefill_calls.append(
                {
                    "id": "mandatory-xy-history-0",
                    "name": "summarize_xy_generation_history",
                    "arguments": {"limit": 12, "offset": 0, "include_failed": True},
                }
            )
            prefill_results = [
                self._execute_tool_call_with_session_state(call, seen_mechanism_evidence_calls)
                for call in prefill_calls
            ]
            result_payload = [
                {"id": result.id, "name": result.name, "ok": result.ok, "output": dict(result.output)}
                for result in prefill_results
            ]
            if any(_mandatory_tool_result_ok(result, "summarize_mechanism_evidence") for result in prefill_results):
                mechanism_rag_satisfied = True
                forced_mandatory_rag_sent = True
                trace["prefilled_mandatory_mechanism_rag"] = True
            if any(_mandatory_tool_result_ok(result, "summarize_xy_generation_history") for result in prefill_results):
                xy_generation_rag_satisfied = True
                forced_xy_generation_rag_sent = True
                trace["prefilled_mandatory_xy_generation_rag"] = True
            trace["tool_steps"].append(
                {
                    "step": -1,
                    "prefilled": True,
                    "reason": "mandatory_xy_sequential_rag",
                    "requested_count": len(prefill_calls),
                    "executed_count": len(result_payload),
                    "results": _truncate_for_trace(result_payload, max_chars=self.max_tool_result_chars),
                }
            )
            transcript = (
                self._append_tool_results(transcript, result_payload)
                + "\nMANDATORY_XY_SEQUENTIAL_RAG_PREFILLED_BY_CONTROLLER:\n"
                "The controller attempted to supply the required compact mechanism and X/Y generation-history summaries. "
                "Use successful tool results directly; if a required summary is missing, the controller will request or force that missing summary before final JSON. "
                "Use this evidence directly in the final role JSON. Do not request RAG tools, do not output `rag_required`, "
                "and do not ask the controller to provide these summaries again.\n"
            )
        try:
            for step in range(max(0, self.max_steps) + 1):
                call_metadata = {**dict(metadata), "local_agent_step": step}
                text = client.complete_text(system=augmented_system, user=transcript, metadata=call_metadata)
                last_text = text
                trailing_final = self.parse_trailing_final_after_tool_request(text)
                if step > 0 and trailing_final is not None:
                    if not mechanism_rag_satisfied:
                        if not missing_rag_retry_sent:
                            missing_rag_retry_sent = True
                            trace["mandatory_mechanism_rag_retry"] = True
                            transcript = self._append_mandatory_rag_missing(transcript)
                            continue
                        if not forced_mandatory_rag_sent:
                            forced_mandatory_rag_sent = True
                            mechanism_rag_satisfied = True
                            include_xy_history = mandatory_xy_generation_rag and not xy_generation_rag_satisfied
                            if include_xy_history:
                                forced_xy_generation_rag_sent = True
                                xy_generation_rag_satisfied = True
                            transcript = self._force_mandatory_mechanism_rag(
                                transcript,
                                trace,
                                seen_mechanism_evidence_calls,
                                step=step,
                                include_xy_generation_history=include_xy_history,
                            )
                            continue
                        raise LLMError("mechanism role returned final JSON before summarize_mechanism_evidence RAG")
                    if not candidate_pool_satisfied:
                        if not missing_candidate_pool_retry_sent:
                            missing_candidate_pool_retry_sent = True
                            trace["mandatory_candidate_pool_retry"] = True
                            transcript = self._append_candidate_pool_missing(transcript, role_name)
                            continue
                        if not forced_candidate_pool_sent:
                            forced_candidate_pool_sent = True
                            transcript = self._force_mandatory_candidate_pool(
                                transcript,
                                trace,
                                f"{system}\n{user}",
                                step=step,
                            )
                            candidate_pool_satisfied = True
                            continue
                        raise LLMError(f"{role_name or 'role'} returned final JSON before query_candidate_pool")
                    if not xy_generation_rag_satisfied:
                        if not missing_xy_generation_rag_retry_sent:
                            missing_xy_generation_rag_retry_sent = True
                            trace["mandatory_xy_generation_rag_retry"] = True
                            transcript = self._append_xy_generation_rag_missing(transcript)
                            continue
                        if not forced_xy_generation_rag_sent:
                            forced_xy_generation_rag_sent = True
                            xy_generation_rag_satisfied = True
                            transcript = self._force_mandatory_xy_generation_rag(transcript, trace, step=step)
                            continue
                        raise LLMError("sequential X/Y role returned final JSON before summarize_xy_generation_history RAG")
                    trace["final_status"] = "trailing_final_after_tool_results"
                    return json.dumps(trailing_final, ensure_ascii=False)
                request = self.parse_tool_request(text)
                if request is None:
                    if not mechanism_rag_satisfied:
                        if not missing_rag_retry_sent:
                            missing_rag_retry_sent = True
                            trace["mandatory_mechanism_rag_retry"] = True
                            transcript = self._append_mandatory_rag_missing(transcript)
                            continue
                        if not forced_mandatory_rag_sent:
                            forced_mandatory_rag_sent = True
                            mechanism_rag_satisfied = True
                            include_xy_history = mandatory_xy_generation_rag and not xy_generation_rag_satisfied
                            if include_xy_history:
                                forced_xy_generation_rag_sent = True
                                xy_generation_rag_satisfied = True
                            transcript = self._force_mandatory_mechanism_rag(
                                transcript,
                                trace,
                                seen_mechanism_evidence_calls,
                                step=step,
                                include_xy_generation_history=include_xy_history,
                            )
                            continue
                        raise LLMError("mechanism role returned final text before summarize_mechanism_evidence RAG")
                    if not candidate_pool_satisfied:
                        if not missing_candidate_pool_retry_sent:
                            missing_candidate_pool_retry_sent = True
                            trace["mandatory_candidate_pool_retry"] = True
                            transcript = self._append_candidate_pool_missing(transcript, role_name)
                            continue
                        if not forced_candidate_pool_sent:
                            forced_candidate_pool_sent = True
                            transcript = self._force_mandatory_candidate_pool(
                                transcript,
                                trace,
                                f"{system}\n{user}",
                                step=step,
                            )
                            candidate_pool_satisfied = True
                            continue
                        raise LLMError(f"{role_name or 'role'} returned final text before query_candidate_pool")
                    if not xy_generation_rag_satisfied:
                        if not missing_xy_generation_rag_retry_sent:
                            missing_xy_generation_rag_retry_sent = True
                            trace["mandatory_xy_generation_rag_retry"] = True
                            transcript = self._append_xy_generation_rag_missing(transcript)
                            continue
                        if not forced_xy_generation_rag_sent:
                            forced_xy_generation_rag_sent = True
                            xy_generation_rag_satisfied = True
                            transcript = self._force_mandatory_xy_generation_rag(transcript, trace, step=step)
                            continue
                        raise LLMError("sequential X/Y role returned final text before summarize_xy_generation_history RAG")
                    trace["final_status"] = "model_final"
                    return text
                if step >= self.max_steps:
                    raise LLMError(f"local agent exceeded max_steps={self.max_steps} while requesting tools")
                calls = request.get("tool_calls")
                if not isinstance(calls, list) or not calls:
                    raise LLMError("local agent tool_request did not include non-empty tool_calls")
                limited_calls = [call for call in calls if isinstance(call, Mapping)][: self.max_tool_calls_per_step]
                results = [
                    self._execute_tool_call_with_session_state(call, seen_mechanism_evidence_calls)
                    for call in limited_calls
                ]
                result_payload = [
                    {"id": item.id, "name": item.name, "ok": item.ok, "output": dict(item.output)}
                    for item in results
                ]
                if mandatory_mechanism_rag and any(
                    _mandatory_tool_result_ok(item, "summarize_mechanism_evidence") for item in results
                ):
                    mechanism_rag_satisfied = True
                if mandatory_candidate_pool and any(item.name == "query_candidate_pool" and item.ok for item in results):
                    candidate_pool_satisfied = True
                if mandatory_xy_generation_rag and any(
                    _mandatory_tool_result_ok(item, "summarize_xy_generation_history") for item in results
                ):
                    xy_generation_rag_satisfied = True
                trace["tool_steps"].append(
                    {
                        "step": step,
                        "requested_count": len(calls),
                        "executed_count": len(result_payload),
                        "results": _truncate_for_trace(result_payload, max_chars=self.max_tool_result_chars),
                    }
                )
                transcript = self._append_tool_results(transcript, result_payload)
                if trailing_final is not None:
                    trace["ignored_trailing_final_after_tool_request"] = True
                    transcript += (
                        "\nLOCAL_AGENT_JSON_BOUNDARY_REPAIR:\n"
                        "Your previous response included a tool_request JSON object immediately followed by a final JSON object. "
                        "The controller executed only the leading tool_request and ignored the trailing final JSON. "
                        "Use the tool results above and return exactly one JSON object now: either one focused tool_request if more "
                        "evidence is essential, or the final role JSON. Do not concatenate two top-level JSON objects.\n"
                    )
            raise LLMError("local agent loop ended without a final response")
        finally:
            trace["elapsed_seconds"] = round(time.time() - started, 3)
            trace["last_text_preview"] = _short_text(last_text, 1000)
            self._write_trace(trace)

    def parse_tool_request(self, text: str) -> dict[str, Any] | None:
        try:
            parsed = extract_json_object(text)
        except Exception:
            parsed = _parse_first_json_value(text)
        if not isinstance(parsed, Mapping):
            return None
        payload = dict(parsed)
        if isinstance(payload.get("tool_request"), Mapping):
            payload = dict(payload["tool_request"])
        if str(payload.get("status") or "").strip().lower() == "tool_request":
            return _coerce_tool_request_payload(payload)
        if "tool_calls" in payload and str(payload.get("status") or "").strip().lower() in {"", "needs_tools"}:
            return _coerce_tool_request_payload({"status": "tool_request", "tool_calls": payload.get("tool_calls")})
        if str(payload.get("type") or "").strip().lower() == "tool_request":
            return _coerce_tool_request_payload(payload)
        if payload.get("tool") or payload.get("name"):
            return _coerce_tool_request_payload({"status": "tool_request", **payload})
        return None

    def parse_trailing_final_after_tool_request(self, text: str) -> Any | None:
        values = _parse_leading_json_values(text, max_values=2)
        if len(values) < 2:
            return None
        first, second = values[0], values[1]
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            return None
        first_payload = dict(first.get("tool_request")) if isinstance(first.get("tool_request"), Mapping) else dict(first)
        if str(first_payload.get("status") or "").strip().lower() != "tool_request":
            return None
        second_payload = dict(second.get("final")) if isinstance(second.get("final"), Mapping) else dict(second)
        if str(second_payload.get("status") or "").strip().lower() == "tool_request":
            return None
        return second_payload

    def execute_tool_call(self, call: Mapping[str, Any]) -> ToolResult:
        call_id = str(call.get("id") or f"call_{call.get('name', 'unknown')}")
        name = str(call.get("name") or call.get("tool") or "").strip()
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            arguments = {}
        try:
            if name == "read_project_file":
                output = self._read_project_file(arguments)
            elif name == "read_pdf_text":
                output = self._read_pdf_text(arguments)
            elif name == "search_project":
                output = self._search_project(arguments)
            elif name == "list_project_files":
                output = self._list_project_files(arguments)
            elif name == "summarize_round_history":
                output = self._summarize_round_history(arguments)
            elif name == "summarize_mechanism_evidence":
                try:
                    output = self._summarize_mechanism_evidence(arguments)
                except Exception as exc:
                    output = self._summarize_mechanism_evidence_fallback(arguments, exc)
            elif name == "summarize_xy_generation_history":
                output = self._summarize_xy_generation_history(arguments)
            elif name == "query_candidate_pool":
                if not self.allow_candidate_pool_tools:
                    raise ValueError("query_candidate_pool is disabled for this local-agent run")
                output = self._query_candidate_pool(arguments)
            elif name == "run_shell":
                output = self._run_shell(arguments)
            elif name == "write_agent_artifact":
                output = self._write_agent_artifact(arguments)
            elif name == "write_project_file":
                output = self._write_project_file(arguments)
            else:
                raise ValueError(f"unknown local agent tool: {name}")
        except Exception as exc:
            return ToolResult(id=call_id, name=name, ok=False, output={"error": f"{type(exc).__name__}: {exc}"})
        limited_output = _limit_tool_output(output, self.max_tool_result_chars)
        if _tool_output_declares_error(limited_output):
            return ToolResult(id=call_id, name=name, ok=False, output=limited_output)
        return ToolResult(id=call_id, name=name, ok=True, output=limited_output)

    def _augment_system(self, system: str) -> str:
        project_write_tool = (
            " write_project_file(path,content)."
            if self.allow_project_writes
            else ""
        )
        candidate_pool_tool = (
            " query_candidate_pool(query,count,seed)."
            if self.allow_candidate_pool_tools
            else ""
        )
        candidate_pool_guidance = (
            "pool checks, "
            if self.allow_candidate_pool_tools
            else ""
        )
        return (
            f"{system}\n\n"
            "LOCAL_AGENT_RUNTIME: controller tools are available for RAG, "
            f"{candidate_pool_guidance}and light inspection. If evidence is needed or mandated, return exactly one JSON "
            "tool request, then after LOCAL_AGENT_TOOL_RESULTS return the final role JSON. "
            "Do not claim tools are unavailable. Do not concatenate a tool_request JSON object with final JSON.\n"
            "Tool request shape:\n"
            '{"status":"tool_request","tool_calls":[{"id":"short-id","name":"tool_name","arguments":{}}]}\n'
            "Tools: read_project_file(path,max_chars); read_pdf_text(path,pattern,max_chars,max_matches); "
            "search_project(pattern,path,glob,max_matches); list_project_files(glob,max_files); "
            "summarize_round_history(limit,status); summarize_mechanism_evidence(limit,offset,include_failed,brief); "
            "summarize_xy_generation_history(limit,offset,include_failed);"
            f"{candidate_pool_tool}"
            " run_shell(command); write_agent_artifact(path,content)."
            f"{project_write_tool}"
            " Do not request secrets/.env/credentials. Keep final JSON compact and evidence-backed.\n"
        )

    def _append_tool_results(self, transcript: str, results: Sequence[Mapping[str, Any]]) -> str:
        prompt_results = _compact_tool_results_for_prompt(
            results,
            max_chars=max(800, min(self.max_tool_result_chars, PROMPT_TOOL_RESULT_CHAR_CAP)),
        )
        return (
            f"{transcript}\n\n"
            "LOCAL_AGENT_TOOL_RESULTS:\n"
            "```json\n"
            f"{json.dumps(prompt_results, ensure_ascii=False)}\n"
            "```\n"
            "Use these results as evidence. If needed, request one focused new tool batch; otherwise return final role JSON. "
            "Return exactly one top-level JSON object; do not concatenate a tool_request object and a final role object.\n"
        )

    def _append_mandatory_rag_missing(self, transcript: str) -> str:
        return (
            f"{transcript}\n\n"
            "MANDATORY_MECHANISM_RAG_MISSING:\n"
            "Before final JSON, request summarize_mechanism_evidence limit 6 offset 0 include_failed true brief true. "
            "Optionally add summarize_round_history or targeted file/search tools. Then write final JSON.\n"
        )

    def _append_candidate_pool_missing(self, transcript: str, role_name: str) -> str:
        return (
            f"{transcript}\n\n"
            "MANDATORY_CANDIDATE_POOL_CHECK_MISSING:\n"
            f"Role `{role_name or 'unknown'}` must query candidate-pool availability before final prediction/execution JSON. "
            "Use this shape and adapt predicates:\n"
            '{"status":"tool_request","tool_calls":[{"id":"primary","name":"query_candidate_pool","arguments":{"query":{"elements_all":["O","F"],"elements_any":["Li","Na"]},"count":5}},{"id":"control","name":"query_candidate_pool","arguments":{"query":{"elements_all":["O"],"elements_any":["Li","Na"]},"count":5}}]}\n'
            "After results, write final role JSON with compact evidence.\n"
        )

    def _append_xy_generation_rag_missing(self, transcript: str) -> str:
        return (
            f"{transcript}\n\n"
            "MANDATORY_XY_GENERATION_HISTORY_RAG_MISSING:\n"
            "Sequential X/Y optimization roles must inspect their own previous material-generation history before returning final JSON. "
            "Request summarize_xy_generation_history now with limit 12, offset 0, and include_failed true. Use the results to decide what to exploit, avoid, or change next.\n"
        )

    def _force_mandatory_mechanism_rag(
        self,
        transcript: str,
        trace: dict[str, Any],
        seen_mechanism_evidence_calls: set[str],
        *,
        step: int,
        include_xy_generation_history: bool = False,
    ) -> str:
        """Guarantee mechanism roles see the mandatory evidence page.

        Some providers/models ignore the JSON tool protocol and answer that
        tools are unavailable. The science requirement is that mechanism agents
        inspect historical evidence before final JSON; after one explicit retry,
        the controller supplies the required first evidence page so the next
        model turn can reason from it instead of burning rounds.
        """

        mechanism_call = {
            "id": "mandatory-evidence-0",
            "name": "summarize_mechanism_evidence",
            "arguments": {"limit": 6, "offset": 0, "include_failed": True, "brief": True},
        }
        results = [self._execute_tool_call_with_session_state(mechanism_call, seen_mechanism_evidence_calls)]
        if include_xy_generation_history:
            xy_call = {
                "id": "mandatory-xy-history-0",
                "name": "summarize_xy_generation_history",
                "arguments": {"limit": 12, "offset": 0, "include_failed": True},
            }
            results.append(self.execute_tool_call(xy_call))
        result_payload = [
            {"id": result.id, "name": result.name, "ok": result.ok, "output": dict(result.output)}
            for result in results
        ]
        trace["forced_mandatory_mechanism_rag"] = True
        if include_xy_generation_history:
            trace["forced_mandatory_xy_generation_rag"] = True
        trace["tool_steps"].append(
            {
                "step": step,
                "forced": True,
                "requested_count": len(result_payload),
                "executed_count": len(result_payload),
                "results": _truncate_for_trace(result_payload, max_chars=self.max_tool_result_chars),
            }
        )
        extra_note = (
            " The controller also supplied the mandatory X/Y generation-history page in the same forced batch."
            if include_xy_generation_history
            else ""
        )
        return (
            self._append_tool_results(transcript, result_payload)
            + "\nMANDATORY_MECHANISM_RAG_FORCED_BY_CONTROLLER:\n"
            "The controller supplied the required first evidence page because your previous response did not emit "
            "the JSON tool_request protocol."
            f"{extra_note} Use this evidence in the final role JSON. You may request another "
            "different evidence page if needed, but do not claim tools are unavailable.\n"
        )

    def _force_mandatory_xy_generation_rag(
        self,
        transcript: str,
        trace: dict[str, Any],
        *,
        step: int,
    ) -> str:
        call = {
            "id": "mandatory-xy-history-0",
            "name": "summarize_xy_generation_history",
            "arguments": {"limit": 12, "offset": 0, "include_failed": True},
        }
        result = self.execute_tool_call(call)
        result_payload = [{"id": result.id, "name": result.name, "ok": result.ok, "output": dict(result.output)}]
        trace["forced_mandatory_xy_generation_rag"] = True
        trace["tool_steps"].append(
            {
                "step": step,
                "forced": True,
                "reason": "mandatory_xy_generation_history_rag",
                "requested_count": 1,
                "executed_count": 1,
                "results": _truncate_for_trace(result_payload, max_chars=self.max_tool_result_chars),
            }
        )
        return (
            self._append_tool_results(transcript, result_payload)
            + "\nThe controller supplied the mandatory X/Y generation-history page because your previous response did not emit "
            "the JSON tool_request protocol. Use this history in the final role JSON.\n"
        )

    def _force_mandatory_candidate_pool(
        self,
        transcript: str,
        trace: dict[str, Any],
        prompt_text: str,
        *,
        step: int,
    ) -> str:
        calls = _candidate_pool_calls_from_prompt(prompt_text, max_calls=self.max_tool_calls_per_step)
        if not calls:
            raise LLMError("mandatory query_candidate_pool check could not be inferred from prompt context")
        results = [self.execute_tool_call(call) for call in calls]
        result_payload = [
            {"id": item.id, "name": item.name, "ok": item.ok, "output": dict(item.output)}
            for item in results
        ]
        trace["forced_mandatory_candidate_pool"] = True
        trace["tool_steps"].append(
            {
                "step": step,
                "forced": True,
                "requested_count": len(calls),
                "executed_count": len(result_payload),
                "results": _truncate_for_trace(result_payload, max_chars=self.max_tool_result_chars),
            }
        )
        return (
            self._append_tool_results(transcript, result_payload)
            + "\nMANDATORY_CANDIDATE_POOL_FORCED_BY_CONTROLLER:\n"
            "The controller supplied candidate-pool query results inferred from the current proposal/context because "
            "your previous response did not emit the JSON tool_request protocol. Use these results in the final JSON. "
            "You may request another focused query_candidate_pool batch if needed, but do not claim tools are unavailable.\n"
        )

    def _execute_tool_call_with_session_state(
        self,
        call: Mapping[str, Any],
        seen_mechanism_evidence_calls: set[str],
    ) -> ToolResult:
        name = str(call.get("name") or call.get("tool") or "").strip()
        if name != "summarize_mechanism_evidence":
            return self.execute_tool_call(call)
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            arguments = {}
        key = json.dumps(_mechanism_evidence_call_key(arguments), ensure_ascii=False, sort_keys=True)
        if key in seen_mechanism_evidence_calls:
            return ToolResult(
                id=str(call.get("id") or "duplicate_mechanism_evidence"),
                name=name,
                ok=True,
                output={
                    "duplicate_request": True,
                    "message": (
                        "This effective summarize_mechanism_evidence page was already returned in this agent turn. "
                        "Use a different offset/page or focused read/search tools for new evidence; otherwise synthesize the final JSON now."
                    ),
                    "repeated_arguments": dict(arguments),
                },
            )
        seen_mechanism_evidence_calls.add(key)
        return self.execute_tool_call(call)

    def _safe_project_path(self, raw_path: Any, *, must_exist: bool = True) -> Path:
        raw_text = str(raw_path or "").strip()
        if not raw_text:
            raise ValueError("path is required")
        path = Path(raw_text)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("path must be project-relative and cannot contain '..'")
        if _looks_sensitive_path(path):
            raise ValueError("refusing to read sensitive path")
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("path escapes project root")
        if must_exist and not resolved.exists():
            raise FileNotFoundError(str(path))
        return resolved

    def _read_project_file(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = self._safe_project_path(arguments.get("path"))
        if path.is_dir():
            raise ValueError("read_project_file path must be a file")
        max_chars = _bounded_int(arguments.get("max_chars"), default=2000, minimum=1, maximum=8000)
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(self.root)
        return {
            "path": str(rel),
            "chars_returned": min(len(text), max_chars),
            "truncated": len(text) > max_chars,
            "text": text[:max_chars],
        }

    def _read_pdf_text(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = self._safe_project_path(arguments.get("path"))
        if path.suffix.lower() != ".pdf":
            raise ValueError("read_pdf_text path must end with .pdf")
        max_chars = _bounded_int(arguments.get("max_chars"), default=3000, minimum=1, maximum=10000)
        completed = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=self.command_timeout,
            check=False,
        )
        text = completed.stdout
        pattern = str(arguments.get("pattern") or "").strip()
        if pattern:
            regex = re.compile(pattern, re.IGNORECASE)
            max_matches = _bounded_int(arguments.get("max_matches"), default=20, minimum=1, maximum=80)
            lines = [line for line in text.splitlines() if regex.search(line)]
            filtered = "\n".join(lines[:max_matches])
            return {
                "path": str(path.relative_to(self.root)),
                "returncode": completed.returncode,
                "pattern": pattern,
                "match_count_returned": min(len(lines), max_matches),
                "omitted_count": max(0, len(lines) - max_matches),
                "truncated": len(filtered) > max_chars,
                "text": filtered[:max_chars],
                "stderr": _short_text(completed.stderr, 1000),
            }
        return {
            "path": str(path.relative_to(self.root)),
            "returncode": completed.returncode,
            "chars_returned": min(len(text), max_chars),
            "truncated": len(text) > max_chars,
            "text": text[:max_chars],
            "stderr": _short_text(completed.stderr, 1000),
        }

    def _search_project(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        pattern = str(arguments.get("pattern") or "").strip()
        if not pattern:
            raise ValueError("pattern is required")
        max_matches = _bounded_int(arguments.get("max_matches"), default=30, minimum=1, maximum=80)
        path_arg = arguments.get("path") or "."
        search_path = self._safe_project_path(path_arg)
        cmd = [
            "rg",
            "-n",
            "--no-heading",
            "--color",
            "never",
            "-m",
            str(max_matches),
            "-g",
            "!.env",
            "-g",
            "!.log/**",
        ]
        glob_arg = arguments.get("glob")
        for glob in _as_string_list(glob_arg):
            cmd.extend(["-g", glob])
        cmd.extend([pattern, str(search_path.relative_to(self.root))])
        completed = subprocess.run(
            cmd,
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=self.command_timeout,
            check=False,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) > max_matches:
            lines = lines[:max_matches]
        return {
            "command": _command_text(cmd),
            "returncode": completed.returncode,
            "match_count_returned": len(lines),
            "truncated": len(completed.stdout) > self.max_tool_result_chars,
            "matches": lines,
            "stderr": _short_text(completed.stderr, 1000),
        }

    def _list_project_files(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        max_files = _bounded_int(arguments.get("max_files"), default=80, minimum=1, maximum=200)
        cmd = ["rg", "--files", "-g", "!.env", "-g", "!.log/**"]
        for glob in _as_string_list(arguments.get("glob")):
            cmd.extend(["-g", glob])
        completed = subprocess.run(
            cmd,
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=self.command_timeout,
            check=False,
        )
        files = [line for line in completed.stdout.splitlines() if line.strip()]
        return {
            "command": _command_text(cmd),
            "returncode": completed.returncode,
            "files": files[:max_files],
            "omitted_count": max(0, len(files) - max_files),
            "stderr": _short_text(completed.stderr, 1000),
        }

    def _summarize_round_history(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        state_path = self.state_path
        if state_path is None:
            state_path = self.root / "physics_mvp_runs" / "current" / "state.json"
        if not state_path.is_absolute():
            state_path = self.root / state_path
        if not state_path.exists():
            return {"state_path": str(state_path.relative_to(self.root)), "exists": False, "history": []}
        state = json.loads(state_path.read_text(encoding="utf-8"))
        history = state.get("history", []) if isinstance(state, Mapping) else []
        if not isinstance(history, list):
            history = []
        status_filter = str(arguments.get("status") or "").strip()
        if status_filter:
            history = [
                item
                for item in history
                if isinstance(item, Mapping) and str(item.get("status") or "complete") == status_filter
            ]
        limit = _bounded_int(arguments.get("limit"), default=8, minimum=1, maximum=30)
        compact = [_compact_history_item(item) for item in history[-limit:] if isinstance(item, Mapping)]
        best = None
        for item in history:
            if not isinstance(item, Mapping):
                continue
            summary = item.get("evaluation_summary")
            if not isinstance(summary, Mapping):
                continue
            support = summary.get("support_rate")
            if isinstance(support, (int, float)) and (best is None or support > best.get("support_rate", -1)):
                best = {
                    "round": item.get("round"),
                    "support_rate": support,
                    "min_e_hull": summary.get("min_e_hull"),
                    "mean_e_hull": summary.get("mean_e_hull"),
                }
        return {
            "state_path": str(state_path.relative_to(self.root)),
            "exists": True,
            "status": state.get("status") if isinstance(state, Mapping) else None,
            "current_round": state.get("current_round") if isinstance(state, Mapping) else None,
            "history_len": len(history),
            "best": best,
            "history_tail": compact,
        }

    def _summarize_mechanism_evidence(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        state_path = self.state_path
        if state_path is None:
            state_path = self.root / "physics_mvp_runs" / "current" / "state.json"
        if not state_path.is_absolute():
            state_path = self.root / state_path
        if not state_path.exists():
            return {
                "tool_status": "ok",
                "state_path": str(state_path.relative_to(self.root)),
                "exists": False,
                "history_len": 0,
                "rounds": [],
            }
        state = json.loads(state_path.read_text(encoding="utf-8"))
        history = state.get("history", []) if isinstance(state, Mapping) else []
        if not isinstance(history, list):
            history = []
        include_failed = _as_bool(arguments.get("include_failed"), default=True)
        brief = _as_bool(arguments.get("brief"), default=False)
        requested_limit = _bounded_int(arguments.get("limit"), default=12, minimum=1, maximum=30)
        limit = min(requested_limit, 3 if brief else 6)
        offset = _bounded_int(arguments.get("offset"), default=0, minimum=0, maximum=max(len(history), 0))
        work_dir = state_path.parent
        records = []
        for item in history:
            if not isinstance(item, Mapping):
                continue
            record = _compact_mechanism_evidence_round(item, work_dir=work_dir, include_failed=include_failed)
            records.append(record)
        records.sort(key=_mechanism_evidence_sort_key, reverse=True)
        selected = records[offset : offset + limit]
        next_offset = offset + len(selected)
        evidence_views = _mechanism_evidence_views(
            records,
            history=history,
            candidate_pool_loader=self._safe_candidate_pool_for_exploration,
        )
        active_round = _active_mechanism_round_number(state if isinstance(state, Mapping) else {}, state_path)
        payload = {
            "tool_status": "ok",
            "state_path": str(state_path.relative_to(self.root)),
            "exists": True,
            "controller_state_status": state.get("status") if isinstance(state, Mapping) else None,
            "current_round": state.get("current_round") if isinstance(state, Mapping) else None,
            "active_round": active_round,
            "history_len": len(history),
            "search_policy": _mechanism_search_policy_for_round(active_round),
            "search_policy_scope": "A/F mechanism-discovery schedule, not a binding X/Y sequential material-acquisition policy",
            "xy_sequential_policy_warning": (
                "For X/Y sequential roles, use this tool's search_policy only as background about A/B's mechanism "
                "exploration cadence. The binding X/Y search mode is CONTEXT_JSON.controller_constraints.search_policy.current_search_mode."
            ),
            "current_principle_program": _compact_principle_program(state.get("current_principle_program"))
            if isinstance(state, Mapping)
            else None,
            "principle_book_tail": _compact_principle_book(state.get("principle_book"))
            if isinstance(state, Mapping)
            else [],
            "selection": "ranked_by_sun_stability_support_and_counterexample_strength",
            "evidence_views": evidence_views,
            "offset": offset,
            "limit": limit,
            "requested_limit": requested_limit,
            "next_offset": next_offset if next_offset < len(records) else None,
            "has_more": next_offset < len(records),
            "rounds": selected,
            "usage_note": (
                "Use evidence_views.top_successes for exploitation, near_misses and underexplored_clusters for exploration, "
                "failure_boundaries as counterexamples, recent_repetition as a collapse warning, and high sun_score or stable_count as stronger evidence. Pages are capped for readability; use next_offset for another page; "
                "Use current_principle_program and principle_book_tail to refine or close the active principle program. Separate primary_sun_count, control_sun_count, and mechanism_validated_sun_count; do not treat control-branch SUN as original-mechanism success. "
                "Do not repeat the same offset unless you intentionally want the same page."
            ),
        }
        if brief:
            payload["usage_note"] = (
                "Use top_successes, near_misses, failure_boundaries, current_principle_program, and principle_book_tail only as compact design evidence. "
                "Do not ask for or paste the full principle book. For X/Y sequential roles, do not treat this tool's "
                "search_policy.current_mode as binding; use CONTEXT_JSON.controller_constraints.search_policy.current_search_mode instead."
            )
            payload["brief"] = True
        return payload

    def _summarize_mechanism_evidence_fallback(self, arguments: Mapping[str, Any], exc: Exception) -> dict[str, Any]:
        state_path = self.state_path
        if state_path is None:
            state_path = self.root / "physics_mvp_runs" / "current" / "state.json"
        if not state_path.is_absolute():
            state_path = self.root / state_path
        state_path = state_path.resolve()
        payload: dict[str, Any] = {
            "tool_status": "fallback",
            "rag_fallback_used": True,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
            "state_path": str(state_path.relative_to(self.root)) if state_path.is_relative_to(self.root) else str(state_path),
            "exists": state_path.exists(),
            "usage_note": (
                "Mechanism evidence summarization hit an internal error, so the controller supplied a compact fallback. "
                "Use this as minimal principle context and do not treat the prior controller_state_status as a tool failure."
            ),
        }
        if not state_path.exists():
            payload.update({"history_len": 0, "rounds": [], "principle_book_tail": []})
            return payload
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as state_exc:
            payload.update(
                {
                    "tool_status": "error",
                    "state_read_error": f"{type(state_exc).__name__}: {state_exc}",
                    "history_len": 0,
                    "rounds": [],
                    "principle_book_tail": [],
                }
            )
            return payload
        if not isinstance(state, Mapping):
            payload.update({"history_len": 0, "rounds": [], "principle_book_tail": []})
            return payload
        history = state.get("history") if isinstance(state.get("history"), list) else []
        payload.update(
            {
                "controller_state_status": state.get("status"),
                "current_round": state.get("current_round"),
                "active_round": _active_mechanism_round_number(state, state_path),
                "history_len": len(history),
                "current_principle_program": _compact_principle_program(state.get("current_principle_program")),
                "principle_book_tail": _compact_principle_book(state.get("principle_book")),
                "rounds": [],
                "brief": _as_bool(arguments.get("brief"), default=False),
            }
        )
        return payload

    def _summarize_xy_generation_history(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        history_path = arguments.get("path")
        if history_path:
            path = self._safe_project_path(history_path, must_exist=False)
        else:
            path = self.xy_history_path or (self.root / "xy_runs" / "sequential_memory.json")
            if not path.is_absolute():
                path = self.root / path
            path = path.resolve()
            if self.root not in path.parents and path != self.root:
                raise ValueError(f"path escapes project root: {path}")
            if _looks_sensitive_path(path):
                raise ValueError(f"refusing sensitive path: {path.relative_to(self.root)}")
        if not path.exists():
            return {
                "history_path": str(path.relative_to(self.root)),
                "exists": False,
                "record_count": 0,
                "summary": {
                    "evaluated_count": 0,
                    "sun_count": 0,
                    "best": None,
                    "recent_formulas": [],
                    "failure_patterns": [],
                },
                "records": [],
                "usage_note": "No X/Y generation history exists yet; use A/B principles for the first proposal and create a falsifiable one-material design.",
            }
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping):
            records = raw.get("records", [])
        else:
            records = raw
        if not isinstance(records, list):
            records = []
        include_failed = _as_bool(arguments.get("include_failed"), default=True)
        compact_records = [
            _compact_xy_generation_record(item)
            for item in records
            if isinstance(item, Mapping) and (include_failed or _xy_record_is_success(item))
        ]
        limit = _bounded_int(arguments.get("limit"), default=12, minimum=1, maximum=50)
        offset = _bounded_int(arguments.get("offset"), default=0, minimum=0, maximum=max(len(compact_records), 0))
        selected = compact_records[offset : offset + limit]
        next_offset = offset + len(selected)
        evaluated = [item for item in compact_records if item.get("evaluated")]
        sun_records = [item for item in evaluated if item.get("is_sun")]
        near_records = [
            item
            for item in evaluated
            if isinstance(item.get("e_hull"), (int, float)) and item.get("e_hull") < 0.03
        ]
        failed = [item for item in compact_records if not item.get("is_sun")]
        best = None
        for item in evaluated:
            e_hull = item.get("e_hull")
            if not isinstance(e_hull, (int, float)):
                continue
            if best is None or e_hull < best.get("e_hull", float("inf")):
                best = item
        recent_formulas = []
        seen_formulas: set[str] = set()
        for item in reversed(compact_records):
            formula_candidates = [str(item.get("formula") or "").strip()]
            failed_formulas = item.get("failed_generated_formulas")
            if isinstance(failed_formulas, list):
                formula_candidates.extend(
                    str(entry.get("formula") or "").strip()
                    for entry in failed_formulas
                    if isinstance(entry, Mapping)
                )
            for formula in formula_candidates:
                if not formula or formula in seen_formulas:
                    continue
                seen_formulas.add(formula)
                recent_formulas.append(formula)
                if len(recent_formulas) >= 20:
                    break
            if len(recent_formulas) >= 20:
                break
        return {
            "history_path": str(path.relative_to(self.root)),
            "exists": True,
            "record_count": len(compact_records),
            "offset": offset,
            "limit": limit,
            "next_offset": next_offset if next_offset < len(compact_records) else None,
            "has_more": next_offset < len(compact_records),
            "summary": {
                "evaluated_count": len(evaluated),
                "sun_count": len(sun_records),
                "sun_ratio": (len(sun_records) / len(evaluated)) if evaluated else None,
                "near_stable_0_03_count": len(near_records),
                "best": best,
                "recent_formulas": recent_formulas,
                "top_successes": sorted(
                    sun_records,
                    key=lambda item: item.get("e_hull") if isinstance(item.get("e_hull"), (int, float)) else 999,
                )[:6],
                "near_misses": sorted(
                    [item for item in near_records if not item.get("is_sun")],
                    key=lambda item: item.get("e_hull") if isinstance(item.get("e_hull"), (int, float)) else 999,
                )[:6],
                "recent_failures": failed[-8:],
                "failure_patterns": _xy_failure_patterns(failed),
            },
            "records": selected,
            "usage_note": (
                "Use top_successes for exploitation, near_misses for local repairs, recent_failures and failure_patterns as negative strategy memory, "
                "and recent_formulas to avoid duplicate reduced_formula sampling unless a deliberate control is justified."
            ),
        }

    def _safe_candidate_pool_for_exploration(self) -> list[dict[str, Any]]:
        try:
            return self._load_candidate_pool()
        except Exception:
            return []

    def _query_candidate_pool(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, Mapping):
            raise ValueError("query_candidate_pool requires a query object")
        query = dict(query)
        errors = validate_query(query)
        if errors:
            return {"valid_query": False, "validation_errors": errors, "match_count": 0, "examples": []}
        records = self._load_candidate_pool()
        matches = [record for record in records if query_matches(record, query)]
        count = _bounded_int(arguments.get("count"), default=10, minimum=0, maximum=50)
        seed = _bounded_int(arguments.get("seed"), default=0, minimum=0, maximum=10**9)
        selected = select_matches(matches, query, count=count, seed=seed) if count else []
        scan_count = min(len(matches), max(count, CANDIDATE_TOOL_DISTINCT_SCAN_LIMIT))
        ordered_probe = select_matches(matches, query, count=scan_count, seed=seed) if scan_count else []
        formula_counts = Counter(_candidate_formula(record) for record in matches if _candidate_formula(record))
        selected_formula_counts = Counter(_candidate_formula(record) for record in selected if _candidate_formula(record))
        first_distinct_examples = _first_distinct_candidate_records(
            ordered_probe,
            limit=CANDIDATE_TOOL_DISTINCT_EXAMPLE_LIMIT,
        )
        duplicate_formula_examples = [
            {"formula": formula, "count": formula_count}
            for formula, formula_count in selected_formula_counts.most_common()
            if formula_count > 1
        ][:CANDIDATE_TOOL_DISTINCT_EXAMPLE_LIMIT]
        limited_examples = selected[:CANDIDATE_TOOL_EXAMPLE_LIMIT]
        return {
            "valid_query": True,
            "candidate_pool_path": str(self.candidate_pool_path.relative_to(self.root)) if self.candidate_pool_path else None,
            "match_count": len(matches),
            "distinct_formula_count": len(formula_counts),
            "requested_examples": count,
            "examples_returned": len(limited_examples),
            "returned_distinct_formula_count": len(selected_formula_counts),
            "first_distinct_formulas": [_candidate_formula(record) for record in first_distinct_examples],
            "first_distinct_examples": [_compact_candidate_record(record) for record in first_distinct_examples],
            "duplicate_formulas_in_examples": duplicate_formula_examples,
            "examples": [_compact_candidate_record(record) for record in limited_examples],
            "usage_note": (
                "distinct_formula_count is computed across all matching pool records. "
                "first_distinct_examples follows the requested preferred_order and is intended for one-formula-per-selection feasibility checks."
            ),
        }

    def _run_shell(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        command = arguments.get("command")
        if isinstance(command, str):
            argv = shlex.split(command)
        elif isinstance(command, Sequence) and not isinstance(command, (bytes, bytearray)):
            argv = [str(item) for item in command]
        else:
            raise ValueError("command must be a string or argv list")
        if not argv:
            raise ValueError("command is empty")
        executable = Path(argv[0]).name
        if executable not in self.allowed_commands:
            raise ValueError(f"command {executable!r} is not allowed")
        for token in argv[1:]:
            if _looks_sensitive_path(Path(str(token))):
                raise ValueError("refusing command that references a sensitive path")
        completed = subprocess.run(
            argv,
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=self.command_timeout,
            check=False,
        )
        return {
            "command": _command_text(argv),
            "returncode": completed.returncode,
            "stdout": _short_text(completed.stdout, self.max_tool_result_chars // 2),
            "stderr": _short_text(completed.stderr, self.max_tool_result_chars // 2),
        }

    def _write_agent_artifact(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        raw_path = str(arguments.get("path") or "").strip()
        if not raw_path:
            raise ValueError("path is required")
        rel = Path(raw_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("artifact path must be relative and cannot contain '..'")
        if _looks_sensitive_path(rel):
            raise ValueError("refusing to write sensitive artifact path")
        path = (self.writable_dir / rel).resolve()
        if not path.is_relative_to(self.writable_dir):
            raise ValueError("artifact path escapes writable directory")
        content = str(arguments.get("content", ""))
        if len(content) > 8000:
            raise ValueError("artifact content exceeds 8000 characters")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "path": str(path.relative_to(self.root)),
            "bytes_written": len(content.encode("utf-8")),
            "exists": path.exists(),
        }

    def _write_project_file(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if not self.allow_project_writes:
            raise PermissionError("write_project_file is disabled; use write_agent_artifact or enable project writes explicitly")
        path = self._safe_project_path(arguments.get("path"), must_exist=False)
        if path.exists() and path.is_dir():
            raise ValueError("write_project_file path must be a file")
        content = str(arguments.get("content", ""))
        if len(content) > 20000:
            raise ValueError("project file content exceeds 20000 characters")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "path": str(path.relative_to(self.root)),
            "bytes_written": len(content.encode("utf-8")),
            "exists": path.exists(),
        }

    def _load_candidate_pool(self) -> list[dict[str, Any]]:
        if self._candidate_pool_cache is not None:
            return self._candidate_pool_cache
        if self.candidate_pool_path is None:
            raise ValueError("candidate_pool_path is not configured")
        path = self.candidate_pool_path if self.candidate_pool_path.is_absolute() else self.root / self.candidate_pool_path
        if not path.exists():
            raise FileNotFoundError(str(path))
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
        self._candidate_pool_cache = records
        return records

    def _write_trace(self, trace: Mapping[str, Any]) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.trace_dir / f"agent_trace_{stamp}.json"
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _mechanism_evidence_call_key(arguments: Mapping[str, Any]) -> dict[str, Any]:
    brief = _as_bool(arguments.get("brief"), default=False)
    requested_limit = _bounded_int(arguments.get("limit"), default=12, minimum=1, maximum=30)
    return {
        "effective_limit": min(requested_limit, 3 if brief else 6),
        "offset": _bounded_int(arguments.get("offset"), default=0, minimum=0, maximum=10**9),
        "include_failed": _as_bool(arguments.get("include_failed"), default=True),
        "brief": brief,
    }


def _looks_sensitive_path(path: Path) -> bool:
    parts = [part.lower() for part in path.parts if part not in {"", "."}]
    joined = "/".join(parts)
    return any(marker in parts or marker in joined for marker in SENSITIVE_PATH_MARKERS)


def _short_text(value: Any, max_chars: int = 1000) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _compact_json_len(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))


def _compact_tool_results_for_prompt(results: Sequence[Mapping[str, Any]], max_chars: int) -> list[Any]:
    """Return a short, structured tool result payload for the next LLM turn."""

    compact: list[Any] = []
    result_count = max(1, len(results))
    per_result_budget = max(420, (max_chars - 80) // result_count)
    for raw in results:
        if not isinstance(raw, Mapping):
            compact.append(_short_text(raw, min(240, per_result_budget)))
            continue
        name = str(raw.get("name") or "").strip()
        item: dict[str, Any] = {
            key: raw.get(key)
            for key in ("id", "name", "ok")
            if raw.get(key) not in (None, "", [])
        }
        output = raw.get("output")
        if isinstance(output, Mapping):
            item["output"] = _compact_tool_output_for_prompt(name, output, max_chars=max(220, per_result_budget - 120))
        elif output not in (None, "", []):
            item["output"] = _short_text(output, max(160, per_result_budget - 120))
        if _compact_json_len(compact + [item]) > max_chars:
            item["output"] = _short_text(item.get("output"), max(120, per_result_budget // 2))
        if _compact_json_len(compact + [item]) > max_chars:
            compact.append({"id": raw.get("id"), "name": name, "ok": raw.get("ok"), "_truncated": True})
            break
        compact.append(item)
    if _compact_json_len(compact) <= max_chars:
        return compact
    return _limit_tool_output(compact, max_chars)


def _compact_tool_output_for_prompt(name: str, output: Mapping[str, Any], *, max_chars: int) -> Any:
    if output.get("duplicate_request") is True:
        return {
            "duplicate_request": True,
            "message": _short_text(output.get("message"), 260),
            "repeated_arguments": output.get("repeated_arguments"),
        }
    if name == "summarize_mechanism_evidence":
        compact = _compact_mechanism_evidence_output_for_prompt(output)
    elif name == "summarize_xy_generation_history":
        compact = _compact_xy_generation_history_output_for_prompt(output)
    elif name == "query_candidate_pool":
        compact = _compact_candidate_pool_output_for_prompt(output, max_chars=max_chars)
    else:
        compact = _limit_tool_output(output, max_chars)
    if _compact_json_len(compact) <= max_chars:
        return compact
    return _limit_tool_output(compact, max_chars)


def _compact_mechanism_evidence_output_for_prompt(output: Mapping[str, Any]) -> dict[str, Any]:
    search_policy = output.get("search_policy")
    if isinstance(search_policy, Mapping):
        search_policy = {
            key: search_policy.get(key)
            for key in ("round", "cycle_slot", "current_mode", "directive")
            if search_policy.get(key) not in (None, "", [])
        }
    else:
        search_policy = None
    rounds = output.get("rounds") if isinstance(output.get("rounds"), list) else []
    principle_book = output.get("principle_book_tail") if isinstance(output.get("principle_book_tail"), list) else []
    compact = {
        "tool_status": output.get("tool_status"),
        "exists": output.get("exists"),
        "current_round": output.get("current_round"),
        "active_round": output.get("active_round"),
        "history_len": output.get("history_len"),
        "search_policy": search_policy,
        "current_principle_program": _limit_tool_output(output.get("current_principle_program"), 420),
        "principle_book_tail": _limit_tool_output(principle_book[-1:], 360) if principle_book else [],
        "evidence_views": _compact_evidence_views_for_prompt(output.get("evidence_views")),
        "offset": output.get("offset"),
        "limit": output.get("limit"),
        "next_offset": output.get("next_offset"),
        "has_more": output.get("has_more"),
        "rounds": [_very_compact_evidence_round(item) for item in rounds[:2] if isinstance(item, Mapping)],
        "brief": output.get("brief"),
        "usage_note": "Use top_successes/near_misses/failure_boundaries/underexplored_clusters; do not paste raw RAG.",
    }
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_evidence_views_for_prompt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for key in ("top_successes", "near_misses", "failure_boundaries"):
        items = value.get(key)
        if isinstance(items, list):
            compact[key] = [_limit_tool_output(item, 260) for item in items[:2]]
    underexplored = value.get("underexplored_clusters")
    if isinstance(underexplored, list):
        compact["underexplored_clusters"] = [_limit_tool_output(item, 220) for item in underexplored[:3]]
    recent_repetition = value.get("recent_repetition")
    if isinstance(recent_repetition, Mapping):
        compact["recent_repetition"] = _limit_tool_output(recent_repetition, 320)
    return {key: item for key, item in compact.items() if item not in (None, "", [], {})}


def _compact_xy_generation_history_output_for_prompt(output: Mapping[str, Any]) -> dict[str, Any]:
    summary = output.get("summary") if isinstance(output.get("summary"), Mapping) else {}
    records = output.get("records") if isinstance(output.get("records"), list) else []
    compact_summary = {
        "evaluated_count": summary.get("evaluated_count"),
        "sun_count": summary.get("sun_count"),
        "sun_ratio": summary.get("sun_ratio"),
        "near_stable_0_03_count": summary.get("near_stable_0_03_count"),
        "best": _limit_tool_output(summary.get("best"), 260),
        "recent_formulas": (summary.get("recent_formulas") or [])[:8]
        if isinstance(summary.get("recent_formulas"), list)
        else [],
        "top_successes": _limit_tool_output((summary.get("top_successes") or [])[:3], 480)
        if isinstance(summary.get("top_successes"), list)
        else [],
        "near_misses": _limit_tool_output((summary.get("near_misses") or [])[:3], 480)
        if isinstance(summary.get("near_misses"), list)
        else [],
        "failure_patterns": _limit_tool_output((summary.get("failure_patterns") or [])[:4], 420)
        if isinstance(summary.get("failure_patterns"), list)
        else [],
    }
    compact = {
        "exists": output.get("exists"),
        "record_count": output.get("record_count"),
        "offset": output.get("offset"),
        "limit": output.get("limit"),
        "next_offset": output.get("next_offset"),
        "has_more": output.get("has_more"),
        "summary": {key: value for key, value in compact_summary.items() if value not in (None, "", [], {})},
        "records": _limit_tool_output(records[:3], 560),
        "usage_note": "Use successes/near_misses/failures; avoid duplicate reduced_formula unless justified.",
    }
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_candidate_record_for_prompt(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {"summary": _short_text(record, 160)}
    compact = {
        key: record.get(key)
        for key in COMPACT_RECORD_KEYS
        if record.get(key) not in (None, "", [], {})
    }
    return compact or {"summary": _short_text(record, 160)}


def _compact_candidate_records_for_prompt(records: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    return [_compact_candidate_record_for_prompt(item) for item in records[:limit]]


def _compact_candidate_pool_output_for_prompt(output: Mapping[str, Any], *, max_chars: int) -> dict[str, Any]:
    first_distinct_examples = _compact_candidate_records_for_prompt(output.get("first_distinct_examples"), limit=3)
    examples = _compact_candidate_records_for_prompt(output.get("examples"), limit=2)
    compact = {
        "valid_query": output.get("valid_query"),
        "match_count": output.get("match_count"),
        "distinct_formula_count": output.get("distinct_formula_count"),
        "requested_examples": output.get("requested_examples"),
        "examples_returned": output.get("examples_returned"),
        "returned_distinct_formula_count": output.get("returned_distinct_formula_count"),
        "first_distinct_formulas": (output.get("first_distinct_formulas") or [])[:8]
        if isinstance(output.get("first_distinct_formulas"), list)
        else [],
        "first_distinct_examples": first_distinct_examples,
        "examples": examples,
        "validation_errors": output.get("validation_errors"),
        "usage_note": _short_text(output.get("usage_note"), 220),
    }
    compact = {key: value for key, value in compact.items() if value not in (None, "", [], {})}
    if _compact_json_len(compact) <= max_chars:
        return compact
    compact.pop("usage_note", None)
    compact["first_distinct_examples"] = first_distinct_examples[:2]
    compact["examples"] = examples[:1]
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _stale_mattergen_backend_gate_text(value: Any) -> bool:
    text = str(value or "").lower()
    normalized = re.sub(r"[\s_\-]+", " ", text)
    if any(
        (
            "backend gate" in normalized and ("unrepaired" in normalized or "repair" in normalized),
            "backend remains unrepaired" in normalized,
            "backend is unrepaired" in normalized,
            "until backend repair" in normalized,
        )
    ):
        return True
    if "mattergen" not in normalized:
        return False
    return any(
        (
            "backend repair" in normalized,
            "repair confirmation" in normalized,
            "repair/cuda" in normalized,
            "cuda compatibility" in normalized and ("repair" in normalized or "confirmed" in normalized),
            "no kernel" in normalized,
            "nokernel" in normalized,
            "kernel image" in normalized,
            "do not dispatch" in normalized and ("repair" in normalized or "confirmed" in normalized),
            "dispatch nothing" in normalized and ("repair" in normalized or "confirmed" in normalized),
            "not explicitly confirmed" in normalized and ("repair" in normalized or "backend" in normalized),
        )
    )


def _command_text(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def _compact_candidate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in COMPACT_RECORD_KEYS if key in record}


def _candidate_formula(record: Mapping[str, Any]) -> str:
    formula = record.get("formula") or record.get("reduced_formula") or record.get("pretty_formula")
    return str(formula or "").strip()


def _failed_generator_formulas(errors: Any, *, limit: int = 8) -> list[dict[str, str]]:
    if not isinstance(errors, list):
        return []
    formulas: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in errors:
        text = str(item or "")
        matches = list(GENERATED_FORMULA_ERROR_RE.finditer(text))
        matches.extend(GENERATED_FORMULA_EXCLUDED_RE.finditer(text))
        for match in matches:
            formula = str(match.group(1) or "").strip()
            template = str(match.group(2) or "").strip() if len(match.groups()) >= 2 else ""
            if not formula:
                continue
            key = (formula, template)
            if key in seen:
                continue
            seen.add(key)
            entry = {"formula": formula, "error": _short_text(text, 140)}
            if template:
                entry["template"] = template
            formulas.append(entry)
            if len(formulas) >= limit:
                break
        if len(formulas) >= limit:
            break
    return formulas


def _first_distinct_candidate_records(records: Sequence[Mapping[str, Any]], *, limit: int) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        formula = _candidate_formula(record)
        if not formula or formula in seen:
            continue
        seen.add(formula)
        selected.append(record)
        if len(selected) >= limit:
            break
    return selected


def _requires_mandatory_mechanism_rag(user: str) -> bool:
    return (
        "MANDATORY_MECHANISM_RAG" in user
        or "MANDATORY_HISTORY_RAG" in user
        or "MANDATORY_XY_SEQUENTIAL_RAG" in user
        or ("summarize_mechanism_evidence" in user and "Sequential" in user)
    )


def _requires_mandatory_xy_generation_rag(user: str) -> bool:
    return "MANDATORY_XY_SEQUENTIAL_RAG" in user or "summarize_xy_generation_history" in user and "Sequential" in user


def _compact_history_item(item: Mapping[str, Any]) -> dict[str, Any]:
    summary = item.get("evaluation_summary")
    if not isinstance(summary, Mapping):
        summary = {}
    skip_summary = item.get("skip_summary")
    return {
        "round": item.get("round"),
        "status": item.get("status", "complete"),
        "accepted_mechanism_count": len(item.get("accepted_mechanisms", []) or []),
        "accepted_prediction_count": len(item.get("accepted_predictions", []) or []),
        "accepted_bundle_count": len(item.get("accepted_bundles", []) or []),
        "support_rate": summary.get("support_rate"),
        "min_e_hull": summary.get("min_e_hull"),
        "mean_e_hull": summary.get("mean_e_hull"),
        "skip_stage_status": skip_summary.get("stage_status") if isinstance(skip_summary, Mapping) else None,
        "skip_reason": _short_text(skip_summary.get("reason"), 240) if isinstance(skip_summary, Mapping) else None,
    }


def _compact_mechanism_evidence_round(
    item: Mapping[str, Any],
    *,
    work_dir: Path,
    include_failed: bool,
) -> dict[str, Any]:
    summary = item.get("evaluation_summary")
    if not isinstance(summary, Mapping):
        summary = {}
    round_number = item.get("round")
    analysis = _read_round_analysis(work_dir, round_number)
    novelty = analysis.get("merged_evaluator_novelty") if isinstance(analysis.get("merged_evaluator_novelty"), Mapping) else {}
    mechanisms = [
        _compact_mechanism_record(mechanism)
        for mechanism in (item.get("accepted_mechanisms") or [])
        if isinstance(mechanism, Mapping)
    ][:1]
    predictions = {
        str(prediction.get("id") or ""): prediction
        for prediction in (item.get("accepted_predictions") or [])
        if isinstance(prediction, Mapping)
    }
    supported_predictions = []
    failed_predictions = []
    for bundle in item.get("bundle_results") or []:
        if not isinstance(bundle, Mapping):
            continue
        supported = bool(bundle.get("supported"))
        if not supported and not include_failed:
            continue
        prediction_ids = bundle.get("prediction_ids")
        if not isinstance(prediction_ids, list):
            prediction_ids = []
        for prediction_id in prediction_ids[:2]:
            prediction = predictions.get(str(prediction_id), {})
            evidence = {
                "prediction_id": str(prediction_id),
                "claim": _short_text(prediction.get("claim"), 90) if isinstance(prediction, Mapping) else "",
                "bundle_id": bundle.get("bundle_id"),
                "delta": _compact_number(bundle.get("delta"), digits=4),
            }
            if supported:
                supported_predictions.append(evidence)
            else:
                failed_predictions.append(evidence)
    record = {
        "round": round_number,
        "status": item.get("status") if item.get("status") not in (None, "complete") else None,
        "support_rate": _compact_number(summary.get("support_rate"), digits=3),
        "supported_bundle_count": summary.get("supported_bundle_count"),
        "bundle_count": summary.get("bundle_count"),
        "min_e_hull": _compact_number(summary.get("min_e_hull"), digits=4),
        "mean_e_hull": _compact_number(summary.get("mean_e_hull"), digits=4),
        "stable_count": summary.get("stable_count"),
        "primary_sun_count": summary.get("primary_sun_count"),
        "control_sun_count": summary.get("control_sun_count"),
        "mechanism_validated_sun_count": summary.get("mechanism_validated_sun_count"),
        "sun_score": _compact_number(
            novelty.get("sun_score") if isinstance(novelty, Mapping) else analysis.get("sun_score"),
            digits=3,
        ),
        "sun_count": novelty.get("sun_both_novel_count") if isinstance(novelty, Mapping) else None,
        "e_hull_lt_0_03": analysis.get("e_hull_lt_0_03"),
        "e_hull_lt_0_10": analysis.get("e_hull_lt_0_10"),
        "accepted_mechanisms": mechanisms,
        "principle_postmortem": _compact_principle_postmortem(item.get("principle_postmortem"))
        if isinstance(item.get("principle_postmortem"), Mapping)
        else None,
        "supported_predictions": supported_predictions[:1],
        "failed_predictions": failed_predictions[:1],
    }
    return {key: value for key, value in record.items() if value not in (None, [], "")}


def _compact_mechanism_record(mechanism: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "id": mechanism.get("id"),
            "claim": _short_text(mechanism.get("claim"), 90),
            "causal_driver": _short_text(mechanism.get("causal_driver"), 90),
            "intervention": _short_text(mechanism.get("intervention"), 90),
            "confidence": mechanism.get("confidence"),
        }.items()
        if value not in (None, "")
    }


def _compact_principle_postmortem(postmortem: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "status": postmortem.get("status"),
            "hypothesis_status": postmortem.get("hypothesis_status"),
            "principle_update_action": postmortem.get("principle_update_action"),
            "current_principle_statement": _short_text(postmortem.get("current_principle_statement"), 120),
            "causal_interpretation": _short_text(postmortem.get("causal_interpretation"), 160),
            "next_test_focus": _short_text(postmortem.get("next_test_focus"), 120),
        }.items()
        if value not in (None, [], "")
    }


def _compact_principle_program(program: Any) -> dict[str, Any] | None:
    if not isinstance(program, Mapping):
        return None
    recent = []
    evidence = program.get("evidence_rounds")
    if isinstance(evidence, list):
        for item in evidence[-4:]:
            if not isinstance(item, Mapping):
                continue
            recent.append(
                {
                    key: value
                    for key, value in {
                        "round": item.get("round"),
                        "hypothesis_status": item.get("hypothesis_status"),
                        "principle_update_action": item.get("principle_update_action"),
                        "support_rate": item.get("support_rate"),
                        "primary_sun_count": item.get("primary_sun_count"),
                        "control_sun_count": item.get("control_sun_count"),
                        "mechanism_validated_sun_count": item.get("mechanism_validated_sun_count"),
                    }.items()
                    if value not in (None, [], "")
                }
            )
    return {
        key: value
        for key, value in {
            "program_id": program.get("program_id"),
            "status": program.get("status"),
            "inner_iteration": program.get("inner_iteration"),
            "current_principle_statement": _short_text(program.get("current_principle_statement"), 160),
            "micro_mechanism": _short_text(program.get("micro_mechanism"), 160),
            "recent_evidence_rounds": recent,
        }.items()
        if value not in (None, [], "")
    }


def _compact_principle_book(book: Any) -> list[dict[str, Any]]:
    if not isinstance(book, list):
        return []
    compact = []
    for item in book[-4:]:
        if not isinstance(item, Mapping):
            continue
        compact.append(
            {
                key: value
                for key, value in {
                    "program_id": item.get("program_id"),
                    "status": item.get("status"),
                    "principle_statement": _short_text(item.get("principle_statement"), 160),
                    "evidence_rounds": item.get("evidence_rounds"),
                    "boundaries": item.get("boundaries", [])[:3] if isinstance(item.get("boundaries"), list) else None,
                }.items()
                if value not in (None, [], "")
            }
        )
    return compact


def _xy_record_is_success(record: Mapping[str, Any]) -> bool:
    result = record.get("evaluation_result")
    if isinstance(result, Mapping):
        if result.get("is_sun") is True:
            return True
        e_hull = _numeric(result.get("e_hull"))
        return e_hull < 0
    evaluation = record.get("evaluation_summary")
    if isinstance(evaluation, Mapping):
        if _numeric(evaluation.get("sun_count")) > 0:
            return True
        return _numeric(evaluation.get("min_e_hull")) < 0
    return False


def _compact_xy_generation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = record.get("evaluation_result")
    if not isinstance(result, Mapping):
        result = {}
    summary = record.get("evaluation_summary")
    if not isinstance(summary, Mapping):
        summary = {}
    description = record.get("material_description")
    if not isinstance(description, Mapping):
        description = {}
    selected = record.get("selected_record")
    if not isinstance(selected, Mapping):
        selected = {}
    candidate = record.get("candidate_spec")
    if not isinstance(candidate, Mapping):
        candidate = {}
    materialization_errors = record.get("materialization_errors")
    failed_generated = _failed_generator_formulas(materialization_errors)
    e_hull_value = result.get("e_hull")
    if e_hull_value is None:
        e_hull_value = summary.get("min_e_hull")
    e_hull = _numeric(e_hull_value)
    is_sun = bool(result.get("is_sun")) or _numeric(summary.get("sun_count")) > 0 or e_hull < 0
    formula = (
        result.get("formula")
        or selected.get("formula")
        or record.get("formula")
        or _candidate_formula(selected)
    )
    postmortem = record.get("xy_postmortem")
    if not isinstance(postmortem, Mapping):
        postmortem = {}
    next_strategy = _short_text(postmortem.get("next_strategy"), 320)
    if _stale_mattergen_backend_gate_text(next_strategy):
        next_strategy = (
            "Stale MatterGen backend repair/CUDA gate omitted from compact history; "
            "follow controller_constraints.mattergen_operational_status and latest_xy_strategy_constraints."
        )
    return {
        key: value
        for key, value in {
            "iteration": record.get("iteration"),
            "status": record.get("status"),
            "evaluated": bool(result) or bool(summary),
            "formula": str(formula or "").strip() or None,
            "e_hull": _compact_number(e_hull, digits=5) if e_hull < 10**8 else None,
            "is_sun": is_sun,
            "near_stable_0_03": e_hull < 0.03,
            "material_description": {
                key: _short_text(description.get(key), 260)
                for key in ("natural_language_description", "target_family", "mechanism_rationale", "expected_local_motif")
                if description.get(key) not in (None, "", [])
            },
            "generator": {
                "id": candidate.get("id"),
                "formula_probes": candidate.get("formula_probes"),
                "formula_probe_strings": candidate.get("formula_probe_strings"),
                "generator_template": candidate.get("generator_template"),
                "selection_policy": candidate.get("selection_policy"),
            },
            "failed_generated_formulas": failed_generated,
            "materialization_errors_tail": [
                _short_text(error, 180)
                for error in (materialization_errors[-3:] if isinstance(materialization_errors, list) else [])
            ],
            "failure_or_success_interpretation": _short_text(
                postmortem.get("causal_interpretation")
                or postmortem.get("strategy_update")
                or postmortem.get("summary")
                or record.get("failure_reason"),
                420,
            ),
            "next_strategy": next_strategy,
        }.items()
        if value not in (None, "", [], {})
    }


def _xy_failure_patterns(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for record in records:
        reason = str(record.get("failure_or_success_interpretation") or record.get("status") or "unspecified").lower()
        if "duplicate" in reason:
            key = "duplicate_or_replayed_formula"
        elif "template" in reason or "proxy" in reason:
            key = "unfaithful_or_insufficient_template"
        elif "charge" in reason or "oxidation" in reason:
            key = "charge_or_oxidation_mismatch"
        elif "high" in reason or "e_hull" in reason or "unstable" in reason:
            key = "evaluated_high_e_hull"
        elif "materializ" in reason or "generator" in reason or "parse" in reason:
            key = "generator_materialization_failure"
        else:
            key = "other_failure_or_boundary"
        counter[key] += 1
        examples.setdefault(key, _short_text(reason, 220))
    return [
        {"pattern": key, "count": count, "example": examples.get(key)}
        for key, count in counter.most_common(8)
    ]


def _read_round_analysis(work_dir: Path, round_number: Any) -> dict[str, Any]:
    try:
        number = int(round_number)
    except (TypeError, ValueError):
        return {}
    path = work_dir / f"round_{number:03d}" / "analysis" / "summary.json"
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _mechanism_evidence_sort_key(record: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    sun_score = _numeric(record.get("sun_score"))
    stable_count = _numeric(record.get("stable_count"))
    support_rate = _numeric(record.get("support_rate"))
    min_e_hull = _numeric(record.get("min_e_hull"))
    failed_count = float(len(record.get("failed_predictions") or []))
    round_number = _numeric(record.get("round"))
    stability_bonus = max(0.0, -min_e_hull)
    return (sun_score * 10.0 + stable_count * 2.0 + support_rate + stability_bonus + failed_count * 0.1, sun_score, support_rate, stable_count, round_number)


def _mechanism_evidence_views(
    records: Sequence[Mapping[str, Any]],
    *,
    history: Sequence[Any],
    candidate_pool_loader: Any,
) -> dict[str, Any]:
    top_successes = [_very_compact_evidence_round(item) for item in records[:3]]
    near_misses = [
        _very_compact_evidence_round(item)
        for item in sorted(records, key=_near_miss_sort_key)
        if _is_near_miss_record(item)
    ][:3]
    failure_boundaries = [
        _very_compact_evidence_round(item)
        for item in sorted(records, key=_failure_boundary_sort_key, reverse=True)
        if _is_failure_boundary_record(item)
    ][:3]
    candidate_pool = candidate_pool_loader()
    recent_repetition = _recent_repetition_summary(history, candidate_pool)
    underexplored = _underexplored_candidate_clusters(history, candidate_pool, limit=5)
    return {
        "top_successes": top_successes,
        "near_misses": near_misses,
        "failure_boundaries": failure_boundaries,
        "underexplored_clusters": underexplored,
        "recent_repetition": recent_repetition,
        "view_usage": (
            "Balance exploitation against exploration: top_successes are strong evidence, near_misses are promising variants, "
            "failure_boundaries define limits, underexplored_clusters suggest wider search, and recent_repetition warns against local collapse."
        ),
    }


def _very_compact_evidence_round(record: Mapping[str, Any]) -> dict[str, Any]:
    mechanisms = record.get("accepted_mechanisms") if isinstance(record.get("accepted_mechanisms"), list) else []
    supported = record.get("supported_predictions") if isinstance(record.get("supported_predictions"), list) else []
    failed = record.get("failed_predictions") if isinstance(record.get("failed_predictions"), list) else []
    compact = {
        "round": record.get("round"),
        "status": record.get("status"),
        "support_rate": record.get("support_rate"),
        "min_e_hull": record.get("min_e_hull"),
        "mean_e_hull": record.get("mean_e_hull"),
        "stable_count": record.get("stable_count"),
        "primary_sun_count": record.get("primary_sun_count"),
        "control_sun_count": record.get("control_sun_count"),
        "mechanism_validated_sun_count": record.get("mechanism_validated_sun_count"),
        "sun_score": record.get("sun_score"),
        "e_hull_lt_0_03": record.get("e_hull_lt_0_03"),
        "mechanism": _short_text(mechanisms[0].get("claim"), 80) if mechanisms and isinstance(mechanisms[0], Mapping) else None,
        "supported": _short_text(supported[0].get("claim"), 80) if supported and isinstance(supported[0], Mapping) else None,
        "failed": _short_text(failed[0].get("claim"), 80) if failed and isinstance(failed[0], Mapping) else None,
    }
    return {key: value for key, value in compact.items() if value not in (None, [], "")}


def _near_miss_sort_key(record: Mapping[str, Any]) -> tuple[float, float, float]:
    min_e_hull = _numeric(record.get("min_e_hull"))
    support_rate = _numeric(record.get("support_rate"))
    lt03 = _numeric(record.get("e_hull_lt_0_03"))
    return (abs(min_e_hull), -lt03, -support_rate)


def _is_near_miss_record(record: Mapping[str, Any]) -> bool:
    if _numeric(record.get("stable_count")) > 0 or _numeric(record.get("sun_score")) > 0:
        return False
    if not isinstance(record.get("min_e_hull"), (int, float)):
        return False
    min_e_hull = _numeric(record.get("min_e_hull"))
    return 0.0 <= min_e_hull <= 0.03 or _numeric(record.get("e_hull_lt_0_03")) > 0


def _failure_boundary_sort_key(record: Mapping[str, Any]) -> tuple[float, float, float]:
    failed_count = float(len(record.get("failed_predictions") or []))
    support_rate = _numeric(record.get("support_rate"))
    round_number = _numeric(record.get("round"))
    return (failed_count, 1.0 - support_rate, round_number)


def _is_failure_boundary_record(record: Mapping[str, Any]) -> bool:
    status = str(record.get("status") or "").strip().lower()
    if status and status != "complete":
        return True
    if record.get("failed_predictions"):
        return True
    support_rate = record.get("support_rate")
    return isinstance(support_rate, (int, float)) and support_rate <= 0


def _mechanism_search_policy_for_round(round_number: Any) -> dict[str, Any]:
    try:
        number = int(round_number)
    except (TypeError, ValueError):
        number = 1
    if number <= 0:
        number = 1
    slot = (number - 1) % 10
    if slot < 7:
        mode = "exploitation"
        directive = "Refine the strongest supported mechanism, but include a near-miss or failure boundary."
    elif slot < 9:
        mode = "neighbor_exploration"
        directive = "Explore a one-axis mutation of a supported mechanism."
    else:
        mode = "far_exploration"
        directive = "Explore a plausible under-tested cluster outside recent repetition."
    return {
        "schedule": "10_round_cycle",
        "cycle_fraction": {"exploitation": 0.7, "neighbor_exploration": 0.2, "far_exploration": 0.1},
        "round": number,
        "cycle_slot": slot + 1,
        "current_mode": mode,
        "directive": directive,
    }


def _active_mechanism_round_number(state: Mapping[str, Any], state_path: Path) -> int:
    """Return the round currently being debated, not just the last completed round."""
    history = state.get("history") if isinstance(state.get("history"), list) else []
    try:
        completed_round = int(state.get("current_round"))
    except (TypeError, ValueError):
        completed_round = 0
    if completed_round <= 0:
        completed_round = max(
            [
                int(item.get("round"))
                for item in history
                if isinstance(item, Mapping) and isinstance(item.get("round"), int)
            ]
            or [0]
        )
    next_round = completed_round + 1
    if next_round > 0 and (state_path.parent / f"round_{next_round:03d}").exists():
        return next_round
    return completed_round if completed_round > 0 else 1


def _recent_repetition_summary(history: Sequence[Any], candidate_pool: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    id_to_record = {
        str(record.get("material_id")): record
        for record in candidate_pool
        if isinstance(record, Mapping) and record.get("material_id")
    }
    cluster_counts: Counter[tuple[str, str, str]] = Counter()
    motif_counts: Counter[str] = Counter()
    for item in list(history)[-8:]:
        if not isinstance(item, Mapping):
            continue
        for material_id in item.get("selected_material_ids") or []:
            record = id_to_record.get(str(material_id))
            if isinstance(record, Mapping):
                cluster_counts[_candidate_exploration_cluster(record)] += 1
        text = json.dumps(
            {
                "mechanisms": item.get("accepted_mechanisms"),
                "predictions": item.get("accepted_predictions"),
            },
            ensure_ascii=False,
        )
        for motif in _material_motifs_from_text(text):
            motif_counts[motif] += 1
    return {
        "recent_window_rounds": min(8, len(history)),
        "dominant_selected_clusters": [_cluster_summary_from_key(key, selected_count=count) for key, count in cluster_counts.most_common(4)],
        "dominant_text_motifs": [{"motif": motif, "count": count} for motif, count in motif_counts.most_common(6)],
    }


def _underexplored_candidate_clusters(
    history: Sequence[Any],
    candidate_pool: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not candidate_pool:
        return []
    candidate_counts: Counter[tuple[str, str, str]] = Counter()
    examples: dict[tuple[str, str, str], list[str]] = {}
    id_to_record: dict[str, Mapping[str, Any]] = {}
    for record in candidate_pool:
        if not isinstance(record, Mapping):
            continue
        key = _candidate_exploration_cluster(record)
        candidate_counts[key] += 1
        formula = str(record.get("formula") or "").strip()
        if formula:
            examples.setdefault(key, [])
            if len(examples[key]) < 3 and formula not in examples[key]:
                examples[key].append(formula)
        material_id = record.get("material_id")
        if material_id:
            id_to_record[str(material_id)] = record
    tested_counts: Counter[tuple[str, str, str]] = Counter()
    for item in history:
        if not isinstance(item, Mapping):
            continue
        for material_id in item.get("selected_material_ids") or []:
            record = id_to_record.get(str(material_id))
            if isinstance(record, Mapping):
                tested_counts[_candidate_exploration_cluster(record)] += 1
    candidate_keys = [
        key
        for key, count in candidate_counts.items()
        if _cluster_is_exploration_relevant(key) and ((_cluster_has_mixed_anions(key) and count >= 2) or count >= 5)
    ]
    candidate_keys.sort(
        key=lambda key: (
            tested_counts.get(key, 0),
            0 if _cluster_has_mixed_anions(key) else 1,
            -candidate_counts[key],
            key,
        )
    )
    return [
        _cluster_summary_from_key(
            key,
            candidate_count=candidate_counts[key],
            selected_count=tested_counts.get(key, 0),
            example_formulas=examples.get(key, []),
        )
        for key in candidate_keys[:limit]
    ]


def _candidate_exploration_cluster(record: Mapping[str, Any]) -> tuple[str, str, str]:
    elements = {str(item) for item in record.get("elements") or [] if str(item).strip()}
    anions = sorted(elements & EXPLORATION_ANIONS)
    family = _cation_family_label(elements - EXPLORATION_ANIONS)
    anion_label = "+".join(anions) if anions else "other"
    try:
        nelements = int(record.get("nelements") or len(elements))
    except (TypeError, ValueError):
        nelements = len(elements)
    if nelements <= 2:
        size_label = "binary"
    elif nelements == 3:
        size_label = "ternary"
    elif nelements == 4:
        size_label = "quaternary"
    else:
        size_label = "5plus"
    return family, anion_label, size_label


def _cation_family_label(cations: set[str]) -> str:
    families = []
    if cations & RARE_EARTH_ELEMENTS:
        families.append("rare_earth")
    if cations & ALKALI_ELEMENTS:
        families.append("alkali")
    if cations & ALKALINE_EARTH_ELEMENTS:
        families.append("alkaline_earth")
    if cations & TRANSITION_METAL_ELEMENTS:
        families.append("transition_metal")
    known = set().union(RARE_EARTH_ELEMENTS, ALKALI_ELEMENTS, ALKALINE_EARTH_ELEMENTS, TRANSITION_METAL_ELEMENTS)
    if cations - known:
        families.append("other_cation")
    if not families:
        return "no_common_cation"
    if len(families) == 1:
        return families[0]
    return "+".join(families[:3])


def _cluster_summary_from_key(
    key: tuple[str, str, str],
    *,
    candidate_count: int | None = None,
    selected_count: int | None = None,
    example_formulas: Sequence[str] | None = None,
) -> dict[str, Any]:
    family, anion_label, size_label = key
    summary = {
        "cation_family": family,
        "anion_motif": anion_label,
        "size": size_label,
        "candidate_count": candidate_count,
        "recent_or_total_selected_count": selected_count,
        "example_formulas": list(example_formulas or [])[:3],
    }
    return {item_key: value for item_key, value in summary.items() if value not in (None, [], "")}


def _cluster_is_exploration_relevant(key: tuple[str, str, str]) -> bool:
    _, anion_label, size_label = key
    return _cluster_has_mixed_anions(key) or size_label in {"quaternary", "5plus"} or anion_label not in {"O", "F"}


def _cluster_has_mixed_anions(key: tuple[str, str, str]) -> bool:
    return "+" in key[1]


def _material_motifs_from_text(text: str) -> list[str]:
    patterns = [
        r"\b[A-Z][a-z]?(?:[-/][A-Z][a-z]?){1,4}\b",
        r"\bO[-/]F[-/]S\b",
        r"\bO/F/S\b",
        r"\bO[-/]F\b",
        r"\bF[-/]S\b",
    ]
    motifs: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            motif = str(match).replace("/", "-")
            if motif not in motifs:
                motifs.append(motif)
    return motifs[:12]


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _compact_number(value: Any, *, digits: int) -> Any:
    if not isinstance(value, (int, float)):
        return value
    rounded = round(float(value), digits)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def _limit_tool_output(value: Any, max_chars: int) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return value
    if max_chars <= 160:
        return _short_text(value, max_chars)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        priority_keys = [
            "id",
            "name",
            "ok",
            "error",
            "tool_status",
            "state_path",
            "exists",
            "status",
            "controller_state_status",
            "current_round",
            "active_round",
            "history_len",
            "search_policy",
            "search_policy_scope",
            "xy_sequential_policy_warning",
            "current_principle_program",
            "principle_book_tail",
            "selection",
            "evidence_views",
            "offset",
            "limit",
            "next_offset",
            "has_more",
            "rounds",
            "usage_note",
            "valid_query",
            "candidate_pool_path",
            "match_count",
            "distinct_formula_count",
            "requested_examples",
            "examples_returned",
            "returned_distinct_formula_count",
            "first_distinct_formulas",
            "first_distinct_examples",
            "duplicate_formulas_in_examples",
            "examples",
            "usage_note",
        ]
        ordered_keys = [key for key in priority_keys if key in value]
        ordered_keys.extend(key for key in value if key not in set(ordered_keys))
        for key in ordered_keys:
            current_size = len(json.dumps(result, ensure_ascii=False, default=str))
            available = max_chars - current_size - 80
            if available <= 160:
                break
            item = value[key]
            item_text = json.dumps(item, ensure_ascii=False, default=str)
            if len(item_text) > available:
                item = _limit_tool_output(item, available)
            candidate = dict(result)
            candidate[str(key)] = item
            if len(json.dumps(candidate, ensure_ascii=False, default=str)) > max_chars - 40:
                candidate[str(key)] = _short_text(value[key], max(80, available // 2))
            result = candidate
        result["_truncated"] = True
        if max_chars > 400:
            result["_preview"] = text[: min(500, max_chars // 4)]
        return result
    if isinstance(value, list):
        if not value:
            return value
        per_item_budget = max(220, (max_chars - 120) // max(1, len(value)) - 40)
        limited: list[Any] = []
        for item in value:
            current_size = len(json.dumps(limited, ensure_ascii=False, default=str))
            available = max_chars - current_size - 80
            if available <= 160:
                break
            item_budget = max(160, min(per_item_budget, available))
            item_text = json.dumps(item, ensure_ascii=False, default=str)
            if len(item_text) > item_budget:
                limited.append(_limit_tool_output(item, item_budget))
            else:
                limited.append(item)
        if len(limited) < len(value):
            limited.append({"_truncated": True, "omitted_items": max(0, len(value) - len(limited))})
        if len(json.dumps(limited, ensure_ascii=False, default=str)) > max_chars and limited:
            smaller_budget = max(160, (max_chars - 120) // max(1, len(value)) - 100)
            limited = [
                _limit_tool_output(item, smaller_budget)
                for item in value[: len(value)]
            ]
            if len(json.dumps(limited, ensure_ascii=False, default=str)) > max_chars:
                limited = [
                    _short_text(item, max(80, smaller_budget // 2))
                    for item in value[: max(1, len(value) - 1)]
                ]
                limited.append({"_truncated": True, "omitted_items": max(0, len(value) - len(limited))})
        return limited
    return _short_text(value, max_chars)


def _truncate_for_trace(value: Any, max_chars: int) -> Any:
    return _limit_tool_output(_safe_json(value), max_chars)


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)
    return value


def _parse_first_json_value(text: str) -> Any:
    """Parse the first JSON value from text that may contain extra objects.

    Some models incorrectly emit a valid tool_request JSON object and then
    continue with a final answer JSON object in the same response.  For agent
    control, the leading tool_request must win so the runtime can execute the
    requested tools and ask for a grounded final response.
    """

    values = _parse_leading_json_values(text, max_values=1)
    return values[0] if values else None


def _parse_leading_json_values(text: str, *, max_values: int) -> list[Any]:
    stripped = text.strip()
    if not stripped:
        return []
    decoder = json.JSONDecoder()
    cursor_candidates = [index for index in (stripped.find("{"), stripped.find("[")) if index >= 0]
    if not cursor_candidates:
        return []
    cursor = min(cursor_candidates)
    values: list[Any] = []
    while len(values) < max_values and cursor < len(stripped):
        while cursor < len(stripped) and stripped[cursor].isspace():
            cursor += 1
        if cursor >= len(stripped) or stripped[cursor] not in "[{":
            break
        try:
            value, end = decoder.raw_decode(stripped[cursor:])
        except json.JSONDecodeError:
            break
        values.append(value)
        cursor += end
    return values
