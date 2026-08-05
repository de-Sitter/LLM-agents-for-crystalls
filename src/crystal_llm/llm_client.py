"""Small OpenAI-compatible Responses API client.

The project intentionally avoids depending on the OpenAI SDK so the same code
can run on compute nodes with only the existing requirements installed.  The
endpoint is expected to accept OpenAI Responses-style POST requests at
``/v1/responses``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


_DIRECT_OPENER = build_opener(ProxyHandler({}))


def urlopen(request: Request, timeout: float | int | None = None):
    """Open LLM requests directly, ignoring proxy environment variables."""

    return _DIRECT_OPENER.open(request, timeout=timeout)


class LLMError(RuntimeError):
    """Raised when the configured LLM endpoint cannot return usable text."""


class _WallClockLLMTimeout(TimeoutError):
    """Raised when one LLM HTTP attempt exceeds the configured wall-clock limit."""


@contextmanager
def _wall_clock_timeout(seconds: float | int | None):
    """Best-effort wall-clock timeout for blocking relay reads on the main thread."""

    if (
        seconds is None
        or float(seconds) <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
    ):
        yield
        return

    timeout_seconds = float(seconds)
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(signum, frame):  # type: ignore[no-untyped-def]
        raise _WallClockLLMTimeout(f"LLM HTTP attempt exceeded wall timeout {timeout_seconds:.1f}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Load a minimal dotenv file into ``os.environ``.

    Only ``KEY=value`` lines are supported.  Quotes around values are stripped.
    Existing environment variables win unless ``override`` is true.
    """

    dotenv_path = Path(path)
    loaded: dict[str, str] = {}
    if not dotenv_path.exists():
        return loaded

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_llm_timeout_default() -> int:
    if os.getenv("LLM_TIMEOUT", "").strip():
        return _env_int("LLM_TIMEOUT", 120)
    max_call_seconds = _env_float("LLM_MAX_CALL_SECONDS", 0.0)
    if max_call_seconds > 0:
        return max(1, int(max_call_seconds))
    return 120


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_headers_json(name: str) -> dict[str, str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise LLMError(f"{name} must be a JSON object")
    headers: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not key.strip():
            raise LLMError(f"{name} contains an invalid header name")
        if not isinstance(value, str):
            raise LLMError(f"{name} header values must be strings")
        headers[key.strip()] = value
    return headers


def _env_param_names(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return ()
    names = {
        item.strip()
        for chunk in raw.split(",")
        for item in chunk.split()
        if item.strip()
    }
    return tuple(sorted(names))


def responses_endpoint(base_url: str) -> str:
    """Normalize ``LLM_BASE_URL`` to a concrete ``/v1/responses`` URL."""

    value = base_url.strip().rstrip("/")
    if not value:
        raise LLMError("LLM_BASE_URL is empty")
    if value.endswith("/v1/responses"):
        return value
    if value.endswith("/v1"):
        return f"{value}/responses"
    return f"{value}/v1/responses"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 8192
    timeout: int = 120
    user_agent: str = ""
    extra_headers: Mapping[str, str] | None = None
    stream: bool = False
    omit_params: tuple[str, ...] = ()

    @classmethod
    def from_env(
        cls,
        *,
        dotenv: str | Path | None = ".env",
        role: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
    ) -> "LLMConfig":
        if dotenv:
            load_dotenv(dotenv)
        role_key = role.upper() if role else ""
        role_model = os.getenv(f"LLM_{role_key}_MODEL") if role_key else None
        selected_model = model or role_model or os.getenv("LLM_MODEL", "")
        return cls(
            base_url=responses_endpoint(os.getenv("LLM_BASE_URL", "")),
            api_key=os.getenv("LLM_API_KEY", ""),
            model=selected_model,
            temperature=temperature if temperature is not None else _env_float("LLM_TEMPERATURE", 0.2),
            max_tokens=max_tokens if max_tokens is not None else _env_int("LLM_MAX_TOKENS", 8192),
            timeout=timeout if timeout is not None else _env_llm_timeout_default(),
            user_agent=os.getenv("LLM_USER_AGENT", "").strip(),
            extra_headers=_env_headers_json("LLM_EXTRA_HEADERS_JSON"),
            stream=_env_bool("LLM_STREAM", False),
            omit_params=_env_param_names("LLM_OMIT_PARAMS"),
        )

    def validate(self) -> None:
        missing = []
        if not self.base_url:
            missing.append("LLM_BASE_URL")
        if not self.api_key:
            missing.append("LLM_API_KEY")
        if not self.model:
            missing.append("LLM_MODEL")
        if missing:
            raise LLMError(f"missing LLM configuration: {', '.join(missing)}")


_METADATA_UNSUPPORTED_ENDPOINTS: set[str] = set()
_OPTIONAL_PARAM_UNSUPPORTED_ENDPOINTS: dict[str, set[str]] = {}
_TRANSIENT_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def message(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "content": [{"type": "input_text", "text": text}]}


def extract_response_text(payload: Mapping[str, Any]) -> str:
    """Extract assistant text from common Responses API response shapes."""

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    output = payload.get("output")
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
                for part in content:
                    if not isinstance(part, Mapping):
                        continue
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                    elif isinstance(part.get("content"), str):
                        chunks.append(str(part["content"]))
            elif isinstance(content, str):
                chunks.append(content)
    if chunks:
        text = "\n".join(chunk for chunk in chunks if chunk).strip()
        if text:
            return text

    choices = payload.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            msg = choice.get("message")
            if isinstance(msg, Mapping) and isinstance(msg.get("content"), str):
                chunks.append(msg["content"])
    if chunks:
        text = "\n".join(chunks).strip()
        if text:
            return text

    raise LLMError("could not extract assistant text from LLM response")


def parse_responses_sse(raw: str) -> dict[str, Any]:
    """Parse Responses API SSE text and synthesize a normal response object."""

    output_chunks: list[str] = []
    final_response: dict[str, Any] | None = None
    last_error: Any = None
    event_count = 0

    for block in raw.split("\n\n"):
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            continue
        data_text = "\n".join(data_lines).strip()
        if not data_text or data_text == "[DONE]":
            continue
        event_count += 1
        try:
            event = json.loads(data_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        event_type = event.get("type")
        if event_type == "response.output_text.delta" and isinstance(event.get("delta"), str):
            output_chunks.append(event["delta"])
        elif event_type == "response.completed" and isinstance(event.get("response"), Mapping):
            final_response = dict(event["response"])
        elif event_type in {"response.failed", "response.incomplete"}:
            last_error = event.get("error") or event.get("response") or event

    if final_response is None:
        if last_error is not None:
            raise LLMError(f"LLM stream ended with error: {last_error}")
        final_response = {"object": "response", "status": "completed"}
    output_text = "".join(output_chunks).strip()
    if output_text and not final_response.get("output_text"):
        final_response["output_text"] = output_text
    final_response["_stream_event_count"] = event_count
    return final_response


def extract_json_object(text: str) -> Any:
    """Parse JSON from raw model text or a fenced Markdown code block."""

    stripped = text.strip()
    if not stripped:
        raise ValueError("empty LLM text")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1).strip())

    start_candidates = [index for index in (stripped.find("{"), stripped.find("[")) if index >= 0]
    if not start_candidates:
        raise ValueError("LLM text does not contain a JSON object or array")
    start = min(start_candidates)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end <= start:
        raise ValueError("LLM text does not contain a complete JSON object or array")
    return json.loads(stripped[start : end + 1])


class ResponsesClient:
    """Minimal synchronous client for OpenAI-compatible Responses endpoints."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        log_dir: str | Path | None = None,
        retries: int | None = None,
        retry_sleep: float | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.retries = max(0, _env_int("LLM_RETRIES", 20) if retries is None else retries)
        self.retry_sleep = max(0.0, _env_float("LLM_RETRY_SLEEP", 5.0) if retry_sleep is None else retry_sleep)
        self.max_call_seconds = max(0.0, _env_float("LLM_MAX_CALL_SECONDS", 0.0))
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        *,
        input_messages: Sequence[Mapping[str, Any]],
        instructions: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        unsupported_params = set(self.config.omit_params)
        unsupported_params.update(_OPTIONAL_PARAM_UNSUPPORTED_ENDPOINTS.get(self.config.base_url, set()))
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": list(input_messages),
        }
        if "temperature" not in unsupported_params:
            payload["temperature"] = self.config.temperature if temperature is None else temperature
        token_limit = self.config.max_tokens if max_tokens is None else max_tokens
        if "max_output_tokens" not in unsupported_params:
            payload["max_output_tokens"] = token_limit
        elif "max_tokens" not in unsupported_params:
            payload["max_tokens"] = token_limit
        if instructions:
            payload["instructions"] = instructions
        if metadata and self.config.base_url not in _METADATA_UNSUPPORTED_ENDPOINTS:
            payload["metadata"] = dict(metadata)
        if self.config.stream:
            payload["stream"] = True

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if self.config.stream else "application/json",
        }
        if self.config.user_agent:
            headers["User-Agent"] = self.config.user_agent
        if self.config.extra_headers:
            headers.update(dict(self.config.extra_headers))

        last_error: Exception | None = None
        metadata_fallback_used = False
        max_output_tokens_fallback_used = False
        max_tokens_fallback_used = False
        temperature_fallback_used = False
        overall_started = time.time()
        transient_attempt = 0
        compatibility_fallbacks = 0
        max_compatibility_fallbacks = 8
        total_attempts = self.retries + 1
        deadline = overall_started + self.max_call_seconds if self.max_call_seconds > 0 else None
        while transient_attempt <= self.retries:
            attempt_number = transient_attempt + 1
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    last_error = TimeoutError(
                        f"LLM call exceeded LLM_MAX_CALL_SECONDS={self.max_call_seconds:.1f}s "
                        f"after {transient_attempt} transient attempt(s)"
                    )
                    break
                request_timeout: float | int | None = min(float(self.config.timeout), max(1.0, remaining))
            else:
                request_timeout = self.config.timeout
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = Request(
                self.config.base_url,
                data=body,
                headers=headers,
                method="POST",
            )
            started = time.time()
            try:
                with _wall_clock_timeout(request_timeout):
                    with urlopen(request, timeout=request_timeout) as response:
                        raw = response.read().decode("utf-8")
                stripped_raw = raw.lstrip()
                data = parse_responses_sse(raw) if stripped_raw.startswith(("event:", "data:")) else json.loads(raw)
                try:
                    extract_response_text(data)
                except LLMError:
                    status = data.get("status") if isinstance(data, Mapping) else None
                    response_error = data.get("error") if isinstance(data, Mapping) else None
                    last_error = LLMError(
                        f"LLM response status={status or 'unknown'} did not include assistant text"
                        + (f"; error={response_error}" if response_error else "")
                    )
                    self._log_call(payload, data, str(last_error), started)
                    if transient_attempt < self.retries:
                        time.sleep(self.retry_sleep * (transient_attempt + 1))
                        transient_attempt += 1
                        continue
                    break
                self._log_call(payload, data, None, started)
                return data
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if (
                    exc.code == 400
                    and "metadata" in payload
                    and "Unsupported parameter: metadata" in detail
                    and not metadata_fallback_used
                ):
                    fallback_payload = dict(payload)
                    fallback_payload.pop("metadata", None)
                    self._log_call(payload, None, f"metadata unsupported, retrying without metadata: {detail[:1000]}", started)
                    _METADATA_UNSUPPORTED_ENDPOINTS.add(self.config.base_url)
                    payload = fallback_payload
                    metadata_fallback_used = True
                    continue
                if (
                    exc.code == 400
                    and "max_output_tokens" in payload
                    and "Unsupported parameter: max_output_tokens" in detail
                    and not max_output_tokens_fallback_used
                ):
                    fallback_payload = dict(payload)
                    token_limit = fallback_payload.pop("max_output_tokens", None)
                    _OPTIONAL_PARAM_UNSUPPORTED_ENDPOINTS.setdefault(self.config.base_url, set()).add("max_output_tokens")
                    if (
                        token_limit is not None
                        and "max_tokens" not in self.config.omit_params
                        and "max_tokens" not in _OPTIONAL_PARAM_UNSUPPORTED_ENDPOINTS.get(self.config.base_url, set())
                    ):
                        fallback_payload["max_tokens"] = token_limit
                        fallback_note = "max_output_tokens unsupported, retrying with max_tokens"
                    else:
                        fallback_note = "max_output_tokens unsupported, retrying without token limit"
                    compatibility_fallbacks += 1
                    if compatibility_fallbacks > max_compatibility_fallbacks:
                        last_error = LLMError(f"too many LLM compatibility fallbacks after: {detail[:1000]}")
                        break
                    self._log_call(
                        payload,
                        None,
                        f"{fallback_note}: {detail[:1000]}",
                        started,
                    )
                    payload = fallback_payload
                    max_output_tokens_fallback_used = True
                    continue
                if (
                    exc.code == 400
                    and "max_tokens" in payload
                    and "Unsupported parameter: max_tokens" in detail
                    and not max_tokens_fallback_used
                ):
                    fallback_payload = dict(payload)
                    fallback_payload.pop("max_tokens", None)
                    _OPTIONAL_PARAM_UNSUPPORTED_ENDPOINTS.setdefault(self.config.base_url, set()).add("max_tokens")
                    compatibility_fallbacks += 1
                    if compatibility_fallbacks > max_compatibility_fallbacks:
                        last_error = LLMError(f"too many LLM compatibility fallbacks after: {detail[:1000]}")
                        break
                    self._log_call(
                        payload,
                        None,
                        f"max_tokens unsupported, retrying without token limit: {detail[:1000]}",
                        started,
                    )
                    payload = fallback_payload
                    max_tokens_fallback_used = True
                    continue
                if (
                    exc.code == 400
                    and "temperature" in payload
                    and "Unsupported parameter: temperature" in detail
                    and not temperature_fallback_used
                ):
                    fallback_payload = dict(payload)
                    fallback_payload.pop("temperature", None)
                    _OPTIONAL_PARAM_UNSUPPORTED_ENDPOINTS.setdefault(self.config.base_url, set()).add("temperature")
                    compatibility_fallbacks += 1
                    if compatibility_fallbacks > max_compatibility_fallbacks:
                        last_error = LLMError(f"too many LLM compatibility fallbacks after: {detail[:1000]}")
                        break
                    self._log_call(
                        payload,
                        None,
                        f"temperature unsupported, retrying without temperature: {detail[:1000]}",
                        started,
                    )
                    payload = fallback_payload
                    temperature_fallback_used = True
                    continue
                last_error = LLMError(f"LLM HTTP {exc.code}: {detail[:1000]}")
                if exc.code not in _TRANSIENT_HTTP_STATUS_CODES:
                    break
            except (
                URLError,
                TimeoutError,
                _WallClockLLMTimeout,
                json.JSONDecodeError,
                http.client.IncompleteRead,
                http.client.HTTPException,
            ) as exc:
                last_error = exc

            if transient_attempt < self.retries:
                self._log_call(
                    payload,
                    None,
                    f"transient LLM attempt {attempt_number}/{total_attempts} failed; retrying: {last_error}",
                    started,
                )
                sleep_seconds = self.retry_sleep * (transient_attempt + 1)
                if deadline is not None:
                    sleep_seconds = min(sleep_seconds, max(0.0, deadline - time.time()))
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            transient_attempt += 1

        self._log_call(payload, None, str(last_error), overall_started)
        raise LLMError(str(last_error))

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        response = self.create(
            instructions=system,
            input_messages=[message("user", user)],
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
        )
        return extract_response_text(response)

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        text = self.complete_text(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
        )
        return extract_json_object(text)

    def _log_call(
        self,
        request_payload: Mapping[str, Any],
        response_payload: Mapping[str, Any] | None,
        error: str | None,
        started: float,
    ) -> None:
        if not self.log_dir:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.log_dir / f"llm_call_{stamp}.json"
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - started, 3),
            "request": {
                key: value
                for key, value in request_payload.items()
                if key != "input"
            },
            "input": request_payload.get("input"),
            "response": response_payload,
            "error": error,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _input_messages_to_text(input_messages: Sequence[Mapping[str, Any]]) -> str:
    chunks: list[str] = []
    for message_payload in input_messages:
        role = str(message_payload.get("role") or "user")
        content = message_payload.get("content")
        parts: list[str] = []
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            for item in content:
                if isinstance(item, Mapping):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    elif isinstance(item.get("content"), str):
                        parts.append(str(item["content"]))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
        elif content is not None:
            parts.append(str(content))
        chunks.append(f"{role.upper()}:\n" + "\n".join(parts).strip())
    return "\n\n".join(chunk for chunk in chunks if chunk.strip())


class CodexCliClient:
    """Use the local Codex CLI as a text-completion backend.

    This is intentionally a narrow adapter for outage recovery when an HTTP
    relay rejects non-official clients.  It is not a drop-in Responses API
    service; each model call starts a short-lived ``codex exec`` process and
    reads its final message from ``--output-last-message``.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        log_dir: str | Path | None = None,
        retries: int | None = None,
        retry_sleep: float | None = None,
    ) -> None:
        self.config = config
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.retries = max(0, _env_int("LLM_CODEX_CLI_RETRIES", 1) if retries is None else retries)
        self.retry_sleep = max(0.0, _env_float("LLM_CODEX_CLI_RETRY_SLEEP", 10.0) if retry_sleep is None else retry_sleep)
        default_timeout = max(
            float(config.timeout or 0),
            _env_float("LLM_MAX_CALL_SECONDS", 0.0),
            900.0,
        )
        self.timeout = max(1.0, _env_float("LLM_CODEX_CLI_TIMEOUT", default_timeout))
        self.executable = os.getenv("LLM_CODEX_CLI_BIN", "codex").strip() or "codex"
        self.model = os.getenv("LLM_CODEX_CLI_MODEL", config.model).strip()
        self.sandbox = os.getenv("LLM_CODEX_CLI_SANDBOX", "read-only").strip() or "read-only"
        self.reasoning_effort = os.getenv("LLM_CODEX_CLI_REASONING_EFFORT", "").strip()
        self.extra_args = shlex.split(os.getenv("LLM_CODEX_CLI_EXTRA_ARGS", ""))
        self.ignore_user_config = _env_bool("LLM_CODEX_CLI_IGNORE_USER_CONFIG", True)

    def create(
        self,
        *,
        input_messages: Sequence[Mapping[str, Any]],
        instructions: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        del temperature, max_tokens
        user_text = _input_messages_to_text(input_messages)
        prompt = self._prompt(instructions=instructions or "", user_text=user_text)
        request_payload = {
            "backend": "codex_cli",
            "model": self.model,
            "input": list(input_messages),
            "instructions": instructions,
            "metadata": dict(metadata or {}),
        }
        last_error: Exception | None = None
        total_attempts = self.retries + 1
        for attempt in range(total_attempts):
            started = time.time()
            try:
                with tempfile.TemporaryDirectory(prefix="crystal_llm_codex_cli_") as tmpdir:
                    output_path = Path(tmpdir) / "last_message.txt"
                    cmd = self._command(output_path)
                    completed = subprocess.run(
                        cmd,
                        input=prompt,
                        text=True,
                        capture_output=True,
                        timeout=self.timeout,
                    )
                    output_text = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""
                    if completed.returncode != 0:
                        raise LLMError(
                            "Codex CLI exited with "
                            f"code {completed.returncode}: {self._tail(completed.stderr or completed.stdout)}"
                        )
                    if not output_text:
                        raise LLMError(f"Codex CLI produced no final message: {self._tail(completed.stdout or completed.stderr)}")
                    response_payload = {
                        "object": "response",
                        "status": "completed",
                        "output_text": output_text,
                        "codex_cli": {
                            "returncode": completed.returncode,
                            "stdout_tail": self._tail(completed.stdout),
                            "stderr_tail": self._tail(completed.stderr),
                            "attempt": attempt + 1,
                        },
                    }
                    self._log_call(request_payload, response_payload, None, started)
                    return response_payload
            except subprocess.TimeoutExpired as exc:
                last_error = LLMError(f"Codex CLI timed out after {self.timeout:.1f}s: {self._tail(exc.stdout or exc.stderr)}")
            except Exception as exc:
                last_error = exc
            self._log_call(
                request_payload,
                None,
                f"codex_cli attempt {attempt + 1}/{total_attempts} failed: {last_error}",
                started,
            )
            if attempt < self.retries and self.retry_sleep > 0:
                time.sleep(self.retry_sleep * (attempt + 1))
        raise LLMError(str(last_error))

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        response = self.create(
            instructions=system,
            input_messages=[message("user", user)],
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
        )
        return extract_response_text(response)

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        return extract_json_object(
            self.complete_text(
                system=system,
                user=user,
                temperature=temperature,
                max_tokens=max_tokens,
                metadata=metadata,
            )
        )

    def _prompt(self, *, instructions: str, user_text: str) -> str:
        return (
            "You are being used as a non-interactive text backend for a scientific controller.\n"
            "Do not inspect or edit files, do not run shell commands, and do not use Codex CLI tools.\n"
            "If the task prompt asks for a JSON object, return exactly that JSON object as the final message.\n"
            "If the task prompt asks for a JSON tool_request for the controller, output that JSON text; do not use native tools.\n"
            "Do not wrap the final answer in Markdown fences unless the task explicitly requires it.\n\n"
            "SYSTEM_INSTRUCTIONS:\n"
            f"{instructions.strip()}\n\n"
            "INPUT_MESSAGES:\n"
            f"{user_text.strip()}\n"
        )

    def _command(self, output_path: Path) -> list[str]:
        cmd = [
            self.executable,
            "exec",
            "--ephemeral",
        ]
        if self.ignore_user_config:
            cmd.append("--ignore-user-config")
        cmd.extend(
            [
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                self.sandbox,
                "--color",
                "never",
            ]
        )
        if self.model:
            cmd.extend(["-m", self.model])
        if self.reasoning_effort:
            cmd.extend(["-c", f'model_reasoning_effort="{self.reasoning_effort}"'])
        cmd.extend(self.extra_args)
        cmd.extend(["-o", str(output_path), "-"])
        return cmd

    @staticmethod
    def _tail(text: str | bytes | None, *, limit: int = 2000) -> str:
        if text is None:
            return ""
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        return str(text).strip()[-limit:]

    def _log_call(
        self,
        request_payload: Mapping[str, Any],
        response_payload: Mapping[str, Any] | None,
        error: str | None,
        started: float,
    ) -> None:
        if not self.log_dir:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.log_dir / f"llm_call_{stamp}.json"
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - started, 3),
            "request": {
                key: value
                for key, value in request_payload.items()
                if key != "input"
            },
            "input": request_payload.get("input"),
            "response": response_payload,
            "error": error,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_llm_client(
    config: LLMConfig,
    *,
    log_dir: str | Path | None = None,
    retries: int | None = None,
    retry_sleep: float | None = None,
) -> ResponsesClient | CodexCliClient:
    backend = os.getenv("LLM_BACKEND", "responses").strip().lower().replace("-", "_")
    if backend in {"responses", "http", "openai"}:
        return ResponsesClient(config, log_dir=log_dir, retries=retries, retry_sleep=retry_sleep)
    if backend in {"codex", "codex_cli", "cli"}:
        return CodexCliClient(config, log_dir=log_dir, retries=retries, retry_sleep=retry_sleep)
    raise LLMError(f"unsupported LLM_BACKEND={backend!r}; expected responses or codex_cli")
