from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
from urllib.error import HTTPError

import pytest

from crystal_llm import llm_client
from crystal_llm.llm_client import CodexCliClient, LLMConfig, ResponsesClient, make_llm_client, message


@pytest.fixture(autouse=True)
def clear_llm_client_compatibility_caches():
    llm_client._METADATA_UNSUPPORTED_ENDPOINTS.clear()
    llm_client._OPTIONAL_PARAM_UNSUPPORTED_ENDPOINTS.clear()
    yield
    llm_client._METADATA_UNSUPPORTED_ENDPOINTS.clear()
    llm_client._OPTIONAL_PARAM_UNSUPPORTED_ENDPOINTS.clear()


class _DummyResponse:
    def __init__(self, body: bytes | None = None) -> None:
        self.body = body

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        if self.body is not None:
            return self.body
        return json.dumps({"output_text": "{\"ok\": true}"}).encode("utf-8")


def test_module_urlopen_uses_direct_opener(monkeypatch) -> None:
    calls: list[tuple[object, object]] = []

    class FakeDirectOpener:
        def open(self, request, timeout=None):
            calls.append((request, timeout))
            return _DummyResponse()

    monkeypatch.setattr(llm_client, "_DIRECT_OPENER", FakeDirectOpener())
    request = llm_client.Request("https://relay.example/v1/responses")

    with llm_client.urlopen(request, timeout=12) as response:
        assert response.read()

    assert calls == [(request, 12)]


def test_responses_client_caches_metadata_unsupported_endpoint(monkeypatch, tmp_path) -> None:
    seen_payloads: list[dict[str, object]] = []

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        seen_payloads.append(payload)
        if len(seen_payloads) == 1:
            detail = b'{"error":{"message":"Unsupported parameter: metadata"}}'
            raise HTTPError(request.full_url, 400, "bad request", hdrs=None, fp=io.BytesIO(detail))
        return _DummyResponse()

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    config = LLMConfig(
        base_url="https://relay.example/v1/responses",
        api_key="test-key",
        model="test-model",
        timeout=1,
    )

    client = ResponsesClient(config, log_dir=tmp_path / "first", retries=1, retry_sleep=0)
    client.create(input_messages=[message("user", "hello")], metadata={"role": "first"})

    assert "metadata" in seen_payloads[0]
    assert "metadata" not in seen_payloads[1]
    assert config.base_url in llm_client._METADATA_UNSUPPORTED_ENDPOINTS

    second = ResponsesClient(config, log_dir=tmp_path / "second", retries=1, retry_sleep=0)
    second.create(input_messages=[message("user", "hello")], metadata={"role": "second"})

    assert "metadata" not in seen_payloads[2]


def test_responses_client_retries_transient_http_errors(monkeypatch, tmp_path) -> None:
    attempts = 0

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            detail = b"<html>502 bad gateway</html>"
            raise HTTPError(request.full_url, 502, "bad gateway", hdrs=None, fp=io.BytesIO(detail))
        return _DummyResponse()

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    config = LLMConfig(
        base_url="https://relay.example/v1/responses",
        api_key="test-key",
        model="test-model",
        timeout=1,
    )

    client = ResponsesClient(config, log_dir=tmp_path, retries=1, retry_sleep=0)
    response = client.create(input_messages=[message("user", "hello")])

    assert response["output_text"] == "{\"ok\": true}"
    assert attempts == 2


def test_responses_client_logs_each_transient_retry(monkeypatch, tmp_path) -> None:
    attempts = 0

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("read timed out")
        return _DummyResponse()

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    config = LLMConfig(
        base_url="https://relay.example/v1/responses",
        api_key="test-key",
        model="test-model",
        timeout=1,
    )

    client = ResponsesClient(config, log_dir=tmp_path, retries=1, retry_sleep=0)
    response = client.create(input_messages=[message("user", "hello")])

    assert response["output_text"] == "{\"ok\": true}"
    logs = [json.loads(path.read_text()) for path in sorted(tmp_path.glob("llm_call_*.json"))]
    assert len(logs) == 2
    assert "transient LLM attempt 1/2 failed; retrying: read timed out" in logs[0]["error"]
    assert logs[1]["error"] is None


def test_responses_client_retries_completed_response_without_text(monkeypatch, tmp_path) -> None:
    attempts = 0

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            body = {
                "id": "resp-empty",
                "status": "completed",
                "output": [],
                "usage": {"output_tokens": 123},
            }
            return _DummyResponse(json.dumps(body).encode("utf-8"))
        return _DummyResponse()

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    config = LLMConfig(
        base_url="https://relay.example/v1/responses",
        api_key="test-key",
        model="test-model",
        timeout=1,
    )

    client = ResponsesClient(config, log_dir=tmp_path, retries=1, retry_sleep=0)
    response = client.create(input_messages=[message("user", "hello")])

    assert response["output_text"] == "{\"ok\": true}"
    assert attempts == 2
    logs = sorted(tmp_path.glob("llm_call_*.json"))
    assert len(logs) == 2
    first = json.loads(logs[0].read_text())
    assert "did not include assistant text" in first["error"]
    assert first["response"]["output"] == []


def test_responses_client_parses_sse_response_when_stream_not_requested(monkeypatch, tmp_path) -> None:
    raw = b"""event: response.output_text.delta
data: {\"type\":\"response.output_text.delta\",\"delta\":\"{\\\"ok\\\": true}\"}

event: response.completed
data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}

"""

    def fake_urlopen(request, timeout):
        return _DummyResponse(raw)

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    config = LLMConfig(
        base_url="https://relay.example/v1/responses",
        api_key="test-key",
        model="test-model",
        timeout=1,
        stream=False,
    )

    client = ResponsesClient(config, log_dir=tmp_path, retries=0, retry_sleep=0)
    response = client.create(input_messages=[message("user", "hello")])

    assert response["output_text"] == "{\"ok\": true}"


def test_responses_client_falls_back_to_max_tokens(monkeypatch, tmp_path) -> None:
    seen_payloads: list[dict[str, object]] = []

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        seen_payloads.append(payload)
        if len(seen_payloads) == 1:
            detail = b'{"error":{"message":"Unsupported parameter: max_output_tokens"}}'
            raise HTTPError(request.full_url, 400, "bad request", hdrs=None, fp=io.BytesIO(detail))
        return _DummyResponse()

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    config = LLMConfig(
        base_url="https://relay.example/v1/responses",
        api_key="test-key",
        model="test-model",
        max_tokens=1234,
        timeout=1,
    )

    client = ResponsesClient(config, log_dir=tmp_path, retries=1, retry_sleep=0)
    response = client.create(input_messages=[message("user", "hello")])

    assert response["output_text"] == "{\"ok\": true}"
    assert seen_payloads[0]["max_output_tokens"] == 1234
    assert "max_tokens" not in seen_payloads[0]
    assert seen_payloads[1]["max_tokens"] == 1234
    assert "max_output_tokens" not in seen_payloads[1]


def test_responses_client_omits_token_limit_when_both_token_params_unsupported(monkeypatch, tmp_path) -> None:
    seen_payloads: list[dict[str, object]] = []

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        seen_payloads.append(payload)
        if "max_output_tokens" in payload:
            detail = b'{"error":{"message":"Unsupported parameter: max_output_tokens"}}'
            raise HTTPError(request.full_url, 400, "bad request", hdrs=None, fp=io.BytesIO(detail))
        if "max_tokens" in payload:
            detail = b'{"error":{"message":"Unsupported parameter: max_tokens"}}'
            raise HTTPError(request.full_url, 400, "bad request", hdrs=None, fp=io.BytesIO(detail))
        return _DummyResponse()

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    config = LLMConfig(
        base_url="https://relay.example/v1/responses",
        api_key="test-key",
        model="test-model",
        max_tokens=1234,
        timeout=1,
    )

    client = ResponsesClient(config, log_dir=tmp_path / "first", retries=0, retry_sleep=0)
    response = client.create(input_messages=[message("user", "hello")])

    assert response["output_text"] == "{\"ok\": true}"
    assert seen_payloads[0]["max_output_tokens"] == 1234
    assert seen_payloads[1]["max_tokens"] == 1234
    assert "max_output_tokens" not in seen_payloads[2]
    assert "max_tokens" not in seen_payloads[2]
    assert llm_client._OPTIONAL_PARAM_UNSUPPORTED_ENDPOINTS[config.base_url] == {
        "max_output_tokens",
        "max_tokens",
    }

    second = ResponsesClient(config, log_dir=tmp_path / "second", retries=0, retry_sleep=0)
    second.create(input_messages=[message("user", "hello")])

    assert "max_output_tokens" not in seen_payloads[3]
    assert "max_tokens" not in seen_payloads[3]


def test_responses_client_omits_unsupported_temperature(monkeypatch, tmp_path) -> None:
    seen_payloads: list[dict[str, object]] = []

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        seen_payloads.append(payload)
        if len(seen_payloads) == 1:
            detail = b'{"error":{"message":"Unsupported parameter: temperature"}}'
            raise HTTPError(request.full_url, 400, "bad request", hdrs=None, fp=io.BytesIO(detail))
        return _DummyResponse()

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    config = LLMConfig(
        base_url="https://relay.example/v1/responses",
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        timeout=1,
    )

    client = ResponsesClient(config, log_dir=tmp_path, retries=1, retry_sleep=0)
    response = client.create(input_messages=[message("user", "hello")])

    assert response["output_text"] == "{\"ok\": true}"
    assert seen_payloads[0]["temperature"] == 0.2
    assert "temperature" not in seen_payloads[1]


def test_responses_client_honors_omit_params_from_env(monkeypatch, tmp_path) -> None:
    seen_payloads: list[dict[str, object]] = []

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        seen_payloads.append(payload)
        return _DummyResponse()

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("LLM_BASE_URL", "https://relay.example")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLM_MAX_TOKENS", "1234")
    monkeypatch.setenv("LLM_OMIT_PARAMS", "temperature,max_output_tokens,max_tokens")
    config = LLMConfig.from_env(dotenv=None)

    client = ResponsesClient(config, log_dir=tmp_path, retries=0, retry_sleep=0)
    client.create(input_messages=[message("user", "hello")])

    assert "temperature" not in seen_payloads[0]
    assert "max_output_tokens" not in seen_payloads[0]
    assert "max_tokens" not in seen_payloads[0]


def test_llm_config_uses_max_call_seconds_as_default_timeout(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://relay.example")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_MAX_CALL_SECONDS", "300")
    monkeypatch.delenv("LLM_TIMEOUT", raising=False)

    config = LLMConfig.from_env(dotenv=None)

    assert config.timeout == 300


def test_llm_config_timeout_env_overrides_max_call_seconds(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://relay.example")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_MAX_CALL_SECONDS", "300")
    monkeypatch.setenv("LLM_TIMEOUT", "45")

    config = LLMConfig.from_env(dotenv=None)

    assert config.timeout == 45


def test_responses_client_reads_retry_defaults_from_env(monkeypatch, tmp_path) -> None:
    attempts = 0
    monkeypatch.setenv("LLM_RETRIES", "2")
    monkeypatch.setenv("LLM_RETRY_SLEEP", "0")

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            detail = b"<html>502 bad gateway</html>"
            raise HTTPError(request.full_url, 502, "bad gateway", hdrs=None, fp=io.BytesIO(detail))
        return _DummyResponse()

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    config = LLMConfig(
        base_url="https://relay.example/v1/responses",
        api_key="test-key",
        model="test-model",
        timeout=1,
    )

    client = ResponsesClient(config, log_dir=tmp_path)
    response = client.create(input_messages=[message("user", "hello")])

    assert response["output_text"] == "{\"ok\": true}"
    assert attempts == 3


def test_responses_client_does_not_retry_non_transient_http_errors(monkeypatch, tmp_path) -> None:
    attempts = 0

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        detail = b'{"error":{"message":"bad request"}}'
        raise HTTPError(request.full_url, 400, "bad request", hdrs=None, fp=io.BytesIO(detail))

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    config = LLMConfig(
        base_url="https://relay.example/v1/responses",
        api_key="test-key",
        model="test-model",
        timeout=1,
    )

    client = ResponsesClient(config, log_dir=tmp_path, retries=3, retry_sleep=0)
    try:
        client.create(input_messages=[message("user", "hello")])
    except llm_client.LLMError as exc:
        assert "LLM HTTP 400" in str(exc)
    else:
        raise AssertionError("expected LLMError")

    assert attempts == 1


def test_responses_client_failure_log_records_total_elapsed(monkeypatch, tmp_path) -> None:
    def fake_urlopen(request, timeout):
        raise TimeoutError("timed out")

    times = iter([100.0, 100.1, 160.0])

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_client.time, "time", lambda: next(times))
    config = LLMConfig(
        base_url="https://relay.example/v1/responses",
        api_key="test-key",
        model="test-model",
        timeout=1,
    )

    client = ResponsesClient(config, log_dir=tmp_path, retries=0, retry_sleep=0)
    try:
        client.create(input_messages=[message("user", "hello")])
    except llm_client.LLMError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected LLMError")

    [log_path] = list(tmp_path.glob("llm_call_*.json"))
    logged = json.loads(log_path.read_text())
    assert logged["elapsed_seconds"] == 60.0


def test_make_llm_client_selects_codex_cli_backend(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_BACKEND", "codex_cli")
    config = LLMConfig(
        base_url="https://relay.example/v1/responses",
        api_key="test-key",
        model="test-model",
        timeout=1,
    )

    client = make_llm_client(config, log_dir=tmp_path, retries=0)

    assert isinstance(client, CodexCliClient)


def test_codex_cli_client_returns_output_last_message(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(cmd, *, input, text, capture_output, timeout):
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_text('{"ok":true,"backend":"codex_cli"}\n', encoding="utf-8")
        calls.append(
            {
                "cmd": cmd,
                "input": input,
                "text": text,
                "capture_output": capture_output,
                "timeout": timeout,
            }
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="codex stdout", stderr="")

    monkeypatch.setattr(llm_client.subprocess, "run", fake_run)
    monkeypatch.setenv("LLM_CODEX_CLI_TIMEOUT", "12")
    config = LLMConfig(
        base_url="https://relay.example/v1/responses",
        api_key="test-key",
        model="test-model",
        timeout=1,
    )
    client = CodexCliClient(config, log_dir=tmp_path, retries=0)

    response = client.create(
        instructions="Return JSON.",
        input_messages=[message("user", "hello")],
        metadata={"role": "test"},
    )

    assert response["output_text"] == '{"ok":true,"backend":"codex_cli"}'
    assert calls
    cmd = calls[0]["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:2] == ["codex", "exec"]
    assert "--ephemeral" in cmd
    assert "--ignore-user-config" in cmd
    assert "--ignore-rules" in cmd
    assert "--skip-git-repo-check" in cmd
    assert "-m" in cmd
    assert cmd[cmd.index("-m") + 1] == "test-model"
    assert calls[0]["timeout"] == 12.0
    assert "SYSTEM_INSTRUCTIONS" in str(calls[0]["input"])
    [log_path] = list(tmp_path.glob("llm_call_*.json"))
    logged = json.loads(log_path.read_text())
    assert logged["request"]["backend"] == "codex_cli"
    assert logged["error"] is None


def test_codex_cli_client_raises_on_failed_process(monkeypatch, tmp_path) -> None:
    def fake_run(cmd, *, input, text, capture_output, timeout):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="bad auth")

    monkeypatch.setattr(llm_client.subprocess, "run", fake_run)
    config = LLMConfig(
        base_url="https://relay.example/v1/responses",
        api_key="test-key",
        model="test-model",
        timeout=1,
    )
    client = CodexCliClient(config, log_dir=tmp_path, retries=0)

    with pytest.raises(llm_client.LLMError) as exc_info:
        client.create(input_messages=[message("user", "hello")])

    assert "Codex CLI exited with code 2" in str(exc_info.value)
