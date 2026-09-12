from __future__ import annotations

import json
import sys

import pytest

from mmco.executor import Executor, ExecutorError, detect_refusal, parse_claude_output
from mmco.utils import ProcessResult, run_process

SUCCESS = {
    "type": "result", "subtype": "success", "is_error": False, "result": "Created app.py",
    "session_id": "abc-123", "total_cost_usd": 0.42, "duration_ms": 5100, "num_turns": 4,
    "stop_reason": "end_turn", "permission_denials": [],
}


class FakeRunner:
    def __init__(self, result: ProcessResult):
        self.result = result
        self.calls = []

    def __call__(self, cmd, cwd, stdin_text, timeout):
        self.calls.append({"cmd": cmd, "cwd": cwd, "stdin": stdin_text, "timeout": timeout})
        return self.result


def proc(stdout="", stderr="", returncode=0, timed_out=False):
    return ProcessResult(stdout=stdout, stderr=stderr, returncode=returncode, timed_out=timed_out, duration_ms=77)


def test_successful_run_is_parsed(settings, tmp_path):
    runner = FakeRunner(proc(json.dumps(SUCCESS)))
    result = Executor(settings, runner).run_task("build it", tmp_path)
    assert not result.is_error and result.result_text == "Created app.py"
    assert result.claude_session_id == "abc-123" and result.cost_usd == 0.42 and result.duration_ms == 5100
    call = runner.calls[0]
    assert call["stdin"] == "build it" and call["cwd"] == str(tmp_path)
    assert call["timeout"] == settings.claude_timeout_seconds
    cmd = call["cmd"]
    assert cmd[1] == "-p" and cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--allowedTools") + 1] == "Read,Write,Edit,Bash,Glob,Grep"
    assert "--resume" not in cmd


def test_resume_and_budget_flags(settings, tmp_path):
    settings.claude_max_budget_usd_per_task = 1.5
    settings.claude_model = "sonnet"
    runner = FakeRunner(proc(json.dumps(SUCCESS)))
    Executor(settings, runner).run_task("fix it", tmp_path, resume_session_id="abc-123")
    cmd = runner.calls[0]["cmd"]
    assert cmd[cmd.index("--resume") + 1] == "abc-123"
    assert cmd[cmd.index("--max-budget-usd") + 1] == "1.5"
    assert cmd[cmd.index("--model") + 1] == "sonnet"


def test_nonzero_exit_is_error(settings, tmp_path):
    result = Executor(settings, FakeRunner(proc("", "boom", returncode=1))).run_task("x", tmp_path)
    assert result.is_error and result.stderr == "boom" and result.exit_code == 1


def test_error_subtype_is_error(settings, tmp_path):
    payload = SUCCESS | {"subtype": "error_max_turns", "is_error": False}
    result = Executor(settings, FakeRunner(proc(json.dumps(payload)))).run_task("x", tmp_path)
    assert result.is_error and result.subtype == "error_max_turns"


def test_timeout_is_reported(settings, tmp_path):
    result = Executor(settings, FakeRunner(proc("", "", returncode=-15, timed_out=True))).run_task("x", tmp_path)
    assert result.timed_out and result.is_error and result.raw_output is None


def test_refusal_is_flagged(settings, tmp_path):
    payload = SUCCESS | {"result": "I can't help with modifying that system."}
    result = Executor(settings, FakeRunner(proc(json.dumps(payload)))).run_task("x", tmp_path)
    assert result.refusal_suspected


def test_missing_binary(settings):
    settings.claude_binary = "definitely-not-a-real-claude-binary"
    with pytest.raises(ExecutorError, match="not found on PATH"):
        Executor(settings).resolve_binary()


@pytest.mark.parametrize(
    ("text", "stop_reason", "expected"),
    [
        ("I can't assist with that request.", None, True),
        ("I won't do that.", None, True),
        ("Anything", "refusal", True),
        ("Created app.py and tests. All 4 tests pass.", None, False),
        # Refusal-like wording deep inside a normal report is not a refusal.
        ("Done. " + "x" * 500 + " I can't run the server here, but tests pass.", None, False),
    ],
)
def test_detect_refusal(text, stop_reason, expected):
    assert detect_refusal(text, stop_reason) is expected


def test_parse_output_variants():
    assert parse_claude_output(json.dumps(SUCCESS))["session_id"] == "abc-123"
    events = [{"type": "system"}, {"type": "assistant"}, SUCCESS]
    assert parse_claude_output(json.dumps(events))["result"] == "Created app.py"
    assert parse_claude_output("warning: something\n" + json.dumps(SUCCESS))["num_turns"] == 4
    assert parse_claude_output("") is None
    assert parse_claude_output("plain text") is None


def test_run_process_kills_on_timeout(tmp_path):
    result = run_process([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, None, timeout=0.5)
    assert result.timed_out and result.duration_ms < 15000


def _failed(text="", status=None, stderr="", is_error=True, timed_out=False):
    from mmco.models import ExecutionResult
    return ExecutionResult(prompt="p", result_text=text, stderr=stderr, is_error=is_error, timed_out=timed_out,
                           raw_output={"api_error_status": status} if status is not None else {})


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_failed("API Error: 529 Overloaded", status=529), "transient"),
        (_failed("", status=429), "transient"),
        (_failed("Request failed: socket hang up"), "transient"),
        (_failed("Credit balance is too low"), "fatal"),
        (_failed("Invalid API key · Please run /login"), "fatal"),
        (_failed("", status=401), "fatal"),
        (_failed("x" * 2000 + " implemented rate limiting middleware", is_error=True), None),
        (_failed("Done", is_error=False), None),
        (_failed("", timed_out=True), None),
    ],
)
def test_classify_failure(result, expected):
    from mmco.executor import classify_failure
    assert classify_failure(result) == expected


def test_system_prompt_choice_and_override(settings, tmp_path, isolated_config):
    from mmco.rules import project_config_dir
    project = tmp_path / "proj"
    runner = FakeRunner(proc(json.dumps(SUCCESS)))
    executor = Executor(settings, runner)
    executor.run_task("x", project, system_prompt="executor_system_minimal.txt")
    cmd = runner.calls[0]["cmd"]
    assert "mmco" not in cmd[cmd.index("--append-system-prompt") + 1]

    override = project_config_dir(project) / "prompts" / "executor_system.txt"
    override.parent.mkdir(parents=True)
    override.write_text("MY SYSTEM PROMPT")
    executor.run_task("x", project)
    cmd = runner.calls[1]["cmd"]
    assert cmd[cmd.index("--append-system-prompt") + 1] == "MY SYSTEM PROMPT"
