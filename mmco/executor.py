"""Executor: runs one coding task through the Claude Code CLI in headless mode."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from .config import Settings
from .models import ExecutionResult
from .rules import prompt_dirs
from .utils import MMCOError, ProcessResult, load_prompt, run_process

REFUSAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bI can'?t\b",
        r"\bI won'?t\b",
        r"\bI'?m not able\b",
        r"\bcannot assist\b",
        r"\bagainst my\b",
    )
]
# Refusals are stated up front; scanning the whole report flags lines like
# "I can't find config.py, so I created it".
REFUSAL_SCAN_CHARS = 400

_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
_TRANSIENT_TEXT = re.compile(
    r"overloaded|rate.?limit|too many requests|temporarily unavailable|ECONNRESET|ETIMEDOUT|socket hang up"
    r"|network error|API Error: 5\d\d",
    re.IGNORECASE,
)
_FATAL_TEXT = re.compile(
    r"credit balance is too low|invalid api key|please run /login|not logged in|authentication_error"
    r"|usage limit|limit reached|out of extra usage",
    re.IGNORECASE,
)
_BINARY_FALLBACKS = ("~/.local/bin/claude", "~/.claude/local/claude", "/opt/homebrew/bin/claude", "/usr/local/bin/claude")

Runner = Callable[[list[str], str, str | None, float], ProcessResult]


class ExecutorError(MMCOError):
    pass


def detect_refusal(result_text: str, stop_reason: str | None = None) -> bool:
    if stop_reason == "refusal":
        return True
    head = (result_text or "").strip()[:REFUSAL_SCAN_CHARS]
    return any(pattern.search(head) for pattern in REFUSAL_PATTERNS)


def classify_failure(result: ExecutionResult) -> Literal["transient", "fatal"] | None:
    """Separate infrastructure failures (retry later / needs the user) from coding failures (planner decides)."""
    if not result.is_error or result.timed_out:
        return None
    status = (result.raw_output or {}).get("api_error_status")
    text = f"{result.result_text}\n{result.stderr}"
    if status in (401, 403) or _FATAL_TEXT.search(text[-4000:]):
        return "fatal"
    # A long report is a real work session that happened to end in an error, not an API outage.
    if status in _TRANSIENT_STATUS or (len(result.result_text) < 600 and _TRANSIENT_TEXT.search(text[-4000:])):
        return "transient"
    return None


def parse_claude_output(stdout: str) -> dict[str, Any] | None:
    """Parse `--output-format json` defensively: a dict, a list of events, or JSON lines."""
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
        for line in reversed(text.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                data = candidate
                break
    if isinstance(data, list):
        results = [item for item in data if isinstance(item, dict) and item.get("type") == "result"]
        data = results[-1] if results else None
    return data if isinstance(data, dict) else None


class Executor:
    def __init__(self, settings: Settings, runner: Runner = run_process):
        self.settings = settings
        self.runner = runner

    def resolve_binary(self) -> str:
        binary = os.path.expanduser(self.settings.claude_binary)
        path = shutil.which(binary)
        if not path and self.settings.claude_binary == "claude":
            path = next((shutil.which(os.path.expanduser(c)) for c in _BINARY_FALLBACKS
                         if shutil.which(os.path.expanduser(c))), None)
        if not path:
            raise ExecutorError(
                f"Claude Code binary '{self.settings.claude_binary}' not found on PATH. Shell aliases are not "
                "visible to subprocesses; set MMCO_CLAUDE_BINARY to an absolute path (e.g. ~/.local/bin/claude)."
            )
        return path

    def build_command(
        self,
        resume_session_id: str | None = None,
        project_dir: str | Path | None = None,
        system_prompt: str = "executor_system.txt",
    ) -> list[str]:
        s = self.settings
        # The prompt goes through stdin: it can be long, and --allowedTools is variadic,
        # so a trailing positional prompt would be swallowed as a tool name.
        cmd = [
            self.resolve_binary(),
            "-p",
            "--output-format", "json",
            "--allowedTools", s.claude_allowed_tools,
            "--permission-mode", s.claude_permission_mode,
            "--append-system-prompt", load_prompt(system_prompt, *prompt_dirs(project_dir, s.prompts_dir)),
        ]
        if s.claude_model:
            cmd += ["--model", s.claude_model]
        if s.claude_max_budget_usd_per_task:
            cmd += ["--max-budget-usd", str(s.claude_max_budget_usd_per_task)]
        if resume_session_id:
            cmd += ["--resume", resume_session_id]
        return cmd

    def run_task(
        self,
        prompt: str,
        project_dir: str | Path,
        resume_session_id: str | None = None,
        system_prompt: str = "executor_system.txt",
    ) -> ExecutionResult:
        cmd = self.build_command(resume_session_id, project_dir, system_prompt)
        logger.bind(event="executor_call", prompt=prompt, resume=resume_session_id).debug(
            "running claude in {} (resume={})", project_dir, resume_session_id
        )
        proc = self.runner(cmd, str(project_dir), prompt, self.settings.claude_timeout_seconds)
        data = parse_claude_output(proc.stdout)

        if data is not None:
            result_text = str(data.get("result") or "")
        else:
            result_text = proc.stdout.strip()
        subtype = data.get("subtype") if data else None
        stop_reason = data.get("stop_reason") if data else None
        is_error = (
            proc.timed_out
            or proc.returncode != 0
            or data is None
            or bool(data.get("is_error"))
            or (subtype is not None and subtype != "success")
        )
        denials = data.get("permission_denials") if data else None

        result = ExecutionResult(
            prompt=prompt,
            raw_output=data,
            result_text=result_text,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            duration_ms=int(data.get("duration_ms") or proc.duration_ms) if data else proc.duration_ms,
            timed_out=proc.timed_out,
            is_error=is_error,
            subtype=subtype,
            stop_reason=stop_reason,
            refusal_suspected=detect_refusal(result_text, stop_reason),
            permission_denials=denials if isinstance(denials, list) else [],
            claude_session_id=data.get("session_id") if data else None,
            cost_usd=_as_float(data.get("total_cost_usd")) if data else None,
            num_turns=data.get("num_turns") if data else None,
        )
        if proc.timed_out:
            logger.warning("claude timed out after {}s", self.settings.claude_timeout_seconds)
        elif is_error:
            logger.warning("claude run failed (exit={}, subtype={}): {}", proc.returncode, subtype,
                           (proc.stderr or result_text)[:300])
        logger.bind(event="executor_result", result=result.model_dump(exclude={"prompt"})).debug(
            "claude finished in {:.1f}s, cost ${}", result.duration_ms / 1000, result.cost_usd
        )
        return result


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
