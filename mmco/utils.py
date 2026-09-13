"""Shared helpers: logging, prompts, JSON extraction, subprocesses, crash dumps."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

from loguru import logger

PROMPTS_DIR = Path(__file__).parent / "prompts"

_session_sinks: dict[str, int] = {}


class MMCOError(Exception):
    """An error that is reported to the user without a traceback."""


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    _session_sinks.clear()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format="<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> {message}",
    )


def add_session_log(log_dir: str | Path, session_id: str) -> Path:
    """Attach a structured (JSON lines) log file for one session."""
    path = Path(log_dir).expanduser().resolve() / f"mmco-{session_id}.log"
    if str(path) not in _session_sinks:
        # One process may run several sessions (chat mode); each file gets only its own session.
        for sink_id in _session_sinks.values():
            logger.remove(sink_id)
        _session_sinks.clear()
        path.parent.mkdir(parents=True, exist_ok=True)
        _session_sinks[str(path)] = logger.add(path, level="DEBUG", serialize=True)
    return path


def load_prompt(name: str, *override_dirs: str | Path | None) -> str:
    """Return the first override found (most specific directory first), else the built-in prompt."""
    for directory in override_dirs:
        if directory:
            candidate = Path(directory).expanduser() / name
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def render(template: str, **values: Any) -> str:
    return Template(template).safe_substitute({k: str(v) for k, v in values.items()})


def truncate(text: str, limit: int) -> str:
    """Keep the head and tail of long text, which is where errors usually are."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    half = max(limit // 2, 1)
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n... [{omitted} chars truncated] ...\n{text[-half:]}"


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse JSON from an LLM response, tolerating fences and surrounding prose."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for match in _FENCE.finditer(text):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text, index)
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in response")


@dataclass
class ProcessResult:
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool
    duration_ms: int


def run_process(cmd: list[str], cwd: str | Path, stdin_text: str | None, timeout: float) -> ProcessResult:
    """Run a command in its own process group so a timeout kills its children too."""
    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(stdin_text or "", timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
        stdout, stderr = proc.communicate()
    except KeyboardInterrupt:
        _kill_group(proc)
        raise
    return ProcessResult(
        stdout=stdout or "",
        stderr=stderr or "",
        returncode=proc.returncode,
        timed_out=timed_out,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def run_process_streaming(
    cmd: list[str], cwd: str | Path, stdin_text: str | None, timeout: float, on_line
) -> ProcessResult:
    """Like run_process, but call on_line(line) for each stdout line as it arrives.

    Used to stream Claude Code's stream-json output live. The full stdout is still accumulated and
    returned, so callers parse the final result exactly as with run_process. A watchdog kills the
    process group on timeout; stderr is drained in a thread to avoid a full-pipe deadlock.
    """
    started = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    out_lines: list[str] = []
    err_chunks: list[str] = []
    timed_out = {"value": False}

    def drain_err() -> None:
        try:
            for line in proc.stderr:  # type: ignore[union-attr]
                err_chunks.append(line)
        except (OSError, ValueError):
            pass

    def watchdog() -> None:
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out["value"] = True
            _kill_group(proc)

    err_thread = threading.Thread(target=drain_err, daemon=True)
    watch_thread = threading.Thread(target=watchdog, daemon=True)
    err_thread.start()
    watch_thread.start()
    try:
        if stdin_text:
            proc.stdin.write(stdin_text)  # type: ignore[union-attr]
        proc.stdin.close()  # type: ignore[union-attr]
    except (BrokenPipeError, OSError, ValueError):
        pass
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            out_lines.append(line)
            try:
                on_line(line)
            except Exception:  # noqa: BLE001 - a bad line must not kill the run
                pass
    except KeyboardInterrupt:
        _kill_group(proc)
        raise
    proc.wait()
    err_thread.join(timeout=2)
    watch_thread.join(timeout=2)
    return ProcessResult(
        stdout="".join(out_lines),
        stderr="".join(err_chunks),
        returncode=proc.returncode,
        timed_out=timed_out["value"],
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def dump_crash(payload: dict[str, Any], directory: str | Path = ".") -> Path:
    path = Path(directory) / f"mmco-crash-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)
