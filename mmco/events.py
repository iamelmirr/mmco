"""Live event stream shared by the planner and executor.

While a session runs, both models stream what they are doing — DeepSeek's reasoning, text and tool
calls, and Claude Code's thinking, tool use and results — into one per-session JSON-lines file
(`<log_dir>/events/<session_id>.jsonl`). The dashboard and `mmco logs -f` tail that file; when a
session runs in the foreground the events are also echoed to the terminal as they happen.

The active sink is a process-global set by the orchestrator for the session it is driving, so the
planner and executor can emit without threading a sink argument through every call. When no sink is
active (unit tests, or streaming disabled) emitting is a no-op and the models fall back to plain,
non-streaming calls.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

SOURCE_LABEL = {"planner": "DeepSeek", "claude": "Claude", "orchestrator": "mmco", "checks": "checks"}

_active: "EventSink | None" = None


def set_active_sink(sink: "EventSink | None") -> None:
    global _active
    _active = sink


def clear_active_sink(sink: "EventSink | None" = None) -> None:
    """Clear the active sink (only if it is `sink`, when given, to avoid clobbering a newer one)."""
    global _active
    if sink is None or _active is sink:
        _active = None


def active_sink() -> "EventSink | None":
    return _active


def _clip(text: str, limit: int = 200) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def summarize(source: str, etype: str, text: str, data: dict[str, Any]) -> str:
    """A short one-line human description of an event, for terminal echo and `mmco logs`."""
    if etype == "thinking":
        return f"… {_clip(text, 160)}" if text.strip() else ""
    if etype == "text":
        return _clip(text, 200) if text.strip() else ""
    if etype == "tool":
        name = data.get("name", "tool")
        args = data.get("args") or data.get("input") or ""
        if isinstance(args, (dict, list)):
            args = json.dumps(args, ensure_ascii=False)
        detail = _clip(args, 120)
        return f"⚙ {name}({detail})" if detail else f"⚙ {name}"
    if etype == "tool_result":
        flag = "✗" if data.get("is_error") else "↳"
        return f"{flag} {_clip(text, 140)}" if text.strip() else f"{flag} (done)"
    if etype == "task_start":
        return f"▶ {data.get('title', '')} — {SOURCE_LABEL.get(data.get('executor', ''), data.get('executor', ''))}"
    if etype == "verdict":
        return f"⇒ {data.get('action', '')}: {_clip(text, 160)}"
    if etype in ("task_done", "task_failed", "session_done", "note", "call"):
        return _clip(text, 200)
    return _clip(text, 200)


class EventSink:
    def __init__(self, session_id: str, log_dir: str | Path, echo: bool = True):
        self.session_id = session_id
        self.path = Path(log_dir).expanduser() / "events" / f"{session_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self.echo = echo

    def emit(self, source: str, etype: str, text: str = "", echo: bool | None = None, **data: Any) -> None:
        event: dict[str, Any] = {"ts": time.time(), "source": source, "type": etype}
        if text:
            event["text"] = text
        event.update(data)
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            try:
                self._fh.write(line + "\n")
                self._fh.flush()
            except (OSError, ValueError):
                pass
            if echo if echo is not None else self.echo:
                summary = summarize(source, etype, text, data)
                if summary:
                    print(f"  {SOURCE_LABEL.get(source, source)} › {summary}", flush=True)

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except (OSError, ValueError):
                pass


class Batcher:
    """Coalesces streamed text/reasoning deltas into readable chunks before emitting.

    Token-level deltas would flood the log; this flushes on newlines or when the buffer grows past a
    threshold, so the stream stays live but the events stay legible.
    """

    def __init__(self, sink: EventSink, source: str, etype: str, limit: int = 180):
        self.sink = sink
        self.source = source
        self.etype = etype
        self.limit = limit
        self._buf = ""

    def add(self, delta: str) -> None:
        if not delta:
            return
        self._buf += delta
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.sink.emit(self.source, self.etype, text=line)
        if len(self._buf) >= self.limit:
            self.sink.emit(self.source, self.etype, text=self._buf)
            self._buf = ""

    def flush(self) -> None:
        if self._buf.strip():
            self.sink.emit(self.source, self.etype, text=self._buf)
        self._buf = ""
