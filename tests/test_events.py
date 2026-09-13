from __future__ import annotations

import json

from mmco.events import Batcher, EventSink, active_sink, clear_active_sink, set_active_sink, summarize


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_sink_writes_jsonl_and_echoes(tmp_path, capsys):
    sink = EventSink("sess-1", tmp_path, echo=True)
    sink.emit("planner", "thinking", text="counting the r's")
    sink.emit("claude", "tool", name="Bash", args="pytest -q")
    sink.close()
    events = read_events(tmp_path / "events" / "sess-1.jsonl")
    assert events[0]["source"] == "planner" and events[0]["type"] == "thinking"
    assert events[1]["name"] == "Bash"
    out = capsys.readouterr().out
    assert "DeepSeek" in out and "Claude" in out and "Bash" in out


def test_emit_echo_false_is_silent(tmp_path, capsys):
    sink = EventSink("s", tmp_path, echo=True)
    sink.emit("orchestrator", "task_start", title="x", executor="planner", echo=False)
    assert capsys.readouterr().out == ""
    assert read_events(sink.path)[0]["executor"] == "planner"


def test_batcher_flushes_on_newline_and_threshold(tmp_path):
    sink = EventSink("s", tmp_path, echo=False)
    b = Batcher(sink, "planner", "thinking", limit=20)
    b.add("hello ")
    b.add("world\nsecond line ")  # newline flushes "hello world"
    b.add("x" * 30)  # threshold flushes
    b.flush()
    texts = [e["text"] for e in read_events(sink.path)]
    assert texts[0] == "hello world"
    assert any("second line" in t for t in texts)


def test_active_sink_global(tmp_path):
    assert active_sink() is None
    sink = EventSink("s", tmp_path, echo=False)
    set_active_sink(sink)
    assert active_sink() is sink
    clear_active_sink(sink)
    assert active_sink() is None


def test_summarize_shapes():
    assert summarize("claude", "tool", "", {"name": "Bash", "args": "ls"}).startswith("⚙ Bash(")
    assert summarize("claude", "tool_result", "3 files", {"is_error": False}).startswith("↳")
    assert summarize("claude", "tool_result", "boom", {"is_error": True}).startswith("✗")
    assert summarize("planner", "thinking", "   ", {}) == ""  # blank thinking is dropped
