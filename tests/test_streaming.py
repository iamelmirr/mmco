from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from mmco.events import EventSink, clear_active_sink, set_active_sink
from mmco.utils import ProcessResult, run_process_streaming


@pytest.fixture
def sink(tmp_path):
    s = EventSink("stream-test", tmp_path, echo=False)
    set_active_sink(s)
    yield s
    clear_active_sink(s)
    s.close()


def events_of(sink, source=None):
    rows = [json.loads(line) for line in sink.path.read_text().splitlines() if line.strip()]
    return [r for r in rows if source is None or r["source"] == source]


# ---- run_process_streaming --------------------------------------------------


def test_streaming_runner_calls_on_line_and_captures_stdout(tmp_path):
    seen = []
    result = run_process_streaming(
        [sys.executable, "-c", "print('a'); print('b'); print('c')"], tmp_path, None, timeout=10,
        on_line=seen.append,
    )
    assert [s.strip() for s in seen] == ["a", "b", "c"]
    assert result.stdout.splitlines() == ["a", "b", "c"] and result.returncode == 0


def test_streaming_runner_kills_on_timeout(tmp_path):
    result = run_process_streaming(
        [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, None, timeout=0.5, on_line=lambda _l: None,
    )
    assert result.timed_out and result.duration_ms < 15000


# ---- executor streaming -----------------------------------------------------

CLAUDE_STREAM = "\n".join([
    json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}),
    json.dumps({"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "let me look"}]}}),
    json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}}]}}),
    json.dumps({"type": "user", "message": {"content": [
        {"type": "tool_result", "content": "3 passed", "is_error": False}]}}),
    json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "All tests pass."}]}}),
    json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "All tests pass.",
                "session_id": "s1", "total_cost_usd": 0.12, "duration_ms": 4200, "num_turns": 3}),
]) + "\n"


def test_executor_streams_events_and_parses_result(settings, tmp_path, sink, monkeypatch):
    from mmco import executor as executor_mod

    def fake_stream(cmd, cwd, stdin, timeout, on_line):
        assert cmd[cmd.index("--output-format") + 1] == "stream-json"  # streaming format chosen
        for line in CLAUDE_STREAM.splitlines():
            on_line(line + "\n")
        return ProcessResult(stdout=CLAUDE_STREAM, stderr="", returncode=0, timed_out=False, duration_ms=4200)

    monkeypatch.setattr(executor_mod, "run_process_streaming", fake_stream)
    result = executor_mod.Executor(settings).run_task("do it", tmp_path)  # default runner => streaming path

    assert not result.is_error and result.result_text == "All tests pass."
    assert result.claude_session_id == "s1" and result.cost_usd == 0.12
    kinds = [(e["type"], e.get("name") or e.get("text", "")[:12]) for e in events_of(sink, "claude")]
    assert ("thinking", "let me look") in kinds
    assert ("tool", "Bash") in kinds
    assert any(t == "tool_result" for t, _ in kinds)
    assert any(t == "text" for t, _ in kinds)


def test_executor_uses_plain_json_with_injected_runner(settings, tmp_path):
    # A test-injected runner (not the default run_process) always takes the non-streaming json path,
    # even when a sink is active — so unit tests never spawn a streaming subprocess.
    from mmco import executor as executor_mod

    calls = {}

    def fake_run(cmd, cwd, stdin, timeout):
        calls["fmt"] = cmd[cmd.index("--output-format") + 1]
        return ProcessResult(
            stdout=json.dumps({"type": "result", "subtype": "success", "result": "ok", "session_id": "x"}),
            stderr="", returncode=0, timed_out=False, duration_ms=1,
        )

    result = executor_mod.Executor(settings, runner=fake_run).run_task("x", tmp_path)
    assert result.result_text == "ok" and calls["fmt"] == "json"


def test_emit_claude_event_parses_blocks(tmp_path):
    from mmco.executor import _emit_claude_event

    s = EventSink("e", tmp_path, echo=False)
    _emit_claude_event(s, json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Edit", "input": {"file_path": "app.py"}}]}}))
    _emit_claude_event(s, "not json")  # ignored
    ev = [json.loads(x) for x in s.path.read_text().splitlines() if x.strip()]
    assert ev[0]["type"] == "tool" and ev[0]["name"] == "Edit" and "app.py" in ev[0]["args"]


# ---- planner streaming ------------------------------------------------------


def _delta(content=None, reasoning=None, tool_calls=None):
    d = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning=reasoning)
    d.model_extra = {"reasoning": reasoning} if reasoning else {}
    return SimpleNamespace(choices=[SimpleNamespace(delta=d)], usage=None)


def _tc(index, name=None, args=None, cid=None):
    return SimpleNamespace(index=index, id=cid,
                           function=SimpleNamespace(name=name, arguments=args))


class StreamingClient:
    def __init__(self, chunks):
        self._chunks = chunks
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.streamed = False

    def _create(self, **kwargs):
        assert kwargs.get("stream") is True
        self.streamed = True
        return iter(self._chunks)


def test_planner_invoke_streams_text_reasoning_and_tools(settings, db, tmp_path, sink):
    from mmco.planner import Planner

    chunks = [
        _delta(reasoning="I should read the file. "),
        _delta(reasoning="Then answer.\n"),
        _delta(content='{"ok": '),
        _delta(content="true}"),
        _delta(tool_calls=[_tc(0, name="read_file", cid="c1")]),
        _delta(tool_calls=[_tc(0, args='{"path": "app.py"}')]),
        SimpleNamespace(choices=[], usage=SimpleNamespace(cost=0.002)),
    ]
    client = StreamingClient(chunks)
    planner = Planner(settings, db, client=client, sleep=lambda _: None)
    message, cost = planner._invoke({"model": "m", "messages": []}, "plan")

    assert client.streamed and message.content == '{"ok": true}'
    assert message.tool_calls[0].function.name == "read_file"
    assert message.tool_calls[0].function.arguments == '{"path": "app.py"}'
    planner_events = events_of(sink, "planner")
    assert any(e["type"] == "thinking" for e in planner_events)
    assert any(e["type"] == "tool" and e["name"] == "read_file" for e in planner_events)


def test_planner_invoke_blocking_without_sink(settings, db):
    from mmco.planner import Planner
    from tests.conftest import FakeOpenAI

    clear_active_sink()
    client = FakeOpenAI(["hello"])
    planner = Planner(settings, db, client=client, sleep=lambda _: None)
    message, cost = planner._invoke({"model": "m", "messages": []}, "plan")
    assert message.content == "hello" and cost == 0.001
    assert "stream" not in client.completions.calls[0]  # plain call, no streaming
