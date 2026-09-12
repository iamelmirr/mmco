import json
from types import SimpleNamespace

import openai

try:  # openai>=3 ships its HTTP client as httpx2
    import httpx2 as httpx
except ImportError:
    import httpx

from mmco.planner import Planner
from mmco.models import Session


def tool_call(cid, name, args):
    return SimpleNamespace(
        id=cid,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class ScriptedClient:
    """Returns tool calls, then a final text answer."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.sent = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        item = self.turns.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, list):
            msg = SimpleNamespace(content=None, tool_calls=item)
        else:
            msg = SimpleNamespace(content=item, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(cost=0.001))


def test_planner_runs_tools_then_returns_json(settings, db, tmp_path):
    (tmp_path / "app.py").write_text("def hello(): return 'hi'\n")
    session = db.create_session("x", str(tmp_path))
    client = ScriptedClient(
        [
            [tool_call("c1", "read_file", {"path": "app.py"})],
            json.dumps({"tasks": [{"description": "task using hello()"}]}),
        ]
    )
    planner = Planner(settings, db, client=client, sleep=lambda _: None)
    plan = planner.plan(session, "", [], read_tools=True)
    assert plan.tasks[0].description
    second = client.sent[1]["messages"]
    assert any(m.get("role") == "tool" and "hello" in str(m.get("content")) for m in second)


def test_tool_loop_has_a_cap(settings, db, tmp_path):
    settings.planner_max_tool_calls = 2
    session = db.create_session("x", str(tmp_path))
    (tmp_path / "a.py").write_text("x=1")
    always = [tool_call("c", "read_file", {"path": "a.py"})]
    client = ScriptedClient([always, always, always, json.dumps({"tasks": []})])
    planner = Planner(settings, db, client=client, sleep=lambda _: None)
    plan = planner.plan(session, "", [], read_tools=True)
    assert plan is not None


def test_transient_error_in_tool_loop_is_retried(settings, db, tmp_path):
    (tmp_path / "app.py").write_text("def hello(): return 'hi'\n")
    session = db.create_session("x", str(tmp_path))
    error = openai.APIConnectionError(request=httpx.Request("POST", "https://openrouter.ai"))
    client = ScriptedClient(
        [
            error,
            [tool_call("c1", "read_file", {"path": "app.py"})],
            json.dumps({"tasks": [{"description": "task using hello()"}]}),
        ]
    )
    planner = Planner(settings, db, client=client, sleep=lambda _: None)
    plan = planner.plan(session, "", [], read_tools=True)
    assert plan.tasks[0].description
    # Three completions happened: the transient failure, the tool call, and the final answer.
    assert len(client.sent) == 3
    calls = db.list_planner_calls(session.id)
    assert calls[0]["error"] and calls[-1]["response_text"]


def test_read_tools_pass_read_only_schemas(settings, db, tmp_path):
    (tmp_path / "app.py").write_text("x=1\n")
    session = db.create_session("x", str(tmp_path))
    client = ScriptedClient(
        [
            [tool_call("c1", "list_dir", {"path": "."})],
            json.dumps({"tasks": [{"description": "do a thing"}]}),
        ]
    )
    planner = Planner(settings, db, client=client, sleep=lambda _: None)
    planner.plan(session, "", [], read_tools=True)
    first = client.sent[0]
    tool_names = {t["function"]["name"] for t in first["tools"]}
    assert tool_names <= {"read_file", "search", "list_dir"}
    assert "write_file" not in tool_names and "edit_file" not in tool_names
    # JSON mode must not be set while tools may still be called.
    assert "response_format" not in first
