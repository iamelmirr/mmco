from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from mmco.db import Database
from mmco.models import Execution, ExecutionResult, TaskSpec
from mmco.web import ApiError, Dashboard, make_handler


@pytest.fixture
def dashboard(tmp_path, monkeypatch, isolated_config):
    monkeypatch.setenv("MMCO_DB_PATH", str(tmp_path / "state" / "mmco.db"))
    monkeypatch.setenv("MMCO_LOG_DIR", str(tmp_path / "state" / "logs"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    d = Dashboard()
    d.spawned = []
    monkeypatch.setattr(d, "_spawn", lambda sid, args: d.spawned.append((sid, args)))
    return d


@pytest.fixture
def server(dashboard):
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(dashboard))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def call(url, method="GET", body=None, token=None, host=None):
    request = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None)
    if token:
        request.add_header("X-MMCO-Token", token)
    if host:
        request.add_header("Host", host)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_page_is_served_with_token_and_api_requires_it(server, dashboard):
    status, html = call(server + "/")
    assert status == 200 and dashboard.token.encode() in html
    assert call(server + "/api/overview")[0] == 403
    status, body = call(server + "/api/overview", token=dashboard.token)
    assert status == 200 and "projects" in json.loads(body)


def test_foreign_host_header_is_rejected(server, dashboard):
    assert call(server + "/api/overview", token=dashboard.token, host="evil.example")[0] == 403


def test_errors_are_reported_as_json(server, dashboard):
    status, body = call(server + "/api/project?path=relative/path", token=dashboard.token)
    assert status == 400 and "absolute" in json.loads(body)["error"]


def test_projects_and_runs(dashboard, tmp_path):
    folder = tmp_path / "new-app"
    dashboard.add_project({"path": str(folder), "create": True})
    assert [p["name"] for p in dashboard.overview()["projects"]] == ["new-app"]
    started = dashboard.start_run({"path": str(folder), "request": "build a thing", "approve_plan": True})
    sid, args = dashboard.spawned[0]
    assert sid == started["session_id"]
    assert args[:3] == ["start", str(folder), "build a thing"] and "--approve-later" in args and "--no-input" in args
    with pytest.raises(ApiError, match="describe"):
        dashboard.start_run({"path": str(folder), "request": "  "})


def test_rules_roundtrip_and_context(dashboard, tmp_path):
    project = tmp_path / "app"
    saved = dashboard.put_rules({"scope": "project", "path": str(project), "executor_context": "minimal",
                                 "body": "Only technical tasks."})
    assert saved["executor_context"] == "minimal" and saved["body"] == "Only technical tasks."
    assert dashboard.project({"path": str(project)})["context"] == "minimal"
    cleared = dashboard.put_rules({"scope": "project", "path": str(project), "executor_context": None,
                                   "body": "Only technical tasks."})
    assert cleared["executor_context"] is None and cleared["inherited_context"] == "full"
    with pytest.raises(ApiError):
        dashboard.put_rules({"scope": "global", "executor_context": "everything", "body": ""})
    preview = dashboard.preview({"path": str(project)})["system_prompt"]
    assert "USER RULES" in preview and "Only technical tasks." in preview


def test_prompt_override_lifecycle(dashboard, tmp_path):
    project = str(tmp_path / "app")
    base = dashboard.get_prompt({"name": "planner_plan.txt", "scope": "project", "path": project})
    assert not base["overridden"] and base["inherited_from"] == "built-in"
    dashboard.put_prompt({"name": "planner_plan.txt", "scope": "global", "content": "GLOBAL"})
    inherited = dashboard.get_prompt({"name": "planner_plan.txt", "scope": "project", "path": project})
    assert inherited["content"] == "GLOBAL" and inherited["inherited_from"] == "global"
    dashboard.put_prompt({"name": "planner_plan.txt", "scope": "project", "path": project, "content": "MINE"})
    listing = {p["name"]: p["source"] for p in dashboard.list_prompts({"path": project})["prompts"]}
    assert listing["planner_plan.txt"] == "project" and listing["planner_eval.txt"] == "built-in"
    reset = dashboard.delete_prompt({"name": "planner_plan.txt", "scope": "project", "path": project})
    assert reset["content"] == "GLOBAL"
    with pytest.raises(ApiError):
        dashboard.get_prompt({"name": "../../etc/passwd"})
    with pytest.raises(ApiError, match="empty"):
        dashboard.put_prompt({"name": "planner_plan.txt", "scope": "global", "content": "  "})


def test_settings_update_masks_secret_and_rolls_back_invalid(dashboard, isolated_config):
    dashboard.put_settings({"values": {"OPENROUTER_API_KEY": "sk-or-v1-abcd1234", "MMCO_CLAUDE_MODEL": "sonnet",
                                       "MMCO_FINAL_REVIEW": False}})
    fields = {f["key"]: f for f in dashboard.get_settings()["fields"]}
    assert fields["OPENROUTER_API_KEY"]["value"] == "" and "1234" in fields["OPENROUTER_API_KEY"]["hint"]
    assert fields["MMCO_CLAUDE_MODEL"]["value"] == "sonnet" and fields["MMCO_FINAL_REVIEW"]["value"] is False
    dashboard.put_settings({"values": {"OPENROUTER_API_KEY": ""}})  # empty keeps the key
    assert "1234" in {f["key"]: f for f in dashboard.get_settings()["fields"]}["OPENROUTER_API_KEY"]["hint"]
    with pytest.raises(ApiError):
        dashboard.put_settings({"values": {"MMCO_MAX_ATTEMPTS_PER_TASK": "lots"}})
    assert "lots" not in (isolated_config / ".env").read_text()
    with pytest.raises(ApiError, match="cannot be changed"):
        dashboard.put_settings({"values": {"PATH": "/tmp"}})


def test_plan_approval_and_answers_resume_the_session(dashboard, tmp_path):
    settings = dashboard.settings()
    db = Database(settings.db_path)
    project = tmp_path / "app"
    project.mkdir()
    session = db.create_session("x", str(project), {"proposed_plan": [TaskSpec(title="a", description="do a").model_dump()]})
    db.close()

    with pytest.raises(Exception, match="at least one"):
        dashboard.plan_decision(session.id, {"action": "approve", "tasks": []})
    dashboard.plan_decision(session.id, {"action": "approve", "tasks": [{"title": "edited", "description": "do b"}]})
    assert dashboard.spawned[-1][1][:2] == ["resume", session.id]
    db = Database(settings.db_path)
    stored = db.find_session(session.id).metadata["plan_decision"]
    assert stored["tasks"][0]["title"] == "edited"
    question = db.add_questions(session.id, None, ["SQLite?"])[0]
    db.close()
    dashboard.answer(session.id, {"answers": {question.id: "yes"}})
    db = Database(settings.db_path)
    assert db.list_clarifications(session.id, answered=True)[0].answer == "yes"
    db.close()
    state = dashboard.session_state(session.id)
    assert state["session"]["decision_pending"] and state["questions"] == []


def test_planner_executed_task_surfaces_its_agent(dashboard, tmp_path):
    settings = dashboard.settings()
    db = Database(settings.db_path)
    project = tmp_path / "app"
    project.mkdir()
    session = db.create_session("x", str(project))
    task = db.insert_tasks(session.id, [TaskSpec(title="a", description="do a")])[0]
    task.executor = "planner"
    task.attempts = 1
    db.save_task(task)
    db.add_execution(Execution(
        task_id=task.id, session_id=session.id, attempt=1, agent="planner",
        result=ExecutionResult(prompt="do a"),
    ))
    db.close()

    state = dashboard.session_state(session.id)
    task_payload = next(t for t in state["tasks"] if t["id"] == task.id)
    assert task_payload["executor"] == "planner"

    dump = dashboard.session_detail(session.id)
    assert any(e["agent"] == "planner" for e in dump["executions"])


def test_session_state_for_a_run_that_has_not_created_its_session(dashboard):
    state = dashboard.session_state("0123abcd-0000-0000-0000-000000000000")
    assert state["session"] is None and state["running"] is False
