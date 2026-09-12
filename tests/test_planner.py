from __future__ import annotations

import json

import openai
import pytest

try:  # openai>=3 ships its HTTP client as httpx2
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from mmco.models import Execution, ExecutionResult, Session, Task, VerifyResult
from mmco.planner import Planner, PlannerError
from mmco.utils import extract_json

from .conftest import FakeOpenAI

PLAN = {
    "tasks": [
        {"order_index": 1, "title": "Routes", "description": "Add routes", "acceptance_criteria": "GET / works"},
        {"order_index": 0, "title": "Skeleton", "description": "Create app", "verify_commands": ["true"]},
    ],
    "questions": [],
}


@pytest.fixture
def session(db) -> Session:
    return db.create_session("build a todo app", "/tmp/project")


def make_planner(settings, db, responses):
    client = FakeOpenAI(responses)
    return Planner(settings, db, client=client, sleep=lambda _: None), client


def test_plan_parses_and_persists_call(settings, db, session):
    planner, client = make_planner(settings, db, [json.dumps(PLAN)])
    plan = planner.plan(session, "", [])
    assert [t.title for t in plan.tasks] == ["Routes", "Skeleton"]
    assert plan.tasks[0].acceptance_criteria == ["GET / works"]  # string coerced to list
    assert client.completions.calls[0]["response_format"] == {"type": "json_object"}
    calls = db.list_planner_calls(session.id)
    assert len(calls) == 1 and calls[0]["purpose"] == "plan" and calls[0]["cost_usd"] == 0.001


def test_plan_extracts_json_from_fenced_prose(settings, db, session):
    planner, _ = make_planner(settings, db, ["Here you go:\n```json\n" + json.dumps(PLAN) + "\n```\nEnjoy"])
    assert len(planner.plan(session, "", []).tasks) == 2


def test_invalid_json_is_fed_back_and_retried(settings, db, session):
    planner, client = make_planner(settings, db, ["not json at all", '{"tasks": [{"title": "no description"}]}',
                                                  json.dumps(PLAN)])
    plan = planner.plan(session, "", [])
    assert len(plan.tasks) == 2
    retry_messages = client.completions.calls[2]["messages"]
    assert retry_messages[-1]["role"] == "user" and "invalid" in retry_messages[-1]["content"]
    assert "description" in retry_messages[-1]["content"]  # validation error is shown to the model


def test_gives_up_after_max_retries(settings, db, session):
    planner, _ = make_planner(settings, db, ["nope", "still nope", "{bad"])
    with pytest.raises(PlannerError, match="invalid JSON"):
        planner.plan(session, "", [])


def test_transient_api_error_is_retried(settings, db, session):
    error = openai.APIConnectionError(request=httpx.Request("POST", "https://openrouter.ai"))
    planner, _ = make_planner(settings, db, [error, json.dumps(PLAN)])
    assert len(planner.plan(session, "", []).tasks) == 2
    calls = db.list_planner_calls(session.id)
    assert calls[0]["error"] and calls[1]["response_text"]


def test_auth_error_is_not_retried(settings, db, session):
    response = httpx.Response(401, request=httpx.Request("POST", "https://openrouter.ai"))
    error = openai.AuthenticationError("bad key", response=response, body=None)
    planner, client = make_planner(settings, db, [error, json.dumps(PLAN)])
    with pytest.raises(PlannerError, match="bad key"):
        planner.plan(session, "", [])
    assert len(client.completions.calls) == 1


def test_json_mode_can_be_disabled(settings, db, session):
    settings.planner_json_mode = False
    planner, client = make_planner(settings, db, [json.dumps(PLAN)])
    planner.plan(session, "", [])
    assert "response_format" not in client.completions.calls[0]


def test_evaluate_sends_evidence(settings, db, session):
    verdict = {"satisfied": False, "reason": "tests fail", "next_action": "RETRY", "feedback_for_executor": "fix it"}
    planner, client = make_planner(settings, db, [json.dumps(verdict)])
    task = Task(session_id=session.id, description="Create app", verify_commands=["pytest"])
    execution = Execution(
        task_id=task.id, session_id=session.id, attempt=1,
        result=ExecutionResult(prompt="p", result_text="All done!"),
        diff_stat="app.py | 3 +++", diff="+print('hi')",
        verify_results=[VerifyResult(command="pytest", exit_code=1, output="1 failed")],
    )
    result = planner.evaluate(session, task, execution, [], [])
    assert result.next_action == "retry" and result.feedback_for_executor == "fix it"
    payload = json.loads(client.completions.calls[0]["messages"][1]["content"])
    assert payload["verification"][0]["passed"] is False
    assert payload["changes"]["diff_stat"] == "app.py | 3 +++"


def test_reformulate_returns_plain_text(settings, db, session):
    planner, client = make_planner(settings, db, ["```markdown\n1. Do the thing\n2. Verify\n```"])
    task = Task(session_id=session.id, description="Create app")
    text = planner.reformulate(session, task, "old prompt", "misunderstood")
    assert text == "1. Do the thing\n2. Verify"
    assert "response_format" not in client.completions.calls[0]


def test_clarify_caps_questions(settings, db, session):
    planner, _ = make_planner(settings, db, [json.dumps({"questions": [f"q{i}" for i in range(8)]})])
    assert planner.clarify(session, "which db?", None, []) == ["q0", "q1", "q2", "q3", "q4"]


def test_missing_api_key_is_reported(settings, db):
    settings.openrouter_api_key = ""
    with pytest.raises(PlannerError, match="OPENROUTER_API_KEY"):
        Planner(settings, db)


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": [1, 2]}} trailing') == {"a": {"b": [1, 2]}}
    assert extract_json("```\n{\"a\": 2}\n```") == {"a": 2}
    with pytest.raises(ValueError):
        extract_json("no json here")
