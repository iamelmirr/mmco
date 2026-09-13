from mmco.db import Database
from mmco.models import Execution, ExecutionResult, TaskSpec


def test_execution_agent_roundtrip(tmp_path):
    db = Database(str(tmp_path / "m.db"))
    s = db.create_session("x", str(tmp_path))
    task = db.insert_tasks(s.id, [TaskSpec(description="d")])[0]
    ex = Execution(task_id=task.id, session_id=s.id, attempt=1, agent="planner",
                   result=ExecutionResult(prompt="p"))
    db.add_execution(ex)
    stored = db.list_executions(task_id=task.id)[0]
    assert stored.agent == "planner"


def test_execution_agent_defaults_to_claude(tmp_path):
    db = Database(str(tmp_path / "m1.db"))
    s = db.create_session("x", str(tmp_path))
    task = db.insert_tasks(s.id, [TaskSpec(description="d")])[0]
    ex = Execution(task_id=task.id, session_id=s.id, attempt=1, result=ExecutionResult(prompt="p"))
    db.add_execution(ex)
    assert db.list_executions(task_id=task.id)[0].agent == "claude"


def test_task_executor_roundtrip(tmp_path):
    db = Database(str(tmp_path / "m2.db"))
    s = db.create_session("x", str(tmp_path))
    task = db.insert_tasks(s.id, [TaskSpec(description="d")])[0]
    assert task.executor == "claude"  # default
    task.executor = "planner"
    db.save_task(task)
    assert db.get_task(task.id).executor == "planner"


def test_migration_version_advances(tmp_path):
    db = Database(str(tmp_path / "m3.db"))
    assert db.conn.execute("PRAGMA user_version").fetchone()[0] >= 3
