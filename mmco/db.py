"""SQLite persistence with versioned migrations (PRAGMA user_version)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import (
    Clarification,
    EvalResult,
    Execution,
    ExecutionResult,
    Session,
    Task,
    TaskSpec,
    VerifyResult,
    new_id,
    utcnow,
)
from .utils import MMCOError

MIGRATIONS: list[str] = [
    """
    CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        original_request TEXT NOT NULL,
        project_dir TEXT NOT NULL,
        status TEXT NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}'
    );

    CREATE TABLE tasks (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        parent_task_id TEXT REFERENCES tasks(id),
        order_index INTEGER NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        description TEXT NOT NULL,
        acceptance_criteria TEXT NOT NULL DEFAULT '[]',
        files_involved TEXT NOT NULL DEFAULT '[]',
        interface_contracts TEXT NOT NULL DEFAULT '{}',
        verify_commands TEXT NOT NULL DEFAULT '[]',
        needs_continuity INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        cycle_attempts INTEGER NOT NULL DEFAULT 0,
        reformulations INTEGER NOT NULL DEFAULT 0,
        prompt_override TEXT,
        last_feedback TEXT,
        resume_claude_session_id TEXT,
        start_commit TEXT,
        end_commit TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX idx_tasks_session_order ON tasks(session_id, order_index);

    CREATE TABLE executions (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        attempt INTEGER NOT NULL,
        prompt_sent TEXT NOT NULL,
        claude_output TEXT,
        result_text TEXT NOT NULL DEFAULT '',
        stderr TEXT NOT NULL DEFAULT '',
        exit_code INTEGER,
        duration_ms INTEGER NOT NULL DEFAULT 0,
        timed_out INTEGER NOT NULL DEFAULT 0,
        is_error INTEGER NOT NULL DEFAULT 0,
        subtype TEXT,
        stop_reason TEXT,
        refusal_detected INTEGER NOT NULL DEFAULT 0,
        permission_denials TEXT NOT NULL DEFAULT '[]',
        claude_session_id TEXT,
        cost_usd REAL,
        num_turns INTEGER,
        diff_stat TEXT NOT NULL DEFAULT '',
        diff TEXT NOT NULL DEFAULT '',
        verify_results TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_executions_task ON executions(task_id, created_at);

    CREATE TABLE evaluations (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
        execution_id TEXT NOT NULL REFERENCES executions(id) ON DELETE CASCADE,
        satisfied INTEGER NOT NULL,
        next_action TEXT NOT NULL,
        reason TEXT NOT NULL DEFAULT '',
        feedback_for_executor TEXT NOT NULL DEFAULT '',
        new_tasks TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL
    );

    CREATE TABLE planner_calls (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        task_id TEXT,
        purpose TEXT NOT NULL,
        model TEXT NOT NULL DEFAULT '',
        request_json TEXT NOT NULL,
        response_text TEXT,
        error TEXT,
        duration_ms INTEGER NOT NULL DEFAULT 0,
        cost_usd REAL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_planner_calls_session ON planner_calls(session_id, created_at);

    CREATE TABLE clarifications (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        task_id TEXT,
        question TEXT NOT NULL,
        answer TEXT,
        created_at TEXT NOT NULL,
        answered_at TEXT
    );
    """,
    """
    ALTER TABLE clarifications ADD COLUMN kind TEXT NOT NULL DEFAULT 'question';
    ALTER TABLE executions ADD COLUMN regression_results TEXT NOT NULL DEFAULT '[]';
    CREATE INDEX idx_sessions_project ON sessions(project_dir, created_at);
    """,
]

_TASK_JSON_FIELDS = ("acceptance_criteria", "files_involved", "interface_contracts", "verify_commands")
_TASK_UPDATABLE = (
    "title",
    "description",
    "acceptance_criteria",
    "files_involved",
    "interface_contracts",
    "verify_commands",
    "needs_continuity",
    "status",
    "attempts",
    "cycle_attempts",
    "reformulations",
    "prompt_override",
    "last_feedback",
    "resume_claude_session_id",
    "start_commit",
    "end_commit",
)


class NotFoundError(MMCOError):
    pass


def _now() -> str:
    return utcnow().isoformat()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path) if str(path) == ":memory:" else str(Path(path).expanduser())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.migrate()

    def close(self) -> None:
        self.conn.close()

    def migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        for index, script in enumerate(MIGRATIONS[version:], start=version + 1):
            with self.conn:
                self.conn.executescript(script)
                self.conn.execute(f"PRAGMA user_version = {index}")

    # ---- sessions -------------------------------------------------------

    def create_session(
        self, request: str, project_dir: str, metadata: dict[str, Any] | None = None, session_id: str | None = None
    ) -> Session:
        session = Session(original_request=request, project_dir=project_dir, metadata=metadata or {})
        if session_id:
            session.id = session_id
        with self.conn:
            self.conn.execute(
                "INSERT INTO sessions (id, created_at, updated_at, original_request, project_dir, status, metadata)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session.id,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    session.original_request,
                    session.project_dir,
                    session.status,
                    _dumps(session.metadata),
                ),
            )
        return session

    def update_session(self, session: Session) -> None:
        session.updated_at = utcnow()
        with self.conn:
            self.conn.execute(
                "UPDATE sessions SET status = ?, metadata = ?, updated_at = ? WHERE id = ?",
                (session.status, _dumps(session.metadata), session.updated_at.isoformat(), session.id),
            )

    def find_session(self, id_or_prefix: str) -> Session:
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ? OR id LIKE ? ORDER BY created_at DESC",
            (id_or_prefix, f"{id_or_prefix}%"),
        ).fetchall()
        exact = [r for r in rows if r["id"] == id_or_prefix]
        if exact:
            rows = exact
        if not rows:
            raise NotFoundError(f"no session matches '{id_or_prefix}'")
        if len(rows) > 1:
            raise NotFoundError(f"'{id_or_prefix}' matches {len(rows)} sessions; use a longer prefix")
        return self._session(rows[0])

    def list_sessions(self, limit: int = 20) -> list[Session]:
        rows = self.conn.execute("SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._session(r) for r in rows]

    def list_projects(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT project_dir, MAX(updated_at) AS last_activity FROM sessions GROUP BY project_dir"
            " ORDER BY last_activity DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def list_sessions_for_project(self, project_dir: str, limit: int = 10) -> list[Session]:
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE project_dir = ? ORDER BY created_at DESC LIMIT ?", (project_dir, limit)
        ).fetchall()
        return [self._session(r) for r in rows]

    @staticmethod
    def _session(row: sqlite3.Row) -> Session:
        data = dict(row)
        data["metadata"] = json.loads(data["metadata"] or "{}")
        return Session.model_validate(data)

    # ---- tasks ----------------------------------------------------------

    def insert_tasks(
        self,
        session_id: str,
        specs: list[TaskSpec],
        position: int | None = None,
        parent_task_id: str | None = None,
    ) -> list[Task]:
        """Insert tasks at `position` (shifting later tasks) or append them."""
        tasks: list[Task] = []
        with self.conn:
            if position is None:
                position = self.conn.execute(
                    "SELECT COALESCE(MAX(order_index) + 1, 0) FROM tasks WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
            else:
                self.conn.execute(
                    "UPDATE tasks SET order_index = order_index + ? WHERE session_id = ? AND order_index >= ?",
                    (len(specs), session_id, position),
                )
            for offset, spec in enumerate(specs):
                task = Task(
                    **spec.model_dump(exclude={"order_index"}),
                    order_index=position + offset,
                    session_id=session_id,
                    parent_task_id=parent_task_id,
                )
                columns = ["id", "session_id", "parent_task_id", "order_index", "created_at", "updated_at", *_TASK_UPDATABLE]
                values = [self._task_value(task, c) for c in columns]
                self.conn.execute(
                    f"INSERT INTO tasks ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})", values
                )
                tasks.append(task)
        return tasks

    def save_task(self, task: Task) -> None:
        """Persist mutable task fields. order_index is only changed by insert_tasks."""
        task.updated_at = utcnow()
        assignments = ", ".join(f"{c} = ?" for c in (*_TASK_UPDATABLE, "updated_at"))
        values = [self._task_value(task, c) for c in (*_TASK_UPDATABLE, "updated_at")]
        with self.conn:
            self.conn.execute(f"UPDATE tasks SET {assignments} WHERE id = ?", (*values, task.id))

    def list_tasks(self, session_id: str) -> list[Task]:
        rows = self.conn.execute(
            "SELECT * FROM tasks WHERE session_id = ? ORDER BY order_index", (session_id,)
        ).fetchall()
        return [self._task(r) for r in rows]

    def get_task(self, task_id: str) -> Task:
        row = self.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"task {task_id} not found")
        return self._task(row)

    def find_task(self, session_id: str, id_or_prefix: str) -> Task:
        # An all-digits reference is a 1-based task number (as shown in the UI/CLI). Resolve it
        # before UUID-prefix matching, so a task whose UUID happens to start with that digit
        # cannot shadow the numbered task.
        if id_or_prefix.isdigit():
            numbered = self.conn.execute(
                "SELECT * FROM tasks WHERE session_id = ? AND order_index = ?", (session_id, int(id_or_prefix) - 1)
            ).fetchall()
            if numbered:
                return self._task(numbered[0])
        rows = self.conn.execute(
            "SELECT * FROM tasks WHERE session_id = ? AND (id = ? OR id LIKE ?)",
            (session_id, id_or_prefix, f"{id_or_prefix}%"),
        ).fetchall()
        if not rows:
            raise NotFoundError(f"no task matches '{id_or_prefix}' in session {session_id}")
        if len(rows) > 1:
            raise NotFoundError(f"'{id_or_prefix}' matches {len(rows)} tasks; use a longer prefix")
        return self._task(rows[0])

    def next_pending_task(self, session_id: str) -> Task | None:
        row = self.conn.execute(
            "SELECT * FROM tasks WHERE session_id = ? AND status = 'pending' ORDER BY order_index LIMIT 1",
            (session_id,),
        ).fetchone()
        return self._task(row) if row else None

    def block_pending_tasks(self, session_id: str) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE tasks SET status = 'blocked', updated_at = ? WHERE session_id = ? AND status = 'pending'",
                (_now(), session_id),
            )
        return cursor.rowcount

    def count_subtasks(self, task_id: str) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM tasks WHERE parent_task_id = ?", (task_id,)).fetchone()[0]

    @staticmethod
    def _task_value(task: Task, column: str) -> Any:
        value = getattr(task, column)
        if column in _TASK_JSON_FIELDS:
            return _dumps(value)
        if column == "needs_continuity":
            return int(value)
        if column in ("created_at", "updated_at"):
            return value.isoformat()
        return value

    @staticmethod
    def _task(row: sqlite3.Row) -> Task:
        data = dict(row)
        for field in _TASK_JSON_FIELDS:
            data[field] = json.loads(data[field])
        data["needs_continuity"] = bool(data["needs_continuity"])
        return Task.model_validate(data)

    # ---- executions -----------------------------------------------------

    def add_execution(self, execution: Execution) -> None:
        r = execution.result
        with self.conn:
            self.conn.execute(
                """INSERT INTO executions (id, task_id, session_id, attempt, prompt_sent, claude_output, result_text,
                   stderr, exit_code, duration_ms, timed_out, is_error, subtype, stop_reason, refusal_detected,
                   permission_denials, claude_session_id, cost_usd, num_turns, diff_stat, diff, verify_results,
                   regression_results, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    execution.id,
                    execution.task_id,
                    execution.session_id,
                    execution.attempt,
                    r.prompt,
                    _dumps(r.raw_output) if r.raw_output is not None else None,
                    r.result_text,
                    r.stderr,
                    r.exit_code,
                    r.duration_ms,
                    int(r.timed_out),
                    int(r.is_error),
                    r.subtype,
                    r.stop_reason,
                    int(execution.refusal_detected),
                    _dumps(r.permission_denials),
                    r.claude_session_id,
                    r.cost_usd,
                    r.num_turns,
                    execution.diff_stat,
                    execution.diff,
                    _dumps([v.model_dump() for v in execution.verify_results]),
                    _dumps([v.model_dump() for v in execution.regression_results]),
                    execution.created_at.isoformat(),
                ),
            )

    def update_execution_checks(self, execution: Execution) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE executions SET verify_results = ?, regression_results = ? WHERE id = ?",
                (
                    _dumps([v.model_dump() for v in execution.verify_results]),
                    _dumps([v.model_dump() for v in execution.regression_results]),
                    execution.id,
                ),
            )

    def has_evaluation(self, execution_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM evaluations WHERE execution_id = ? LIMIT 1", (execution_id,))
        return row.fetchone() is not None

    def list_executions(self, session_id: str | None = None, task_id: str | None = None) -> list[Execution]:
        if task_id:
            rows = self.conn.execute(
                "SELECT * FROM executions WHERE task_id = ? ORDER BY created_at", (task_id,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM executions WHERE session_id = ? ORDER BY created_at", (session_id,)
            ).fetchall()
        return [self._execution(r) for r in rows]

    def last_execution(self, task_id: str) -> Execution | None:
        row = self.conn.execute(
            "SELECT * FROM executions WHERE task_id = ? ORDER BY created_at DESC, attempt DESC LIMIT 1", (task_id,)
        ).fetchone()
        return self._execution(row) if row else None

    @staticmethod
    def _execution(row: sqlite3.Row) -> Execution:
        d = dict(row)
        result = ExecutionResult(
            prompt=d["prompt_sent"],
            raw_output=json.loads(d["claude_output"]) if d["claude_output"] else None,
            result_text=d["result_text"],
            stderr=d["stderr"],
            exit_code=d["exit_code"],
            duration_ms=d["duration_ms"],
            timed_out=bool(d["timed_out"]),
            is_error=bool(d["is_error"]),
            subtype=d["subtype"],
            stop_reason=d["stop_reason"],
            permission_denials=json.loads(d["permission_denials"]),
            claude_session_id=d["claude_session_id"],
            cost_usd=d["cost_usd"],
            num_turns=d["num_turns"],
        )
        return Execution(
            id=d["id"],
            task_id=d["task_id"],
            session_id=d["session_id"],
            attempt=d["attempt"],
            result=result,
            refusal_detected=bool(d["refusal_detected"]),
            diff_stat=d["diff_stat"],
            diff=d["diff"],
            verify_results=[VerifyResult.model_validate(v) for v in json.loads(d["verify_results"])],
            regression_results=[VerifyResult.model_validate(v) for v in json.loads(d["regression_results"])],
            created_at=d["created_at"],
        )

    # ---- evaluations ----------------------------------------------------

    def add_evaluation(self, execution: Execution, evaluation: EvalResult) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO evaluations (id, session_id, task_id, execution_id, satisfied, next_action, reason,
                   feedback_for_executor, new_tasks, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_id(),
                    execution.session_id,
                    execution.task_id,
                    execution.id,
                    int(evaluation.satisfied),
                    evaluation.next_action,
                    evaluation.reason,
                    evaluation.feedback_for_executor,
                    _dumps([t.model_dump() for t in evaluation.new_tasks]),
                    _now(),
                ),
            )

    def list_evaluations(self, task_id: str | None = None, session_id: str | None = None) -> list[dict[str, Any]]:
        column, value = ("task_id", task_id) if task_id else ("session_id", session_id)
        rows = self.conn.execute(
            f"SELECT e.*, x.attempt FROM evaluations e JOIN executions x ON x.id = e.execution_id"
            f" WHERE e.{column} = ? ORDER BY e.created_at",
            (value,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["satisfied"] = bool(d["satisfied"])
            d["new_tasks"] = json.loads(d["new_tasks"])
            result.append(d)
        return result

    # ---- planner calls --------------------------------------------------

    def add_planner_call(
        self,
        session_id: str,
        purpose: str,
        model: str,
        request: Any,
        response_text: str | None,
        duration_ms: int,
        task_id: str | None = None,
        error: str | None = None,
        cost_usd: float | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO planner_calls (id, session_id, task_id, purpose, model, request_json, response_text,
                   error, duration_ms, cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (new_id(), session_id, task_id, purpose, model, _dumps(request), response_text, error,
                 duration_ms, cost_usd, _now()),
            )

    def list_planner_calls(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM planner_calls WHERE session_id = ? ORDER BY created_at", (session_id,)
        ).fetchall()
        return [dict(r) | {"request_json": json.loads(r["request_json"])} for r in rows]

    # ---- clarifications -------------------------------------------------

    def add_questions(
        self, session_id: str, task_id: str | None, questions: list[str], kind: str = "question"
    ) -> list[Clarification]:
        items = [Clarification(session_id=session_id, task_id=task_id, question=q, kind=kind)
                 for q in questions if q.strip()]
        with self.conn:
            for c in items:
                self.conn.execute(
                    "INSERT INTO clarifications (id, session_id, task_id, kind, question, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (c.id, c.session_id, c.task_id, c.kind, c.question, c.created_at.isoformat()),
                )
        return items

    def answer_clarification(self, clarification_id: str, answer: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE clarifications SET answer = ?, answered_at = ? WHERE id = ?",
                (answer, _now(), clarification_id),
            )

    def list_clarifications(
        self, session_id: str, answered: bool | None = None, kind: str | None = None
    ) -> list[Clarification]:
        query, params = "SELECT * FROM clarifications WHERE session_id = ?", [session_id]
        if answered is True:
            query += " AND answer IS NOT NULL"
        elif answered is False:
            query += " AND answer IS NULL"
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        rows = self.conn.execute(query + " ORDER BY created_at", params).fetchall()
        return [Clarification.model_validate(dict(r)) for r in rows]

    # ---- reporting ------------------------------------------------------

    def summary(self, session_id: str) -> dict[str, Any]:
        tasks = self.list_tasks(session_id)
        counts = {status: sum(1 for t in tasks if t.status == status)
                  for status in ("done", "failed", "blocked", "skipped", "pending", "in_progress")}
        executions, exec_ms, exec_cost = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(duration_ms), 0), COALESCE(SUM(cost_usd), 0)"
            " FROM executions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        planner_calls, planner_ms, planner_cost = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(duration_ms), 0), COALESCE(SUM(cost_usd), 0)"
            " FROM planner_calls WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return {
            "tasks": len(tasks),
            **counts,
            "executions": executions,
            "reformulations": sum(t.reformulations for t in tasks),
            "executor_time_ms": exec_ms,
            "executor_cost_usd": round(exec_cost, 4),
            "planner_calls": planner_calls,
            "planner_time_ms": planner_ms,
            "planner_cost_usd": round(planner_cost, 4),
        }

    def session_cost(self, session_id: str) -> float:
        s = self.summary(session_id)
        return s["executor_cost_usd"] + s["planner_cost_usd"]

    def task_stats(self, task_id: str) -> dict[str, Any]:
        duration, cost = self.conn.execute(
            "SELECT COALESCE(SUM(duration_ms), 0), COALESCE(SUM(cost_usd), 0) FROM executions WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        return {"duration_ms": duration, "cost_usd": cost}

    def dump_session(self, session_id: str) -> dict[str, Any]:
        session = self.find_session(session_id)
        return {
            "session": session.model_dump(mode="json"),
            "tasks": [t.model_dump(mode="json") for t in self.list_tasks(session.id)],
            "executions": [e.model_dump(mode="json") for e in self.list_executions(session_id=session.id)],
            "evaluations": self.list_evaluations(session_id=session.id),
            "planner_calls": self.list_planner_calls(session.id),
            "clarifications": [c.model_dump(mode="json") for c in self.list_clarifications(session.id)],
            "summary": self.summary(session.id),
        }
