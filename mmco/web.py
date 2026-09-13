"""Local dashboard (`mmco ui`): a JSON API over the database, rules and prompts, plus one HTML page.

Work runs in separate `mmco` worker processes, so closing the dashboard never stops a session.
Decisions (plan approval, answers) are written to the database and a worker is started to continue.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import traceback
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from .config import Settings, config_dir, global_env_file, load_settings, read_env_file, write_env_file
from .db import Database
from .executor import Executor
from .models import new_id
from .orchestrator import Orchestrator
from .planner import compose_system_prompt
from .rules import (
    EXECUTOR_CONTEXTS,
    RulesError,
    global_rules_path,
    load_rules,
    parse_rules,
    project_config_dir,
    project_rules_path,
)
from .utils import PROMPTS_DIR, MMCOError
from .workspace import _pid_alive

UI_FILE = Path(__file__).parent / "ui" / "index.html"
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
RUNNING_STATUSES = {"planning", "executing"}

PROMPT_INFO: dict[str, tuple[str, str]] = {
    "planner_plan.txt": ("Planning", "Turns your request into a list of tasks."),
    "planner_eval.txt": ("Reviewing each attempt", "Judges Claude's work from the diff and test results."),
    "planner_reformulate.txt": ("Rewriting a failed prompt", "Writes a new prompt when an attempt went wrong."),
    "planner_clarify.txt": ("Asking you questions", "Phrases questions when only you can decide."),
    "planner_review.txt": ("Final review", "Compares the finished project with your request."),
    "executor_system.txt": ("System prompt", "Standing instructions for Claude (full and task context)."),
    "executor_system_minimal.txt": ("System prompt, minimal", "Standing instructions when context is minimal."),
    "executor_task.txt": ("Task prompt, full context", "Goal, plan, your answers and the task."),
    "executor_task_isolated.txt": ("Task prompt, task only", "Only the task, its checks and feedback."),
    "executor_task_minimal.txt": ("Task prompt, minimal", "Only the planner's instructions and checks."),
    "executor_retry.txt": ("Retry with feedback", "Sent when Claude continues after a failed review."),
}
TEMPLATE_VARIABLES = {
    "executor_task.txt": ["$goal", "$plan_overview", "$task_block", "$verify_commands", "$clarifications", "$feedback"],
    "executor_task_isolated.txt": ["$task_block", "$verify_commands", "$feedback"],
    "executor_task_minimal.txt": ["$task_block", "$verify_commands", "$feedback"],
    "executor_retry.txt": ["$feedback", "$verify_report"],
}
SETTINGS_FIELDS: list[dict[str, Any]] = [
    {"key": "OPENROUTER_API_KEY", "field": "openrouter_api_key", "kind": "secret", "group": "Models",
     "label": "OpenRouter API key"},
    {"key": "MMCO_PLANNER_MODEL", "field": "planner_model", "kind": "text", "group": "Models",
     "label": "Planner model", "help": "Any OpenRouter model id."},
    {"key": "MMCO_CLAUDE_MODEL", "field": "claude_model", "kind": "text", "group": "Models",
     "label": "Claude model", "help": "Empty uses your Claude Code default. 'sonnet' is cheaper."},
    {"key": "MMCO_EXECUTOR_CONTEXT", "field": "executor_context", "kind": "choice", "choices": list(EXECUTOR_CONTEXTS),
     "group": "Models", "label": "Default context for Claude", "help": "Rules can override this per project."},
    {"key": "MMCO_CLAUDE_MAX_BUDGET_USD_PER_TASK", "field": "claude_max_budget_usd_per_task", "kind": "number",
     "group": "Limits", "label": "Claude budget per task (USD)", "help": "Empty means no limit."},
    {"key": "MMCO_MAX_SESSION_COST_USD", "field": "max_session_cost_usd", "kind": "number", "group": "Limits",
     "label": "Pause a session above (USD)", "help": "Empty means no limit."},
    {"key": "MMCO_MAX_ATTEMPTS_PER_TASK", "field": "max_attempts_per_task", "kind": "int", "group": "Limits",
     "label": "Attempts per prompt"},
    {"key": "MMCO_MAX_REFORMULATIONS_PER_TASK", "field": "max_reformulations_per_task", "kind": "int",
     "group": "Limits", "label": "Prompt rewrites per task"},
    {"key": "MMCO_CLAUDE_TIMEOUT_SECONDS", "field": "claude_timeout_seconds", "kind": "int", "group": "Limits",
     "label": "Claude timeout per attempt (seconds)"},
    {"key": "MMCO_ESCALATE_TO_USER", "field": "escalate_to_user", "kind": "bool", "group": "Behaviour",
     "label": "Ask me when a task is stuck", "help": "Off marks stuck tasks as failed."},
    {"key": "MMCO_FINAL_REVIEW", "field": "final_review", "kind": "bool", "group": "Behaviour",
     "label": "Final review against my request"},
    {"key": "MMCO_REGRESSION_CHECKS", "field": "regression_checks", "kind": "bool", "group": "Behaviour",
     "label": "Rerun earlier checks after every task"},
    {"key": "MMCO_CLAUDE_ALLOWED_TOOLS", "field": "claude_allowed_tools", "kind": "text", "group": "Behaviour",
     "label": "Tools Claude may use"},
    {"key": "MMCO_CLAUDE_PERMISSION_MODE", "field": "claude_permission_mode", "kind": "choice",
     "choices": ["bypassPermissions", "acceptEdits", "plan", "default"], "group": "Behaviour",
     "label": "Claude permission mode",
     "help": "bypassPermissions lets Claude run anything without asking. Narrow it for untrusted work."},
]


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _require(value: Any, message: str) -> Any:
    if value in (None, ""):
        raise ApiError(400, message)
    return value


def _project_path(raw: str | None) -> Path:
    path = Path(_require(raw, "choose a project folder")).expanduser()
    if not path.is_absolute():
        raise ApiError(400, "use an absolute folder path")
    return path.resolve()


class Dashboard:
    def __init__(self) -> None:
        self.token = secrets.token_urlsafe(24)
        self.processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    # ---- plumbing -------------------------------------------------------

    @staticmethod
    def settings() -> Settings:
        return load_settings()

    def db(self) -> Database:
        return Database(self.settings().db_path)

    def _state_file(self) -> Path:
        return Path(self.settings().db_path).expanduser().parent / "ui.json"

    def _registered_projects(self) -> list[str]:
        try:
            return json.loads(self._state_file().read_text(encoding="utf-8")).get("projects", [])
        except (OSError, ValueError):
            return []

    def _save_projects(self, projects: list[str]) -> None:
        self._state_file().parent.mkdir(parents=True, exist_ok=True)
        self._state_file().write_text(json.dumps({"projects": projects}, indent=2), encoding="utf-8")

    def _runs_dir(self) -> Path:
        path = Path(self.settings().db_path).expanduser().parent / "runs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _reap(self) -> None:
        with self._lock:
            for session_id, proc in list(self.processes.items()):
                if proc.poll() is not None:
                    del self.processes[session_id]

    def _worker_pid(self, project_dir: str | Path, session_id: str | None = None) -> int | None:
        self._reap()
        if session_id and session_id in self.processes:
            return self.processes[session_id].pid
        try:
            pid = int((Path(project_dir) / ".git" / "mmco.lock").read_text().strip())
        except (OSError, ValueError):
            return None
        return pid if _pid_alive(pid) else None

    def _spawn(self, session_id: str, args: list[str]) -> None:
        with self._lock:
            existing = self.processes.get(session_id)
            if existing and existing.poll() is None:
                raise ApiError(409, "This session is already running.")
            cwd = config_dir()
            cwd.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "COLUMNS": "120", "NO_COLOR": "1", "TERM": "dumb", "PYTHONUNBUFFERED": "1"}
            with open(self._runs_dir() / f"{session_id}.log", "a", encoding="utf-8") as log:
                self.processes[session_id] = subprocess.Popen(
                    [sys.executable, "-m", "mmco.main", *args],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    cwd=str(cwd), env=env, start_new_session=True,
                )

    def _log_tail(self, session_id: str, lines: int = 250) -> str:
        path = self._runs_dir() / f"{session_id}.log"
        if not path.exists():
            return ""
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 120_000))
            text = fh.read().decode("utf-8", errors="replace")
        return ANSI.sub("", "\n".join(text.splitlines()[-lines:]))

    def _orchestrator(self, settings: Settings, db: Database) -> Orchestrator:
        return Orchestrator(settings, db, planner=None, executor=None)

    # ---- overview & projects --------------------------------------------

    def overview(self, **_: Any) -> dict[str, Any]:
        settings = self.settings()
        db = self.db()
        try:
            seen: dict[str, dict[str, Any]] = {}
            for row in db.list_projects():
                seen[row["project_dir"]] = {"path": row["project_dir"]}
            for path in self._registered_projects():
                seen.setdefault(path, {"path": path})
            projects = []
            for path, item in seen.items():
                latest = db.list_sessions_for_project(path, limit=1)
                session = latest[0] if latest else None
                running = bool(self._worker_pid(path, session.id if session else None))
                projects.append({
                    "path": path,
                    "name": Path(path).name or path,
                    "exists": Path(path).is_dir(),
                    "running": running,
                    "status": session.status if session else None,
                    "updated_at": session.updated_at.isoformat() if session else None,
                })
        finally:
            db.close()
        try:
            claude = Executor(settings).resolve_binary()
        except MMCOError:
            claude = None
        return {
            "projects": projects,
            "setup": {
                "key_set": bool(settings.openrouter_api_key),
                "claude": claude,
                "git": bool(shutil.which("git")),
                "planner_model": settings.planner_model,
                "claude_model": settings.claude_model,
            },
        }

    def add_project(self, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        path = _project_path(body.get("path"))
        if body.get("create"):
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise ApiError(400, f"{path} is not a folder")
        projects = self._registered_projects()
        if str(path) not in projects:
            projects.insert(0, str(path))
            self._save_projects(projects)
        return {"path": str(path)}

    def remove_project(self, query: dict[str, str], **_: Any) -> dict[str, Any]:
        path = str(_project_path(query.get("path")))
        self._save_projects([p for p in self._registered_projects() if p != path])
        return {"removed": path}

    def project(self, query: dict[str, str], **_: Any) -> dict[str, Any]:
        path = _project_path(query.get("path"))
        settings = self.settings()
        db = self.db()
        try:
            sessions = []
            for session in db.list_sessions_for_project(str(path), limit=50):
                summary = db.summary(session.id)
                sessions.append({
                    "id": session.id,
                    "request": session.original_request,
                    "status": session.status,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "tasks": summary["tasks"],
                    "done": summary["done"],
                    "cost": summary["executor_cost_usd"] + summary["planner_cost_usd"],
                    "running": bool(self._worker_pid(path, session.id)) and session.status in RUNNING_STATUSES,
                })
        finally:
            db.close()
        try:
            rules = load_rules(path)
            context = rules.executor_context or settings.executor_context
            project_set = bool(project_rules_path(path).exists() and parse_rules(
                project_rules_path(path).read_text(encoding="utf-8"))[0].get("executor_context"))
        except RulesError:
            context, project_set = settings.executor_context, False
        running = self._worker_pid(path)
        return {
            "path": str(path),
            "name": path.name,
            "exists": path.is_dir(),
            "sessions": sessions,
            "context": context,
            "context_from_project": project_set,
            "busy": bool(running),
        }

    def browse(self, query: dict[str, str], **_: Any) -> dict[str, Any]:
        raw = query.get("path") or str(Path.home())
        path = Path(raw).expanduser().resolve()
        if not path.is_dir():
            raise ApiError(400, f"{path} is not a folder")
        try:
            dirs = sorted((p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")),
                          key=str.lower)
        except PermissionError:
            dirs = []
        return {"path": str(path), "parent": str(path.parent) if path.parent != path else None, "dirs": dirs}

    def open_folder(self, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        path = _project_path(body.get("path"))
        code = shutil.which("code")
        if code:
            subprocess.Popen([code, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-a", "Visual Studio Code", str(path)])
        else:
            raise ApiError(400, "VS Code's `code` command is not on PATH")
        return {"opened": str(path)}

    # ---- sessions -------------------------------------------------------

    def start_run(self, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        path = _project_path(body.get("path"))
        request = _require((body.get("request") or "").strip(), "describe what should be built")
        if self._worker_pid(path):
            raise ApiError(409, "Something is already running in this project. Pause it or wait for it to finish.")
        path.mkdir(parents=True, exist_ok=True)
        self.add_project({"path": str(path)})
        session_id = new_id()
        args = ["start", str(path), request, "--no-input", "--allow-dirty", "--session-id", session_id]
        if body.get("approve_plan", True):
            args.append("--approve-later")
        self._spawn(session_id, args)
        return {"session_id": session_id}

    def session_state(self, sid: str, **_: Any) -> dict[str, Any]:
        db = self.db()
        try:
            try:
                session = db.find_session(sid)
            except MMCOError:
                running = sid in self.processes and self.processes[sid].poll() is None
                return {"session": None, "running": running, "log": self._log_tail(sid)}
            tasks = []
            for task in db.list_tasks(session.id):
                stats = db.task_stats(task.id)
                executions = db.list_executions(task_id=task.id)
                evaluations = db.list_evaluations(task_id=task.id)
                stage = None
                if task.status == "in_progress":
                    last = executions[-1] if executions else None
                    stage = "review" if last and last.attempt == task.attempts and not db.has_evaluation(last.id) \
                        else "build"
                tasks.append({
                    "id": task.id,
                    "title": task.label,
                    "description": task.description,
                    "status": task.status,
                    "stage": stage,
                    "executor": task.executor,
                    "attempts": task.attempts,
                    "reformulations": task.reformulations,
                    "verify_commands": task.verify_commands,
                    "duration_ms": stats["duration_ms"],
                    "cost": stats["cost_usd"],
                    "reformulated": bool(task.prompt_override),
                    "last_verdict": evaluations[-1] if evaluations else None,
                    "checks": [
                        {"command": v.command, "passed": v.passed}
                        for v in (executions[-1].verify_results if executions else [])
                    ],
                })
            pending = [c.model_dump(mode="json") for c in db.list_clarifications(session.id, answered=False)]
            summary = db.summary(session.id)
            running = bool(self._worker_pid(session.project_dir, session.id))
            return {
                "session": {
                    "id": session.id,
                    "request": session.original_request,
                    "project_dir": session.project_dir,
                    "status": session.status,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "note": session.metadata.get("last_error"),
                    "proposed_plan": session.metadata.get("proposed_plan"),
                    "decision_pending": bool(session.metadata.get("plan_decision")),
                },
                "tasks": tasks,
                "questions": pending,
                "summary": summary,
                "running": running,
                "log": self._log_tail(session.id),
            }
        finally:
            db.close()

    def session_detail(self, sid: str, **_: Any) -> dict[str, Any]:
        db = self.db()
        try:
            return db.dump_session(sid)
        finally:
            db.close()

    def _stopped_session(self, sid: str) -> tuple[Settings, Database, Any]:
        settings = self.settings()
        db = self.db()
        session = db.find_session(sid)
        if self._worker_pid(session.project_dir, session.id):
            db.close()
            raise ApiError(409, "Wait a moment: the session is still saving its state.")
        return settings, db, session

    def answer(self, sid: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        settings, db, session = self._stopped_session(sid)
        try:
            answers = body.get("answers") or {}
            if not answers:
                raise ApiError(400, "no answers given")
            self._orchestrator(settings, db).record_answers(session.id, {k: str(v) for k, v in answers.items()})
        finally:
            db.close()
        self._spawn(session.id, ["resume", session.id, "--no-input", "--allow-dirty"])
        return {"resumed": True}

    def plan_decision(self, sid: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        settings, db, session = self._stopped_session(sid)
        try:
            try:
                self._orchestrator(settings, db).set_plan_decision(
                    session.id, body.get("action", ""), body.get("feedback") or "", body.get("tasks"))
            except ValidationError as exc:
                raise ApiError(400, f"a task is missing required fields: {exc.errors()[0]['loc']}") from exc
        finally:
            db.close()
        self._spawn(session.id, ["resume", session.id, "--no-input", "--allow-dirty"])
        return {"resumed": True}

    def pause(self, sid: str, **_: Any) -> dict[str, Any]:
        db = self.db()
        try:
            session = db.find_session(sid)
        finally:
            db.close()
        pid = self._worker_pid(session.project_dir, session.id)
        if not pid:
            raise ApiError(409, "Nothing is running for this session.")
        os.kill(pid, signal.SIGINT)
        return {"paused": True}

    def resume(self, sid: str, **_: Any) -> dict[str, Any]:
        _, db, session = self._stopped_session(sid)
        db.close()
        if session.status in ("done", "cancelled"):
            raise ApiError(400, f"This session is {session.status}.")
        self._spawn(session.id, ["resume", session.id, "--no-input", "--allow-dirty"])
        return {"resumed": True}

    # ---- rules ----------------------------------------------------------

    @staticmethod
    def _rules_file(scope: str, path: str | None) -> Path:
        if scope == "global":
            return global_rules_path()
        if scope == "project":
            return project_rules_path(_project_path(path))
        raise ApiError(400, "scope must be global or project")

    def get_rules(self, query: dict[str, str], **_: Any) -> dict[str, Any]:
        scope = query.get("scope", "global")
        file = self._rules_file(scope, query.get("path"))
        settings = self.settings()
        options: dict[str, str] = {}
        body = ""
        error = None
        if file.exists():
            try:
                options, body = parse_rules(file.read_text(encoding="utf-8"), file)
            except RulesError as exc:
                error = str(exc)
        if scope == "project":
            inherited = load_rules(None).executor_context or settings.executor_context
        else:
            inherited = settings.executor_context
        return {"scope": scope, "file": str(file), "exists": file.exists(), "executor_context":
                options.get("executor_context"), "inherited_context": inherited, "body": body, "error": error}

    def put_rules(self, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        scope = body.get("scope", "global")
        file = self._rules_file(scope, body.get("path"))
        context = body.get("executor_context") or None
        if context is not None and context not in EXECUTOR_CONTEXTS:
            raise ApiError(400, f"context must be one of {', '.join(EXECUTOR_CONTEXTS)}")
        options: dict[str, str] = {}
        if file.exists():
            try:
                options = parse_rules(file.read_text(encoding="utf-8"), file)[0]
            except RulesError:
                options = {}
        options.pop("executor_context", None)
        if context:
            options["executor_context"] = context
        header = "".join(f"{k}: {v}\n" for k, v in options.items())
        text = (f"---\n{header}---\n" if header else "") + (body.get("body") or "").strip() + "\n"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")
        return self.get_rules({"scope": scope, "path": body.get("path")})

    # ---- prompts --------------------------------------------------------

    @staticmethod
    def _prompt_name(name: str | None) -> str:
        if not name or name not in PROMPT_INFO or not (PROMPTS_DIR / name).is_file():
            raise ApiError(404, "unknown prompt")
        return name

    @staticmethod
    def _prompt_dir(scope: str, path: str | None) -> Path:
        if scope == "global":
            return config_dir() / "prompts"
        if scope == "project":
            return project_config_dir(_project_path(path)) / "prompts"
        raise ApiError(400, "scope must be global or project")

    def list_prompts(self, query: dict[str, str], **_: Any) -> dict[str, Any]:
        path = query.get("path")
        items = []
        for name, (label, description) in PROMPT_INFO.items():
            source = "built-in"
            if (config_dir() / "prompts" / name).is_file():
                source = "global"
            if path and (project_config_dir(_project_path(path)) / "prompts" / name).is_file():
                source = "project"
            items.append({"name": name, "label": label, "description": description,
                          "agent": "planner" if name.startswith("planner_") else "executor", "source": source})
        return {"prompts": items}

    def get_prompt(self, query: dict[str, str], **_: Any) -> dict[str, Any]:
        name = self._prompt_name(query.get("name"))
        scope = query.get("scope", "global")
        here = self._prompt_dir(scope, query.get("path")) / name
        builtin = (PROMPTS_DIR / name).read_text(encoding="utf-8")
        inherited, inherited_from = builtin, "built-in"
        global_file = config_dir() / "prompts" / name
        if scope == "project" and global_file.is_file():
            inherited, inherited_from = global_file.read_text(encoding="utf-8"), "global"
        return {
            "name": name,
            "label": PROMPT_INFO[name][0],
            "description": PROMPT_INFO[name][1],
            "scope": scope,
            "overridden": here.is_file(),
            "content": here.read_text(encoding="utf-8") if here.is_file() else inherited,
            "inherited_from": inherited_from,
            "variables": TEMPLATE_VARIABLES.get(name, []),
        }

    def put_prompt(self, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        name = self._prompt_name(body.get("name"))
        content = body.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ApiError(400, "a prompt cannot be empty; use Reset to go back to the default")
        directory = self._prompt_dir(body.get("scope", "global"), body.get("path"))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(content, encoding="utf-8")
        return self.get_prompt({"name": name, "scope": body.get("scope", "global"), "path": body.get("path")})

    def delete_prompt(self, query: dict[str, str], **_: Any) -> dict[str, Any]:
        name = self._prompt_name(query.get("name"))
        file = self._prompt_dir(query.get("scope", "global"), query.get("path")) / name
        if file.exists():
            file.unlink()
        return self.get_prompt(query)

    def preview(self, query: dict[str, str], **_: Any) -> dict[str, Any]:
        name = self._prompt_name(query.get("name") or "planner_plan.txt")
        if not name.startswith("planner_"):
            raise ApiError(400, "preview is for planner prompts")
        path = query.get("path")
        return {"system_prompt": compose_system_prompt(name, str(_project_path(path)) if path else None,
                                                       self.settings())}

    # ---- settings -------------------------------------------------------

    def get_settings(self, **_: Any) -> dict[str, Any]:
        settings = self.settings()
        stored = read_env_file(global_env_file())
        fields = []
        for spec in SETTINGS_FIELDS:
            value = getattr(settings, spec["field"])
            item = {k: v for k, v in spec.items() if k != "field"}
            if spec["kind"] == "secret":
                item["value"] = ""
                item["hint"] = f"saved, ends in {value[-4:]}" if value else "not set"
            else:
                item["value"] = value
            item["saved"] = spec["key"] in stored
            fields.append(item)
        return {"fields": fields, "file": str(global_env_file())}

    def put_settings(self, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        updates = body.get("values") or {}
        kinds = {spec["key"]: spec["kind"] for spec in SETTINGS_FIELDS}
        previous = read_env_file(global_env_file())
        values = dict(previous)
        for key, value in updates.items():
            if key not in kinds:
                raise ApiError(400, f"{key} cannot be changed here")
            if kinds[key] == "secret" and not value:
                continue
            if value is None or value == "":
                values.pop(key, None)
            elif kinds[key] == "bool":
                values[key] = "true" if value in (True, "true", "1", 1) else "false"
            else:
                values[key] = str(value).strip()
        write_env_file(global_env_file(), values)
        try:
            load_settings()
        except ValidationError as exc:
            write_env_file(global_env_file(), previous)
            problem = exc.errors()[0]
            raise ApiError(400, f"{'.'.join(map(str, problem['loc']))}: {problem['msg']}") from exc
        return self.get_settings()


def _routes(d: Dashboard) -> list[tuple[str, re.Pattern[str], Callable[..., Any]]]:
    sid = r"(?P<sid>[0-9a-fA-F-]{8,36})"
    table = [
        ("GET", r"/api/overview", d.overview),
        ("POST", r"/api/projects", d.add_project),
        ("DELETE", r"/api/projects", d.remove_project),
        ("GET", r"/api/project", d.project),
        ("GET", r"/api/fs", d.browse),
        ("POST", r"/api/open", d.open_folder),
        ("POST", r"/api/runs", d.start_run),
        ("GET", rf"/api/sessions/{sid}", d.session_state),
        ("GET", rf"/api/sessions/{sid}/detail", d.session_detail),
        ("POST", rf"/api/sessions/{sid}/answers", d.answer),
        ("POST", rf"/api/sessions/{sid}/plan", d.plan_decision),
        ("POST", rf"/api/sessions/{sid}/pause", d.pause),
        ("POST", rf"/api/sessions/{sid}/resume", d.resume),
        ("GET", r"/api/rules", d.get_rules),
        ("PUT", r"/api/rules", d.put_rules),
        ("GET", r"/api/prompts", d.list_prompts),
        ("GET", r"/api/prompt", d.get_prompt),
        ("PUT", r"/api/prompt", d.put_prompt),
        ("DELETE", r"/api/prompt", d.delete_prompt),
        ("GET", r"/api/preview", d.preview),
        ("GET", r"/api/settings", d.get_settings),
        ("PUT", r"/api/settings", d.put_settings),
    ]
    return [(method, re.compile(pattern), handler) for method, pattern, handler in table]


def make_handler(dashboard: Dashboard) -> type[BaseHTTPRequestHandler]:
    routes = _routes(dashboard)

    class Handler(BaseHTTPRequestHandler):
        server_version = "mmco"

        def log_message(self, *args: Any) -> None:  # keep the terminal quiet
            pass

        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def do_PUT(self) -> None:
            self._handle("PUT")

        def do_DELETE(self) -> None:
            self._handle("DELETE")

        def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
            data = payload if isinstance(payload, bytes) else json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _handle(self, method: str) -> None:
            # Only this machine, and only pages served by this dashboard (they carry the token).
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            if host not in ("127.0.0.1", "localhost"):
                return self._send(403, {"error": "forbidden"})
            url = urlparse(self.path)
            if method == "GET" and url.path in ("/", "/index.html"):
                html = UI_FILE.read_text(encoding="utf-8").replace("{{TOKEN}}", dashboard.token)
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if not url.path.startswith("/api/"):
                return self._send(404, {"error": "not found"})
            if self.headers.get("X-MMCO-Token") != dashboard.token:
                return self._send(403, {"error": "forbidden"})
            try:
                body: dict[str, Any] = {}
                if method in ("POST", "PUT"):
                    length = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(length) or b"{}")
                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                for route_method, pattern, handler in routes:
                    match = pattern.fullmatch(url.path)
                    if match and route_method == method:
                        return self._send(200, handler(query=query, body=body, **match.groupdict()))
                return self._send(404, {"error": "not found"})
            except ApiError as exc:
                return self._send(exc.status, {"error": str(exc)})
            except MMCOError as exc:
                return self._send(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - report instead of dropping the connection
                traceback.print_exc()
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def serve(port: int = 8765, open_browser: bool = True) -> None:
    dashboard = Dashboard()
    handler = make_handler(dashboard)
    server = None
    for candidate in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"no free port between {port} and {port + 19}")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"mmco dashboard: {url}")
    print("In VS Code: Command Palette → 'Simple Browser: Show' → paste the address.")
    print("Press Ctrl+C to close the dashboard (running work keeps going).")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
