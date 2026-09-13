"""Command-line interface. Running `mmco` with no arguments opens chat mode in the current folder."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .config import GLOBAL_ENV_FILE, Settings, config_dir
from .db import Database
from .executor import Executor
from .models import Session, TaskSpec
from .orchestrator import CANCEL_PLAN, Orchestrator
from .planner import Planner
from .rules import (
    EXECUTOR_CONTEXTS,
    eject_prompts,
    ensure_rules_file,
    global_rules_path,
    load_rules,
    project_config_dir,
    project_rules_path,
)
from .utils import PROMPTS_DIR, MMCOError, setup_logging

app = typer.Typer(
    help="Multi-Model Coding Orchestrator: a planner LLM breaks work into tasks that Claude Code executes.\n\n"
    "Run `mmco` inside a project folder to open chat mode.",
    add_completion=False,
)
console = Console()

STATUS_STYLE = {
    "done": "green",
    "failed": "red",
    "blocked": "red",
    "skipped": "yellow",
    "pending": "dim",
    "in_progress": "yellow",
    "planning": "cyan",
    "executing": "yellow",
    "awaiting_input": "magenta",
    "awaiting_approval": "magenta",
    "paused": "magenta",
    "cancelled": "dim",
}
UNFINISHED = ("paused", "awaiting_input", "awaiting_approval", "failed")
CHAT_HELP = """[bold]Type what you want built or changed[/bold] and press Enter. The planner shows a plan; press Enter to run it,
type what to change to get a new plan, or 'no' to cancel. Ctrl+C pauses the current work.

  /status        current session: tasks, attempts, time, cost
  /tasks         full task definitions
  /log [--full]  every planner call, Claude run and verdict
  /resume        continue the last unfinished session in this folder
  /sessions      sessions in this folder
  /rules         edit the planner rules for this project (and how much Claude is told)
  /rules global  edit the rules that apply to every project
  /help          this help
  /quit          exit (also Ctrl+D)"""


class _Holder:
    orchestrator: Orchestrator | None = None


@contextmanager
def _handle_errors(holder: _Holder | None = None) -> Iterator[None]:
    try:
        yield
    except MMCOError as exc:
        console.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        session_id = holder.orchestrator.current_session_id if holder and holder.orchestrator else None
        hint = f" Resume with: [bold]mmco resume {session_id}[/bold]" if session_id else ""
        console.print(f"\n[yellow]Interrupted.[/yellow]{hint}")
        raise typer.Exit(130) from None


def _load() -> tuple[Settings, Database]:
    settings = Settings()
    setup_logging(settings.log_level)
    return settings, Database(settings.db_path)


# ---- interaction hooks ----------------------------------------------------


def _ask_user(questions: list[str]) -> list[str]:
    answers = []
    for index, question in enumerate(questions, 1):
        console.print(Panel(escape(question), title=f"Needs your input ({index}/{len(questions)})",
                            title_align="left", border_style="magenta"))
        try:
            answers.append(console.input("[bold magenta]answer ›[/] ").strip())
        except EOFError:
            raise KeyboardInterrupt from None
    return answers


def _confirm_plan(tasks: list[TaskSpec]) -> str | None:
    table = Table(title="Proposed plan", title_justify="left", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("task")
    table.add_column("checks", justify="right")
    for index, task in enumerate(tasks, 1):
        table.add_row(str(index), escape(task.label), str(len(task.verify_commands)))
    console.print(table)
    try:
        answer = console.input(
            "[bold]Run this plan?[/bold] [dim](Enter = yes · type changes to re-plan · 'no' to cancel)[/dim] "
        ).strip()
    except EOFError:
        return CANCEL_PLAN
    if answer.lower() in ("", "y", "yes", "da", "ok", "go"):
        return None
    if answer.lower() in ("n", "no", "ne", "cancel"):
        return CANCEL_PLAN
    return answer


def _orchestrator(settings: Settings, db: Database, interactive: bool, confirm: bool = False) -> Orchestrator:
    return Orchestrator(
        settings, db, Planner(settings, db), Executor(settings),
        ask_user=_ask_user if interactive else None,
        confirm_plan=_confirm_plan if interactive and confirm else None,
    )


# ---- rendering ------------------------------------------------------------


def _style(status: str) -> str:
    return f"[{STATUS_STYLE.get(status, 'white')}]{status}[/]"


def _duration(ms: int | float) -> str:
    seconds = int(ms / 1000)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _tasks_table(db: Database, session: Session) -> Table:
    table = Table(title="Tasks", title_justify="left")
    for column in ("#", "task id", "title", "status", "attempts", "reform.", "duration", "cost"):
        table.add_column(column, justify="right" if column in ("#", "attempts", "reform.", "cost") else "left")
    for task in db.list_tasks(session.id):
        stats = db.task_stats(task.id)
        table.add_row(
            str(task.order_index + 1), task.id[:8], escape(task.label), _style(task.status),
            str(task.attempts), str(task.reformulations), _duration(stats["duration_ms"]),
            f"${stats['cost_usd']:.2f}",
        )
    return table


def _session_panel(session: Session) -> Panel:
    lines = [
        f"[bold]id[/bold]       {session.id}",
        f"[bold]status[/bold]   {_style(session.status)}",
        f"[bold]project[/bold]  {escape(session.project_dir)}",
        f"[bold]request[/bold]  {escape(session.original_request)}",
        f"[bold]created[/bold]  {session.created_at:%Y-%m-%d %H:%M:%S} UTC",
    ]
    if session.metadata.get("last_error") and session.status != "done":
        lines.append(f"[bold]note[/bold]     [red]{escape(str(session.metadata['last_error']))}[/red]")
    return Panel("\n".join(lines), title="Session", title_align="left")


def _print_summary(db: Database, session: Session, resume_hint: str | None = None) -> None:
    session = db.find_session(session.id)
    s = db.summary(session.id)
    if not s["tasks"]:
        console.print(f"Session {_style(session.status)}"
                      + (f": {escape(str(session.metadata.get('last_error')))}" if session.metadata.get("last_error") else ""))
        return
    wall = (session.updated_at - session.created_at).total_seconds() * 1000
    console.print(_tasks_table(db, session))
    console.print(Panel(
        f"status {_style(session.status)}   tasks {s['tasks']} "
        f"([green]{s['done']} done[/green], {s['skipped']} skipped, [red]{s['failed']} failed[/red], "
        f"{s['blocked']} blocked, {s['pending']} pending)\n"
        f"executions {s['executions']}   reformulations {s['reformulations']}   planner calls {s['planner_calls']}\n"
        f"time: total {_duration(wall)} (claude {_duration(s['executor_time_ms'])}, "
        f"planner {_duration(s['planner_time_ms'])})\n"
        f"cost: claude ${s['executor_cost_usd']:.2f} + planner ${s['planner_cost_usd']:.4f}",
        title="Summary", title_align="left",
    ))
    if session.status in UNFINISHED:
        console.print(resume_hint or f"Continue with: [bold]mmco resume {session.id}[/bold]")


def _show_status(db: Database, session_id: str) -> None:
    session = db.find_session(session_id)
    console.print(_session_panel(session))
    _print_summary(db, session)
    pending = db.list_clarifications(session.id, answered=False)
    if pending:
        console.print(Panel("\n\n".join(escape(c.question) for c in pending),
                            title="Waiting for your input", border_style="magenta"))


def _show_tasks(db: Database, session_id: str) -> None:
    session = db.find_session(session_id)
    for task in db.list_tasks(session.id):
        body = [escape(task.description)]
        if task.acceptance_criteria:
            body.append("\n[bold]Acceptance criteria[/bold]\n" + "\n".join(f"- {escape(c)}" for c in task.acceptance_criteria))
        if task.verify_commands:
            body.append("\n[bold]Verify[/bold]\n" + "\n".join(f"$ {escape(c)}" for c in task.verify_commands))
        if task.files_involved:
            body.append("\n[bold]Files[/bold] " + escape(", ".join(task.files_involved)))
        if task.prompt_override:
            body.append("\n[bold]Reformulated prompt[/bold]\n" + escape(task.prompt_override))
        console.print(Panel(
            "\n".join(body),
            title=f"{task.order_index + 1}. {escape(task.label)}  [{task.id[:8]}]  {_style(task.status)}",
            title_align="left",
            subtitle=f"attempts {task.attempts}, reformulations {task.reformulations}",
            subtitle_align="right",
        ))


def _show_log(db: Database, session_id: str, full: bool = False, as_json: bool = False) -> None:
    dump = db.dump_session(session_id)
    if as_json:
        print(json.dumps(dump, indent=2, default=str, ensure_ascii=False))
        return
    limit = None if full else 1500

    def clip(text: Any) -> str:
        text = "" if text is None else str(text)
        if limit is not None and len(text) > limit:
            text = text[:limit] + f"… (+{len(text) - limit} chars, use --full)"
        return escape(text)

    titles = {t["id"]: f"{t['order_index'] + 1}. {t['title'] or t['description'][:50]}" for t in dump["tasks"]}
    events: list[tuple[str, Panel]] = []
    for call in dump["planner_calls"]:
        request = call["request_json"].get("messages", [])
        user_msg = request[-1]["content"] if request else ""
        body = f"[bold]request (last message)[/bold]\n{clip(user_msg)}\n\n[bold]response[/bold]\n{clip(call['response_text'])}"
        if call["error"]:
            body += f"\n\n[red]error: {escape(call['error'])}[/red]"
        title = f"planner · {call['purpose']} · {_duration(call['duration_ms'])}"
        if call["task_id"]:
            title += f" · {escape(titles.get(call['task_id'], ''))}"
        events.append((call["created_at"], Panel(body, title=title, title_align="left", border_style="cyan")))
    evaluations = {e["execution_id"]: e for e in dump["evaluations"]}
    for ex in dump["executions"]:
        r = ex["result"]
        checks = "\n".join(
            f"{'✓' if v['exit_code'] == 0 and not v['timed_out'] else '✗'} {escape(v['command'])}"
            for v in ex["verify_results"]
        ) or "(none)"
        broken = [v["command"] for v in ex.get("regression_results", []) if v["exit_code"] != 0 or v["timed_out"]]
        if broken:
            checks += "\n[red]regressions: " + escape(", ".join(broken)) + "[/red]"
        body = (
            f"[bold]prompt[/bold]\n{clip(r['prompt'])}\n\n[bold]claude result[/bold]\n{clip(r['result_text'])}\n\n"
            f"[bold]changes[/bold]\n{clip(ex['diff_stat']) or '(none)'}\n\n[bold]verification[/bold]\n{checks}"
        )
        evaluation = evaluations.get(ex["id"])
        if evaluation:
            body += (f"\n\n[bold]verdict[/bold] {evaluation['next_action']} "
                     f"(satisfied={evaluation['satisfied']}): {escape(evaluation['reason'])}")
        flags = [name for name, on in (("timeout", r["timed_out"]), ("error", r["is_error"]),
                                       ("refusal", ex["refusal_detected"])) if on]
        cost = f"${r['cost_usd']:.2f}" if r["cost_usd"] is not None else "$?"
        title = (f"claude · {escape(titles.get(ex['task_id'], ''))} · attempt {ex['attempt']} · "
                 f"{_duration(r['duration_ms'])} · {cost}" + (f" · [red]{', '.join(flags)}[/red]" if flags else ""))
        events.append((ex["created_at"], Panel(body, title=title, title_align="left", border_style="yellow")))
    for _, panel in sorted(events, key=lambda e: datetime.fromisoformat(str(e[0]))):
        console.print(panel)


def _sessions_table(sessions: list[Session], db: Database) -> Table:
    table = Table(title="Sessions", title_justify="left")
    for column in ("id", "created", "status", "tasks", "request"):
        table.add_column(column)
    for session in sessions:
        s = db.summary(session.id)
        table.add_row(session.id[:8], f"{session.created_at:%Y-%m-%d %H:%M}", _style(session.status),
                      f"{s['done']}/{s['tasks']}", escape(session.original_request[:70]))
    return table


def _open_in_editor(path: Path) -> None:
    editor = shutil.which("code") or os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if editor:
        subprocess.run([editor, str(path)], check=False)
        console.print(f"Opened [bold]{escape(str(path))}[/bold]. Changes apply on the next planner call.")
    else:
        console.print(f"Edit this file: [bold]{escape(str(path))}[/bold]")


def _edit_rules(settings: Settings, project: Path | None) -> None:
    if project is None:
        path = ensure_rules_file(global_rules_path(), "Global", settings.executor_context)
    else:
        path = ensure_rules_file(project_rules_path(project), f"Project ({project.name})",
                                 load_rules(None).executor_context or settings.executor_context)
    _open_in_editor(path)


def _context_line(settings: Settings, project: Path) -> str:
    rules = load_rules(project)
    context = rules.executor_context or settings.executor_context
    source = "project rules" if project_rules_path(project) in rules.sources else (
        "global rules" if rules.sources else "default")
    return f"{context} ({source}{', + custom rules' if rules.text else ''})"


def _exit_code(session: Session) -> None:
    if session.status == "failed":
        raise typer.Exit(1)
    if session.status in ("paused", "awaiting_input"):
        raise typer.Exit(2)


# ---- chat mode ------------------------------------------------------------


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        chat(Path.cwd(), yes=False)


@app.command()
def chat(
    project_dir: Annotated[Path, typer.Argument(help="Project folder (default: current folder)")] = Path("."),
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Run plans without asking for approval")] = False,
) -> None:
    """Interactive mode: describe what you want, approve the plan, watch it get built."""
    with _handle_errors():
        settings, db = _load()
        root = project_dir.expanduser().resolve()
        claude = Executor(settings).resolve_binary()
        orchestrator = _orchestrator(settings, db, interactive=True, confirm=not yes)

    console.print(Panel(
        f"[bold]project[/bold]   {escape(str(root))}\n"
        f"[bold]planner[/bold]   {escape(settings.planner_model)}\n"
        f"[bold]executor[/bold]  Claude Code ({escape(settings.claude_model or 'default model')}) · {escape(claude)}\n"
        f"[bold]context[/bold]   {escape(_context_line(settings, root))}\n\n"
        + CHAT_HELP,
        title="mmco", title_align="left", border_style="cyan",
    ))

    def latest_session() -> Session | None:
        sessions = db.list_sessions_for_project(str(root), limit=1)
        return sessions[0] if sessions else None

    def run(action: Callable[[], Session]) -> None:
        try:
            session = action()
            _print_summary(db, session, resume_hint="Type [bold]/resume[/bold] to continue it.")
        except MMCOError as exc:
            console.print(f"[red]error:[/red] {escape(str(exc))}")
        except KeyboardInterrupt:
            console.print("\n[yellow]Paused.[/yellow] Type [bold]/resume[/bold] to continue.")

    last = latest_session()
    if last and last.status in UNFINISHED:
        console.print(f"Unfinished session: [bold]{escape(last.original_request[:100])}[/bold] ({_style(last.status)})")
        try:
            answer = console.input("[bold]Continue it now?[/bold] [dim](Enter = yes · 'no' to skip)[/dim] ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = "no"
        if answer.lower() in ("", "y", "yes", "da"):
            run(lambda: orchestrator.resume(last.id, allow_dirty=True))

    while True:
        try:
            line = console.input("\n[bold cyan]mmco ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not line:
            continue
        if not line.startswith("/"):
            run(lambda: orchestrator.start(root, line, allow_dirty=True))
            continue

        command, _, argument = line[1:].partition(" ")
        argument = argument.strip()
        session = latest_session()
        try:
            if command in ("quit", "exit", "q"):
                return
            if command in ("help", "h", "?"):
                console.print(CHAT_HELP)
            elif command == "rules":
                _edit_rules(settings, None if argument == "global" else root)
            elif command == "sessions":
                console.print(_sessions_table(db.list_sessions_for_project(str(root), limit=20), db))
            elif command in ("status", "tasks", "log", "resume") and session is None:
                console.print("No sessions in this folder yet. Type a request to start one.")
            elif command == "status":
                _show_status(db, argument or session.id)
            elif command == "tasks":
                _show_tasks(db, argument.split()[0] if argument and not argument.startswith("-") else session.id)
            elif command == "log":
                target = next((a for a in argument.split() if not a.startswith("-")), session.id)
                _show_log(db, target, full="--full" in argument)
            elif command == "resume":
                target = db.find_session(argument) if argument else session
                if target.status not in UNFINISHED:
                    console.print(f"The last session is {_style(target.status)}; nothing to resume.")
                else:
                    run(lambda: orchestrator.resume(target.id, allow_dirty=True))
            else:
                console.print(f"Unknown command /{escape(command)}. Type /help.")
        except MMCOError as exc:
            console.print(f"[red]error:[/red] {escape(str(exc))}")


# ---- one-shot commands ------------------------------------------------------


@app.command()
def start(
    project_dir: Annotated[Path, typer.Argument(help="Project directory (created if missing)")],
    request: Annotated[str, typer.Argument(help="What to build")],
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty", help="Commit existing uncommitted changes first")] = False,
    no_input: Annotated[bool, typer.Option("--no-input", help="Never prompt; pause when input is needed")] = False,
    confirm: Annotated[bool, typer.Option("--confirm", help="Ask before running the plan")] = False,
    session_id: Annotated[str | None, typer.Option("--session-id", hidden=True)] = None,
    approve_later: Annotated[bool, typer.Option("--approve-later", hidden=True)] = False,
) -> None:
    """Start a new session non-interactively (for scripts; use `mmco` for chat mode)."""
    holder = _Holder()
    with _handle_errors(holder):
        settings, db = _load()
        holder.orchestrator = _orchestrator(settings, db, interactive=not no_input, confirm=confirm)
        session = holder.orchestrator.start(project_dir, request, allow_dirty=allow_dirty,
                                            session_id=session_id, approve_later=approve_later)
    _print_summary(db, session)
    _exit_code(session)


@app.command()
def resume(
    session_id: Annotated[str, typer.Argument(help="Session id (a unique prefix is enough)")],
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty", help="Commit manual changes before continuing")] = False,
    no_input: Annotated[bool, typer.Option("--no-input", help="Never prompt; pause when input is needed")] = False,
) -> None:
    """Continue a paused, interrupted or failed session."""
    holder = _Holder()
    with _handle_errors(holder):
        settings, db = _load()
        holder.orchestrator = _orchestrator(settings, db, interactive=not no_input)
        session = holder.orchestrator.resume(session_id, allow_dirty=allow_dirty)
    _print_summary(db, session)
    _exit_code(session)


@app.command()
def replay(
    session_id: Annotated[str, typer.Argument(help="Session id")],
    from_task: Annotated[str, typer.Option("--from-task", help="Task id prefix or 1-based task number")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation")] = False,
    no_input: Annotated[bool, typer.Option("--no-input")] = False,
) -> None:
    """Roll the project back to a task's starting commit and re-run from there."""
    holder = _Holder()
    with _handle_errors(holder):
        settings, db = _load()
        session = db.find_session(session_id)
        task = db.find_task(session.id, from_task)
        if not yes:
            typer.confirm(
                f"This runs `git reset --hard` in {session.project_dir} back to before task "
                f"'{task.label}'. Continue?", abort=True,
            )
        holder.orchestrator = _orchestrator(settings, db, interactive=not no_input)
        session = holder.orchestrator.replay(session.id, task.id)
    _print_summary(db, session)
    _exit_code(session)


@app.command()
def status(session_id: Annotated[str, typer.Argument(help="Session id")]) -> None:
    """Show session state and a task table."""
    with _handle_errors():
        _, db = _load()
        _show_status(db, session_id)


@app.command()
def tasks(session_id: Annotated[str, typer.Argument(help="Session id")]) -> None:
    """List tasks with their full definitions."""
    with _handle_errors():
        _, db = _load()
        _show_tasks(db, session_id)


@app.command()
def log(
    session_id: Annotated[str, typer.Argument(help="Session id")],
    full: Annotated[bool, typer.Option("--full", help="Show complete prompts and responses")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Dump everything as JSON")] = False,
) -> None:
    """Dump every planner call, execution and evaluation in order."""
    with _handle_errors():
        _, db = _load()
        _show_log(db, session_id, full=full, as_json=as_json)


@app.command()
def sessions(limit: Annotated[int, typer.Option(help="How many to show")] = 20) -> None:
    """List recent sessions across all projects."""
    with _handle_errors():
        _, db = _load()
        console.print(_sessions_table(db.list_sessions(limit), db))


_EVENT_STYLE = {"planner": "cyan", "claude": "magenta", "orchestrator": "bold white", "checks": "yellow"}


def _render_event(event: dict[str, Any]) -> None:
    from datetime import datetime as _dt

    from .events import SOURCE_LABEL, summarize

    source = event.get("source", "")
    summary = summarize(source, event.get("type", ""), event.get("text", ""), event)
    if not summary:
        return
    style = _EVENT_STYLE.get(source, "white")
    label = SOURCE_LABEL.get(source, source)
    stamp = _dt.fromtimestamp(event.get("ts", 0)).strftime("%H:%M:%S")
    detail_style = "dim" if event.get("type") == "thinking" else ""
    console.print(f"[dim]{stamp}[/dim] [{style}]{label:>8}[/{style}] [{detail_style}]{escape(summary)}[/]")


@app.command()
def logs(
    session_id: Annotated[str | None, typer.Argument(help="Session id (default: most recent)")] = None,
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Keep streaming new events live")] = False,
) -> None:
    """Show the live log of a session: DeepSeek's reasoning and tool calls, and Claude Code's steps."""
    import time as _time

    with _handle_errors():
        settings, db = _load()
        session = db.find_session(session_id) if session_id else next(iter(db.list_sessions(1)), None)
        if session is None:
            console.print("No sessions yet.")
            return
        path = Path(settings.log_dir).expanduser() / "events" / f"{session.id}.jsonl"
        console.print(f"[dim]live log · session {session.id[:8]} · {escape(session.original_request[:70])}[/dim]")
        if not path.exists() and not follow:
            console.print("[dim]No streamed events for this session"
                          " (it ran with streaming off, or hasn't started).[/dim]")
            return
        pos = 0
        try:
            while True:
                if path.exists():
                    with path.open("r", encoding="utf-8") as fh:
                        fh.seek(pos)
                        for line in fh:
                            line = line.strip()
                            if line:
                                try:
                                    _render_event(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                        pos = fh.tell()
                if not follow:
                    break
                _time.sleep(0.4)
        except KeyboardInterrupt:
            console.print("\n[dim]stopped following[/dim]")


@app.command()
def rules(
    project_dir: Annotated[Path, typer.Argument(help="Project folder (default: current folder)")] = Path("."),
    global_: Annotated[bool, typer.Option("--global", help="Edit the rules shared by all projects")] = False,
    show: Annotated[bool, typer.Option("--show", help="Print the effective rules instead of opening an editor")] = False,
) -> None:
    """Edit the planner's rules and how much Claude Code is told (full | task | minimal)."""
    with _handle_errors():
        settings = Settings()
        root = project_dir.expanduser().resolve()
        if show:
            effective = load_rules(None if global_ else root)
            console.print(f"[bold]executor_context:[/bold] {effective.executor_context or settings.executor_context}"
                          f"  [dim](choices: {', '.join(EXECUTOR_CONTEXTS)})[/dim]")
            console.print("[bold]sources:[/bold] " + (", ".join(map(str, effective.sources)) or "(none)"))
            console.print(Panel(escape(effective.text) or "[dim](no custom rules)[/dim]", title="Rules sent to the planner"))
            return
        _edit_rules(settings, None if global_ else root)


@app.command()
def prompts(
    project_dir: Annotated[Path, typer.Argument(help="Project folder (default: current folder)")] = Path("."),
    eject: Annotated[bool, typer.Option("--eject", help="Copy the built-in prompts so you can edit them")] = False,
    global_: Annotated[bool, typer.Option("--global", help="Use the global prompt folder instead of the project's")] = False,
) -> None:
    """Show where prompt overrides go, or copy the built-in prompts there to edit them."""
    root = project_dir.expanduser().resolve()
    target = (config_dir() if global_ else project_config_dir(root)) / "prompts"
    if eject:
        written = eject_prompts(target)
        console.print(f"Copied {len(written)} prompt(s) to [bold]{escape(str(target))}[/bold] "
                      "(existing files were kept). Delete a file to go back to the built-in version.")
        _open_in_editor(target)
        return
    table = Table(title="Prompts", title_justify="left")
    table.add_column("file")
    table.add_column("used from")
    for source in sorted(PROMPTS_DIR.glob("*.txt")):
        project_file = project_config_dir(root) / "prompts" / source.name
        global_file = config_dir() / "prompts" / source.name
        used = project_file if project_file.exists() else global_file if global_file.exists() else None
        table.add_row(source.name, escape(str(used)) if used else "[dim]built-in[/dim]")
    console.print(table)
    console.print("Planner prompts start with 'planner_', Claude Code prompts with 'executor_'. "
                  "Run [bold]mmco prompts --eject[/bold] to customise them for this project.")


@app.command()
def ui(
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1")] = 8765,
    no_browser: Annotated[bool, typer.Option("--no-browser", help="Do not open a browser window")] = False,
) -> None:
    """Open the visual dashboard in your browser."""
    from .web import serve

    serve(port=port, open_browser=not no_browser)


@app.command()
def setup() -> None:
    """Save your OpenRouter key globally and check that Claude Code and git are ready."""
    env_file = GLOBAL_ENV_FILE
    env_file.parent.mkdir(parents=True, exist_ok=True)
    lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
    values = dict(line.split("=", 1) for line in lines if "=" in line and not line.lstrip().startswith("#"))

    key = values.get("OPENROUTER_API_KEY", "")
    if key and not typer.confirm(f"An OpenRouter key is already saved (…{key[-4:]}). Replace it?", default=False):
        pass
    else:
        key = typer.prompt("OpenRouter API key (input hidden)", hide_input=True).strip()
        values["OPENROUTER_API_KEY"] = key

    try:
        request = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read()).get("data", {})
        console.print(f"[green]✓[/green] OpenRouter key works (usage so far ${float(data.get('usage') or 0):.2f})")
    except urllib.error.HTTPError as exc:
        console.print(f"[red]✗[/red] OpenRouter rejected the key (HTTP {exc.code})")
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        console.print(f"[yellow]?[/yellow] Could not verify the key ({exc})")

    settings = Settings(openrouter_api_key=key)
    try:
        claude = Executor(settings).resolve_binary()
        console.print(f"[green]✓[/green] Claude Code found: {claude}")
        if not shutil.which("claude"):
            values.setdefault("MMCO_CLAUDE_BINARY", claude)
    except MMCOError as exc:
        console.print(f"[red]✗[/red] {exc}")
    console.print(f"[green]✓[/green] git found" if shutil.which("git") else "[red]✗[/red] git is not installed")

    env_file.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
    os.chmod(env_file, 0o600)
    console.print(f"Saved settings to {env_file}. Open any folder and run [bold]mmco[/bold].")


if __name__ == "__main__":
    app()
