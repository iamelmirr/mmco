from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mmco.executor import detect_refusal
from mmco.models import EvalResult, ExecutionResult, PlanResponse, ReviewResult, TaskSpec
from mmco.orchestrator import Orchestrator, OrchestratorError
from mmco.workspace import GitError

# ---- fakes ------------------------------------------------------------------


class FakeExecutor:
    """Each scripted step is a callable(project_dir, prompt) that edits files and returns a result."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls: list[dict] = []

    def resolve_binary(self):
        return "/fake/claude"

    def run_task(self, prompt, project_dir, resume_session_id=None, system_prompt="executor_system.txt"):
        self.calls.append({"prompt": prompt, "resume": resume_session_id, "system_prompt": system_prompt})
        step = self.steps.pop(0)
        return step(Path(project_dir), prompt)


def writes(files: dict[str, str], text="Done.", session_id="claude-1", **extra):
    def step(root: Path, prompt: str) -> ExecutionResult:
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        return ExecutionResult(prompt=prompt, result_text=text, claude_session_id=session_id, exit_code=0,
                               cost_usd=0.1, duration_ms=1000, refusal_suspected=detect_refusal(text), **extra)
    return step


def interrupts(files: dict[str, str]):
    def step(root: Path, prompt: str):
        writes(files)(root, prompt)
        raise KeyboardInterrupt
    return step


class FakePlanner:
    def __init__(self, plan, verdicts=(), reformulations=(), questions=(), reviews=()):
        self.plans = list(plan) if isinstance(plan, list) else [plan]
        self.verdicts = list(verdicts)
        self.reformulations = list(reformulations)
        self.questions = list(questions)
        self.reviews = list(reviews)
        self.evaluated = []
        self.plan_calls = []
        self.review_calls = []

    def plan(self, session, listing, clarifications):
        self.plan_calls.append(clarifications)
        return self.plans.pop(0)

    def evaluate(self, session, task, execution, history, clarifications):
        self.evaluated.append((task, execution))
        return self.verdicts.pop(0)

    def reformulate(self, session, task, previous_prompt, failure_reason):
        return self.reformulations.pop(0)

    def clarify(self, session, blocker, task, clarifications):
        return self.questions.pop(0)

    def review(self, session, listing, tasks, reports, checks, clarifications):
        self.review_calls.append({"tasks": tasks, "checks": checks})
        return self.reviews.pop(0) if self.reviews else ReviewResult(complete=True, reason="all good")


def ok(reason="looks good"):
    return EvalResult(satisfied=True, reason=reason, next_action="continue")


def verdict(action, reason="not yet", feedback="", new_tasks=(), satisfied=False):
    return EvalResult(satisfied=satisfied, reason=reason, next_action=action,
                      feedback_for_executor=feedback, new_tasks=list(new_tasks))


def spec(i, title, verify=()):
    return TaskSpec(order_index=i, title=title, description=f"Do {title}", verify_commands=list(verify))


def git(root: Path, *args) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return tmp_path / "project"


def build(settings, db, planner, executor, answers=None, confirm_plan=None):
    if answers is not None:
        queue = list(answers)

        def ask(questions):
            return [queue.pop(0) for _ in questions]
    else:
        ask = None
    return Orchestrator(settings, db, planner, executor, ask_user=ask, confirm_plan=confirm_plan,
                        sleep=lambda _: None)


# ---- tests ------------------------------------------------------------------


def test_happy_path_commits_each_task(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(1, "second"), spec(0, "first")]), [ok(), ok()])
    executor = FakeExecutor([writes({"a.txt": "a"}), writes({"b.txt": "b"})])
    session = build(settings, db, planner, executor).start(project, "build it")

    assert session.status == "done"
    tasks = db.list_tasks(session.id)
    assert [t.title for t in tasks] == ["first", "second"]
    assert all(t.status == "done" and t.end_commit for t in tasks)
    log = git(project, "log", "--format=%s")
    assert "mmco: first" in log and "mmco: second" in log and "mmco: baseline" in log
    assert not git(project, "status", "--porcelain").strip()
    # The executor sees the whole plan and knows which step it is on.
    assert "→ 2. second" in executor.calls[1]["prompt"] and "✓ 1. first" in executor.calls[1]["prompt"]
    # The planner evaluated with the real diff.
    assert "b.txt" in planner.evaluated[1][1].diff_stat
    assert db.summary(session.id)["executions"] == 2


def test_verification_failure_overrides_satisfied_and_retry_resumes_session(settings, db, project):
    check = "test -f done.flag"
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "flag", [check])]),
                          [ok("agent says done"), ok()])
    executor = FakeExecutor([writes({"partial.txt": "x"}, session_id="claude-7"), writes({"done.flag": "1"})])
    session = build(settings, db, planner, executor).start(project, "make a flag")

    assert session.status == "done"
    evaluations = db.list_evaluations(session_id=session.id)
    assert evaluations[0]["next_action"] == "retry" and "[mmco] verification failed" in evaluations[0]["reason"]
    # Retry continues the same Claude conversation with a feedback prompt.
    assert executor.calls[1]["resume"] == "claude-7"
    assert "reviewer checked your work" in executor.calls[1]["prompt"]
    assert "exit code: 1" in executor.calls[1]["prompt"]


def test_reformulate_rolls_back_and_uses_new_prompt(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "api")]),
                          [verdict("reformulate", "misunderstood"), ok()],
                          reformulations=["STEP 1: create api.py exactly"])
    executor = FakeExecutor([writes({"wrong.txt": "oops"}), writes({"api.py": "ok"})])
    session = build(settings, db, planner, executor).start(project, "api")

    assert session.status == "done"
    assert not (project / "wrong.txt").exists()  # rolled back before the second attempt
    assert "STEP 1: create api.py exactly" in executor.calls[1]["prompt"]
    assert executor.calls[1]["resume"] is None
    task = db.list_tasks(session.id)[0]
    assert task.reformulations == 1 and task.description == "Do api"  # original preserved


def test_refusal_forces_reformulation(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "thing")]),
                          [verdict("retry", "did nothing"), ok()], reformulations=["clearer prompt"])
    executor = FakeExecutor([writes({}, text="I can't do that."), writes({"thing.py": "x"})])
    session = build(settings, db, planner, executor).start(project, "thing")

    assert session.status == "done"
    execution = db.list_executions(session_id=session.id)[0]
    assert execution.refusal_detected
    assert db.list_evaluations(session_id=session.id)[0]["next_action"] == "reformulate"


def test_refusal_wording_with_real_changes_is_not_a_refusal(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "thing")]), [ok()])
    executor = FakeExecutor([writes({"thing.py": "x"}, text="I can't find config, so I created thing.py")])
    session = build(settings, db, planner, executor).start(project, "thing")
    assert session.status == "done"
    assert not db.list_executions(session_id=session.id)[0].refusal_detected


def test_exhausted_budget_fails_task_and_blocks_the_rest(settings, db, project):
    # max_attempts_per_task=2, max_reformulations_per_task=1 (see conftest)
    settings.escalate_to_user = False
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "hard"), spec(1, "later")]),
                          [verdict("retry")] * 4, reformulations=["try again differently"])
    executor = FakeExecutor([writes({f"f{i}.txt": "x"}) for i in range(4)])
    session = build(settings, db, planner, executor).start(project, "hard")

    assert session.status == "failed"
    hard, later = db.list_tasks(session.id)
    assert hard.status == "failed" and hard.attempts == 4 and hard.reformulations == 1
    assert later.status == "blocked"
    assert len(executor.calls) == 4


def test_add_task_runs_prerequisite_before_retrying(settings, db, project):
    prerequisite = TaskSpec(title="install deps", description="Add requirements.txt")
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "app")]),
                          [verdict("add_task", "missing deps", new_tasks=[prerequisite]), ok(), ok()])
    executor = FakeExecutor([writes({"app.py": "1"}), writes({"requirements.txt": "flask"}), writes({"app.py": "2"})])
    session = build(settings, db, planner, executor).start(project, "app")

    assert session.status == "done"
    tasks = db.list_tasks(session.id)
    assert [t.title for t in tasks] == ["install deps", "app"]
    assert tasks[0].parent_task_id == tasks[1].id
    assert "install deps" in executor.calls[1]["prompt"].split("# Your task")[1]


def test_clarify_asks_user_and_passes_answers_to_executor(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "db")]),
                          [verdict("clarify", "which database?"), ok()], questions=[["Postgres or SQLite?"]])
    executor = FakeExecutor([writes({}), writes({"db.py": "sqlite"})])
    session = build(settings, db, planner, executor, answers=["SQLite"]).start(project, "db")

    assert session.status == "done"
    assert "Q: Postgres or SQLite?\nA: SQLite" in executor.calls[1]["prompt"]


def test_clarify_without_terminal_pauses_then_resume_continues(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "db")]),
                          [verdict("clarify", "which database?"), ok()], questions=[["Postgres or SQLite?"]])
    executor = FakeExecutor([writes({}), writes({"db.py": "sqlite"})])
    session = build(settings, db, planner, executor).start(project, "db")
    assert session.status == "awaiting_input"

    session = build(settings, db, planner, executor, answers=["SQLite"]).resume(session.id)
    assert session.status == "done"


def test_plan_questions_are_answered_before_planning(settings, db, project):
    planner = FakePlanner([PlanResponse(questions=["Which framework?"]), PlanResponse(tasks=[spec(0, "x")])], [ok()])
    executor = FakeExecutor([writes({"x.py": "1"})])
    session = build(settings, db, planner, executor, answers=["Flask"]).start(project, "web app")
    assert session.status == "done"
    assert planner.plan_calls[1][0].answer == "Flask"


def test_interrupt_then_resume_discards_partial_work(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "one"), spec(1, "two")]), [ok(), ok(), ok()])
    executor = FakeExecutor([writes({"one.txt": "1"}), interrupts({"half.txt": "partial"}), writes({"two.txt": "2"})])
    orchestrator = build(settings, db, planner, executor)
    with pytest.raises(KeyboardInterrupt):
        orchestrator.start(project, "two things")
    session_id = orchestrator.current_session_id
    assert db.find_session(session_id).status == "paused"
    assert (project / "half.txt").exists()

    session = build(settings, db, planner, executor).resume(session_id)
    assert session.status == "done"
    assert not (project / "half.txt").exists()
    assert (project / "one.txt").exists() and (project / "two.txt").exists()


def test_replay_from_task_resets_files(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "one"), spec(1, "two")]), [ok(), ok(), ok()])
    executor = FakeExecutor([writes({"one.txt": "1"}), writes({"two.txt": "v1"}), writes({"two.txt": "v2"})])
    session = build(settings, db, planner, executor).start(project, "two things")
    second = db.list_tasks(session.id)[1]

    session = build(settings, db, planner, executor).replay(session.id, second.id[:8])
    assert session.status == "done"
    assert (project / "two.txt").read_text() == "v2"
    assert db.list_tasks(session.id)[1].attempts == 2


def test_max_loop_iterations_pauses_and_resume_continues(settings, db, project):
    settings.max_loop_iterations = 1
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "one"), spec(1, "two")]), [ok(), ok()])
    executor = FakeExecutor([writes({"one.txt": "1"}), writes({"two.txt": "2"})])
    session = build(settings, db, planner, executor).start(project, "x")
    assert session.status == "paused" and "max loop iterations" in session.metadata["last_error"]
    session = build(settings, db, planner, executor).resume(session.id)
    assert session.status == "done"


def test_dirty_repo_is_refused(settings, db, project):
    project.mkdir()
    git(project, "init", "-q")
    (project / "file.txt").write_text("x")
    git(project, "add", "-A")
    git(project, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    (project / "file.txt").write_text("changed")
    orchestrator = build(settings, db, FakePlanner(PlanResponse()), FakeExecutor([]))
    with pytest.raises(GitError, match="uncommitted changes"):
        orchestrator.start(project, "x")


def test_subdirectory_of_repo_is_refused(settings, db, tmp_path):
    git(tmp_path, "init", "-q")
    orchestrator = build(settings, db, FakePlanner(PlanResponse()), FakeExecutor([]))
    with pytest.raises(GitError, match="inside the git repository"):
        orchestrator.start(tmp_path / "sub", "x")


def test_state_files_inside_project_are_never_committed_or_cleaned(settings, db, tmp_path):
    project = tmp_path / "state"  # the settings fixture puts the DB and logs here
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "x")]), [verdict("reformulate"), ok()],
                          reformulations=["again"])
    executor = FakeExecutor([writes({"x.py": "1"}), writes({"x.py": "2"})])
    session = build(settings, db, planner, executor).start(project, "x")
    assert session.status == "done"
    assert Path(settings.db_path).exists()
    assert "mmco.db" not in git(project, "ls-files")


def test_replay_of_unrun_task_is_rejected(settings, db, project):
    settings.escalate_to_user = False
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "one"), spec(1, "two")]), [verdict("fail", "impossible")])
    executor = FakeExecutor([writes({"one.txt": "1"})])
    session = build(settings, db, planner, executor).start(project, "x")
    assert session.status == "failed"
    with pytest.raises(OrchestratorError, match="never run"):
        build(settings, db, planner, executor).replay(session.id, "2")


def test_integration_hello_world_flask_app(settings, db, project):
    """End-to-end over real git, DB and verify commands, with scripted models."""
    plan = PlanResponse(tasks=[
        TaskSpec(order_index=0, title="Create Flask app", description="Create app.py with a hello route",
                 files_involved=["app.py"], interface_contracts={"GET /": "returns 'Hello, World!'"},
                 verify_commands=["python3 -c \"import ast; ast.parse(open('app.py').read())\"",
                                  "grep -q 'Hello, World!' app.py"]),
        TaskSpec(order_index=1, title="Add requirements", description="Pin flask",
                 verify_commands=["grep -qi flask requirements.txt"]),
    ])
    app_source = (
        "from flask import Flask\n\napp = Flask(__name__)\n\n\n@app.get('/')\n"
        "def hello():\n    return 'Hello, World!'\n"
    )
    planner = FakePlanner(plan, [ok(), ok()])
    executor = FakeExecutor([writes({"app.py": app_source}), writes({"requirements.txt": "flask>=3\n"})])
    session = build(settings, db, planner, executor).start(project, "create a hello world Flask app")

    assert session.status == "done"
    executions = db.list_executions(session_id=session.id)
    assert all(v.passed for e in executions for v in e.verify_results)
    assert len(executions[0].verify_results) == 2
    assert session.metadata["summary"]["done"] == 2


# ---- never give up silently ---------------------------------------------------


def test_stuck_task_is_escalated_and_user_guidance_is_used(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "hard")]),
                          [verdict("retry")] * 4 + [ok()], reformulations=["try differently"])
    executor = FakeExecutor([writes({f"f{i}.txt": "x"}) for i in range(5)])
    session = build(settings, db, planner, executor, answers=["use the stdlib json module"]).start(project, "hard")

    assert session.status == "done"
    assert "Guidance from the user (follow it): use the stdlib json module" in executor.calls[4]["prompt"]
    escalation = db.list_clarifications(session.id, kind="escalation")[0]
    assert "is stuck" in escalation.question and escalation.answer == "use the stdlib json module"
    # Escalation answers are not leaked into later task prompts as "answers from the user".
    assert "Answers from the user" not in executor.calls[4]["prompt"]


def test_escalation_skip_resets_changes_and_finishes(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "hard"), spec(1, "easy")]),
                          [verdict("fail", "impossible"), ok()])
    executor = FakeExecutor([writes({"half.txt": "x"}), writes({"easy.txt": "y"})])
    session = build(settings, db, planner, executor, answers=["skip"]).start(project, "x")

    assert session.status == "done"
    hard, easy = db.list_tasks(session.id)
    assert hard.status == "skipped" and easy.status == "done"
    assert not (project / "half.txt").exists()


def test_escalation_abort_fails_session(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "hard"), spec(1, "later")]), [verdict("fail", "nope")])
    executor = FakeExecutor([writes({"a.txt": "x"})])
    session = build(settings, db, planner, executor, answers=["abort"]).start(project, "x")
    assert session.status == "failed"
    assert [t.status for t in db.list_tasks(session.id)] == ["failed", "blocked"]


def test_escalation_without_terminal_waits_for_resume(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "hard")]), [verdict("fail", "nope"), ok()])
    executor = FakeExecutor([writes({"a.txt": "x"}), writes({"b.txt": "y"})])
    session = build(settings, db, planner, executor).start(project, "x")
    assert session.status == "awaiting_input"
    assert db.list_tasks(session.id)[0].status == "pending"

    session = build(settings, db, planner, executor, answers=[""]).resume(session.id)
    assert session.status == "done"


def test_final_review_adds_missing_work(settings, db, project):
    missing = TaskSpec(title="add delete endpoint", description="DELETE /todos/<id>")
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "crud")]), [ok(), ok()],
                          reviews=[ReviewResult(complete=False, reason="delete missing", new_tasks=[missing]),
                                   ReviewResult(complete=True)])
    executor = FakeExecutor([writes({"crud.py": "1"}), writes({"delete.py": "2"})])
    session = build(settings, db, planner, executor).start(project, "todo crud")

    assert session.status == "done"
    assert [t.title for t in db.list_tasks(session.id)] == ["crud", "add delete endpoint"]
    assert len(planner.review_calls) == 2


def test_final_review_turns_failing_checks_into_a_fix_task(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "one", ["test -f keep.txt"]), spec(1, "two")]),
                          [ok(), ok(), ok()])

    def deletes_keep(root, prompt):
        (root / "keep.txt").unlink()  # breaks task one's check without the regression gate noticing
        return writes({"two.txt": "2"})(root, prompt)

    settings.regression_checks = False
    executor = FakeExecutor([writes({"keep.txt": "1"}), deletes_keep, writes({"keep.txt": "1"})])
    session = build(settings, db, planner, executor).start(project, "x")

    assert session.status == "done"
    tasks = db.list_tasks(session.id)
    assert tasks[-1].title == "Fix failing checks" and tasks[-1].verify_commands == ["test -f keep.txt"]


def test_regression_in_earlier_task_blocks_completion(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "one", ["test -f keep.txt"]), spec(1, "two")]),
                          [ok(), ok("looks done"), ok()])

    def deletes_keep(root, prompt):
        (root / "keep.txt").unlink()
        return writes({"two.txt": "2"}, session_id="claude-2")(root, prompt)

    executor = FakeExecutor([writes({"keep.txt": "1"}), deletes_keep, writes({"keep.txt": "1"})])
    session = build(settings, db, planner, executor).start(project, "x")

    assert session.status == "done"
    evaluation = db.list_evaluations(session_id=session.id)[1]
    assert evaluation["next_action"] == "retry" and "broke earlier tasks" in evaluation["reason"]
    assert "broke checks from earlier tasks" in executor.calls[2]["prompt"]


def api_error(text="API Error: 529 Overloaded", status=529):
    def step(root, prompt):
        return ExecutionResult(prompt=prompt, result_text=text, is_error=True, exit_code=1,
                               raw_output={"api_error_status": status, "is_error": True})
    return step


def test_transient_claude_errors_are_retried_without_spending_attempts(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "one")]), [ok()])
    executor = FakeExecutor([api_error(), api_error(status=429, text="rate limited"), writes({"one.txt": "1"})])
    session = build(settings, db, planner, executor).start(project, "x")
    assert session.status == "done"
    task = db.list_tasks(session.id)[0]
    assert task.attempts == 1 and len(executor.calls) == 3


def test_fatal_claude_error_pauses_session(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "one")]), [ok()])
    executor = FakeExecutor([api_error("Credit balance is too low", status=None)])
    orchestrator = build(settings, db, planner, executor)
    with pytest.raises(OrchestratorError, match="cannot continue"):
        orchestrator.start(project, "x")
    assert db.find_session(orchestrator.current_session_id).status == "paused"


def test_planner_outage_during_evaluation_keeps_the_work(settings, db, project):
    from mmco.planner import PlannerError

    class FlakyPlanner(FakePlanner):
        failed = False

        def evaluate(self, *args):
            if not self.failed:
                self.failed = True
                raise PlannerError("openrouter down")
            return super().evaluate(*args)

    planner = FlakyPlanner(PlanResponse(tasks=[spec(0, "one")]), [ok()])
    executor = FakeExecutor([writes({"one.txt": "1"})])
    session = build(settings, db, planner, executor).start(project, "x")
    assert session.status == "paused"

    session = build(settings, db, planner, executor).resume(session.id)
    assert session.status == "done"
    assert len(executor.calls) == 1  # evaluated the existing run instead of re-running Claude
    assert (project / "one.txt").exists()


def test_planner_can_fix_a_broken_verify_command(settings, db, project):
    fixed = EvalResult(satisfied=True, reason="check pointed at wrong file", next_action="continue",
                       updated_verify_commands=["test -f app.py"])
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "app", ["test -f wrong_name.py"])]), [fixed])
    executor = FakeExecutor([writes({"app.py": "1"})])
    session = build(settings, db, planner, executor).start(project, "x")
    assert session.status == "done"
    task = db.list_tasks(session.id)[0]
    assert task.verify_commands == ["test -f app.py"]
    assert db.list_executions(session_id=session.id)[0].verify_results[0].passed


def test_lock_held_by_another_live_process(tmp_path):
    import os
    from mmco.workspace import Workspace
    workspace = Workspace(tmp_path / "p")
    workspace.root.mkdir()
    workspace.ensure_repo()
    (workspace.root / ".git" / "mmco.lock").write_text(str(os.getppid()))
    with pytest.raises(GitError, match="already working"):
        workspace.acquire_lock()
    (workspace.root / ".git" / "mmco.lock").write_text("999999")  # stale pid
    workspace.acquire_lock()
    workspace.release_lock()
    assert not (workspace.root / ".git" / "mmco.lock").exists()


# ---- plan approval, follow-ups, rules and context levels ------------------------


def test_plan_feedback_replans_and_cancel_cancels(settings, db, project):
    replies = ["use FastAPI instead", None]
    planner = FakePlanner([PlanResponse(tasks=[spec(0, "flask app")]), PlanResponse(tasks=[spec(0, "fastapi app")])],
                          [ok()])
    executor = FakeExecutor([writes({"main.py": "1"})])
    session = build(settings, db, planner, executor, confirm_plan=lambda tasks: replies.pop(0)).start(project, "api")
    assert session.status == "done"
    assert [t.title for t in db.list_tasks(session.id)] == ["fastapi app"]
    assert planner.plan_calls[1][0].answer == "use FastAPI instead"

    from mmco.orchestrator import CANCEL_PLAN
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "x")]))
    session = build(settings, db, planner, FakeExecutor([]), confirm_plan=lambda tasks: CANCEL_PLAN).start(project, "y")
    assert session.status == "cancelled" and not db.list_tasks(session.id)


def test_follow_up_request_knows_previous_requests(settings, db, project):
    planner = FakePlanner([PlanResponse(tasks=[spec(0, "app")]), PlanResponse(tasks=[spec(0, "dark mode")])],
                          [ok(), ok()])
    executor = FakeExecutor([writes({"app.py": "1"}), writes({"theme.css": "dark"})])
    build(settings, db, planner, executor).start(project, "build an app")
    second = build(settings, db, planner, executor, answers=[]).start(project, "add dark mode")
    assert second.status == "done"
    assert second.metadata["previous_requests"] == [{"request": "build an app", "status": "done"}]


def write_rules(isolated_config, project, text):
    from mmco.rules import project_rules_path
    path = project_rules_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.mark.parametrize("context", ["full", "task", "minimal"])
def test_executor_context_controls_what_claude_sees(settings, db, project, isolated_config, context):
    write_rules(isolated_config, project, f"---\nexecutor_context: {context}\n---\nOnly technical instructions.\n")
    task = TaskSpec(title="storage layer", description="Implement TodoStore in store.py",
                    files_involved=["store.py"], verify_commands=["test -f store.py"])
    planner = FakePlanner(PlanResponse(tasks=[task]), [ok()])
    executor = FakeExecutor([writes({"store.py": "1"})])
    build(settings, db, planner, executor).start(project, "SECRET PRODUCT IDEA: a todo app for dentists")

    prompt = executor.calls[0]["prompt"]
    assert "Implement TodoStore in store.py" in prompt and "test -f store.py" in prompt
    assert "Only technical instructions" not in prompt  # rules go to the planner, never to Claude
    if context == "full":
        assert "SECRET PRODUCT IDEA" in prompt and "→ 1. storage layer" in prompt
        assert executor.calls[0]["system_prompt"] == "executor_system.txt"
    else:
        assert "SECRET PRODUCT IDEA" not in prompt and "Plan" not in prompt
    if context == "minimal":
        assert "Files likely involved" not in prompt
        assert executor.calls[0]["system_prompt"] == "executor_system_minimal.txt"
    if context == "task":
        assert "Files likely involved" in prompt


def test_project_prompt_override_is_used(settings, db, project, isolated_config):
    from mmco.rules import project_config_dir
    override = project_config_dir(project) / "prompts" / "executor_task.txt"
    override.parent.mkdir(parents=True)
    override.write_text("CUSTOM TEMPLATE\n$task_block")
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "one")]), [ok()])
    executor = FakeExecutor([writes({"one.txt": "1"})])
    build(settings, db, planner, executor).start(project, "x")
    assert executor.calls[0]["prompt"].startswith("CUSTOM TEMPLATE\n## one")


# ---- dashboard flow: stored plan approval and answers from outside the process ----


def test_stored_plan_approval_flow(settings, db, project):
    planner = FakePlanner([PlanResponse(tasks=[spec(0, "first idea")]), PlanResponse(tasks=[spec(0, "second idea")])],
                          [ok(), ok()])
    executor = FakeExecutor([writes({"b.txt": "b"})])
    session = build(settings, db, planner, executor).start(project, "x", approve_later=True, session_id="11111111-2222-3333-4444-555555555555")
    assert session.id == "11111111-2222-3333-4444-555555555555"
    assert session.status == "awaiting_approval" and not db.list_tasks(session.id)

    # Resuming without a decision keeps waiting.
    assert build(settings, db, planner, executor).resume(session.id).status == "awaiting_approval"

    orchestrator = build(settings, db, planner, executor)
    orchestrator.set_plan_decision(session.id, "change", "try something else")
    session = build(settings, db, planner, executor).resume(session.id)
    assert session.status == "awaiting_approval"
    assert session.metadata["proposed_plan"][0]["title"] == "second idea"
    assert planner.plan_calls[1][0].answer == "try something else"

    edited = [{"title": "edited task", "description": "Do the edited thing"}]
    build(settings, db, planner, executor).set_plan_decision(session.id, "approve", tasks=edited)
    session = build(settings, db, planner, executor).resume(session.id)
    assert session.status == "done"
    assert [t.title for t in db.list_tasks(session.id)] == ["edited task"]
    assert "Do the edited thing" in executor.calls[0]["prompt"]


def test_stored_plan_cancel(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "x")]))
    session = build(settings, db, planner, FakeExecutor([])).start(project, "x", approve_later=True)
    build(settings, db, planner, FakeExecutor([])).set_plan_decision(session.id, "cancel")
    assert build(settings, db, planner, FakeExecutor([])).resume(session.id).status == "cancelled"


def test_escalation_answered_from_dashboard(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "hard"), spec(1, "easy")]), [verdict("fail", "nope"), ok()])
    executor = FakeExecutor([writes({"half.txt": "x"}), writes({"easy.txt": "y"})])
    session = build(settings, db, planner, executor).start(project, "x")
    assert session.status == "awaiting_input"
    question = db.list_clarifications(session.id, answered=False)[0]
    build(settings, db, planner, executor).record_answers(session.id, {question.id: "skip"})
    assert db.list_tasks(session.id)[0].status == "skipped" and not (project / "half.txt").exists()
    session = build(settings, db, planner, executor).resume(session.id)
    assert session.status == "done"
    with pytest.raises(OrchestratorError, match="already answered"):
        build(settings, db, planner, executor).record_answers(session.id, {question.id: "again"})


def test_dependency_folders_and_secrets_are_never_committed_or_deleted(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "setup")]), [verdict("reformulate"), ok()], reformulations=["again"])
    files = {".venv/bin/python": "binary", "node_modules/x/index.js": "x", ".env": "SECRET=1", "app.py": "print(1)",
             ".env.example": "SECRET=", "env/settings.py": "DEBUG = True"}
    executor = FakeExecutor([writes(files), writes(files)])
    session = build(settings, db, planner, executor).start(project, "x")
    assert session.status == "done"
    tracked = git(project, "ls-files")
    assert "app.py" in tracked and ".env.example" in tracked and "env/settings.py" in tracked
    assert ".venv" not in tracked and "node_modules" not in tracked and "\n.env\n" not in "\n" + tracked
    first = db.list_executions(session_id=session.id)[0]
    assert ".venv" not in first.diff_stat and "app.py" in first.diff_stat
    assert (project / ".venv/bin/python").exists()  # the rollback before the second attempt kept it
