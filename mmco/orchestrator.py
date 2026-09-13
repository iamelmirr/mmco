"""The plan → execute → verify → evaluate loop. All state lives in SQLite, so any run can be resumed.

Nothing is silently given up on: a stuck task is escalated to the user, and a finished session is
reviewed against the original request before it is declared done.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from .config import Settings
from .db import Database
from .executor import classify_failure
from .models import EvalResult, Execution, ExecutionResult, Session, SessionStatus, Task, TaskSpec, VerifyResult
from .planner import PlannerError
from .rules import load_rules, prompt_dirs
from .tools import Toolbox
from .utils import MMCOError, add_session_log, bullets, dump_crash, load_prompt, render, truncate
from .workspace import Workspace, run_verify_commands

AskUser = Callable[[list[str]], list[str]]
# Receives the proposed tasks; returns None to approve, CANCEL_PLAN to cancel, or feedback text to re-plan.
ConfirmPlan = Callable[[list[TaskSpec]], str | None]
CANCEL_PLAN = "__cancel__"

SKIP_ANSWERS = {"skip", "s"}
ABORT_ANSWERS = {"abort", "a", "stop", "quit"}
RETRY_ANSWERS = {"", "retry", "r", "continue", "c"}


class OrchestratorError(MMCOError):
    pass


class Orchestrator:
    MAX_PLAN_ROUNDS = 5

    def __init__(
        self,
        settings: Settings,
        db: Database,
        planner: Any,
        executor: Any,
        ask_user: AskUser | None = None,
        confirm_plan: ConfirmPlan | None = None,
        workspace_factory: Callable[[Path], Workspace] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        self.db = db
        self.planner = planner
        self.executor = executor
        self.ask_user = ask_user
        self.confirm_plan = confirm_plan
        self._sleep = sleep
        self.workspace_factory = workspace_factory or (
            lambda root: Workspace(
                root,
                enabled=settings.git_checkpoints,
                protected_paths=[settings.db_path, settings.log_dir],
            )
        )
        self.current_session_id: str | None = None
        self._workspace: Workspace | None = None

    # ---- entry points ---------------------------------------------------

    def start(
        self,
        project_dir: str | Path,
        request: str,
        allow_dirty: bool = False,
        session_id: str | None = None,
        approve_later: bool = False,
    ) -> Session:
        """Start a session. With approve_later, the plan is stored for approval (dashboard) instead of run."""
        self.executor.resolve_binary()
        root = Path(project_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        workspace = self.workspace_factory(root)
        baseline = workspace.ensure_repo(allow_dirty=allow_dirty)
        previous = [
            {"request": s.original_request, "status": s.status}
            for s in reversed(self.db.list_sessions_for_project(str(root), limit=5))
        ]
        metadata: dict[str, Any] = {"baseline_commit": baseline, "previous_requests": previous}
        if approve_later:
            metadata["plan_approval"] = "stored"
        session = self.db.create_session(request, str(root), metadata, session_id=session_id)
        logger.info("session {} started in {}", session.id, root)
        return self._drive(session, workspace)

    def resume(self, session_id: str, allow_dirty: bool = False) -> Session:
        session = self.db.find_session(session_id)
        if session.status in ("done", "cancelled"):
            logger.info("session {} is already {}", session.id, session.status)
            return session
        self.executor.resolve_binary()
        workspace = self.workspace_factory(Path(session.project_dir))
        workspace.verify_root()
        tasks = self.db.list_tasks(session.id)

        interrupted = [t for t in tasks if t.status == "in_progress"]
        for task in interrupted:
            if self._unevaluated_execution(task):
                logger.info("task '{}' finished running before the interruption; it will be evaluated", task.label)
            else:
                logger.warning("task '{}' was interrupted; discarding its partial changes", task.label)
                workspace.reset_to(task.start_commit)
                task.resume_claude_session_id = None
            task.status = "pending"
            self.db.save_task(task)
        # A task paused between attempts (escalation, cost cap, iteration cap) keeps its work on disk on purpose.
        mid_task = any(t.status == "pending" and t.start_commit and not t.end_commit for t in tasks)
        if not interrupted and not mid_task and workspace.is_dirty():
            if not allow_dirty:
                raise OrchestratorError(
                    f"{session.project_dir} has uncommitted changes; commit them or pass --allow-dirty"
                )
            workspace.commit("mmco: manual changes before resume")

        if session.status == "failed":
            for task in tasks:
                if task.status in ("failed", "blocked"):
                    logger.info("re-queueing task '{}' with a fresh attempt budget", task.label)
                    task.status = "pending"
                    task.cycle_attempts = 0
                    task.reformulations = 0
                    self.db.save_task(task)
            session.metadata["final_reviews"] = 0
        logger.info("resuming session {}", session.id)
        return self._drive(session, workspace)

    def replay(self, session_id: str, from_task: str) -> Session:
        session = self.db.find_session(session_id)
        self.executor.resolve_binary()
        workspace = self.workspace_factory(Path(session.project_dir))
        workspace.verify_root()
        target = self.db.find_task(session.id, from_task)
        if not target.start_commit:
            raise OrchestratorError(f"task '{target.label}' has never run, so there is nothing to replay from")
        workspace.reset_to(target.start_commit)
        for task in self.db.list_tasks(session.id):
            if task.order_index >= target.order_index:
                task.status = "pending"
                task.cycle_attempts = 0
                task.reformulations = 0
                task.prompt_override = None
                task.last_feedback = None
                task.resume_claude_session_id = None
                task.start_commit = None
                task.end_commit = None
                self.db.save_task(task)
        session.metadata["final_reviews"] = 0
        logger.info("replaying session {} from task '{}'", session.id, target.label)
        return self._drive(session, workspace)

    # ---- driver ---------------------------------------------------------

    def _drive(self, session: Session, workspace: Workspace) -> Session:
        self.current_session_id = session.id
        self._workspace = workspace
        log_path = add_session_log(self.settings.log_dir, session.id)
        logger.debug("session log: {}", log_path)
        workspace.acquire_lock()
        try:
            if not self.db.list_tasks(session.id) and not self._plan(session, workspace):
                return session
            return self._loop(session, workspace)
        except KeyboardInterrupt:
            self._set_status(session, "paused", last_error="interrupted by user")
            raise
        except PlannerError as exc:
            logger.error("planner failed: {}", exc)
            self._set_status(session, "paused", last_error=str(exc))
            return session
        except sqlite3.Error as exc:
            path = dump_crash(self._crash_state(session, exc))
            raise OrchestratorError(f"database error ({exc}); state dumped to {path}") from exc
        except MMCOError as exc:
            self._set_status(session, "paused", last_error=str(exc))
            raise
        finally:
            workspace.release_lock()

    def _plan(self, session: Session, workspace: Workspace) -> bool:
        proposed = session.metadata.get("proposed_plan")
        if proposed is not None:
            decision = session.metadata.get("plan_decision")
            if not decision:
                self._set_status(session, "awaiting_approval")
                return False
            session.metadata.pop("proposed_plan", None)
            session.metadata.pop("plan_decision", None)
            action = decision.get("action")
            if action == "cancel":
                self._set_status(session, "cancelled", last_error="plan cancelled by user")
                return False
            if action == "approve":
                specs = [TaskSpec.model_validate(t) for t in (decision.get("tasks") or proposed)]
                if specs:
                    return self._accept_plan(session, specs)
            if decision.get("feedback"):
                self._record_plan_feedback(session, decision["feedback"])

        self._set_status(session, "planning")
        for _ in range(self.MAX_PLAN_ROUNDS):
            if not self._answer_pending(session):
                return False
            logger.info("planning with {}...", self.settings.planner_model)
            plan = self.planner.plan(
                session, workspace.listing(), self.db.list_clarifications(session.id, True, kind="question"),
                read_tools=self.settings.allow_planner_executor,
            )
            if plan.tasks:
                specs = sorted(plan.tasks, key=lambda t: t.order_index)
                if session.metadata.get("plan_approval") == "stored":
                    self._set_status(session, "awaiting_approval",
                                     proposed_plan=[spec.model_dump(mode="json") for spec in specs])
                    logger.info("plan with {} tasks is waiting for approval", len(specs))
                    return False
                if self.confirm_plan is not None:
                    feedback = self.confirm_plan(specs)
                    if feedback == CANCEL_PLAN:
                        self._set_status(session, "cancelled", last_error="plan cancelled by user")
                        return False
                    if feedback:
                        self._record_plan_feedback(session, feedback)
                        continue
                return self._accept_plan(session, specs)
            if not plan.questions:
                self._set_status(session, "paused", last_error="planner returned no tasks and no questions")
                return False
            logger.info("planner needs clarification before planning")
            self.db.add_questions(session.id, None, plan.questions)
        self._set_status(session, "paused", last_error="could not agree on a plan; resume to try again")
        return False

    def _accept_plan(self, session: Session, specs: list[TaskSpec]) -> bool:
        tasks = self.db.insert_tasks(session.id, specs)
        logger.info("plan has {} tasks:", len(tasks))
        for task in tasks:
            logger.info("  {}. {}", task.order_index + 1, task.label)
        self._set_status(session, "executing")
        return True

    def _record_plan_feedback(self, session: Session, feedback: str) -> None:
        clarification = self.db.add_questions(session.id, None, ["Feedback on the proposed plan"])[0]
        self.db.answer_clarification(clarification.id, feedback)

    # ---- decisions made outside a running process (dashboard) ---------------

    def record_answers(self, session_id: str, answers: dict[str, str]) -> Session:
        session = self.db.find_session(session_id)
        self._workspace = self.workspace_factory(Path(session.project_dir))
        pending = {c.id: c for c in self.db.list_clarifications(session.id, answered=False)}
        for clarification_id, answer in answers.items():
            clarification = pending.get(clarification_id)
            if clarification is None:
                raise OrchestratorError("that question was already answered or does not exist")
            self._store_answer(session, clarification, answer)
        return session

    def set_plan_decision(
        self, session_id: str, action: str, feedback: str = "", tasks: list[dict[str, Any]] | None = None
    ) -> Session:
        session = self.db.find_session(session_id)
        if session.metadata.get("proposed_plan") is None:
            raise OrchestratorError("this session has no plan waiting for approval")
        if action not in ("approve", "change", "cancel"):
            raise OrchestratorError(f"unknown plan decision '{action}'")
        if action == "change" and not feedback.strip():
            raise OrchestratorError("describe what should change in the plan")
        validated = [TaskSpec.model_validate(t).model_dump(mode="json") for t in tasks] if tasks else None
        if action == "approve" and tasks is not None and not validated:
            raise OrchestratorError("the plan needs at least one task")
        session.metadata["plan_decision"] = {"action": action, "feedback": feedback, "tasks": validated}
        self.db.update_session(session)
        return session

    def _loop(self, session: Session, workspace: Workspace) -> Session:
        s = self.settings
        self._set_status(session, "executing")
        iterations = 0
        while True:
            if not self._answer_pending(session):
                return session
            task = self.db.next_pending_task(session.id)
            if task is None:
                if self._final_review(session, workspace):
                    continue
                break
            if iterations >= s.max_loop_iterations:
                logger.warning("max loop iterations ({}) reached; pausing", s.max_loop_iterations)
                self._set_status(session, "paused", last_error="max loop iterations reached; resume to continue")
                return session
            if s.max_session_cost_usd is not None:
                cost = self.db.session_cost(session.id)
                if cost >= s.max_session_cost_usd:
                    logger.error("session cost ${:.2f} reached the ${:.2f} cap; pausing", cost, s.max_session_cost_usd)
                    self._set_status(session, "paused", last_error="session cost cap reached")
                    return session
            iterations += 1
            self._run_task(session, task, workspace)
            if session.status == "awaiting_input":
                return session
        return self._finish(session)

    # ---- one task attempt -----------------------------------------------

    def _run_task(self, session: Session, task: Task, workspace: Workspace) -> None:
        s = self.settings
        tasks = self.db.list_tasks(session.id)
        if task.start_commit is None:
            task.start_commit = workspace.head()
        task.status = "in_progress"
        self.db.save_task(task)

        execution = self._unevaluated_execution(task)
        if execution is None:
            execution = self._execute(session, task, tasks, workspace)

        self._evaluate_and_dispatch(session, task, execution, workspace)

    def _evaluate_and_dispatch(
        self, session: Session, task: Task, execution: Execution, workspace: Workspace
    ) -> None:
        """Judge one execution, apply the hard guards, persist the verdict, and act on it."""
        s = self.settings
        logger.info("evaluating attempt...")
        evaluation = self.planner.evaluate(
            session, task, execution, self.db.list_evaluations(task_id=task.id),
            self.db.list_clarifications(session.id, True, kind="question"),
            read_tools=self.settings.allow_planner_executor,
        )
        if evaluation.updated_verify_commands and evaluation.updated_verify_commands != task.verify_commands:
            logger.warning("planner replaced the verify commands: {} → {}",
                           task.verify_commands, evaluation.updated_verify_commands)
            task.verify_commands = evaluation.updated_verify_commands
            self.db.save_task(task)
            execution.verify_results = run_verify_commands(
                task.verify_commands, session.project_dir, s.verify_timeout_seconds
            )
            self.db.update_execution_checks(execution)
        evaluation = self._apply_guards(evaluation, execution)
        self.db.add_evaluation(execution, evaluation)
        logger.bind(event="evaluation", evaluation=evaluation.model_dump()).info(
            "verdict: {} (satisfied={}) — {}", evaluation.next_action, evaluation.satisfied, evaluation.reason
        )
        self._dispatch(session, task, execution, evaluation, workspace)

    def _execute(self, session: Session, task: Task, tasks: list[Task], workspace: Workspace) -> Execution:
        s = self.settings
        # A Claude retry (--resume with feedback) must stay with Claude; never re-ask who should run it.
        forced_claude = bool(task.resume_claude_session_id and task.last_feedback)

        executor = "claude"
        if s.allow_planner_executor and not forced_claude:
            executor, reason = self.planner.choose_executor(
                session, task, Toolbox(session.project_dir, allow_write=False)
            )
            logger.info("executor for '{}': {} ({})", task.label, executor, reason or "no reason given")
        # A planner task being retried with feedback restarts from the checkpoint (see _execute_as).
        return self._execute_as(
            session, task, tasks, workspace, executor,
            reset_for_planner_retry=(executor == "planner" and bool(task.last_feedback)),
        )

    def _execute_as(
        self, session: Session, task: Task, tasks: list[Task], workspace: Workspace, executor: str,
        reset_for_planner_retry: bool = False,
    ) -> Execution:
        """Run this task with a specific executor and put the result through the shared verify/regression tail.

        reset_for_planner_retry rolls the tree back to the checkpoint before the planner runs. It is meant
        ONLY for a planner *retry* (which has no conversation to resume, so it must not compound the previous
        attempt's rolled-back-but-still-on-disk edits). A take_over passes False so the planner BUILDS ON the
        coding agent's work instead of discarding the very changes it is supposed to finish.
        """
        s = self.settings
        position = next((i for i, t in enumerate(tasks, 1) if t.id == task.id), "?")
        task.executor = executor
        self.db.save_task(task)

        if executor == "planner":
            logger.info(
                "▶ task {}/{} '{}' (attempt {}, executor: DeepSeek)",
                position, len(tasks), task.label, task.attempts + 1,
            )
            if reset_for_planner_retry:
                workspace.reset_to(task.start_commit)
            report = self.planner.execute_task(session, task, Toolbox(session.project_dir, allow_write=True))
            result = ExecutionResult(
                prompt=self._task_block(task), result_text=str(report), cost_usd=None, claude_session_id=None
            )
            agent: str = "planner"
        else:
            result, agent = self._execute_with_claude(session, task, tasks, position)

        diff_stat, diff = workspace.diff_since(task.start_commit)
        verify: list[VerifyResult] = []
        regressions: list[VerifyResult] = []
        # Planner runs never "time out"; a timed-out Claude run leaves the tree in an unknown state.
        if not result.timed_out:
            verify = run_verify_commands(task.verify_commands, session.project_dir, s.verify_timeout_seconds)
            if s.regression_checks:
                regressions = run_verify_commands(
                    self._regression_commands(tasks, task), session.project_dir, s.verify_timeout_seconds
                )
        task.attempts += 1
        task.cycle_attempts += 1
        execution = Execution(
            task_id=task.id,
            session_id=session.id,
            attempt=task.attempts,
            agent=agent,
            result=result,
            # Only trust refusal phrasing for Claude, and only when it also changed nothing.
            refusal_detected=agent == "claude" and result.refusal_suspected and not diff.strip(),
            diff_stat=diff_stat,
            diff=truncate(diff, s.max_diff_chars),
            verify_results=verify,
            regression_results=regressions,
        )
        self.db.add_execution(execution)
        self.db.save_task(task)
        if execution.refusal_detected:
            logger.warning("refusal detected in executor output")
        return execution

    def _execute_with_claude(
        self, session: Session, task: Task, tasks: list[Task], position: Any
    ) -> tuple[ExecutionResult, str]:
        context = self._executor_context(session)
        resume_id = task.resume_claude_session_id
        if resume_id and task.last_feedback:
            prompt = self._retry_prompt(session, task)
        else:
            resume_id = None
            prompt = self._task_prompt(session, task, tasks, context)
            if task.needs_continuity:
                resume_id = self._previous_claude_session(tasks, task)
        system_prompt = "executor_system_minimal.txt" if context == "minimal" else "executor_system.txt"
        logger.info(
            "▶ task {}/{} '{}' (attempt {}{}, context: {})",
            position, len(tasks), task.label, task.attempts + 1, ", resumed session" if resume_id else "", context,
        )
        result = self._run_claude(prompt, session.project_dir, resume_id, system_prompt)
        return result, "claude"

    def _run_claude(
        self, prompt: str, project_dir: str, resume_id: str | None, system_prompt: str = "executor_system.txt"
    ) -> ExecutionResult:
        """Retry API outages with backoff; stop the session on problems only the user can fix."""
        s = self.settings
        for retry in range(s.max_transient_retries + 1):
            result = self.executor.run_task(prompt, project_dir, resume_id, system_prompt=system_prompt)
            kind = classify_failure(result)
            if kind == "fatal":
                raise OrchestratorError(
                    "Claude Code cannot continue (login, credits or usage limit): "
                    f"{truncate(result.result_text or result.stderr, 400)}. Fix it, then run `mmco resume`."
                )
            if kind != "transient" or retry == s.max_transient_retries:
                return result
            wait = s.transient_backoff_seconds * (2 ** retry)
            logger.warning("Claude API unavailable ({}); retrying in {:.0f}s",
                           truncate(result.result_text or result.stderr, 160), wait)
            self._sleep(wait)
        return result  # pragma: no cover

    def _unevaluated_execution(self, task: Task) -> Execution | None:
        """An execution that finished but was never judged (the planner failed or the user pressed Ctrl+C)."""
        last = self.db.last_execution(task.id)
        if last and last.attempt == task.attempts and not self.db.has_evaluation(last.id):
            return last
        return None

    @staticmethod
    def _regression_commands(tasks: list[Task], task: Task) -> list[str]:
        commands: list[str] = []
        for earlier in tasks:
            if earlier.status == "done" and earlier.order_index < task.order_index:
                commands += [c for c in earlier.verify_commands if c not in commands and c not in task.verify_commands]
        return commands

    @staticmethod
    def _apply_guards(evaluation: EvalResult, execution: Execution) -> EvalResult:
        """Hard rules the planner cannot talk its way around."""
        failed = [v for v in execution.verify_results if not v.passed]
        if evaluation.satisfied and failed:
            commands = ", ".join(f"`{v.command}`" for v in failed)
            evaluation = evaluation.model_copy(update={
                "satisfied": False,
                "next_action": "retry" if evaluation.next_action in ("continue", "add_task") else evaluation.next_action,
                "reason": f"[mmco] verification failed ({commands}); planner had said: {evaluation.reason}",
                "feedback_for_executor": evaluation.feedback_for_executor
                or f"These verification commands fail and must pass: {commands}",
            })
        broken = [v for v in execution.regression_results if not v.passed]
        if evaluation.satisfied and broken:
            commands = ", ".join(f"`{v.command}`" for v in broken)
            evaluation = evaluation.model_copy(update={
                "satisfied": False,
                "next_action": "retry",
                "reason": f"[mmco] this change broke earlier tasks ({commands}); planner had said: {evaluation.reason}",
                "feedback_for_executor": (
                    f"Your changes broke checks from earlier tasks that used to pass: {commands}. "
                    "Fix the regression without removing the new functionality.\n"
                    + "\n".join(f"$ {v.command}\n{truncate(v.output, 1500)}" for v in broken)
                ),
            })
        if not evaluation.satisfied and evaluation.next_action == "continue":
            evaluation = evaluation.model_copy(update={"next_action": "retry"})
        if execution.refusal_detected and not evaluation.satisfied and evaluation.next_action == "retry":
            evaluation = evaluation.model_copy(update={"next_action": "reformulate"})
        return evaluation

    def _dispatch(
        self, session: Session, task: Task, execution: Execution, evaluation: EvalResult, workspace: Workspace
    ) -> None:
        s = self.settings
        action = evaluation.next_action
        if evaluation.satisfied and action != "add_task":
            action = "continue"
        if action == "add_task" and not evaluation.new_tasks:
            action = "continue" if evaluation.satisfied else "retry"

        if action == "continue":
            self._complete(task, workspace)
        elif action == "add_task":
            if self.db.count_subtasks(task.id) + len(evaluation.new_tasks) > s.max_subtasks_per_task:
                self._fail(session, task, f"the planner keeps adding tasks for this step: {evaluation.reason}",
                           workspace)
                return
            if evaluation.satisfied:
                self._complete(task, workspace)
                self.db.insert_tasks(session.id, evaluation.new_tasks, task.order_index + 1, task.id)
                logger.info("added {} follow-up task(s)", len(evaluation.new_tasks))
            else:
                self.db.insert_tasks(session.id, evaluation.new_tasks, task.order_index, task.id)
                task.status = "pending"
                task.last_feedback = evaluation.feedback_for_executor or evaluation.reason
                task.resume_claude_session_id = None
                self.db.save_task(task)
                logger.info("added {} prerequisite task(s); this task runs again after them",
                            len(evaluation.new_tasks))
        elif action == "take_over":
            # With the planner-executor kill-switch off, take_over must not run the planner with a
            # write-enabled toolbox; downgrade it to a normal Claude retry with the reviewer's feedback.
            if self.settings.allow_planner_executor:
                self._take_over(session, task, workspace)
            else:
                self._retry(session, task, execution, evaluation, workspace)
        elif action == "retry":
            self._retry(session, task, execution, evaluation, workspace)
        elif action == "reformulate":
            self._reformulate(session, task, evaluation, workspace)
        elif action == "clarify":
            questions = self.planner.clarify(
                session, evaluation.reason, task, self.db.list_clarifications(session.id, True, kind="question")
            )
            task.status = "pending"
            task.last_feedback = evaluation.feedback_for_executor or evaluation.reason
            task.resume_claude_session_id = None
            self.db.save_task(task)
            self.db.add_questions(session.id, task.id, questions)
            self._answer_pending(session)
        elif action == "fail":
            self._fail(session, task, evaluation.reason, workspace)

    def _complete(self, task: Task, workspace: Workspace) -> None:
        task.end_commit = workspace.commit(f"mmco: {task.label}")
        task.status = "done"
        task.last_feedback = None
        task.resume_claude_session_id = None
        self.db.save_task(task)
        logger.success("✔ task '{}' done", task.label)

    def _take_over(self, session: Session, task: Task, workspace: Workspace) -> None:
        """The planner finishes the task itself, building on the coding agent's on-disk work.

        A per-task hand-off counter stops the two executors from ping-ponging forever: this method →
        _execute_as → _evaluate_and_dispatch → _dispatch may recurse back here on another take_over, but
        the mutual recursion is bounded by max_handoffs_per_task. handoffs is a conservative lifetime cap
        and is intentionally NOT reset on reformulation.
        """
        task.handoffs += 1
        self.db.save_task(task)
        if task.handoffs > self.settings.max_handoffs_per_task:
            self._fail(session, task, "executors kept handing the task back and forth", workspace)
            return
        logger.info("planner is taking over '{}' to finish it (hand-off {})", task.label, task.handoffs)
        tasks = self.db.list_tasks(session.id)
        # Build on whatever the coding agent left on disk; never reset before a take_over.
        execution = self._execute_as(session, task, tasks, workspace, "planner", reset_for_planner_retry=False)
        self._evaluate_and_dispatch(session, task, execution, workspace)

    def _retry(
        self, session: Session, task: Task, execution: Execution, evaluation: EvalResult, workspace: Workspace
    ) -> None:
        if task.cycle_attempts >= self.settings.max_attempts_per_task:
            if task.reformulations < self.settings.max_reformulations_per_task:
                logger.info("attempt budget used up; escalating to reformulation")
                self._reformulate(session, task, evaluation, workspace)
            else:
                self._fail(session, task, evaluation.reason, workspace)
            return
        result = execution.result
        task.status = "pending"
        task.last_feedback = evaluation.feedback_for_executor or evaluation.reason
        # Continue the same Claude conversation with the reviewer's feedback, unless that run broke.
        task.resume_claude_session_id = (
            result.claude_session_id if result.claude_session_id and not result.is_error else None
        )
        self.db.save_task(task)

    def _reformulate(self, session: Session, task: Task, evaluation: EvalResult, workspace: Workspace) -> None:
        if task.reformulations >= self.settings.max_reformulations_per_task:
            self._fail(session, task, evaluation.reason, workspace)
            return
        logger.info("reformulating the prompt for '{}'", task.label)
        previous = task.prompt_override or self._task_block(task)
        new_prompt = self.planner.reformulate(session, task, previous, evaluation.reason)
        workspace.reset_to(task.start_commit)
        task.prompt_override = new_prompt
        task.reformulations += 1
        task.cycle_attempts = 0
        task.last_feedback = None
        task.resume_claude_session_id = None
        task.status = "pending"
        self.db.save_task(task)

    def _fail(self, session: Session, task: Task, reason: str, workspace: Workspace) -> None:
        if self.settings.escalate_to_user:
            self._escalate(session, task, reason)
        else:
            self._hard_fail(session, task, reason)

    def _escalate(self, session: Session, task: Task, reason: str) -> None:
        task.status = "pending"
        task.resume_claude_session_id = None
        self.db.save_task(task)
        logger.warning("task '{}' is stuck; asking the user how to proceed", task.label)
        question = (
            f"Task '{task.label}' is stuck after {task.attempts} attempt(s) and {task.reformulations} "
            f"reformulation(s).\nLast problem: {truncate(reason, 800)}\n"
            "Type guidance for the next attempt (or press Enter to simply try again), "
            "'skip' to leave this task out, or 'abort' to stop."
        )
        self.db.add_questions(session.id, task.id, [question], kind="escalation")
        self._answer_pending(session)

    def _hard_fail(self, session: Session, task: Task, reason: str) -> None:
        task.status = "failed"
        task.resume_claude_session_id = None
        self.db.save_task(task)
        logger.error("✘ task '{}' failed: {}", task.label, reason)
        if self.settings.stop_on_task_failure:
            blocked = self.db.block_pending_tasks(session.id)
            if blocked:
                logger.error("{} remaining task(s) blocked", blocked)

    # ---- final review ---------------------------------------------------

    def _final_review(self, session: Session, workspace: Workspace) -> bool:
        """Compare the finished project with the request. Returns True if it queued more work."""
        s = self.settings
        reviews = int(session.metadata.get("final_reviews", 0))
        tasks = self.db.list_tasks(session.id)
        if not s.final_review or reviews >= s.max_final_reviews or any(t.status == "failed" for t in tasks):
            return False
        logger.info("all tasks finished; running every check and reviewing the result against the request")
        commands: list[str] = []
        for task in tasks:
            if task.status == "done":
                commands += [c for c in task.verify_commands if c not in commands]
        checks = run_verify_commands(commands, session.project_dir, s.verify_timeout_seconds)
        reports = {}
        for task in tasks:
            last = self.db.last_execution(task.id)
            if last:
                reports[task.id] = last.result.result_text
        review = self.planner.review(
            session, workspace.listing(), tasks, reports, checks,
            self.db.list_clarifications(session.id, True, kind="question"),
        )
        self._set_status(session, "executing", final_reviews=reviews + 1)
        failing = [c for c in checks if not c.passed]
        new_tasks = list(review.new_tasks)
        if failing and not new_tasks:
            new_tasks = [TaskSpec(
                title="Fix failing checks",
                description="These checks passed when their tasks finished but fail now. Find the cause and fix "
                            "it without removing functionality.\n\n"
                            + "\n\n".join(f"$ {c.command}\n{truncate(c.output, 1500)}" for c in failing),
                verify_commands=[c.command for c in failing],
            )]
        if review.complete and not failing:
            logger.success("final review: {}", review.reason or "the request is fully implemented")
            return False
        if not new_tasks:
            logger.warning("final review found gaps but proposed no tasks: {}", review.reason)
            return False
        self.db.insert_tasks(session.id, new_tasks)
        logger.info("final review added {} task(s): {}", len(new_tasks), review.reason)
        return True

    # ---- questions ------------------------------------------------------

    def _answer_pending(self, session: Session) -> bool:
        pending = self.db.list_clarifications(session.id, answered=False)
        if not pending:
            return True
        if self.ask_user is None:
            self._set_status(session, "awaiting_input")
            logger.warning("waiting for your input on {} question(s); run `mmco resume {}`", len(pending), session.id)
            return False
        answers = self.ask_user([c.question for c in pending])
        for clarification, answer in zip(pending, answers):
            self._store_answer(session, clarification, answer)
        if session.status == "awaiting_input":
            self._set_status(session, "executing")
        return True

    def _store_answer(self, session: Session, clarification: Any, answer: str) -> None:
        self.db.answer_clarification(clarification.id, answer)
        if clarification.kind == "escalation" and clarification.task_id:
            self._apply_escalation_answer(session, clarification.task_id, answer)

    def _apply_escalation_answer(self, session: Session, task_id: str, answer: str) -> None:
        task = self.db.get_task(task_id)
        text = (answer or "").strip()
        choice = text.lower()
        if choice in SKIP_ANSWERS:
            logger.warning("skipping task '{}' at the user's request", task.label)
            if self._workspace is not None:
                self._workspace.reset_to(task.start_commit)
            task.status = "skipped"
            self.db.save_task(task)
            return
        if choice in ABORT_ANSWERS:
            self._hard_fail(session, task, "aborted by the user")
            return
        task.cycle_attempts = 0
        task.reformulations = 0
        task.resume_claude_session_id = None
        if choice not in RETRY_ANSWERS:
            task.last_feedback = f"Guidance from the user (follow it): {text}"
        self.db.save_task(task)
        logger.info("retrying task '{}' with a fresh budget", task.label)

    # ---- prompts --------------------------------------------------------

    def _prompt(self, session: Session, name: str) -> str:
        return load_prompt(name, *prompt_dirs(session.project_dir, self.settings.prompts_dir))

    def _executor_context(self, session: Session) -> str:
        return load_rules(session.project_dir).executor_context or self.settings.executor_context

    def _task_prompt(self, session: Session, task: Task, tasks: list[Task], context: str = "full") -> str:
        feedback = f"\n# Feedback on the previous attempt\n{task.last_feedback}\n" if task.last_feedback else ""
        verify = bullets(task.verify_commands) or "- (none: re-read your changes carefully instead)"
        if context == "minimal":
            # Only what the planner wrote: no goal, no plan, no user discussion, no file hints.
            return render(
                self._prompt(session, "executor_task_minimal.txt"),
                task_block=task.prompt_override or self._task_block(task, include_files=False),
                verify_commands=verify,
                feedback=feedback,
            )
        if context == "task":
            return render(
                self._prompt(session, "executor_task_isolated.txt"),
                task_block=task.prompt_override or self._task_block(task),
                verify_commands=verify,
                feedback=feedback,
            )
        overview = []
        for index, t in enumerate(tasks, 1):
            marker = "→" if t.id == task.id else ("✓" if t.status == "done" else " ")
            overview.append(f"{marker} {index}. {t.label}")
        clarifications = self.db.list_clarifications(session.id, True, kind="question")
        return render(
            self._prompt(session, "executor_task.txt"),
            goal=session.original_request,
            plan_overview="\n".join(overview),
            task_block=task.prompt_override or self._task_block(task),
            verify_commands=verify,
            clarifications=(
                "\n# Answers from the user\n"
                + "\n".join(f"Q: {c.question}\nA: {c.answer}" for c in clarifications) + "\n"
            ) if clarifications else "",
            feedback=feedback,
        )

    @staticmethod
    def _task_block(task: Task, include_files: bool = True) -> str:
        parts = [f"## {task.label}", task.description.strip()]
        if task.interface_contracts:
            contracts = task.interface_contracts
            if not isinstance(contracts, str):
                contracts = json.dumps(contracts, indent=2, ensure_ascii=False)
            parts += ["## Interface contracts (use these names and shapes exactly)", contracts]
        if task.acceptance_criteria:
            parts += ["## Acceptance criteria", bullets(task.acceptance_criteria)]
        if include_files and task.files_involved:
            parts += ["## Files likely involved", bullets(task.files_involved)]
        return "\n\n".join(parts)

    def _retry_prompt(self, session: Session, task: Task) -> str:
        last = self.db.last_execution(task.id)
        checks = (last.verify_results + last.regression_results) if last else []
        report = "\n\n".join(
            f"$ {v.command}\nexit code: {v.exit_code}{' (timed out)' if v.timed_out else ''}\n"
            f"{truncate(v.output, 2000)}"
            for v in checks
        )
        return render(
            self._prompt(session, "executor_retry.txt"),
            feedback=task.last_feedback or "",
            verify_report=report or "(no verification commands for this task)",
        )

    def _previous_claude_session(self, tasks: list[Task], task: Task) -> str | None:
        earlier = [t for t in tasks if t.order_index < task.order_index and t.status == "done"]
        if not earlier:
            return None
        last = self.db.last_execution(earlier[-1].id)
        return last.result.claude_session_id if last else None

    # ---- bookkeeping ----------------------------------------------------

    def _set_status(self, session: Session, status: SessionStatus, **metadata: Any) -> None:
        session.status = status
        session.metadata.update(metadata)
        self.db.update_session(session)

    def _finish(self, session: Session) -> Session:
        summary = self.db.summary(session.id)
        failed = summary["failed"] or summary["blocked"]
        self._set_status(session, "failed" if failed else "done", summary=summary)
        log = logger.error if failed else logger.success
        log("session {} {}: {} done, {} skipped, {} failed, {} blocked", session.id, session.status,
            summary["done"], summary["skipped"], summary["failed"], summary["blocked"])
        return session

    def _crash_state(self, session: Session, exc: Exception) -> dict[str, Any]:
        state: dict[str, Any] = {"error": repr(exc), "session": session.model_dump(mode="json")}
        try:
            state["tasks"] = [t.model_dump(mode="json") for t in self.db.list_tasks(session.id)]
        except sqlite3.Error as read_exc:
            state["tasks_unavailable"] = repr(read_exc)
        return state
