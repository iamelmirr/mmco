"""Planner: an OpenAI-compatible chat model (DeepSeek via OpenRouter by default)."""

from __future__ import annotations

import json
import time
from typing import Any, TypeVar

import openai
from loguru import logger
from pydantic import BaseModel, ValidationError

from .config import Settings
from .db import Database
from .models import (
    Clarification,
    ClarifyResponse,
    EvalResult,
    Execution,
    ExecutorChoice,
    PlannerPurpose,
    PlanResponse,
    ReviewResult,
    Session,
    Task,
    VerifyResult,
)
from .projectmap import build_map
from .rules import load_rules, prompt_dirs
from .tools import Toolbox
from .utils import MMCOError, extract_json, load_prompt, truncate

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Retrying these cannot succeed: bad key, bad model id, unsupported parameter.
_FATAL_API_ERRORS = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.BadRequestError,
)


class PlannerError(MMCOError):
    pass


def compose_system_prompt(prompt_file: str, project_dir: str | None, settings: Settings) -> str:
    """A planner system prompt with the user's rules appended. Re-read on every call."""
    system = load_prompt(prompt_file, *prompt_dirs(project_dir, settings.prompts_dir))
    rules = load_rules(project_dir)
    if rules.text:
        system += (
            "\n\nUSER RULES\nThe user who runs this pipeline set the following rules. Follow them; when they "
            "conflict with the instructions above, the rules win, except for the required output format.\n\n"
            + rules.text
        )
    return system


class Planner:
    def __init__(self, settings: Settings, db: Database, client: Any = None, sleep=time.sleep):
        self.settings = settings
        self.db = db
        self._sleep = sleep
        if client is None:
            if not settings.openrouter_api_key:
                raise PlannerError("OPENROUTER_API_KEY is not set: run `mmco setup` or add it to ~/.config/mmco/.env")
            client = openai.OpenAI(
                api_key=settings.openrouter_api_key,
                base_url=settings.planner_base_url,
                timeout=settings.planner_timeout_seconds,
                max_retries=0,
                default_headers={"X-Title": "mmco"},
            )
        self.client = client

    # ---- public API -----------------------------------------------------

    def plan(
        self,
        session: Session,
        project_files: str,
        clarifications: list[Clarification],
        read_tools: bool = False,
        toolbox: "Toolbox | None" = None,
    ) -> PlanResponse:
        payload = {
            "request": session.original_request,
            "project_files": project_files or "(empty directory)",
            "project_map": build_map(session.project_dir),
            "previous_requests_in_this_project": session.metadata.get("previous_requests", []),
            "executor_context": self.executor_context(session),
            "user_clarifications": _qa(clarifications),
        }
        if read_tools:
            tb = toolbox or Toolbox(session.project_dir, allow_write=False)
            return self._call_json_with_tools(
                session, "plan", "planner_plan.txt", payload, PlanResponse, tb, allow_write=False
            )
        return self._call_json(session, "plan", "planner_plan.txt", payload, PlanResponse)

    def evaluate(
        self,
        session: Session,
        task: Task,
        execution: Execution,
        history: list[dict[str, Any]],
        clarifications: list[Clarification],
        read_tools: bool = False,
        toolbox: "Toolbox | None" = None,
    ) -> EvalResult:
        s = self.settings
        r = execution.result
        payload = {
            "overall_goal": session.original_request,
            "task": _task_payload(task),
            "attempt": {
                "number": execution.attempt,
                "attempts_since_reformulation": task.cycle_attempts,
                "max_attempts_per_prompt": s.max_attempts_per_task,
                "reformulations_so_far": task.reformulations,
                "max_reformulations": s.max_reformulations_per_task,
            },
            "executor": {
                "final_message": truncate(r.result_text, s.max_output_chars),
                "is_error": r.is_error,
                "subtype": r.subtype,
                "stop_reason": r.stop_reason,
                "timed_out": r.timed_out,
                "exit_code": r.exit_code,
                "num_turns": r.num_turns,
                "permission_denials": r.permission_denials[:10],
                "refusal_detected": execution.refusal_detected,
                "stderr": truncate(r.stderr, 2000),
            },
            "changes": {
                "diff_stat": execution.diff_stat or "(no files changed)",
                "diff": execution.diff or "",
            },
            "verification": [
                {
                    "command": v.command,
                    "passed": v.passed,
                    "exit_code": v.exit_code,
                    "timed_out": v.timed_out,
                    "output": truncate(v.output, 3000),
                }
                for v in execution.verify_results
            ],
            "regression_checks": [
                {"command": v.command, "passed": v.passed, "output": truncate(v.output, 1500)}
                for v in execution.regression_results
            ],
            "previous_evaluations": [
                {"attempt": h.get("attempt"), "next_action": h["next_action"], "reason": h["reason"]}
                for h in history
            ],
            "user_clarifications": _qa(clarifications),
        }
        if read_tools:
            tb = toolbox or Toolbox(session.project_dir, allow_write=False)
            return self._call_json_with_tools(
                session, "eval", "planner_eval.txt", payload, EvalResult, tb, allow_write=False, task_id=task.id
            )
        return self._call_json(session, "eval", "planner_eval.txt", payload, EvalResult, task_id=task.id)

    def review(
        self,
        session: Session,
        project_files: str,
        tasks: list[Task],
        reports: dict[str, str],
        checks: list[VerifyResult],
        clarifications: list[Clarification],
    ) -> ReviewResult:
        payload = {
            "request": session.original_request,
            "user_clarifications": _qa(clarifications),
            "project_files": project_files,
            "tasks": [
                {
                    "title": t.label,
                    "status": t.status,
                    "description": truncate(t.description, 800),
                    "acceptance_criteria": t.acceptance_criteria,
                    "agent_report": truncate(reports.get(t.id, ""), 1500),
                }
                for t in tasks
            ],
            "checks": [
                {"command": c.command, "passed": c.passed, "output": truncate(c.output, 1500)} for c in checks
            ],
        }
        return self._call_json(session, "review", "planner_review.txt", payload, ReviewResult)

    def reformulate(self, session: Session, task: Task, previous_prompt: str, failure_reason: str) -> str:
        payload = {
            "overall_goal": session.original_request,
            "task": _task_payload(task),
            "previous_prompt": truncate(previous_prompt, self.settings.max_output_chars),
            "failure_reason": failure_reason,
            "executor_context": self.executor_context(session),
        }
        messages = self._messages(session, "planner_reformulate.txt", payload)
        text = self._request(session.id, "reformulate", messages, json_mode=False, task_id=task.id).strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.strip("`").removeprefix("markdown").removeprefix("text").strip()
        if not text:
            raise PlannerError("planner returned an empty reformulation")
        return text

    def clarify(
        self, session: Session, blocker: str, task: Task | None, clarifications: list[Clarification]
    ) -> list[str]:
        payload = {
            "original_request": session.original_request,
            "task": _task_payload(task) if task else None,
            "blocker": blocker,
            "already_answered": _qa(clarifications),
        }
        response = self._call_json(
            session, "clarify", "planner_clarify.txt", payload, ClarifyResponse, task_id=task.id if task else None
        )
        questions = [q.strip() for q in response.questions if q.strip()][:5]
        return questions or [f"The pipeline is blocked: {blocker}. How should it proceed?"]

    def choose_executor(self, session: Session, task: Task, toolbox: "Toolbox") -> tuple[str, str]:
        """Decide whether the planner or Claude should execute ``task``.

        Returns ``(executor, reason)`` with executor "planner" or "claude" (anything
        unexpected is normalised to "claude").
        """
        payload = {
            "task": _task_payload(task),
            "project_map": build_map(session.project_dir),
        }
        choice = self._call_json_with_tools(
            session, "choose_executor", "planner_choose_executor.txt", payload, ExecutorChoice,
            toolbox, allow_write=False, task_id=task.id,
        )
        return choice.executor, choice.reason

    def execute_task(
        self, session: Session, task: Task, toolbox: "Toolbox", building_on_work: bool = False
    ) -> str:
        """Execute ``task`` directly via the write-enabled tool loop; return a short report.

        ``building_on_work`` is True when the working tree already holds prior work to build on (a
        take_over, or a first attempt after earlier tasks) and False when it was just rolled back to a
        clean checkpoint (a planner retry). It only affects how the starting point is described.
        """
        payload = {
            "task": _task_payload(task),
            "project_map": build_map(session.project_dir),
            "starting_point": (
                "the current working tree, which already contains prior work you should build on"
                if building_on_work
                else "a clean checkpoint (your previous attempt was rolled back)"
            ),
        }
        if task.last_feedback:
            payload["feedback_on_previous_attempt"] = task.last_feedback
        messages = self._messages(session, "planner_execute.txt", payload)
        report = self._request_with_tools(
            session, "execute", messages, toolbox, allow_write=True, task_id=task.id, json_mode=False,
        ).strip()
        if not report:
            raise PlannerError("planner returned an empty execution report")
        return report

    # ---- plumbing -------------------------------------------------------

    def executor_context(self, session: Session) -> str:
        return load_rules(session.project_dir).executor_context or self.settings.executor_context

    def _messages(self, session: Session, prompt_file: str, payload: dict[str, Any]) -> list[dict[str, str]]:
        # Rules and prompt overrides are re-read on every call, so edits apply to a running session.
        system = compose_system_prompt(prompt_file, session.project_dir, self.settings)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, indent=2, ensure_ascii=False, default=str)},
        ]

    def _call_json(
        self,
        session: Session,
        purpose: PlannerPurpose,
        prompt_file: str,
        payload: dict[str, Any],
        schema: type[SchemaT],
        task_id: str | None = None,
    ) -> SchemaT:
        """Ask for JSON; on parse/validation failure, show the model its error and ask again."""
        session_id = session.id
        messages = self._messages(session, prompt_file, payload)
        return self._parse_json_with_retries(
            purpose,
            messages,
            schema,
            lambda msgs: self._request(session_id, purpose, msgs, json_mode=True, task_id=task_id),
        )

    def _call_json_with_tools(
        self,
        session: Session,
        purpose: PlannerPurpose,
        prompt_file: str,
        payload: dict[str, Any],
        schema: type[SchemaT],
        toolbox: Toolbox,
        allow_write: bool,
        task_id: str | None = None,
    ) -> SchemaT:
        """Like ``_call_json`` but the model may call tools before answering with JSON."""
        messages = self._messages(session, prompt_file, payload)
        return self._parse_json_with_retries(
            purpose,
            messages,
            schema,
            lambda msgs: self._request_with_tools(session, purpose, msgs, toolbox, allow_write, task_id=task_id),
        )

    def _parse_json_with_retries(
        self,
        purpose: PlannerPurpose,
        messages: list[dict[str, Any]],
        schema: type[SchemaT],
        produce,
    ) -> SchemaT:
        """Shared retry/parse loop: ``produce(messages)`` yields the model's text answer."""
        last_error: Exception | None = None
        for attempt in range(1, self.settings.planner_max_retries + 1):
            content = produce(messages)
            try:
                return schema.model_validate(extract_json(content))
            except (ValueError, ValidationError) as exc:
                last_error = exc
                logger.warning("planner {} returned invalid JSON (attempt {}): {}", purpose, attempt, str(exc)[:300])
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": f"That response was invalid: {truncate(str(exc), 1500)}\n"
                        "Reply again with ONLY a JSON object that matches the schema in the instructions.",
                    },
                ]
        raise PlannerError(
            f"planner returned invalid JSON for '{purpose}' after {self.settings.planner_max_retries} attempts: "
            f"{truncate(str(last_error), 500)}"
        )

    def _request_with_tools(
        self,
        session: Session,
        purpose: PlannerPurpose,
        messages: list[dict[str, Any]],
        toolbox: Toolbox,
        allow_write: bool,
        task_id: str | None = None,
        json_mode: bool = True,
    ) -> str:
        """Run a tool-calling loop, then return the model's final text answer.

        Each iteration sends the conversation with the toolbox schemas attached. Tool calls are
        dispatched and their results fed back. When the model stops calling tools we return its
        answer; when the budget is exhausted we ask once more, without tools, for the answer.
        """
        s = self.settings
        messages = list(messages)
        for _ in range(s.planner_max_tool_calls):
            kwargs: dict[str, Any] = {
                "model": s.planner_model,
                "messages": messages,
                "temperature": s.planner_temperature,
                "tools": toolbox.schemas(allow_write),
                "tool_choice": "auto",
            }
            message = self._create_and_persist(session.id, purpose, kwargs, task_id)
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                return message.content or ""
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = toolbox.dispatch(tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # Budget exhausted: force a final answer without any further tool calls.
        nudge = (
            "Tool budget exhausted. Answer now with the required JSON, no more tool calls."
            if json_mode
            else "Tool budget exhausted. Provide your final report now; do not call any more tools."
        )
        messages.append({"role": "user", "content": nudge})
        kwargs = {"model": s.planner_model, "messages": messages, "temperature": s.planner_temperature}
        if json_mode and s.planner_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        message = self._create_and_persist(session.id, purpose, kwargs, task_id)
        content = message.content or ""
        # Fail fast on a genuinely empty answer: returning "" would make extract_json fail and rerun
        # this whole tool loop up to planner_max_retries times. If the model still tried to call tools,
        # hand back its (empty) text so the JSON-retry loop can prompt it once more for a plain answer.
        if not content.strip() and not getattr(message, "tool_calls", None):
            raise PlannerError(f"planner '{purpose}' did not answer after exhausting its tool budget")
        return content

    def _create_and_persist(
        self, session_id: str, purpose: PlannerPurpose, kwargs: dict[str, Any], task_id: str | None
    ) -> Any:
        """One completion with backoff on transient API errors, persisted like ``_request``.

        Fatal errors (bad key / model / request) raise PlannerError immediately; transient ones
        (rate limit, timeout, connection, server) are retried up to ``planner_max_retries``.
        """
        s = self.settings
        delay = 2.0
        for attempt in range(1, s.planner_max_retries + 1):
            started = time.monotonic()
            try:
                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
            except Exception as exc:  # noqa: BLE001 - every failure is logged; fatal ones re-raised below
                elapsed = int((time.monotonic() - started) * 1000)
                self.db.add_planner_call(
                    session_id, purpose, s.planner_model, kwargs, None, elapsed, task_id, error=repr(exc)
                )
                if isinstance(exc, _FATAL_API_ERRORS):
                    hint = ""
                    if isinstance(exc, openai.BadRequestError) and "response_format" in kwargs:
                        hint = " (if the model does not support JSON mode, set MMCO_PLANNER_JSON_MODE=false)"
                    raise PlannerError(f"planner API call failed: {exc}{hint}") from exc
                if attempt == s.planner_max_retries:
                    raise PlannerError(f"planner API call failed after {attempt} attempts: {exc}") from exc
                logger.warning("planner {} call failed ({}); retrying in {:.0f}s", purpose, exc, delay)
                self._sleep(delay)
                delay *= 2
                continue

            elapsed = int((time.monotonic() - started) * 1000)
            content = message.content or ""
            cost = _usage_cost(response)
            self.db.add_planner_call(
                session_id, purpose, s.planner_model, kwargs, content, elapsed, task_id, cost_usd=cost
            )
            logger.bind(event="planner_call", purpose=purpose, request=kwargs["messages"], response=content).debug(
                "planner {} call finished in {:.1f}s", purpose, elapsed / 1000
            )
            return message
        raise PlannerError("unreachable")  # pragma: no cover

    def _request(
        self,
        session_id: str,
        purpose: PlannerPurpose,
        messages: list[dict[str, str]],
        json_mode: bool,
        task_id: str | None = None,
    ) -> str:
        """One completion with exponential backoff on transient API errors. Every call is persisted."""
        s = self.settings
        kwargs: dict[str, Any] = {"model": s.planner_model, "messages": messages, "temperature": s.planner_temperature}
        if json_mode and s.planner_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        delay = 2.0
        for attempt in range(1, s.planner_max_retries + 1):
            started = time.monotonic()
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - every failure is logged, fatal ones re-raised below
                elapsed = int((time.monotonic() - started) * 1000)
                self.db.add_planner_call(
                    session_id, purpose, s.planner_model, kwargs, None, elapsed, task_id, error=repr(exc)
                )
                if isinstance(exc, _FATAL_API_ERRORS):
                    hint = ""
                    if isinstance(exc, openai.BadRequestError) and "response_format" in kwargs:
                        hint = " (if the model does not support JSON mode, set MMCO_PLANNER_JSON_MODE=false)"
                    raise PlannerError(f"planner API call failed: {exc}{hint}") from exc
                if attempt == s.planner_max_retries:
                    raise PlannerError(f"planner API call failed after {attempt} attempts: {exc}") from exc
                logger.warning("planner {} call failed ({}); retrying in {:.0f}s", purpose, exc, delay)
                self._sleep(delay)
                delay *= 2
                continue

            elapsed = int((time.monotonic() - started) * 1000)
            cost = _usage_cost(response)
            self.db.add_planner_call(
                session_id, purpose, s.planner_model, kwargs, content, elapsed, task_id, cost_usd=cost
            )
            logger.bind(event="planner_call", purpose=purpose, request=messages, response=content).debug(
                "planner {} call finished in {:.1f}s", purpose, elapsed / 1000
            )
            if content.strip():
                return content
            if attempt == s.planner_max_retries:
                raise PlannerError(f"planner returned empty content for '{purpose}'")
            logger.warning("planner {} returned empty content; retrying", purpose)
            self._sleep(delay)
            delay *= 2
        raise PlannerError("unreachable")  # pragma: no cover


def _usage_cost(response: Any) -> float | None:
    usage = getattr(response, "usage", None)
    cost = getattr(usage, "cost", None) if usage is not None else None
    try:
        return float(cost) if cost is not None else None
    except (TypeError, ValueError):
        return None


def _qa(clarifications: list[Clarification]) -> list[dict[str, str | None]]:
    return [{"question": c.question, "answer": c.answer} for c in clarifications]


def _task_payload(task: Task) -> dict[str, Any]:
    return {
        "title": task.title,
        "description": task.description,
        "acceptance_criteria": task.acceptance_criteria,
        "files_involved": task.files_involved,
        "interface_contracts": task.interface_contracts,
        "verify_commands": task.verify_commands,
        "current_prompt_override": task.prompt_override,
    }
