"""Pydantic schemas shared by the planner, executor, orchestrator and database."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SessionStatus = Literal["planning", "executing", "awaiting_input", "awaiting_approval", "paused", "done", "failed", "cancelled"]
TaskStatus = Literal["pending", "in_progress", "done", "failed", "blocked", "skipped"]
NextAction = Literal["continue", "retry", "reformulate", "add_task", "clarify", "fail"]
PlannerPurpose = Literal["plan", "eval", "reformulate", "clarify", "review", "choose_executor", "execute"]
ClarificationKind = Literal["question", "escalation"]
AgentName = Literal["claude", "planner"]


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaskSpec(BaseModel):
    """A task as produced by the planner."""

    order_index: int = 0
    title: str = ""
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    files_involved: list[str] = Field(default_factory=list)
    interface_contracts: Any = Field(default_factory=dict)
    verify_commands: list[str] = Field(default_factory=list)
    needs_continuity: bool = False

    @field_validator("acceptance_criteria", "files_involved", "verify_commands", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("interface_contracts", mode="before")
    @classmethod
    def _coerce_contracts(cls, value: Any) -> Any:
        return {} if value is None else value

    @property
    def label(self) -> str:
        if self.title:
            return self.title
        first_line = self.description.strip().splitlines()[0] if self.description.strip() else ""
        return first_line[:70]


class PlanResponse(BaseModel):
    tasks: list[TaskSpec] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)


class Task(TaskSpec):
    id: str = Field(default_factory=new_id)
    session_id: str
    parent_task_id: str | None = None
    status: TaskStatus = "pending"
    attempts: int = 0
    cycle_attempts: int = 0  # attempts since the last reformulation
    reformulations: int = 0
    executor: AgentName = "claude"
    prompt_override: str | None = None
    last_feedback: str | None = None
    resume_claude_session_id: str | None = None
    start_commit: str | None = None
    end_commit: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Session(BaseModel):
    id: str = Field(default_factory=new_id)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    original_request: str
    project_dir: str
    status: SessionStatus = "planning"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """One Claude Code run, parsed from `claude -p --output-format json`."""

    prompt: str
    raw_output: dict[str, Any] | None = None
    result_text: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: int = 0
    timed_out: bool = False
    is_error: bool = False
    subtype: str | None = None
    stop_reason: str | None = None
    refusal_suspected: bool = False
    permission_denials: list[Any] = Field(default_factory=list)
    claude_session_id: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None


class VerifyResult(BaseModel):
    command: str
    exit_code: int | None
    output: str = ""
    timed_out: bool = False
    duration_ms: int = 0

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Execution(BaseModel):
    id: str = Field(default_factory=new_id)
    task_id: str
    session_id: str
    attempt: int
    agent: AgentName = "claude"
    result: ExecutionResult
    refusal_detected: bool = False
    diff_stat: str = ""
    diff: str = ""
    verify_results: list[VerifyResult] = Field(default_factory=list)
    regression_results: list[VerifyResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class EvalResult(BaseModel):
    satisfied: bool
    reason: str = ""
    next_action: NextAction
    feedback_for_executor: str = ""
    new_tasks: list[TaskSpec] = Field(default_factory=list)
    updated_verify_commands: list[str] | None = None

    @field_validator("updated_verify_commands", mode="before")
    @classmethod
    def _empty_means_unchanged(cls, value: Any) -> Any:
        # The planner may fix a broken check, but never remove verification altogether.
        return value or None

    @field_validator("next_action", mode="before")
    @classmethod
    def _normalise_action(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("new_tasks", mode="before")
    @classmethod
    def _coerce_tasks(cls, value: Any) -> Any:
        return [] if value is None else value


class ReviewResult(BaseModel):
    complete: bool
    reason: str = ""
    new_tasks: list[TaskSpec] = Field(default_factory=list)

    @field_validator("new_tasks", mode="before")
    @classmethod
    def _coerce_tasks(cls, value: Any) -> Any:
        return [] if value is None else value


class ExecutorChoice(BaseModel):
    """The planner's decision about who should execute a task."""

    executor: AgentName = "claude"
    reason: str = ""

    @field_validator("executor", mode="before")
    @classmethod
    def _normalise_executor(cls, value: Any) -> Any:
        # Anything unexpected falls back to Claude, the safe default.
        if isinstance(value, str) and value.strip().lower() in ("planner", "claude"):
            return value.strip().lower()
        return "claude"


class ClarifyResponse(BaseModel):
    questions: list[str] = Field(default_factory=list)


class Clarification(BaseModel):
    id: str = Field(default_factory=new_id)
    session_id: str
    task_id: str | None = None
    kind: ClarificationKind = "question"
    question: str
    answer: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    answered_at: datetime | None = None
