"""Runtime configuration from environment variables and .env."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def config_dir() -> Path:
    """Global configuration directory (rules, prompt overrides, API key)."""
    return Path(os.environ.get("MMCO_CONFIG_DIR", Path.home() / ".config" / "mmco")).expanduser()


GLOBAL_ENV_FILE = config_dir() / ".env"


def global_env_file() -> Path:
    return config_dir() / ".env"


def load_settings() -> Settings:
    """Settings from the current global config dir (honours MMCO_CONFIG_DIR at call time)."""
    return Settings(_env_file=(global_env_file(), ".env"))


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def write_env_file(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")
    path.chmod(0o600)


class Settings(BaseSettings):
    # The global file holds the API key for use from any folder; a local .env overrides it.
    model_config = SettingsConfigDict(
        env_file=(GLOBAL_ENV_FILE, ".env"), env_prefix="MMCO_", extra="ignore", populate_by_name=True
    )

    openrouter_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "MMCO_OPENROUTER_API_KEY", "openrouter_api_key"),
    )
    planner_model: str = "deepseek/deepseek-v4.1-flash"
    planner_base_url: str = "https://openrouter.ai/api/v1"
    planner_temperature: float = 0.2
    planner_json_mode: bool = True
    planner_max_retries: int = 3
    planner_max_tool_calls: int = 25
    planner_timeout_seconds: int = 180

    claude_binary: str = "claude"
    claude_model: str | None = "opus"
    claude_allowed_tools: str = "Read,Write,Edit,Bash,Glob,Grep,WebFetch,WebSearch"
    claude_permission_mode: str = "bypassPermissions"
    claude_timeout_seconds: int = 1800
    # Default for how much Claude is told; a rules.md front matter overrides it (full | task | minimal).
    executor_context: str = "full"
    claude_max_budget_usd_per_task: float | None = None
    # Let the planner (DeepSeek) execute suitable tasks itself instead of always dispatching Claude.
    allow_planner_executor: bool = True

    max_attempts_per_task: int = 3
    max_reformulations_per_task: int = 2
    max_subtasks_per_task: int = 5
    # How many times executors may hand a single task back and forth (planner take_over guard).
    max_handoffs_per_task: int = 3
    max_loop_iterations: int = 100
    max_session_cost_usd: float | None = None
    stop_on_task_failure: bool = True
    verify_timeout_seconds: int = 300
    # Instead of failing a stuck task, ask the user for guidance (or skip / abort).
    escalate_to_user: bool = True
    # When every task is done, rerun all checks and let the planner compare the result with the request.
    final_review: bool = True
    max_final_reviews: int = 2
    # After each task, rerun the verify commands of earlier tasks.
    regression_checks: bool = True
    # Claude API overload / rate-limit errors are retried without spending the task's attempts.
    max_transient_retries: int = 4
    transient_backoff_seconds: float = 30

    max_diff_chars: int = 20_000
    max_output_chars: int = 12_000

    # Stream what the planner and executor are doing live (reasoning, tool calls, results) into a
    # per-session event log that `mmco logs -f` and the dashboard show. Off = plain non-streaming calls.
    stream_logs: bool = True

    git_checkpoints: bool = True
    db_path: str = "~/.mmco/mmco.db"
    log_dir: str = "~/.mmco/logs"
    log_level: str = "INFO"
    prompts_dir: str | None = None
