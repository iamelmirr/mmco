from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mmco.config import Settings
from mmco.db import Database


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch) -> Path:
    """Never read the real ~/.config/mmco (rules, prompt overrides) during tests."""
    directory = tmp_path / "config"
    monkeypatch.setenv("MMCO_CONFIG_DIR", str(directory))
    return directory


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        openrouter_api_key="test-key",
        db_path=str(tmp_path / "state" / "mmco.db"),
        log_dir=str(tmp_path / "state" / "logs"),
        claude_binary="true",  # any binary that exists on PATH
        planner_max_retries=3,
        max_attempts_per_task=2,
        max_reformulations_per_task=1,
        transient_backoff_seconds=0,
    )


@pytest.fixture
def db(settings: Settings) -> Database:
    database = Database(settings.db_path)
    yield database
    database.close()


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=item))],
            usage=SimpleNamespace(cost=0.001),
        )


class FakeOpenAI:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)
