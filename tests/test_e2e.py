"""Opt-in tests against the real planner API. Run with: MMCO_E2E=1 pytest -m e2e"""

from __future__ import annotations

import os

import pytest

from mmco.config import Settings
from mmco.planner import Planner

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.environ.get("MMCO_E2E") != "1", reason="set MMCO_E2E=1 to call the real planner"),
]


def test_real_planner_produces_a_valid_plan(db):
    settings = Settings()  # reads OPENROUTER_API_KEY and model from .env
    if not settings.openrouter_api_key:
        pytest.skip("OPENROUTER_API_KEY not configured")
    planner = Planner(settings, db)
    session = db.create_session("Create a Python CLI that prints the first N Fibonacci numbers, with pytest tests",
                                "/tmp/fib")
    plan = planner.plan(session, "", [])
    assert plan.tasks, plan
    assert all(t.description for t in plan.tasks)
    assert any(t.verify_commands for t in plan.tasks)
