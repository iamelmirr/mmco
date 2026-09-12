from __future__ import annotations

import json

import pytest

from mmco.planner import Planner
from mmco.rules import (
    RulesError,
    eject_prompts,
    ensure_rules_file,
    global_rules_path,
    load_rules,
    parse_rules,
    project_config_dir,
    project_rules_path,
)

from .conftest import FakeOpenAI


def test_parse_front_matter_and_strip_comments():
    options, body = parse_rules(
        "---\n# comment\nexecutor_context: task   # inline comment\n---\n<!-- hidden -->\nBe terse.\n"
    )
    assert options == {"executor_context": "task"}
    assert body == "Be terse."


def test_invalid_context_is_rejected():
    with pytest.raises(RulesError, match="executor_context"):
        parse_rules("---\nexecutor_context: everything\n---\n")


def test_unclosed_front_matter_is_rejected():
    with pytest.raises(RulesError, match="never closed"):
        parse_rules("---\nexecutor_context: task\n")


def test_template_parses_to_no_rules(tmp_path):
    path = ensure_rules_file(tmp_path / "rules.md", "Global", "minimal")
    options, body = parse_rules(path.read_text())
    assert options == {"executor_context": "minimal"} and body == ""


def test_project_rules_override_global(tmp_path):
    project = tmp_path / "app"
    global_rules_path().parent.mkdir(parents=True, exist_ok=True)
    global_rules_path().write_text("---\nexecutor_context: full\n---\nGlobal rule.")
    project_rules_path(project).parent.mkdir(parents=True)
    project_rules_path(project).write_text("---\nexecutor_context: minimal\n---\nProject rule.")
    rules = load_rules(project)
    assert rules.executor_context == "minimal"
    assert rules.text == "Global rule.\n\nProject rule."
    assert load_rules(tmp_path / "other").executor_context == "full"


def test_project_config_lives_outside_the_project(tmp_path, isolated_config):
    project = tmp_path / "app"
    assert isolated_config in project_config_dir(project).parents
    assert project_config_dir(project) != project_config_dir(tmp_path / "elsewhere" / "app")


def test_rules_are_added_to_planner_system_prompt(settings, db, tmp_path):
    project = tmp_path / "app"
    project_rules_path(project).parent.mkdir(parents=True)
    project_rules_path(project).write_text("---\nexecutor_context: task\n---\nNever mention the business goal.")
    session = db.create_session("build it", str(project))
    client = FakeOpenAI([json.dumps({"tasks": [{"description": "x"}]})])
    Planner(settings, db, client=client, sleep=lambda _: None).plan(session, "", [])
    messages = client.completions.calls[0]["messages"]
    assert "USER RULES" in messages[0]["content"] and "Never mention the business goal." in messages[0]["content"]
    assert json.loads(messages[1]["content"])["executor_context"] == "task"


def test_eject_prompts_keeps_existing_files(tmp_path):
    target = tmp_path / "prompts"
    target.mkdir()
    (target / "planner_plan.txt").write_text("mine")
    written = eject_prompts(target)
    assert (target / "planner_plan.txt").read_text() == "mine"
    assert target / "planner_eval.txt" in written
