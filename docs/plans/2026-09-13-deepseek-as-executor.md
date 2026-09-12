# DeepSeek as a Co-Executor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let the DeepSeek planner read the codebase and execute some tasks itself (docs, analysis, small edits) while still steering Claude Code for real coding, deciding the executor per task at run time.

**Architecture:** A deterministic project map and a set of file tools (read-only always, write/edit when executing) are given to the planner via OpenRouter tool-calling. The orchestrator's per-task step branches on an executor decision: the planner runs a tool loop, or Claude runs as today. Both go through the same git checkpoint, verification, regression and escalation gates. A new `take_over` evaluation outcome lets the planner finish a task Claude left almost done.

**Tech Stack:** Python 3.11+, OpenAI SDK (OpenRouter, tool-calling), SQLite, pytest. Design doc: `docs/plans/2026-09-13-deepseek-as-executor-design.md`.

**Conventions in this repo (read before starting):**
- Tests: `tests/`, pytest, no network. `conftest.py` has `settings`, `db`, `isolated_config` (autouse, sets `MMCO_CONFIG_DIR`), and `FakeOpenAI`/`FakeCompletions` (a `.chat.completions.create` double that pops scripted responses; supports strings or Exceptions). Orchestrator tests use `FakePlanner`/`FakeExecutor` (see `tests/test_orchestrator.py`).
- Run tests: `.venv/bin/python -m pytest -q`
- DB migrations: append a SQL string to `MIGRATIONS` in `mmco/db.py`; `PRAGMA user_version` advances automatically.
- Commit after each task. End every commit message with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Keep the executor (Claude Code) path untouched except where explicitly noted.

---

## Task 1: Project map

**Files:**
- Create: `mmco/projectmap.py`
- Test: `tests/test_projectmap.py`

**Step 1: Write failing tests**

```python
# tests/test_projectmap.py
from pathlib import Path
from mmco.projectmap import build_map

def test_python_symbols(tmp_path):
    (tmp_path / "a.py").write_text("import os\n\ndef foo(x):\n    return x\n\nclass Bar:\n    def m(self): ...\n")
    out = build_map(tmp_path)
    assert "a.py" in out
    assert "foo" in out and "Bar" in out
    assert "(4 lines)" in out or "4 lines" in out  # line count shown

def test_skips_noise_dirs_and_binaries(tmp_path):
    (tmp_path / ".venv").mkdir(); (tmp_path / ".venv" / "x.py").write_text("def hidden(): ...")
    (tmp_path / "node_modules").mkdir(); (tmp_path / "node_modules" / "y.js").write_text("x")
    (tmp_path / "img.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "main.py").write_text("def real(): ...")
    out = build_map(tmp_path)
    assert "main.py" in out and "hidden" not in out and "node_modules" not in out and "img.png" not in out

def test_non_python_shows_first_line(tmp_path):
    (tmp_path / "README.md").write_text("# My Project\n\nstuff")
    out = build_map(tmp_path)
    assert "README.md" in out

def test_limit_caps_output(tmp_path):
    for i in range(50):
        (tmp_path / f"f{i}.py").write_text("def g(): ...")
    out = build_map(tmp_path, max_files=10)
    assert "more files" in out
```

**Step 2:** Run `.venv/bin/python -m pytest tests/test_projectmap.py -q` — expect FAIL (no module).

**Step 3: Implement `mmco/projectmap.py`**

Requirements:
- `build_map(root: str | Path, max_files: int = 200, max_symbols: int = 12) -> str`
- Reuse the ignore logic already in the repo: import `_LISTING_SKIP_DIRS` and `DEFAULT_EXCLUDES` from `mmco.workspace` for directory skipping; also skip files whose suffix is binary (`.png .jpg .jpeg .gif .pdf .zip .lock .woff .woff2 .ico .so .dylib .pyc`) and files > 500 KB.
- Prefer git for the file list when `root` is a git repo (`git ls-files --cached --others --exclude-standard`), else `os.walk` skipping `_LISTING_SKIP_DIRS`. (Mirror `Workspace.listing`.)
- For `.py` files: parse with `ast`; collect top-level `FunctionDef`, `AsyncFunctionDef`, `ClassDef` names (up to `max_symbols`). On `SyntaxError`, fall back to first non-empty line.
- For other text files: first non-empty line, trimmed to ~80 chars.
- Each line: `f"{relpath:<40} ({n} lines)  {symbols_or_firstline}"`.
- Cap at `max_files`; if more, append `f"... and {extra} more files"`.
- Sort files alphabetically.

**Step 4:** Run the tests — expect PASS.

**Step 5: Commit**
```bash
git add mmco/projectmap.py tests/test_projectmap.py
git commit -m "feat: deterministic project map for the planner"
```

---

## Task 2: File tools (read-only + write/edit)

**Files:**
- Create: `mmco/tools.py`
- Test: `tests/test_tools.py`

**Step 1: Write failing tests**

```python
# tests/test_tools.py
import pytest
from mmco.tools import Toolbox, ToolError

@pytest.fixture
def box(tmp_path):
    (tmp_path / "a.py").write_text("line1\nline2\n")
    (tmp_path / "sub").mkdir(); (tmp_path / "sub" / "b.txt").write_text("hello world")
    return Toolbox(tmp_path)

def test_read_file(box):
    assert "line1" in box.read_file("a.py")

def test_search(box):
    hits = box.search("hello")
    assert "sub/b.txt" in hits

def test_list_dir(box):
    out = box.list_dir(".")
    assert "a.py" in out and "sub" in out

def test_write_and_edit(box, tmp_path):
    box.write_file("new.py", "x = 1\n")
    assert (tmp_path / "new.py").read_text() == "x = 1\n"
    box.edit_file("new.py", "x = 1", "x = 2")
    assert (tmp_path / "new.py").read_text() == "x = 2\n"

def test_edit_requires_unique_match(box):
    box.write_file("d.py", "a\na\n")
    with pytest.raises(ToolError):
        box.edit_file("d.py", "a", "b")   # not unique

def test_path_escape_blocked(box):
    for bad in ["../outside.txt", "/etc/passwd", "sub/../../x"]:
        with pytest.raises(ToolError):
            box.read_file(bad)
        with pytest.raises(ToolError):
            box.write_file(bad, "x")

def test_read_only_mode_blocks_writes(tmp_path):
    box = Toolbox(tmp_path, allow_write=False)
    with pytest.raises(ToolError):
        box.write_file("x.py", "1")

def test_openai_schemas_shape():
    schemas = Toolbox.schemas(allow_write=True)
    names = {s["function"]["name"] for s in schemas}
    assert {"read_file", "search", "list_dir", "write_file", "edit_file"} <= names
    assert Toolbox.schemas(allow_write=False) and all(
        s["function"]["name"] in {"read_file", "search", "list_dir"} for s in Toolbox.schemas(allow_write=False))
```

**Step 2:** Run `pytest tests/test_tools.py -q` — expect FAIL.

**Step 3: Implement `mmco/tools.py`**

- `class ToolError(MMCOError)` (import `MMCOError` from `mmco.utils`).
- `class Toolbox`:
  - `__init__(self, root, allow_write=True, max_bytes=200_000)`; store `self.root = Path(root).resolve()`.
  - `_resolve(path)`: join to root, `.resolve()`, and raise `ToolError` unless the result is `self.root` or inside it (use `Path.is_relative_to`). Reject absolute inputs too.
  - `read_file(path)`: return text, `ToolError` if missing / too large / binary.
  - `search(query, max_results=50)`: walk tracked files (reuse `projectmap`'s file iterator or a small local one skipping `_LISTING_SKIP_DIRS`), return lines `f"{relpath}:{lineno}: {line}"`. Plain substring match (case-insensitive); no regex.
  - `list_dir(path=".")`: names in the dir, `/`-suffixed for subdirs, skipping `_LISTING_SKIP_DIRS`.
  - `write_file(path, content)`: `ToolError` if `not allow_write`; create parent dirs; write text. Normalise to end with a single `\n`.
  - `edit_file(path, old, new)`: read, require `old` to occur exactly once (else `ToolError`), replace, write.
  - `dispatch(name, arguments: dict)`: route to the method, return a string result; catch `ToolError` and return `f"ERROR: {e}"` (so the model can recover instead of the loop crashing).
  - `@staticmethod schemas(allow_write)`: return OpenAI tool schemas (`{"type":"function","function":{"name","description","parameters"}}`) for the allowed tools.

**Step 4:** Run the tests — expect PASS.

**Step 5: Commit**
```bash
git add mmco/tools.py tests/test_tools.py
git commit -m "feat: sandboxed file tools for the planner"
```

---

## Task 3: Planner tool-calling loop

**Files:**
- Modify: `mmco/planner.py` (add a tool-loop path; `plan`/`evaluate` gain optional `toolbox`)
- Modify: `mmco/config.py` (add `planner_max_tool_calls: int = 25`)
- Test: `tests/test_planner_tools.py`

**Context:** `_request` currently sends one completion and returns text. Add a sibling that runs a tool loop. Do NOT change `_request`'s existing callers.

**Step 1: Write failing tests** — extend `FakeOpenAI` to script tool calls. Add to `tests/conftest.py` a helper building a `ChatCompletion`-shaped object with `tool_calls`, or inline in the test:

```python
# tests/test_planner_tools.py
import json
from types import SimpleNamespace
from mmco.config import Settings
from mmco.planner import Planner
from mmco.models import Session

def tool_call(cid, name, args):
    return SimpleNamespace(id=cid, type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)))

class ScriptedClient:
    """Returns tool calls, then a final text answer."""
    def __init__(self, turns):
        self.turns = list(turns); self.sent = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
    def _create(self, **kwargs):
        self.sent.append(kwargs)
        item = self.turns.pop(0)
        if isinstance(item, list):
            msg = SimpleNamespace(content=None, tool_calls=item)
        else:
            msg = SimpleNamespace(content=item, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(cost=0.001))

def test_planner_runs_tools_then_returns_json(settings, db, tmp_path):
    (tmp_path / "app.py").write_text("def hello(): return 'hi'\n")
    session = db.create_session("x", str(tmp_path))
    client = ScriptedClient([
        [tool_call("c1", "read_file", {"path": "app.py"})],
        json.dumps({"tasks": [{"description": "task using hello()"}]}),
    ])
    planner = Planner(settings, db, client=client, sleep=lambda _: None)
    plan = planner.plan(session, "", [], read_tools=True)
    assert plan.tasks[0].description
    # second request carried the tool result back
    second = client.sent[1]["messages"]
    assert any(m.get("role") == "tool" and "hello" in str(m.get("content")) for m in second)

def test_tool_loop_has_a_cap(settings, db, tmp_path):
    settings.planner_max_tool_calls = 2
    session = db.create_session("x", str(tmp_path))
    (tmp_path / "a.py").write_text("x=1")
    always_tools = [tool_call("c", "read_file", {"path": "a.py"})]
    client = ScriptedClient([always_tools, always_tools, always_tools, json.dumps({"tasks": []})])
    planner = Planner(settings, db, client=client, sleep=lambda _: None)
    # after the cap, the loop forces a final answer or raises PlannerError
    plan = planner.plan(session, "", [], read_tools=True)
    assert plan is not None
```

**Step 2:** Run `pytest tests/test_planner_tools.py -q` — expect FAIL.

**Step 3: Implement**
- `config.py`: add `planner_max_tool_calls: int = 25`.
- `planner.py`:
  - Add `_request_with_tools(session, purpose, messages, toolbox, allow_write, json_mode, task_id) -> tuple[str, list]`: loop up to `planner_max_tool_calls`:
    - call `create(**kwargs, tools=toolbox.schemas(allow_write), tool_choice="auto")` (+ `response_format` only on the final turn is hard to know; instead: keep `response_format` OFF while tools may be called, and when the model returns no tool calls, that content is the answer). Persist each call via `db.add_planner_call` (reuse existing logging).
    - if message has `tool_calls`: append the assistant message and one `{"role":"tool","tool_call_id":...,"content": toolbox.dispatch(...)}` per call; continue.
    - else return `message.content`.
    - On reaching the cap, append a user message: "Tool budget exhausted. Answer now with the required JSON." and do one final `create` without tools.
  - `plan(...)` and `evaluate(...)`: add param `read_tools: bool = False` and `toolbox: Toolbox | None = None`. When enabled, use the tool loop via a new `_call_json_with_tools` that mirrors `_call_json` (same JSON extraction + retry-on-invalid) but sources content from `_request_with_tools`. Keep `_call_json` for the no-tools path unchanged.
  - Reuse `extract_json`, validation, and `add_planner_call` exactly as today.

**Step 4:** Run tests — expect PASS. Then full suite: `.venv/bin/python -m pytest -q` (nothing else should break; new params default off).

**Step 5: Commit**
```bash
git add mmco/planner.py mmco/config.py tests/test_planner_tools.py tests/conftest.py
git commit -m "feat: planner tool-calling loop (read/write files)"
```

---

## Task 4: Data model — executor columns

**Files:**
- Modify: `mmco/db.py` (append migration 3; persist/read `agent`)
- Modify: `mmco/models.py` (`Execution.agent`, `Task.executor`)
- Test: `tests/test_db_executor.py`

**Step 1: Write failing test**

```python
# tests/test_db_executor.py
from mmco.db import Database
from mmco.models import Execution, ExecutionResult

def test_execution_agent_roundtrip(tmp_path):
    db = Database(str(tmp_path / "m.db"))
    s = db.create_session("x", str(tmp_path))
    task = db.insert_tasks(s.id, [__import__("mmco.models", fromlist=["TaskSpec"]).TaskSpec(description="d")])[0]
    ex = Execution(task_id=task.id, session_id=s.id, attempt=1, agent="planner",
                   result=ExecutionResult(prompt="p"))
    db.add_execution(ex)
    assert db.list_executions(task_id=task.id)[0].agent == "planner"

def test_existing_rows_default_to_claude(tmp_path):
    # migration adds column with default 'claude'
    db = Database(str(tmp_path / "m2.db"))
    assert db.conn.execute("PRAGMA user_version").fetchone()[0] >= 3
```

**Step 2:** Run — expect FAIL.

**Step 3: Implement**
- `models.py`: `AgentName = Literal["claude", "planner"]`; add `agent: AgentName = "claude"` to `Execution`; add `executor: AgentName = "claude"` to `Task` (and thus `_TASK_UPDATABLE` in `db.py`).
- `db.py`:
  - Append migration string: `ALTER TABLE executions ADD COLUMN agent TEXT NOT NULL DEFAULT 'claude'; ALTER TABLE tasks ADD COLUMN executor TEXT NOT NULL DEFAULT 'claude';`
  - Add `agent` to the `add_execution` INSERT column list + value `execution.agent`; add to `_execution` reader.
  - Add `"executor"` to `_TASK_UPDATABLE`.

**Step 4:** Run test + full suite — expect PASS.

**Step 5: Commit**
```bash
git add mmco/db.py mmco/models.py tests/test_db_executor.py
git commit -m "feat: record which agent executed each task"
```

---

## Task 5: Orchestrator — planner executes a task

**Files:**
- Modify: `mmco/orchestrator.py` (`_execute` branches on executor; new `_execute_by_planner`)
- Modify: `mmco/planner.py` (add `execute_task(session, task, toolbox) -> str` returning a short report; and `choose_executor(session, task, toolbox) -> tuple[str, str]`)
- Test: `tests/test_orchestrator.py` (extend `FakePlanner`)

**Design:** `_execute` currently always runs Claude. Change it to:
1. Ask `self.planner.choose_executor(...)` → `("planner"|"claude", reason)`. Guard with settings (see Task 8 default) and rules.
2. If `claude`: existing path, `agent="claude"`.
3. If `planner`: build a `Toolbox(project_dir, allow_write=True)`, call `self.planner.execute_task(...)` which runs the write-enabled tool loop and returns a report string. Wrap it into an `ExecutionResult(prompt=<task block>, result_text=report, cost_usd=..., claude_session_id=None)` with `agent="planner"`. Then the SAME diff/verify/regression code runs (do not duplicate it — restructure so both branches converge before `run_verify_commands`).

**Step 1: Write failing tests** (extend `FakePlanner` with `choose_executor`/`execute_task`, and a writes-to-disk fake):

```python
def test_planner_executes_task_itself(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "docs")]), [ok()])
    planner.executor_choice = ("planner", "just docs")
    def planner_writes(root, prompt):
        (root / "README.md").write_text("# Docs\n"); return "wrote README"
    planner.execute_impl = planner_writes
    executor = FakeExecutor([])  # Claude must NOT be called
    session = build(settings, db, planner, executor).start(project, "x")
    assert session.status == "done"
    assert (project / "README.md").exists()
    assert not executor.calls  # Claude untouched
    assert db.list_executions(session_id=session.id)[0].agent == "planner"

def test_planner_work_still_passes_through_checks(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "docs", ["test -f README.md"])]),
                          [ok(), ok()])
    planner.executor_choice = ("planner", "docs")
    planner.execute_impl = lambda root, prompt: "did nothing"  # first attempt writes nothing -> check fails
    ...
```

(Model the fakes on the existing `writes()` helper; `FakePlanner.choose_executor` returns `self.executor_choice` default `("claude","")`; `execute_task` calls `self.execute_impl(root, prompt)`.)

**Step 2:** Run — expect FAIL.

**Step 3: Implement**
- `planner.py`:
  - `choose_executor(session, task, toolbox)`: a `_call_json_with_tools` call using a new prompt `planner_choose_executor.txt` returning `{"executor": "...", "reason": "..."}`. Read-only tools. (Fakes bypass this.)
  - `execute_task(session, task, toolbox)`: write-enabled tool loop using new prompt `planner_execute.txt`; returns the final assistant text (a short report).
- `orchestrator.py`: refactor `_execute` so both executors produce a `result: ExecutionResult` and `agent: str`, then shared diff/verify/regression/`Execution(...)` code runs once. Respect `settings` from Task 8.

**Step 4:** Run tests + full suite — expect PASS.

**Step 5: Commit**
```bash
git add mmco/orchestrator.py mmco/planner.py mmco/prompts/planner_choose_executor.txt mmco/prompts/planner_execute.txt tests/test_orchestrator.py
git commit -m "feat: planner can execute a task itself via file tools"
```

---

## Task 6: `take_over` outcome + hand-off guard

**Files:**
- Modify: `mmco/models.py` (`NextAction` gains `take_over`)
- Modify: `mmco/orchestrator.py` (`_dispatch` branch; per-task hand-off counter)
- Modify: `mmco/prompts/planner_eval.txt` (document `take_over`)
- Test: `tests/test_orchestrator.py`

**Step 1: Write failing tests**

```python
def test_take_over_finishes_a_claude_task(settings, db, project):
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "feature", ["test -f done.txt"])]),
                          [verdict("take_over", "claude almost did it")])
    planner.executor_choice = ("claude", "")
    planner.execute_impl = lambda root, prompt: (root / "done.txt").write_text("1") or "finished it"
    executor = FakeExecutor([writes({"partial.py": "x"})])  # Claude's attempt
    session = build(settings, db, planner, executor).start(project, "x")
    assert session.status == "done"
    agents = [e.agent for e in db.list_executions(session_id=session.id)]
    assert agents == ["claude", "planner"]

def test_handoff_counter_stops_pingpong(settings, db, project):
    # planner keeps take_over-ing but check never passes -> escalation after N
    ...
```

**Step 2:** Run — expect FAIL.

**Step 3: Implement**
- `models.py`: add `"take_over"` to `NextAction`.
- `orchestrator.py` `_dispatch`: new branch — run `_execute_by_planner` for the same task (a fresh `Execution` with `agent="planner"`), then evaluate again. Increment `task`-scoped hand-off count (store on the Task via a new field `handoffs: int = 0`, added to `_TASK_UPDATABLE`, or in metadata); when it exceeds `settings.max_handoffs_per_task` (Task 8), escalate.
- `planner_eval.txt`: add `take_over` to the allowed `next_action` list with a one-line description.

**Step 4:** Run tests + full suite — expect PASS.

**Step 5: Commit**
```bash
git add mmco/models.py mmco/orchestrator.py mmco/prompts/planner_eval.txt tests/test_orchestrator.py
git commit -m "feat: take_over outcome with a per-task hand-off guard"
```

---

## Task 7: Settings, prompts wiring, default context

**Files:**
- Modify: `mmco/config.py` (`allow_planner_executor: bool = True`, `max_handoffs_per_task: int = 3`; default `executor_context` stays `full` unless you decide otherwise — DO NOT change silently)
- Modify: `mmco/planner.py` / `orchestrator.py` to actually enable read-tools during plan/eval when `allow_planner_executor` and a git project map is available
- Modify: `mmco/prompts/planner_plan.txt` (mention the project map + that it may read files)
- Test: `tests/test_orchestrator.py` (a flag-off test: planner never executes when disabled)

**Step 1: Write failing test**

```python
def test_disabling_planner_executor_forces_claude(settings, db, project):
    settings.allow_planner_executor = False
    planner = FakePlanner(PlanResponse(tasks=[spec(0, "docs")]), [ok()])
    planner.executor_choice = ("planner", "docs")  # planner WANTS to, but flag is off
    executor = FakeExecutor([writes({"x.py": "1"})])
    session = build(settings, db, planner, executor).start(project, "x")
    assert session.status == "done"
    assert executor.calls  # Claude was used despite the planner's choice
```

**Step 2:** Run — expect FAIL.

**Step 3: Implement** the flag checks in `_execute`/`choose_executor` gating; inject the project map into the planner payload (Task 1's `build_map`) in `planner.plan`. Add settings to `.env.example`.

**Step 4:** Run tests + full suite — expect PASS.

**Step 5: Commit**
```bash
git add mmco/config.py mmco/planner.py mmco/orchestrator.py mmco/prompts/planner_plan.txt .env.example tests/test_orchestrator.py
git commit -m "feat: settings and project-map wiring for planner execution"
```

---

## Task 8: Dashboard + web API surface

**Files:**
- Modify: `mmco/web.py` (`session_state` already returns tasks; add `agent` to execution/task payloads)
- Modify: `mmco/ui/index.html` (timeline: colour/label the step by executor; history filter optional)
- Test: `tests/test_web.py`

**Step 1: Write failing test**

```python
def test_session_state_exposes_agent(dashboard, tmp_path):
    # create a session with a planner-executed execution, assert the API returns agent
    ...
```

**Step 2:** Run — expect FAIL.

**Step 3: Implement** — add `agent` to the execution dicts in `web.session_state`/`session_detail`; in `index.html` `taskDetail`, show "Built by DeepSeek" vs "Built by Claude" per attempt, and tint the build segment accordingly.

**Step 4:** Run tests + full suite. Also do a manual smoke check with `mmco ui` and a headless screenshot (see prior sessions).

**Step 5: Commit**
```bash
git add mmco/web.py mmco/ui/index.html tests/test_web.py
git commit -m "feat: show which agent executed each task in the dashboard"
```

---

## Task 9: README + end-to-end verification

**Files:**
- Modify: `README.md` (document co-executor, the new settings, `executor_context` interplay)
- Test: manual `MMCO_E2E`-style run in a throwaway folder (not in CI)

**Steps:**
1. Update the README table of guarantees and the configuration table.
2. Run the full suite: `.venv/bin/python -m pytest -q` — all green.
3. Manual: in a scratch folder, a small request that should produce a docs task (planner-executed) and a code task (Claude), confirm both paths work and both pass checks.
4. Commit README.
5. Use superpowers:finishing-a-development-branch to open a PR into `main`.

---

## Notes for the executor

- Keep the Claude Code path byte-for-byte where possible; all new behaviour is additive and gated by `allow_planner_executor`.
- Every planner-produced result must pass the SAME `run_verify_commands` + regression + git-checkpoint gates as Claude. Do not add a shortcut that lets planner work skip checks.
- Tool loops are bounded (`planner_max_tool_calls`); hand-offs are bounded (`max_handoffs_per_task`). Overruns escalate, never loop forever.
- Do not let file tools escape `project_dir`; the `test_path_escape_blocked` test is load-bearing.
