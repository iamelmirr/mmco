# MMCO: Multi-Model Coding Orchestrator

You talk to a **planner** (DeepSeek via OpenRouter). It reads your codebase, breaks your request into technical
tasks and, one at a time, decides who executes each: **Claude Code** for real code development, or **DeepSeek
itself** for docs, content, config, analysis and small edits. After every task the planner checks the real
evidence (git diff + test results) and decides what happens next — including finishing a task Claude left almost
done — until the whole request is done.

```
you ─► mmco chat ─► planner.plan (reads your code) ─► you approve / change the plan
                                         │
                ┌────────────────────────┘
                ▼
     git checkpoint ─► planner.choose_executor
                │            ├── Claude Code (task prompt, with as much context as your rules allow)
                │            └── DeepSeek     (write-enabled file tools, sandboxed to the project dir)
                ▲                              │
                │        git diff + task checks + checks of earlier tasks (the same gates for both)
                │                              │
                │                    planner.evaluate
                ├── retry        (same Claude session, with reviewer feedback)
                ├── reformulate  (roll back, planner rewrites the prompt)
                ├── take_over    (DeepSeek finishes what Claude left almost done, bounded per task)
                ├── add_task     (prerequisites first / follow-ups next)
                ├── clarify      (asks you)
                ├── stuck        (asks you: guidance / skip / abort; never fails silently)
                └── continue ─► git commit ─► next task
                                         │
                      all tasks done ─► rerun every check + final review against your request
                                         └─► missing work becomes new tasks
```

## Install (once)

Requires Python 3.11+, git, and Claude Code logged in (`claude` works in your terminal).

```bash
./install.sh     # creates .venv, links `mmco` into ~/.local/bin, asks for your OpenRouter key
```

The key is stored in `~/.config/mmco/.env` (mode 600). Re-run `mmco setup` to change it.

## Dashboard (visual)

```bash
mmco ui          # opens http://127.0.0.1:8765 in your browser
```

To keep it inside VS Code: Command Palette, `Simple Browser: Show`, paste the address.

- **Projects** in the sidebar, with a live status dot. *Add a project folder* browses your disk or creates a folder.
- **Work**: write a request, then review DeepSeek's plan. You can edit task titles, descriptions and checks, reorder or
  remove tasks, ask for a new plan, or run it. Every task shows a four-step track (DeepSeek writes, Claude builds,
  checks run, DeepSeek reviews) that updates live. Open a task to see every attempt: the exact prompt Claude received,
  its report, changed files, which checks passed and DeepSeek's verdict. Questions and stuck tasks appear as cards
  with buttons (send guidance, try again, skip, stop). Pause and Resume work like Ctrl+C and `mmco resume`.
- **History**: every request in the project with status, tasks and cost.
- **Rules and prompts** (per project, or global from the sidebar): pick how much Claude knows, write DeepSeek's rules,
  see the full planning prompt DeepSeek receives, and edit any planner or Claude prompt with Save and Reset.
- **Settings**: OpenRouter key, models, budgets, attempt limits and behaviour switches.

Runs are separate background processes: closing the browser or stopping `mmco ui` does not stop the work.
The dashboard only answers on 127.0.0.1 and only to the page it served.

## Daily use in VS Code (terminal)

1. Open your project folder in VS Code (an empty folder works too).
2. Open the terminal (`` Ctrl+` ``) and run `mmco`.
3. Type what you want, e.g. `Build a REST API for a book library with auth, CRUD and search`.
4. The planner shows the plan. Press Enter to run it, type a change ("use FastAPI, skip auth") to get a new plan,
   or `no` to cancel.
5. Watch the tasks run. If the planner needs a decision, or a task is stuck, you are asked right there.
6. When it is done, type the next request (`add pagination to search`). The planner knows what was built before.

Every finished task is a git commit named `mmco: <task>`, so the VS Code Source Control panel shows exactly what
changed and lets you revert anything. You can edit files yourself between requests; mmco commits your edits
before starting the next request.

Chat commands: `/status`, `/tasks`, `/log [--full]`, `/resume`, `/sessions`, `/rules`, `/rules global`, `/help`,
`/quit`. Ctrl+C pauses the running work; `/resume` (or reopening `mmco` in the folder) continues it.

Why not opencode or another agent as the front end? It would be a second orchestrator with its own loop and state,
competing with this one. `mmco` is the single place that holds the plan, the checks and the history.

## Controlling the planner: rules

`/rules` (or `mmco rules`) opens this project's rules in VS Code; `/rules global` opens the rules for all projects.

```markdown
---
executor_context: task
---
- Give Claude purely technical instructions: exact file names, function signatures and expected behaviour.
- Never mention the product name, the company, its users or the business purpose.
- Let Claude choose implementation details inside each task.
```

- The text is added to **every** planner prompt (planning, reviewing, rewriting prompts, final review).
  Edits apply on the next planner call, even in a running session.
- `executor_context` decides what Claude Code is shown. This is enforced by mmco, not just requested from the planner:

  | value | Claude sees |
  | --- | --- |
  | `full` | your request, the whole plan with its position in it, your answers, its task |
  | `task` | only its own task (description, contracts, criteria, files, checks), no goal, no plan, no discussion |
  | `minimal` | only the instructions the planner wrote plus the checks; a generic system prompt with no mention of mmco |

  With `task`/`minimal`, the planner is told that every task must be fully self-contained.
- Project rules override global ones. Rules live in `~/.config/mmco/` (project rules under
  `~/.config/mmco/projects/<folder>-<hash>/`), never inside the project, because Claude can read every file there.
- `mmco rules --show` prints the effective rules.

For complete control, `mmco prompts --eject` copies all built-in prompts into the project's config folder
(`--global` for all projects). Files starting with `planner_` are DeepSeek's system prompts, files starting with
`executor_` are what Claude Code receives. Delete a copied file to return to the built-in version.

## DeepSeek as a co-executor

By default the planner is more than a task dispatcher: it reads your code and can do some of the work itself.

- **It sees the codebase, not just a file list.** Every planning and evaluation prompt is given a *project map* —
  the functions and classes per file, built deterministically (`mmco/projectmap.py`) — and read-only file tools
  (`read_file`, `search`, `list_dir`) it can call in a bounded loop while it plans and reviews. So it reasons about
  the real code instead of guessing from names.
- **It picks the executor per task, at run time.** For each task the planner decides who builds it: Claude Code for
  real code development, or DeepSeek itself for docs, content, config, analysis and small edits, using
  write-enabled file tools (`write_file`, `edit_file`). Claude still makes the implementation decisions inside its
  own tasks — the default `executor_context` is unchanged.
- **It can take over.** After Claude's attempt, the planner may choose the `take_over` outcome: DeepSeek finishes a
  task Claude left almost done, building on Claude's on-disk work rather than starting over.
- **The same gates apply to DeepSeek's own work.** Whoever builds a task, it goes through the identical pipeline —
  git checkpoint, `verify_commands`, regression checks against earlier tasks, and the escalation path. A task whose
  checks fail is never marked done, regardless of who built it. DeepSeek's file tools are sandboxed to the project
  directory, and take_over / executor ping-pong is bounded per task (`MMCO_MAX_HANDOFFS_PER_TASK`) before the task
  is escalated. The tool-calling loop is capped by `MMCO_PLANNER_MAX_TOOL_CALLS`.

Which agent executed each task is recorded and shown in the dashboard ("Built by DeepSeek" / "Built by Claude").
Set `MMCO_ALLOW_PLANNER_EXECUTOR=false` to turn all of this off: Claude does everything and the planner gets no
tools, exactly as before.

## What it guarantees and how

| Situation | Behaviour |
| --- | --- |
| Claude says "done" but is not | Each task has checks (`verify_commands`) that mmco runs itself. A task whose checks fail is never marked done, whatever the planner says. |
| A later task breaks earlier work | After every task the checks of all earlier tasks are rerun; a regression sends the task back with the failing output. |
| Something from the request is missing at the end | Final review: all checks rerun, the planner compares the project with the request and adds tasks for gaps (up to `MMCO_MAX_FINAL_REVIEWS` rounds). Checks that fail at that point become a fix task automatically. |
| A task keeps failing | Retry (same Claude session with feedback), then reformulation (rollback + new prompt). When the budget is used up, **you are asked** for guidance (fresh budget), `skip` or `abort`. Nothing fails silently. |
| DeepSeek executes a task itself | Its work goes through the same git checkpoint, `verify_commands` and regression checks as Claude's; a task whose checks fail is never marked done, whoever built it. Its file tools are sandboxed to the project directory. |
| Executors keep handing work back and forth | take_over and executor ping-pong are bounded per task (`MMCO_MAX_HANDOFFS_PER_TASK`); past that the task is escalated instead of looping. |
| The planner wrote a wrong check | It can replace the check (`updated_verify_commands`); the new check is run immediately. It can never remove all checks. |
| Claude API overloaded / rate limited | Retried with backoff without using the task's attempts. |
| Claude login, credits or usage limit | Session pauses with the reason; `mmco resume` after fixing it. |
| OpenRouter down during evaluation | Session pauses; on resume the finished Claude run is evaluated instead of redone. |
| Ctrl+C, crash, closed terminal | State is in SQLite. Resume discards only the half-finished attempt. DB errors dump `mmco-crash-<ts>.json`. |
| Two mmco processes on one folder | Refused (lock in `.git/mmco.lock`, stale locks are cleared). |
| Runaway loops or cost | `MMCO_MAX_LOOP_ITERATIONS` and `MMCO_MAX_SESSION_COST_USD` pause the session (resumable). |
| Refusal-looking wording | Counted as a refusal only if it is at the start of the reply **and** nothing changed; triggers a reformulation. |

It cannot guarantee that every request is achievable; when it cannot proceed, it stops and asks you instead of
marking work done or giving up.

## Commands outside chat

```bash
mmco start ./todo-app "Build a TODO app with Flask and SQLite" [--no-input] [--confirm]
mmco resume <session>                     # continue paused / waiting / failed work
mmco status <session>                     # task table: status, attempts, duration, cost
mmco tasks <session>                      # full task definitions and reformulated prompts
mmco log <session> [--full] [--json]      # every planner call, Claude run and verdict
mmco replay <session> --from-task 3       # git reset to before task 3 and re-run from there
mmco sessions                             # all sessions
mmco rules [--global] [--show]            # planner rules and executor_context
mmco prompts [--eject] [--global]         # prompt overrides
mmco setup                                # OpenRouter key and environment check
```

A unique prefix of a session or task id is enough. `--no-input` never prompts: the session waits with status
`awaiting_input` until `mmco resume`. Exit codes: `0` done, `1` failed, `2` paused or waiting, `130` interrupted.

## Safety notes

- Claude runs with `--allowedTools Read,Write,Edit,Bash,Glob,Grep` and `--permission-mode acceptEdits`, so it can run
  any shell command in the project. Narrow `MMCO_CLAUDE_ALLOWED_TOOLS` (e.g. `Bash(pytest *)`) or use a container
  for untrusted work. Checks come from the planner and run through `/bin/sh`.
- mmco commits to the project repository and uses `git reset --hard` + `git clean -fd` when it rolls back
  (reformulation, skip, resume after an interrupt, replay). It refuses a folder nested inside another repository.

## Configuration

Environment variables, read from `~/.config/mmco/.env`, then `./.env`. See [.env.example](.env.example).

| Variable | Default | |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | | required (`mmco setup`) |
| `MMCO_PLANNER_MODEL` | `deepseek/deepseek-v4.1-flash` | any OpenRouter model id |
| `MMCO_CLAUDE_MODEL` | Claude Code default | e.g. `sonnet` (cheaper) |
| `MMCO_EXECUTOR_CONTEXT` | `full` | default when no rules file sets it |
| `MMCO_ALLOW_PLANNER_EXECUTOR` | `true` | let the planner read code and execute some tasks itself; `false` = Claude does everything, no planner tools |
| `MMCO_MAX_HANDOFFS_PER_TASK` | `3` | bound on take_over / executor ping-pong before a task is escalated |
| `MMCO_PLANNER_MAX_TOOL_CALLS` | `25` | cap on the planner's tool-calling loop per plan / eval / execute |
| `MMCO_CLAUDE_MAX_BUDGET_USD_PER_TASK` | none | passed as `--max-budget-usd` |
| `MMCO_MAX_SESSION_COST_USD` | none | pauses the session when reached |
| `MMCO_MAX_ATTEMPTS_PER_TASK` / `MMCO_MAX_REFORMULATIONS_PER_TASK` | `3` / `2` | before asking you |
| `MMCO_ESCALATE_TO_USER` | `true` | `false` = mark stuck tasks failed (for scripts/CI) |
| `MMCO_FINAL_REVIEW` / `MMCO_MAX_FINAL_REVIEWS` | `true` / `2` | |
| `MMCO_REGRESSION_CHECKS` | `true` | rerun earlier tasks' checks after each task |
| `MMCO_DB_PATH` / `MMCO_LOG_DIR` | `~/.mmco/mmco.db` / `~/.mmco/logs` | JSON-lines log per session |

## Development

```bash
.venv/bin/python -m pytest -q                        # 80 tests, mocked models, no network
MMCO_E2E=1 .venv/bin/python -m pytest -q -m e2e      # real planner call (costs a fraction of a cent)
```
