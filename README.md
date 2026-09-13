# mmco — Multi-Model Coding Orchestrator

`mmco` is a command-line coding agent with two brains. You describe what you want; a **planner**
(DeepSeek, via [OpenRouter](https://openrouter.ai)) reads your codebase and breaks the request into
technical tasks. For each task it decides who is the best executor — **Claude Code** for real code
development, or **DeepSeek itself** for docs, content, config, analysis and small edits. After every
task the planner inspects the real evidence (the git diff plus your test results) and decides what
happens next, until the whole request is genuinely done.

It's for developers who want an agent that plans deliberately, checks its own work against commands
you can see, and commits each finished task to git so nothing is a black box.

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

## Table of contents

- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Usage](#usage)
  - [Chat mode](#chat-mode)
  - [The dashboard](#the-dashboard)
  - [One-shot CLI commands](#one-shot-cli-commands)
- [Configuration](#configuration)
- [How it stays reliable](#how-it-stays-reliable)
- [DeepSeek as a co-executor](#deepseek-as-a-co-executor)
- [Controlling the planner](#controlling-the-planner)
- [Safety](#safety)
- [Development and tests](#development-and-tests)
- [License](#license)

## Prerequisites

- **Python 3.11+**
- **git**
- **Claude Code**, installed and logged in — `claude` should work in your terminal.
- **An OpenRouter account** and API key — get one at [openrouter.ai/keys](https://openrouter.ai/keys).

## Quickstart

```bash
git clone <this-repo-url>
cd mmco
./install.sh          # creates .venv, links `mmco` into ~/.local/bin, then runs first-time setup
```

`install.sh` finds a suitable Python, builds a virtualenv in `.venv`, and puts the `mmco` command on
your `PATH`. It finishes by running `mmco setup`, which asks for your OpenRouter key and checks that
Claude Code and git are ready. The key is stored in `~/.config/mmco/.env` (mode `600`); re-run
`mmco setup` any time to change it.

If the installer prints a note about your `PATH`, add this to your shell profile and reopen the terminal:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Now `cd` into any project (an empty folder is fine) and start:

```bash
cd ~/code/my-project
mmco            # interactive chat mode
# or
mmco ui         # visual dashboard at http://127.0.0.1:8765
```

> **Heads up:** by default Claude Code runs in `bypassPermissions` mode (it can run any shell command
> in the project directory) and there are no spend caps. This is fast and hands-off, but you should
> point it at code you trust. See [Safety](#safety) to dial it back.

## Usage

### Chat mode

Run `mmco` (or `mmco chat`) inside your project folder:

1. Type what you want, e.g. `Build a REST API for a book library with auth, CRUD and search`.
2. The planner reads your code and shows a plan. Press **Enter** to run it, type a change
   (`use FastAPI, skip auth`) to get a new plan, or `no` to cancel. Use `mmco chat --yes` to run
   plans without asking.
3. Watch the tasks run. If the planner needs a decision, or a task is stuck, you're asked right there.
4. When it's done, type the next request (`add pagination to search`). The planner remembers what was
   built before.

Every finished task becomes a git commit named `mmco: <task>`, so your Source Control panel shows
exactly what changed and lets you revert anything. You can edit files yourself between requests; mmco
commits your edits before starting the next one. **Ctrl+C** pauses the running work; `/resume` (or
just reopening `mmco` in the folder) continues it.

In-chat commands:

| Command | Does |
| --- | --- |
| `/status` | current session: tasks, attempts, time, cost |
| `/tasks` | full task definitions |
| `/log [--full]` | every planner call, Claude run and verdict |
| `/resume` | continue the last unfinished session in this folder |
| `/sessions` | sessions in this folder |
| `/rules` / `/rules global` | edit the planner rules (and how much Claude is told) |
| `/help` | show help |
| `/quit` | exit (also Ctrl+D) |

### The dashboard

```bash
mmco ui          # opens http://127.0.0.1:8765 in your browser
```

To keep it inside VS Code: Command Palette → `Simple Browser: Show`, then paste the address.

- **Projects** — the sidebar lists your project folders with a live status dot. *Add a project folder*
  browses your disk or creates one.
- **Work** — write a request, then review DeepSeek's plan. You can edit task titles, descriptions and
  checks, reorder or remove tasks, ask for a new plan, or run it. Each task shows a live track of its
  stages — plan, build (by **whichever** executor the planner picked, Claude or DeepSeek), checks,
  review. Open a task to see every attempt: the exact prompt the executor received, its report, the
  changed files, which checks passed and DeepSeek's verdict. Questions and stuck tasks appear as cards
  with buttons (send guidance, try again, skip, stop). Pause and Resume behave like Ctrl+C and
  `mmco resume`.
- **History** — every request in the project, with status, tasks and cost.
- **Rules and prompts** (per project, or global from the sidebar) — choose how much Claude knows,
  write DeepSeek's rules, read the full planning prompt DeepSeek receives, and edit any planner or
  Claude prompt with Save and Reset.
- **Settings** — OpenRouter key, models, budgets, attempt limits and behaviour switches.

Runs are separate background processes: closing the browser or stopping `mmco ui` doesn't stop the
work. The dashboard only answers on `127.0.0.1` and only to the page it served.

### One-shot CLI commands

For scripts and headless use, everything is also a plain command:

```bash
mmco start ./todo-app "Build a TODO app with Flask and SQLite" [--no-input] [--confirm] [--allow-dirty]
mmco resume <session>                     # continue paused / interrupted / failed work
mmco status <session>                     # task table: status, attempts, duration, cost
mmco tasks <session>                      # full task definitions and reformulated prompts
mmco log <session> [--full] [--json]      # every planner call, Claude run and verdict
mmco replay <session> --from-task 3       # git reset to before task 3 and re-run from there
mmco sessions [--limit N]                 # recent sessions across all projects
mmco rules [--global] [--show]            # planner rules and executor_context
mmco prompts [--eject] [--global]         # prompt overrides
mmco setup                                # OpenRouter key and environment check
```

A unique prefix of a session or task id is enough. `--no-input` never prompts: the session waits with
status `awaiting_input` until `mmco resume`. Exit codes: `0` done, `1` failed, `2` paused or waiting,
`130` interrupted.

## Configuration

Everything is editable in the dashboard's **Settings** panel, or as environment variables read from
`~/.config/mmco/.env` first, then a local `./.env`. See [.env.example](.env.example) for the full,
commented list. The settings you'll reach for most:

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — | required (set by `mmco setup`) |
| `MMCO_PLANNER_MODEL` | `deepseek/deepseek-v4.1-flash` | any OpenRouter model id |
| `MMCO_CLAUDE_MODEL` | `opus` | any Claude Code alias/name; e.g. `sonnet` is cheaper; empty = your Claude Code default |
| `MMCO_CLAUDE_PERMISSION_MODE` | `bypassPermissions` | `bypassPermissions` / `acceptEdits` / `plan` / `default` — see [Safety](#safety) |
| `MMCO_CLAUDE_ALLOWED_TOOLS` | `Read,Write,Edit,Bash,Glob,Grep,WebFetch,WebSearch` | narrow it (e.g. `Bash(pytest *)`) for untrusted work |
| `MMCO_EXECUTOR_CONTEXT` | `full` | how much Claude is told (`full` / `task` / `minimal`); a rules file overrides it |
| `MMCO_ALLOW_PLANNER_EXECUTOR` | `true` | let the planner read code and execute some tasks itself; `false` = Claude does everything |
| `MMCO_MAX_HANDOFFS_PER_TASK` | `3` | bound on take_over / executor ping-pong before a task is escalated |
| `MMCO_PLANNER_MAX_TOOL_CALLS` | `25` | cap on the planner's tool-calling loop per plan / eval / execute |
| `MMCO_CLAUDE_MAX_BUDGET_USD_PER_TASK` | none | passed to Claude Code as `--max-budget-usd` |
| `MMCO_MAX_SESSION_COST_USD` | none | pauses the session when reached |
| `MMCO_MAX_ATTEMPTS_PER_TASK` / `MMCO_MAX_REFORMULATIONS_PER_TASK` | `3` / `2` | before asking you |
| `MMCO_ESCALATE_TO_USER` | `true` | `false` = mark stuck tasks failed (for scripts/CI) |
| `MMCO_FINAL_REVIEW` / `MMCO_MAX_FINAL_REVIEWS` | `true` / `2` | end-of-run check that nothing is missing |
| `MMCO_REGRESSION_CHECKS` | `true` | rerun earlier tasks' checks after each task |
| `MMCO_DB_PATH` / `MMCO_LOG_DIR` | `~/.mmco/mmco.db` / `~/.mmco/logs` | state and JSON-lines log per session |

There are no per-task or per-session spend caps by default; set `MMCO_CLAUDE_MAX_BUDGET_USD_PER_TASK`
and/or `MMCO_MAX_SESSION_COST_USD` if you want them.

## How it stays reliable

mmco is built so an executor can't declare victory it hasn't earned. The same gates apply no matter
who built a task.

| Situation | Behaviour |
| --- | --- |
| Claude says "done" but isn't | Each task has checks (`verify_commands`) that mmco runs itself. A task whose checks fail is never marked done, whatever the planner says. |
| A later task breaks earlier work | After every task the checks of all earlier tasks are rerun; a regression sends the task back with the failing output. |
| Something from the request is missing at the end | Final review: all checks rerun, the planner compares the project with the request and adds tasks for gaps (up to `MMCO_MAX_FINAL_REVIEWS` rounds). Checks that fail at that point become a fix task automatically. |
| A task keeps failing | Retry (same Claude session with feedback), then reformulation (rollback + new prompt). When the budget is used up, **you are asked** for guidance (fresh budget), `skip` or `abort`. Nothing fails silently. |
| DeepSeek executes a task itself | Its work goes through the same git checkpoint, `verify_commands` and regression checks as Claude's; a task whose checks fail is never marked done, whoever built it. Its file tools are sandboxed to the project directory. |
| Executors keep handing work back and forth | take_over and executor ping-pong are bounded per task (`MMCO_MAX_HANDOFFS_PER_TASK`); past that the task is escalated instead of looping. |
| The planner wrote a wrong check | It can replace the check (`updated_verify_commands`); the new check runs immediately. It can never remove all checks. |
| Claude API overloaded / rate limited | Retried with backoff without using the task's attempts. |
| Claude login, credits or usage limit | Session pauses with the reason; `mmco resume` after fixing it. |
| OpenRouter down during evaluation | Session pauses; on resume the finished Claude run is evaluated instead of redone. |
| Ctrl+C, crash, closed terminal | State is in SQLite. Resume discards only the half-finished attempt. DB errors dump `mmco-crash-<ts>.json`. |
| Two mmco processes on one folder | Refused (lock in `.git/mmco.lock`; stale locks are cleared). |
| Runaway loops or cost | `MMCO_MAX_LOOP_ITERATIONS` and `MMCO_MAX_SESSION_COST_USD` pause the session (resumable). |
| Refusal-looking wording | Counted as a refusal only if it's at the start of the reply **and** nothing changed; triggers a reformulation. |

It can't guarantee that every request is achievable; when it can't proceed, it stops and asks you
instead of marking work done or giving up.

## DeepSeek as a co-executor

By default the planner is more than a task dispatcher — it reads your code and can do some of the work
itself.

- **It sees the codebase, not just a file list.** Every planning and evaluation prompt is given a
  *project map* — the functions and classes per file, built deterministically (`mmco/projectmap.py`) —
  plus read-only file tools (`read_file`, `search`, `list_dir`) it can call in a bounded loop while it
  plans and reviews. So it reasons about the real code instead of guessing from names.
- **It picks the executor per task, at run time.** For each task the planner decides who builds it:
  Claude Code for real code development, or DeepSeek itself for docs, content, config, analysis and
  small edits, using write-enabled file tools (`write_file`, `edit_file`). Claude still makes the
  implementation decisions inside its own tasks — the default `executor_context` is unchanged.
- **It can take over.** After Claude's attempt, the planner may choose the `take_over` outcome:
  DeepSeek finishes a task Claude left almost done, building on Claude's on-disk work rather than
  starting over.
- **The same gates apply to DeepSeek's own work.** Whoever builds a task, it goes through the identical
  pipeline — git checkpoint, `verify_commands`, regression checks against earlier tasks, and the
  escalation path. DeepSeek's file tools are sandboxed to the project directory, and take_over /
  executor ping-pong is bounded per task (`MMCO_MAX_HANDOFFS_PER_TASK`) before the task is escalated.
  The tool-calling loop is capped by `MMCO_PLANNER_MAX_TOOL_CALLS`.

Which agent executed each task is recorded and shown in the dashboard ("Built by DeepSeek" / "Built by
Claude"). Set `MMCO_ALLOW_PLANNER_EXECUTOR=false` to turn all of this off: Claude does everything and
the planner gets no tools.

## Controlling the planner

`mmco rules` (or `/rules` in chat) opens this project's rules in your editor; `mmco rules --global`
opens the rules shared by all projects. A rules file is Markdown with optional front matter:

```markdown
---
executor_context: task
---
- Give Claude purely technical instructions: exact file names, function signatures and expected behaviour.
- Never mention the product name, the company, its users or the business purpose.
- Let Claude choose implementation details inside each task.
```

- The text is added to **every** planner prompt (planning, reviewing, rewriting prompts, final
  review). Edits apply on the next planner call, even in a running session.
- `executor_context` decides what Claude Code is shown. This is enforced by mmco, not merely requested
  from the planner:

  | value | Claude sees |
  | --- | --- |
  | `full` | your request, the whole plan with its position in it, your answers, its task |
  | `task` | only its own task (description, contracts, criteria, files, checks) — no goal, no plan, no discussion |
  | `minimal` | only the instructions the planner wrote plus the checks; a generic system prompt with no mention of mmco |

  With `task` / `minimal`, the planner is told every task must be fully self-contained.
- Project rules override global ones. Rules live in `~/.config/mmco/` (project rules under
  `~/.config/mmco/projects/<folder>-<hash>/`), never inside the project — because Claude can read
  every file there. `mmco rules --show` prints the effective rules.

For complete control, `mmco prompts --eject` copies all built-in prompts into the project's config
folder (`--global` for all projects). Files starting with `planner_` are DeepSeek's system prompts;
files starting with `executor_` are what Claude Code receives. Delete a copied file to return to the
built-in version.

## Safety

Read this before pointing mmco at code you care about.

- **The default is hands-off and powerful.** Claude Code runs with
  `--permission-mode bypassPermissions` and `--allowedTools Read,Write,Edit,Bash,Glob,Grep,WebFetch,WebSearch`,
  so it can run **any** shell command in the project directory without asking. The planner's checks
  run through `/bin/sh`. This is deliberate — it means no interruptions mid-run — but it's real power.
- **No spend caps by default.** Neither per-task nor per-session cost is limited unless you set
  `MMCO_CLAUDE_MAX_BUDGET_USD_PER_TASK` and/or `MMCO_MAX_SESSION_COST_USD`.
- **How to dial it back.** In the dashboard **Settings**, or in `~/.config/mmco/.env`:
  - Set `MMCO_CLAUDE_PERMISSION_MODE=acceptEdits` (or `plan` / `default`) so shell commands aren't run
    unattended.
  - Narrow `MMCO_CLAUDE_ALLOWED_TOOLS` (e.g. `Bash(pytest *)`) to restrict what Claude may do.
  - Set the spend caps above.
- **For untrusted work**, run mmco against a throwaway directory or inside a container.
- mmco commits to the project repository and uses `git reset --hard` + `git clean -fd` when it rolls
  back (reformulation, skip, resume after an interrupt, replay). It refuses a folder nested inside
  another git repository.

## Development and tests

```bash
.venv/bin/python -m pytest -q                        # full suite: mocked models, no network
MMCO_E2E=1 .venv/bin/python -m pytest -q -m e2e      # opt-in: one real planner call (a fraction of a cent)
```

The design notes and implementation plans live in [`docs/plans/`](docs/plans/).

## License

No license file is currently set, so all rights are reserved by default. If you intend to open the
project for reuse, add a `LICENSE` file with the terms you want.
