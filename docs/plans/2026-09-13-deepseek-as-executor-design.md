# DeepSeek as a co-executor

Date: 2026-09-13
Status: design agreed, not yet implemented

## Problem

Today DeepSeek only plans and reviews; Claude Code executes every task. Two gaps:

1. **DeepSeek does not know the code.** The planner sees only a file listing, not file
   contents, so on an existing project it can plan duplicates, target the wrong files, or
   ignore conventions. With `task`/`minimal` context (which we want as the norm) every task
   must be self-contained, and DeepSeek cannot write self-contained tasks without reading code.
2. **DeepSeek cannot do work itself.** Some tasks (docs, content, analysis, small edits) are
   wasteful to send to Claude. DeepSeek should be able to do them directly and keep steering
   Claude for real coding — a back-and-forth where the boss also gets its hands dirty.

## Roles

- **DeepSeek — the boss.** Plans, reads code, reviews, and now also executes some tasks itself
  and decides, per task, at run time, who does the next step.
- **Claude — the builder.** Executes technical tasks when DeepSeek sends it, as today. Claude
  still makes implementation decisions inside a task; it is not reduced to boilerplate.

## Knowledge asymmetry (context levels)

Default becomes `task` (not `minimal`).

| | DeepSeek | Claude (`task`, default) |
| --- | --- | --- |
| Original user request | yes | no |
| Whole plan / other tasks | yes | no |
| Project map + read any file | yes | only files the task names |
| User rules and answers | yes | no |
| Intent / why | yes | no |
| **How to implement this task** | fixes only cross-task contracts | **Claude decides** |

`minimal` stays as an option for deliberately narrow, literal work. `full` stays for cases
where Claude benefits from the whole picture (large refactors).

Division is by **level**, not "who thinks vs who types":

| Decision level | Who |
| --- | --- |
| What to build, order, how parts fit | DeepSeek |
| Contracts crossing task boundaries (file names, signatures, API shape) | DeepSeek fixes |
| How a task is implemented internally | Claude |
| Whether a result is good enough | DeepSeek (+ checks) |

Rule of thumb: **boundaries fixed, insides free.**

## How DeepSeek sees the project

**Layer 1 — the map (cheap, always).** Before planning, mmco builds a short map and adds it to
the planner prompt: per file, line count and top-level `def`/`class` names (Python via `ast`;
other languages via first line + regex signatures). Deterministic, free, fast.

**Layer 2 — read tools (on demand).** DeepSeek gets read-only tools via OpenRouter tool-calling
(all DeepSeek models support tools): `read_file`, `search`, `list_dir`. It calls them as needed
before returning a plan or a verdict. The same machinery powers `write_file` / `edit_file` when
DeepSeek executes a task itself.

All tools are confined to `project_dir` (the git root); paths are normalised to block `../`
escape. No shell execution for DeepSeek — file tools only (YAGNI).

## The loop (one task)

```
task up next
   │ DeepSeek decides executor  -> {"executor": "planner"|"claude", "reason": ...}
   │
   ├─ planner: tool loop (read -> write/edit -> read ...) until it says done
   │           all writes land in project_dir
   └─ claude:  existing flow (prompt, subprocess, --resume)
   │
   git diff since checkpoint         <- same for both
   verify + regression checks        <- same for both
   DeepSeek evaluates the result (whoever produced it)
      -> continue / retry / reformulate / take_over / add_task / clarify / stuck
```

**`take_over`** is a new evaluation outcome: DeepSeek finishes a task Claude left almost done
(a small remainder, or Claude stuck on something trivial) instead of retrying in circles.

## Gates unchanged (flexibility must not break safety)

Same for both executors: git checkpoint before each task; `git reset` on failure/rollback;
checks decide "done" (a task whose checks fail can never be done, whoever built it); regression
checks; escalation instead of silent failure; final review; SQLite state; resume; dashboard.

**Loop guards:** DeepSeek's tool loop has a call-count and cost budget per task (like Claude's
budget); overrun -> treated as stuck -> escalation. A per-task hand-off counter (e.g. 3) stops
ping-pong between executors -> escalation.

## Choosing the executor

Decided at run time, not hard-coded. The planner prompt carries a guideline, editable via
`rules.md`:

> Do it yourself when the task is text, config, analysis or a small edit. Send Claude for real
> code development where implementation decisions are made.

## Data model changes

Migration 3: `executions.agent` (`claude` | `planner`), `tasks.executor` (the decision).
Existing rows default to `claude`.

## Implementation order

Each step leaves a working, tested system.

1. **Project map** — new `mmco/projectmap.py`; inject into planner prompt. Pure addition.
2. **DeepSeek tools** — new `mmco/tools.py` (read_file, search, list_dir, write_file, edit_file;
   confined to project_dir). Tool-calling loop in `planner.py`.
3. **Data model** — migration 3 (`executions.agent`, `tasks.executor`).
4. **Executor per task** — branch in `orchestrator._execute`: planner runs via tools, or Claude
   as today. Same checks and checkpoint for both.
5. **`take_over`** — new branch in `_dispatch` + hand-off counter + escalation.
6. **Default `task` context** and executor-choice guideline in prompt and `rules.md`.
7. **Dashboard** — executor label on the timeline track; filter by executor in history.
8. **README + tests** throughout (mock tools; no real calls in CI).

## Out of scope (YAGNI)

Parallel executors; a third model; DeepSeek running shell commands (file tools only). Revisit
if needed.
