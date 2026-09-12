"""User-editable planner rules and prompt overrides, global and per project.

Everything lives outside the project directory (in ~/.config/mmco) so the coding agent,
which can read any file in the project, never sees the rules or the planner's instructions.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

from .config import config_dir
from .utils import MMCOError, PROMPTS_DIR

ExecutorContext = Literal["full", "task", "minimal"]
EXECUTOR_CONTEXTS: tuple[str, ...] = get_args(ExecutorContext)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

RULES_TEMPLATE = """---
# executor_context: how much Claude Code is told
#   full    - overall goal, the whole plan, your answers, and its task
#   task    - only its own technical task (no goal, no plan, no discussion)
#   minimal - only the instructions the planner wrote, with nothing about mmco or the project
executor_context: {context}
---
<!--
{scope} rules for the planner (DeepSeek). Write them below in plain language, outside this comment.
They are added to every planner prompt: planning, reviewing Claude's work, rewriting prompts and the
final review. Changes apply immediately, even to a running session.

Examples:
- Give Claude purely technical instructions: exact files, function signatures, data shapes and expected behaviour.
- Never describe the product, its users or the business purpose to Claude.
- Allow Claude to choose implementation details inside each task.
- Use TypeScript and pnpm; never introduce new dependencies without listing them in the task.
-->
"""


class RulesError(MMCOError):
    pass


@dataclass
class Rules:
    text: str = ""
    options: dict[str, str] = field(default_factory=dict)
    sources: list[Path] = field(default_factory=list)

    @property
    def executor_context(self) -> ExecutorContext | None:
        value = self.options.get("executor_context")
        return value if value in EXECUTOR_CONTEXTS else None  # type: ignore[return-value]


def global_rules_path() -> Path:
    return config_dir() / "rules.md"


def project_config_dir(project_dir: str | Path) -> Path:
    root = Path(project_dir).expanduser().resolve()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", root.name).strip("-") or "project"
    digest = hashlib.sha1(str(root).encode()).hexdigest()[:8]
    return config_dir() / "projects" / f"{slug}-{digest}"


def project_rules_path(project_dir: str | Path) -> Path:
    return project_config_dir(project_dir) / "rules.md"


def prompt_dirs(project_dir: str | Path | None, extra: str | None = None) -> list[Path]:
    """Directories searched for prompt overrides, most specific first."""
    dirs = []
    if project_dir:
        dirs.append(project_config_dir(project_dir) / "prompts")
    dirs.append(config_dir() / "prompts")
    if extra:
        dirs.append(Path(extra).expanduser())
    return dirs


def parse_rules(text: str, source: Path | None = None) -> tuple[dict[str, str], str]:
    options: dict[str, str] = {}
    body = text
    if text.lstrip().startswith("---"):
        _, _, rest = text.lstrip().partition("---")
        header, sep, body = rest.partition("\n---")
        if not sep:
            raise RulesError(f"{source or 'rules'}: front matter starts with --- but is never closed")
        for line in header.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            key, colon, value = line.partition(":")
            if not colon:
                raise RulesError(f"{source or 'rules'}: expected 'key: value' in front matter, got '{line}'")
            options[key.strip()] = value.strip()
    context = options.get("executor_context")
    if context is not None and context not in EXECUTOR_CONTEXTS:
        raise RulesError(
            f"{source or 'rules'}: executor_context must be one of {', '.join(EXECUTOR_CONTEXTS)}, got '{context}'"
        )
    body = _COMMENT.sub("", body).strip()
    return options, body


def load_rules(project_dir: str | Path | None) -> Rules:
    """Global rules first, then project rules; project options win."""
    rules = Rules()
    paths = [global_rules_path()] + ([project_rules_path(project_dir)] if project_dir else [])
    parts = []
    for path in paths:
        if not path.is_file():
            continue
        options, body = parse_rules(path.read_text(encoding="utf-8"), path)
        rules.options.update(options)
        rules.sources.append(path)
        if body:
            parts.append(body)
    rules.text = "\n\n".join(parts)
    return rules


def ensure_rules_file(path: Path, scope: str, context: str = "full") -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(RULES_TEMPLATE.format(scope=scope, context=context), encoding="utf-8")
    return path


def eject_prompts(target: Path, overwrite: bool = False) -> list[Path]:
    """Copy the built-in prompts somewhere editable."""
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for source in sorted(PROMPTS_DIR.glob("*.txt")):
        destination = target / source.name
        if destination.exists() and not overwrite:
            continue
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(destination)
    return written
