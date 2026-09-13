"""A deterministic, compact map of a project's files for the planner's prompt."""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

from .workspace import _LISTING_SKIP_DIRS

# Binary or otherwise uninteresting files: no useful text to summarise.
_SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".lock",
    ".woff", ".woff2", ".ico", ".so", ".dylib", ".pyc",
}
_MAX_FILE_BYTES = 500 * 1024
_FIRST_LINE_CHARS = 80


def build_map(root: str | Path, max_files: int = 200, max_symbols: int = 12) -> str:
    """Return a compact per-file summary of the repo at `root`.

    Each line is ``{relpath:<40} ({n} lines)  {symbols_or_firstline}``. Python
    files show their top-level function/class names; other text files show their
    first non-empty line. Files are sorted by relative path and capped at
    `max_files`, with a trailing "... and N more files" line when truncated.
    """
    root = Path(root).expanduser().resolve()
    rels = sorted(_list_files(root))

    lines: list[str] = []
    for rel in rels[:max_files]:
        path = root / rel
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        text = _read_text(path)
        if text is None:
            continue
        n = _line_count(text)
        summary = _summarise(path, text, max_symbols)
        lines.append(f"{rel:<40} ({n} lines)  {summary}")

    extra = len(rels) - max_files
    if extra > 0:
        lines.append(f"... and {extra} more files")

    return "\n".join(lines)


def _list_files(root: Path) -> list[str]:
    """Relative paths of interesting files, git-aware with an os.walk fallback."""
    files = _git_ls_files(root)
    if files is None:
        files = _walk_files(root)
    return [f for f in files if not _skip_file(f)]


def _git_ls_files(root: Path) -> list[str] | None:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            text=True,
        )
    except (OSError, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None
    return [line for line in proc.stdout.splitlines() if line]


def _walk_files(root: Path) -> list[str]:
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _LISTING_SKIP_DIRS)
        rel = Path(dirpath).relative_to(root)
        for name in sorted(filenames):
            files.append(str(rel / name) if str(rel) != "." else name)
    return files


def _skip_file(relpath: str) -> bool:
    return Path(relpath).suffix.lower() in _SKIP_SUFFIXES


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())


def _summarise(path: Path, text: str, max_symbols: int) -> str:
    if path.suffix == ".py":
        symbols = _python_symbols(text, max_symbols)
        if symbols is not None:
            return symbols
    return _first_line(text)


def _python_symbols(text: str, max_symbols: int) -> str | None:
    """Comma-joined top-level def/class names, or None to fall back to first line."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not names:
        return _first_line(text)
    return ", ".join(names[:max_symbols])


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:_FIRST_LINE_CHARS]
    return ""
