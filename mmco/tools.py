"""Sandboxed file tools for the planner, plus their OpenAI tool-call schemas.

The :class:`Toolbox` confines every operation to a single root directory and
exposes read (and optionally write) helpers. :meth:`Toolbox.schemas` produces the
OpenAI function-calling schemas so the planner can invoke these tools.
"""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path

from .utils import MMCOError
from .workspace import DEFAULT_EXCLUDES, _LISTING_SKIP_DIRS


class ToolError(MMCOError):
    """A tool failed in a way the planner can recover from."""


class Toolbox:
    """File tools confined to ``root``."""

    def __init__(self, root, allow_write: bool = True, max_bytes: int = 200_000) -> None:
        self.root = Path(root).expanduser().resolve()
        self.allow_write = allow_write
        self.max_bytes = max_bytes

    # -- sandbox ---------------------------------------------------------

    def _resolve(self, path) -> Path:
        """Resolve ``path`` relative to root, rejecting absolute paths and escapes."""
        raw = Path(path)
        if raw.is_absolute():
            raise ToolError(f"absolute paths are not allowed: {path}")
        resolved = (self.root / raw).resolve()
        if not resolved.is_relative_to(self.root):
            raise ToolError(f"path escapes the sandbox: {path}")
        return resolved

    # -- exclusion rules -------------------------------------------------

    def _in_protected_dir(self, relpath: str) -> bool:
        """True if any path component is a directory we never touch (``.git``, ``.venv``, ...)."""
        return any(part in _LISTING_SKIP_DIRS for part in Path(relpath).parts)

    def _is_excluded(self, relpath: str) -> bool:
        """Shared rule: is ``relpath`` under a skip-dir or matched by DEFAULT_EXCLUDES?

        This works without git so it also guards non-git directories. It does NOT itself
        consult ``.gitignore``; :meth:`read_file` layers ``git check-ignore`` on top and
        :meth:`search` enumerates via ``git ls-files`` so gitignored files are never scanned.
        """
        if self._in_protected_dir(relpath):
            return True
        parts = Path(relpath).parts
        basename = parts[-1] if parts else relpath
        for pattern in DEFAULT_EXCLUDES:
            trimmed = pattern.rstrip("/")
            if pattern.endswith("/"):
                if trimmed in parts:
                    return True
            elif (
                fnmatch.fnmatch(basename, trimmed)
                or fnmatch.fnmatch(relpath, trimmed)
                or any(fnmatch.fnmatch(part, trimmed) for part in parts)
            ):
                return True
        return False

    def _git_check_ignore(self, relpath: str) -> bool:
        """True if git would ignore ``relpath`` (returns False when not a git repo)."""
        try:
            proc = subprocess.run(
                ["git", "check-ignore", "-q", "--", relpath],
                cwd=self.root, capture_output=True, text=True,
            )
        except OSError:
            return False
        return proc.returncode == 0

    def _git_listing(self) -> list[str] | None:
        """Files git tracks or would show (tracked + untracked, minus ignored); None if not a git repo."""
        try:
            proc = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=self.root, capture_output=True, text=True,
            )
        except OSError:
            return None
        if proc.returncode != 0:
            return None
        return [line for line in proc.stdout.splitlines() if line]

    def _candidate_files(self):
        """Yield sandbox-relative posix paths worth scanning, skipping gitignored/excluded files."""
        tracked = self._git_listing()
        if tracked is not None:
            for rel in tracked:
                if not self._is_excluded(rel):
                    yield rel
            return
        for current, dirnames, filenames in _walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in _LISTING_SKIP_DIRS)
            for filename in sorted(filenames):
                rel = (current / filename).relative_to(self.root).as_posix()
                if not self._is_excluded(rel):
                    yield rel

    # -- read ------------------------------------------------------------

    def read_file(self, path) -> str:
        target = self._resolve(path)
        relpath = target.relative_to(self.root).as_posix()
        if self._is_excluded(relpath) or self._git_check_ignore(relpath):
            raise ToolError(
                f"refusing to read {path}: it is gitignored (may contain secrets or build artifacts)"
            )
        if not target.exists():
            raise ToolError(f"file not found: {path}")
        if not target.is_file():
            raise ToolError(f"not a file: {path}")
        size = target.stat().st_size
        if size > self.max_bytes:
            raise ToolError(f"file too large ({size} bytes > {self.max_bytes}): {path}")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"file is not valid UTF-8 text: {path}")

    def search(self, query, max_results: int = 50) -> str:
        needle = query.lower()
        hits: list[str] = []
        for relpath in self._candidate_files():
            file_path = self.root / relpath
            try:
                # Skip symlinks (or anything) that resolves outside the sandbox.
                if not file_path.resolve().is_relative_to(self.root):
                    continue
                if file_path.stat().st_size > self.max_bytes:
                    continue
                text = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue  # skip binary-looking / unreadable files
            for lineno, line in enumerate(text.splitlines(), start=1):
                if needle in line.lower():
                    hits.append(f"{relpath}:{lineno}: {line.rstrip()}")
                    if len(hits) >= max_results:
                        return "\n".join(hits)
        if not hits:
            return f"no matches for {query!r}"
        return "\n".join(hits)

    def list_dir(self, path=".") -> str:
        target = self._resolve(path)
        if not target.exists():
            raise ToolError(f"directory not found: {path}")
        if not target.is_dir():
            raise ToolError(f"not a directory: {path}")
        entries: list[str] = []
        for child in target.iterdir():
            if child.is_dir():
                if child.name in _LISTING_SKIP_DIRS:
                    continue
                entries.append(f"{child.name}/")
            else:
                entries.append(child.name)
        return "\n".join(sorted(entries))

    # -- write -----------------------------------------------------------

    def write_file(self, path, content) -> str:
        if not self.allow_write:
            raise ToolError("write access is disabled (read-only mode)")
        target = self._resolve(path)
        if self._in_protected_dir(target.relative_to(self.root).as_posix()):
            raise ToolError(f"refusing to write {path}: it is under a protected directory (.git/, .venv/, ...)")
        text = content.rstrip("\n") + "\n"
        data = text.encode("utf-8")
        if len(data) > self.max_bytes:
            raise ToolError(f"content too large ({len(data)} bytes > {self.max_bytes}): {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return f"wrote {path} ({len(data)} bytes)"

    def edit_file(self, path, old, new) -> str:
        if not self.allow_write:
            raise ToolError("write access is disabled (read-only mode)")
        target = self._resolve(path)
        if self._in_protected_dir(target.relative_to(self.root).as_posix()):
            raise ToolError(f"refusing to edit {path}: it is under a protected directory (.git/, .venv/, ...)")
        if not target.is_file():
            raise ToolError(f"file not found: {path}")
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"file is not valid UTF-8 text: {path}")
        count = text.count(old)
        if count == 0:
            raise ToolError(f"old text not found in {path}")
        if count > 1:
            raise ToolError(
                f"old text is not unique in {path} ({count} occurrences); "
                "add more surrounding context to make it unique"
            )
        updated = text.replace(old, new)
        updated = updated.rstrip("\n") + "\n"
        data = updated.encode("utf-8")
        if len(data) > self.max_bytes:
            raise ToolError(f"content too large ({len(data)} bytes > {self.max_bytes}): {path}")
        target.write_text(updated, encoding="utf-8")
        return f"edited {path} ({len(data)} bytes)"

    # -- dispatch --------------------------------------------------------

    def dispatch(self, name, arguments: dict) -> str:
        methods = {
            "read_file": lambda a: self.read_file(a["path"]),
            "search": lambda a: self.search(a["query"], a.get("max_results", 50)),
            "list_dir": lambda a: self.list_dir(a.get("path", ".")),
            "write_file": lambda a: self.write_file(a["path"], a["content"]),
            "edit_file": lambda a: self.edit_file(a["path"], a["old"], a["new"]),
        }
        method = methods.get(name)
        if method is None:
            return f"ERROR: unknown tool {name!r}"
        try:
            return method(arguments)
        except KeyError as e:
            return f"ERROR: missing argument {e} for tool {name!r}"
        except (ToolError, TypeError, ValueError, OSError) as e:
            return f"ERROR: {e}"

    # -- schemas ---------------------------------------------------------

    @staticmethod
    def schemas(allow_write: bool = True) -> list[dict]:
        schemas: list[dict] = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a UTF-8 text file within the sandbox and return its contents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path relative to the sandbox root.",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Case-insensitive substring search across text files in the sandbox.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The substring to search for.",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of matching lines to return.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List the entries of a directory within the sandbox.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path relative to the sandbox root (default '.').",
                            }
                        },
                        "required": [],
                    },
                },
            },
        ]
        if allow_write:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "description": "Create or overwrite a file within the sandbox.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {
                                        "type": "string",
                                        "description": "Path relative to the sandbox root.",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "The full file contents to write.",
                                    },
                                },
                                "required": ["path", "content"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "description": "Replace a unique snippet of text in an existing file within the sandbox.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {
                                        "type": "string",
                                        "description": "Path relative to the sandbox root.",
                                    },
                                    "old": {
                                        "type": "string",
                                        "description": "The exact text to replace; must occur exactly once.",
                                    },
                                    "new": {
                                        "type": "string",
                                        "description": "The replacement text.",
                                    },
                                },
                                "required": ["path", "old", "new"],
                            },
                        },
                    },
                ]
            )
        return schemas


def _walk(root: Path):
    """os.walk over a Path, yielding (Path, dirnames, filenames)."""
    import os

    for current, dirnames, filenames in os.walk(root):
        yield Path(current), dirnames, filenames
