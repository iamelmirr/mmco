"""Sandboxed file tools for the planner, plus their OpenAI tool-call schemas.

The :class:`Toolbox` confines every operation to a single root directory and
exposes read (and optionally write) helpers. :meth:`Toolbox.schemas` produces the
OpenAI function-calling schemas so the planner can invoke these tools.
"""

from __future__ import annotations

from pathlib import Path

from .utils import MMCOError
from .workspace import _LISTING_SKIP_DIRS


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
        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise ToolError(f"path escapes the sandbox: {path}")
        return resolved

    def _relpath(self, target: Path) -> str:
        return target.relative_to(self.root).as_posix()

    # -- read ------------------------------------------------------------

    def read_file(self, path) -> str:
        target = self._resolve(path)
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
        for current, dirnames, filenames in _walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in _LISTING_SKIP_DIRS)
            for filename in sorted(filenames):
                file_path = current / filename
                try:
                    if file_path.stat().st_size > self.max_bytes:
                        continue
                    text = file_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue  # skip binary-looking / unreadable files
                relpath = file_path.relative_to(self.root).as_posix()
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
        target.parent.mkdir(parents=True, exist_ok=True)
        text = content.rstrip("\n") + "\n"
        data = text.encode("utf-8")
        target.write_text(text, encoding="utf-8")
        return f"wrote {path} ({len(data)} bytes)"

    def edit_file(self, path, old, new) -> str:
        if not self.allow_write:
            raise ToolError("write access is disabled (read-only mode)")
        target = self._resolve(path)
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
        except ToolError as e:
            return f"ERROR: {e}"
        except KeyError as e:
            return f"ERROR: missing argument {e} for tool {name!r}"

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
