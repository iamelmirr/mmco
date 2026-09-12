"""Git checkpoints for the project directory and verification commands."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from .models import VerifyResult
from .utils import MMCOError, run_process, truncate

_LISTING_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}
# Never committed, never shown to the planner as changes, never deleted by a rollback (git clean skips ignored files).
DEFAULT_EXCLUDES = (
    ".venv/", "venv/", "node_modules/", "__pycache__/", "*.pyc", ".pytest_cache/", ".mypy_cache/",
    ".ruff_cache/", ".tox/", ".next/", ".turbo/", ".parcel-cache/", ".DS_Store", ".env", ".env.local", ".env.*.local",
)


class GitError(MMCOError):
    pass


class Workspace:
    """Every task starts from a commit, so a bad attempt can be rolled back and a session replayed."""

    def __init__(self, root: str | Path, enabled: bool = True, protected_paths: Iterable[str | Path] = ()):
        self.root = Path(root).expanduser().resolve()
        self.enabled = enabled
        self.protected_paths = [Path(p).expanduser().resolve() for p in protected_paths]

    # ---- setup ----------------------------------------------------------

    def ensure_repo(self, allow_dirty: bool = False) -> str | None:
        """Initialise git if needed and return the baseline commit."""
        if not self.enabled:
            return None
        if shutil.which("git") is None:
            raise GitError("git is not installed; install it or set MMCO_GIT_CHECKPOINTS=false")
        if self._toplevel() is None:
            logger.info("initialising git repository in {}", self.root)
            self._git("init", "-q")
        self.verify_root()
        if self.head() is None:
            self._git("add", "-A")
            self._git("commit", "-q", "--allow-empty", "-m", "mmco: baseline")
        elif self.is_dirty():
            if not allow_dirty:
                raise GitError(
                    f"{self.root} has uncommitted changes. Commit or stash them, or pass --allow-dirty "
                    "to snapshot them in a commit first."
                )
            self.commit("mmco: snapshot of pre-existing changes")
        return self.head()

    def verify_root(self) -> None:
        if not self.enabled:
            return
        top = self._toplevel()
        if top is None:
            raise GitError(f"{self.root} is not a git repository")
        if top != self.root:
            raise GitError(
                f"{self.root} is inside the git repository {top}. mmco rolls back with `git reset --hard`, "
                "which would affect that whole repository; use its root or a directory outside it."
            )
        self._exclude_protected()

    # ---- checkpoints ----------------------------------------------------

    def head(self) -> str | None:
        if not self.enabled:
            return None
        proc = self._run("rev-parse", "--verify", "-q", "HEAD")
        return proc.stdout.strip() or None

    def is_dirty(self) -> bool:
        return self.enabled and bool(self._git("status", "--porcelain").strip())

    def diff_since(self, commit: str | None) -> tuple[str, str]:
        """Diff of everything (including new files) since `commit`."""
        if not self.enabled or not commit:
            return "", ""
        self._git("add", "-A")
        stat = self._git("diff", "--cached", "--stat", commit)
        diff = self._git("diff", "--cached", commit)
        return stat.strip(), diff

    def commit(self, message: str) -> str | None:
        if not self.enabled:
            return None
        self._git("add", "-A")
        if self._run("diff", "--cached", "--quiet").returncode != 0:
            self._git("commit", "-q", "-m", message)
        return self.head()

    def reset_to(self, commit: str | None) -> None:
        if not self.enabled or not commit:
            return
        logger.info("rolling project back to {}", commit[:10])
        self._git("reset", "-q", "--hard", commit)
        self._git("clean", "-q", "-fd")

    # ---- locking --------------------------------------------------------

    def acquire_lock(self) -> None:
        """Refuse to let two mmco processes drive the same project at once."""
        if not self.enabled:
            return
        lock = self._lock_path()
        if lock.exists():
            try:
                pid = int(lock.read_text().strip() or 0)
            except ValueError:
                pid = 0
            if pid and pid != os.getpid() and _pid_alive(pid):
                raise GitError(f"another mmco process (pid {pid}) is already working in {self.root}")
        lock.write_text(str(os.getpid()))

    def release_lock(self) -> None:
        if not self.enabled:
            return
        lock = self._lock_path()
        try:
            if lock.read_text().strip() == str(os.getpid()):
                lock.unlink()
        except FileNotFoundError:
            pass

    def _lock_path(self) -> Path:
        return self.root / ".git" / "mmco.lock"

    # ---- inspection -----------------------------------------------------

    def listing(self, limit: int = 300) -> str:
        if self.enabled and self._toplevel() == self.root:
            files = [f for f in self._git("ls-files", "--cached", "--others", "--exclude-standard").splitlines() if f]
        else:
            files = []
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = sorted(d for d in dirnames if d not in _LISTING_SKIP_DIRS)
                rel = Path(dirpath).relative_to(self.root)
                files.extend(str(rel / name) if str(rel) != "." else name for name in sorted(filenames))
        if not files:
            return ""
        shown = "\n".join(files[:limit])
        return shown if len(files) <= limit else f"{shown}\n... and {len(files) - limit} more files"

    # ---- internals ------------------------------------------------------

    def _toplevel(self) -> Path | None:
        if not self.root.exists():
            return None
        proc = self._run("rev-parse", "--show-toplevel")
        return Path(proc.stdout.strip()).resolve() if proc.returncode == 0 and proc.stdout.strip() else None

    def _exclude_protected(self) -> None:
        """Keep mmco's own DB and logs, dependency folders and secrets out of commits and out of `git clean`."""
        exclude = self.root / ".git" / "info" / "exclude"
        existing = exclude.read_text(encoding="utf-8").splitlines() if exclude.exists() else []
        additions = [pattern for pattern in DEFAULT_EXCLUDES if pattern not in existing]
        for path in self.protected_paths:
            try:
                rel = path.relative_to(self.root)
            except ValueError:
                continue
            pattern = f"/{rel.as_posix()}*" if path.suffix else f"/{rel.as_posix()}/"
            if pattern not in existing and pattern not in additions:
                additions.append(pattern)
        if additions:
            exclude.parent.mkdir(parents=True, exist_ok=True)
            with exclude.open("a", encoding="utf-8") as fh:
                fh.write("\n# mmco state\n" + "\n".join(additions) + "\n")

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-c", "user.name=mmco", "-c", "user.email=mmco@localhost", "-c", "commit.gpgsign=false", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
        )

    def _git(self, *args: str) -> str:
        proc = self._run(*args)
        if proc.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_verify_commands(commands: list[str], cwd: str | Path, timeout: float) -> list[VerifyResult]:
    results = []
    for command in commands:
        proc = run_process(["/bin/sh", "-c", command], cwd, None, timeout)
        output = (proc.stdout + ("\n" + proc.stderr if proc.stderr else "")).strip()
        result = VerifyResult(
            command=command,
            exit_code=proc.returncode,
            output=truncate(output, 8000),
            timed_out=proc.timed_out,
            duration_ms=proc.duration_ms,
        )
        logger.info("verify {} `{}`", "✓" if result.passed else "✗", command)
        results.append(result)
    return results
