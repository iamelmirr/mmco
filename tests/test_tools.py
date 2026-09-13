import subprocess

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


def test_read_only_mode_blocks_edits(tmp_path):
    (tmp_path / "x.py").write_text("a\n")
    box = Toolbox(tmp_path, allow_write=False)
    with pytest.raises(ToolError):
        box.edit_file("x.py", "a", "b")


def test_search_does_not_follow_symlink_outside_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOPSECRET value")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link.txt").symlink_to(outside / "secret.txt")
    box = Toolbox(root)
    # A hit would render as "link.txt:1: ...", so the symlinked file's path
    # must not appear in the results.
    assert "link.txt" not in box.search("TOPSECRET")


def test_dispatch_handles_bad_arg_types(box):
    # "line" matches a.py, so the bad max_results type is exercised in the cap check.
    result = box.dispatch("search", {"query": "line", "max_results": "5"})
    assert isinstance(result, str) and result.startswith("ERROR:")


def test_write_too_large_raises(tmp_path):
    box = Toolbox(tmp_path, max_bytes=10)
    with pytest.raises(ToolError):
        box.write_file("big.py", "x" * 100)


def test_edit_too_large_raises(tmp_path):
    (tmp_path / "big.py").write_text("small\n")
    box = Toolbox(tmp_path, max_bytes=10)
    with pytest.raises(ToolError):
        box.edit_file("big.py", "small", "x" * 100)


def _init_git(root):
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)


def _make_secret_tree(root):
    (root / ".env").write_text("SECRET=sk-live-xyz\n")
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "x.py").write_text("SECRET=sk-live-xyz\n")
    (root / "app.py").write_text("value = 'findme'\n")


# ---- FIX 1: file tools must not leak gitignored secrets ----------------------

def test_search_skips_secrets_in_git_repo(tmp_path):
    _init_git(tmp_path)
    (tmp_path / ".gitignore").write_text(".env\n.venv/\n")
    _make_secret_tree(tmp_path)
    subprocess.run(["git", "add", "app.py", ".gitignore"], cwd=tmp_path, check=True)
    box = Toolbox(tmp_path)
    hits = box.search("SECRET")
    assert ".env" not in hits and ".venv" not in hits
    assert "findme" in box.search("findme")  # a normal file is still searched


def test_search_skips_secrets_without_git(tmp_path):
    _make_secret_tree(tmp_path)
    box = Toolbox(tmp_path)
    hits = box.search("SECRET")
    assert ".env" not in hits and ".venv" not in hits
    assert "findme" in box.search("findme")


def test_read_file_refuses_gitignored_secret(tmp_path):
    _init_git(tmp_path)
    (tmp_path / ".gitignore").write_text(".env\n")
    _make_secret_tree(tmp_path)
    subprocess.run(["git", "add", "app.py", ".gitignore"], cwd=tmp_path, check=True)
    box = Toolbox(tmp_path)
    with pytest.raises(ToolError):
        box.read_file(".env")
    assert "findme" in box.read_file("app.py")


def test_read_file_refuses_excluded_without_git(tmp_path):
    _make_secret_tree(tmp_path)
    box = Toolbox(tmp_path)
    with pytest.raises(ToolError):
        box.read_file(".env")
    with pytest.raises(ToolError):
        box.read_file(".venv/lib/x.py")
    assert "findme" in box.read_file("app.py")


# ---- FIX 2: dispatch must not crash on OSError -------------------------------

def test_dispatch_survives_oserror(tmp_path):
    box = Toolbox(tmp_path)
    result = box.dispatch("write_file", {"path": "", "content": "x"})
    assert isinstance(result, str) and result.startswith("ERROR:")


# ---- FIX 3: deny writes into .git/ and skip-dirs ----------------------------

def test_write_denied_into_git_dir(tmp_path):
    _init_git(tmp_path)
    box = Toolbox(tmp_path)
    with pytest.raises(ToolError):
        box.write_file(".git/config", "x")
    box.write_file("sub/ok.py", "x")
    assert (tmp_path / "sub" / "ok.py").exists()


def test_edit_denied_into_git_dir(tmp_path):
    _init_git(tmp_path)
    box = Toolbox(tmp_path)
    with pytest.raises(ToolError):
        box.edit_file(".git/config", "a", "b")


def test_openai_schemas_shape():
    schemas = Toolbox.schemas(allow_write=True)
    names = {s["function"]["name"] for s in schemas}
    assert {"read_file", "search", "list_dir", "write_file", "edit_file"} <= names
    assert Toolbox.schemas(allow_write=False) and all(
        s["function"]["name"] in {"read_file", "search", "list_dir"} for s in Toolbox.schemas(allow_write=False))
