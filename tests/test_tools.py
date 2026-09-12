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
