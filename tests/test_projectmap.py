from pathlib import Path

from mmco.projectmap import build_map


def test_python_symbols(tmp_path):
    (tmp_path / "a.py").write_text("import os\n\ndef foo(x):\n    return x\n\nclass Bar:\n    def m(self): ...\n")
    out = build_map(tmp_path)
    assert "a.py" in out
    assert "foo" in out and "Bar" in out
    assert "(7 lines)" in out  # line count shown


def test_skips_noise_dirs_and_binaries(tmp_path):
    (tmp_path / ".venv").mkdir(); (tmp_path / ".venv" / "x.py").write_text("def hidden(): ...")
    (tmp_path / "node_modules").mkdir(); (tmp_path / "node_modules" / "y.js").write_text("x")
    (tmp_path / "img.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "main.py").write_text("def real(): ...")
    out = build_map(tmp_path)
    assert "main.py" in out and "hidden" not in out and "node_modules" not in out and "img.png" not in out


def test_non_python_shows_first_line(tmp_path):
    (tmp_path / "README.md").write_text("# My Project\n\nstuff")
    out = build_map(tmp_path)
    assert "README.md" in out
    assert "My Project" in out


def test_limit_caps_output(tmp_path):
    for i in range(50):
        (tmp_path / f"f{i}.py").write_text("def g(): ...")
    out = build_map(tmp_path, max_files=10)
    assert "more files" in out
