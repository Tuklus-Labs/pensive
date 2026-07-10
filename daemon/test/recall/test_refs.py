"""Ref-to-filesystem-path resolution for serve-time enrichment."""
from pathlib import Path

from recall.refs import refToPath

_HOME = Path("/fake/home")


def test_projects_root():
    assert refToPath("projects/obol/api/rate.go#c11", home=_HOME) == \
        _HOME / "Projects/obol/api/rate.go"


def test_reference_library_root():
    assert refToPath("reference-library/53-hw.md#c2", home=_HOME) == \
        _HOME / "Projects/Aegis/AEGIS/docs/reference-library/53-hw.md"


def test_dotfile_roots():
    assert refToPath("claude-home/hooks/emit.py", home=_HOME) == \
        _HOME / ".claude/hooks/emit.py"
    assert refToPath("codex-home/config.toml#c0", home=_HOME) == \
        _HOME / ".codex/config.toml"


def test_unknown_root_and_junk_return_none():
    assert refToPath("kv_cache/vector_meta.db#rowid=7", home=_HOME) is None
    assert refToPath("", home=_HOME) is None
    assert refToPath(None, home=_HOME) is None


def test_traversal_rejected():
    # A ref must never escape its root: reject any .. segment outright.
    assert refToPath("projects/../../etc/passwd", home=_HOME) is None
    assert refToPath("projects/x/../../../etc/shadow#c1", home=_HOME) is None


def test_root_prefix_with_empty_rest_returns_none():
    # "projects/" alone is a bare root prefix with nothing after it: no file
    # named, so there is nothing to resolve.
    assert refToPath("projects/", home=_HOME) is None
