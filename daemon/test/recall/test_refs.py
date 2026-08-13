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


def test_absolute_rest_rejected():
    # The doubled slash makes the rest an ABSOLUTE path, and joining an
    # absolute right operand makes pathlib DISCARD everything on the left:
    # root / rel / "/etc/passwd" is just "/etc/passwd". The .. guard never
    # fires because no segment is ".."; the ref walks straight out of the
    # root while still looking like a well-formed projects/ ref.
    assert refToPath("projects//etc/passwd", home=_HOME) is None
    assert refToPath("codex-home//etc/shadow#c0", home=_HOME) is None
    # Rest of "/" alone collapses the whole ref to the filesystem root.
    assert refToPath("projects//", home=_HOME) is None


def test_absolute_rest_rejected_even_when_it_lands_inside_root(tmp_path):
    # An absolute rest that happens to name a file INSIDE the root passes the
    # containment check, so containment alone would admit it. It is still not
    # a ref: the convention is root-relative, and accepting the absolute
    # spelling gives one file two refs. enrich echoes the ref text straight
    # back into its output ("at <base>#L..."), so the spelling is not private
    # to this module.
    home = tmp_path / "home"
    (home / "Projects" / "pkg").mkdir(parents=True)
    inside = home / "Projects" / "pkg" / "api.go"
    inside.write_text("package pkg\n")
    assert refToPath(f"projects/{inside}", home=home) is None


def test_absolute_rest_cannot_reach_secrets():
    # The reachable form of the bug: enrich reads the resolved path, so an
    # absolute rest turns a planted sourceRef into a read of any file the
    # daemon user can open.
    assert refToPath(
        "claude-home//home/aegis/.ssh/id_ed25519", home=_HOME) is None


def test_symlink_out_of_root_rejected(tmp_path):
    # The second escape route: no ".." and no absolute rest, but the named
    # file is a symlink whose target lives outside the root, so the read
    # still lands outside. Containment has to be judged after resolution,
    # not on the joined string.
    home = tmp_path / "home"
    (home / "Projects" / "pkg").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.env").write_text("token\n")
    (home / "Projects" / "pkg" / "cfg.env").symlink_to(outside / "secret.env")
    assert refToPath("projects/pkg/cfg.env", home=home) is None


def test_unresolvable_ref_returns_none_without_raising(tmp_path):
    # A ref is store data, so it can hold bytes no path can: an embedded NUL
    # makes the OS path calls raise ValueError. The caller (enrich.lines)
    # invokes refToPath OUTSIDE its try, so a raise here would break recall
    # enrichment outright rather than degrading to "no attachment".
    assert refToPath("projects/\x00etc/passwd", home=_HOME) is None
    assert refToPath("projects/pkg/cfg\x00.env", home=tmp_path) is None


def test_symlinked_root_still_resolves(tmp_path):
    # Containment is checked with BOTH sides resolved, so a home (or a root)
    # that is itself a symlink stays usable: the file is really inside the
    # root, it just reaches it through a link.
    real = tmp_path / "real"
    (real / "Projects" / "pkg").mkdir(parents=True)
    (real / "Projects" / "pkg" / "api.go").write_text("package pkg\n")
    home = tmp_path / "home"
    home.symlink_to(real)
    assert refToPath("projects/pkg/api.go#c3", home=home) == \
        home / "Projects/pkg/api.go"
