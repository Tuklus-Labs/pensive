"""Ref-to-filesystem-path resolution for serve-time enrichment."""
import os
import signal
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from recall.refs import openRef, refToPath

_HOME = Path("/fake/home")

# What an attacker is trying to get the daemon to read back to them.
_SECRET = "AWS_SECRET_ACCESS_KEY=phosphor-7482-not-a-real-key\n"
_INNOCENT = "listen = 0.0.0.0:5999\n"


class _Blocked(Exception):
    """Raised by the alarm below. Deliberately NOT an OSError.

    TimeoutError, the obvious choice, IS an OSError subclass, so refs.openRef's
    own ``except (OSError, ValueError): return None`` swallows it and the test
    sees a tidy None -- the guard reports success because the code it is
    guarding caught the alarm. Found by the sabotage gate: removing O_NONBLOCK
    left the suite green at 42 passed, ten seconds slower than usual and
    silent about why.
    """


@contextmanager
def _mustNotBlock(seconds=5):
    """Turn a hang into a failure.

    A fifo opened O_RDONLY blocks until a writer appears and no writer ever
    appears in a test, so an unguarded fifo case does not fail the suite, it
    STOPS it -- on the live daemon that is the recall serving thread, parked
    forever on a path a poisoned ref chose. Measured against the pre-fix code:
    is_file() reported True on a regular file, the file was swapped for a
    fifo, and read_text() never returned.
    """
    def onAlarm(sig, frame):
        raise _Blocked(f"blocked for more than {seconds}s")

    previous = signal.signal(signal.SIGALRM, onAlarm)
    signal.alarm(seconds)
    try:
        yield
    except _Blocked as exc:
        pytest.fail(f"call blocked instead of refusing: {exc}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _homeTree(tmp_path):
    """(home, outside): a populated projects root and a secret beyond it."""
    home = tmp_path / "home"
    (home / "Projects" / "pkg").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.env").write_text(_SECRET)
    return home, outside


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
        "claude-home//home/user/.ssh/id_ed25519", home=_HOME) is None


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


# ---- openRef: the read path ----------------------------------------------- #
#
# refToPath answers "what does this ref NAME", which is a question about a
# string. Every test below is about the different question the reader has to
# ask -- "what does this descriptor HOLD" -- because a name is re-walked by
# the kernel on every syscall and can mean a different file each time.
#
# The shape of each attack test is the shape of the bug: CHECK (refToPath
# validates and is asserted to pass, so the test cannot silently degrade into
# proving the ref was rejected up front), then the SWAP an attacker with write
# access to a parent directory can perform, then the USE.


def test_openRef_reads_a_contained_file(tmp_path):
    # Regression guard: the fix must not turn ordinary enrichment off.
    home, _ = _homeTree(tmp_path)
    (home / "Projects" / "pkg" / "api.go").write_text("package pkg\n")
    with openRef("projects/pkg/api.go#c0", home=home) as fh:
        assert fh.read() == b"package pkg\n"


def test_openRef_leaf_swapped_after_validation_is_not_read(tmp_path):
    # The window the old code left open: validated as an ordinary file inside
    # the root, replaced by a link out of it before the read.
    home, outside = _homeTree(tmp_path)
    leaf = home / "Projects" / "pkg" / "cfg.env"
    leaf.write_text("innocent = true\n")
    assert refToPath("projects/pkg/cfg.env", home=home) is not None
    leaf.unlink()
    leaf.symlink_to(outside / "secret.env")
    assert openRef("projects/pkg/cfg.env", home=home) is None


def test_openRef_parent_swapped_after_validation_is_not_read(tmp_path):
    # The variant a leaf-only O_NOFOLLOW does NOT catch: the final component
    # stays an ordinary file, and the DIRECTORY above it becomes the link. Only
    # a walk anchored at the root refuses this one.
    home, outside = _homeTree(tmp_path)
    pkg = home / "Projects" / "pkg"
    (pkg / "cfg.env").write_text("innocent = true\n")
    (outside / "cfg.env").write_text(_SECRET)
    assert refToPath("projects/pkg/cfg.env", home=home) is not None
    (pkg / "cfg.env").unlink()
    pkg.rmdir()
    pkg.symlink_to(outside)
    assert openRef("projects/pkg/cfg.env", home=home) is None


def test_openRef_refuses_a_parent_created_after_validation(tmp_path):
    # resolve() is non-strict, so a ref naming a path that does not exist YET
    # validates as contained. The attacker does not have to swap anything: they
    # create the missing parent afterwards, pointing out of the root.
    home, outside = _homeTree(tmp_path)
    (outside / "cfg.env").write_text(_SECRET)
    assert refToPath("projects/later/cfg.env", home=home) is not None
    (home / "Projects" / "later").symlink_to(outside)
    assert openRef("projects/later/cfg.env", home=home) is None


def test_openRef_refuses_a_fifo_without_blocking(tmp_path):
    # A fifo is the denial of service hiding inside the file-type check:
    # opening one for reading blocks until a writer appears, so the refusal has
    # to survive the OPEN, not merely follow it.
    home, _ = _homeTree(tmp_path)
    os.mkfifo(home / "Projects" / "pkg" / "pipe.env")
    with _mustNotBlock():
        assert openRef("projects/pkg/pipe.env", home=home) is None


def test_openRef_refuses_a_directory(tmp_path):
    home, _ = _homeTree(tmp_path)
    assert openRef("projects/pkg", home=home) is None


def test_openRef_enforces_the_byte_cap_on_the_open_descriptor(tmp_path):
    home, _ = _homeTree(tmp_path)
    big = home / "Projects" / "pkg" / "big.bin"
    big.write_bytes(b"x" * 2048)
    assert openRef("projects/pkg/big.bin", home=home, maxBytes=1024) is None
    with openRef("projects/pkg/big.bin", home=home, maxBytes=4096) as fh:
        assert len(fh.read()) == 2048


def test_openRef_missing_file_returns_none(tmp_path):
    home, _ = _homeTree(tmp_path)
    assert openRef("projects/pkg/gone.go", home=home) is None


def test_openRef_shares_the_ref_guards(tmp_path):
    # openRef re-uses refToPath's parsing rather than restating it, so every
    # ref refToPath refuses is refused here too. Asserted rather than assumed:
    # a fix that opened first and parsed second would pass every test above.
    home, _ = _homeTree(tmp_path)
    (home / "Projects" / "pkg" / "api.go").write_text("package pkg\n")
    assert openRef("kv_cache/vector_meta.db#rowid=7", home=home) is None
    assert openRef("projects/../../etc/passwd", home=home) is None
    assert openRef("projects//etc/passwd", home=home) is None
    assert openRef("projects/", home=home) is None
    assert openRef("", home=home) is None
    assert openRef(None, home=home) is None
    assert openRef("projects/pkg/cfg\x00.env", home=home) is None


def test_openRef_reads_through_a_symlinked_root(tmp_path):
    # The ROOT may be reached through a symlink: home is configuration, not ref
    # data, so it is followed. Only the components the REF supplies are refused,
    # which is what keeps a symlinked $HOME working (parity with refToPath's
    # both-sides-resolved containment).
    real = tmp_path / "real"
    (real / "Projects" / "pkg").mkdir(parents=True)
    (real / "Projects" / "pkg" / "api.go").write_text("package pkg\n")
    home = tmp_path / "home"
    home.symlink_to(real)
    with openRef("projects/pkg/api.go#c3", home=home) as fh:
        assert fh.read() == b"package pkg\n"


def test_openRef_has_no_window_between_validating_and_reading(tmp_path):
    """The bug itself, rather than the shapes it takes: a NAME is re-walked by
    the kernel on every syscall, so a reader that validates one lookup and
    reads another can be aimed between the two.

    A writer flips the leaf between an ordinary in-root file and a link to the
    secret while the reader runs. The reader may return the innocent bytes or
    refuse; it may never return the secret. This is the one test here that
    grades the WINDOW instead of the post-swap state, so it is also the only
    one an externally staged swap cannot express -- the swap has to land while
    the call is in flight.

    Can only fail in the true direction: a run where the flip never lands
    inside the window passes, and no run where it does can pass a reader that
    holds one descriptor. Measured against the pre-fix shape, the secret came
    back within the first few dozen attempts.
    """
    home, outside = _homeTree(tmp_path)
    pkg = home / "Projects" / "pkg"
    cfg = pkg / "cfg.env"
    stage = pkg / ".stage"
    cfg.write_text(_INNOCENT)

    stop = threading.Event()

    def flipper():
        # os.replace is atomic at the cfg name, so the reader always sees one
        # or the other, never a partially written file.
        while not stop.is_set():
            stage.write_text(_INNOCENT)
            os.replace(stage, cfg)
            os.symlink(outside / "secret.env", stage)
            os.replace(stage, cfg)

    writer = threading.Thread(target=flipper, daemon=True)
    writer.start()
    try:
        for _ in range(2000):
            fh = openRef("projects/pkg/cfg.env", home=home)
            if fh is None:
                continue
            with fh:
                assert _SECRET not in fh.read().decode("utf-8", "replace")
    finally:
        stop.set()
        writer.join(timeout=5)


def test_openRef_refuses_a_symlink_that_stays_inside_the_root(tmp_path):
    # DOCUMENTED NARROWING, asserted so it stays deliberate: a symlink whose
    # target is also inside the root used to be readable (refToPath resolves
    # and finds it contained) and is not any more. Telling that link apart from
    # one swapped a microsecond later requires re-validating after resolution,
    # which is the window this whole module exists to close.
    #
    # Priced against the live store 2026-08-13: 0 of 279,818 resolvable live
    # chunk source_refs traverse a symlink at any component, so the narrowing
    # costs no enrichment that exists today.
    home, _ = _homeTree(tmp_path)
    (home / "Projects" / "pkg" / "real.go").write_text("package pkg\n")
    (home / "Projects" / "pkg" / "link.go").symlink_to(
        home / "Projects" / "pkg" / "real.go")
    assert openRef("projects/pkg/link.go", home=home) is None
