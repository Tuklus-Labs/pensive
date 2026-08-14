"""Ref-to-filesystem-path resolution: the serve-time inverse of the import
waves' ref convention.

A source_ref names a file relative to one of the known import roots plus an
optional ``#<fragment>`` (chunk index). This module maps a ref back to the
filesystem. Two entry points, and the difference between them is the whole
security story:

- ``refToPath`` answers what a ref NAMES. That is a question about a string.
- ``openRef`` answers what a ref currently HOLDS, and hands back a descriptor.

A name is not a capability. The kernel re-walks it on every syscall, so a
caller that validates a name and then reads it has asked two questions of two
different lookups, and anything with write access to a parent directory
between them chooses what the second one answers. Refs are data from the
store, so that is a reachable position, not a theoretical one. Every READ goes
through ``openRef``; ``refToPath`` is for callers that only need the name.

Unknown roots (including the kv_cache refs Phase A could not recover), refs
containing a ``..`` segment, refs whose rest is absolute, and refs that land
outside their root all yield nothing from either entry point.
"""
import os
import stat
from pathlib import Path

__all__ = ["refToPath", "openRef"]

# refRoot -> path relative to $HOME. Ordered; first prefix match wins.
_ROOTS = (
    ("projects/", "Projects"),
    ("reference-library/", "Projects/Aegis/AEGIS/docs/reference-library"),
    ("claude-home/", ".claude"),
    ("codex-home/", ".codex"),
)

# O_NOFOLLOW makes a symlink an ERROR rather than a redirection, which is what
# turns the walk below into a containment proof.
#
# O_NONBLOCK is not an optimization: opening a fifo O_RDONLY BLOCKS until a
# writer appears, and for a fifo named by a poisoned ref no writer ever
# appears. Without it the file-type check below is unreachable -- the daemon
# never gets there, because it is parked inside open() on the recall serving
# thread. Measured on the pre-fix code: a regular file passed is_file(), was
# swapped for a fifo, and read_text() never returned. On a regular file the
# flag has no effect on the open or on any read that follows.
_LEAF_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK

# O_DIRECTORY on the interior components fails a non-directory at the
# component that is wrong, rather than several syscalls later with ENOTDIR
# against a name that no longer explains anything.
_WALK_FLAGS = _LEAF_FLAGS | os.O_DIRECTORY


def _contains(top, path):
    """True if path is still inside top once both sides are fully resolved.

    Resolving is what catches the escapes the joined string hides: a file
    inside the root that is a symlink pointing out of it reads from outside
    the root while its ref looks ordinary. Both sides are resolved so a root
    reached through a symlinked $HOME still compares equal to itself.

    resolve() is non-strict, so a ref naming a file that does not exist is
    still contained (missing files remain the caller's problem). It does
    raise on a ref no path can express, such as an embedded NUL: that is
    reported as not-contained rather than allowed to propagate. Both entry
    points are called outside their caller's try, so a raise here costs the
    whole enrichment rather than one attachment.

    Sound for a name and useless for a read: by the time the caller acts on
    the answer, the answer describes a lookup that has already happened. That
    is why openRef does not use it, and why the containment openRef relies on
    is the walk rather than this.
    """
    try:
        return path.resolve().is_relative_to(top.resolve())
    except (OSError, ValueError):
        return False


def _rootFor(ref, home):
    """``(top, rest)`` for a well-formed ref under a known root, else None.

    Shared by both entry points on purpose. These guards are about the ref's
    SHAPE, so they hold no matter what the filesystem looks like, and a guard
    that lived in only one entry point would be a guard the other caller
    silently does without.
    """
    if not ref:
        return None
    base = ref.split("#", 1)[0]
    if not base:
        return None
    parts = base.split("/")
    if ".." in parts:
        return None
    root = home if home is not None else Path.home()
    for prefix, rel in _ROOTS:
        if base.startswith(prefix):
            rest = base[len(prefix):]
            if not rest or rest.startswith("/"):
                # An absolute rest is the ..-free way out of the root: a
                # doubled slash ("projects//etc/passwd") leaves rest holding
                # "/etc/passwd", and pathlib DISCARDS the left operand when
                # the right one is absolute, so root / rel / rest collapses
                # to "/etc/passwd". The .. guard never fires, because no
                # segment is "..". An empty rest names no file at all.
                return None
            return root / rel, rest
    return None


def refToPath(ref, home=None):
    """Absolute Path for a store ref, or None for unknown/unsafe refs.

    NAMES a file. It does not open one, and the Path it returns is not
    permission to read that name later: see the module docstring. Callers that
    intend to read call openRef instead.
    """
    split = _rootFor(ref, home)
    if split is None:
        return None
    top, rest = split
    path = top / rest
    if not _contains(top, path):
        return None
    return path


def openRef(ref, home=None, maxBytes=None):
    """Open the file a ref names, read-only and binary, or None.

    The caller owns the returned handle and must close it (it is a context
    manager). None means "no such readable file", which every caller here
    treats as "no attachment", never as an error.

    Containment is STRUCTURAL rather than checked. The walk starts at a
    descriptor for the ROOT and opens one ref component at a time with
    O_NOFOLLOW, relative to the descriptor for the component before it. A
    ``..`` segment is already refused by _rootFor and a symlink at any
    component is an error rather than a redirection, so nothing the store
    controls can name a file outside the root -- not by racing, because there
    is no second lookup to race: every component is resolved once, against a
    descriptor the kernel is holding open, and the descriptor that comes out
    is the one that gets read.

    That is why there is no st_dev/st_ino comparison against a resolved path
    here, and no readlink of /proc/self/fd. Both would re-introduce a lookup
    by NAME to check the result of a lookup by name, and the identity they
    establish is the weaker one -- "this descriptor is what that name meant a
    moment ago" rather than "this descriptor is a descendant of the root,
    reached without following anything a ref chose". This is
    openat2(RESOLVE_BENEATH) assembled from stdlib parts; Linux-only, which
    this daemon already is.

    The ROOT itself is followed normally: home is configuration, not ref data,
    so a symlinked $HOME (or a symlinked import root) keeps working.

    NARROWING, deliberate: a symlink whose target is also inside the root used
    to be readable and is not any more, because telling it apart from one
    swapped a moment later is exactly the question that cannot be answered by
    name. Priced against the live store 2026-08-13: 0 of 279,818 resolvable
    live chunk source_refs traverse a symlink at any component.
    """
    split = _rootFor(ref, home)
    if split is None:
        return None
    top, rest = split
    # PurePath drops "" and "." when joining, so the walk drops them too and
    # the two entry points cannot disagree about which file a ref names.
    names = [name for name in rest.split("/") if name not in ("", ".")]
    if not names:
        return None
    try:
        walkFd = os.open(top, os.O_RDONLY | os.O_DIRECTORY)
    except (OSError, ValueError):
        return None
    fd = None
    try:
        for name in names[:-1]:
            nextFd = os.open(name, _WALK_FLAGS, dir_fd=walkFd)
            os.close(walkFd)
            walkFd = nextFd
        fd = os.open(names[-1], _LEAF_FLAGS, dir_fd=walkFd)
    except (OSError, ValueError):
        # ELOOP (a symlink met under O_NOFOLLOW), ENOTDIR, ENOENT, EACCES, or
        # a ref holding bytes no path can express. All of them mean the same
        # thing to every caller: no file to read.
        return None
    finally:
        os.close(walkFd)
    try:
        st = os.fstat(fd)
        # On the descriptor, so the answer is about the file that will be
        # read rather than about whatever the name meant when it was asked.
        # S_ISREG is what keeps a fifo, a device, or a directory from being
        # fed to a reader that expects to reach EOF.
        readable = stat.S_ISREG(st.st_mode) and (
            maxBytes is None or st.st_size <= maxBytes)
    except OSError:
        readable = False
    if not readable:
        os.close(fd)
        return None
    return os.fdopen(fd, "rb")
