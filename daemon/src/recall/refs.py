"""Ref-to-filesystem-path resolution: the serve-time inverse of the import
waves' ref convention.

A source_ref names a file relative to one of the known import roots plus an
optional ``#<fragment>`` (chunk index). This module maps a ref back to the
absolute path; existence and readability stay the caller's concern (enrich
handles missing files as "no attachment"). Symlinks are the one thing it does
read, because containment cannot be judged without following them.

Unknown roots (including the kv_cache refs Phase A could not recover), refs
containing a ``..`` segment, refs whose rest is absolute, and refs that land
outside their root once resolved all return None: a ref is data from the
store, not a trusted path, and must never escape its root.
"""
from pathlib import Path

__all__ = ["refToPath"]

# refRoot -> path relative to $HOME. Ordered; first prefix match wins.
_ROOTS = (
    ("projects/", "Projects"),
    ("reference-library/", "Projects/Aegis/AEGIS/docs/reference-library"),
    ("claude-home/", ".claude"),
    ("codex-home/", ".codex"),
)


def _contains(top, path):
    """True if path is still inside top once both sides are fully resolved.

    Resolving is what catches the escapes the joined string hides: a file
    inside the root that is a symlink pointing out of it reads from outside
    the root while its ref looks ordinary. Both sides are resolved so a root
    reached through a symlinked $HOME still compares equal to itself.

    resolve() is non-strict, so a ref naming a file that does not exist is
    still contained (missing files remain the caller's problem). It does
    raise on a ref no path can express, such as an embedded NUL: that is
    reported as not-contained rather than allowed to propagate, because
    enrich.lines() calls refToPath outside its try and would otherwise lose
    the whole enrichment instead of one attachment.
    """
    try:
        return path.resolve().is_relative_to(top.resolve())
    except (OSError, ValueError):
        return False


def refToPath(ref, home=None):
    """Absolute Path for a store ref, or None for unknown/unsafe refs."""
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
            top = root / rel
            path = top / rest
            if not _contains(top, path):
                return None
            return path
    return None
