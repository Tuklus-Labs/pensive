"""Ref-to-filesystem-path resolution: the serve-time inverse of the import
waves' ref convention.

A source_ref names a file relative to one of the known import roots plus an
optional ``#<fragment>`` (chunk index). This module maps a ref back to the
absolute path WITHOUT touching the filesystem; existence and readability are
the caller's concern (enrich handles missing files as "no attachment").

Unknown roots (including the kv_cache refs Phase A could not recover) and any
ref containing a ``..`` segment resolve to None: a ref is data from the store,
not a trusted path, and must never escape its root.
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
            if not rest:
                return None
            return root / rel / rest
    return None
