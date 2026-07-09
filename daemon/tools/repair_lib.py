"""Pure helpers for the Phase A corpus repair: summary parsing and ref mapping.

The retired kv_cache store's ``meta.summary`` embeds the original absolute path
for file-derived rows (``[files] file /abs/path <action> ...``); everything else
in that table is reasoning text with no path. These helpers turn that summary
into the store's ref convention and derive project attribution from paths and
refs. Anything that does not parse returns None: the repair tool reports those
rows, it never guesses.

Ref convention (matches the existing import waves):
  /home/aegis/Projects/<name>/<rest>  ->  projects/<name>/<rest>, project <name>
  /home/aegis/.claude/<rest>          ->  claude-home/<rest>, no project
  /home/aegis/.codex/<rest>           ->  codex-home/<rest>, no project
Reference-library refs map to project Aegis (the library lives inside the Aegis
repo at AEGIS/docs/reference-library/).
"""
import re

__all__ = ["parseFilesSummary", "abspathToRef", "refToProject"]

# "[files] file <abspath> <rest>" -- the path is the token after "file ".
# Paths with embedded spaces do not occur in this corpus; a path token that
# does not start with "/" is rejected rather than guessed at.
_FILES_RE = re.compile(r"^\[files\] file (\S+)")

# (prefix, refRoot, projectSegment) -- projectSegment True means the first
# path segment under the prefix is the project name.
_ROOTS = (
    ("/home/aegis/Projects/", "projects/", True),
    ("/home/aegis/.claude/", "claude-home/", False),
    ("/home/aegis/.codex/", "codex-home/", False),
)


def parseFilesSummary(summary):
    """Absolute path out of a ``[files] file <path> ...`` summary, or None."""
    if not summary:
        return None
    m = _FILES_RE.match(summary)
    if m is None:
        return None
    path = m.group(1)
    if not path.startswith("/"):
        return None
    return path


def abspathToRef(abspath):
    """Map an absolute path to ``(ref, project)`` or None for unknown roots."""
    for prefix, refRoot, hasProject in _ROOTS:
        if abspath.startswith(prefix):
            rest = abspath[len(prefix):]
            project = rest.split("/", 1)[0] if hasProject and "/" in rest else (
                rest if hasProject else None)
            # A bare filename directly under Projects/ has no project dir.
            if hasProject and "/" not in rest:
                project = None
            return refRoot + rest, project
    return None


def refToProject(ref):
    """Project derivable from a ref, for null-project backfill; else None.

    ``projects/<name>/...`` yields the name; ``reference-library/...`` yields
    ``Aegis`` (the library lives inside the Aegis repo). Dotfile roots and the
    broken kv_cache refs yield None: dotfiles are not projects, and a kv_cache
    ref carries no path information.
    """
    if ref.startswith("projects/"):
        rest = ref[len("projects/"):]
        if "/" in rest:
            return rest.split("/", 1)[0]
        return None
    if ref.startswith("reference-library/"):
        return "Aegis"
    return None
