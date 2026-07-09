"""Serve-time enrichment for code-chunk recall results.

Two attachments, computed lazily for SURFACED hits only (top-k is small, so
this is a handful of file reads and one facet query per recall):

- **Location line** ``at <ref-path>#L<start>-<end>``: the chunk's stored text
  located in its source file by whitespace-normalized search. The chunker that
  produced the corpus is irrelevant because the text itself is the key.
- **Relation lines** ``relates -> p3://<atomId> <gist>``: memory-kind atoms
  sharing entity facets with the chunk, rarest-shared-entity first (specificity
  = 1/frequency, summed over shared entities). Same grammar as Tier-2 edge
  lines, so when the Phase C campaign materializes real relates edges the
  payload format does not change.

Failure policy: every miss (unknown ref root, missing/oversized/binary file,
text not found, no shared facets) yields NO line, never an exception. Recall
must answer even when the filesystem has moved on. Enrichment failures are
invisible by design; the dry corpus-health numbers live in the repair tool's
report, not in serve-time noise.
"""
import re

from recall.refs import refToPath
from store.store import getAtom

__all__ = ["locateChunk", "relatedMemory", "Enricher"]

# Files over this size are not searched (binary blobs, giant logs).
_MAX_FILE_BYTES = 5 * 1024 * 1024

# Collapse all whitespace runs to single spaces for matching; the stored chunk
# and the on-disk file may disagree about blank lines and indentation width.
_WS_RE = re.compile(r"\s+")

# Gist rendering matches payload._gist (80 chars, whitespace collapsed).
_GIST_CHARS = 80


def _norm(text):
    return _WS_RE.sub(" ", text).strip()


def _gist(text):
    flat = _norm(text)
    return flat[:_GIST_CHARS]


def locateChunk(chunkText, path):
    """1-indexed inclusive (start, end) line span of chunkText in path, or None.

    Match is on whitespace-normalized text. The span is found by scanning
    normalized prefixes: walk the file's lines accumulating a normalized
    window, and slide the window start forward when it can no longer prefix
    the target. O(lines * window) worst case, fine for source files.
    """
    try:
        if not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
            return None
        lines = path.read_text(errors="strict").splitlines()
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    target = _norm(chunkText)
    if not target:
        return None
    for start in range(len(lines)):
        if not _norm(lines[start]):
            # A blank line normalizes to "" and can never be the real start of
            # a match (target is already stripped, so it never begins with
            # whitespace). Without this guard the empty accumulator trivially
            # "prefixes" the target and the scan walks forward into the real
            # content, reporting the match as starting on the blank line
            # instead of where the text actually begins.
            continue
        acc = ""
        for end in range(start, len(lines)):
            piece = _norm(lines[end])
            acc = (acc + " " + piece).strip() if piece else acc
            if acc == target:
                return (start + 1, end + 1)
            if len(acc) > len(target) or (acc and not target.startswith(acc)):
                break
    return None


_MEMORY_KINDS = ("atom", "narrative", "snapshot")


def relatedMemory(store, atomId, limit=3):
    """Memory atoms sharing entity facets with atomId, rarest-first.

    Score per candidate = sum over shared entity values of 1/freq(value),
    where freq counts LIVE atoms carrying that entity facet, so rare shared
    entities dominate and hub entities (FILES-style) contribute almost
    nothing. Live memory kinds only; the source atom itself excluded.
    Returns [(memoryAtomId, gist)] best-first, at most limit.
    """
    kindMarks = ",".join("?" for _ in _MEMORY_KINDS)
    rows = store._conn.execute(
        "WITH mine AS ("
        "  SELECT value FROM facets WHERE atom_id = ? AND key = 'entity'"
        "), freq AS ("
        "  SELECT f.value, COUNT(*) AS n FROM facets f"
        "  JOIN atoms a ON a.id = f.atom_id"
        "  WHERE f.key = 'entity' AND a.status = 'live'"
        "  AND f.value IN (SELECT value FROM mine)"
        "  GROUP BY f.value"
        ") "
        "SELECT f.atom_id, SUM(1.0 / freq.n) AS score "
        "FROM facets f "
        "JOIN freq ON freq.value = f.value "
        "JOIN atoms a ON a.id = f.atom_id "
        f"WHERE f.key = 'entity' AND a.status = 'live' AND a.kind IN ({kindMarks}) "
        "AND f.atom_id != ? "
        "GROUP BY f.atom_id ORDER BY score DESC, f.atom_id LIMIT ?",
        (atomId, *_MEMORY_KINDS, atomId, limit),
    ).fetchall()
    out = []
    for memId, _score in rows:
        atom = getAtom(store, memId)
        if atom is None:
            continue
        out.append((memId, _gist(atom["text"])))
    return out


class Enricher:
    """Per-recall enrichment: furniture lines for one result dict.

    ``lines(result)`` returns [] for non-chunk atoms and on every failure
    path. ``home`` overrides the filesystem root for tests.
    """

    def __init__(self, store, home=None, relatedLimit=3):
        self._store = store
        self._home = home
        self._relatedLimit = relatedLimit

    def lines(self, result):
        atom = getAtom(self._store, result["atomId"])
        if atom is None or atom["kind"] != "document_chunk":
            return []
        out = []
        ref = self._sourceRef(result["atomId"])
        if ref:
            path = refToPath(ref, home=self._home)
            if path is not None:
                span = locateChunk(atom["text"], path)
                if span is not None:
                    base = ref.split("#", 1)[0]
                    out.append(f"at {base}#L{span[0]}-{span[1]}")
        for memId, gist in relatedMemory(
                self._store, result["atomId"], self._relatedLimit):
            out.append(f"relates -> p3://{memId} {gist}")
        return out

    def _sourceRef(self, atomId):
        row = self._store._conn.execute(
            "SELECT source_ref FROM provenance WHERE atom_id = ? "
            "AND source_ref IS NOT NULL ORDER BY recorded_at LIMIT 1",
            (atomId,),
        ).fetchone()
        return row[0] if row else None
