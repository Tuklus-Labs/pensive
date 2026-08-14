"""Serve-time enrichment for code-chunk recall results.

Two attachments, computed lazily for SURFACED hits only (top-k is small, so
this is a handful of file reads and one facet query per recall):

- **Location line** ``at <ref-path>#L<start>-<end>``: the chunk's stored text
  located in its source file by whitespace-normalized search. The chunker that
  produced the corpus is irrelevant because the text itself is the key.
- **Relation lines** ``relates -> p3://<atomId> <gist>``: when the chunk has
  LIVE outgoing ``relates`` edges to live memory atoms, those render (weight
  desc, capped at ``relatedLimit``). Chunks the Phase C campaign has not
  covered yet have no such edges, so they fall back to memory-kind atoms
  sharing entity facets with the chunk, rarest-shared-entity first
  (specificity = 1/frequency, summed over shared entities). Same grammar
  either way, so the payload format does not change across the campaign.

Failure policy: every miss (unknown ref root, missing/oversized/binary file,
text not found, no shared facets) yields NO line, never an exception. Recall
must answer even when the filesystem has moved on. Enrichment failures are
invisible by design; the dry corpus-health numbers live in the repair tool's
report, not in serve-time noise.
"""
import os
import re

from recall.payload import GIST_CHARS as _GIST_CHARS, gistOf as _gistOf
from recall.refs import openRef
from store.store import getAtom

__all__ = ["locateChunk", "relatedMemory", "Enricher"]

# Files over this size are not searched (binary blobs, giant logs). Spent on
# the BYTES READ, not on a size measured beforehand: a stat describes a name
# at one instant, and a file can grow between the stat and the read. Measured
# on the pre-fix code, a stat reporting 6 bytes was followed by a read that
# returned 6 MiB, because a writer got in between the two lookups.
#
# Also passed to openRef, which refuses an oversized file from its fstat
# before a byte is read. Two locks that fail differently, the same shape as
# mcp._rejectEscapingSourceRef sitting on top of refs.py: the fstat is an
# early out on one number, the bounded read is what actually holds.
_MAX_FILE_BYTES = 5 * 1024 * 1024

# An entity value carried by more than this many live atoms is a HUB and is
# excluded from the related-memory query entirely.
#
# relatedMemory scores by sum(1/freq), so a hub already contributes almost
# nothing to the ranking. It contributed nearly all of the COST, because it was
# filtered after the join rather than before it: the value was joined, grouped
# and summed across every atom carrying it, then down-weighted to irrelevance.
# Measured on the live store 2026-08-13, per chunk:
#
#     getAtom (+ all provenance)      0.01 ms
#     _sourceRef                      0.01 ms
#     _edgeRelations                  0.08 ms
#     relatedMemory                1182.23 ms   <- the whole enrichment cost
#
# Only 2,810 `relates` edges exist across ~283k live chunks, so nearly every
# chunk takes this fallback, and L3-with-enrichment measured 2,323 ms p50
# against a 125 ms budget.
#
# 100 against ~675k entity facets over ~212k distinct values (mean 3.2 per
# value): a value at the ceiling contributes 0.01 to a score where a singleton
# contributes 1.0, so nothing that could plausibly decide a ranking is dropped.
# Identical in shape to the lexical DF prune in recall/signals.py, and worth
# recognizing on sight: a term whose score weight is near zero dominating the
# work because the filter sits downstream of the join.
HUB_ENTITY_MAX_FREQ = 100

# Collapse all whitespace runs to single spaces for matching; the stored chunk
# and the on-disk file may disagree about blank lines and indentation width.
_WS_RE = re.compile(r"\s+")

# Gist rendering shares payload's GIST_CHARS (imported above) so the two can
# never diverge if the gist width is ever tuned.


def _norm(text):
    return _WS_RE.sub(" ", text).strip()


def _gist(text):
    """Shares payload's gist SELECTION, not merely its width.

    A shared GIST_CHARS was never enough: what a reader sees is which line was
    picked, so a divergence here would render the same atom two different ways
    depending on which surface returned it."""
    return _gistOf(text)


def locateChunk(chunkText, fh):
    """1-indexed inclusive (start, end) line span of chunkText in fh, or None.

    ``fh`` is an ALREADY OPEN binary handle, normally from ``refs.openRef``.
    Taking a handle rather than a path is the fix, not a convenience: a reader
    handed a name goes back to the filesystem for a lookup of its own, and
    whatever that lookup finds is a different question from the one the ref
    was validated against. There is nothing to re-open here, so there is no
    window to race.

    Match is on whitespace-normalized text. The span is found by scanning
    normalized prefixes: walk the file's lines accumulating a normalized
    window, and slide the window start forward when it can no longer prefix
    the target. O(lines * window) worst case, fine for source files.
    """
    if isinstance(fh, (str, bytes, os.PathLike)):
        # Loud, because a quiet None would let a later refactor go back to
        # reading by name and still see a green suite.
        raise TypeError(
            "locateChunk reads an open binary handle, not a path; open the "
            "ref with recall.refs.openRef and pass the handle"
        )
    try:
        raw = fh.read(_MAX_FILE_BYTES + 1)
        if len(raw) > _MAX_FILE_BYTES:
            return None
        # Decoding moves here with the handle. read_text() used the locale's
        # preferred encoding; utf-8 is what the corpus is, and pinning it
        # keeps a security-relevant read from depending on the daemon's
        # environment. A binary blob still raises and still yields no line.
        lines = raw.decode("utf-8", errors="strict").splitlines()
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
        # The ceiling belongs HERE, upstream of the outer join, so a hub value
        # never reaches it. Putting it in the score instead is what made this
        # query cost a second per chunk.
        "  HAVING n <= ?"
        ") "
        "SELECT f.atom_id, SUM(1.0 / freq.n) AS score "
        "FROM facets f "
        "JOIN freq ON freq.value = f.value "
        "JOIN atoms a ON a.id = f.atom_id "
        # The value predicate has to be repeated HERE, not left to the join.
        # Without it SQLite drives the outer query from `f` with
        # `idx_facets_kv (key=?)` alone -- scanning every entity facet in the
        # store (~676k), joining each to atoms, and probing the freq co-routine
        # per row. EXPLAIN QUERY PLAN says so in one line; five rounds of
        # reasoning about it did not. Restricting on `freq` rather than `mine`
        # applies the hub ceiling to the scan as well as to the score.
        #
        #   1182 ms  original
        #     46 ms  + value predicate (index becomes key AND value)
        #     24 ms  + restricted to the ceiling-filtered values
        f"WHERE f.key = 'entity' AND f.value IN (SELECT value FROM freq) "
        f"AND a.status = 'live' AND a.kind IN ({kindMarks}) "
        "AND f.atom_id != ? "
        "GROUP BY f.atom_id ORDER BY score DESC, f.atom_id LIMIT ?",
        (atomId, HUB_ENTITY_MAX_FREQ, *_MEMORY_KINDS, atomId, limit),
    ).fetchall()
    out = []
    for memId, _score in rows:
        atom = getAtom(store, memId)
        if atom is None:
            continue
        out.append((memId, _gist(atom["text"])))
    return out


def _edgeRelations(store, chunkId, limit):
    """LIVE relates edges from a chunk -> [(memId, gist)] by weight desc.

    Only edges whose destination is a live atom render (parity with the
    Tier-2 rule: never dangle a pointer at a non-recallable atom). Returns
    [] when the chunk has no live relates edges, which is the signal to fall
    back to the facet join.
    """
    rows = store._conn.execute(
        "SELECT e.dst_atom FROM edges e "
        "JOIN atoms a ON a.id = e.dst_atom "
        "WHERE e.src_atom = ? AND e.type = 'relates' "
        "AND a.status = 'live' "
        "ORDER BY e.weight DESC, e.dst_atom LIMIT ?",
        (chunkId, limit),
    ).fetchall()
    out = []
    for (memId,) in rows:
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
            # A sourceRef is store data, so this string selects a file the
            # daemon opens during ordinary recall. openRef returns a
            # descriptor or nothing; the name is never handed onward.
            fh = openRef(ref, home=self._home, maxBytes=_MAX_FILE_BYTES)
            if fh is not None:
                with fh:
                    span = locateChunk(atom["text"], fh)
                if span is not None:
                    base = ref.split("#", 1)[0]
                    out.append(f"at {base}#L{span[0]}-{span[1]}")
        relations = _edgeRelations(
            self._store, result["atomId"], self._relatedLimit)
        if not relations:
            relations = relatedMemory(
                self._store, result["atomId"], self._relatedLimit)
        for memId, gist in relations:
            out.append(f"relates -> p3://{memId} {gist}")
        return out

    def _sourceRef(self, atomId):
        row = self._store._conn.execute(
            "SELECT source_ref FROM provenance WHERE atom_id = ? "
            "AND source_ref IS NOT NULL ORDER BY recorded_at LIMIT 1",
            (atomId,),
        ).fetchone()
        return row[0] if row else None
