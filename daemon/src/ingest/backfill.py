"""Backfill an existing corpus export into the canonical v3 store.

This is the one-time bridge from "the memory Gary already has" to "the memory v3
serves". It consumes a read-only export of the live corpus and writes, through
the canonical store API only, one atom per source record plus its provenance and
facets. It does NOT embed: embedding is a separate ``embedMissing`` pass the
caller runs once the whole corpus is in (the fixed ``backfill(store, srcExport)``
signature carries no embedder, and keeping embedding out means the ingest unit
stays GPU-free and independently testable). The end-to-end backfill is therefore
two calls::

    backfill(store, export)           # atoms + provenance + facets  (this module)
    embedMissing(store, embedder)     # vectors                       (recall.embedder)

Export record shape (every field but ``text`` optional):

    {
      "sourceId":   <original id or path>,   # -> provenance.source_ref
      "text":       <atom body>,             # REQUIRED
      "kind":       "atom"|"narrative"|...,  # -> atom.kind (default "atom")
      "project":    <slug> | None,           # -> atom.project
      "occurredAt": <unix seconds> | None,   # -> atom.occurred_at (when the
                                             #    remembered thing happened)
      "createdAt":  <unix seconds> | None,   # -> atom.created_at OVERRIDE (the
                                             #    original record time; see below)
      "importance": <float>,                 # -> atom.importance (default 0.0)
      "tags":       ["src:<name>", ...],     # -> facets key='tag', value verbatim
      "sessionId":  <str> | None,            # -> provenance.session_id
      "agent":      <str> | None,            # -> provenance.agent
    }

Provenance: every backfilled atom is stamped ``source='bulk-import'`` with
``source_ref`` = the original id, plus the original ``session_id`` and ``agent``
when the export carries them. The eval gate (and later the MCP layer) map a
recalled atomId back to its original document id THROUGH ``source_ref``, so it is
load-bearing, not decorative.

created_at override: ``putAtom`` stamps ``created_at`` to now, which is right for
a live emit but wrong for a bulk import of historical memory -- a decade-old atom
must not read as recorded today. When a record carries ``createdAt`` this module
corrects the row to the real record time after the insert (occurred_at stays the
event time; created_at becomes the ingest time), so the store's effective time
COALESCE(occurred_at, created_at) reflects the atom's true age. Absent
``createdAt`` the putAtom default (now) stands.

src: tags -> facets ``(key='tag', value='src:<name>')`` verbatim. Entity
enrichment -> facets ``(key='entity', value=<lowercase surface form>)`` in the
convention pinned at Task 6: the write side here and the read side in
``recall.signals.facetSignal`` MUST construct the extractor identically or entity
boosts silently die. Both use ``MegaExtractor(build_pattern_set(REAL_DATA_PATTERNS))``
over the LOCAL ``src/pensive`` tree (never a pip-installed pypensive). A format
change touches this module and ``facetSignal`` together; the negative pins in
``test_backfill`` and ``test_signals`` turn any drift into a loud failure.

Loudness: a record with no ``text`` has nothing to remember and RAISES rather than
being silently skipped -- on a decades-scale store a swallowed record is memory
lost forever. Each record is its own transaction (putAtom + its facets), so an
abort preserves the records already committed (honest partial progress) instead of
rolling back a multi-hour backfill.
"""
import sys
from pathlib import Path

from store.store import putAtom, addFacet

__all__ = ["backfill"]

# The v2 entity extractor lives in THIS repo's local tree at <repo>/src/pensive,
# not in the daemon package. parents: ingest -> src -> daemon -> <repo>. Mirror
# recall.signals exactly so writer and reader share one extractor construction.
_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from pensive.mega_extract import MegaExtractor  # noqa: E402  (path set above)
from pensive.patterns import REAL_DATA_PATTERNS, build_pattern_set  # noqa: E402

# Fixed provenance source for anything brought in by the bulk migration. The
# schema comments enumerate this exact value.
_BULK_SOURCE = "bulk-import"

# Lazily-built, process-wide extractor (pattern compilation is not free). Same
# construction as recall.signals._getExtractor -- keep them identical.
_extractor = None


def _getExtractor():
    global _extractor
    if _extractor is None:
        _extractor = MegaExtractor(build_pattern_set(REAL_DATA_PATTERNS))
    return _extractor


def backfill(store, srcExport):
    """Ingest ``srcExport`` records into ``store``; return truthful stats.

    ``srcExport`` is any iterable of export records (see the module docstring).
    For each record this writes one atom with ``bulk-import`` provenance, its
    ``src:`` tags as ``tag`` facets, and its extracted entities as ``entity``
    facets in the pinned convention. Facet writes are idempotent, so an entity
    named twice in one body yields exactly one facet.

    Returns ``{ingested, byKind, tagFacets, entityFacets}`` where the counts equal
    what actually landed in the store (facets are counted DEDUPED per atom, so the
    totals match ``SELECT COUNT(*) FROM facets`` for each key -- the report cites
    these numbers, so they must not lie).

    Raises ``KeyError`` on a record with no ``text`` (loudness over silent skip);
    records committed before the offending one are preserved.
    """
    extractor = _getExtractor()
    ingested = 0
    byKind = {}
    tagFacets = 0
    entityFacets = 0

    for record in srcExport:
        # Required. A missing text is a loud KeyError, not a silent drop.
        text = record["text"]
        kind = record.get("kind") or "atom"

        provenance = {"source": _BULK_SOURCE, "sourceRef": record.get("sourceId")}
        # Carry the original session/agent when the export has them (the real
        # corpus does; the synthetic fixture does not). putAtom defaults them NULL.
        if record.get("sessionId") is not None:
            provenance["sessionId"] = record["sessionId"]
        if record.get("agent") is not None:
            provenance["agent"] = record["agent"]

        atomInput = {
            "text": text,
            "kind": kind,
            "project": record.get("project"),
            "occurredAt": record.get("occurredAt"),
            "importance": record.get("importance", 0.0),
            "provenance": provenance,
        }
        atomId = putAtom(store, atomInput)

        # created_at override for historical imports (see module docstring). One
        # UPDATE, committed, only when the record supplies the real record time.
        createdAt = record.get("createdAt")
        if createdAt is not None:
            store._conn.execute(
                "UPDATE atoms SET created_at = ? WHERE id = ?", (createdAt, atomId)
            )
            store._conn.commit()

        ingested += 1
        byKind[kind] = byKind.get(kind, 0) + 1

        # src: tags -> tag facets, verbatim. Dedup within the record so the count
        # matches the store (the facet PK dedups too, this keeps the stat honest).
        for tag in dict.fromkeys(record.get("tags") or []):
            addFacet(store, atomId, "tag", tag)
            tagFacets += 1

        # Entity enrichment -> entity facets in the pinned convention. extract()
        # returns the lowercase surface form already; dedup labels per atom so a
        # repeated entity is one facet and the count matches the store.
        labels = {label for label, _etype in extractor.extract(text)}
        for label in labels:
            addFacet(store, atomId, "entity", label)
            entityFacets += 1

    return {
        "ingested": ingested,
        "byKind": byKind,
        "tagFacets": tagFacets,
        "entityFacets": entityFacets,
    }
