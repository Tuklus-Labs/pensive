"""Backfill an existing corpus export into the canonical v3 store.

This is the one-time bridge from "the memory Gary already has" to "the memory v3
serves". It consumes a read-only export of the live corpus and writes to the
canonical store schema: one atom per source record plus its provenance and
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
corrects the row to the real record time inside the record transaction
(occurred_at stays the event time; created_at becomes the original record time),
so the store's effective time COALESCE(occurred_at, created_at) reflects the
atom's true age. Absent
``createdAt`` the putAtom default (now) stands.

src: tags -> facets ``(key='tag', value='src:<name>')`` verbatim. Entity
enrichment -> facets ``(key='entity', value=<lowercase surface form>)`` in the
convention pinned at Task 6: the write side here and the read side in
``recall.signals.facetSignal`` MUST construct the extractor identically or entity
boosts silently die. Both use ``MegaExtractor(build_pattern_set(REAL_DATA_PATTERNS))``
over the LOCAL ``src/pensive`` tree (never a pip-installed pypensive). A format
change touches this module and ``facetSignal`` together; the negative pins in
``test_backfill`` and ``test_signals`` turn any drift into a loud failure.

Re-run idempotency (the migration path runs against a real store): a record whose
``sourceId`` already exists as a ``bulk-import`` ``source_ref`` in the store is
SKIPPED, not re-ingested -- so running the same export twice does not silently
double the corpus. The guard also dedups WITHIN a run (a repeated sourceId in one
export lands once). Skips are counted and returned as ``skipped``; a record with no
``sourceId`` (``source_ref`` NULL) has no identity to dedup on and is always
ingested.

Loudness: a record with no ``text`` has nothing to remember and RAISES rather than
being silently skipped -- on a decades-scale store a swallowed record is memory
lost forever. Each record is its own transaction (atom + provenance + created_at
override + facets), so an abort preserves the records already committed (honest
partial progress) while rolling back the in-flight record completely.
"""
import sys
import time
from pathlib import Path

from store.migrate import CURRENT_SCHEMA_VERSION
from util.ulid import ulid

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


def _insertAtomNoCommit(conn, atomInput):
    now = int(time.time())
    atomId = ulid()
    prov = atomInput["provenance"]
    conn.execute(
        "INSERT INTO atoms(id, text, kind, project, created_at, occurred_at, "
        "importance, status, schema_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            atomId,
            atomInput["text"],
            atomInput["kind"],
            atomInput.get("project"),
            now,
            atomInput.get("occurredAt"),
            atomInput.get("importance", 0.0),
            "live",
            CURRENT_SCHEMA_VERSION,
        ),
    )
    conn.execute(
        "INSERT INTO provenance(id, atom_id, source, session_id, agent, "
        "source_ref, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            ulid(),
            atomId,
            prov["source"],
            prov.get("sessionId"),
            prov.get("agent"),
            prov.get("sourceRef"),
            now,
        ),
    )
    return atomId


def _addFacetNoCommit(conn, atomId, key, value):
    conn.execute(
        "INSERT OR IGNORE INTO facets(atom_id, key, value) VALUES (?, ?, ?)",
        (atomId, key, value),
    )


def backfill(store, srcExport):
    """Ingest ``srcExport`` records into ``store``; return truthful stats.

    ``srcExport`` is any iterable of export records (see the module docstring).
    For each record this writes one atom with ``bulk-import`` provenance, its
    ``src:`` tags as ``tag`` facets, and its extracted entities as ``entity``
    facets in the pinned convention. Facet writes are idempotent, so an entity
    named twice in one body yields exactly one facet.

    Returns ``{ingested, skipped, byKind, tagFacets, entityFacets}`` where the
    counts equal what actually landed in the store (facets are counted DEDUPED per
    atom, so the totals match ``SELECT COUNT(*) FROM facets`` for each key -- the
    report cites these numbers, so they must not lie). ``skipped`` counts records
    whose ``sourceId`` was already present as a bulk-import ``source_ref`` (re-run
    idempotency); those write nothing.

    Raises ``KeyError`` on a record with no ``text`` (loudness over silent skip);
    records committed before the offending one are preserved.
    """
    extractor = _getExtractor()
    ingested = 0
    skipped = 0
    byKind = {}
    tagFacets = 0
    entityFacets = 0

    # Re-run guard: existing bulk-import refs already in the store. Seeding the
    # set from the store (one SELECT) makes a second run against the same store a
    # no-op; adding to it as we go dedups repeats within THIS run too. NULL refs
    # are excluded -- they carry no identity to dedup on.
    seenRefs = {
        row[0]
        for row in store._conn.execute(
            "SELECT source_ref FROM provenance "
            "WHERE source = ? AND source_ref IS NOT NULL",
            (_BULK_SOURCE,),
        ).fetchall()
    }

    for record in srcExport:
        # Required. A missing text is a loud KeyError, not a silent drop.
        text = record["text"]
        kind = record.get("kind") or "atom"

        sourceRef = record.get("sourceId")
        # Already backfilled (prior run or earlier in this run): skip, do not
        # double. A NULL sourceRef has no identity, so it is never deduped.
        if sourceRef is not None and sourceRef in seenRefs:
            skipped += 1
            continue

        provenance = {"source": _BULK_SOURCE, "sourceRef": sourceRef}
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
        createdAt = record.get("createdAt")

        # src: tags -> tag facets, verbatim. Dedup within the record so the count
        # matches the store (the facet PK dedups too, this keeps the stat honest).
        tags = list(dict.fromkeys(record.get("tags") or []))

        # Entity enrichment -> entity facets in the pinned convention. extract()
        # returns the lowercase surface form already; dedup labels per atom so a
        # repeated entity is one facet and the count matches the store.
        labels = {label for label, _etype in extractor.extract(text)}
        conn = store._conn
        try:
            atomId = _insertAtomNoCommit(conn, atomInput)
            # created_at override for historical imports (see module docstring).
            # Fold it into the record transaction so a crash cannot leave a
            # wrong-created_at atom that the rerun guard skips forever.
            if createdAt is not None:
                conn.execute(
                    "UPDATE atoms SET created_at = ? WHERE id = ?", (createdAt, atomId)
                )
            for tag in tags:
                _addFacetNoCommit(conn, atomId, "tag", tag)
            for label in labels:
                _addFacetNoCommit(conn, atomId, "entity", label)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        ingested += 1
        byKind[kind] = byKind.get(kind, 0) + 1
        tagFacets += len(tags)
        entityFacets += len(labels)
        if sourceRef is not None:
            seenRefs.add(sourceRef)  # dedup a repeated ref later in this same run

    return {
        "ingested": ingested,
        "skipped": skipped,
        "byKind": byKind,
        "tagFacets": tagFacets,
        "entityFacets": entityFacets,
    }
