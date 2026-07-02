"""Backfill of an existing corpus into the v3 store.

Risk model (what a broken backfill would silently do, and the test that catches
each):

  1. Drop or duplicate records            -> count + per-kind assertions.
  2. Mislabel provenance                   -> every atom must read source
     'bulk-import' with sourceRef = the original id (the gate maps atomId back to
     the original doc id THROUGH this field; a wrong sourceRef silently zeroes
     every recall hit in the eval).
  3. Flatten kind                          -> atom vs narrative must survive.
  4. Lose or mangle src: tags              -> tag facets key='tag',
     value='src:<name>' verbatim.
  5. Diverge the entity-facet format       -> key='entity', value=lowercase
     surface form from MegaExtractor (the convention facetSignal reads; a case or
     prefix drift silently kills every entity boost). Pinned positively (the
     canonical form lands) and negatively (no PascalCase / type-prefixed variant).
  6. Double-write a repeated entity        -> idempotent facet (one row).
  7. Swallow a malformed record            -> a record with no text must raise,
     not silently vanish (loudness: a silent skip loses memory forever on a
     decades-scale store).
  8. Lie in the returned stats             -> the numbers the report cites must
     equal what is actually in the store.

The fixture is 200 SYNTHETIC atoms (committed; no PII). Anchors are found by
scanning the fixture with the SAME extractor the writer uses, so regenerating the
fixture cannot silently rot the test.
"""
import json
from pathlib import Path

import pytest

from ingest.backfill import backfill
from store.store import openStore, atomCount, getAtom, facetsOf

# The writer-side extractor construction, mirrored so the test's expectations are
# computed the same way backfill computes them (and the same way facetSignal reads
# them). If these drift, both this test and recall break together, loudly.
import sys

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
from pensive.mega_extract import MegaExtractor  # noqa: E402
from pensive.patterns import REAL_DATA_PATTERNS, build_pattern_set  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synthetic_atoms.jsonl"


def _load_fixture():
    with FIXTURE.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def records():
    recs = _load_fixture()
    assert len(recs) == 200, "fixture must be exactly 200 synthetic atoms"
    return recs


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "backfill.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def extractor():
    return MegaExtractor(build_pattern_set(REAL_DATA_PATTERNS))


def _atom_id_by_source_ref(store, sourceRef):
    row = store._conn.execute(
        "SELECT atom_id FROM provenance WHERE source_ref = ?", (sourceRef,)
    ).fetchone()
    return row[0] if row else None


# --------------------------------------------------------------------------- #
# 1 + 3: count and per-kind fidelity                                          #
# --------------------------------------------------------------------------- #

def test_backfill_ingests_every_record_once(store, records):
    stats = backfill(store, records)
    assert atomCount(store) == len(records)          # no drops, no dupes
    assert stats["ingested"] == len(records)


def test_backfill_preserves_kind_atom_vs_narrative(store, records):
    backfill(store, records)
    from collections import Counter
    expected = Counter(r["kind"] for r in records)
    rows = store._conn.execute(
        "SELECT kind, COUNT(*) FROM atoms GROUP BY kind"
    ).fetchall()
    got = {k: c for k, c in rows}
    assert got == dict(expected)
    # The fixture must actually contain both kinds or this test proves nothing.
    assert expected["atom"] > 0 and expected["narrative"] > 0


# --------------------------------------------------------------------------- #
# 2: provenance mapping (the gate's atomId -> original-doc bridge)            #
# --------------------------------------------------------------------------- #

def test_backfill_stamps_bulk_import_provenance_for_every_atom(store, records):
    backfill(store, records)
    rows = store._conn.execute("SELECT source FROM provenance").fetchall()
    assert len(rows) == len(records)
    assert all(source == "bulk-import" for (source,) in rows)


def test_backfill_maps_source_ref_to_original_id(store, records):
    backfill(store, records)
    refs = {
        r[0] for r in store._conn.execute(
            "SELECT source_ref FROM provenance"
        ).fetchall()
    }
    assert refs == {r["sourceId"] for r in records}
    # And a spot atom round-trips its own ref (the exact lookup the gate does).
    sample = records[7]
    atomId = _atom_id_by_source_ref(store, sample["sourceId"])
    assert atomId is not None
    atom = getAtom(store, atomId)
    assert atom["provenance"][0]["source"] == "bulk-import"
    assert atom["provenance"][0]["sourceRef"] == sample["sourceId"]
    assert atom["text"] == sample["text"]


def test_backfill_preserves_project_occurred_at_and_null_holes(store, records):
    backfill(store, records)
    # A record with a real project + occurredAt round-trips both.
    withBoth = next(
        r for r in records if r["project"] is not None and r["occurredAt"] is not None
    )
    a = getAtom(store, _atom_id_by_source_ref(store, withBoth["sourceId"]))
    assert a["project"] == withBoth["project"]
    assert a["occurredAt"] == withBoth["occurredAt"]
    # A project-less record stays NULL (not coerced to "" or a placeholder).
    projNull = next(r for r in records if r["project"] is None)
    a2 = getAtom(store, _atom_id_by_source_ref(store, projNull["sourceId"]))
    assert a2["project"] is None


# --------------------------------------------------------------------------- #
# 4: src: tags land as facets verbatim                                        #
# --------------------------------------------------------------------------- #

def test_backfill_lands_src_tags_as_tag_facets(store, records):
    backfill(store, records)
    tagged = next(r for r in records if r["tags"])
    a = getAtom(store, _atom_id_by_source_ref(store, tagged["sourceId"]))
    facets = facetsOf(store, a["id"])
    tagValues = {f["value"] for f in facets if f["key"] == "tag"}
    assert tagValues == set(tagged["tags"])
    # Every value keeps the src: prefix verbatim -- the read side matches on it.
    assert all(v.startswith("src:") for v in tagValues)


# --------------------------------------------------------------------------- #
# 5 + 6: entity facets in the pinned convention, deduped                      #
# --------------------------------------------------------------------------- #

def test_backfill_writes_entity_facets_in_canonical_convention(store, records, extractor):
    backfill(store, records)
    # Find a record whose text carries the 'pensive' project entity.
    withPensive = next(
        r for r in records
        if "pensive" in {lab for lab, _t in extractor.extract(r["text"])}
    )
    a = getAtom(store, _atom_id_by_source_ref(store, withPensive["sourceId"]))
    facets = facetsOf(store, a["id"])
    entityValues = {f["value"] for f in facets if f["key"] == "entity"}
    assert "pensive" in entityValues                    # canonical lowercase form
    # Negative pin: NO divergent-format variant is ever written.
    assert "PENSIVE" not in entityValues
    assert "project:pensive" not in entityValues
    # The value really is the lowercase surface form extract() returns.
    for lab, _etype in extractor.extract(withPensive["text"]):
        assert lab == lab.lower()
        assert lab in entityValues


def test_backfill_entity_facet_is_idempotent_for_repeated_entity(store, records, extractor):
    # A body that names the same entity twice must yield exactly ONE entity facet
    # for it (addFacet is INSERT OR IGNORE on (atom_id, key, value)).
    def _repeats(text):
        labels = [lab for lab, _t in extractor.extract(text)]
        for lab in set(labels):
            if labels.count(lab) >= 2:
                return lab
        return None

    target = None
    for r in records:
        lab = _repeats(r["text"])
        if lab is not None:
            target = (r, lab)
            break
    assert target is not None, "fixture must contain a repeated-entity atom"
    r, lab = target

    backfill(store, records)
    a = getAtom(store, _atom_id_by_source_ref(store, r["sourceId"]))
    facets = facetsOf(store, a["id"])
    matching = [f for f in facets if f["key"] == "entity" and f["value"] == lab]
    assert len(matching) == 1


# --------------------------------------------------------------------------- #
# 7: loudness -- a malformed record raises, it is not silently dropped        #
# --------------------------------------------------------------------------- #

def test_backfill_raises_on_record_missing_text(store):
    # Loudness over silent-skip: on a decades-scale memory store a swallowed
    # record is memory lost forever. A record with no text has nothing to store
    # and must surface, not vanish. The good record BEFORE it is already
    # committed (per-record transaction); backfill aborts loudly on the bad one.
    good = {"sourceId": "ok-1", "text": "a well formed atom about aegis",
            "kind": "atom", "tags": ["src:ok"]}
    bad = {"sourceId": "bad-1", "kind": "atom", "tags": ["src:bad"]}  # no text
    with pytest.raises(KeyError):
        backfill(store, [good, bad])
    # The good record persisted (partial progress is preserved and honest);
    # the bad one wrote nothing.
    assert atomCount(store) == 1
    assert _atom_id_by_source_ref(store, "bad-1") is None


def test_backfill_carries_session_agent_and_created_at_for_historical_imports(store):
    # The real corpus export carries the original session/agent and the ingest
    # time; a bulk import of decade-old memory must not read as recorded today, so
    # createdAt overrides putAtom's now(). occurredAt stays the event time.
    recs = [{
        "sourceId": "kv_cache/vector_meta.db#rowid=42",
        "text": "a historical reasoning atom about aegis",
        "kind": "atom",
        "project": "aegis-infra",
        "occurredAt": 1_600_000_000,   # when the thing happened
        "createdAt": 1_600_500_000,    # when it was recorded (the real ingest time)
        "sessionId": "sess-historical",
        "agent": "claude",
    }]
    backfill(store, recs)
    a = getAtom(store, _atom_id_by_source_ref(store, "kv_cache/vector_meta.db#rowid=42"))
    assert a["createdAt"] == 1_600_500_000       # override applied, not now()
    assert a["occurredAt"] == 1_600_000_000       # event time preserved distinctly
    prov = a["provenance"][0]
    assert prov["source"] == "bulk-import"
    assert prov["sessionId"] == "sess-historical"
    assert prov["agent"] == "claude"


def test_backfill_tolerates_empty_tags_and_absent_optionals(store):
    # tags=[] and no project/occurredAt/importance: ingest succeeds, no tag
    # facets, defaults applied -- an atom with nothing but text is still memory.
    recs = [{"sourceId": "min-1", "text": "bare atom naming kairos once"}]
    stats = backfill(store, recs)
    assert stats["ingested"] == 1
    a = getAtom(store, _atom_id_by_source_ref(store, "min-1"))
    assert a["kind"] == "atom"          # kind defaults to 'atom'
    assert a["project"] is None
    assert a["importance"] == 0.0
    tagFacets = [f for f in facetsOf(store, a["id"]) if f["key"] == "tag"]
    assert tagFacets == []
    # entity enrichment still ran (kairos is a known project entity).
    entityFacets = {f["value"] for f in facetsOf(store, a["id"]) if f["key"] == "entity"}
    assert "kairos" in entityFacets


# --------------------------------------------------------------------------- #
# 8: the stats the report cites are true                                      #
# --------------------------------------------------------------------------- #

def test_backfill_is_idempotent_across_runs(store, records):
    # Re-run safety (the production migration path): running the SAME export into
    # the SAME store twice must NOT double the corpus. The guard is on the
    # bulk-import source_ref (the original id); the second run skips every record
    # whose ref is already present and reports it, leaving atom and facet counts
    # exactly where run 1 left them.
    stats1 = backfill(store, records)
    count1 = atomCount(store)
    entity1 = store._conn.execute(
        "SELECT COUNT(*) FROM facets WHERE key = 'entity'"
    ).fetchone()[0]
    tag1 = store._conn.execute(
        "SELECT COUNT(*) FROM facets WHERE key = 'tag'"
    ).fetchone()[0]
    assert stats1["ingested"] == len(records)
    assert stats1["skipped"] == 0

    stats2 = backfill(store, records)          # identical export, identical store
    assert atomCount(store) == count1          # no doubling
    assert stats2["ingested"] == 0
    assert stats2["skipped"] == len(records)
    assert store._conn.execute(
        "SELECT COUNT(*) FROM facets WHERE key = 'entity'"
    ).fetchone()[0] == entity1                 # facets unchanged
    assert store._conn.execute(
        "SELECT COUNT(*) FROM facets WHERE key = 'tag'"
    ).fetchone()[0] == tag1


def test_backfill_passes_through_document_chunk_kind(store):
    # document_chunk is a real kind (the chat-export backfill uses it exclusively)
    # but the 200-atom fixture never exercises it. Pin the passthrough directly.
    recs = [
        {"sourceId": "doc-1", "text": "a chunk of a longer document about aegis",
         "kind": "document_chunk"},
        {"sourceId": "doc-2", "text": "another chunk naming pensive once",
         "kind": "document_chunk"},
    ]
    stats = backfill(store, recs)
    assert stats["byKind"].get("document_chunk") == 2
    a = getAtom(store, _atom_id_by_source_ref(store, "doc-1"))
    assert a["kind"] == "document_chunk"


def test_backfill_stats_match_the_store(store, records):
    stats = backfill(store, records)
    entityRows = store._conn.execute(
        "SELECT COUNT(*) FROM facets WHERE key = 'entity'"
    ).fetchone()[0]
    tagRows = store._conn.execute(
        "SELECT COUNT(*) FROM facets WHERE key = 'tag'"
    ).fetchone()[0]
    assert stats["entityFacets"] == entityRows
    assert stats["tagFacets"] == tagRows
    assert stats["ingested"] == atomCount(store)
    # byKind sums to the total.
    assert sum(stats["byKind"].values()) == stats["ingested"]
