"""Serve-time enrichment: chunk location + related-memory attachment."""
import pytest

from recall.enrich import locateChunk, relatedMemory, Enricher
from store.store import openStore, putAtom, addFacet


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text, kind="atom", project="aegis", sourceRef=None):
    prov = {"source": "claude-code"}
    if sourceRef is not None:
        prov["sourceRef"] = sourceRef
    return putAtom(store, {
        "text": text, "kind": kind, "project": project,
        "importance": 0.0, "provenance": prov,
    })


# ---- locateChunk ----------------------------------------------------------

def test_locate_exact_span(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("line one\ndef target():\n    return 42\nline four\n")
    assert locateChunk("def target():\n    return 42", f) == (2, 3)


def test_locate_whitespace_normalized(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("a\n\ndef target():\n\n    return   42\nz\n")
    # The stored chunk collapsed blank lines and run-length spaces; the span
    # still resolves to the file's own line numbers.
    assert locateChunk("def target():\n    return 42", f) == (3, 5)


def test_locate_miss_missing_and_oversize(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("nothing relevant\n")
    assert locateChunk("absent text", f) is None
    assert locateChunk("anything", tmp_path / "gone.py") is None
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    assert locateChunk("anything", big) is None


# ---- relatedMemory --------------------------------------------------------

def test_related_ranks_by_shared_entity_specificity(store):
    chunk = _put(store, "def compile(): pass", kind="document_chunk",
                 sourceRef="projects/pensive/src/x.py#c0")
    rare = _put(store, "decision about the csr publish lock")
    common = _put(store, "note mentioning python again")
    other = _put(store, "unrelated memory")
    codeTwin = _put(store, "another chunk", kind="document_chunk")
    addFacet(store, chunk, "entity", "csr_lock")
    addFacet(store, chunk, "entity", "python")
    addFacet(store, rare, "entity", "csr_lock")          # csr_lock: 2 atoms (rarer)
    addFacet(store, common, "entity", "python")
    addFacet(store, codeTwin, "entity", "python")        # python: 3 atoms
    # codeTwin shares only "python" with chunk (not "csr_lock"): it is a
    # document_chunk, so it must be excluded from results by kind even
    # though it shares an entity, without also equalizing csr_lock's and
    # python's frequencies (which would make the specificity ranking a tie).
    got = relatedMemory(store, chunk, limit=3)
    ids = [g[0] for g in got]
    assert ids[0] == rare                 # rarer shared entity outranks common
    assert common in ids
    assert other not in ids               # no shared entity
    assert codeTwin not in ids            # memory kinds only


def test_related_empty_when_no_facets(store):
    chunk = _put(store, "body", kind="document_chunk")
    assert relatedMemory(store, chunk, limit=3) == []


# ---- Enricher -------------------------------------------------------------

def test_enricher_lines_for_chunk(store, tmp_path):
    f = tmp_path / "Projects/obol/api/rate.go"
    f.parent.mkdir(parents=True)
    f.write_text("package api\nfunc Rate() int { return 1 }\n")
    chunk = _put(store, "func Rate() int { return 1 }",
                 kind="document_chunk",
                 sourceRef="projects/obol/api/rate.go#c0")
    mem = _put(store, "we capped the rate limiter at one")
    addFacet(store, chunk, "entity", "rate_limiter")
    addFacet(store, mem, "entity", "rate_limiter")
    e = Enricher(store, home=tmp_path)
    lines = e.lines({"atomId": chunk})
    assert lines[0] == "at projects/obol/api/rate.go#L2-2"
    assert lines[1].startswith(f"relates -> p3://{mem} ")


def test_enricher_empty_for_memory_kind_and_on_any_failure(store):
    mem = _put(store, "a memory atom")
    assert Enricher(store).lines({"atomId": mem}) == []
    chunk = _put(store, "body", kind="document_chunk",
                 sourceRef="kv_cache/vector_meta.db#rowid=1")
    # Unknown root: no location; no facets: no relations; empty, no raise.
    assert Enricher(store).lines({"atomId": chunk}) == []
