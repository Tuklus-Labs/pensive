"""Edge proposer: specificity-scored memory-to-chunk candidate pairs."""
import json
import sys
from pathlib import Path

import pytest

_DAEMON = Path(__file__).resolve().parents[2]
if str(_DAEMON / "tools") not in sys.path:
    sys.path.insert(0, str(_DAEMON / "tools"))

from edge_proposer import proposeForAtom, emitProposals
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


def test_proposes_rare_shared_entity_chunk_first(store):
    mem = _put(store, "decision about the csr publish lock")
    hot = _put(store, "def compile(): csr publish lock impl",
               kind="document_chunk", sourceRef="projects/p/x.py#c0")
    cold = _put(store, "def unrelated(): pass", kind="document_chunk")
    other = _put(store, "another chunk sharing a commoner entity",
                 kind="document_chunk")
    addFacet(store, mem, "entity", "csr_lock")
    addFacet(store, mem, "entity", "python")
    addFacet(store, hot, "entity", "csr_lock")
    addFacet(store, other, "entity", "python")
    # freq: csr_lock=2 (mem+hot), python=3 (mem+other+below)
    extra = _put(store, "yet another python mention", kind="document_chunk")
    addFacet(store, extra, "entity", "python")
    got = proposeForAtom(store, mem)
    ids = [p["chunkId"] for p in got]
    assert ids[0] == hot                       # rare entity wins
    assert cold not in ids                     # no shared entity
    assert all(p["memId"] == mem for p in got)
    assert got[0]["score"] > 0


def test_hub_entities_are_damped_out(store):
    mem = _put(store, "a memory atom")
    chunk = _put(store, "a chunk", kind="document_chunk")
    addFacet(store, mem, "entity", "hub_value")
    addFacet(store, chunk, "entity", "hub_value")
    # Push hub_value's frequency past the hubCap.
    for i in range(6):
        extra = _put(store, f"filler {i}", kind="document_chunk")
        addFacet(store, extra, "entity", "hub_value")
    got = proposeForAtom(store, mem, hubCap=5)
    assert got == []                           # only shared entity was a hub


def test_cap_and_min_score(store):
    mem = _put(store, "memory with one rare entity")
    addFacet(store, mem, "entity", "rare_e")
    chunks = []
    for i in range(5):
        c = _put(store, f"chunk {i}", kind="document_chunk")
        addFacet(store, c, "entity", "rare_e")
        chunks.append(c)
    got = proposeForAtom(store, mem, maxPerAtom=2)
    assert len(got) == 2
    # A ridiculous min score filters everything.
    assert proposeForAtom(store, mem, minScore=99.0) == []


def test_only_live_memory_and_live_chunks(store):
    mem = _put(store, "memory")
    chunk = _put(store, "chunk", kind="document_chunk")
    addFacet(store, mem, "entity", "shared_e")
    addFacet(store, chunk, "entity", "shared_e")
    store._conn.execute(
        "UPDATE atoms SET status='superseded' WHERE id=?", (chunk,))
    store._conn.commit()
    assert proposeForAtom(store, mem) == []


def test_emitProposals_batches_and_self_contained_lines(store, tmp_path):
    mem = _put(store, "the decision text " + "x" * 3000)
    for i in range(3):
        c = _put(store, f"chunk body {i} " + "y" * 3000,
                 kind="document_chunk", sourceRef=f"projects/p/f{i}.py#c0")
        addFacet(store, c, "entity", "rare_e")
    addFacet(store, mem, "entity", "rare_e")
    out = tmp_path / "props"
    report = emitProposals(store, out, batchSize=2, textCap=100)
    assert report["proposals"] == 3
    assert report["batches"] == 2              # 2 + 1
    files = sorted(out.glob("proposals-*.jsonl"))
    assert len(files) == 2
    lines = [json.loads(l) for f in files for l in f.read_text().splitlines()]
    assert len(lines) == 3
    for line in lines:
        # Self-contained: verify agents never need the store.
        assert set(line) >= {"memId", "chunkId", "score", "entities",
                             "memKind", "memText", "chunkRef", "chunkText"}
        assert len(line["memText"]) <= 100
        assert len(line["chunkText"]) <= 100
        assert "\n" not in line["memText"]     # whitespace-collapsed


from edge_proposer import buildFreqMap, proposeDense

MODEL_ID = "BAAI/bge-small-en-v1.5"


def test_freqMap_matches_per_atom_scoring(store):
    mem = _put(store, "memory text")
    hot = _put(store, "chunk text", kind="document_chunk")
    addFacet(store, mem, "entity", "rare_e")
    addFacet(store, hot, "entity", "rare_e")
    fm = buildFreqMap(store)
    assert fm["rare_e"] == 2
    withMap = proposeForAtom(store, mem, freqMap=fm)
    without = proposeForAtom(store, mem)
    assert withMap == without           # identical scoring, no freq SQL


def test_proposeDense_finds_semantic_neighbor(store):
    import pytest
    emb = pytest.importorskip("recall.embedder")
    from recall.embedder import Embedder, embedMissing
    from recall.vector_index import buildClassIndexes
    embedder = Embedder(MODEL_ID)
    mem = _put(store, "we fixed the csr matrix publish race with a lock")
    close = _put(store, "def compile(): publish csr matrix under build lock",
                 kind="document_chunk")
    far = _put(store, "banana bread recipe with walnuts",
               kind="document_chunk")
    embedMissing(store, embedder)
    codeIndex = buildClassIndexes(store, MODEL_ID)["code"]
    got = proposeDense(store, codeIndex, mem, MODEL_ID, topK=2, minSim=0.3)
    ids = [p["chunkId"] for p in got]
    assert close in ids
    assert all(p["channel"] == "dense" for p in got)
    assert all(p["score"] >= 0.3 for p in got)
    tight = proposeDense(store, codeIndex, mem, MODEL_ID, topK=2, minSim=0.99)
    assert tight == []                  # threshold filters


def test_proposeDense_missing_embedding_returns_empty(store):
    mem = _put(store, "never embedded")
    class _BombIndex:
        def search(self, vec, k):
            raise AssertionError("must not search without a vector")
    assert proposeDense(store, _BombIndex(), mem, MODEL_ID) == []


def test_emitProposals_merges_channels_dedup_keeps_higher(store, tmp_path):
    import json
    import pytest
    pytest.importorskip("recall.embedder")
    from recall.embedder import Embedder, embedMissing
    embedder = Embedder(MODEL_ID)
    mem = _put(store, "we fixed the csr matrix publish race with a lock")
    both = _put(store, "def compile(): publish csr matrix under build lock",
                kind="document_chunk")
    addFacet(store, mem, "entity", "rare_e")
    addFacet(store, both, "entity", "rare_e")   # entity AND dense candidate
    embedMissing(store, embedder)
    out = tmp_path / "props"
    report = emitProposals(store, out, minSim=0.3)
    lines = [json.loads(l) for f in sorted(out.glob("*.jsonl"))
             for l in f.read_text().splitlines()]
    ours = [l for l in lines if l["chunkId"] == both]
    assert len(ours) == 1               # deduped across channels
    assert report["merged"] == report["proposals"]
    assert report["entityProposals"] >= 1
    assert report["denseProposals"] >= 1
