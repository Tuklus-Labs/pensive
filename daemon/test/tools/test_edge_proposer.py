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
