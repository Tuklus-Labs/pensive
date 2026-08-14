"""Serve-time enrichment: chunk location + related-memory attachment."""
import os
import signal
from contextlib import contextmanager

import pytest

from recall.enrich import _MAX_FILE_BYTES, locateChunk, relatedMemory, Enricher
from store.store import openStore, putAtom, addFacet, addEdge


class _Blocked(Exception):
    """Raised by the alarm below. Deliberately NOT an OSError.

    TimeoutError, the obvious choice, IS an OSError subclass, so the read
    path's own ``except OSError`` swallows it and the test sees a tidy None --
    the guard reports success because the code it is guarding caught the
    alarm. Found by the sabotage gate, not by reading the code.
    """


@contextmanager
def _mustNotBlock(seconds=5):
    """Turn a hang into a failure.

    Duplicated from test_refs deliberately: it guards the enrichment path
    end-to-end, and a shared helper that lived in one of these files would be
    the file the other one stops importing during a refactor. A fifo reached
    through a ref parks the reader forever, so without this the suite does not
    fail, it stops.
    """
    def onAlarm(sig, frame):
        raise _Blocked(f"blocked for more than {seconds}s")

    previous = signal.signal(signal.SIGALRM, onAlarm)
    signal.alarm(seconds)
    try:
        yield
    except _Blocked as exc:
        pytest.fail(f"enrichment blocked instead of skipping: {exc}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


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
#
# locateChunk takes an OPEN handle, not a path. Opening is refs.openRef's job
# (it is the only code that knows which root a ref is confined to), and a
# reader that cannot be handed a name cannot be aimed at a different file
# between the check and the read.


def test_locate_exact_span(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("line one\ndef target():\n    return 42\nline four\n")
    with open(f, "rb") as fh:
        assert locateChunk("def target():\n    return 42", fh) == (2, 3)


def test_locate_whitespace_normalized(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("a\n\ndef target():\n\n    return   42\nz\n")
    # The stored chunk collapsed blank lines and run-length spaces; the span
    # still resolves to the file's own line numbers.
    with open(f, "rb") as fh:
        assert locateChunk("def target():\n    return 42", fh) == (3, 5)


def test_locate_miss(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("nothing relevant\n")
    with open(f, "rb") as fh:
        assert locateChunk("absent text", fh) is None


def test_locate_refuses_a_file_over_the_cap(tmp_path):
    # The cap is spent on the BYTES READ, not on a size measured beforehand. A
    # stat is a snapshot of a NAME: measured against the pre-fix code, a stat
    # reporting 6 bytes was followed by a read that returned 6 MiB, because a
    # writer got in between the two lookups.
    #
    # The needle sits past the cap so the assertion is loud: drop the bound and
    # this returns a span instead of None.
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * _MAX_FILE_BYTES + b"\nneedle here\n")
    with open(big, "rb") as fh:
        assert locateChunk("needle here", fh) is None


def test_locate_reads_a_file_at_exactly_the_cap(tmp_path):
    # The other side of the boundary, so the cap cannot be "fixed" by refusing
    # everything: a file exactly at the limit is still searched.
    needle = b"needle here\n"
    edge = tmp_path / "edge.py"
    edge.write_bytes(b"x" * (_MAX_FILE_BYTES - len(needle) - 1) + b"\n" + needle)
    assert edge.stat().st_size == _MAX_FILE_BYTES
    with open(edge, "rb") as fh:
        assert locateChunk("needle here", fh) == (2, 2)


def test_locate_refuses_a_path(tmp_path):
    # The pre-fix calling convention, refused loudly. Handing the reader a name
    # is the ONLY way the check/use window can reopen -- the reader would go
    # back to the filesystem for a fresh lookup, and whatever it found there
    # would be a different question from the one the ref was validated against.
    # A quiet failure here would let the next refactor reintroduce the bug and
    # still see green, so this is a raise, not a None.
    f = tmp_path / "x.py"
    f.write_text("def target(): pass\n")
    with pytest.raises(TypeError):
        locateChunk("def target(): pass", f)
    with pytest.raises(TypeError):
        locateChunk("def target(): pass", str(f))


def test_locate_refuses_undecodable_bytes(tmp_path):
    # A binary blob under the cap decodes to nothing usable; enrichment yields
    # no line rather than an exception (the module's failure policy).
    blob = tmp_path / "x.bin"
    blob.write_bytes(b"\xff\xfe\x00\x01 not utf-8")
    with open(blob, "rb") as fh:
        assert locateChunk("anything", fh) is None


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


def test_enricher_prefers_real_relates_edges(store):
    chunk = _put(store, "chunk body", kind="document_chunk")
    viaEdge = _put(store, "memory linked by verified edge")
    viaFacet = _put(store, "memory linked only by facet")
    addFacet(store, chunk, "entity", "shared_e")
    addFacet(store, viaFacet, "entity", "shared_e")
    addEdge(store, {"src": chunk, "dst": viaEdge, "type": "relates",
                    "weight": 0.9})
    lines = Enricher(store).lines({"atomId": chunk})
    joined = "\n".join(lines)
    assert f"relates -> p3://{viaEdge}" in joined
    assert viaFacet not in joined              # edges replace the facet join


def test_edge_to_dead_memory_falls_back_to_facets(store):
    chunk = _put(store, "chunk body", kind="document_chunk")
    dead = _put(store, "superseded memory")
    live = _put(store, "facet-linked memory")
    addEdge(store, {"src": chunk, "dst": dead, "type": "relates",
                    "weight": 0.9})
    store._conn.execute(
        "UPDATE atoms SET status='superseded' WHERE id=?", (dead,))
    store._conn.commit()
    addFacet(store, chunk, "entity", "shared_e")
    addFacet(store, live, "entity", "shared_e")
    lines = Enricher(store).lines({"atomId": chunk})
    joined = "\n".join(lines)
    assert dead not in joined                  # dead edge target never renders
    assert f"relates -> p3://{live}" in joined # fallback still works


# ---- Enricher: the poisoned-ref path -------------------------------------- #
#
# Reachability, end to end: a sourceRef is store data, and a document_chunk's
# provenance is resolved and READ during ordinary recall enrichment. These
# assert on the whole path rather than on refs.openRef alone, because the
# guarantee that matters is "recall never reads that file", not "one function
# returns None".

def _secretTree(tmp_path, secret="ROOT_PASSWORD=phosphor-7482\n"):
    """A home whose projects root sits beside a secret it must never read."""
    (tmp_path / "Projects" / "pkg").mkdir(parents=True)
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    (outside / "secret.env").write_text(secret)
    return outside / "secret.env", secret


def test_enricher_never_locates_a_chunk_through_an_outbound_symlink(
        store, tmp_path):
    secretFile, secret = _secretTree(tmp_path)
    (tmp_path / "Projects" / "pkg" / "cfg.env").symlink_to(secretFile)
    # The chunk text IS the secret, so a located span is proof the daemon read
    # the file: the span could not be found any other way.
    chunk = _put(store, secret.strip(), kind="document_chunk",
                 sourceRef="projects/pkg/cfg.env#c0")
    lines = Enricher(store, home=tmp_path).lines({"atomId": chunk})
    assert not any(l.startswith("at ") for l in lines)


def test_enricher_never_locates_a_chunk_through_a_symlinked_parent(
        store, tmp_path):
    secretFile, secret = _secretTree(tmp_path)
    (secretFile.parent / "cfg.env").write_text(secret)
    (tmp_path / "Projects" / "pkg").rmdir()
    (tmp_path / "Projects" / "pkg").symlink_to(secretFile.parent)
    chunk = _put(store, secret.strip(), kind="document_chunk",
                 sourceRef="projects/pkg/cfg.env#c0")
    lines = Enricher(store, home=tmp_path).lines({"atomId": chunk})
    assert not any(l.startswith("at ") for l in lines)


def test_enricher_does_not_hang_on_a_fifo_ref(store, tmp_path):
    # The denial of service: a ref naming a fifo parks the recall thread inside
    # open() forever, and recall answers nothing while it waits.
    (tmp_path / "Projects" / "pkg").mkdir(parents=True)
    os.mkfifo(tmp_path / "Projects" / "pkg" / "pipe.env")
    chunk = _put(store, "anything", kind="document_chunk",
                 sourceRef="projects/pkg/pipe.env#c0")
    with _mustNotBlock():
        lines = Enricher(store, home=tmp_path).lines({"atomId": chunk})
    assert not any(l.startswith("at ") for l in lines)


def test_enricher_skips_a_ref_whose_file_is_gone(store, tmp_path):
    # The ordinary miss, kept next to the attacks so a fix that refuses
    # everything is not mistaken for a fix that refuses the right things.
    (tmp_path / "Projects" / "pkg").mkdir(parents=True)
    chunk = _put(store, "func Gone() {}", kind="document_chunk",
                 sourceRef="projects/pkg/gone.go#c0")
    lines = Enricher(store, home=tmp_path).lines({"atomId": chunk})
    assert not any(l.startswith("at ") for l in lines)


def test_edges_ordered_by_weight_and_capped(store):
    chunk = _put(store, "chunk body", kind="document_chunk")
    mems = [_put(store, f"memory {i}") for i in range(4)]
    weights = [0.2, 0.9, 0.5, 0.7]
    for m, w in zip(mems, weights):
        addEdge(store, {"src": chunk, "dst": m, "type": "relates",
                        "weight": w})
    lines = Enricher(store, relatedLimit=2).lines({"atomId": chunk})
    rel = [l for l in lines if l.startswith("relates ->")]
    assert len(rel) == 2
    assert f"p3://{mems[1]}" in rel[0]         # 0.9 first
    assert f"p3://{mems[3]}" in rel[1]         # 0.7 second
