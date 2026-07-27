"""Aux dense signal: second-model fusion, degradation, and scoping.

Risk model (what could silently break, and the test that catches it):

- **Degradation regression.** The aux path failing at query time must leave
  recall BYTE-identical to the two-signal engine -- a partial fusion, a raised
  exception, or a half-tagged signalHits would corrupt live recall the moment
  the API blips. Caught by the exact-equality degradation test (failing aux vs
  aux=None over real models, frozen clock).
- **Evidence mis-tagging.** Aux hits must count as dense-family evidence for
  trust ("dense") while staying distinguishable ("openai"). A tag drop weakens
  agreement scoring silently. Caught by the assessTrust-spy test.
- **Filter leak.** A project filterSet that does not apply to the aux list
  would let out-of-project atoms into fusion. Caught by the rrf-spy test.
- **Class leak.** The aux index is scoped to the memory class; a code-class
  atom with an aux vector must never enter the memory aux index. Caught by the
  buildIndexes scoping test.
- **Key plumbing.** The daemon runs under systemd without the shell env, so
  ~/.keys parsing is load-bearing; both quoted and bare export forms must
  parse, and a missing key must disable (RuntimeError), not crash later.

The aux embedder double returns unit-normalized float32 vectors of fixed width
(the same invariants the API guarantees); the failure double raises the same
family of exception a network outage does. Live-shape verification (real API,
real backfilled vectors) is a deploy-time check, deliberately not a unit test.
"""
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

import recall.engine as engine
import recall.aux_dense as aux_dense
from recall.aux_dense import AuxDense, OpenAIEmbedder, readOpenAiKey
from recall.embedder import Embedder, vecToBlob
from recall.engine import recall, _auxHits
from recall.fusion import rrf as realRrf
from recall.trust import assessTrust as realAssessTrust
from store.store import openStore, putAtom

MODEL_ID = "BAAI/bge-small-en-v1.5"
AUX_MODEL_ID = "fake-aux-model"

# Frozen reference clock (2033-05-18 UTC): recall samples now once per call, and
# the degradation test asserts EXACT equality across two calls, so both must see
# the same clock.
NOW = 2_000_000_000

pytestmark = pytest.mark.filterwarnings(
    "ignore:builtin type SwigPy.* has no __module__ attribute:DeprecationWarning"
)


# --------------------------------------------------------------------------- #
# Doubles                                                                     #
# --------------------------------------------------------------------------- #


class FakeAuxEmbedder:
    """Duck-type twin of OpenAIEmbedder minus the network.

    Deterministic: equal text embeds equal (crc32-seeded), vectors are
    unit-normalized float32 of fixed width -- the invariants the engine and the
    flat index actually rely on.
    """

    def __init__(self, modelId=AUX_MODEL_ID, dim=8):
        self.modelId = modelId
        self.dim = dim

    def embed(self, texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(zlib.crc32(t.encode()))
            v = rng.standard_normal(self.dim).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return out


class FailingAuxEmbedder(FakeAuxEmbedder):
    """The outage double: same contract, every embed raises like a dead API."""

    def embed(self, texts):
        raise ConnectionError("synthetic aux outage")


class FakeIndex:
    """A VectorIndex double preloaded with its ranked answer."""

    def __init__(self, pairs):
        self._pairs = list(pairs)

    def search(self, vec, k):
        return self._pairs[:k]


def fakeAux(pairs, embedder=None):
    aux = AuxDense(embedder or FakeAuxEmbedder())
    aux.indexes = {"memory": FakeIndex(pairs)}
    return aux


# --------------------------------------------------------------------------- #
# Fixtures (session-scoped real models, per-test store: the house pattern)    #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def embedder():
    return Embedder(MODEL_ID)


@pytest.fixture(scope="session")
def _rerankerWarm():
    from recall.rerank import _getReranker

    _getReranker()


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def frozenClock(monkeypatch):
    monkeypatch.setattr(engine, "time", SimpleNamespace(time=lambda: NOW))


def _put(store, text, kind="atom", project=None):
    return putAtom(store, {
        "text": text, "kind": kind, "project": project,
        "provenance": {"source": "test"},
    })


def _plantAuxVector(store, atomId, embedder, text):
    vec = embedder.embed([text])[0]
    store._conn.execute(
        "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) "
        "VALUES (?, ?, ?, ?)",
        (atomId, embedder.modelId, vecToBlob(vec), NOW),
    )
    store._conn.commit()


# --------------------------------------------------------------------------- #
# _auxHits unit seam                                                          #
# --------------------------------------------------------------------------- #


def test_auxHitsSearchesEveryAuxIndex():
    aux = fakeAux([("a1", 0.9), ("a2", 0.5)])
    aux.indexes["extra"] = FakeIndex([("b1", 0.7)])
    hits = _auxHits(aux, "any query", 10)
    assert ("a1", 0.9) in hits and ("a2", 0.5) in hits and ("b1", 0.7) in hits


def test_auxHitsEmptyOnEmbedFailureAndLogsThrottled(capsys):
    engine._auxFailures = 0
    aux = fakeAux([("a1", 0.9)], embedder=FailingAuxEmbedder())
    for _ in range(3):
        assert _auxHits(aux, "q", 10) == []
    err = capsys.readouterr().err
    # Powers of two: occurrences 1 and 2 log, occurrence 3 does not.
    assert "(1x)" in err and "(2x)" in err and "(3x)" not in err
    assert "synthetic aux outage" in err


# --------------------------------------------------------------------------- #
# Engine integration                                                          #
# --------------------------------------------------------------------------- #


def test_failingAuxIsByteIdenticalToNoAux(store, embedder, _rerankerWarm,
                                          frozenClock):
    _put(store, "the rocm allocator fragments under mixed batch sizes")
    _put(store, "kestrel tail-sitter transition needs a pitch rate limit")
    _put(store, "sqlite wal checkpoint starvation on long readers")
    from recall.embedder import embedMissing
    from recall.vector_index import buildClassIndexes
    embedMissing(store, embedder)
    indexes = buildClassIndexes(store, MODEL_ID)

    engine._auxFailures = 0
    query = "wal checkpoint starvation"
    base = recall(store, indexes, embedder, query, aux=None)
    degraded = recall(store, indexes, embedder, query,
                      aux=fakeAux([], embedder=FailingAuxEmbedder()))
    assert degraded == base


def test_auxHitsTagDenseAndOpenai(store, embedder, _rerankerWarm, frozenClock,
                                  monkeypatch):
    target = _put(store, "turboquant kv cache norm correction pass")
    from recall.embedder import embedMissing
    from recall.vector_index import buildClassIndexes
    embedMissing(store, embedder)
    indexes = buildClassIndexes(store, MODEL_ID)

    captured = {}

    def spy(reranked, signalHits, store_, now):
        captured.update(signalHits)
        return realAssessTrust(reranked, signalHits, store_, now)

    monkeypatch.setattr(engine, "assessTrust", spy)
    recall(store, indexes, embedder, "kv cache quantization",
           aux=fakeAux([(target, 0.95)]))
    assert {"dense", "openai"} <= captured[target]


def test_projectFilterAppliesToAuxList(store, embedder, _rerankerWarm,
                                       frozenClock, monkeypatch):
    inProject = _put(store, "warden assessor confidence floor", project="alpha")
    outOfProject = _put(store, "parlor websocket reconnect backoff",
                        project="beta")
    from recall.embedder import embedMissing
    from recall.vector_index import buildClassIndexes
    embedMissing(store, embedder)
    indexes = buildClassIndexes(store, MODEL_ID)

    seen = {}

    def spy(lists, k=60, weights=None):
        seen["lists"] = [list(l) for l in lists]
        return realRrf(lists, k=k, weights=weights)

    monkeypatch.setattr(engine, "rrf", spy)
    recall(store, indexes, embedder, "confidence floor", project="alpha",
           aux=fakeAux([(outOfProject, 0.99), (inProject, 0.4)]))
    assert len(seen["lists"]) == 3
    auxList = seen["lists"][2]
    assert all(atomId != outOfProject for atomId, _ in auxList)
    assert any(atomId == inProject for atomId, _ in auxList)


# --------------------------------------------------------------------------- #
# AuxDense index scoping over real backfilled rows                            #
# --------------------------------------------------------------------------- #


def test_buildIndexesScopedToMemoryClassOnly(store):
    auxEmbedder = FakeAuxEmbedder()
    memAtom = _put(store, "memory class resident fact")
    codeAtom = _put(store, "def code_chunk(): pass", kind="document_chunk")
    _plantAuxVector(store, memAtom, auxEmbedder, "memory class resident fact")
    _plantAuxVector(store, codeAtom, auxEmbedder, "def code_chunk(): pass")

    aux = AuxDense(auxEmbedder).buildIndexes(store)
    assert set(aux.indexes) == {"memory"}
    queryVec = auxEmbedder.embed(["memory class resident fact"])[0]
    got = [atomId for atomId, _ in aux.indexes["memory"].search(queryVec, 10)]
    assert memAtom in got
    assert codeAtom not in got


def test_reindexScopingMatchesServeContext(store):
    auxEmbedder = FakeAuxEmbedder()
    aux = AuxDense(auxEmbedder).buildIndexes(store)
    a1 = _put(store, "first resident fact")
    _plantAuxVector(store, a1, auxEmbedder, "first resident fact")

    # A code-class write must not refresh the memory aux index...
    aux.reindex(store, kinds=("document_chunk",))
    vec = auxEmbedder.embed(["first resident fact"])[0]
    assert aux.indexes["memory"].search(vec, 5) == []
    # ...a memory-class write must.
    aux.reindex(store, kinds=("atom",))
    got = [atomId for atomId, _ in aux.indexes["memory"].search(vec, 5)]
    assert got == [a1]


# --------------------------------------------------------------------------- #
# Key plumbing                                                                #
# --------------------------------------------------------------------------- #


def test_readOpenAiKeyParsesKeysFileForms(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    keys = tmp_path / "keys"
    keys.write_text('# comment\nexport OPENAI_API_KEY="sk-test-abc123"\n')
    monkeypatch.setattr(aux_dense, "_KEYS_FILE", keys)
    assert readOpenAiKey() == "sk-test-abc123"

    keys.write_text("OPENAI_API_KEY=sk-bare-form-456\n")
    assert readOpenAiKey() == "sk-bare-form-456"

    monkeypatch.setattr(aux_dense, "_KEYS_FILE", tmp_path / "absent")
    assert readOpenAiKey() is None

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-wins")
    assert readOpenAiKey() == "sk-env-wins"


def test_openAIEmbedderWithoutKeyRaises(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(aux_dense, "_KEYS_FILE", tmp_path / "absent")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIEmbedder("text-embedding-3-large")
