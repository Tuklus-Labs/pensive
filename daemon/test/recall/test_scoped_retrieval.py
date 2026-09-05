"""Regression coverage for scope-before-limit recall."""
import time

import numpy as np
import pytest

from recall.aux_dense import AuxDense
from recall.engine import _auxHits, recall
from recall.hnsw_index import HnswIndex
from recall.vector_index import FlatIndex, buildClassIndexes
from store.store import openStore, putAtom


MODEL_ID = "synthetic/scoped-v1"
QUERY = np.asarray([1.0, 0.0], dtype=np.float32)
TARGET = np.asarray([0.8, 0.6], dtype=np.float32)


class StaticEmbedder:
    modelId = MODEL_ID
    dim = 2

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [QUERY.copy() for _ in texts]


@pytest.fixture
def store(tmp_path):
    opened = openStore(tmp_path / "scoped.db")
    try:
        yield opened
    finally:
        opened.close()


def _put(store, text, *, kind="atom", project="other", agent="other",
         occurredAt=None, vector=None):
    atomId = putAtom(store, {
        "text": text,
        "kind": kind,
        "project": project,
        "occurredAt": occurredAt,
        "provenance": {"source": "claude-code", "agent": agent},
    })
    if vector is not None:
        store._conn.execute(
            "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) "
            "VALUES (?, ?, ?, ?)",
            (atomId, MODEL_ID, np.asarray(vector, dtype=np.float32).tobytes(),
             int(time.time())),
        )
        store._conn.commit()
    return atomId


def _denseDistractors(store, count=201, **atomFields):
    return [
        _put(store, f"opaque distractor {i}", vector=QUERY, **atomFields)
        for i in range(count)
    ]


def _ids(result):
    return [row["atomId"] for row in result["results"]]


def test_project_scope_precedes_bm25_cutoff(store):  # I1, B1, C2
    for i in range(201):
        _put(store, f"needle needle needle needle distractor {i}")
    target = _put(store, "needle scoped target", project="wanted")

    result = recall(store, {}, StaticEmbedder(), "needle", project="wanted",
                    tier="L2", k=10)

    got = _ids(result)
    assert target in got, (
        "scoped-recall project-before-limit invariant violated: "
        f"target={target} results={got} distractors=201 signal_cap=200")


def test_agent_scope_precedes_dense_cutoff(store):  # I1, B1
    _denseDistractors(store)
    target = _put(store, "opaque scoped target", agent="grok", vector=TARGET)
    embedder = StaticEmbedder()
    indexes = buildClassIndexes(store, MODEL_ID)

    result = recall(store, indexes, embedder, "semantic query", agent="grok",
                    tier="L2", k=10)

    got = _ids(result)
    assert target in got, (
        "scoped-recall agent-before-dense-limit invariant violated: "
        f"target={target} results={got} distractors=201 signal_cap=200")


def test_effective_time_scope_precedes_dense_cutoff_and_is_inclusive(store):  # I1, B2
    _denseDistractors(store, occurredAt=99)
    atStart = _put(store, "opaque start boundary", occurredAt=100, vector=TARGET)
    atEnd = _put(store, "opaque end boundary", occurredAt=200,
                 vector=np.asarray([0.7, np.sqrt(0.51)], dtype=np.float32))
    createdFallback = _put(
        store, "opaque created fallback", occurredAt=None,
        vector=np.asarray([0.65, np.sqrt(1 - 0.65 ** 2)], dtype=np.float32),
    )
    store._conn.execute(
        "UPDATE atoms SET created_at = 150 WHERE id = ?", (createdFallback,))
    store._conn.commit()
    outside = _put(store, "opaque outside boundary", occurredAt=201,
                   vector=np.asarray([0.6, 0.8], dtype=np.float32))
    embedder = StaticEmbedder()
    indexes = buildClassIndexes(store, MODEL_ID)

    result = recall(store, indexes, embedder, "semantic query",
                    timeScope=(100, 200), tier="L2", k=10)

    got = _ids(result)
    assert got == [atStart, atEnd, createdFallback], (
        "scoped-recall inclusive-time-before-limit invariant violated: "
        f"expected={[atStart, atEnd, createdFallback]} "
        f"outside={outside} results={got}")


def test_explicit_narrative_kind_precedes_dense_cutoff(store):  # I1, C3
    _denseDistractors(store, kind="atom")
    target = _put(store, "opaque narrative target", kind="narrative", vector=TARGET)
    embedder = StaticEmbedder()
    indexes = buildClassIndexes(store, MODEL_ID)

    result = recall(store, indexes, embedder, "semantic query",
                    kinds=["narrative"], k=10, rerankEnabled=False)

    got = _ids(result)
    assert got == [target], (
        "scoped-recall explicit-kind-before-limit invariant violated: "
        f"target={target} results={got} distractors=201 signal_cap=200")


def test_explicit_narrative_kind_precedes_bm25_cutoff(
        store):  # I1, B1, C2, C3
    for i in range(201):
        _put(store, f"needle needle needle needle atom distractor {i}", kind="atom")
    target = _put(store, "needle narrative target", kind="narrative")

    result = recall(
        store, {}, StaticEmbedder(), "needle", kinds=["narrative"], k=10,
        rerankEnabled=False,
    )

    got = _ids(result)
    assert got == [target], (
        "scoped-recall explicit-kind-BM25-before-limit invariant violated: "
        f"target={target} results={got} distractors=201 signal_cap=200")


def test_combined_scope_intersects_all_constraints(store):  # I2
    _denseDistractors(store, kind="narrative", project="wanted",
                      agent="other", occurredAt=150)
    target = _put(store, "opaque combined target", kind="narrative",
                  project="wanted", agent="grok", occurredAt=100, vector=TARGET)
    embedder = StaticEmbedder()
    indexes = buildClassIndexes(store, MODEL_ID)

    result = recall(
        store, indexes, embedder, "semantic query", project="wanted",
        agent="grok", kinds=["narrative"], timeScope=(100, 200), k=10,
        rerankEnabled=False,
    )

    got = _ids(result)
    assert got == [target], (
        "scoped-recall combined-intersection invariant violated: "
        f"target={target} results={got} distractors=201")


def test_combined_scope_precedes_bm25_cutoff(store):  # I1, I2, C2
    for i in range(201):
        _put(
            store, f"needle needle needle needle distractor {i}",
            kind="narrative", project="wanted", agent="other", occurredAt=150,
        )
    target = _put(
        store, "needle combined target", kind="narrative", project="wanted",
        agent="grok", occurredAt=100,
    )

    result = recall(
        store, {}, StaticEmbedder(), "needle", project="wanted", agent="grok",
        kinds=["narrative"], timeScope=(100, 200), k=10,
        rerankEnabled=False,
    )

    got = _ids(result)
    assert got == [target], (
        "scoped-recall combined-BM25-before-limit invariant violated: "
        f"target={target} results={got} distractors=201")


def test_empty_combined_scope_skips_both_embedders(store):  # S2, B3
    _put(store, "project side", project="wanted", agent="other", vector=QUERY)
    _put(store, "agent side", project="other", agent="grok", vector=QUERY)
    base = StaticEmbedder()
    auxEmbedder = StaticEmbedder()
    aux = AuxDense(auxEmbedder)
    aux.indexes = {"memory": FlatIndex().add("unused", QUERY)}

    result = recall(
        store, buildClassIndexes(store, MODEL_ID), base, "semantic query",
        project="wanted", agent="grok", tier="L2", aux=aux, k=10,
    )

    assert result["results"] == [], (
        "scoped-recall empty-intersection invariant violated: "
        f"results={result['results']}")
    assert (base.calls, auxEmbedder.calls) == (0, 0), (
        "scoped-recall empty-intersection work invariant violated: "
        f"base_calls={base.calls} aux_calls={auxEmbedder.calls}")


def test_scoped_flat_and_hnsw_agree_and_exclude_retired_rows():  # I3, S1, M1, P1
    vectors = {
        "disallowed-best": QUERY,
        "retired": np.asarray([0.95, np.sqrt(1 - 0.95 ** 2)], dtype=np.float32),
        "allowed-a": TARGET,
        "allowed-b": np.asarray([0.7, np.sqrt(0.51)], dtype=np.float32),
    }
    allowed = {"retired", "allowed-a", "allowed-b", "absent"}
    indexes = [FlatIndex(), HnswIndex()]
    for index in indexes:
        for atomId, vector in vectors.items():
            index.add(atomId, vector)
        assert index.remove("retired") is True, (
            "scoped-recall retirement setup has no power: "
            f"index={type(index).__name__} retired=retired")

    flat = indexes[0].search(QUERY, 10, allowedIds=allowed)
    hnsw = indexes[1].search(QUERY, 10, allowedIds=allowed)
    expected = ["allowed-a", "allowed-b"]
    assert [atomId for atomId, _ in flat] == expected, (
        "scoped-recall Flat allowed-set invariant violated: "
        f"expected={expected} hits={flat}")
    assert [atomId for atomId, _ in hnsw] == expected, (
        "scoped-recall HNSW allowed-set invariant violated: "
        f"expected={expected} hits={hnsw}")
    assert np.allclose([score for _, score in flat],
                       [score for _, score in hnsw], atol=1e-6), (
        "scoped-recall index score-parity invariant violated: "
        f"flat={flat} hnsw={hnsw}")


def test_l3_embeds_base_query_once_across_two_classes(store):  # I4, C1
    memory = _put(store, "opaque memory", kind="atom", vector=QUERY)
    code = _put(store, "opaque code", kind="document_chunk", vector=TARGET)
    embedder = StaticEmbedder()

    result = recall(store, buildClassIndexes(store, MODEL_ID), embedder,
                    "semantic query", tier="L3", k=10)

    got = set(_ids(result))
    assert got == {memory, code}, (
        "scoped-recall L3 class coverage invariant violated: "
        f"expected={{'{memory}', '{code}'}} results={got}")
    assert embedder.calls == 1, (
        "scoped-recall base-query embed-count invariant violated: "
        f"calls={embedder.calls} active_classes=2")


def test_l2_full_class_keeps_old_duck_index_call_shape(store):  # I5, C1
    atomId = _put(store, "opaque duck result")

    class TwoArgumentIndex:
        def search(self, vector, k):
            return [(atomId, float(np.dot(vector, QUERY)))][:k]

    embedder = StaticEmbedder()
    result = recall(store, {"memory": TwoArgumentIndex()}, embedder,
                    "semantic query", tier="L2", k=10)

    got = _ids(result)
    assert got == [atomId], (
        "scoped-recall unscoped duck-index contract violated: "
        f"target={atomId} results={got}")
    assert embedder.calls == 1, (
        "scoped-recall unscoped duck-index embed invariant violated: "
        f"calls={embedder.calls}")


def test_l2_full_class_skips_out_of_scope_aux_indexes(store):  # I1, I5, C5
    memory = _put(store, "opaque memory target", kind="atom")
    code = _put(store, "opaque code distractor", kind="document_chunk")

    class CountingIndex:
        def __init__(self, hits):
            self.hits = hits
            self.calls = 0

        def search(self, vector, k):
            self.calls += 1
            return self.hits[:k]

    memoryIndex = CountingIndex([(memory, 0.8)])
    codeIndex = CountingIndex([(code, 1.0)])
    auxEmbedder = StaticEmbedder()
    aux = AuxDense(auxEmbedder)
    aux.indexes = {"memory": memoryIndex, "code": codeIndex}

    result = recall(
        store, {}, StaticEmbedder(), "semantic query", tier="L2", aux=aux,
        k=10,
    )

    got = _ids(result)
    assert got == [memory], (
        "scoped-recall native-class result invariant violated: "
        f"memory={memory} code={code} results={got}")
    assert (memoryIndex.calls, codeIndex.calls) == (1, 0), (
        "scoped-recall native-class candidate-universe invariant violated: "
        f"memory_calls={memoryIndex.calls} code_calls={codeIndex.calls}")


def test_aux_dense_scopes_before_top_k_and_embeds_once():  # I1, I4, C4
    embedder = StaticEmbedder()
    aux = AuxDense(embedder, classNames=("memory", "code"))
    targets = []
    for className in aux.classNames:
        index = FlatIndex()
        for i in range(201):
            index.add(f"{className}-other-{i}", QUERY)
        target = f"{className}-target"
        targets.append(target)
        index.add(target, TARGET)
        aux.indexes[className] = index

    hits = _auxHits(aux, "semantic query", 200, allowedIds=set(targets))

    got = [atomId for atomId, _ in hits]
    assert got == targets, (
        "scoped-recall auxiliary-before-limit invariant violated: "
        f"expected={targets} hits={got} distractors_per_class=201")
    assert embedder.calls == 1, (
        "scoped-recall auxiliary embed-count invariant violated: "
        f"calls={embedder.calls} active_classes=2")
