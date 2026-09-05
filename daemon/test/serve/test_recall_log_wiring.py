import pytest

from lifecycle.importance import accrueImportance
from serve import mcp
from serve.mcp import ServeContext, dispatch
from store.store import getAtom, openStore, putAtom


class FakeEmbedder:
    modelId = "fake-model"

    def embed(self, texts):
        import hashlib
        import numpy as np

        vectors = []
        for text in texts:
            vec = np.zeros(128, dtype=np.float32)
            for token in text.lower().split():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                vec[int.from_bytes(digest, "big") % vec.shape[0]] += 1.0
            norm = float(np.linalg.norm(vec))
            if norm:
                vec = vec / norm
            vectors.append(vec)
        return vectors


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def ctx(store):
    return ServeContext(store, FakeEmbedder(), FakeEmbedder.modelId, agent="heph")


def _put(store, text, importance=0.0):
    return putAtom(store, {
        "text": text,
        "kind": "atom",
        "project": "pensive",
        "importance": importance,
        "provenance": {"source": "bulk-import"},
    })


def _recall_result(atom_ids):
    return {
        "results": [
            {
                "atomId": atom_id,
                "confidence": 0.91 - (i * 0.01),
                "shouldTrust": True,
                "why": "test result",
            }
            for i, atom_id in enumerate(atom_ids)
        ],
        "payload": "\n\n".join(f"p3://{atom_id} | payload" for atom_id in atom_ids),
        "tokensUsed": 1,
        "lowConfidence": False,
    }


def _recall_log_rows(store):
    return store._conn.execute(
        "SELECT atom_id, query, source_ref, weight, processed_at "
        "FROM recall_log ORDER BY rowid"
    ).fetchall()


def test_native_recall_logs_exactly_returned_atom_ids(ctx, monkeypatch):
    """...and stamps the TIER it served from.

    source_ref moved from "mcp.recall" to "mcp.recall.<tier>" on 2026-08-13 when
    the L2/L3 tiers landed. Deliberate: recall_log is the only record of what this
    daemon was actually asked to do, and without the tier a traffic split cannot
    be reconstructed after the fact. A cross-vendor review asked for exactly that
    breakdown after 91.4% of all historical traffic turned out to be one caller
    nobody had attributed. Nothing else in the tree asserts the old literal.
    """
    first = _put(ctx.store, "first returned memory")
    second = _put(ctx.store, "second returned memory")
    omitted = _put(ctx.store, "candidate not returned")
    monkeypatch.setattr(mcp, "recall", lambda *a, **kw: _recall_result([first, second]))

    text, is_error = dispatch(ctx, "recall", {"query": "returned memories", "k": 2})

    assert is_error is False
    assert f"p3://{first}" in text
    assert f"p3://{second}" in text
    assert omitted not in [row[0] for row in _recall_log_rows(ctx.store)]
    assert _recall_log_rows(ctx.store) == [
        (first, "returned memories", "mcp.recall.L2", 1.0, None),
        (second, "returned memories", "mcp.recall.L2", 1.0, None),
    ]


def test_compat_recall_logs_exactly_returned_atom_ids(ctx, monkeypatch):
    first = _put(ctx.store, "compat returned alpha")
    second = _put(ctx.store, "compat returned beta")
    monkeypatch.setattr(mcp, "recall", lambda *a, **kw: _recall_result([first, second]))

    text, is_error = dispatch(ctx, "pensive_recall", {"query": "compat returned", "limit": 2})

    assert is_error is False
    assert text.startswith("Found 2 memories:")
    assert _recall_log_rows(ctx.store) == [
        (first, "compat returned", "mcp.pensive_recall", 1.0, None),
        (second, "compat returned", "mcp.pensive_recall", 1.0, None),
    ]


def test_empty_recall_results_write_no_recall_log_rows(ctx, monkeypatch):
    monkeypatch.setattr(mcp, "recall", lambda *a, **kw: _recall_result([]))

    text, is_error = dispatch(ctx, "pensive_recall", {"query": "empty", "project": "ghost"})

    assert is_error is False
    assert text == "No memories found for query: empty"
    assert _recall_log_rows(ctx.store) == []


def test_log_recall_failure_leaves_recall_response_intact(ctx, monkeypatch):
    atom_id = _put(ctx.store, "response survives telemetry failure")
    monkeypatch.setattr(mcp, "recall", lambda *a, **kw: _recall_result([atom_id]))

    def boom(*_args, **_kwargs):
        raise RuntimeError("database is briefly locked")

    monkeypatch.setattr(mcp, "logRecall", boom)

    text, is_error = dispatch(ctx, "recall", {"query": "survives"})

    assert is_error is False
    assert f"p3://{atom_id}" in text
    assert ctx.recallLogErrors == 1
    assert _recall_log_rows(ctx.store) == []


def test_recall_log_rows_are_exposure_not_helpfulness_end_to_end(ctx, monkeypatch):
    atom_id = _put(ctx.store, "retrieval alone cannot prove usefulness", importance=0.10)
    monkeypatch.setattr(mcp, "recall", lambda *a, **kw: _recall_result([atom_id]))

    text, is_error = dispatch(ctx, "recall", {"query": "importance signal"})
    report = accrueImportance(ctx.store)

    assert is_error is False, 'ordinary recall still succeeds'
    assert f"p3://{atom_id}" in text, 'served handle stays usable'
    assert report == {"processed": 0, "updated": 0, "missingAtoms": 0}, 'exposure is not helpful feedback'
    assert getAtom(ctx.store, atom_id)["importance"] == pytest.approx(0.10), 'retrieval cannot inflate importance'
    assert _recall_log_rows(ctx.store)[0][4] is None, 'legacy telemetry remains historical exposure data'
