"""assemblePayload enrichment: attachment lines + per-entry degrade order."""
import pytest

from recall.payload import assemblePayload, estimateTokens
from store.store import openStore, putAtom


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text, kind="document_chunk"):
    return putAtom(store, {
        "text": text, "kind": kind, "project": "aegis",
        "importance": 0.0, "provenance": {"source": "bulk-import"},
    })


def _result(atomId):
    return {"atomId": atomId, "score": 1.0, "confidence": 0.9,
            "shouldTrust": True, "why": "why"}


class _FakeEnricher:
    def __init__(self, lines):
        self._lines = lines

    def lines(self, result):
        return list(self._lines)


def test_no_enricher_is_legacy_identical(store):
    aid = _put(store, "chunk body")
    legacy = assemblePayload(store, [_result(aid)], 1500)
    explicit = assemblePayload(store, [_result(aid)], 1500, enricher=None)
    assert legacy == explicit


def test_enricher_lines_appended_after_provenance(store):
    aid = _put(store, "chunk body")
    enr = _FakeEnricher(["at projects/x/y.py#L1-2",
                         "relates -> p3://SOME_ID a gist"])
    payload, _, _ = assemblePayload(store, [_result(aid)], 1500, enricher=enr)
    lines = payload.splitlines()
    assert lines[-2] == "at projects/x/y.py#L1-2"
    assert lines[-1] == "relates -> p3://SOME_ID a gist"
    assert "chunk body" in payload


def test_degrade_drops_relates_then_at_before_entry(store):
    aid = _put(store, "chunk body")
    enr = _FakeEnricher(["at projects/x/y.py#L1-2",
                         "relates -> p3://A first gist",
                         "relates -> p3://B second gist"])
    bare, bareTokens, _ = assemblePayload(store, [_result(aid)], 1500)
    # Budget exactly the bare entry: every attachment must drop, entry kept.
    payload, _, _ = assemblePayload(store, [_result(aid)], bareTokens,
                                    enricher=enr)
    assert payload == bare
    # Budget = bare + the at-line only: relates lines drop, at-line stays.
    atLine = "at projects/x/y.py#L1-2"
    midBudget = estimateTokens(bare + "\n" + atLine)
    payload2, _, _ = assemblePayload(store, [_result(aid)], midBudget,
                                     enricher=enr)
    assert atLine in payload2
    assert "relates ->" not in payload2


def test_enricher_exception_never_breaks_payload(store):
    aid = _put(store, "chunk body")

    class _Bomb:
        def lines(self, result):
            raise RuntimeError("enrichment exploded")

    payload, _, _ = assemblePayload(store, [_result(aid)], 1500,
                                    enricher=_Bomb())
    assert "chunk body" in payload  # entry rendered bare, no raise
