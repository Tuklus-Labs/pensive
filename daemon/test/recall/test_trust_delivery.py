"""Regressions for RISK_MODEL_TRUST_DELIVERY.md; no embedding models required."""
import pytest

from recall.payload import (
    SENTINEL_BUDGET_TOO_SMALL, SENTINEL_LOW_CONFIDENCE, assemblePayload,
    estimateTokens, tier0Handle,
)
from recall.trust import assessTrust, TRUST_FLOOR
from store.store import openStore, putAtom, supersede

NOW = 2_000_000_000


@pytest.fixture
def store(tmp_path):
    value = openStore(tmp_path / "delivery.db")
    try:
        yield value
    finally:
        value.close()


def put(store, text, kind="atom", source="explicit-emit"):
    return putAtom(store, {
        "text": text, "kind": kind, "project": "pensive",
        "occurredAt": NOW, "provenance": {"source": source},
    })


def result(atom, trusted, **extra):
    return dict(atomId=atom, score=1.0, confidence=0.9 if trusted else 0.4,
                shouldTrust=trusted, why="test evidence", **extra)


@pytest.mark.parametrize("signals", [
    {"bm25"}, {"dense"}, {"facet"}, {"dense", "openai"},
])
def test_one_import_signal_family_cannot_establish_trust(store, signals):
    # T1: a lone recent winner gives the blend every opportunity to trust it.
    atom = put(store, "imported evidence", "document_chunk", "bulk-import")
    row = assessTrust([(atom, 1.0)], {atom: signals}, store, NOW)[0]
    assert row["shouldTrust"] is False and row["confidence"] < TRUST_FLOOR, (
        f"T1 independent-evidence rule: signals={signals!r}, result={row!r}")
    assert "uncorroborated" in row["why"], (
        f"T1 explanation must disclose missing corroboration: result={row!r}")


@pytest.mark.parametrize("signals", [
    {"bm25", "dense"}, {"bm25", "facet"}, {"dense", "facet"},
])
def test_two_independent_import_signal_families_remain_eligible(store, signals):
    # T5: prevent fixing single-signal trust by suppressing the entire corpus.
    atom = put(store, "corroborated evidence", "document_chunk", "bulk-import")
    row = assessTrust([(atom, 1.0)], {atom: signals}, store, NOW)[0]
    assert row["shouldTrust"] is True, (
        f"T5 corroborated imports remain eligible: signals={signals!r}, row={row!r}")


@pytest.mark.parametrize("enriched", [False, True])
def test_mixed_payload_limits_weak_result_to_labelled_handle(store, enriched):
    # T2/T6: a trusted neighbor cannot upgrade a weak entry's representation.
    strong = put(store, "the supported conclusion")
    weak_text = "uncertain lead " * 100 + "UNVERIFIED_DETAIL_MUST_NOT_BE_INLINED"
    weak = put(store, weak_text)
    rows = [result(strong, True), result(weak, False)]
    calls = []

    class Enricher:
        def lines(self, row):
            calls.append(row["atomId"])
            return []

    payload, used, low = assemblePayload(
        store, rows, 3000, enricher=Enricher() if enriched else None)
    assert "  the supported conclusion" in payload, (
        f"T6 trusted bodies remain available: payload={payload!r}")
    assert f"p3://{weak}" in payload and "untrusted" in payload, (
        f"T2 weak results remain discoverable and labelled: payload={payload!r}")
    assert not any(line.startswith("  ") and "uncertain lead" in line
                   for line in payload.splitlines()), (
        f"T2 weak full body must not be inlined: payload={payload!r}")
    assert calls == ([strong] if enriched else []), (
        f"T6 enrichment only sees trusted rows: calls={calls!r}")
    assert payload.index(strong) < payload.index(weak) and not low and used <= 3000, (
        f"T6 rank/budget metadata preserved: used={used}, low={low}, payload={payload!r}")


def test_superseded_handle_keeps_successor_pointer(store):
    # T4: handle degradation cannot orphan a historical claim from its correction.
    old = put(store, "outdated conclusion")
    new = put(store, "current conclusion")
    supersede(store, old, new, {"source": "explicit-emit"})
    line = tier0Handle(store, result(old, False, supersededBy=new))
    assert f"superseded by p3://{new}" in line and "untrusted" in line, (
        f"T4 historical handles retain trust state and successor: line={line!r}")
    assert len(line.splitlines()) == 1, (
        f"T4 handle framing stays one physical line: line={line!r}")


@pytest.mark.parametrize("mode", ["empty", "weak", "trusted"])
def test_every_small_budget_is_respected(store, mode):
    # T3: both low-confidence and no-fit sentinels must fit the caller's budget.
    atom = put(store, "a body much too large for the tiny budget " * 20)
    rows = [] if mode == "empty" else [result(atom, mode == "trusted")]
    maximum = max(map(estimateTokens, [SENTINEL_BUDGET_TOO_SMALL,
                                       SENTINEL_LOW_CONFIDENCE]))
    for budget in range(maximum + 1):
        payload, used, low = assemblePayload(store, rows, budget)
        assert used == estimateTokens(payload) and used <= budget, (
            f"T3 estimated budget exceeded: mode={mode}, budget={budget}, "
            f"used={used}, payload={payload!r}")
        assert low is (mode != "trusted"), (
            f"T3 trust state survives empty payload: mode={mode}, low={low}")


def test_negative_payload_budget_is_rejected(store):
    # T3 malformed boundary: negative capacity cannot have a valid payload.
    with pytest.raises(ValueError, match="tokenBudget"):
        assemblePayload(store, [], -1)
