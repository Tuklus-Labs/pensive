"""Gate for what an emit actually STORES, as distinct from what it returns.

THE CLAIM UNDER TEST, stated so a drifted question reads as a wrong sentence:

    A one-argument shorthand emit stores a body that IS the caller's sentence.
    It does not wrap that sentence in synthesized scaffolding, and it never
    writes the sentence twice. The shape ("discovery", "failed approach") is
    carried as structure, not as prose at the head of the body.

Why the distinction matters, and why the legacy result string is asserted here
too: the compat RESULT string is a pinned contract (the shadow diff pins it, and
a drift reads as a broken agent), while the stored TEXT never was. Those are two
different artifacts and only one of them is frozen. This gate holds the frozen
one still while the other changes.

Born 2026-08-12. ``handle_emit_failure`` synthesized ``shape="failed approach"``,
``approach="attempted"``, ``outcome="failed"`` and set BOTH ``reason`` and
``principle`` to the caller's single sentence, so ``_composeAtomText`` rendered
"failed approach / approach: attempted / outcome: failed. <sentence> /
principle: <the same sentence>". Measured cost across the live store: 1,091
discovery-template atoms and 202 failed-approach stutters, each carrying an
identical 49-to-53 character prefix. That prefix is a constant contribution to
every one of those embeddings, and it occupies most of the 80 characters a
reader sees in a brief, so the two tools written to REMOVE the emit tax were
producing the least legible rows in the store.
"""
import re

import pytest

from serve import mcp as M
from serve.mcp import ServeContext, dispatch
from recall.embedder import Embedder
from store.store import openStore, facetsOf

MODEL_ID = "BAAI/bge-small-en-v1.5"
_UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"

pytestmark = pytest.mark.filterwarnings(
    "ignore:builtin type SwigPy.* has no __module__ attribute:DeprecationWarning"
)


@pytest.fixture(scope="session")
def embedder():
    return Embedder(MODEL_ID)


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def ctx(store, embedder):
    return ServeContext(store, embedder, MODEL_ID, agent="heph")


def _lastRow(ctx):
    return ctx.store._conn.execute(
        "SELECT id, text FROM atoms ORDER BY created_at DESC, id DESC LIMIT 1"
    ).fetchone()


SENTENCE = "spreading activation stays sub-millisecond at fifty million documents"
FAILURE = "networkx blew the memory budget at ten million nodes"


# --------------------------------------------------------------------------- #
# what gets STORED
# --------------------------------------------------------------------------- #


def test_discovery_stores_the_sentence_not_scaffolding(ctx):
    dispatch(ctx, "engram_emit_discovery",
             {"project": "pensive", "principle": SENTENCE})
    _, text = _lastRow(ctx)
    assert text.strip() == SENTENCE, (
        "a one-argument discovery must store the caller's sentence verbatim; "
        f"got {text!r}"
    )


def test_failure_stores_the_sentence_not_scaffolding(ctx):
    dispatch(ctx, "engram_emit_failure",
             {"project": "pensive", "principle": FAILURE})
    _, text = _lastRow(ctx)
    assert text.strip() == FAILURE


def test_no_synthesized_scaffolding_survives_in_a_shorthand_body(ctx):
    """The exact strings that were eating the visible gist."""
    for tool, sentence in (("engram_emit_discovery", SENTENCE),
                           ("engram_emit_failure", FAILURE)):
        dispatch(ctx, tool, {"project": "pensive", "principle": sentence})
        _, text = _lastRow(ctx)
        for scaffold in ("approach: observed", "approach: attempted",
                         "outcome: succeeded", "outcome: failed",
                         "failed approach", "principle:"):
            assert scaffold not in text, (
                f"{tool} still writes {scaffold!r} into the stored body: {text!r}"
            )


def test_a_shorthand_never_writes_the_sentence_twice(ctx):
    """reason and principle were both set to the caller's one sentence, so the
    body carried it twice and the embedding double-weighted one phrasing."""
    dispatch(ctx, "engram_emit_failure",
             {"project": "pensive", "principle": FAILURE})
    _, text = _lastRow(ctx)
    assert text.count(FAILURE) == 1, f"sentence stored {text.count(FAILURE)}x"


def test_shape_is_carried_as_structure_not_prose(ctx):
    """Shape is a property of the atom, not the first line of its body."""
    dispatch(ctx, "engram_emit_discovery",
             {"project": "pensive", "principle": SENTENCE})
    atomId, _ = _lastRow(ctx)
    shapes = [f["value"] for f in facetsOf(ctx.store, atomId) if f["key"] == "shape"]
    assert shapes == ["discovery"], f"expected a shape facet, got {shapes!r}"

    dispatch(ctx, "engram_emit_failure",
             {"project": "pensive", "principle": FAILURE})
    atomId, _ = _lastRow(ctx)
    shapes = [f["value"] for f in facetsOf(ctx.store, atomId) if f["key"] == "shape"]
    assert shapes == ["failed approach"], f"expected a shape facet, got {shapes!r}"


def test_the_full_atom_form_still_composes_its_fields(ctx):
    """Regression guard: a caller who supplies real distinct fields still gets
    them composed. Only the SYNTHESIZED placeholders are gone."""
    dispatch(ctx, "engram_emit_atom", {
        "project": "pensive", "shape": "a real shape", "approach": "a real approach",
        "outcome": "succeeded", "reason": "a real reason", "principle": "a real principle",
    })
    _, text = _lastRow(ctx)
    assert "a real approach" in text
    assert "a real reason" in text
    assert "a real principle" in text


# --------------------------------------------------------------------------- #
# what gets RETURNED (the frozen compat contract)
# --------------------------------------------------------------------------- #


def test_legacy_result_string_is_unchanged_for_discovery(ctx):
    out, isError = dispatch(ctx, "engram_emit_discovery",
                            {"project": "pensive", "principle": SENTENCE})
    assert isError is False
    assert re.match(
        r"atom \[succeeded\] emitted \(emission_id: " + _UUID_RE + r"\): ", out)
    assert SENTENCE[:80] in out


def test_legacy_result_string_is_unchanged_for_failure(ctx):
    out, isError = dispatch(ctx, "engram_emit_failure",
                            {"project": "pensive", "principle": FAILURE})
    assert isError is False
    assert re.match(
        r"atom \[failed\] emitted \(emission_id: " + _UUID_RE + r"\): ", out)
    assert FAILURE[:80] in out


def test_shorthands_still_stamp_the_agent(ctx):
    """The old delegation rebuilt an argument dict and copied a fixed key list
    across, so a key left off that list was dropped in silence and the emit
    reported success while stamping nothing. Whatever replaces the delegation
    must keep every emit on one provenance path."""
    for tool, sentence in (("engram_emit_discovery", SENTENCE),
                           ("engram_emit_failure", FAILURE)):
        dispatch(ctx, tool, {"project": "pensive", "principle": sentence,
                             "agent": "fable-agent"})
        atomId, _ = _lastRow(ctx)
        row = ctx.store._conn.execute(
            "SELECT agent FROM provenance WHERE atom_id = ?", (atomId,)).fetchone()
        assert row[0] == "fable-agent", f"{tool} dropped the agent"


def test_shorthands_still_record_the_project(ctx):
    for tool in ("engram_emit_discovery", "engram_emit_failure"):
        dispatch(ctx, tool, {"project": "porchlight", "principle": SENTENCE})
        atomId, _ = _lastRow(ctx)
        row = ctx.store._conn.execute(
            "SELECT project FROM atoms WHERE id = ?", (atomId,)).fetchone()
        assert row[0] == "porchlight"
