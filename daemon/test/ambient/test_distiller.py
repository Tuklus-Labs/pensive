"""Distiller (Claude Code source) tests: two-stage capture, dedup, anti-storm.

Risk model (what could silently break, and the test that catches it):

- **An explicit emit gets re-summarized by the model.** An ``engram_emit_*`` call
  in a transcript is the agent's OWN curated atom -- running it through the lossy
  stage-2 model would corrupt a deliberate record. The Step-1 test asserts the
  span passes straight through as a live atom (source='claude-code', provenance
  pointing at the span) AND that the FakeModelClient was never called for it.

- **Atom storm.** The distiller tails; the same span WILL be seen twice (a re-tail,
  an overlapping delta). If each sighting inserts, memory floods with duplicates.
  The anti-storm tests feed the same span twice (-> one atom, importance raised)
  and a repetitive transcript (-> bounded atom count), asserting the count stays
  small and importance rises instead.

- **Importance runs away.** A bump-on-dup that never caps sends a hot span's
  importance to infinity, breaking the ranking prior. The cap test feeds the same
  span 100x and asserts importance stops at the documented cap.

- **The verbatim seam regresses.** v3.1 will feed person-sourced material that must
  be preserved verbatim (no distillation, no dedup-merge). The seam test proves a
  policy='verbatim' source inserts EVERY span unmodified, never calls the model,
  and never merges duplicates -- the branch v3.1 fills, exercised now so it cannot
  rot.

- **A bad transcript kills the loop.** An empty delta, a malformed event, or a
  model that errors on one span must not take down the batch. Sabotage tests assert
  empty -> no atoms, malformed -> skipped, model-error -> that span skipped and the
  rest processed.

All tests use the FakeModelClient (no live 35B) and a deterministic FakeEmbedder
(bag-of-words cosine: identical normalized summaries dup, unrelated text does not),
so the suite is fast and never touches the network or GPU.
"""
from ambient.distiller import (
    distill, DISTILL_POLICY, VERBATIM_POLICY, SOURCE_CLAUDE_CODE, IMPORTANCE_CAP,
)
from ambient.dedup import dedup, DEDUP_THRESHOLD
from ambient.segment import segment, KIND_EXPLICIT_EMIT
from store.store import openStore, putAtom, getAtom, atomCount

import pytest


# --------------------------------------------------------------------------- #
# Fakes: a deterministic summarizer and a deterministic embedder               #
# --------------------------------------------------------------------------- #


class FakeModelClient:
    """Stage-2 stand-in. Deterministic: identical span text -> identical summary,
    so the dedup path is exercisable without a live model. Records every call so a
    test can assert the model was (or was not) invoked."""

    def __init__(self):
        self.calls = []

    def summarizeSpan(self, spanText):
        self.calls.append(spanText)
        # A canned house-format atom derived from the span. Deterministic in the
        # span text so two identical spans yield identical (dup-able) summaries.
        gist = " ".join(spanText.split())
        return {
            "text": f"principle: {gist}",
            "kind": "atom",
        }


class FakeEmbedder:
    """Deterministic bag-of-words embedder. cosine(A,B) = shared-words /
    sqrt(|A||B|): identical text -> 1.0, one-word-different -> just under 1.0,
    unrelated -> ~0. No RNG, no network -- the dedup plumbing is what is under
    test, not embedding quality."""

    DIM = 4096

    def embed(self, texts):
        import numpy as np
        out = []
        for t in texts:
            vec = np.zeros(self.DIM, dtype=np.float32)
            for tok in t.lower().split():
                vec[hash(tok) % self.DIM] += 1.0
            n = float(np.linalg.norm(vec))
            if n > 0:
                vec = vec / n
            out.append(vec)
        return out


# --------------------------------------------------------------------------- #
# Fixtures + transcript builders                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _assistantText(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _emitEvent(tool, args):
    return {"role": "assistant", "content": [
        {"type": "text", "text": "recording an atom"},
        {"type": "tool_use", "name": tool, "input": args},
    ]}


def _source(sessionId, events, policy=DISTILL_POLICY, offset=0, **extra):
    src = {"sessionId": sessionId, "policy": policy,
           "deltas": [{"offset": offset, "events": events}]}
    src.update(extra)
    return src


def _atoms(store):
    ids = [r[0] for r in store._conn.execute(
        "SELECT id FROM atoms ORDER BY id").fetchall()]
    return [getAtom(store, i) for i in ids]


# --------------------------------------------------------------------------- #
# Step 1: an explicit emit passes straight through, bypassing the model        #
# --------------------------------------------------------------------------- #


def test_explicit_emit_passes_through_without_model(store):
    model = FakeModelClient()
    embedder = FakeEmbedder()
    principle = "usearch HNSW must pin its score scale to the flat index"
    source = _source("sess-A", [
        _emitEvent("engram_emit_discovery",
                   {"project": "pensive", "principle": principle}),
    ])

    result = distill(store, source, model, embedder)

    # exactly one atom, written as a passthrough (not through stage 2)
    assert atomCount(store) == 1
    assert result["passthrough"] == 1
    assert result["inserted"] == 0
    # the FakeModelClient was NEVER called for the explicit emit
    assert model.calls == [], f"model should not run on an explicit emit: {model.calls}"

    atom = getAtom(store, result["atomIds"][0])
    assert principle in atom["text"]
    prov = atom["provenance"][0]
    assert prov["source"] == SOURCE_CLAUDE_CODE
    assert prov["sessionId"] == "sess-A"
    assert prov["sourceRef"] and "sess-A" in prov["sourceRef"]
    assert atom["status"] == "live"


# --------------------------------------------------------------------------- #
# A heuristic span goes through the model and lands with CC provenance         #
# --------------------------------------------------------------------------- #


def test_heuristic_span_is_summarized_and_inserted(store):
    model = FakeModelClient()
    embedder = FakeEmbedder()
    para = "I'll pin the HNSW score scale to the flat index so recall stays comparable."
    source = _source("sess-H", [_assistantText(para)])

    result = distill(store, source, model, embedder)

    assert result["inserted"] == 1
    assert result["passthrough"] == 0
    assert model.calls == [para]                 # the model DID run on a heuristic span
    atom = getAtom(store, result["atomIds"][0])
    assert atom["text"] == f"principle: {para}"  # the fake's house-format summary
    prov = atom["provenance"][0]
    assert prov["source"] == SOURCE_CLAUDE_CODE
    assert prov["sessionId"] == "sess-H"
    assert "sess-H#" in prov["sourceRef"]


# --------------------------------------------------------------------------- #
# Step 5: anti-storm -- the same span twice is ONE atom, importance raised      #
# --------------------------------------------------------------------------- #


def test_same_span_twice_one_atom_importance_raised(store):
    model = FakeModelClient()
    embedder = FakeEmbedder()
    para = "I'll switch the KV cache to turbo3 so the 262k context fits in RAM."
    source = _source("sess-S", [_assistantText(para)])

    first = distill(store, source, model, embedder)
    second = distill(store, source, model, embedder)   # exact same span re-tailed

    assert atomCount(store) == 1                        # never a second copy
    assert first["inserted"] == 1
    assert second["inserted"] == 0 and second["bumped"] == 1
    # The re-tail bumps WITHOUT re-summarizing: the model ran once, not twice.
    assert model.calls == [para]
    atom = getAtom(store, first["atomIds"][0])
    assert atom["importance"] > 0.0                     # importance was raised


def test_repetitive_transcript_bounded_atom_count(store):
    model = FakeModelClient()
    embedder = FakeEmbedder()
    # Ten assistant turns, each restating the SAME decision at a different offset.
    # Distinct sourceRefs, so each is summarized; identical summaries, so dedup folds
    # them into one atom. The storm is bounded to a single atom + nine bumps.
    para = "I'll cap the distiller importance bump so a hot span never runs away."
    events = [_assistantText(para) for _ in range(10)]
    source = _source("sess-R", events)

    result = distill(store, source, model, embedder)

    assert atomCount(store) == 1                        # bounded, not ten
    assert result["inserted"] == 1
    assert result["bumped"] == 9
    assert len(model.calls) == 10                       # each distinct span summarized


def test_importance_bump_is_capped(store):
    model = FakeModelClient()
    embedder = FakeEmbedder()
    para = "I'll feed this exact span over and over to test the importance ceiling."
    source = _source("sess-C", [_assistantText(para)])

    for _ in range(100):                                # same span, a hundred times
        distill(store, source, model, embedder)

    assert atomCount(store) == 1
    atom = getAtom(store, store._conn.execute(
        "SELECT id FROM atoms").fetchone()[0])
    assert atom["importance"] == pytest.approx(IMPORTANCE_CAP)   # capped, not infinite
    assert atom["importance"] <= IMPORTANCE_CAP


def test_near_dup_bumps_not_inserts(store):
    model = FakeModelClient()
    embedder = FakeEmbedder()
    # Two DIFFERENT spans (different offsets) whose summaries are identical: the
    # second must dedup against the first by SIMILARITY (not sourceRef) and bump.
    para = "I'll route dedup through the recent window, not the whole store."
    source = _source("sess-N", [_assistantText(para), _assistantText(para)])

    result = distill(store, source, model, embedder)

    assert atomCount(store) == 1
    assert result["inserted"] == 1 and result["bumped"] == 1


# --------------------------------------------------------------------------- #
# The verbatim seam (v3.1): insert every span as-is, no model, no dedup-merge   #
# --------------------------------------------------------------------------- #


def test_verbatim_policy_inserts_every_span_unmodified(store):
    model = FakeModelClient()
    embedder = FakeEmbedder()
    # Two identical spans under a verbatim source. The distill policy would fold them
    # into one; verbatim must keep BOTH, unmodified, and never call the model.
    para = "I'll keep every word of this exactly as written, twice."
    source = _source("sess-V", [_assistantText(para), _assistantText(para)],
                     policy=VERBATIM_POLICY)

    result = distill(store, source, model, embedder)

    assert atomCount(store) == 2                        # repetition preserved, not merged
    assert result["inserted"] == 2 and result["bumped"] == 0
    assert model.calls == []                            # stage 2 never runs on verbatim
    for atom in _atoms(store):
        assert atom["text"] == para                     # inserted raw, not summarized


# --------------------------------------------------------------------------- #
# Sabotage: empty delta, malformed event, model error                          #
# --------------------------------------------------------------------------- #


def test_empty_delta_yields_no_atoms(store):
    model = FakeModelClient()
    embedder = FakeEmbedder()
    source = _source("sess-E", [])                      # a delta with no events
    result = distill(store, source, model, embedder)
    assert atomCount(store) == 0
    assert result == {"inserted": 0, "bumped": 0, "passthrough": 0,
                      "skipped": 0, "atomIds": []}
    # And a source with no deltas at all.
    empty = {"sessionId": "sess-E2", "deltas": []}
    assert distill(store, empty, model, embedder)["inserted"] == 0
    assert atomCount(store) == 0


def test_malformed_events_are_skipped_not_raised(store):
    model = FakeModelClient()
    embedder = FakeEmbedder()
    # A garbage event, a shapeless dict, and a valid explicit emit all in one delta.
    events = [
        "not an event at all",
        42,
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": [{"nonsense": True}, {"type": "text"}]},
        _emitEvent("engram_emit_discovery",
                   {"project": "pensive", "principle": "malformed events must not crash the loop"}),
    ]
    source = _source("sess-M", events)

    result = distill(store, source, model, embedder)   # must not raise

    assert result["passthrough"] == 1                  # the one valid span survived
    assert atomCount(store) == 1


def test_model_error_skips_that_span_processes_the_rest(store):
    embedder = FakeEmbedder()
    bad = "I'll trigger the model error on this exact span."
    good = "I'll let this span summarize cleanly instead."

    class ErroringModelClient(FakeModelClient):
        def summarizeSpan(self, spanText):
            if spanText == bad:
                raise RuntimeError("model is down for this span")
            return super().summarizeSpan(spanText)

    model = ErroringModelClient()
    source = _source("sess-X", [_assistantText(bad), _assistantText(good)])

    result = distill(store, source, model, embedder)   # must not raise

    assert result["skipped"] == 1                       # the erroring span dropped
    assert result["inserted"] == 1                      # the healthy span landed
    assert atomCount(store) == 1


# --------------------------------------------------------------------------- #
# Unit: segment -- explicit emits + heuristic kinds, user turns ignored         #
# --------------------------------------------------------------------------- #


def test_segment_detects_emit_and_heuristic_kinds(store):
    delta = {
        "sessionId": "sess-U",
        "offset": 0,
        "events": [
            {"role": "user", "content": "please fix the recall bug"},   # ignored
            {"role": "assistant", "content": [
                {"type": "text",
                 "text": ("The root cause is a stale index.\n\n"
                          "I'll rebuild it on every emit.")},
                {"type": "tool_use", "name": "engram_emit_failure",
                 "input": {"project": "pensive", "principle": "stale index breaks recall"}},
            ]},
        ],
    }
    spans = segment(delta)
    kinds = [s["kind"] for s in spans]
    assert KIND_EXPLICIT_EMIT in kinds                  # the emit tool_use
    assert "discovery" in kinds                         # "the root cause is"
    assert "decision" in kinds                          # "I'll rebuild"
    # No span came from the user turn.
    assert all("please fix the recall bug" not in s["text"] for s in spans)
    # Offsets are unique per span (stable pointers back into the session).
    assert len({s["offset"] for s in spans}) == len(spans)


def test_segment_empty_and_malformed_return_empty():
    assert segment({}) == []
    assert segment({"events": []}) == []
    assert segment("garbage") == []
    assert segment({"events": "not a list"}) == []


# --------------------------------------------------------------------------- #
# Unit: dedup -- identical text dups, unrelated does not                        #
# --------------------------------------------------------------------------- #


def test_dedup_identical_is_dup_unrelated_is_not(store):
    embedder = FakeEmbedder()
    text = "the KV cache turbo3 quant keeps the full context resident in system RAM"
    atomId = putAtom(store, {"text": text, "kind": "atom", "project": "pensive",
                             "provenance": {"source": SOURCE_CLAUDE_CODE}})

    same = dedup(store, embedder, text)
    assert same["isDup"] is True
    assert same["nearId"] == atomId
    assert same["similarity"] == pytest.approx(1.0)

    other = dedup(store, embedder,
                  "titanium pressure hull rated to four hundred meters depth")
    assert other["isDup"] is False
    assert other["similarity"] < DEDUP_THRESHOLD


def test_dedup_empty_store_and_blank_candidate(store):
    embedder = FakeEmbedder()
    assert dedup(store, embedder, "anything at all")["isDup"] is False   # empty store
    putAtom(store, {"text": "a real atom", "kind": "atom",
                    "provenance": {"source": SOURCE_CLAUDE_CODE}})
    assert dedup(store, embedder, "")["isDup"] is False                  # blank candidate
    assert dedup(store, embedder, "   ")["isDup"] is False


def test_dedup_ignores_superseded_atoms(store):
    embedder = FakeEmbedder()
    from store.store import supersede
    text = "the deep zoom perturbation ceiling is the float64 exponent at 1e300"
    old = putAtom(store, {"text": text, "kind": "atom",
                          "provenance": {"source": SOURCE_CLAUDE_CODE}})
    new = putAtom(store, {"text": "an unrelated replacement fact about fan curves",
                          "kind": "atom", "provenance": {"source": SOURCE_CLAUDE_CODE}})
    supersede(store, old, new, {"source": SOURCE_CLAUDE_CODE})
    # The superseded atom is retired: an identical candidate must NOT dedup into it.
    verdict = dedup(store, embedder, text)
    assert verdict["nearId"] != old
    assert verdict["isDup"] is False
