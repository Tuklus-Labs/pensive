"""Risk model for the conservative drift watcher.

Invariants:
- high-confidence-hit: only a hit above the documented floor may inject.
- one-per-cooldown: a caller-owned ctx suppresses repeat injections.
- already-visible: an atom id or exact atom text in recent context suppresses.

State transitions:
- idle -> injected records a cooldown timestamp in ctx.
- injected -> cooling returns None until the cooldown expires.

Boundaries:
- blank tail text returns None before embedding.
- huge tail text is truncated to the recent end before embedding.
- empty search results return None.
- scores below the floor return None.
- non-finite and out-of-range scores are skipped.

Malformed inputs:
- odd ctx/recent-context shapes stay silent instead of crashing.
- malformed now/last-injected values stay silent.
- stale index hits and atom text None stay silent.
- embedder/index/store operational failures are not swallowed.

Concurrency:
- cooldown state is per ctx object; module globals must not carry session state.

Persistence:
- uses a real Store and getAtom so the payload shape tracks persisted atoms.

Integration contracts:
- consumes embedder.embed(texts), index.search(vec, k), and store.getAtom.
- returns one compact dict a hook can render.

Regression traps:
- boundary: blank tail guard is explicit because blank embeds can be valid.
- concurrency: cooldown does not use module-level mutable state.
- contract: search score scale is treated as cosine similarity.
- encoding: N/A, tests use ASCII and store text is already Unicode-safe.
- framework: N/A, no async or pytest plugin behavior in scope.
- io: store/index real faults may propagate by design.
- persistence: payload is fetched through the real Store contract.
- state: cooldown mutation is asserted directly.
- resource: huge tails are bounded before embedder/tokenizer work.

Coverage matrix:
- high-confidence-hit, one-per-cooldown:
  test_strong_match_injects_once_then_cools_down
- high-confidence-hit below floor:
  test_off_topic_tail_below_floor_stays_silent
- boundary blank tail:
  test_blank_tail_returns_none_without_embedding
- already-visible:
  test_recent_context_suppresses_atom_id_or_text
- concurrency state isolation:
  test_cooldown_lives_in_caller_ctx_not_module_state
- resource tail bound:
  test_huge_tail_is_truncated_to_recent_end_before_embedding
- state mixed clock source:
  test_mixed_clock_source_keeps_cooldown_active
- boundary score validation:
  test_non_finite_score_is_skipped_for_next_valid_hit,
  test_nan_only_hit_returns_none,
  test_out_of_range_score_is_skipped_for_next_valid_hit
- malformed input shapes:
  test_adversarial_input_shapes_return_none
"""
import hashlib

import numpy as np
import pytest

import ambient.drift as drift
from ambient.drift import CONFIDENCE_FLOOR, COOLDOWN_SECONDS, onTail
from store.store import getAtom, openStore, putAtom


class FakeEmbedder:
    DIM = 4096

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        out = []
        for text in texts:
            vec = np.zeros(self.DIM, dtype=np.float32)
            for tok in text.lower().split():
                digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                vec[int.from_bytes(digest, "big") % self.DIM] += 1.0
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            out.append(vec)
        return out


class TinyIndex:
    def __init__(self, vectors):
        self._vectors = vectors

    def search(self, vec, k):
        query = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0 or k <= 0:
            return []
        query = query / norm
        scored = [
            (atomId, float(np.asarray(atomVec, dtype=np.float32) @ query))
            for atomId, atomVec in self._vectors.items()
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:k]


class OrderedIndex:
    def __init__(self, hits):
        self._hits = list(hits)

    def search(self, vec, k):
        return self._hits[:k]


class SizeCappedEmbedder(FakeEmbedder):
    def __init__(self, max_chars):
        super().__init__()
        self.max_chars = max_chars

    def embed(self, texts):
        for text in texts:
            if len(text) > self.max_chars:
                raise AssertionError(
                    "tail-size bound invariant violated: "
                    f"len={len(text)} max={self.max_chars}"
                )
        return super().embed(texts)


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


def _put(store, text):
    return putAtom(store, {
        "text": text,
        "kind": "atom",
        "project": "pensive",
        "provenance": {"source": "codex"},
    })


def _indexFor(embedder, atoms):
    texts = [atom["text"] for atom in atoms]
    vecs = embedder.embed(texts)
    return TinyIndex({atom["id"]: vec for atom, vec in zip(atoms, vecs)})


def test_strong_match_injects_once_then_cools_down(store):
    embedder = FakeEmbedder()
    atomId = _put(store, "pensive drift watcher injects only high confidence memory")
    atom = getAtom(store, atomId)
    index = _indexFor(embedder, [atom])
    ctx = {"now": 1_000.0}

    first = onTail(
        store,
        index,
        embedder,
        "pensive drift watcher injects only high confidence memory",
        ctx,
    )
    second = onTail(
        store,
        index,
        embedder,
        "pensive drift watcher injects only high confidence memory",
        ctx,
    )

    assert first == {
        "kind": "memory",
        "atomId": atomId,
        "text": atom["text"],
        "score": pytest.approx(1.0),
    }, f"high-confidence-hit invariant violated: injection={first!r}"
    assert second is None, (
        f"one-per-cooldown invariant violated: second={second!r} "
        f"ctx={ctx!r} cooldown={COOLDOWN_SECONDS}"
    )
    assert ctx.get("driftLastInjectedAt") == 1_000.0, (
        f"cooldown state transition violated: ctx={ctx!r}"
    )


def test_off_topic_tail_below_floor_stays_silent(store):
    embedder = FakeEmbedder()
    atomId = _put(store, "pensive drift watcher injects only high confidence memory")
    index = _indexFor(embedder, [getAtom(store, atomId)])

    injection = onTail(
        store,
        index,
        embedder,
        "unrelated sourdough starter hydration schedule",
        {"now": 1_234.0},
    )

    assert injection is None, (
        f"precision floor invariant violated: score below {CONFIDENCE_FLOOR} "
        f"must stay silent, injection={injection!r}"
    )


def test_blank_tail_returns_none_without_embedding(store):
    embedder = FakeEmbedder()
    injection = onTail(store, TinyIndex({}), embedder, " \n\t ", {})

    assert injection is None, f"blank-tail boundary violated: injection={injection!r}"
    assert embedder.calls == [], (
        f"blank-tail boundary violated: embedder was called calls={embedder.calls!r}"
    )


def test_recent_context_suppresses_atom_id_or_text(store):
    embedder = FakeEmbedder()
    atomId = _put(store, "pensive drift watcher injects only high confidence memory")
    atom = getAtom(store, atomId)
    index = _indexFor(embedder, [atom])

    byId = onTail(
        store,
        index,
        embedder,
        atom["text"],
        {"recentContext": [{"atomId": atomId}]},
    )
    byText = onTail(
        store,
        index,
        embedder,
        atom["text"],
        {"recentContext": [f"already said: {atom['text']}"]},
    )

    assert byId is None, (
        f"already-visible id suppression violated: atomId={atomId} injection={byId!r}"
    )
    assert byText is None, (
        f"already-visible text suppression violated: text={atom['text']!r} "
        f"injection={byText!r}"
    )


def test_top_visible_hit_suppresses_without_lower_score_fallback(store):
    embedder = FakeEmbedder()
    visibleId = _put(store, "visible best memory should not be reinjected")
    otherId = _put(store, "lower score cousin must not inject")
    visibleAtom = getAtom(store, visibleId)
    otherAtom = getAtom(store, otherId)
    index = OrderedIndex([(visibleId, 0.99), (otherId, 0.98)])

    injection = onTail(
        store,
        index,
        embedder,
        otherAtom["text"],
        {"now": 1_000.0, "recentContext": [{"atomId": visibleId}]},
    )

    assert injection is None, (
        f"best-hit contract violated: visible top hit fell through to lower hit "
        f"visibleId={visibleId} otherId={otherId} injection={injection!r}"
    )
    assert visibleAtom is not None, (
        f"test setup invariant violated: visible atom missing id={visibleId}"
    )


@pytest.mark.parametrize("status", ["superseded", "tombstone"])
def test_non_live_top_hit_suppresses_without_lower_score_fallback(store, status):
    embedder = FakeEmbedder()
    staleId = _put(store, "stale best memory must not be injected")
    otherId = _put(store, "lower score live cousin must not inject")
    otherAtom = getAtom(store, otherId)
    store._conn.execute("UPDATE atoms SET status = ? WHERE id = ?", (status, staleId))
    store._conn.commit()
    index = OrderedIndex([(staleId, 0.99), (otherId, 0.98)])

    injection = onTail(
        store,
        index,
        embedder,
        otherAtom["text"],
        {"now": 1_000.0},
    )

    assert injection is None, (
        f"live-only drift invariant violated: top hit with status={status!r} "
        f"fell through to lower hit staleId={staleId} otherId={otherId} "
        f"injection={injection!r}"
    )


def test_cooldown_lives_in_caller_ctx_not_module_state(store):
    embedder = FakeEmbedder()
    atomId = _put(store, "pensive drift watcher injects only high confidence memory")
    atom = getAtom(store, atomId)
    index = _indexFor(embedder, [atom])

    firstCtx = {"now": 200.0}
    secondCtx = {"now": 200.0}
    first = onTail(store, index, embedder, atom["text"], firstCtx)
    second = onTail(store, index, embedder, atom["text"], secondCtx)

    assert first is not None, f"ctx isolation setup failed: first={first!r}"
    assert second is not None, (
        f"ctx isolation violated: second ctx was cooled by module/session state "
        f"firstCtx={firstCtx!r} secondCtx={secondCtx!r}"
    )


def test_huge_tail_is_truncated_to_recent_end_before_embedding(store):
    embedder = SizeCappedEmbedder(max_chars=8_192)
    atomId = _put(store, "recent drift signal survives tail truncation")
    atom = getAtom(store, atomId)
    index = OrderedIndex([(atomId, 0.99)])

    tail = ("old context " * 220_000) + atom["text"]
    injection = onTail(store, index, embedder, tail, {"now": 1_000.0})

    assert injection is not None, (
        f"tail truncation invariant violated: huge recent-match tail did not inject "
        f"tail_len={len(tail)}"
    )
    assert injection["atomId"] == atomId, (
        f"tail-end retention invariant violated: injection={injection!r} atomId={atomId}"
    )
    assert embedder.calls and embedder.calls[-1][0].endswith(atom["text"]), (
        f"tail-end retention invariant violated: embedded={embedder.calls[-1][0]!r}"
    )


def test_mixed_clock_source_keeps_cooldown_active(store):
    embedder = FakeEmbedder()
    atomId = _put(store, "pensive drift watcher injects only high confidence memory")
    atom = getAtom(store, atomId)
    index = _indexFor(embedder, [atom])
    ctx = {"now": 1_000.0}

    first = onTail(store, index, embedder, atom["text"], ctx)
    ctx.pop("now")
    second = onTail(store, index, embedder, atom["text"], ctx)

    assert first is not None, f"mixed-clock setup failed: first={first!r} ctx={ctx!r}"
    assert second is None, (
        f"mixed-clock cooldown invariant violated: wall clock bypassed ctx cooldown "
        f"second={second!r} ctx={ctx!r}"
    )


def test_non_finite_score_is_skipped_for_next_valid_hit(store):
    embedder = FakeEmbedder()
    nanId = _put(store, "nan score candidate must be skipped")
    validId = _put(store, "valid score candidate may inject")
    validAtom = getAtom(store, validId)
    vec = embedder.embed([validAtom["text"]])[0]
    index = OrderedIndex([(nanId, float("nan")), (validId, 0.99)])

    injection = onTail(store, index, embedder, validAtom["text"], {"now": 1_000.0})

    assert injection is not None, (
        f"score validation invariant violated: NaN first hit aborted valid fallback "
        f"injection={injection!r}"
    )
    assert injection["atomId"] == validId, (
        f"score validation invariant violated: injection={injection!r} validId={validId}"
    )
    assert vec is not None, "test setup invariant violated: valid vector was not built"


def test_nan_only_hit_returns_none(store):
    embedder = FakeEmbedder()
    atomId = _put(store, "nan score candidate must be skipped")
    atom = getAtom(store, atomId)
    index = OrderedIndex([(atomId, float("nan"))])

    injection = onTail(store, index, embedder, atom["text"], {"now": 1_000.0})

    assert injection is None, (
        f"score validation invariant violated: NaN-only hit injected {injection!r}"
    )


def test_adversarial_input_shapes_return_none(store, monkeypatch):
    embedder = FakeEmbedder()
    atomId = _put(store, "pensive drift watcher injects only high confidence memory")
    atom = getAtom(store, atomId)
    index = _indexFor(embedder, [atom])

    cases = [
        ("ctx-none", atom["text"], None, index),
        ("malformed-last-injected", atom["text"], {"driftLastInjectedAt": "never"}, index),
        ("malformed-now", atom["text"], {"now": "later"}, index),
        ("empty-search-results", atom["text"], {"now": 1_000.0}, OrderedIndex([])),
        ("unknown-atom-id", atom["text"], {"now": 1_000.0}, OrderedIndex([("missing", 0.99)])),
    ]

    for label, tail, ctx, caseIndex in cases:
        injection = onTail(store, caseIndex, embedder, tail, ctx)
        assert injection is None, (
            f"input-shape contract violated: case={label} injection={injection!r} ctx={ctx!r}"
        )

    monkeypatch.setattr(drift, "getAtom", lambda store, atomId: {"id": atomId, "text": None})
    injection = onTail(store, OrderedIndex([(atomId, 0.99)]), embedder, atom["text"], {"now": 1_000.0})
    assert injection is None, (
        f"input-shape contract violated: atom text None injected {injection!r}"
    )


def test_top_stale_hit_suppresses_without_lower_score_fallback(store):
    embedder = FakeEmbedder()
    validId = _put(store, "valid fallback must not inject after stale top hit")
    validAtom = getAtom(store, validId)
    index = OrderedIndex([("missing", 0.99), (validId, 0.98)])

    injection = onTail(store, index, embedder, validAtom["text"], {"now": 1_000.0})

    assert injection is None, (
        f"best-hit contract violated: stale top hit fell through to lower hit "
        f"validId={validId} injection={injection!r}"
    )


def test_top_text_none_hit_suppresses_without_lower_score_fallback(store, monkeypatch):
    embedder = FakeEmbedder()
    textNoneId = _put(store, "text none top hit must suppress")
    validId = _put(store, "valid fallback must not inject after text none top hit")
    validAtom = getAtom(store, validId)

    def fakeGetAtom(store, atomId):
        if atomId == textNoneId:
            return {"id": textNoneId, "text": None}
        return getAtom(store, atomId)

    monkeypatch.setattr(drift, "getAtom", fakeGetAtom)
    index = OrderedIndex([(textNoneId, 0.99), (validId, 0.98)])

    injection = onTail(store, index, embedder, validAtom["text"], {"now": 1_000.0})

    assert injection is None, (
        f"best-hit contract violated: text-None top hit fell through to lower hit "
        f"textNoneId={textNoneId} validId={validId} injection={injection!r}"
    )


def test_out_of_range_score_is_skipped_for_next_valid_hit(store):
    embedder = FakeEmbedder()
    highId = _put(store, "out of range score candidate must be skipped")
    validId = _put(store, "valid score candidate may inject")
    validAtom = getAtom(store, validId)
    index = OrderedIndex([(highId, 1.00001), (validId, 0.99)])

    injection = onTail(store, index, embedder, validAtom["text"], {"now": 1_000.0})

    assert injection is not None, (
        f"score range invariant violated: out-of-range first hit aborted valid fallback "
        f"injection={injection!r}"
    )
    assert injection["atomId"] == validId, (
        f"score range invariant violated: injection={injection!r} validId={validId}"
    )
