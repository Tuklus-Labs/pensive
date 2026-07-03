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
- empty search results return None.
- scores below the floor return None.

Malformed inputs:
- odd ctx/recent-context shapes stay silent instead of crashing.
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
- resource: N/A, fake embedder and tiny in-memory index.
- state: cooldown mutation is asserted directly.

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
"""
import hashlib
import time

import numpy as np
import pytest

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
        {"now": time.time()},
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
