"""Stage-2 dedup: does this candidate atom already exist, near-verbatim, recently?

The distiller's anti-storm gate. Before a freshly-summarized span is inserted, it
is checked against RECENT atoms by embedding similarity: if it is near-identical to
one already in the store, it is a duplicate and the distiller bumps that atom's
importance instead of inserting a second copy. The same reasoning recurs across a
long session (and across re-tails of the same transcript), so without this the
store floods with paraphrases of one idea.

Two design choices, both named constants below:

- **A bounded recent window, not the whole store.** "Duplicate" is a recency
  phenomenon -- the agent restating something it just said. Scanning the last
  :data:`RECENT_WINDOW` live atoms bounds the cost on a decades-scale store while
  catching the storms that actually happen. An old atom outside the window is not a
  dedup target; if the same idea resurfaces months later that is arguably a fresh
  reinforcement worth its own atom, not a merge.

- **A high similarity threshold.** :data:`DEDUP_THRESHOLD` is set to catch
  *near-verbatim* restatements (the model normalizes recurring reasoning to nearly
  the same house-format text), NOT loosely-related atoms. Dedup must never merge two
  genuinely different memories; a false merge silently loses a memory, the worst
  failure in a decades store. Better to keep a borderline near-dup than to eat a
  real one, so the bar is deliberately high.

"Same-provenance suppression" (the same session restating a point) falls out of
this naturally: an identical restatement embeds to the same vector, so its
similarity is ~1.0 and it trips the threshold regardless of which session it came
from. The signature is fixed to ``dedup(store, embedder, candidateText)``, so the
gate keys on the candidate's TEXT; identical text is the same-provenance case at
its limit.

Similarity is cosine. The project embedder returns unit vectors (cosine == dot),
but this computes the full normalized cosine anyway so a non-normalizing embedder
(or the deterministic test fake) still gives a correct score.
"""
import numpy as np

__all__ = ["dedup", "DEDUP_THRESHOLD", "RECENT_WINDOW", "cosine"]

# Cosine-similarity bar for calling two atoms the same memory. High on purpose:
# near-verbatim only. A false merge loses a memory; a missed near-dup only costs a
# redundant atom, so we err toward keeping. Calibrated 2026-07-03 against local
# BAAI/bge-small-en-v1.5 with 10 near-verbatim house-atom restatement pairs
# (mean 0.9949, min 0.9863) and 10 distinct-but-related same-subsystem pairs
# (mean 0.6974, max 0.8069). The recall harness owns final tuning.
DEDUP_THRESHOLD = 0.96

# How many recent live atoms a candidate is checked against. Bounds the scan on a
# decades store; dedup is a recency phenomenon (see module docstring), so the
# recent tail is the right (and sufficient) place to look.
RECENT_WINDOW = 200


def cosine(a, b):
    """Cosine similarity of two 1-D float vectors, 0.0 if either is a zero vector.

    Guards the zero-vector case (an empty-text embedding) so a blank candidate
    scores 0 similarity rather than dividing by zero."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _recentLiveAtoms(store):
    """The most recent :data:`RECENT_WINDOW` LIVE atoms as ``[(id, text)]``,
    newest first. Only live atoms are dedup targets -- a superseded/tombstoned
    atom is retired, and bumping or merging into it would resurrect a dead memory.
    """
    rows = store._conn.execute(
        "SELECT id, text FROM atoms WHERE status = 'live' "
        "ORDER BY created_at DESC, id DESC LIMIT ?",
        (RECENT_WINDOW,),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def dedup(store, embedder, candidateText):
    """Is ``candidateText`` a near-duplicate of a recent live atom?

    Returns ``{"isDup": bool, "nearId": <atom id>|None, "similarity": <float>}``.
    ``isDup`` is True when the max cosine similarity between ``candidateText`` and
    any atom in the recent window reaches :data:`DEDUP_THRESHOLD`; ``nearId`` is the
    atom it matched (the best match), and the distiller bumps that atom's importance
    instead of inserting. A blank candidate, an empty store, or no match above the
    threshold all yield ``{"isDup": False, "nearId": None, "similarity": <max>}``.

    One batched ``embedder.embed`` call covers the candidate and the whole window,
    so a dedup check is a single model round-trip, not one per recent atom.
    """
    if not candidateText or not candidateText.strip():
        return {"isDup": False, "nearId": None, "similarity": 0.0}

    recent = _recentLiveAtoms(store)
    if not recent:
        return {"isDup": False, "nearId": None, "similarity": 0.0}

    ids = [rid for rid, _ in recent]
    texts = [txt for _, txt in recent]
    # Candidate first, then the window -- one batched embed call for all of them.
    vecs = embedder.embed([candidateText] + texts)
    candidateVec = vecs[0]
    windowVecs = vecs[1:]

    bestId = None
    bestSim = 0.0
    for atomId, vec in zip(ids, windowVecs):
        sim = cosine(candidateVec, vec)
        if sim > bestSim:
            bestSim = sim
            bestId = atomId

    isDup = bestSim >= DEDUP_THRESHOLD
    return {
        "isDup": isDup,
        "nearId": bestId if isDup else None,
        "similarity": bestSim,
    }
