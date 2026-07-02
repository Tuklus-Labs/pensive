"""Cross-encoder rerank of the fused top-50 -- the spec's largest quality lever
above fusion.

Fusion (Task 7) reads ONLY rank positions across signals; it never scores the
query against a document. This stage does exactly that: a BAAI/bge-reranker-base
cross-encoder consumes each ``(query, atom.text)`` pair TOGETHER and emits one
relevance score. That joint scoring is why a paraphrase sharing little vocabulary
with the query -- low on BM25, middling on dense -- can still be pulled to the top
on meaning. Higher score = better, matching every other signal's sign convention.

Cost discipline (the whole point of batching): the pairs are scored in ONE
``predict`` call, one GPU forward pass -- never a per-pair loop. Fifty sequential
GPU round-trips is the ~800 ms failure mode this module exists to avoid; one batch
is the point. Atom texts are fetched in ONE batched ``SELECT ... IN`` (the
fusion.py pattern).

VRAM etiquette: the model is ~1.1GB and shares this box's AMD 7900 XTX with the
embedder and the llama-servers, so exactly one copy is loaded per process, lazily
on the first ``rerank`` call -- never at import.

Runtime note: torch + sentence-transformers are the Phase 0 runtime decision (the
only stack that reaches the card via ROCm), sanctioned here as in the embedder,
and imported lazily inside the loader so importing this module stays cheap and
side-effect free.
"""
import os

__all__ = ["rerank"]

# Phase 0 model. A cross-encoder (not a bi-encoder): it reads the query and a
# document jointly and emits one relevance logit, which is why it reranks better
# than the dense cosine signal but costs a forward pass per pair.
_RERANK_MODEL_ID = "BAAI/bge-reranker-base"

# The contract cap: rerank scores at most the fused top-50. Fusion hands results
# best-first, so ``candidates[:_RERANK_CAP]`` keeps the strongest fused candidates
# and drops the long tail the cross-encoder would rarely promote anyway. This cap
# is part of the interface -- callers rely on it.
_RERANK_CAP = 50

# Explicit truncation window. Atom texts can exceed the model's 512-token context
# (Task 18 posts conversation tails into recall as raw queries), so truncation
# must be a DELIBERATE max_length, not an accidental tokenizer default that could
# silently change with the model. 512 is bge-reranker-base's positional limit.
_MAX_LENGTH = 512

# Lazy module-level singleton, mirroring the embedder: one copy per process,
# loaded on first rerank(), never at import.
_reranker = None


def _getReranker():
    """Load (once) and return the process-wide cross-encoder singleton."""
    global _reranker
    if _reranker is None:
        # Force offline resolution from the HF cache; setdefault (not assignment)
        # so an explicit caller/CI override still wins. The progress bar and the
        # load-time advisory notes are silenced so a daemon log -- and the test
        # output -- stays clean: this is a fixed, known-good model, not one whose
        # load chatter we need to see.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        import torch
        from sentence_transformers import CrossEncoder
        from transformers.utils import logging as hfLogging

        hfLogging.set_verbosity_error()
        # ROCm presents as CUDA to torch, so cuda.is_available() is the GPU probe
        # on this box; fall back to CPU where there is no device.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker = CrossEncoder(
            _RERANK_MODEL_ID, device=device, max_length=_MAX_LENGTH
        )
    return _reranker


def rerank(query, candidates, store):
    """Rescore fused candidates against ``query`` -> ``[(atomId, score)]`` best-first.

    ``candidates`` is the fused ``[(atomId, fusedScore)]`` list, best-first (the
    Task 7 convention). Only the FIRST ``_RERANK_CAP`` (50) are rescored -- the cap
    is part of the contract -- and the fused scores are DISCARDED: the
    cross-encoder assigns its own relevance score to each ``(query, atom.text)``
    pair. The result is sorted best-first by that cross-encoder score (higher =
    better); ties keep the incoming fused order via a stable sort, so the output is
    deterministic.

    Empty ``candidates`` -> ``[]`` WITHOUT loading the model (no reason to page in
    1.1GB to rerank nothing). Atom texts are fetched in ONE batched
    ``SELECT ... IN``; a candidate id absent from the store means the index and
    store have desynced -- the decades rule says surface that loudly, so it raises
    ``ValueError`` naming the id rather than dropping it (parity with
    :func:`recall.fusion.applyPriors`). The desync check runs BEFORE the model is
    touched, so a broken candidate set never wastes a forward pass. All surviving
    pairs are scored in ONE ``predict`` call.
    """
    if not candidates:
        return []

    capped = candidates[:_RERANK_CAP]
    atomIds = [atomId for atomId, _ in capped]
    placeholders = ",".join("?" for _ in atomIds)
    rows = store._conn.execute(
        f"SELECT id, text FROM atoms WHERE id IN ({placeholders})",
        tuple(atomIds),
    ).fetchall()
    textById = {r[0]: r[1] for r in rows}

    pairs = []
    for atomId in atomIds:
        if atomId not in textById:
            raise ValueError(
                f"rerank candidate {atomId!r} is absent from the store "
                "(index/store desync)"
            )
        pairs.append((query, textById[atomId]))

    model = _getReranker()
    # ONE batched forward pass: batch_size == the pair count (<=50) so every pair
    # goes through the GPU together. A per-pair loop here would be up to 50
    # sequential GPU round-trips (~800 ms); this single call is the point of the
    # module.
    scores = model.predict(
        pairs,
        batch_size=len(pairs),
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    scored = [(atomId, float(score)) for atomId, score in zip(atomIds, scores)]
    # Stable sort: descending score, ties fall back to the incoming fused order.
    scored.sort(key=lambda kv: -kv[1])
    return scored
