"""Model-keyed text embedder and the embeddings-table writer.

Embeddings are DERIVED data (cattle, not pet): rebuildable from atom text at any
time, and keyed by ``model_id`` so re-embedding under a new model is additive --
it never overwrites another model's vectors. The blob format is fixed by
``schema.sql``: float32 little-endian bytes, one vector per row, dimension set by
the model (384 for bge-small-en-v1.5).

Runtime note: torch + sentence-transformers are the Phase 0 runtime decision (the
only stack that reaches this box's AMD 7900 XTX via ROCm), so they are sanctioned
here. They are imported lazily inside ``Embedder`` so that importing this module
for the blob helpers alone stays cheap and side-effect free.
"""
import os
import time

import numpy as np

__all__ = ["Embedder", "OnnxEmbedder", "makeEmbedder", "embedMissing",
           "vecToBlob", "blobToVec"]

# Path to an exported ONNX graph of the SAME model. Unset = feature off and this
# module behaves exactly as before. See OnnxEmbedder for why the model id does
# not change when this is set.
_ONNX_PATH_ENV = "PENSIVE_V3_ONNX_MODEL"

# Plan-specified batch size. The spike used 32; the plan's number governs here.
_BATCH_SIZE = 64

# schema.sql fixes the on-disk vector encoding as float32 little-endian. "<f4"
# forces that byte order regardless of host endianness, so the store stays
# portable even though this box happens to be little-endian already.
_BLOB_DTYPE = "<f4"


def vecToBlob(vec):
    """Serialize one vector to the schema's float32 little-endian BLOB bytes."""
    return np.asarray(vec, dtype=_BLOB_DTYPE).tobytes()


def blobToVec(blob):
    """Read one vector back from a float32 little-endian BLOB.

    Returns a 1-D float32 array. ``np.frombuffer`` yields a read-only view over
    the bytes; callers that need to mutate or stack copy first (``FlatIndex.build``
    stacks, which copies). The round-trip ``blobToVec(vecToBlob(v))`` reproduces
    every float32 value exactly -- the cast is float32 -> float32.
    """
    return np.frombuffer(blob, dtype=_BLOB_DTYPE)


# Torch thread count for the SERVE path.
#
# House rule (Gary, 2026-08-13): batch encodes on the GPU, stream encodes on the
# CPU. A stream encode is ONE short sequence, and torch defaults to every core it
# can see -- 12 on this box -- which spends more on thread synchronization than
# the work itself. Measured, single-query p50 / p95:
#
#     threads=1   10.41 / 11.64 ms
#     threads=2    8.74 / 10.62 ms
#     threads=4    5.77 /  6.09 ms   <- chosen
#     threads=8    6.60 /  7.71 ms
#     threads=12   7.79 /  8.68 ms   (the default)
#
# Four is both the fastest and the tightest, and it leaves the remaining cores
# for the rest of the pipeline: an earlier interleaved measurement of the same
# encode read 66 ms because torch and numpy were fighting over all 12.
#
# Env-overridable because the right number is a property of the HOST, not of this
# code, and a box with a different core count will want a different one.
_SERVE_THREADS_ENV = "PENSIVE_V3_TORCH_THREADS"
_DEFAULT_SERVE_THREADS = 4


def _applyServeThreadCap():
    """Cap torch's intra-op threads for the latency-sensitive serve path.

    Returns the applied value, or None when no cap was applied. Setting the env
    var to 0 disables the cap entirely: a BATCH job wanting every core says so
    explicitly rather than fighting a default tuned for single-sequence latency.
    """
    import os
    try:
        want = int(os.environ.get(_SERVE_THREADS_ENV, _DEFAULT_SERVE_THREADS))
    except ValueError:
        want = _DEFAULT_SERVE_THREADS
    if want <= 0:
        return None
    try:
        import torch
        torch.set_num_threads(want)
        return want
    except Exception:
        # A thread cap that cannot be applied is a missed optimization, never a
        # reason to fail an embed.
        return None


class Embedder:
    """Loads one sentence-transformers model and embeds text to unit vectors.

    Construct once and reuse: the model is ~130MB and shares VRAM with other
    processes on this box, so a single resident copy is the rule (tests load it in
    a session-scoped fixture). ``embed`` batches internally and returns
    unit-normalized float32 vectors, so a flat cosine search reduces to a dot
    product.
    """

    def __init__(self, modelId):
        _applyServeThreadCap()
        # Force offline resolution from the HF cache. setdefault (not assignment)
        # so an explicit caller/CI override still wins.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from sentence_transformers import SentenceTransformer

        # ROCm presents as CUDA to torch, so cuda.is_available() is the GPU probe
        # on this box; fall back to CPU where there is no device.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelId = modelId
        self._model = SentenceTransformer(modelId, device=self.device)
        # sentence-transformers 5.x renamed get_sentence_embedding_dimension ->
        # get_embedding_dimension; prefer the new name, keep the old for the
        # >=2.2 floor. Reading the attribute this way avoids the FutureWarning
        # the deprecated call emits on 5.x.
        if hasattr(self._model, "get_embedding_dimension"):
            self.dim = self._model.get_embedding_dimension()
        else:
            self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts):
        """Embed ``texts`` -> list of unit-normalized float32 vectors (dim ``self.dim``).

        Batched at ``_BATCH_SIZE`` in one ``encode`` call (not a per-text loop);
        an empty input yields an empty list without touching the model.
        """
        if not texts:
            return []
        matrix = self._model.encode(
            texts,
            batch_size=_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        # .copy() so each returned vector owns its buffer -- a row view would
        # alias the batch matrix, and a consumer's in-place op would silently
        # mutate its siblings. 384 float32s per copy, negligible.
        return [matrix[i].copy() for i in range(matrix.shape[0])]


class OnnxEmbedder:
    """Same model, same embedding space, same model id -- different runtime.

    WHY THIS EXISTS: the query embed is a STREAM encode on the serve path, and
    it was the largest single component of L2 latency. Running the identical
    bge-small graph under onnxruntime instead of sentence-transformers measured
    3.49/4.55/5.03/5.75 ms at 30/95/154/223-char queries against 6.44/8.05/9.58/
    11.97 for the current path (the four lengths are the real p10/p50/p90/p99 of
    this store's queries).

    WHY ``modelId`` IS UNCHANGED, and why that is not a lie: embeddings are keyed
    ``(atom_id, model_id)``. This runtime produces the same vectors from the same
    weights, verified below, so the 340,722 stored vectors remain valid and no
    re-embed is required. Reporting a different id here would strand every one of
    them and silently trigger a full re-embed on next startup. The id names the
    EMBEDDING SPACE, not the inference library.

    VERIFIED before shipping, on this box, against 375 real texts (250 live atom
    bodies + 125 real recall_log queries):

        paired cosine vs sentence-transformers   min 0.99999940, mean 1.00000000
        samples below 0.9999                     0
        negative control (shuffled pairing)      median 0.6458, p99 0.9233

    The control is there because a cosine near 1.0 proves nothing on its own; it
    is exactly what comparing a thing to itself would produce. The control shows
    the metric discriminates.

    A NOTE ON THAT CONTROL, because it bit me: an earlier verdict gated on
    ``paired.min - control.max``, and control.max is 0.9756. That is not encoder
    error, it is two genuinely near-duplicate atoms in a corpus where the same
    source chunks were re-emitted thousands of times. Gating on the max of a
    control asks a question about the corpus, not about the encoder.

    Pooling is folded INTO the exported graph (CLS token, then L2 normalize) so
    this class cannot drift from the reference by reimplementing it here.
    """

    def __init__(self, modelId, onnxPath, threads=None):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if not os.path.exists(onnxPath):
            raise FileNotFoundError(f"ONNX graph not found: {onnxPath}")

        if threads is None:
            try:
                threads = int(os.environ.get(_SERVE_THREADS_ENV,
                                             _DEFAULT_SERVE_THREADS))
            except ValueError:
                threads = _DEFAULT_SERVE_THREADS
        opts = ort.SessionOptions()
        if threads > 0:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.modelId = modelId
        self.onnxPath = onnxPath
        self.device = "cpu"
        self._tok = AutoTokenizer.from_pretrained(modelId)
        self._sess = ort.InferenceSession(onnxPath, opts,
                                          providers=["CPUExecutionProvider"])
        outs = self._sess.get_outputs()
        if len(outs) != 1:
            raise ValueError(
                f"expected exactly one graph output, got {[o.name for o in outs]}")
        self._outName = outs[0].name
        self._inNames = {i.name for i in self._sess.get_inputs()}
        # dim comes from a real forward pass, not from the declared output shape.
        # onnxruntime happens to resolve this graph's trailing axis to a literal
        # 384, but that is a property of how it was exported: raw graph
        # inspection shows a SYMBOLIC name there, and an export that kept the
        # symbol would make int(shape[-1]) raise at construction. That failure
        # degrades to the torch path, which is safe but silently costs the
        # optimization. One tiny inference at startup makes the number measured
        # rather than declared, which is the rule applied everywhere else here.
        self.dim = len(self.embed(["dimension probe"])[0])

    def embed(self, texts):
        """Embed ``texts`` -> list of unit-normalized float32 vectors.

        Same contract as ``Embedder.embed``: empty in, empty out; each returned
        vector owns its buffer.
        """
        if not texts:
            return []
        out = []
        for i in range(0, len(texts), _BATCH_SIZE):
            enc = self._tok(texts[i:i + _BATCH_SIZE], padding=True,
                            truncation=True, max_length=512, return_tensors="np")
            feed = {"input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64)}
            # bge exports with token_type_ids; a graph without it must not be fed
            # one, so the input set decides rather than an assumption about BERT.
            if "token_type_ids" in self._inNames:
                tt = enc.get("token_type_ids")
                feed["token_type_ids"] = (
                    np.zeros_like(feed["input_ids"]) if tt is None
                    else tt.astype(np.int64))
            matrix = self._sess.run([self._outName], feed)[0]
            matrix = matrix.astype(np.float32, copy=False)
            out.extend(matrix[j].copy() for j in range(matrix.shape[0]))
        return out


def makeEmbedder(modelId):
    """Build the serve-path embedder: ONNX when configured, torch otherwise.

    A construction failure on the ONNX path is NOT fatal. The feature is a
    latency optimization over an already-working encoder, so it degrades to the
    torch path and says so loudly. Silence here would be the bad outcome: an
    operator who set the env var deserves to know it did not take.
    """
    onnxPath = os.environ.get(_ONNX_PATH_ENV)
    if not onnxPath:
        return Embedder(modelId)
    try:
        return OnnxEmbedder(modelId, onnxPath)
    except Exception as exc:  # noqa: BLE001
        print(f"[pensive] ONNX embedder unavailable ({exc}); "
              f"falling back to sentence-transformers", flush=True)
        return Embedder(modelId)


def embedOne(store, embedder, atomId):
    """Embed exactly ``atomId`` if it is live and not already embedded.

    Returns 1 if a row was written, 0 if there was nothing to do.

    WHY THIS EXISTS ALONGSIDE ``embedMissing``. The emit path knows precisely
    which atom it just wrote, and used to call ``embedMissing``, which scans
    every live atom with a correlated NOT EXISTS to rediscover it. Measured on
    this store that scan costs 90ms and returns ZERO rows in the steady state,
    on the event-loop thread, on every emit. Asking "which atoms lack an
    embedding" when the answer is "the one I am holding" is the defect.

    ``embedMissing`` remains correct and stays the right call for startup and
    for any caller that cannot name what changed; this is the scoped form.
    """
    row = store._conn.execute(
        "SELECT a.text FROM atoms a WHERE a.id = ? AND a.status = 'live' "
        "AND NOT EXISTS (SELECT 1 FROM embeddings e "
        "  WHERE e.atom_id = a.id AND e.model_id = ?)",
        (atomId, embedder.modelId),
    ).fetchone()
    if row is None:
        return 0
    vec = embedder.embed([row[0]])[0]
    now = int(time.time())
    try:
        store._conn.execute(
            "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) "
            "VALUES (?, ?, ?, ?)",
            (atomId, embedder.modelId, vecToBlob(vec), now),
        )
        store._conn.commit()
    except Exception:
        store._conn.rollback()
        raise
    return 1


def embedMissing(store, embedder):
    """Embed every LIVE atom lacking an ``embeddings`` row for ``embedder.modelId``.

    Idempotent and additive: it selects only atoms with no row under this model,
    so a second call with nothing new to do embeds nothing and returns 0. Because
    embeddings are keyed by ``(atom_id, model_id)``, embedding under a different
    model adds rows rather than replacing this model's. Non-live atoms
    (superseded/tombstone) are skipped -- their text stays readable but they are
    not recallable, so spending GPU time embedding them would be waste.

    Returns the number of atoms embedded this call. All writes commit as one
    transaction; a failure mid-batch rolls back and leaves the table untouched.
    """
    conn = store._conn
    rows = conn.execute(
        "SELECT a.id, a.text FROM atoms a "
        "WHERE a.status = 'live' AND NOT EXISTS ("
        "  SELECT 1 FROM embeddings e "
        "  WHERE e.atom_id = a.id AND e.model_id = ?"
        ") ORDER BY a.id",
        (embedder.modelId,),
    ).fetchall()
    if not rows:
        return 0

    atomIds = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    vecs = embedder.embed(texts)
    now = int(time.time())
    try:
        conn.executemany(
            "INSERT INTO embeddings(atom_id, model_id, vector, embedded_at) "
            "VALUES (?, ?, ?, ?)",
            [
                (atomId, embedder.modelId, vecToBlob(vec), now)
                for atomId, vec in zip(atomIds, vecs)
            ],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(atomIds)
