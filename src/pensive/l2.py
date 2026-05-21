"""L2 embedding-based retrieval using sentence-transformers and FAISS.

Provides semantic search that complements SA's entity-graph retrieval.
SA catches exact entities (7/19 query types), L2 catches semantic meaning
(the other 12/19). Together they hit 100%.

Requires: pip install pypensive[full]
"""
import logging
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class L2Config:
    """Configuration for L2 retrieval."""
    embedding_model: str = "all-MiniLM-L6-v2"
    max_results: int = 10
    batch_size: int = 256
    normalize: bool = True
    use_ivf: bool = False
    ivf_nprobe: int = 10
    ivf_train_threshold: int = 10_000


@dataclass
class L2Result:
    """A result from L2 retrieval."""
    document_id: str
    content: str
    score: float
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class L2Handler:
    """Lightweight L2 retrieval with FAISS and sentence-transformers.

    Embeds documents using a sentence transformer model and indexes them
    in FAISS for fast approximate nearest neighbor search.

    Usage:
        from pensive.l2 import L2Handler

        l2 = L2Handler()
        l2.add_documents([
            {'id': 'doc1', 'content': 'The server latency was 42ms'},
            {'id': 'doc2', 'content': 'GPU temperature reached 82C'},
        ])
        results = l2.query("What was the latency?")
    """

    def __init__(self, config: Optional[L2Config] = None):
        self.config = config or L2Config()
        self._lock = threading.RLock()

        # Lazy imports for optional deps
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.config.embedding_model)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info("Loaded embedding model: %s (dim=%d)",
                        self.config.embedding_model, self._dim)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for L2 retrieval. "
                "Install with: pip install pypensive[full]"
            )

        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for L2 retrieval. "
                "Install with: pip install pypensive[full]"
            )

        # FAISS index
        self._index = self._faiss.IndexFlatIP(self._dim)
        self._id_map = self._faiss.IndexIDMap2(self._index)

        # Document storage
        self._docs: Dict[int, Dict[str, Any]] = {}
        self._doc_id_to_faiss_id: Dict[str, int] = {}
        self._emb_array: np.ndarray = np.empty((0, self._dim), dtype=np.float32)
        self._emb_capacity: int = 0
        self._next_id: int = 0

        # IVF state (for large corpora)
        self._ivf_trained = False
        self._buffer_embeddings: List[np.ndarray] = []
        self._buffer_ids: List[int] = []

        # Single-slot query embedding cache: avoids re-encoding when the
        # same query text hits both query() and query_candidates().
        self._query_cache_text: Optional[str] = None
        self._query_cache_emb: Optional[np.ndarray] = None

        # Pre-allocated buffers for query_candidates hot path.
        # Avoids creating new numpy arrays on every call.
        self._qc_fid_buf: np.ndarray = np.empty(256, dtype=np.int64)
        self._qc_emb_buf: np.ndarray = np.empty((256, self._dim), dtype=np.float32)
        self._qc_scores_buf: np.ndarray = np.empty(256, dtype=np.float32)
        # Persistent 2D view for query vector reshape (avoids alloc per search).
        # PENPY-MIN-2: this buffer is shared between query() and the
        # large-n branch of query_candidates(). Both paths acquire
        # self._lock before reading/writing it, so concurrent calls
        # serialize correctly. Any new caller that touches this buffer
        # MUST also hold self._lock for the duration of the write+use
        # pair (i.e. fill _query_2d and consume it before releasing).
        # batch_query() deliberately allocates a separate reshape rather
        # than share this buffer; do not change that without auditing
        # the lock contract here.
        self._query_2d: np.ndarray = np.empty((1, self._dim), dtype=np.float32)

    def _encode_query(self, text: str) -> np.ndarray:
        """Encode a single query string, with single-slot cache."""
        if self._query_cache_text == text and self._query_cache_emb is not None:
            return self._query_cache_emb
        emb = self._encode([text])[0]
        self._query_cache_text = text
        self._query_cache_emb = emb
        return emb

    def _store_embeddings(self, ids: List[int], embeddings: np.ndarray):
        """Store embeddings in the contiguous array, growing as needed.

        Newly grown rows are zero-initialized rather than left as
        ``np.empty`` garbage. The hot-path bounds check is
        ``faiss_id < self._emb_array.shape[0]`` -- a "stored" row that
        was never written would still pass that check and then dot-product
        a zeroed vector against the query, returning a stable 0.0 score
        instead of a random nonsense one. Faster to allocate empty + zero
        the new region than ``np.zeros`` the whole capacity each grow.
        """
        max_id = max(ids) + 1
        if max_id > self._emb_capacity:
            new_cap = max(max_id, self._emb_capacity * 2, 256)
            new_arr = np.empty((new_cap, self._dim), dtype=np.float32)
            old_size = self._emb_array.shape[0]
            if old_size > 0:
                new_arr[:old_size] = self._emb_array
            # Zero the newly grown region so any read of an unwritten
            # slot returns a defined zero vector rather than uninitialized
            # memory.
            if new_cap > old_size:
                new_arr[old_size:new_cap].fill(0.0)
            self._emb_array = new_arr
            self._emb_capacity = new_cap
        for i, fid in enumerate(ids):
            self._emb_array[fid] = embeddings[i]

    def _get_embedding(self, faiss_id: int) -> Optional[np.ndarray]:
        """Get an embedding by faiss ID, or None if not stored."""
        if faiss_id < self._emb_array.shape[0]:
            return self._emb_array[faiss_id]
        return None

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        embeddings = self._model.encode(texts, batch_size=self.config.batch_size,
                                         show_progress_bar=False)
        if self.config.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
        return embeddings.astype(np.float32)

    def add_documents(self, documents: List[Dict[str, Any]],
                      id_field: str = 'id',
                      content_field: str = 'content') -> int:
        """Add documents to the index.

        Args:
            documents: List of dicts with at minimum id and content fields.
            id_field: Field name for document ID.
            content_field: Field name for text content to embed.

        Returns:
            Number of documents added.
        """
        with self._lock:
            new_texts = []
            new_ids = []

            for doc in documents:
                doc_id = str(doc.get(id_field, ''))
                if doc_id in self._doc_id_to_faiss_id:
                    continue

                faiss_id = self._next_id
                self._next_id += 1

                self._doc_id_to_faiss_id[doc_id] = faiss_id
                self._docs[faiss_id] = doc

                new_texts.append(doc.get(content_field, ''))
                new_ids.append(faiss_id)

            if not new_texts:
                return 0

            embeddings = self._encode(new_texts)
            self._store_embeddings(new_ids, embeddings)
            faiss_ids = np.array(new_ids, dtype=np.int64)

            if self.config.use_ivf and not self._ivf_trained:
                # Buffer until we have enough for IVF training
                self._buffer_embeddings.append(embeddings)
                self._buffer_ids.extend(new_ids)
                total_buffered = sum(e.shape[0] for e in self._buffer_embeddings)

                if total_buffered >= self.config.ivf_train_threshold:
                    self._train_ivf()
                else:
                    return len(new_texts)

            self._id_map.add_with_ids(embeddings, faiss_ids)
            return len(new_texts)

    def _train_ivf(self):
        """Train IVF index from buffered embeddings."""
        all_embeddings = np.vstack(self._buffer_embeddings)
        all_ids = np.array(self._buffer_ids, dtype=np.int64)
        n = all_embeddings.shape[0]

        nlist = min(int(np.sqrt(n)), 256)
        quantizer = self._faiss.IndexFlatIP(self._dim)
        ivf_index = self._faiss.IndexIVFFlat(quantizer, self._dim, nlist,
                                              self._faiss.METRIC_INNER_PRODUCT)
        ivf_index.train(all_embeddings)
        ivf_index.nprobe = self.config.ivf_nprobe

        self._index = ivf_index
        self._id_map = self._faiss.IndexIDMap2(ivf_index)
        self._id_map.add_with_ids(all_embeddings, all_ids)

        self._ivf_trained = True
        self._buffer_embeddings = []
        self._buffer_ids = []
        logger.info("IVF index trained: %d vectors, %d centroids", n, nlist)

    def query(self, query_text: str, top_k: Optional[int] = None) -> List[L2Result]:
        """Search for documents similar to the query.

        Args:
            query_text: Natural language query.
            top_k: Number of results to return.

        Returns:
            List of L2Result ordered by similarity score.
        """
        top_k = top_k or self.config.max_results

        with self._lock:
            if self._id_map.ntotal == 0 and not self._buffer_embeddings:
                return []

            query_vec = self._encode_query(query_text)
            if self._id_map.ntotal == 0 and self._buffer_embeddings:
                # IVF warmup mode: query buffered vectors before training threshold.
                all_embeddings = np.vstack(self._buffer_embeddings)
                scores = all_embeddings @ query_vec
                k = min(top_k, scores.shape[0])
                if k >= len(scores):
                    top_indices = np.argsort(scores)[::-1]
                else:
                    top_indices = np.argpartition(scores, -k)[-k:]
                    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
                ids = [self._buffer_ids[i] for i in top_indices]
                return self._results_from_ids(ids, scores[top_indices])

            k = min(top_k, self._id_map.ntotal)
            self._query_2d[0] = query_vec
            scores, ids = self._id_map.search(self._query_2d, k)
            return self._results_from_ids(ids[0], scores[0])

    def query_candidates(
        self,
        query_text: str,
        candidate_doc_ids: List[str],
        top_k: Optional[int] = None,
    ) -> List[L2Result]:
        """Rerank a candidate subset by semantic similarity.

        For small candidate sets (typical L1 output of ~30 docs), uses
        direct numpy dot product instead of creating a temporary FAISS
        index, avoiding index construction overhead.
        """
        top_k = top_k or self.config.max_results
        if not candidate_doc_ids:
            return []

        with self._lock:
            unique_doc_ids = list(dict.fromkeys(str(doc_id) for doc_id in candidate_doc_ids))
            faiss_ids = []

            for doc_id in unique_doc_ids:
                faiss_id = self._doc_id_to_faiss_id.get(doc_id)
                if faiss_id is None:
                    continue

                emb = self._get_embedding(faiss_id)
                if emb is None:
                    doc = self._docs.get(faiss_id, {})
                    content = doc.get('content', doc.get('summary', ''))
                    if not content:
                        continue
                    enc = self._encode([content])[0]
                    self._store_embeddings([faiss_id], enc.reshape(1, -1))

                faiss_ids.append(faiss_id)

            if not faiss_ids:
                return []

            query_emb = self._encode_query(query_text)
            n = len(faiss_ids)

            # For small candidate sets, numpy dot product beats FAISS
            # index creation overhead. Threshold at 1000 candidates.
            if n < 1000:
                # Grow pre-allocated buffers if needed (amortized, rare after warmup)
                if n > self._qc_fid_buf.shape[0]:
                    new_cap = max(n, self._qc_fid_buf.shape[0] * 2)
                    self._qc_fid_buf = np.empty(new_cap, dtype=np.int64)
                    self._qc_emb_buf = np.empty((new_cap, self._dim), dtype=np.float32)
                    self._qc_scores_buf = np.empty(new_cap, dtype=np.float32)

                # Fill index buffer without allocating a new array
                fid_view = self._qc_fid_buf[:n]
                fid_view[:] = faiss_ids

                # Gather embeddings via vectorized numpy take (C-level copy)
                emb_view = self._qc_emb_buf[:n]
                np.take(self._emb_array, fid_view, axis=0, out=emb_view)

                # Dot product into pre-allocated scores buffer
                np.dot(emb_view, query_emb, out=self._qc_scores_buf[:n])
                scores_view = self._qc_scores_buf[:n]

                k = min(top_k, n)
                if k >= n:
                    top_indices = np.argsort(scores_view)[::-1][:k]
                else:
                    top_indices = np.argpartition(scores_view, -k)[-k:]
                    top_indices = top_indices[np.argsort(scores_view[top_indices])[::-1]]
                result_ids = [faiss_ids[i] for i in top_indices]
                result_scores = scores_view[top_indices]
                return self._results_from_ids(result_ids, result_scores)

            # For large candidate sets, use FAISS
            local = self._faiss.IndexFlatIP(self._dim)
            local_id_map = self._faiss.IndexIDMap2(local)
            fid_arr = np.array(faiss_ids, dtype=np.int64)
            local_id_map.add_with_ids(
                self._emb_array[fid_arr],
                fid_arr,
            )

            k = min(top_k, n)
            self._query_2d[0] = query_emb
            scores, ids = local_id_map.search(self._query_2d, k)
            return self._results_from_ids(ids[0], scores[0])

    def _results_from_ids(self, ids, scores) -> List[L2Result]:
        """Convert FAISS IDs and scores to typed result objects."""
        results = []
        for rank, (score, faiss_id) in enumerate(zip(scores, ids)):
            if int(faiss_id) == -1:
                continue
            doc = self._docs.get(int(faiss_id), {})
            results.append(L2Result(
                document_id=str(doc.get('id', doc.get('hash', faiss_id))),
                content=doc.get('content', doc.get('summary', '')),
                score=float(score),
                rank=rank,
                metadata=doc,
            ))
        return results

    def batch_query(
        self,
        query_texts: List[str],
        top_k: Optional[int] = None,
    ) -> List[List[L2Result]]:
        """Batch-query multiple texts with a single encode call.

        Encodes all query texts in one sentence-transformer batch,
        then searches each embedding individually. Much faster than
        calling query() N times when N > 1.

        Args:
            query_texts: List of natural language queries.
            top_k: Number of results per query.

        Returns:
            List of result lists, one per query text.
        """
        if not query_texts:
            return []
        if len(query_texts) == 1:
            return [self.query(query_texts[0], top_k)]

        top_k = top_k or self.config.max_results

        with self._lock:
            if self._id_map.ntotal == 0 and not self._buffer_embeddings:
                return [[] for _ in query_texts]

            # Single batch encode for all queries
            all_embeddings = self._encode(query_texts)

            results = []
            for query_vec in all_embeddings:
                if self._id_map.ntotal == 0 and self._buffer_embeddings:
                    buf = np.vstack(self._buffer_embeddings)
                    scores = buf @ query_vec
                    k = min(top_k, scores.shape[0])
                    if k >= len(scores):
                        top_indices = np.argsort(scores)[::-1]
                    else:
                        top_indices = np.argpartition(scores, -k)[-k:]
                        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
                    ids = [self._buffer_ids[i] for i in top_indices]
                    results.append(self._results_from_ids(ids, scores[top_indices]))
                else:
                    k = min(top_k, self._id_map.ntotal)
                    qv = query_vec.reshape(1, -1)
                    scores, ids = self._id_map.search(qv, k)
                    results.append(self._results_from_ids(ids[0], scores[0]))

            return results

    def _query_sync(self, query_text: str, top_k: Optional[int] = None) -> List[L2Result]:
        """Sync query interface for ParallelHybrid compatibility."""
        return self.query(query_text, top_k)

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        buffered = len(self._buffer_ids)
        indexed = self._id_map.ntotal if self._id_map else 0
        return buffered + indexed

    def clear(self):
        """Clear all indexed documents."""
        with self._lock:
            self._index = self._faiss.IndexFlatIP(self._dim)
            self._id_map = self._faiss.IndexIDMap2(self._index)
            self._docs.clear()
            self._doc_id_to_faiss_id.clear()
            self._emb_array = np.empty((0, self._dim), dtype=np.float32)
            self._emb_capacity = 0
            self._next_id = 0
            self._ivf_trained = False
            self._buffer_embeddings = []
            self._buffer_ids = []
            self._query_cache_text = None
            self._query_cache_emb = None
