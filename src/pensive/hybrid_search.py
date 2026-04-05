"""Hybrid search combining dense (FAISS) and sparse (BM25) retrieval.

Addresses the limitation of dense vector embeddings struggling with
precise numeric identifiers and exact string matches.

Combines:
1. Dense vector search (FAISS) for semantic similarity
2. Sparse BM25 search for exact/keyword matching
3. Reciprocal Rank Fusion (RRF) to combine results

Reference: "Reciprocal Rank Fusion outperforms Condorcet and individual
Rank Learning Methods" (Cormack et al., 2009)

Requires: pip install pypensive[full]
"""
import logging
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

try:
    from rank_bm25 import BM25Plus
    HAS_RANK_BM25 = True
except ImportError:
    HAS_RANK_BM25 = False

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Unified search result from any retrieval method."""
    document_id: str
    content: str
    score: float
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"  # 'dense', 'sparse', or 'hybrid'


def reciprocal_rank_fusion(
    ranked_lists: List[List[SearchResult]],
    k: int = 60,
    weights: Optional[List[float]] = None
) -> List[SearchResult]:
    """Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for document d:
        RRF(d) = sum_{r in rankings} weight_r / (k + rank_r(d))
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError("Number of weights must match number of ranked lists")

    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_results: Dict[str, SearchResult] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        weight = weights[list_idx]
        for rank, result in enumerate(ranked_list, start=1):
            doc_id = result.document_id
            rrf_scores[doc_id] += weight / (k + rank)

            if doc_id not in doc_results:
                doc_results[doc_id] = result
            elif result.source == 'dense' and doc_results[doc_id].source != 'dense':
                doc_results[doc_id] = result

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    for new_rank, (doc_id, rrf_score) in enumerate(sorted_docs, start=1):
        result = doc_results[doc_id]
        final_results.append(SearchResult(
            document_id=result.document_id,
            content=result.content,
            score=rrf_score,
            rank=new_rank,
            metadata=result.metadata,
            source='hybrid'
        ))

    return final_results


_BM25_TOKEN_RE = re.compile(r'0x[0-9a-f]+|[\w]+(?:[-.][\w]+)*')


class BM25Index:
    """Sparse retrieval index using BM25.

    Excels at exact keyword matching, numeric identifiers, proper nouns,
    and cases where semantic similarity fails.
    """

    def __init__(self, tokenizer: Optional[callable] = None):
        if not HAS_RANK_BM25:
            raise ImportError(
                "rank_bm25 is required for BM25Index. "
                "Install with: pip install pypensive[full]"
            )

        self.tokenizer = tokenizer or self._default_tokenizer
        self.bm25 = None
        self._bm25_dirty = False
        self.documents: List[Dict[str, Any]] = []
        self.doc_id_to_idx: Dict[str, int] = {}
        self._corpus_tokens: List[List[str]] = []

    def _default_tokenizer(self, text: str) -> List[str]:
        """Tokenizer that preserves numeric identifiers and codes."""
        if not text:
            return []
        return _BM25_TOKEN_RE.findall(text.lower())

    def add_documents(self, documents: List[Dict[str, Any]], id_field: str = 'id',
                      content_field: str = 'content'):
        """Add documents to the BM25 index."""
        for doc in documents:
            doc_id = str(doc.get(id_field, len(self.documents)))
            content = doc.get(content_field, '')

            if doc_id in self.doc_id_to_idx:
                continue

            idx = len(self.documents)
            self.doc_id_to_idx[doc_id] = idx
            self.documents.append(doc)
            self._corpus_tokens.append(self.tokenizer(content))

        if self._corpus_tokens:
            self._bm25_dirty = True

    def _ensure_bm25(self):
        """Rebuild BM25 index if dirty (deferred from add_documents)."""
        if self._bm25_dirty and self._corpus_tokens:
            self.bm25 = BM25Plus(self._corpus_tokens)
            self._bm25_dirty = False

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search the BM25 index."""
        self._ensure_bm25()
        if self.bm25 is None or not self.documents:
            return []

        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Use argpartition for O(n) partial sort when top_k << n
        n = len(scores)
        if top_k < n:
            part_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = part_idx[np.argsort(scores[part_idx])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = scores[idx]
            if score <= 0:
                continue
            doc = self.documents[idx]
            results.append(SearchResult(
                document_id=str(doc.get('id', doc.get('hash', idx))),
                content=doc.get('content', doc.get('summary', '')),
                score=float(score),
                rank=rank,
                metadata=doc,
                source='sparse'
            ))
        return results

    @property
    def size(self) -> int:
        return len(self.documents)

    def clear(self):
        self.bm25 = None
        self._bm25_dirty = False
        self.documents = []
        self.doc_id_to_idx = {}
        self._corpus_tokens = []


class HybridSearcher:
    """Combines dense (FAISS) and sparse (BM25) retrieval using RRF."""

    def __init__(
        self,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
        tokenizer: Optional[callable] = None
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.bm25_index = BM25Index(tokenizer=tokenizer)

    def index_documents(self, documents: List[Dict[str, Any]],
                        id_field: str = 'id', content_field: str = 'content'):
        """Index documents for BM25 sparse retrieval."""
        self.bm25_index.add_documents(documents, id_field, content_field)

    def search(self, query: str, dense_results: List[SearchResult],
               top_k: int = 10) -> List[SearchResult]:
        """Hybrid search combining dense and sparse results."""
        sparse_results = self.bm25_index.search(query, top_k=top_k * 2)

        if not dense_results and not sparse_results:
            return []

        ranked_lists = []
        weights = []

        if dense_results:
            ranked_lists.append(dense_results)
            weights.append(self.dense_weight)

        if sparse_results:
            ranked_lists.append(sparse_results)
            weights.append(self.sparse_weight)

        return reciprocal_rank_fusion(ranked_lists, k=self.rrf_k, weights=weights)[:top_k]

    @property
    def bm25_size(self) -> int:
        return self.bm25_index.size


_ID_MEGA = re.compile(
    r'(?P<hex>0x[0-9a-f]+)'
    r'|(?:(?:subsystem|system|unit|module|sector|node)\s*[#]?(?P<sub>\d+))'
    r'|(?:(?:error|err|fault|code)\s*[#:-]?\s*(?P<err>[0-9a-fx]+))'
    r'|(?P<ver>v\d+(?:\.\d+)+)'
    r'|(?P<num>\b\d{3,}\b)'
)


def extract_identifiers(text: str) -> Set[str]:
    """Extract potential identifiers (hex codes, subsystem refs, error codes, versions)."""
    identifiers = set()
    for m in _ID_MEGA.finditer(text.lower()):
        val = m.group('hex') or m.group('sub') or m.group('err') or m.group('ver') or m.group('num')
        if val:
            identifiers.add(val)
    return identifiers


def boost_identifier_matches(
    results: List[SearchResult],
    query: str,
    boost_factor: float = 1.5
) -> List[SearchResult]:
    """Boost scores for results containing exact identifier matches from the query."""
    query_ids = extract_identifiers(query)
    if not query_ids:
        return results

    boosted = []
    for result in results:
        content_ids = extract_identifiers(result.content)
        matching_ids = query_ids & content_ids
        if matching_ids:
            boost = boost_factor ** len(matching_ids)
            boosted.append(SearchResult(
                document_id=result.document_id,
                content=result.content,
                score=result.score * boost,
                rank=result.rank,
                metadata={**result.metadata, 'boosted_ids': list(matching_ids)},
                source=result.source
            ))
        else:
            boosted.append(result)

    boosted.sort(key=lambda x: x.score, reverse=True)
    for i, result in enumerate(boosted, start=1):
        result.rank = i

    return boosted
