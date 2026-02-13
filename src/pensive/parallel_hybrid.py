"""Parallel hybrid retrieval.

Fire SA and L2 in parallel, agreement boosting where both find the same
document. Optional cross-encoder reranking and adaptive pattern learning.

Usage:
    from pensive import SpreadingActivation
    from pensive.l2 import L2Handler
    from pensive.parallel_hybrid import ParallelHybrid

    sa = SpreadingActivation()
    sa.build(documents)

    l2 = L2Handler()
    l2.add_documents(documents)

    hybrid = ParallelHybrid(spreading_activation=sa, l2_handler=l2)
    results = hybrid.query("What was the P99 latency?")
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .pattern_learner import PatternLearner

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False


@dataclass
class HybridResult:
    """A result from parallel hybrid retrieval."""
    doc_id: str
    content: str
    summary: str
    score: float
    sa_score: Optional[float]
    l2_score: Optional[float]
    source: str  # 'both', 'sa', 'l2'
    rank: int


class ParallelHybrid:
    """Parallel SA + L2 with agreement boosting.

    The insight: if both SA (graph-based) and L2 (semantic)
    find the same document, it's probably the right answer.
    """

    def __init__(
        self,
        spreading_activation=None,
        l2_handler=None,
        use_cross_encoder: bool = False,
        cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        enable_pattern_learning: bool = True,
        pattern_learner: Optional[PatternLearner] = None,
    ):
        self.sa = spreading_activation
        self.l2 = l2_handler
        self._executor = ThreadPoolExecutor(max_workers=4)

        self.cross_encoder = None
        if use_cross_encoder and CROSS_ENCODER_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model)
                logger.info("Loaded cross-encoder: %s", cross_encoder_model)
            except Exception as e:
                logger.warning("Failed to load cross-encoder: %s", e)

        self.pattern_learner = None
        if enable_pattern_learning:
            self.pattern_learner = pattern_learner or PatternLearner()

    def query(
        self,
        query: str,
        top_k: int = 5,
        sa_top_k: int = 30,
        l2_top_k: int = 20,
        context: Optional[List[str]] = None,
    ) -> List[HybridResult]:
        """Parallel retrieval with agreement boosting.

        Args:
            query: The question
            top_k: Final results to return
            sa_top_k: Candidates from SA
            l2_top_k: Candidates from L2
            context: Optional conversation context for SA

        Returns:
            List of HybridResult, ranked by combined score
        """
        t0 = time.perf_counter()

        # Fire both in parallel
        sa_future = self._executor.submit(self._query_sa, query, sa_top_k, context)
        l2_future = self._executor.submit(self._query_l2, query, l2_top_k)

        sa_results = sa_future.result()
        l2_results = l2_future.result()

        parallel_ms = (time.perf_counter() - t0) * 1000
        logger.info("Parallel: SA=%d, L2=%d in %.0fms",
                     len(sa_results), len(l2_results), parallel_ms)

        # Build lookup tables
        sa_by_id = {r['doc_id']: r for r in sa_results}
        l2_by_id = {r['doc_id']: r for r in l2_results}

        # Merge with agreement boosting
        all_ids = set(sa_by_id.keys()) | set(l2_by_id.keys())

        candidates = []
        for doc_id in all_ids:
            sa_hit = sa_by_id.get(doc_id)
            l2_hit = l2_by_id.get(doc_id)

            if sa_hit and l2_hit:
                # BOTH found it - strong signal
                sa_rank = sa_hit['rank']
                l2_rank = l2_hit['rank']
                combined_rank = 2 / (1/(sa_rank+1) + 1/(l2_rank+1))
                score = 100.0 / combined_rank
                source = 'both'
                content = l2_hit.get('content') or sa_hit.get('content', '')
                summary = l2_hit.get('summary') or sa_hit.get('summary', content[:500])

            elif l2_hit:
                score = 50.0 / (l2_hit['rank'] + 1)
                source = 'l2'
                content = l2_hit.get('content', '')
                summary = l2_hit.get('summary', content[:500])

            else:
                score = 30.0 / (sa_hit['rank'] + 1)
                source = 'sa'
                content = sa_hit.get('content', '')
                summary = sa_hit.get('summary', content[:500])

            candidates.append({
                'doc_id': doc_id,
                'content': content,
                'summary': summary,
                'score': score,
                'sa_score': sa_hit['sa_score'] if sa_hit else None,
                'l2_score': l2_hit['l2_score'] if l2_hit else None,
                'source': source,
            })

        candidates.sort(key=lambda x: -x['score'])

        # Optional cross-encoder reranking
        if self.cross_encoder and candidates:
            candidates = self._rerank_with_cross_encoder(query, candidates[:top_k * 2])

        results = [
            HybridResult(
                doc_id=c['doc_id'],
                content=c['content'],
                summary=c['summary'],
                score=c['score'],
                sa_score=c['sa_score'],
                l2_score=c['l2_score'],
                source=c['source'],
                rank=i,
            )
            for i, c in enumerate(candidates[:top_k])
        ]

        total_ms = (time.perf_counter() - t0) * 1000
        both_count = sum(1 for r in results if r.source == 'both')
        logger.info("Hybrid complete: %d results (%d from both) in %.0fms",
                     len(results), both_count, total_ms)

        # Learn from SA/L2 gaps
        if self.pattern_learner:
            learned = self.pattern_learner.observe(query, sa_results, l2_results)
            if learned:
                logger.info("Learned new patterns: %s", learned)

        return results

    def _query_sa(
        self, query: str, top_k: int, context: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Query spreading activation."""
        if not self.sa or not getattr(self.sa, '_built', False):
            return []

        try:
            results = self.sa.query_with_doc_ids(query, top_k=top_k, context=context)
            return [
                {
                    'doc_id': doc_id,
                    'content': value,
                    'summary': value[:500],
                    'sa_score': score,
                    'rank': i,
                }
                for i, (doc_id, value, score) in enumerate(results)
            ]
        except Exception as e:
            logger.warning("SA query failed: %s", e)
            return []

    def _query_l2(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Query L2 vector store."""
        if not self.l2:
            return []

        try:
            if hasattr(self.l2, '_query_sync'):
                results = self.l2._query_sync(query, top_k=top_k)
            elif hasattr(self.l2, 'query_sync'):
                results = self.l2.query_sync(query, top_k=top_k)
            else:
                results = self.l2.query(query, top_k=top_k)

            return [
                {
                    'doc_id': r.document_id if hasattr(r, 'document_id') else r.get('document_id', ''),
                    'content': r.content if hasattr(r, 'content') else r.get('content', ''),
                    'summary': (r.content if hasattr(r, 'content') else r.get('content', ''))[:500],
                    'l2_score': r.score if hasattr(r, 'score') else r.get('score', 0),
                    'rank': i,
                }
                for i, r in enumerate(results)
            ]
        except Exception as e:
            logger.warning("L2 query failed: %s", e)
            return []

    def _rerank_with_cross_encoder(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank using cross-encoder."""
        if not self.cross_encoder or not candidates:
            return candidates

        try:
            pairs = [(query, c['content'][:512]) for c in candidates]
            scores = self.cross_encoder.predict(pairs)

            for c, score in zip(candidates, scores):
                c['score'] = 0.4 * c['score'] + 0.6 * (score * 10)

            candidates.sort(key=lambda x: -x['score'])
            return candidates

        except Exception as e:
            logger.warning("Cross-encoder rerank failed: %s", e)
            return candidates

    def get_learned_entities(self) -> set:
        """Get all learned entity terms."""
        if self.pattern_learner:
            return self.pattern_learner.get_learned_terms()
        return set()

    def shutdown(self):
        """Shutdown thread pool."""
        self._executor.shutdown(wait=False)
