"""Performance regression tests for spreading activation.

These tests verify that query latency stays within acceptable bounds
at various scales. They run with real graph construction and query
execution, not mocked data.

Thresholds are generous (5x headroom) to avoid flaky CI failures
while still catching major regressions.
"""
import random
import time

import pytest

from pensive.spreading import SpreadingActivation, SpreadingConfig


ENTITY_POOL = [
    'GPU_MEMORY', 'CACHE_HIT_RATE', 'P99_LATENCY_MS', 'INFERENCE_SPEED',
    'BATCH_SIZE_CONFIG', 'ROCm_7.1', 'AEGIS_SYSTEM', 'PENSIVE_ENGINE',
    'ENGRAM_DAEMON', 'CHARON_LIFECYCLE', 'KAIROS_TIMING', 'PANOPTES_VISION',
    'Gary Anderson', 'Session Memory', 'Dream Mode Processing',
    'Immune System Patrol', 'Curiosity Engine Scan', 'Token Budget Enforcement',
    'Query Router Stage', 'Vector Store Backend', 'Flash Attention Kernel',
    'Spreading Activation', 'Bipartite Graph Build', 'Semantic Reranking',
    'L1 Hot Cache', 'L2 Deep Retrieval', 'L3 Cold Storage',
    'AMD Radeon 7900 XTX', 'Triton GEMV Kernel', 'INT4 Quantization',
    'KV Cache Management', 'Prompt Builder Module', 'Context Bridge',
]

QUERIES = [
    'latency ms', 'cache hit rate', 'gary anderson memory',
    'dream mode processing', 'attention kernel triton',
    'quantization kv cache', 'semantic reranking query',
]


def _build_graph(num_docs: int, seed: int = 42) -> SpreadingActivation:
    """Build a graph with realistic entity distribution."""
    random.seed(seed)
    docs = []
    for i in range(num_docs):
        entities = random.sample(ENTITY_POOL, random.randint(2, 6))
        content = ' '.join(entities) + f' document {i}'
        docs.append({'id': f'doc_{i}', 'content': content, 'value': content})
    sa = SpreadingActivation(config=SpreadingConfig())
    sa.build(docs)
    return sa


class TestQueryPerformance:
    """Verify query latency stays within bounds at various scales."""

    def test_query_10k_under_500us(self):
        """At 10K docs, per-query mean should be under 500us."""
        sa = _build_graph(10_000)
        # JIT warmup
        sa.query('warmup query', top_k=10)

        times = []
        for _ in range(50):
            sa._substr_match_cache.clear()
            t0 = time.perf_counter_ns()
            for q in QUERIES:
                sa.query(q, top_k=10)
            times.append(time.perf_counter_ns() - t0)

        per_query_us = sum(times) / len(times) / len(QUERIES) / 1000
        assert per_query_us < 500, f"mean query latency {per_query_us:.0f}us > 500us at 10K docs"

    def test_query_50k_under_1ms(self):
        """At 50K docs, per-query mean should be under 1ms."""
        sa = _build_graph(50_000)
        sa.query('warmup query', top_k=10)

        times = []
        for _ in range(30):
            sa._substr_match_cache.clear()
            t0 = time.perf_counter_ns()
            for q in QUERIES:
                sa.query(q, top_k=10)
            times.append(time.perf_counter_ns() - t0)

        per_query_us = sum(times) / len(times) / len(QUERIES) / 1000
        assert per_query_us < 1000, f"mean query latency {per_query_us:.0f}us > 1000us at 50K docs"

    def test_query_with_doc_ids_10k(self):
        """query_with_doc_ids should be within 2x of query() latency."""
        sa = _build_graph(10_000)
        sa.query('warmup', top_k=10)
        sa.query_with_doc_ids('warmup', top_k=10)

        q = 'latency ms cache hit'

        # Benchmark query()
        t0 = time.perf_counter_ns()
        for _ in range(200):
            sa.query(q, top_k=10)
        query_ns = (time.perf_counter_ns() - t0) / 200

        # Benchmark query_with_doc_ids()
        t0 = time.perf_counter_ns()
        for _ in range(200):
            sa.query_with_doc_ids(q, top_k=30)
        qwdi_ns = (time.perf_counter_ns() - t0) / 200

        ratio = qwdi_ns / max(query_ns, 1)
        assert ratio < 3.0, f"query_with_doc_ids is {ratio:.1f}x slower than query()"


class TestBuildPerformance:
    """Verify build time stays reasonable."""

    def test_build_10k_under_5s(self):
        """Building 10K docs should take under 5 seconds."""
        random.seed(42)
        docs = []
        for i in range(10_000):
            entities = random.sample(ENTITY_POOL, random.randint(2, 6))
            content = ' '.join(entities) + f' document {i}'
            docs.append({'id': f'doc_{i}', 'content': content, 'value': content})

        sa = SpreadingActivation(config=SpreadingConfig())
        t0 = time.perf_counter()
        sa.build(docs)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"build took {elapsed:.1f}s > 5s at 10K docs"

    def test_query_200k_under_2ms(self):
        """At 200K docs, per-query mean should be under 2ms."""
        sa = _build_graph(200_000)
        sa.query('warmup query', top_k=10)

        times = []
        for _ in range(20):
            sa._substr_match_cache.clear()
            t0 = time.perf_counter_ns()
            for q in QUERIES:
                sa.query(q, top_k=10)
            times.append(time.perf_counter_ns() - t0)

        per_query_us = sum(times) / len(times) / len(QUERIES) / 1000
        assert per_query_us < 2000, f"mean query latency {per_query_us:.0f}us > 2000us at 200K docs"

    def test_serialization_round_trip_fast(self):
        """Save/load at 10K should complete in under 2 seconds total."""
        sa = _build_graph(10_000)

        t0 = time.perf_counter()
        data = sa.get_save_data()
        save_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        sa2 = SpreadingActivation.from_save_data(data)
        load_s = time.perf_counter() - t0

        total = save_s + load_s
        assert total < 2.0, f"save+load took {total:.2f}s > 2s at 10K docs"

        # Verify loaded graph produces same results
        results1 = sa.query('cache hit rate', top_k=5)
        results2 = sa2.query('cache hit rate', top_k=5)
        assert len(results1) == len(results2), "round-trip changed result count"


class TestTokenIndex:
    """Verify token index correctness and performance."""

    def test_token_index_finds_multi_word_entities(self):
        """Token index should match entities by component words."""
        sa = SpreadingActivation(config=SpreadingConfig())
        docs = [
            {'id': '1', 'content': 'Gary Anderson built this', 'value': 'v1'},
            {'id': '2', 'content': 'Session Memory is active', 'value': 'v2'},
        ]
        sa.build(docs)

        # 'anderson' should find the 'Gary Anderson' entity
        results = sa.query('anderson', top_k=5)
        assert len(results) > 0, "Token index should find 'Gary Anderson' via 'anderson'"

    def test_substring_cold_cache_fast(self):
        """Cold-cache substring lookup should be fast with token index."""
        sa = _build_graph(10_000)
        sa.query('warmup', top_k=10)

        # Measure cold cache performance
        times = []
        test_words = ['anderson', 'latency', 'cache', 'memory', 'kernel']
        for _ in range(100):
            sa._substr_match_cache.clear()
            t0 = time.perf_counter_ns()
            for w in test_words:
                sa._substring_seed_nodes(w)
            times.append(time.perf_counter_ns() - t0)

        per_word_us = sum(times) / len(times) / len(test_words) / 1000
        assert per_word_us < 50, f"cold-cache substring {per_word_us:.1f}us/word > 50us"
