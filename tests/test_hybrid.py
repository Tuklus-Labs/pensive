"""Tests for hybrid retrieval stack (requires pypensive[full])."""
import pytest

# Skip all tests if full deps not available
try:
    from pensive.l2 import L2Handler, L2Config, L2Result
    from pensive.hybrid_search import (
        BM25Index, HybridSearcher, SearchResult,
        reciprocal_rank_fusion, extract_identifiers, boost_identifier_matches,
    )
    from pensive.parallel_hybrid import ParallelHybrid, HybridResult
    FULL_AVAILABLE = True
except ImportError:
    FULL_AVAILABLE = False

pytestmark = pytest.mark.skipif(not FULL_AVAILABLE, reason="pypensive[full] not installed")


SAMPLE_DOCS = [
    {'id': 'n1', 'content': 'System latency on 2025-07-16 was 199ms at the 99th percentile.',
     'value': '199ms'},
    {'id': 'n2', 'content': 'GPU temperature reached 82C during the training run on server gpu-node-3.',
     'value': '82C'},
    {'id': 'n3', 'content': 'Meeting with Sarah Chen about Project Atlas budget was scheduled for Friday.',
     'value': 'Atlas budget meeting'},
    {'id': 'n4', 'content': 'Error code 0x4F2A occurred in Subsystem 916 at 03:42 UTC.',
     'value': 'Error 0x4F2A in Subsystem 916'},
    {'id': 'n5', 'content': 'The quarterly revenue was $2.3M, up 15% from last quarter.',
     'value': '$2.3M quarterly revenue'},
]


class TestBM25Index:
    def test_basic_search(self):
        idx = BM25Index()
        idx.add_documents(SAMPLE_DOCS)
        results = idx.search("GPU temperature")
        assert len(results) > 0
        assert results[0].document_id == 'n2'

    def test_numeric_identifier_search(self):
        idx = BM25Index()
        idx.add_documents(SAMPLE_DOCS)
        results = idx.search("Subsystem 916")
        assert any(r.document_id == 'n4' for r in results)

    def test_hex_code_search(self):
        idx = BM25Index()
        idx.add_documents(SAMPLE_DOCS)
        results = idx.search("error 0x4F2A")
        assert any(r.document_id == 'n4' for r in results)

    def test_empty_index(self):
        idx = BM25Index()
        results = idx.search("anything")
        assert results == []

    def test_size(self):
        idx = BM25Index()
        assert idx.size == 0
        idx.add_documents(SAMPLE_DOCS)
        assert idx.size == 5


class TestRRF:
    def test_basic_fusion(self):
        list1 = [SearchResult('a', '', 1.0, 1), SearchResult('b', '', 0.8, 2)]
        list2 = [SearchResult('b', '', 1.0, 1), SearchResult('c', '', 0.8, 2)]
        fused = reciprocal_rank_fusion([list1, list2])
        # 'b' should be top since it appears in both lists
        assert fused[0].document_id == 'b'

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []


class TestIdentifiers:
    def test_extract_hex(self):
        ids = extract_identifiers("Error 0x4F2A in the system")
        assert '0x4f2a' in ids

    def test_extract_subsystem(self):
        ids = extract_identifiers("Subsystem 916 reported failure")
        assert '916' in ids

    def test_boost(self):
        results = [
            SearchResult('a', 'Error 0x4F2A occurred', 1.0, 1),
            SearchResult('b', 'No identifiers here', 1.0, 2),
        ]
        boosted = boost_identifier_matches(results, "What happened with 0x4F2A?")
        assert boosted[0].document_id == 'a'
        assert boosted[0].score > 1.0


class TestL2Handler:
    def test_add_and_query(self):
        l2 = L2Handler(config=L2Config(embedding_model='all-MiniLM-L6-v2'))
        l2.add_documents(SAMPLE_DOCS)
        assert l2.size == 5

        results = l2.query("What was the GPU temperature?")
        assert len(results) > 0
        # Semantic search should find the GPU doc
        doc_ids = [r.document_id for r in results[:3]]
        assert 'n2' in doc_ids

    def test_empty_query(self):
        l2 = L2Handler()
        results = l2.query("anything")
        assert results == []


class TestParallelHybrid:
    def test_sa_only(self):
        from pensive import SpreadingActivation
        from pensive.patterns import SYNTHETIC_PATTERNS

        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build([
            {'id': 'n1', 'content': 'Latency was 199ms on 2025-07-16',
             'value': '199ms', 'query': 'P99 latency on 2025-07-16?'},
        ])

        hybrid = ParallelHybrid(spreading_activation=sa, l2_handler=None)
        results = hybrid.query("P99 latency on 2025-07-16?")
        assert len(results) > 0
        assert results[0].source == 'sa'

    def test_l2_only(self):
        l2 = L2Handler()
        l2.add_documents(SAMPLE_DOCS)

        hybrid = ParallelHybrid(spreading_activation=None, l2_handler=l2)
        results = hybrid.query("GPU temperature")
        assert len(results) > 0
        assert results[0].source == 'l2'

    def test_full_hybrid(self):
        from pensive import SpreadingActivation
        from pensive.patterns import SYNTHETIC_PATTERNS

        docs = [
            {'id': 'n1', 'content': 'Latency was 199ms on 2025-07-16',
             'value': '199ms', 'query': 'P99 latency on 2025-07-16?'},
            {'id': 'n2', 'content': 'GPU temp hit 82C during training run',
             'value': '82C', 'query': 'GPU temperature during training?'},
        ]

        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(docs)

        l2 = L2Handler()
        l2.add_documents(docs)

        hybrid = ParallelHybrid(spreading_activation=sa, l2_handler=l2)
        results = hybrid.query("P99 latency on 2025-07-16?")
        assert len(results) > 0

    def test_no_retrievers(self):
        hybrid = ParallelHybrid(spreading_activation=None, l2_handler=None)
        results = hybrid.query("anything")
        assert results == []
