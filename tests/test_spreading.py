"""Tests for spreading activation retrieval."""
import pytest
from unittest.mock import MagicMock

from pensive import SpreadingActivation, SpreadingConfig
from pensive.parallel_hybrid import ParallelHybrid
from pensive.pattern_learner import PatternLearner, integrate_with_sa
from pensive.patterns import SYNTHETIC_PATTERNS


def make_ambiguous_docs():
    """Two docs with same date, different answers - the core disambiguation problem."""
    return [
        {'content': 'System latency on 2025-07-16 was 199ms at the 99th percentile.',
         'id': 'n1', 'value': '199ms',
         'query': 'What was the P99 latency on 2025-07-16?'},
        {'content': 'System latency on 2025-07-16 was 257ms at the 99th percentile.',
         'id': 'n2', 'value': '257ms',
         'query': 'What was the P99 latency on 2025-07-16?'},
    ]


def make_multi_type_docs():
    """Mix of fact, entity, and numeric types."""
    return [
        {'content': 'System latency on 2025-07-16 was 199ms at the 99th percentile.',
         'id': 'n1', 'value': '199ms',
         'query': 'What was the P99 latency on 2025-07-16?'},
        {'content': 'System latency on 2025-07-16 was 257ms at the 99th percentile.',
         'id': 'n2', 'value': '257ms',
         'query': 'What was the P99 latency on 2025-07-16?'},
        {'content': 'Meeting room A-512 has a capacity of 11 people.',
         'id': 'n3', 'value': '11 people',
         'query': 'What is the capacity of meeting room A-512?'},
        {'content': 'Dr. Taylor Smith is the lead researcher on the Horizon initiative.',
         'id': 'n4', 'value': 'Dr. Taylor Smith',
         'query': 'Who leads the Horizon initiative?'},
    ]


class TestBasicQuery:
    def test_query_without_context_returns_both(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_ambiguous_docs())
        results = sa.query("What was the P99 latency on 2025-07-16?")
        answers = [r[0] for r in results]
        assert len(answers) == 2

    def test_fact_type_retrieval(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_multi_type_docs())
        results = sa.query("What is the capacity of meeting room A-512?")
        assert results[0][0] == "11 people"

    def test_entity_type_retrieval(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_multi_type_docs())
        results = sa.query("Who leads the Horizon initiative?")
        assert any("Taylor Smith" in r[0] for r in results)

    def test_empty_graph_raises(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        with pytest.raises(ValueError, match="Graph not built"):
            sa.query("anything")


class TestContextDisambiguation:
    def test_context_disambiguates_199(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_ambiguous_docs())
        results = sa.query(
            "What was the P99 latency on 2025-07-16?",
            context=["199ms"]
        )
        assert results[0][0] == "199ms"
        assert results[0][1] > results[1][1]

    def test_context_disambiguates_257(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_ambiguous_docs())
        results = sa.query(
            "What was the P99 latency on 2025-07-16?",
            context=["257ms"]
        )
        assert results[0][0] == "257ms"

    def test_context_none_matches_no_context(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_ambiguous_docs())
        r1 = sa.query("What was the P99 latency on 2025-07-16?")
        r2 = sa.query("What was the P99 latency on 2025-07-16?", context=None)
        assert r1 == r2

    def test_unmatched_context_degrades_gracefully(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_ambiguous_docs())
        results = sa.query(
            "What was the P99 latency on 2025-07-16?",
            context=["unrelated-entity-xyz"]
        )
        assert len(results) >= 1


class TestContextProvider:
    def test_set_context_provider(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_ambiguous_docs())

        sa.set_context_provider(lambda q: ["199ms"])
        results = sa.query("What was the P99 latency on 2025-07-16?")
        assert results[0][0] == "199ms"

    def test_explicit_context_overrides_provider(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_ambiguous_docs())
        sa.set_context_provider(lambda q: ["199ms"])

        results = sa.query(
            "What was the P99 latency on 2025-07-16?",
            context=["257ms"]
        )
        assert results[0][0] == "257ms"

    def test_provider_not_called_when_explicit_context(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_ambiguous_docs())

        provider = MagicMock(return_value=["199ms"])
        sa.set_context_provider(provider)
        sa.query("What was the P99 latency on 2025-07-16?", context=["257ms"])
        provider.assert_not_called()


class TestIncrementalAdd:
    def test_add_document_builds_graph(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.add_document({
            'content': 'Meeting room A-512 has a capacity of 11 people.',
            'id': 'n1', 'value': '11 people',
            'query': 'What is the capacity of meeting room A-512?'
        })
        assert sa._built is True
        assert sa.stats()['value_nodes'] == 1

    def test_add_document_queryable(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.add_document({
            'content': 'Meeting room A-512 has a capacity of 11 people.',
            'id': 'n1', 'value': '11 people',
            'query': 'What is the capacity of meeting room A-512?'
        })
        results = sa.query("What is the capacity of meeting room A-512?")
        assert any("11 people" in r[0] for r in results)

    def test_incremental_matches_batch_build(self):
        docs = make_multi_type_docs()

        sa_batch = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa_batch.build(docs)

        sa_inc = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa_inc.add_documents(docs)

        assert sa_batch.stats()['nodes'] == sa_inc.stats()['nodes']
        assert sa_batch.stats()['edges'] == sa_inc.stats()['edges']

    def test_incremental_query_matches_batch_build(self):
        docs = [
            {'content': 'Error 0x4F2A on node alpha', 'id': 'a', 'value': 'A', 'query': 'error 0x4F2A'},
            {'content': 'Error 0x4F2A on node beta', 'id': 'b', 'value': 'B', 'query': 'error 0x4F2A'},
            {'content': 'Error 0x4F2A on node gamma', 'id': 'c', 'value': 'C', 'query': 'error 0x4F2A'},
        ]
        query = "error 0x4F2A"

        sa_batch = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa_batch.build(docs)

        sa_inc = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa_inc.add_documents(docs)

        r_batch = sa_batch.query_with_doc_ids(query, top_k=3)
        r_inc = sa_inc.query_with_doc_ids(query, top_k=3)
        assert r_batch == r_inc

    def test_add_after_build_extends_graph(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_ambiguous_docs())
        initial_nodes = sa.stats()['nodes']

        sa.add_document({
            'content': 'Dr. Taylor Smith is the lead researcher on the Horizon initiative.',
            'id': 'n_extra', 'value': 'Dr. Taylor Smith',
            'query': 'Who leads the Horizon initiative?'
        })
        assert sa.stats()['nodes'] > initial_nodes


class TestMegaExtractor:
    def test_extract_returns_entities(self):
        from pensive.mega_extract import MegaExtractor
        from pensive.patterns import SYNTHETIC_PATTERNS

        ext = MegaExtractor(SYNTHETIC_PATTERNS)
        results = ext.extract("System latency on 2025-07-16 was 199ms")
        assert len(results) > 0
        labels = [r[0] for r in results]
        assert any("2025-07-16" in l for l in labels)

    def test_extract_with_raw_preserves_case(self):
        from pensive.mega_extract import MegaExtractor
        from pensive.patterns import REAL_DATA_PATTERNS

        ext = MegaExtractor(REAL_DATA_PATTERNS)
        results = ext.extract_with_raw("Meeting with Dr. Sarah Chen tomorrow")
        for lower, etype, raw in results:
            assert lower == raw.lower()


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_multi_type_docs())

        data = sa.get_save_data()
        sa2 = SpreadingActivation.from_save_data(data)

        assert sa.stats() == sa2.stats()

        r1 = sa.query("What is the capacity of meeting room A-512?")
        r2 = sa2.query("What is the capacity of meeting room A-512?")
        assert r1[0][0] == r2[0][0]

    def test_add_after_load_keeps_existing_edges(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_multi_type_docs())

        data = sa.get_save_data()
        sa2 = SpreadingActivation.from_save_data(data)

        sa2.add_document({
            'content': 'Room B-111 capacity is 23 people.',
            'id': 'n_extra',
            'value': '23 people',
            'query': 'What is room B-111 capacity?',
        })

        existing = sa2.query("What is the capacity of meeting room A-512?")
        added = sa2.query("What is room B-111 capacity?")

        assert any("11 people" in r[0] for r in existing)
        assert any("23 people" in r[0] for r in added)


class TestPatternLearnerIntegration:
    def test_integrate_with_sparse_sa(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build([
            {'content': 'GPU temperature reached 82C during training.',
             'id': 'n1', 'value': 'GPU temperature 82C', 'query': 'gpu temperature'},
        ])

        learner = PatternLearner()
        learner.add_manual('gpu')
        integrate_with_sa(sa, learner)

        results = sa.query_with_doc_ids("gpu temperature", top_k=5)
        assert len(results) > 0
        assert results[0][0] == 'n1'


class _FakeL2Result:
    def __init__(self, document_id, content, score):
        self.document_id = document_id
        self.content = content
        self.score = score


class _FakeL2:
    def __init__(self):
        self.candidate_calls = 0
        self.global_calls = 0

    def query_candidates(self, query_text, candidate_doc_ids, top_k=10):
        self.candidate_calls += 1
        docs = list(dict.fromkeys(candidate_doc_ids))[:top_k]
        return [_FakeL2Result(doc_id, f"doc {doc_id}", 1.0 / (i + 1)) for i, doc_id in enumerate(docs)]

    def query(self, query_text, top_k=10):
        self.global_calls += 1
        return [_FakeL2Result('global', 'global', 0.1)]


class TestParallelHybridRouting:
    def test_uses_l2_candidates_when_sa_hits_exist(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build([
            {'id': 'n1', 'content': 'Latency was 199ms on 2025-07-16',
             'value': '199ms', 'query': 'P99 latency on 2025-07-16?'},
        ])
        fake_l2 = _FakeL2()
        hybrid = ParallelHybrid(
            spreading_activation=sa,
            l2_handler=fake_l2,
            l2_on_sa_hits=True,
            l2_fallback_global=False,
            enable_pattern_learning=False,
        )

        results = hybrid.query("P99 latency on 2025-07-16?", top_k=1)
        assert len(results) == 1
        assert fake_l2.candidate_calls == 1
        assert fake_l2.global_calls == 0

    def test_falls_back_to_global_l2_when_no_sa_hits(self):
        fake_l2 = _FakeL2()
        hybrid = ParallelHybrid(
            spreading_activation=None,
            l2_handler=fake_l2,
            l2_on_sa_hits=True,
            l2_fallback_global=True,
            enable_pattern_learning=False,
        )

        results = hybrid.query("anything", top_k=1)
        assert len(results) == 1
        assert fake_l2.candidate_calls == 0
        assert fake_l2.global_calls == 1


class TestQueryWithDocIds:
    def test_returns_doc_ids(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_multi_type_docs())
        results = sa.query_with_doc_ids("What is the capacity of meeting room A-512?")
        assert len(results) > 0
        doc_id, value, score = results[0]
        assert doc_id == "n3"
        assert value == "11 people"


class TestStats:
    def test_stats_keys(self):
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(make_multi_type_docs())
        s = sa.stats()
        assert 'nodes' in s
        assert 'edges' in s
        assert 'entity_nodes' in s
        assert 'value_nodes' in s
        assert 'unique_entities' in s
        assert s['value_nodes'] == 4
