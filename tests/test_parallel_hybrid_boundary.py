"""Tests for boundary-aware hybrid fallback behavior."""
from dataclasses import dataclass

from pensive import SpreadingActivation
from pensive.parallel_hybrid import ParallelHybrid
from pensive.patterns import SYNTHETIC_PATTERNS


def _build_ambiguous_sa():
    docs = [
        {
            'id': 'n1',
            'content': 'System latency on 2025-07-16 was 199ms at the 99th percentile.',
            'value': '199ms',
            'query': 'What was the P99 latency on 2025-07-16?',
        },
        {
            'id': 'n2',
            'content': 'System latency on 2025-07-16 was 257ms at the 99th percentile.',
            'value': '257ms',
            'query': 'What was the P99 latency on 2025-07-16?',
        },
    ]
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build(docs)
    return sa


def _build_distinct_sa():
    docs = [
        {
            'id': 'n1',
            'content': 'Meeting room A-512 has capacity of 11 people.',
            'value': '11 people',
            'query': 'Capacity of meeting room A-512?',
        },
        {
            'id': 'n2',
            'content': 'Dr. Taylor Smith leads the Horizon initiative.',
            'value': 'Dr. Taylor Smith',
            'query': 'Who leads Horizon?',
        },
    ]
    sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
    sa.build(docs)
    return sa


@dataclass
class _DummyL2Result:
    document_id: str
    content: str
    score: float


class _DummyL2:
    def __init__(self):
        self.global_calls = 0
        self.candidate_calls = 0
        self.last_candidate_ids = None

    def query(self, query: str, top_k: int = 20):
        self.global_calls += 1
        return [_DummyL2Result('global-doc', 'Global semantic answer', 0.99)]

    def query_candidates(self, query: str, candidate_doc_ids, top_k: int = 20):
        self.candidate_calls += 1
        self.last_candidate_ids = list(candidate_doc_ids)
        if not candidate_doc_ids:
            return []
        return [
            _DummyL2Result(candidate_doc_ids[0], 'Candidate semantic answer', 0.75)
        ]


class TestBoundaryAwareHybridFallback:
    def test_low_confidence_sa_uses_global_l2_when_enabled(self):
        sa = _build_ambiguous_sa()
        l2 = _DummyL2()
        hybrid = ParallelHybrid(
            spreading_activation=sa,
            l2_handler=l2,
            l2_fallback_on_low_confidence=True,
            enable_pattern_learning=False,
        )

        results = hybrid.query("What was the P99 latency on 2025-07-16?")

        assert l2.global_calls == 1
        assert l2.candidate_calls == 0
        assert any(r.doc_id == 'global-doc' for r in results)

    def test_high_confidence_sa_stays_on_candidate_rerank(self):
        sa = _build_distinct_sa()
        l2 = _DummyL2()
        hybrid = ParallelHybrid(
            spreading_activation=sa,
            l2_handler=l2,
            l2_fallback_on_low_confidence=True,
            enable_pattern_learning=False,
        )

        hybrid.query("Capacity of meeting room A-512?")

        assert l2.global_calls == 0
        assert l2.candidate_calls == 1
        assert l2.last_candidate_ids == ['n1']

    def test_low_confidence_fallback_is_opt_in(self):
        sa = _build_ambiguous_sa()
        l2 = _DummyL2()
        hybrid = ParallelHybrid(
            spreading_activation=sa,
            l2_handler=l2,
            l2_fallback_on_low_confidence=False,
            enable_pattern_learning=False,
        )

        hybrid.query("What was the P99 latency on 2025-07-16?")

        assert l2.global_calls == 0
        assert l2.candidate_calls == 1
