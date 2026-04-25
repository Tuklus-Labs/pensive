"""Tests for the boundary-analysis research benchmark harness."""
import json

import pytest

from pensive import SpreadingActivation
from pensive.boundary_bench import (
    load_boundary_eval_cases,
    mine_boundary_candidates,
    run_boundary_benchmark,
    run_boundary_candidate_mining,
)
from pensive.ingestion.pipeline import IngestPipeline
from pensive.patterns import SYNTHETIC_PATTERNS


def _reports_by_id(result):
    return {report['case_id']: report for report in result['reports']}


class TestBoundaryBenchmarkHarness:
    def test_run_boundary_benchmark_returns_summary_and_reports(self):
        result = run_boundary_benchmark()

        assert result['case_count'] == len(result['reports'])
        assert result['doc_count'] > 0
        assert 'summary' in result
        assert result['summary']['answerable_cases'] > 0

    def test_ambiguous_case_is_flagged_unreliable(self):
        reports = _reports_by_id(run_boundary_benchmark())
        ambiguous = reports['ambiguous-no-context']

        assert ambiguous['context_needed'] is True
        assert ambiguous['predicted_unreliable'] is True
        assert ambiguous['should_trust'] is False
        assert ambiguous['recommended_action'] == 'request_context'
        assert ambiguous['confidence'] == 'low'
        assert ambiguous['disambiguation_gap'] is not None
        assert ambiguous['disambiguation_gap'] < 0.05

    def test_context_case_recovers_expected_answer(self):
        reports = _reports_by_id(run_boundary_benchmark())
        contextual = reports['ambiguous-with-context']

        assert contextual['correct'] is True
        assert contextual['predicted_unreliable'] is False
        assert contextual['should_trust'] is True

    def test_band_crossing_can_appear_on_reliable_queries(self):
        result = run_boundary_benchmark()

        assert result['summary']['band_crossing_cases'] >= 1
        assert any(
            report['band_crossing']
            and report['expect_reliable']
            and report['predicted_unreliable'] is False
            for report in result['reports']
        )

    def test_load_boundary_eval_cases_from_json(self, tmp_path):
        cases_path = tmp_path / 'cases.json'
        payload = [
            {
                'case_id': 'demo',
                'query': 'What happened?',
                'expected_top': 'demo-answer',
                'expect_reliable': True,
                'context': ['demo'],
                'notes': 'demo case',
            }
        ]
        cases_path.write_text(json.dumps(payload), encoding='utf-8')

        cases = load_boundary_eval_cases(str(cases_path))

        assert len(cases) == 1
        assert cases[0].case_id == 'demo'
        assert cases[0].context == ['demo']

    def test_real_graph_mode_requires_cases_file(self):
        with pytest.raises(ValueError, match='requires --cases'):
            run_boundary_benchmark(graph_path='graph.pkl')

    def test_run_boundary_benchmark_with_saved_graph_and_cases(self, tmp_path):
        docs = [
            {
                'content': 'System latency on 2025-07-16 was 199ms at the 99th percentile.',
                'id': 'n1',
                'value': '199ms',
                'query': 'What was the P99 latency on 2025-07-16?',
            },
            {
                'content': 'System latency on 2025-07-16 was 257ms at the 99th percentile.',
                'id': 'n2',
                'value': '257ms',
                'query': 'What was the P99 latency on 2025-07-16?',
            },
        ]
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(docs)

        graph_path = tmp_path / 'graph.pkl'
        IngestPipeline(sa=sa).save_graph(str(graph_path))

        cases_path = tmp_path / 'cases.json'
        cases_path.write_text(json.dumps([
            {
                'case_id': 'ambiguous',
                'query': 'What was the P99 latency on 2025-07-16?',
                'expected_top': None,
                'expect_reliable': False,
                'context': None,
                'notes': 'ambiguous no-context case',
            },
            {
                'case_id': 'contextual',
                'query': 'What was the P99 latency on 2025-07-16?',
                'expected_top': '199ms',
                'expect_reliable': True,
                'context': ['199ms'],
                'notes': 'context-resolved case',
            },
        ]), encoding='utf-8')

        result = run_boundary_benchmark(
            graph_path=str(graph_path),
            cases_path=str(cases_path),
        )

        assert result['graph_path'] == str(graph_path)
        assert result['cases_path'] == str(cases_path)
        assert result['case_count'] == 2
        reports = _reports_by_id(result)
        assert reports['ambiguous']['predicted_unreliable'] is True
        assert reports['contextual']['correct'] is True

    def test_mine_boundary_candidates_finds_recoverable_query(self):
        docs = [
            {
                'content': 'System latency on 2025-07-16 was 199ms at the 99th percentile.',
                'id': 'n1',
                'value': '199ms',
                'query': 'What was the P99 latency on 2025-07-16?',
            },
            {
                'content': 'System latency on 2025-07-16 was 257ms at the 99th percentile.',
                'id': 'n2',
                'value': '257ms',
                'query': 'What was the P99 latency on 2025-07-16?',
            },
        ]
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(docs)

        candidates = mine_boundary_candidates(
            sa,
            queries=['What was the P99 latency on 2025-07-16?'],
            max_candidates=1,
        )

        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate['query'] == 'What was the P99 latency on 2025-07-16?'
        assert candidate['baseline_confidence'] == 'low'
        assert candidate['recovery_confidence'] in {'medium', 'high'}
        assert candidate['recovery_top'] in {'199ms', '257ms'}
        assert len(candidate['benchmark_case_templates']) == 2
        assert candidate['benchmark_case_templates'][0]['expect_reliable'] is False
        assert candidate['benchmark_case_templates'][1]['expect_reliable'] is True

    def test_run_boundary_candidate_mining_with_saved_graph(self, tmp_path):
        docs = [
            {
                'content': 'System latency on 2025-07-16 was 199ms at the 99th percentile.',
                'id': 'n1',
                'value': '199ms',
                'query': 'What was the P99 latency on 2025-07-16?',
            },
            {
                'content': 'System latency on 2025-07-16 was 257ms at the 99th percentile.',
                'id': 'n2',
                'value': '257ms',
                'query': 'What was the P99 latency on 2025-07-16?',
            },
        ]
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(docs)

        graph_path = tmp_path / 'graph.pkl'
        IngestPipeline(sa=sa).save_graph(str(graph_path))

        result = run_boundary_candidate_mining(
            graph_path=str(graph_path),
            queries=['What was the P99 latency on 2025-07-16?'],
            max_candidates=1,
        )

        assert result['graph_path'] == str(graph_path)
        assert result['candidate_count'] == 1
        assert result['candidates'][0]['query'] == 'What was the P99 latency on 2025-07-16?'
