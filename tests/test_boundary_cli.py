"""End-to-end CLI smoke tests for boundary analysis."""
import subprocess
import sys

from pensive import SpreadingActivation
from pensive.ingestion.pipeline import IngestPipeline
from pensive.patterns import SYNTHETIC_PATTERNS


def _write_ambiguous_graph(path):
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
    IngestPipeline(sa=sa).save_graph(str(path))


class TestBoundaryCli:
    def test_query_analyze_flags_ambiguous_result(self, tmp_path):
        graph_path = tmp_path / 'boundary_graph.pkl'
        _write_ambiguous_graph(graph_path)

        proc = subprocess.run(
            [
                sys.executable,
                '-m',
                'pensive.ingestion.cli',
                'query',
                '--graph',
                str(graph_path),
                '--analyze',
                'What was the P99 latency on 2025-07-16?',
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        assert 'Confidence: low' in proc.stdout
        assert 'Should trust: False' in proc.stdout
        assert 'Recommended action: request_context' in proc.stdout
        assert 'Suggested context: 199ms, 257ms' in proc.stdout

    def test_query_analyze_with_context_restores_trust(self, tmp_path):
        graph_path = tmp_path / 'boundary_graph.pkl'
        _write_ambiguous_graph(graph_path)

        proc = subprocess.run(
            [
                sys.executable,
                '-m',
                'pensive.ingestion.cli',
                'query',
                '--graph',
                str(graph_path),
                '--analyze',
                '--context',
                '199ms',
                'What was the P99 latency on 2025-07-16?',
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        assert 'Confidence: high' in proc.stdout
        assert 'Should trust: True' in proc.stdout
        assert 'Recommended action: trust' in proc.stdout
