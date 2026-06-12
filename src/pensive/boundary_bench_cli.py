"""CLI report layer for the boundary-analysis research benchmark.

REFACTOR-6: the print-based reporting used to live in boundary_bench.py
next to the importable scoring functions, so importing the scorer pulled
in CLI presentation. The pure benchmark/scoring logic stays in
boundary_bench; this module is the one place in the package (besides the
`pensive` ingestion CLI) that is allowed to print.

Run via the repo-root wrapper or as a module:

    python bench_boundary.py [--json] [--graph ... --cases ...]
    python -m pensive.boundary_bench_cli --mine-candidates --graph ...
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from .boundary_bench import (
    run_boundary_benchmark,
    run_boundary_candidate_mining,
)


def print_boundary_benchmark(result: Dict[str, Any]) -> None:
    """Print a human-readable benchmark summary."""
    summary = result['summary']

    def _metric(value: Optional[float]) -> str:
        return 'n/a' if value is None else f'{value:.2f}'

    print('Boundary Analysis Research Benchmark')
    print('=' * 40)
    print(f"Docs: {result['doc_count']}")
    print(f"Cases: {result['case_count']}")
    print(f"Top-1 accuracy on answerable cases: {_metric(summary['top1_accuracy'])}")
    print(f"Unreliable recall: {_metric(summary['unreliable_recall'])}")
    print(f"Unreliable precision: {_metric(summary['unreliable_precision'])}")
    print(
        'Reliable false-positive rate: '
        f"{_metric(summary['reliable_false_positive_rate'])}"
    )
    print(f"Context-case accuracy: {_metric(summary['context_case_accuracy'])}")
    print(f"Band-crossing cases: {summary['band_crossing_cases']}")

    print('\nConfidence buckets:')
    for label, bucket in summary['confidence_buckets'].items():
        print(
            f"  {label:>6}: cases={bucket['cases']}, "
            f"answerable={bucket['answerable_cases']}, "
            f"top1_accuracy={_metric(bucket['top1_accuracy'])}"
        )

    print('\nCases:')
    for report in result['reports']:
        print(
            f"  {report['case_id']}: top={report['top_value']!r}, "
            f"confidence={report['confidence']}, "
            f"boundary={report['boundary_distance']}, "
            f"context_needed={report['context_needed']}, "
            f"band_crossing={report['band_crossing']}, "
            f"predicted_unreliable={report['predicted_unreliable']}"
        )


def print_boundary_candidate_mining(result: Dict[str, Any]) -> None:
    """Print a human-readable summary of mined recovery candidates."""
    print('Boundary Candidate Mining')
    print('=' * 28)
    print(f"Graph: {result['graph_path']}")
    print(f"Candidates: {result['candidate_count']}")
    print(
        'Search params: '
        f"max_queries={result['max_queries']}, "
        f"max_candidates={result['max_candidates']}, "
        f"min_query_frequency={result['min_query_frequency']}, "
        f"top_k={result['top_k']}, "
        f"max_context_terms={result['max_context_terms']}"
    )

    print('\nCandidates:')
    for candidate in result['candidates']:
        print(
            f"  query={candidate['query']!r}, "
            f"baseline={candidate['baseline_confidence']} -> "
            f"recovery={candidate['recovery_confidence']}, "
            f"context={candidate['recovery_context']}, "
            f"top={candidate['recovery_top']!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run the synthetic boundary-analysis research benchmark.',
    )
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--graph',
                        help='Optional saved Pensive graph for real-corpus evaluation.')
    parser.add_argument('--cases',
                        help='JSON file with benchmark cases for --graph runs.')
    parser.add_argument('--mine-candidates', action='store_true',
                        help='Mine recoverable low-confidence queries from a saved graph.')
    parser.add_argument('--max-queries', type=int, default=50)
    parser.add_argument('--max-candidates', type=int, default=10)
    parser.add_argument('--min-query-frequency', type=int, default=2)
    parser.add_argument('--max-context-terms', type=int, default=4)
    parser.add_argument('--json', action='store_true',
                        help='Emit the full benchmark report as JSON.')
    args = parser.parse_args()

    if args.mine_candidates:
        if args.graph is None:
            raise ValueError('--mine-candidates requires --graph')
        result = run_boundary_candidate_mining(
            graph_path=args.graph,
            max_queries=args.max_queries,
            max_candidates=args.max_candidates,
            min_query_frequency=args.min_query_frequency,
            top_k=args.top_k,
            max_context_terms=args.max_context_terms,
        )
        if args.json:
            print(json.dumps(result, indent=2))
            return
        print_boundary_candidate_mining(result)
        return

    result = run_boundary_benchmark(
        top_k=args.top_k,
        graph_path=args.graph,
        cases_path=args.cases,
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print_boundary_benchmark(result)


if __name__ == '__main__':
    main()
