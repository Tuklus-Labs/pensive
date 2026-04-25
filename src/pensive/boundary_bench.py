"""Research harness for evaluating boundary-analysis usefulness.

This is intentionally small and opinionated: it builds a synthetic corpus
with both clear and ambiguous retrieval cases, runs `query_analyzed()`,
and reports whether the diagnostic signals line up with the cases where
retrieval should or should not be trusted.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import SpreadingActivation
from .ingestion.pipeline import IngestPipeline
from .patterns import SYNTHETIC_PATTERNS


@dataclass(frozen=True)
class BoundaryEvalCase:
    case_id: str
    query: str
    expected_top: Optional[str]
    expect_reliable: bool
    context: Optional[List[str]] = None
    notes: str = ""


def build_boundary_eval_docs() -> List[Dict[str, str]]:
    """Build a tiny corpus with clear, ambiguous, and cross-band queries."""
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
        {
            'id': 'n3',
            'content': 'Meeting room A-512 has capacity of 11 people.',
            'value': '11 people',
            'query': 'Capacity of meeting room A-512?',
        },
        {
            'id': 'n4',
            'content': 'Dr. Taylor Smith leads the Horizon initiative.',
            'value': 'Dr. Taylor Smith',
            'query': 'Who leads Horizon?',
        },
    ]

    atlas_values = [121, 130, 147, 160, 171, 180, 194, 153]
    for offset, value in enumerate(atlas_values, start=1):
        date = f"2025-08-{offset:02d}"
        docs.append(
            {
                'id': f'atlas-{offset}',
                'content': (
                    f'Project Atlas latency on {date} was {value}ms in production. '
                    'Atlas telemetry was reviewed by the performance team.'
                ),
                'value': f'{value}ms',
                'query': f'What was the Project Atlas latency on {date}?',
            }
        )

    return docs


def get_boundary_eval_cases() -> List[BoundaryEvalCase]:
    """Named benchmark cases for deciding whether the signals are useful."""
    return [
        BoundaryEvalCase(
            case_id='ambiguous-no-context',
            query='What was the P99 latency on 2025-07-16?',
            expected_top=None,
            expect_reliable=False,
            notes='Classic ambiguity: same query activates two different answers.',
        ),
        BoundaryEvalCase(
            case_id='ambiguous-with-context',
            query='What was the P99 latency on 2025-07-16?',
            expected_top='199ms',
            expect_reliable=True,
            context=['199ms'],
            notes='Context should convert the same query into a trustworthy hit.',
        ),
        BoundaryEvalCase(
            case_id='clear-room-capacity',
            query='Capacity of meeting room A-512?',
            expected_top='11 people',
            expect_reliable=True,
            notes='Straightforward one-answer retrieval, but it still checks whether '
                  'band crossing can appear on a reliable result.',
        ),
        BoundaryEvalCase(
            case_id='clear-entity-query',
            query='Who leads Horizon?',
            expected_top='Dr. Taylor Smith',
            expect_reliable=True,
            notes='Named-entity retrieval without ambiguity.',
        ),
        BoundaryEvalCase(
            case_id='atlas-latency-query',
            query='What was the Project Atlas latency on 2025-08-08?',
            expected_top='153ms',
            expect_reliable=True,
            notes='Common project + specific date; useful for checking how often '
                  'band structure actually shows up in practice.',
        ),
        BoundaryEvalCase(
            case_id='nonsense-query',
            query='xyzzy nonexistent gibberish',
            expected_top=None,
            expect_reliable=False,
            notes='No retrieval result should be treated as unreliable.',
        ),
    ]


def load_boundary_eval_cases(path: str) -> List[BoundaryEvalCase]:
    """Load benchmark cases from a JSON file for real-corpus evaluation."""
    with open(path, 'r', encoding='utf-8') as f:
        raw_cases = json.load(f)
    return [BoundaryEvalCase(**case) for case in raw_cases]


def _confidence_rank(label: str) -> int:
    """Map confidence labels to a sortable score."""
    return {
        'none': 0,
        'low': 1,
        'medium': 2,
        'high': 3,
    }.get(label, -1)


def _build_context_trials(suggested_context: List[str],
                          max_context_terms: int = 4,
                          include_pairs: bool = True) -> List[List[str]]:
    """Generate single-term and pairwise context trials from suggestions."""
    limited = suggested_context[:max_context_terms]
    trials = [[term] for term in limited]
    if include_pairs:
        trials.extend([list(pair) for pair in combinations(limited, 2)])
    return trials


def _slugify(text: str) -> str:
    """Small helper for stable benchmark case ids."""
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    return slug or 'query'


def candidate_to_boundary_eval_cases(candidate: Dict[str, Any],
                                     prefix: str = 'candidate'
                                     ) -> List[BoundaryEvalCase]:
    """Convert a mined recovery candidate into benchmark-case templates."""
    query_slug = _slugify(candidate['query'])
    context_slug = _slugify(' '.join(candidate['recovery_context']))
    return [
        BoundaryEvalCase(
            case_id=f'{prefix}-{query_slug}-ambiguous',
            query=candidate['query'],
            expected_top=None,
            expect_reliable=False,
            context=None,
            notes=(
                'Mined ambiguous real-graph query. Boundary analysis requested '
                'context before the result should be trusted.'
            ),
        ),
        BoundaryEvalCase(
            case_id=f'{prefix}-{query_slug}-{context_slug}',
            query=candidate['query'],
            expected_top=candidate['recovery_top'],
            expect_reliable=True,
            context=candidate['recovery_context'],
            notes=(
                'Mined recovery case. Suggested context upgraded the same query '
                'to a trustworthy result.'
            ),
        ),
    ]


def mine_boundary_candidates(
    sa: SpreadingActivation,
    queries: Optional[List[str]] = None,
    max_queries: int = 50,
    max_candidates: int = 10,
    min_query_frequency: int = 2,
    top_k: int = 5,
    max_context_terms: int = 4,
    include_pairs: bool = True,
) -> List[Dict[str, Any]]:
    """Mine low-trust queries that become trustworthy with suggested context.

    This is meant for research triage, not automatic benchmark generation.
    The output is a list of candidate recoveries plus case templates that can
    be manually curated into a real benchmark file.
    """
    if queries is None:
        ranked_entities = sorted(
            sa.entity_freq.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        query_rows = [
            (query, freq)
            for query, freq in ranked_entities
            if freq >= min_query_frequency
        ][:max_queries]
    else:
        query_rows = [
            (query, sa.entity_freq.get(query, 0))
            for query in queries[:max_queries]
        ]

    candidates: List[Dict[str, Any]] = []
    seen_queries = set()
    for query, query_frequency in query_rows:
        if query in seen_queries:
            continue
        seen_queries.add(query)

        baseline = sa.query_analyzed(query, top_k=top_k)
        baseline_analysis = baseline.analysis
        if (
            baseline_analysis.recommended_action != 'request_context'
            or not baseline_analysis.suggested_context
        ):
            continue

        baseline_top = baseline.results[0][0] if baseline.results else None
        baseline_rank = _confidence_rank(baseline_analysis.confidence)

        for context in _build_context_trials(
            baseline_analysis.suggested_context,
            max_context_terms=max_context_terms,
            include_pairs=include_pairs,
        ):
            recovered = sa.query_analyzed(query, top_k=top_k, context=context)
            if not recovered.analysis.should_trust or not recovered.results:
                continue

            recovered_top = recovered.results[0][0]
            recovered_rank = _confidence_rank(recovered.analysis.confidence)
            if (
                recovered_rank < baseline_rank
                or (
                    recovered_rank == baseline_rank
                    and recovered_top == baseline_top
                )
            ):
                continue

            candidate = {
                'query': query,
                'query_frequency': query_frequency,
                'baseline_top': baseline_top,
                'baseline_confidence': baseline_analysis.confidence,
                'baseline_action': baseline_analysis.recommended_action,
                'baseline_boundary_distance': (
                    baseline_analysis.boundary_distance
                ),
                'baseline_disambiguation_gap': (
                    baseline_analysis.disambiguation_gap
                ),
                'baseline_suggested_context': (
                    baseline_analysis.suggested_context[: max_context_terms * 2]
                ),
                'recovery_context': context,
                'recovery_top': recovered_top,
                'recovery_confidence': recovered.analysis.confidence,
                'recovery_action': recovered.analysis.recommended_action,
                'recovery_boundary_distance': (
                    recovered.analysis.boundary_distance
                ),
            }
            case_templates = candidate_to_boundary_eval_cases(
                candidate,
                prefix='mined',
            )
            candidate['benchmark_case_templates'] = [
                case.__dict__
                for case in case_templates
            ]
            candidates.append(candidate)
            break

        if len(candidates) >= max_candidates:
            break

    return candidates


def evaluate_case(sa: SpreadingActivation, case: BoundaryEvalCase,
                  top_k: int = 3) -> Dict[str, Any]:
    """Run a single evaluation case through query_analyzed()."""
    diagnosed = sa.query_analyzed(case.query, top_k=top_k, context=case.context)
    top_value = diagnosed.results[0][0] if diagnosed.results else None
    report = {
        'case_id': case.case_id,
        'query': case.query,
        'context': case.context,
        'expected_top': case.expected_top,
        'expect_reliable': case.expect_reliable,
        'notes': case.notes,
        'top_value': top_value,
        'top_scores': diagnosed.analysis.top_scores,
        'correct': None if case.expected_top is None else top_value == case.expected_top,
        'confidence': diagnosed.analysis.confidence,
        'boundary_distance': diagnosed.analysis.boundary_distance,
        'disambiguation_gap': diagnosed.analysis.disambiguation_gap,
        'band_crossing': diagnosed.analysis.band_crossing,
        'context_needed': diagnosed.analysis.context_needed,
        'suggested_context': diagnosed.analysis.suggested_context,
        'fundamentally_ambiguous': diagnosed.analysis.fundamentally_ambiguous,
        'should_trust': diagnosed.analysis.should_trust,
        'recommended_action': diagnosed.analysis.recommended_action,
    }
    report['predicted_unreliable'] = not diagnosed.analysis.should_trust
    return report


def summarize_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate high-signal research metrics from per-case reports."""
    total_cases = len(reports)
    answerable = [r for r in reports if r['expected_top'] is not None]
    reliable = [r for r in reports if r['expect_reliable']]
    unreliable = [r for r in reports if not r['expect_reliable']]
    predicted_bad = [r for r in reports if r['predicted_unreliable']]

    confidence_buckets: Dict[str, Dict[str, Any]] = {}
    for label in ('high', 'medium', 'low', 'none'):
        bucket = [r for r in reports if r['confidence'] == label]
        bucket_answerable = [r for r in bucket if r['correct'] is not None]
        accuracy = None
        if bucket_answerable:
            accuracy = sum(1 for r in bucket_answerable if r['correct']) / len(bucket_answerable)
        confidence_buckets[label] = {
            'cases': len(bucket),
            'answerable_cases': len(bucket_answerable),
            'top1_accuracy': accuracy,
        }

    def _mean(values: List[Optional[float]]) -> Optional[float]:
        filtered = [v for v in values if v is not None]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)

    context_answerable = [r for r in answerable if r['context']]

    return {
        'total_cases': total_cases,
        'answerable_cases': len(answerable),
        'top1_accuracy': (
            sum(1 for r in answerable if r['correct']) / len(answerable)
            if answerable else None
        ),
        'unreliable_recall': (
            sum(1 for r in unreliable if r['predicted_unreliable']) / len(unreliable)
            if unreliable else None
        ),
        'unreliable_precision': (
            sum(1 for r in predicted_bad if not r['expect_reliable']) / len(predicted_bad)
            if predicted_bad else None
        ),
        'reliable_false_positive_rate': (
            sum(1 for r in reliable if r['predicted_unreliable']) / len(reliable)
            if reliable else None
        ),
        'context_case_accuracy': (
            sum(1 for r in context_answerable if r['correct']) / len(context_answerable)
            if context_answerable else None
        ),
        'band_crossing_cases': sum(1 for r in reports if r['band_crossing']),
        'mean_boundary_distance_correct': _mean(
            [r['boundary_distance'] for r in answerable if r['correct']]
        ),
        'mean_boundary_distance_incorrect': _mean(
            [r['boundary_distance'] for r in answerable if r['correct'] is False]
        ),
        'confidence_buckets': confidence_buckets,
    }


def run_boundary_benchmark(
    top_k: int = 3,
    graph_path: Optional[str] = None,
    cases_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the synthetic suite or a real-corpus boundary benchmark."""
    if graph_path is not None:
        if cases_path is None:
            raise ValueError("--graph requires --cases")
        sa = IngestPipeline.load_graph(graph_path).sa
        cases = load_boundary_eval_cases(cases_path)
        doc_count = sa.stats()['value_nodes']
    else:
        docs = build_boundary_eval_docs()
        sa = SpreadingActivation(patterns=SYNTHETIC_PATTERNS)
        sa.build(docs)
        cases = get_boundary_eval_cases()
        doc_count = len(docs)

    reports = [evaluate_case(sa, case, top_k=top_k) for case in cases]
    summary = summarize_reports(reports)
    return {
        'summary': summary,
        'reports': reports,
        'case_count': len(cases),
        'doc_count': doc_count,
        'graph_path': str(Path(graph_path)) if graph_path is not None else None,
        'cases_path': str(Path(cases_path)) if cases_path is not None else None,
    }


def run_boundary_candidate_mining(
    graph_path: str,
    max_queries: int = 50,
    max_candidates: int = 10,
    min_query_frequency: int = 2,
    top_k: int = 5,
    max_context_terms: int = 4,
    include_pairs: bool = True,
    queries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load a saved graph and mine candidate recoverable ambiguity cases."""
    sa = IngestPipeline.load_graph(graph_path).sa
    candidates = mine_boundary_candidates(
        sa,
        queries=queries,
        max_queries=max_queries,
        max_candidates=max_candidates,
        min_query_frequency=min_query_frequency,
        top_k=top_k,
        max_context_terms=max_context_terms,
        include_pairs=include_pairs,
    )
    return {
        'graph_path': str(Path(graph_path)),
        'candidate_count': len(candidates),
        'max_queries': max_queries,
        'max_candidates': max_candidates,
        'min_query_frequency': min_query_frequency,
        'top_k': top_k,
        'max_context_terms': max_context_terms,
        'candidates': candidates,
    }


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
