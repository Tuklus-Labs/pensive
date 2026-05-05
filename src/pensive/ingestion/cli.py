#!/usr/bin/env python3
"""CLI for building and querying the spreading activation graph.

Usage:
    pensive build \
        --chatgpt ~/chatgpt-export/ \
        --facebook ~/facebook-export/ \
        --google ~/google-takeout/ \
        -o sa_graph.pkl

    pensive query --graph sa_graph.pkl "What did we discuss about project X?"

    pensive stats --graph sa_graph.pkl
"""
import argparse
import os
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def _progress(source: str, count: int) -> None:
    """Print ingestion progress."""
    print(f"  [{source}] {count:,} docs", flush=True)


def cmd_build(args):
    """Build SA graph from data sources."""
    from .pipeline import IngestPipeline
    from .chunker import SentenceAwareChunker
    from .parsers.chatgpt import ChatGPTParser
    from .parsers.facebook import FacebookParser
    from .parsers.google import (
        GoogleCalendarParser,
        YouTubeCommentsParser,
        GoogleMapsParser,
    )
    from .parsers.email import EmailJSONLParser

    chunker = SentenceAwareChunker(target_size=args.chunk_size)
    parsers = []

    if args.chatgpt:
        parsers.append(ChatGPTParser(args.chatgpt, chunker=chunker))
    if args.facebook:
        for fb_dir in args.facebook:
            parsers.append(FacebookParser(fb_dir, chunker=chunker))
    if args.google:
        parsers.append(GoogleCalendarParser(args.google))
        parsers.append(YouTubeCommentsParser(args.google))
        parsers.append(GoogleMapsParser(args.google))
    if args.email:
        parsers.append(EmailJSONLParser(args.email, chunker=chunker))

    if not parsers:
        print("No data sources specified. Use --chatgpt, --facebook, or --google.")
        sys.exit(1)

    # Resolve and canonicalize the output path so the existence check
    # operates on the final path, not on a relative form that race
    # conditions could exploit. Refuse to overwrite an existing file
    # unless --force is set so an accidental rerun does not wipe a
    # multi-hour graph build.
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.force:
        print(
            f"ERROR: output file already exists: {output_path}\n"
            "Pass --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)
    args.output = str(output_path)

    print(f"Building SA graph from {len(parsers)} source(s)...")
    pipeline = IngestPipeline(batch_size=args.batch_size, progress_callback=_progress)

    t0 = time.time()
    stats = pipeline.ingest_all(parsers)
    elapsed = time.time() - t0

    print(f"\nIngestion complete in {elapsed:.1f}s:")
    total = 0
    for source, count in stats.items():
        print(f"  {source}: {count:,} documents")
        total += count
    print(f"  TOTAL: {total:,} documents")

    graph_stats = pipeline.sa.stats()
    print(f"\nGraph stats:")
    print(f"  Nodes: {graph_stats['nodes']:,}")
    print(f"  Edges: {graph_stats['edges']:,}")
    print(f"  Entity nodes: {graph_stats['entity_nodes']:,}")
    print(f"  Value nodes: {graph_stats['value_nodes']:,}")
    print(f"  Unique entities: {graph_stats['unique_entities']:,}")

    pipeline.save_graph(args.output)
    print(f"\nSaved to: {args.output}")


def cmd_query(args):
    """Query an existing SA graph."""
    from .pipeline import IngestPipeline

    pipeline = IngestPipeline.load_graph(args.graph, trusted=bool(getattr(args, 'trusted', False)))
    query = ' '.join(args.query_text)
    context = args.context.split(',') if args.context else None

    t0 = time.perf_counter()
    if args.analyze:
        diagnosed = pipeline.sa.query_analyzed(
            query, top_k=args.top_k, context=context
        )
        results = diagnosed.results
    else:
        diagnosed = None
        results = pipeline.sa.query(query, top_k=args.top_k, context=context)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Query: {query}")
    if args.context:
        print(f"Context: {args.context}")
    print(f"Latency: {elapsed_ms:.2f}ms")
    print(f"\nResults ({len(results)}):")
    for i, (value, score) in enumerate(results):
        print(f"  {i+1}. [{score:.4f}] {value[:120]}...")

    if diagnosed is not None:
        analysis = diagnosed.analysis
        print("\nBoundary analysis:")
        print(f"  Confidence: {analysis.confidence}")
        print(f"  Should trust: {analysis.should_trust}")
        print(f"  Recommended action: {analysis.recommended_action}")
        print(f"  Boundary distance: {analysis.boundary_distance}")
        print(f"  Disambiguation gap: {analysis.disambiguation_gap}")
        print(f"  Band crossing: {analysis.band_crossing}")
        print(f"  Context needed: {analysis.context_needed}")
        print(f"  Fundamentally ambiguous: {analysis.fundamentally_ambiguous}")
        if analysis.suggested_context:
            print(f"  Suggested context: {', '.join(analysis.suggested_context)}")


def cmd_stats(args):
    """Show stats for an existing SA graph."""
    from .pipeline import IngestPipeline

    pipeline = IngestPipeline.load_graph(args.graph, trusted=bool(getattr(args, 'trusted', False)))

    print("Graph statistics:")
    for k, v in pipeline.sa.stats().items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")

    if pipeline.stats:
        print("\nIngestion sources:")
        for source, count in pipeline.stats.items():
            print(f"  {source}: {count:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Pensive - Spreading activation retrieval',
    )
    sub = parser.add_subparsers(dest='command')

    # build
    build_p = sub.add_parser('build', help='Build SA graph from data sources')
    build_p.add_argument('--chatgpt', help='Path to chatgpt-export/ directory')
    build_p.add_argument('--facebook', action='append', help='Path to Facebook export directory (can specify multiple)')
    build_p.add_argument('--google', help='Path to Google Takeout/ directory')
    build_p.add_argument('--email', help='Path to email JSONL file or directory')
    build_p.add_argument('-o', '--output', required=True,
                        help='Output graph pickle path')
    build_p.add_argument('--chunk-size', type=int, default=800,
                        help='Target chunk size in characters (default: 800)')
    build_p.add_argument('--batch-size', type=int, default=1000,
                        help='Batch size for add_documents (default: 1000)')
    build_p.add_argument('--force', action='store_true',
                        help='Overwrite the output file if it already exists. '
                             'Without this flag, build refuses to clobber an '
                             'existing graph.')

    # query
    query_p = sub.add_parser('query', help='Query the SA graph')
    query_p.add_argument('--graph', required=True, help='Path to graph pickle')
    query_p.add_argument('--top-k', type=int, default=10)
    query_p.add_argument('--context', help='Comma-separated context entities')
    query_p.add_argument('--analyze', action='store_true',
                         help='Include boundary-analysis diagnostics')
    query_p.add_argument('--trusted', action='store_true',
                         help='Load unsigned legacy pickles (pre-HMAC-signing era). '
                              'Unsafe on untrusted files -- only use on graphs you built yourself.')
    query_p.add_argument('query_text', nargs='+')

    # stats
    stats_p = sub.add_parser('stats', help='Show graph statistics')
    stats_p.add_argument('--graph', required=True, help='Path to graph pickle')
    stats_p.add_argument('--trusted', action='store_true',
                         help='Load unsigned legacy pickles (pre-HMAC-signing era). '
                              'Unsafe on untrusted files -- only use on graphs you built yourself.')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {'build': cmd_build, 'query': cmd_query, 'stats': cmd_stats}[args.command](args)


if __name__ == '__main__':
    main()
