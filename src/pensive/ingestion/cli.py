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
import sys
import time
import logging

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

    chunker = SentenceAwareChunker(target_size=args.chunk_size)
    parsers = []

    if args.chatgpt:
        parsers.append(ChatGPTParser(args.chatgpt, chunker=chunker))
    if args.facebook:
        parsers.append(FacebookParser(args.facebook, chunker=chunker))
    if args.google:
        parsers.append(GoogleCalendarParser(args.google))
        parsers.append(YouTubeCommentsParser(args.google))
        parsers.append(GoogleMapsParser(args.google))

    if not parsers:
        print("No data sources specified. Use --chatgpt, --facebook, or --google.")
        sys.exit(1)

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

    pipeline = IngestPipeline.load_graph(args.graph)
    query = ' '.join(args.query_text)

    t0 = time.perf_counter()
    results = pipeline.sa.query(query, top_k=args.top_k,
                                 context=args.context.split(',') if args.context else None)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Query: {query}")
    if args.context:
        print(f"Context: {args.context}")
    print(f"Latency: {elapsed_ms:.2f}ms")
    print(f"\nResults ({len(results)}):")
    for i, (value, score) in enumerate(results):
        print(f"  {i+1}. [{score:.4f}] {value[:120]}...")


def cmd_stats(args):
    """Show stats for an existing SA graph."""
    from .pipeline import IngestPipeline

    pipeline = IngestPipeline.load_graph(args.graph)

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
    build_p.add_argument('--facebook', help='Path to Facebook export directory')
    build_p.add_argument('--google', help='Path to Google Takeout/ directory')
    build_p.add_argument('-o', '--output', required=True,
                        help='Output graph pickle path')
    build_p.add_argument('--chunk-size', type=int, default=800,
                        help='Target chunk size in characters (default: 800)')
    build_p.add_argument('--batch-size', type=int, default=1000,
                        help='Batch size for add_documents (default: 1000)')

    # query
    query_p = sub.add_parser('query', help='Query the SA graph')
    query_p.add_argument('--graph', required=True, help='Path to graph pickle')
    query_p.add_argument('--top-k', type=int, default=10)
    query_p.add_argument('--context', help='Comma-separated context entities')
    query_p.add_argument('query_text', nargs='+')

    # stats
    stats_p = sub.add_parser('stats', help='Show graph statistics')
    stats_p.add_argument('--graph', required=True, help='Path to graph pickle')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {'build': cmd_build, 'query': cmd_query, 'stats': cmd_stats}[args.command](args)


if __name__ == '__main__':
    main()
