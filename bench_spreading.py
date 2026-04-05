"""Benchmark for spreading activation hot paths.

Measures:
- Entity extraction (MegaExtractor)
- Graph build (batch and parallel)
- Query latency (single, burst)
- _spread() inner loop
- _seed_from_words()
- _collect_results()
- Serialization round-trip
"""
import time
import random
import string
import sys
import numpy as np
from pensive import SpreadingActivation, SpreadingConfig
from pensive.mega_extract import MegaExtractor
from pensive.patterns import REAL_DATA_PATTERNS, SYNTHETIC_PATTERNS


def generate_docs(n, seed=42):
    """Generate n synthetic documents with realistic entity density."""
    rng = random.Random(seed)
    techs = ["PyTorch", "TensorFlow", "ROCm", "CUDA", "Triton", "FAISS",
             "numpy", "scipy", "Flask", "FastAPI", "Redis", "PostgreSQL"]
    people = ["Dr. Smith", "Dr. Chen", "Dr. Garcia", "Dr. Patel", "Dr. Kim",
              "Prof. Johnson", "Prof. Williams", "Prof. Brown"]
    metrics = ["latency", "throughput", "accuracy", "loss", "VRAM usage",
               "temperature", "bandwidth", "error rate"]
    projects = ["AEGIS", "Pensive", "Engram", "Cortex", "Kennel", "Panoptes",
                "Nemequ", "Hearth", "Warden", "Colosseum"]

    docs = []
    for i in range(n):
        date = f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        tech = rng.choice(techs)
        person = rng.choice(people)
        metric = rng.choice(metrics)
        project = rng.choice(projects)
        value_num = rng.randint(1, 9999)
        unit = rng.choice(["ms", "tok/s", "GB", "%", "C", "MB/s"])

        content = (
            f"{person} reported that {project} {metric} on {date} was "
            f"{value_num}{unit} using {tech}. "
            f"The v{rng.randint(1,5)}.{rng.randint(0,9)} release showed "
            f"{'improvement' if rng.random() > 0.5 else 'regression'} "
            f"compared to the previous build."
        )
        query = f"What was the {project} {metric} on {date}?"
        value = f"{value_num}{unit}"

        docs.append({
            'content': content,
            'id': f'doc_{i}',
            'value': value,
            'query': query,
        })
    return docs


def bench_extraction(docs, patterns):
    """Benchmark entity extraction."""
    ext = MegaExtractor(patterns)
    texts = [d['content'] + ' ' + d.get('query', '') for d in docs]

    t0 = time.perf_counter()
    total_entities = 0
    for text in texts:
        entities = ext.extract(text)
        total_entities += len(entities)
    elapsed = time.perf_counter() - t0

    return {
        'docs': len(docs),
        'total_entities': total_entities,
        'time_s': elapsed,
        'docs_per_sec': len(docs) / elapsed,
        'entities_per_doc': total_entities / len(docs),
    }


def bench_build(docs, patterns):
    """Benchmark graph build."""
    sa = SpreadingActivation(patterns=patterns)
    t0 = time.perf_counter()
    sa.build(docs)
    elapsed = time.perf_counter() - t0
    stats = sa.stats()
    return {
        'time_s': elapsed,
        'docs_per_sec': len(docs) / elapsed,
        **stats,
    }, sa


def bench_query(sa, queries, top_k=10):
    """Benchmark query latency."""
    # Warmup
    for q in queries[:3]:
        sa.query(q, top_k=top_k)

    times = []
    for q in queries:
        t0 = time.perf_counter()
        results = sa.query(q, top_k=top_k)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    times_us = [t * 1e6 for t in times]
    return {
        'n_queries': len(queries),
        'p50_us': np.percentile(times_us, 50),
        'p95_us': np.percentile(times_us, 95),
        'p99_us': np.percentile(times_us, 99),
        'mean_us': np.mean(times_us),
        'min_us': np.min(times_us),
        'max_us': np.max(times_us),
        'qps': len(queries) / sum(times),
    }


def bench_spread_isolated(sa, n_trials=100):
    """Benchmark just the _spread() method."""
    words_list = [
        ["pytorch", "latency", "2025-03-15"],
        ["aegis", "throughput", "rocm"],
        ["temperature", "gpu", "chen"],
        ["nemequ", "accuracy", "triton"],
        ["pensive", "bandwidth", "2025-07-16"],
    ]

    times = []
    for i in range(n_trials):
        words = words_list[i % len(words_list)]
        activations = sa._seed_from_words(words)
        t0 = time.perf_counter()
        sa._spread(activations)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    times_us = [t * 1e6 for t in times]
    return {
        'n_trials': n_trials,
        'p50_us': np.percentile(times_us, 50),
        'p95_us': np.percentile(times_us, 95),
        'mean_us': np.mean(times_us),
    }


def bench_seed_isolated(sa, n_trials=100):
    """Benchmark just _seed_from_words()."""
    words_list = [
        ["pytorch", "latency", "2025-03-15"],
        ["aegis", "throughput", "rocm"],
        ["temperature", "gpu", "chen"],
        ["nemequ", "accuracy", "triton"],
        ["pensive", "bandwidth", "2025-07-16"],
    ]

    times = []
    for i in range(n_trials):
        words = words_list[i % len(words_list)]
        t0 = time.perf_counter()
        sa._seed_from_words(words)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    times_us = [t * 1e6 for t in times]
    return {
        'n_trials': n_trials,
        'p50_us': np.percentile(times_us, 50),
        'p95_us': np.percentile(times_us, 95),
        'mean_us': np.mean(times_us),
    }


def bench_collect_results(sa, n_trials=100):
    """Benchmark _collect_results()."""
    words = ["pytorch", "latency", "2025-03-15"]
    activations = sa._seed_from_words(words)
    activations = sa._spread(activations)

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        sa._collect_results(activations, top_k=10)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    times_us = [t * 1e6 for t in times]
    return {
        'n_trials': n_trials,
        'activation_size': len(activations),
        'p50_us': np.percentile(times_us, 50),
        'p95_us': np.percentile(times_us, 95),
        'mean_us': np.mean(times_us),
    }


def bench_serialization(sa):
    """Benchmark save/load round-trip."""
    t0 = time.perf_counter()
    data = sa.get_save_data()
    save_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sa2 = SpreadingActivation.from_save_data(data)
    load_time = time.perf_counter() - t0

    return {
        'save_s': save_time,
        'load_s': load_time,
    }


def run_suite(n_docs, patterns=REAL_DATA_PATTERNS):
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {n_docs:,} documents")
    print(f"{'='*60}")

    docs = generate_docs(n_docs)
    queries = [d['query'] for d in docs[:min(200, n_docs)]]

    # Extraction
    ext_result = bench_extraction(docs, patterns)
    print(f"\n  Extraction:")
    print(f"    Time: {ext_result['time_s']:.3f}s ({ext_result['docs_per_sec']:.0f} docs/s)")
    print(f"    Entities/doc: {ext_result['entities_per_doc']:.1f}")

    # Build
    build_result, sa = bench_build(docs, patterns)
    print(f"\n  Build:")
    print(f"    Time: {build_result['time_s']:.3f}s ({build_result['docs_per_sec']:.0f} docs/s)")
    print(f"    Nodes: {build_result['nodes']:,} ({build_result['entity_nodes']:,} entity, {build_result['value_nodes']:,} value)")
    print(f"    Edges: {build_result['edges']:,}")
    print(f"    Unique entities: {build_result['unique_entities']:,}")

    # Query
    query_result = bench_query(sa, queries)
    print(f"\n  Query ({query_result['n_queries']} queries):")
    print(f"    p50: {query_result['p50_us']:.1f}us")
    print(f"    p95: {query_result['p95_us']:.1f}us")
    print(f"    p99: {query_result['p99_us']:.1f}us")
    print(f"    QPS: {query_result['qps']:.0f}")

    # Spread isolated
    spread_result = bench_spread_isolated(sa)
    print(f"\n  _spread() isolated:")
    print(f"    p50: {spread_result['p50_us']:.1f}us")
    print(f"    p95: {spread_result['p95_us']:.1f}us")

    # Seed isolated
    seed_result = bench_seed_isolated(sa)
    print(f"\n  _seed_from_words() isolated:")
    print(f"    p50: {seed_result['p50_us']:.1f}us")
    print(f"    p95: {seed_result['p95_us']:.1f}us")

    # Collect results
    collect_result = bench_collect_results(sa)
    print(f"\n  _collect_results() isolated:")
    print(f"    activation_size: {collect_result['activation_size']}")
    print(f"    p50: {collect_result['p50_us']:.1f}us")
    print(f"    p95: {collect_result['p95_us']:.1f}us")

    # Serialization
    ser_result = bench_serialization(sa)
    print(f"\n  Serialization:")
    print(f"    Save: {ser_result['save_s']:.3f}s")
    print(f"    Load: {ser_result['load_s']:.3f}s")

    return {
        'n_docs': n_docs,
        'extraction': ext_result,
        'build': build_result,
        'query': query_result,
        'spread': spread_result,
        'seed': seed_result,
        'collect': collect_result,
        'serialization': ser_result,
    }


if __name__ == '__main__':
    sizes = [100, 1_000, 10_000]
    if '--large' in sys.argv:
        sizes.append(100_000)

    results = {}
    for n in sizes:
        results[n] = run_suite(n)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Docs':>10} | {'Build':>10} | {'Query p50':>10} | {'QPS':>10} | {'Spread p50':>10}")
    print(f"{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for n, r in results.items():
        print(f"{n:>10,} | {r['build']['time_s']:>9.3f}s | {r['query']['p50_us']:>8.1f}us | {r['query']['qps']:>10,.0f} | {r['spread']['p50_us']:>8.1f}us")
