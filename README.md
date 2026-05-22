# Pensive

Spreading activation retrieval for document collections. Sub-millisecond queries at 50M+ documents.

Pensive builds a sparse entity graph from your documents using regex-based extraction, then retrieves answers via spreading activation -- a biologically-inspired algorithm where query terms "light up" connected entities and the activation spreads to relevant answers.

## Install

```bash
pip install pypensive            # Core SA engine (numpy + scipy only)
pip install pypensive[full]      # + L2 semantic search, BM25, hybrid retrieval
```

## Quickstart

```python
from pensive import SpreadingActivation

sa = SpreadingActivation()
sa.build([
    {'id': '1', 'content': 'The P99 latency was 42ms on 2025-10-08', 'value': '42ms on 2025-10-08'},
    {'id': '2', 'content': 'Build 1234 completed in 320s with rss 18GB', 'value': '320s build, 18GB rss'},
    {'id': '3', 'content': 'Meeting with Sarah Chen about Project Atlas budget', 'value': 'Atlas budget meeting'},
])

# Queries are entity-exact: pass the entity surface form, NOT a natural-
# language question. Pensive extracts entities (metrics, dates, IDs,
# people, projects, ...) via regex and matches the query against those
# extracted entities. A query like "42ms" returns answers connected to
# that entity in the graph; a query like "What was the P99 latency?"
# returns [] because none of those query tokens are recognized entities.
results = sa.query("42ms")
# [('42ms on 2025-10-08', 4.5)]

results = sa.query("sarah chen")
# [('Atlas budget meeting', 2.4)]

results = sa.query("2025-10-08")
# [('42ms on 2025-10-08', 4.5)]
```

### Natural-language queries

`sa.query()` does NOT parse natural language. If you need to map a
question like "What was the P99 latency?" to entities, run your own
extractor first and pass the extracted entity to `sa.query()`. The
`pensive.mega_extract.MegaExtractor` class is the same one Pensive uses
internally and accepts any custom pattern set.

```python
from pensive import SpreadingActivation
from pensive.mega_extract import MegaExtractor
from pensive.patterns import REAL_DATA_PATTERNS

extractor = MegaExtractor(REAL_DATA_PATTERNS)
question = "What was the latency on 2025-10-08?"
for entity, _etype in extractor.extract(question):
    hits = sa.query(entity)
    if hits:
        print(entity, '->', hits)
```

## Parallel Build (large corpora)

```python
# 14x faster at 1M docs using multiprocessing
sa = SpreadingActivation()
sa.build_parallel(documents, workers=8)
```

## Ingestion from Data Exports

```python
from pensive.ingestion import IngestPipeline
from pensive.ingestion.parsers.chatgpt import ChatGPTParser
from pensive.ingestion.parsers.facebook import FacebookParser

pipe = IngestPipeline()
pipe.ingest_all([
    ChatGPTParser("/path/to/chatgpt-export/"),
    FacebookParser("/path/to/facebook-export/"),
])

# Query (entity-exact -- see Quickstart on extracting entities from
# natural-language questions before calling query()).
results = pipe.sa.query("2025-10-08")
results = pipe.sa.query("project atlas")

# Save/load
pipe.save_graph("my_graph.pkl")
pipe = IngestPipeline.load_graph("my_graph.pkl")
```

## CLI

```bash
pensive build --chatgpt ~/chatgpt-export/ --facebook ~/fb-export/ -o graph.pkl
# Queries are entity-exact -- pass the entity itself, not a natural-
# language question. For NL questions, run MegaExtractor over the
# question first (see Quickstart) and feed the extracted entity here.
pensive query --graph graph.pkl "2025-10-08"
pensive stats --graph graph.pkl
```

## Document Format

Each document is a dict with:
- `id` (str): Unique identifier
- `content` (str): Text to extract entities from
- `value` (str): The answer/snippet to retrieve
- `query` (str, optional): Additional text for entity extraction

## Configuration

```python
from pensive import SpreadingActivation, SpreadingConfig

sa = SpreadingActivation(config=SpreadingConfig(
    max_hops=2,        # Spreading depth (default: 4)
    max_active=200,    # Max active nodes per hop (default: 50)
    decay=0.6,         # Activation decay per hop (default: 0.6)
    threshold=0.15,    # Min activation to keep spreading (default: 0.15)
))
```

## Contextual Disambiguation

Provide conversation context to disambiguate queries. The query is
still an entity (Pensive does not parse natural language); context
boosts candidates that also activate from the context entities.

```python
# Two docs mention "199ms" and "199gb" respectively. A bare query for
# the project entity returns both; context biases toward the GPU one.
sa.build([
    {'id': '1', 'content': 'GPU memory bandwidth was 199GB on Project Atlas',
     'value': 'Atlas GPU 199GB bandwidth'},
    {'id': '2', 'content': 'Project Atlas API P99 latency was 199ms',
     'value': 'Atlas API 199ms latency'},
])

results = sa.query(
    "atlas",
    context=["GPU", "training run"]  # Steers toward the GPU/bandwidth doc
)
```

## Boundary Analysis

Inspect whether a query is close to the activation boundary, mixes rare and
common entities, or should ask the caller for more context:

```python
# Entity-exact query, same as sa.query() -- the analyzer wraps the
# regular query path with a diagnostic envelope. A natural-language
# input like "What was the P99 latency on 2025-07-16?" lands on
# confidence='none', should_trust=False because none of those tokens
# are entities. Use an extracted entity instead.
diagnosed = sa.query_analyzed("2025-07-16")

print(diagnosed.analysis.confidence)         # "low", "medium", "high", or "none"
print(diagnosed.analysis.should_trust)       # False when retrieval looks unreliable
print(diagnosed.analysis.recommended_action) # "trust", "request_context", "no_result", ...
print(diagnosed.analysis.boundary_distance)  # score - threshold
print(diagnosed.analysis.context_needed)     # True when SA sees ambiguity
print(diagnosed.analysis.suggested_context)  # e.g. ["199ms", "257ms"]
```

From the CLI (still entity-exact):

```bash
pensive query --graph graph.pkl --analyze "2025-07-16"
```

## Hybrid Retrieval (`pypensive[full]`)

Default flow is two-stage retrieval:
1. L1 SA generates fast candidate IDs.
2. L2 FAISS reranks those L1 hits semantically.

If L1 returns nothing, hybrid can fall back to global L2 search.

```python
from pensive import SpreadingActivation
from pensive.l2 import L2Handler
from pensive.parallel_hybrid import ParallelHybrid

sa = SpreadingActivation()
sa.build(documents)

l2 = L2Handler()  # defaults to all-MiniLM-L6-v2
l2.add_documents(documents)

hybrid = ParallelHybrid(
    spreading_activation=sa,
    l2_handler=l2,
    l2_on_sa_hits=True,       # default
    l2_fallback_global=True,  # default
    l2_fallback_on_low_confidence=True,  # Escalate ambiguous SA queries to global L2
)
results = hybrid.query("What was the P99 latency on 2025-07-16?")

for r in results:
    print(f"{r.doc_id}: {r.summary} (score={r.score:.1f}, source={r.source})")
```

### BM25 Sparse Search

For keyword/identifier matching without embeddings:

```python
from pensive.hybrid_search import BM25Index

idx = BM25Index()
idx.add_documents(documents)
results = idx.search("error 0x4F2A")  # Exact identifier matching
```

### What `[full]` adds

| Component | Purpose | Dependency |
|-----------|---------|------------|
| L2Handler | Semantic vector search | sentence-transformers, faiss-cpu |
| BM25Index | Sparse keyword matching | rank-bm25 |
| ParallelHybrid | SA + L2 agreement boosting | (uses both above) |
| Cross-encoder reranking | Optional reranker | sentence-transformers |

## Scale Characteristics

| Scale | Query Latency | Peak RSS | Build Time (parallel) |
|-------|--------------|----------|----------------------|
| 1M docs | ~1ms | 13 GB | ~12s |
| 5M docs | ~0.4ms | 16 GB | ~5 min |
| 10M docs | ~0.4ms | 30 GB | ~10 min |
| 50M docs | ~0.45ms | 139 GB | ~28 min |

### Long-lived processes: structural memory growth

The graph is append-only: `add_documents()` extends `_idx_to_node`,
`_node_to_idx`, `_node_label`, and the COO edge buffers without
compacting. A pure-query workload has flat RSS, but a daemon that
indefinitely interleaves `add_documents()` with queries grows RSS
roughly linearly with corpus size. Pass-7 measured the per-node cost
at ~50 bytes (Python dict + list slot overhead, not counting edge
buffers), so 1M incrementally-added nodes adds ~50 MB on top of the
adjacency matrix.

This is not a leak; it is the structural cost of an append-only graph
with no eviction or compaction path. For very-long-lived processes
that ingest forever, the recommended pattern is to periodically
serialize the graph with `get_save_data()`, discard the old
`SpreadingActivation` instance, and rebuild from the save -- which
also defragments the underlying Python data structures. A dedicated
`compact()` method is on the roadmap; track in the issue tracker.

## License

MIT
