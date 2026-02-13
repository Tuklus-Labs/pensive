# Pensive

Spreading activation retrieval for document collections. Sub-millisecond queries at 50M+ documents.

Pensive builds a sparse entity graph from your documents using regex-based extraction, then retrieves answers via spreading activation -- a biologically-inspired algorithm where query terms "light up" connected entities and the activation spreads to relevant answers.

## Install

```bash
pip install pensive
```

## Quickstart

```python
from pensive import SpreadingActivation

sa = SpreadingActivation()
sa.build([
    {'id': '1', 'content': 'The P99 latency was 42ms on 2025-10-08', 'value': '42ms on 2025-10-08'},
    {'id': '2', 'content': 'GPU temp hit 82C during the training run', 'value': '82C during training'},
    {'id': '3', 'content': 'Meeting with Sarah Chen about Project Atlas budget', 'value': 'Atlas budget meeting'},
])

results = sa.query("What was the P99 latency?")
# [('42ms on 2025-10-08', 1.623), ...]
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

# Query
results = pipe.sa.query("What did we talk about last week?")

# Save/load
pipe.save_graph("my_graph.pkl")
pipe = IngestPipeline.load_graph("my_graph.pkl")
```

## CLI

```bash
pensive build --chatgpt ~/chatgpt-export/ --facebook ~/fb-export/ -o graph.pkl
pensive query --graph graph.pkl "What was the deployment date?"
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

Provide conversation context to disambiguate queries:

```python
results = sa.query(
    "What was the temperature?",
    context=["GPU", "training run"]  # Disambiguates toward GPU temp, not weather
)
```

## Scale Characteristics

| Scale | Query Latency | Peak RSS | Build Time (parallel) |
|-------|--------------|----------|----------------------|
| 1M docs | ~1ms | 13 GB | ~12s |
| 5M docs | ~0.4ms | 16 GB | ~5 min |
| 10M docs | ~0.4ms | 30 GB | ~10 min |
| 50M docs | ~0.45ms | 139 GB | ~28 min |

## License

MIT
