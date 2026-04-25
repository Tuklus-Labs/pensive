# Boundary Evaluation Checklist

Goal: decide whether boundary-analysis signals in Pensive are worth keeping,
calibrating, and eventually exporting to NSSIO.

## Hypotheses

1. Low `boundary_distance` and/or tiny `disambiguation_gap` correlate with
   retrievals that should not be trusted without more context.
2. `context_needed` is a useful gate for "ask for clarification" behavior.
3. `band_crossing` is explanatory, but not sufficient on its own to mark a
   result as bad.
4. Explicit context should convert at least some low-confidence queries into
   reliable hits.

## What To Measure

- Top-1 accuracy on answerable cases
- Unreliable-case recall
  Definition: among cases we marked as inherently unreliable, how often does
  the heuristic flag them as unreliable?
- Unreliable-case precision
  Definition: when the heuristic says "don't trust this," how often is that
  actually true?
- Reliable-case false-positive rate
  Definition: how often do we over-warn on queries that are actually fine?
- Context-case accuracy
  Definition: on cases where we provide disambiguating context, do we recover
  the expected answer?
- Accuracy by confidence bucket
  Definition: if the confidence buckets are meaningful, `high` should outperform
  `medium`, which should outperform `low`.

## Benchmark Cases

The synthetic benchmark in `bench_boundary.py` includes:

- ambiguous no-context latency query
- the same ambiguous query with explicit context
- clear fact retrieval
- clear named-entity retrieval
- clear query where band crossing can coexist with reliability
- nonsense / no-result query

These are not meant to prove production behavior. They are meant to answer a
smaller research question first: do the diagnostics move in the right direction
on representative failure modes?

## How To Run

```bash
python bench_boundary.py
python bench_boundary.py --json
python bench_boundary.py --graph /path/to/graph.pkl --cases /path/to/cases.json
python bench_boundary.py --graph /home/aegis/Projects/pensive/dell_ingest.pkl --cases /home/aegis/Projects/pensive/research/dell-real-graph-cases.json
pytest -q tests/test_boundary.py tests/test_boundary_bench.py
```

Case-file schema for `--cases`:

```json
[
  {
    "case_id": "ambiguous-latency",
    "query": "What was the P99 latency on 2025-07-16?",
    "expected_top": "199ms",
    "expect_reliable": true,
    "context": ["199ms"],
    "notes": "Optional free-form note"
  }
]
```

## Decision Gates For Advancing To NSSIO

Move from A to B only if the benchmark and real-query sampling show:

- `context_needed` reliably lights up on ambiguous queries
- low-confidence cases are materially worse than medium/high cases
- contextual retries actually recover good answers often enough to matter
- `band_crossing` adds interpretability without becoming the main decision rule

If those gates fail, keep the work in Pensive as optional diagnostics only and
do not build NSSIO policy around it yet.

## Near-Term Follow-Ups

1. Add a small real-query eval set from the actual corpus, not just synthetic docs.
2. Compare multiple distrust heuristics:
   - `confidence == "low"`
   - `context_needed`
   - `boundary_distance < x`
   - combinations of the above
3. Log suggested-context usefulness:
   does feeding one of the suggested entities actually recover the target answer?
4. Only after that, export the raw signals into NSSIO's reflective loop.
