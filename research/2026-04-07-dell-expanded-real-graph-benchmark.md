# Dell Expanded Real-Graph Boundary Benchmark

Date: 2026-04-07

Graph:
`/home/aegis/Projects/pensive/dell_ingest.pkl`

Candidate mining output:
`/home/aegis/Projects/pensive/research/dell-real-graph-mined-candidates.json`

Expanded cases:
`/home/aegis/Projects/pensive/research/dell-real-graph-expanded-cases.json`

Mining command:

```bash
python bench_boundary.py \
  --graph /home/aegis/Projects/pensive/dell_ingest.pkl \
  --mine-candidates \
  --max-queries 20 \
  --max-candidates 10 \
  --json
```

Benchmark command:

```bash
python bench_boundary.py \
  --graph /home/aegis/Projects/pensive/dell_ingest.pkl \
  --cases /home/aegis/Projects/pensive/research/dell-real-graph-expanded-cases.json \
  --json
```

## Summary

- Case count: 12
- Answerable cases: 6
- Top-1 accuracy on answerable cases: 1.00
- Unreliable recall: 1.00
- Unreliable precision: 1.00
- Reliable false-positive rate: 0.00
- Context-case accuracy: 1.00
- Band-crossing cases: 11

## What Validated

1. The original `vendor-a` result still validates on the larger run:
   - no-context query is `low` confidence and `request_context`
   - two different subject contexts recover trustworthy results
2. The same pattern generalizes to other high-frequency boilerplate queries:
   - `privacy policy`
   - `read more`
   - `united states`
   - `free shipping`
3. The recovery is not limited to one behavior:
   - sometimes context keeps the same top hit but upgrades trust
   - sometimes context changes the top hit and upgrades trust
4. `no_result` still behaves cleanly on the real graph for `vendor-b`

## Research Findings

1. Boundary analysis is doing useful triage work on the saved graph, not just on synthetic examples.
2. The most actionable signals are still:
   - `context_needed`
   - `disambiguation_gap`
   - `no_result`
3. `band_crossing` remains explanatory rather than decisive. It fired on 11 of 12 cases, including every trustworthy recovery.
4. High-frequency boilerplate phrases are a productive source of real ambiguous queries for evaluation.

## Caveat

These new cases are stronger than the original 4-case smoke test, but they are
still only semi-curated. The extra candidates were mined from Pensive's own
`suggested_context` output and then hand-selected into the expanded case file.
That makes this a good internal validation step, not an externally labeled
ground-truth benchmark yet.

## Implication For Next Steps

- keep boundary diagnostics in Pensive
- keep `should_trust` and `recommended_action` as the main standalone API surface
- continue treating `band_crossing` as metadata
- build a more independent real-world evaluation set before promoting these signals into NSSIO policy
