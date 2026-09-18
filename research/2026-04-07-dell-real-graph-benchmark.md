# Dell Real-Graph Boundary Benchmark

> **Note:** the graph and case files referenced below are not distributed with
> this repository. They were derived from a private personal corpus, so the
> fixtures stay local and the query terms appear here as `vendor-a` and
> `vendor-b`. The methodology and the numbers reproduce against any corpus
> you build yourself with `pensive ingest`.

Date: 2026-04-07

Graph:
`~/Projects/pensive/dell_ingest.pkl`

Cases:
`~/Projects/pensive/research/dell-real-graph-cases.json`

Run command:

```bash
python bench_boundary.py \
  --graph ~/Projects/pensive/dell_ingest.pkl \
  --cases ~/Projects/pensive/research/dell-real-graph-cases.json \
  --json
```

## Summary

- Case count: 4
- Answerable cases: 2
- Top-1 accuracy on answerable cases: 1.00
- Unreliable recall: 1.00
- Unreliable precision: 1.00
- Reliable false-positive rate: 0.00
- Context-case accuracy: 1.00

## What Validated

1. A real ambiguous query (`vendor-a`) is flagged as:
   - `confidence = low`
   - `should_trust = false`
   - `recommended_action = request_context`
2. Adding real subject context recovers the intended chunk:
   - `vendor-a + [black, friday, hidden, gems]`
   - `vendor-a + [custom, boot]`
3. A real no-hit query (`vendor-b`) maps cleanly to:
   - `confidence = none`
   - `recommended_action = no_result`

## Research Findings

1. The trust/action layer is doing real work on the saved graph. The same base
   query can move from `request_context` to `trust` once the caller supplies
   disambiguating context.
2. `band_crossing` is not a standalone failure signal on this graph. It fired
   on 3 of the 4 cases, including both reliable context-resolved wins.
3. The strongest signals on the real graph were:
   - `context_needed`
   - `disambiguation_gap`
   - `no_result`

## Implication For Next Steps

This is enough validation to keep pushing A:

- keep exposing raw signals from Pensive
- prefer `should_trust` / `recommended_action` as the standalone API surface
- treat `band_crossing` as explanatory metadata, not a primary decision rule
- expand the real-case file with more manually curated examples before wiring
  anything stronger into NSSIO policy
