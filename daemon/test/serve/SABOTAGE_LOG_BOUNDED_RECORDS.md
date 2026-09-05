# Sabotage Log: bounded structured recall records

Source and assertion mutations ran in the isolated snapshot
`/tmp/pensive-bounded-mutants.KWqnHR/repo`. The canonical worktree was not
mutated. Every run used the CPU-only targeted command:

```text
HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
/home/aegis/Documents/Codex/2026-09-05/fi/work/text-test-runtime/bin/python \
-m pytest -q <exact selected test> --tb=short
```

After reversing all mutations, the isolated bounded selection reported `10
passed, 56 deselected in 0.52s`.

## Source mutations

| Mutation | Prediction | Observed result | Conclusion |
|---|---|---|---|
| Remove `provenanceLimit=65` from the handler's `getAtom` call. | The 1,000-row case detects an eager unbounded fetch. | 1 failed, 2 passed; `sizes=[1000]` violated the 65-row read bound. | The test observes the store-side resource bound, not only the 64-row output slice. |
| Suppress `provenanceTruncated` on records with more than 64 rows. | The 65-row case loses explicit disclosure. | 1 failed; marker was absent while 64 rows were served. | Omitted history cannot be silent. |
| Build each candidate's envelope flag from only the current record. | A complete second result clears an earlier truncation. | 1 failed; two records were served with envelope `truncated=false`. | Truncation state must accumulate across admitted records. |
| Add `provenanceTruncated=true` to every record. | The exact-64 ordinary-shape control fails. | 1 failed; the optional marker appeared with no omitted row. | The additive field is emitted only when needed. |
| Replace the final empty-body fallback with `[body omitted]`. | A 1-token budget is exceeded. | 1 failed; the returned notice cost 5 tokens under budget 1. | The degradation ladder must be allowed to choose an empty body while retaining ID and cost metadata. |
| Replace full-body `record.estimatedTokens` with the compact notice cost. | Both normal and maximum-body starvation tests lose retry sizing. | 2 failed; costs became 9 instead of 20 and 45 instead of 10,667. | Record metadata carries full body cost; envelope accounting carries returned notice cost. |

## Assertion mutations

| Mutation | Prediction | Observed result | Conclusion |
|---|---|---|---|
| Change the marker boundary expectation from `total > 64` to `total >= 64`. | The exact-64 control fails. | 1 failed; no marker was present at 64. | The first omitted row boundary is pinned precisely. |
| Flip the tiny-budget assertion from `estimatedTokens <= budget` to `> budget`. | The 1-token control fails. | 1 failed; actual returned-body cost was 0. | The body-budget assertion executes and constrains the returned notice. |

## Defect found during review

A valid 32,000-character emit body has an estimated full cost of 10,667. The
initial patch preserved that cost in the stub but retained an 8,000 maximum in
the record schema, causing a schema error. The new regression failed RED at
`records.0.estimatedTokens: validator=maximum`. Record-level cost now permits the
safe JSON integer maximum; the envelope body budget remains 8,000.

## Limits

This was a targeted campaign over the six requested source mutations and two
assertion mutations. It is not exhaustive mutation of `mcp.py` or all 136
assertions in the structured test file. Generic-run limitations for this
campaign are recorded in `../recall/SABOTAGE_LOG_TRUST_DELIVERY.md`; this file
claims only its executed targeted mutations. Earlier wording attributed every
generic-run failure to NumPy, which was too broad.
No live database, service, GPU, or production index was used.
