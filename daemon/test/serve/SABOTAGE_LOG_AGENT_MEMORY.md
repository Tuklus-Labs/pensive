# Sabotage Log: atomic corrections, fork history, and native handles

All mutations below were applied to the working tree, run with the isolated CPU
runtime, and reversed before the final suite. The targeted runner was:

```text
HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
/home/aegis/Documents/Codex/2026-09-05/fi/work/text-test-runtime/bin/python \
-m pytest -q daemon/test/serve/test_agent_memory_corrections.py -k '<selector>'
```

The restored correction suite passed 39 cases. `git diff --check` was clean and
the four scoped files were searched for mutation markers before final testing.

## Production mutations

| Mutation | Prediction | Observed result | Conclusion |
|---|---|---|---|
| Replace retained successor importance with `0.0`. | Metadata and reopen persistence tests fail. | 2 failed: both importance assertions reported `0.0` instead of `0.75` / `0.625`. | Importance continuity is directly constrained. |
| Copy `entity` along with `pin` and `tag`. | The selected-facet assertion fails. | 1 failed with the content-derived entity present on the successor. | Content-derived entities cannot silently become claims about replacement text. |
| Remove both stale-status guards: bypass the prewrite status check and remove the live predicate from the update. | Stale targets commit and two concurrent corrections both win. | 3 failed: superseded and tombstone cases did not raise; concurrent outcomes contained two successor IDs. | The transaction lock and both live-state checks enforce single-successor correction. |
| Omit incoming successor edges while discovering a history component. | History invoked from a branch leaf misses its sibling. | 1 failed; the sibling handle was absent. | Traversal must cross oldward to a fork and newward down every branch. |
| Continue to incremental retirement after a successful add-failure rebuild. | Recovery makes redundant calls and can fail after the index is already repaired. | 1 failed; calls were `index, rebuild, retire, rebuild` instead of `index, rebuild`. | A complete class rebuild ends recovery. |
| Swallow incremental retirement failure instead of rebuilding. | Recovery-call and committed-error tests fail. | 2 failed; rebuild was absent and the double-failure case incorrectly returned success. | Remove failure must rebuild, and unrecovered committed state stays loud. |
| Keep the `p3://` prefix instead of stripping it in the shared normalizer. | The prefixed handler path fails before a correction edge exists. | 1 failed. The first audit exposed an incidental `IndexError`; after assertion ordering was fixed, the named normalization assertion guards this path. | Displayed handles must round-trip through all native handlers. |
| Raise the fork threshold from two unique successors to three. | A two-branch legacy fork is reported healthy. | 2 failed; `forkedPredecessors` was empty and `ok` was true. | The first real fork is exactly two unique successors. |
| Commit instead of rolling back the correction exception path. | Mid- and late-transaction failure tests observe leaked canonical rows. | 2 failed; the mid failure left a second atom, and the late failure changed counts from `(1, 1, 0, 1)` to `(2, 3, 1, 2)`. | Rollback covers the complete successor/provenance/edge/facet/status unit. |

## Test mutation

| Mutation | Prediction | Observed result | Conclusion |
|---|---|---|---|
| Flip the stale-target zero-write assertion from `after == before` to `after != before`. | Both parameter cases fail at the named invariant. | 2 failed with identical before/after tuples. | The zero-write assertion executes and is not incidental to the exception check. |

## Limits

This is a targeted campaign over high-risk branches, not exhaustive mutation of
every byte or two mutations per each of the 39 parameterized cases. Generic-run
limitations are recorded in `../recall/SABOTAGE_LOG_TRUST_DELIVERY.md`; those
attempts do not measure this correction test set.
The campaign does not cover live daemon behavior, real HNSW/FAISS recovery,
service restart, or a production database. Those operations were outside scope.
