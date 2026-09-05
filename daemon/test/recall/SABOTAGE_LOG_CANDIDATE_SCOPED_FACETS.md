# Sabotage Log: candidate-scoped facet boosts

Each mutant used a private copy of `daemon/src` under
`work/campaign-two/candidate-mutants`; production files were never mutated in
place. The named regression test was run alone against the mutant. Each
production mutant exited 1. Replacing that test body with a no-op through
`work/campaign-two/candidate_weaken_plugin.py` made the same mutant exit 0, so
the shipped assertion caused the failure.

An initial mutation that disabled facet boosting passed the first version of the
frozen-clock test. That exposed a real test hole: both candidate and reference
arms shared the broken boost stage. The test was strengthened to require the
matching lower-ranked candidate to move first and to carry `facet match` trust
evidence; the same mutant then failed.

| Test | Production mutation | Observation |
|---|---|---|
| `test_candidate_scope_equals_global_intersection_for_live_candidates` | Ignored `candidateIds` and ran the global label lookup. | Failed because the matching noncandidate entered `boostSet`; weakened test passed. |
| `test_candidate_scope_equals_global_intersection_for_live_candidates` | Removed the canonical live-status predicate from the candidate query. | Failed because the superseded candidate entered `boostSet`; weakened test passed. |
| `test_omitted_candidate_scope_preserves_global_facet_contract` | Treated omitted/None candidate scope as an empty tuple. | Failed because the legacy global match disappeared; weakened test passed. |
| `test_empty_candidate_scope_skips_entity_extraction` | Removed the explicit-empty guard before entity extraction. | Failed on the extractor bomb; weakened test passed. |
| `test_candidate_scope_forces_atom_first_facet_index` | Forced `(key,value,atom_id)` instead of the atom-first facets primary key. | Failed with the executed SQL in the assertion message; weakened test passed. |
| `test_recall_candidate_scoped_facets_match_frozen_clock_reference` | Removed `candidateIds` from the engine callsite. | Failed because the facet spy saw no fused candidate pool; weakened test passed. |
| `test_recall_candidate_scoped_facets_match_frozen_clock_reference` | Disabled the post-fusion facet score boost. | Failed because the lower-ranked matching candidate stayed second; weakened test passed. |
| `test_empty_fused_pool_skips_facet_work` | Removed the post-RRF empty-pool short circuit. | Failed on the facet bomb before a result could be assembled; weakened test passed. |

All mutation runs used the CPU-only text runtime with
`HIP_VISIBLE_DEVICES=-1`, `ROCR_VISIBLE_DEVICES=-1`, and
`CUDA_VISIBLE_DEVICES=-1`.

