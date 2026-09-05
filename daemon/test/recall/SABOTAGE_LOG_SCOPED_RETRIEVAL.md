# Sabotage Log: scoped recall candidate generation

The production mutations were loaded through the temporary
`work/scoped_mutation_plugin.py` pytest plugin. Every production mutant failed
the named test. Replacing that test's body with a no-op in the same run made the
mutant pass, demonstrating that the shipped assertions caused the failure.
The complete matrix was rerun after the scoped-retrieval review on 2026-09-05:
all 14 production runs exited 1 and all 14 weakened-test runs exited 0.

| Test | Production mutation and observed failure | Test mutation and observation |
|------|------------------------------------------|-------------------------------|
| `test_project_scope_precedes_bm25_cutoff` | Removed the project predicate from the engine's BM25 call. Failed with the scoped target absent beyond 200 distractors. | Removed the test body; mutant passed. |
| `test_agent_scope_precedes_dense_cutoff` | Replaced the combined allowed set with `None`. Failed with the agent target absent beyond 200 vectors. | Removed the test body; mutant passed. |
| `test_effective_time_scope_precedes_dense_cutoff_and_is_inclusive` | Replaced the combined allowed set with `None`. Failed with both boundary targets absent. | Removed the test body; mutant passed. |
| `test_explicit_narrative_kind_precedes_dense_cutoff` | Replaced the combined allowed set with `None`. Failed with the narrative target absent. | Removed the test body; mutant passed. |
| `test_explicit_narrative_kind_precedes_bm25_cutoff` | Widened the engine's BM25 kind argument from `narrative` to the full memory class. Failed with the narrative target absent beyond 200 atom distractors. | Removed the test body; mutant passed. |
| `test_combined_scope_intersects_all_constraints` | Replaced the combined allowed set with `None`. Failed with the combined target absent beyond 200 vectors. | Removed the test body; mutant passed. |
| `test_combined_scope_precedes_bm25_cutoff` | Removed project, agent, and time predicates from BM25. Failed with the combined target absent beyond 200 lexical distractors. | Removed the test body; mutant passed. |
| `test_empty_combined_scope_skips_both_embedders` | Converted an empty allowed set to unscoped. Failed loudly on the now-reached auxiliary candidate, before the call-count assertion. | Removed the test body; mutant passed. |
| `test_scoped_flat_and_hnsw_agree_and_exclude_retired_rows` | Independently made Flat and HNSW ignore `allowedIds`. Each run failed on the disallowed best vector entering results. | Removed the test body under the Flat mutant; mutant passed. |
| `test_l3_embeds_base_query_once_across_two_classes` | Disabled the one-query embedding cache. Failed with two base embed calls for two classes. | Removed the test body; mutant passed. |
| `test_l2_full_class_keeps_old_duck_index_call_shape` | Forced a whole-class L2 request through the scoped call. Failed because the old two-argument duck index received `allowedIds`. | Removed the test body; mutant passed. |
| `test_l2_full_class_skips_out_of_scope_aux_indexes` | Removed the active-class selection from auxiliary search. Failed because the code index was searched once during an L2 memory recall. | Removed the test body; mutant passed. |
| `test_aux_dense_scopes_before_top_k_and_embeds_once` | Dropped `allowedIds` from auxiliary searches. Failed with 400 distractors instead of the two allowed targets. | Removed the test body; mutant passed. |

Commands used the form:

```text
PENSIVE_SCOPED_MUTATION=<mutation> PYTHONPATH=daemon/src:.../work \
  pytest -q -p scoped_mutation_plugin \
  daemon/test/recall/test_scoped_retrieval.py::<test>
```

The corresponding assertion mutation added
`PENSIVE_WEAKEN_TEST=<test>`. Production runs exited 1; weakened-test runs exited
0. The temporary plugin is outside the repository and is not part of the diff.

`mutmut` was also attempted for the wider daemon campaign, but it crashes while
re-importing NumPy under CPython 3.14. The targeted plugin mutations above are the
executed fallback; no result is inferred from the generic runner failure.
