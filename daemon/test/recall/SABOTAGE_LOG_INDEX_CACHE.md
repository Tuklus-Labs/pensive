# Sabotage Log: bounded derived index snapshot cache

The focused test suite ran green before this audit. An actual `mutmut` run was
performed from `daemon/` with a temporary configuration targeting
`src/recall/index_cache.py` and `test/recall/test_index_cache.py`; the generated
`daemon/mutants/` checkout and configuration were removed afterward. The run
exercised 298 mutants: 220 were killed, 76 survived, and 2 timed out. Surviving mutations are
mostly defensive branches, canonicalization variants, and filesystem-error
branches not deterministically inducible by this small suite; the targeted
critical mutations below were independently run and killed.

| Test | Production mutation | Observation | Weakened-test observation |
|---|---|---|---|
| `test_load_checks_source_and_content_before_loader` | Replaced the filename/content SHA comparison with `if False`, allowing corrupted bytes to reach `loader`. | Failed because the corrupted snapshot returned an object and the loader-call list became non-empty. | After removing both the corruption-result and loader-order assertions, the same mutant passed; those assertions carry the hash-before-loader rule. |
| `test_retention_keeps_two_namespace_files_only` | Changed retention from `entries[2:]` to `entries[1:]`, deleting one of the two newest files. | Failed because only one same-namespace snapshot remained after three saves. | Removing the exact-two assertion let the mutant pass; that assertion names the inclusive retention boundary. |

The first production mutation and the corresponding test weakening were run
against the working tree with `pytest -q` on the named test. The `mutmut` run
was the broad critical-path mutation gate; these two rows are the explicit
targeted evidence for hash ordering and retention bounds.

Independent Daybreak review found corrupt files consuming retention slots. The
new regression observed loss of the second loadable snapshot before the fix.
The parent also reproduced publication disappearing after a clock rollback and
an invalid optional path escaping as TypeError. All three cases passed after
valid-only retention, explicit publication priority, and path-error containment.
A separate real HNSW regression proved that an injected snapshot-writer exception
cannot discard a correctly built index.
