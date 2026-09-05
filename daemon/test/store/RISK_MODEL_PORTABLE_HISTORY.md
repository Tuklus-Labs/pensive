# Portable history risk model

Scope: JSONL export format 2, legacy reads and complete history snapshots.

| Axis | Risk |
|---|---|
| Invariants | E1: atoms/provenance/edges/facets plus usage and supersession-review state survive a round trip exactly. |
| State transitions | E2: processed usage remains processed; pending usage accrues once after restore. |
| Boundaries | E3: legacy four-file dumps remain readable; current empty tables still require their files. |
| Malformed inputs | E4: missing current-format files and unsupported manifests fail before creating a target. |
| Concurrency | E5: all tables come from one SQLite read snapshot even when another connection writes. E7: a rebuild racing publication either reads one manifest generation or rejects a digest mismatch; a legacy read rejects a transition to a manifested export before commit. E8: target reservation is atomic against a competing creator. |
| Persistence | E6: staging failure leaves a prior completed dump intact; interrupted publication is visibly incomplete. E9: staged files and publication state are flushed in an order that leaves Linux filesystem interruption fail-closed. Power-loss behavior still depends on filesystem and storage guarantees. |
| Integration contracts | JSONL columns remain raw schema names; derived vectors/FTS remain rebuildable and excluded; unrelated destination files remain untouched. Format-2 rebuild is read-only on its source directory. Editing a format-2 JSONL file requires updating its manifest digest deliberately. |
| Regression traps | boundary: E3; concurrency: E5/E7/E8; contract: E1/E4; encoding: Unicode query/source strings in E1; framework: N/A, plain JSONL; io: E6/E9; persistence: E1/E2/E6/E9; resource: table rows are streamed and hashed through the parsing descriptor; state: E2/E7. |

The format manifest declares the required tables and the SHA-256 digest of each
JSONL file. Rebuild hashes bytes through the same descriptor it parses and checks
the digest only after consuming that file, so malformed rows still raise their
specific parse or SQLite error before an integrity mismatch. A publication marker
prevents a partially replaced file set from being mistaken for a legacy dump;
digests close the race where publication begins after rebuild's initial marker
check. Existing dump files are replaced only after staging a complete snapshot.

## Coverage

| Risk | Test |
|---|---|
| E1/E2 | `test_restore_preserves_usage_and_review_history_and_accrues_once` |
| E3 | `test_legacy_four_file_dump_remains_readable`, `test_current_dump_requires_each_declared_history_file` |
| E4 | `test_unknown_export_version_is_refused_before_target_creation` |
| E5 | `test_export_reads_all_tables_from_one_snapshot` |
| E6 | `test_staging_failure_preserves_previous_complete_export`, `test_interrupted_publication_is_refused_as_incomplete` |
| E7 | `test_rebuild_rejects_generation_changed_after_manifest_read`, `test_legacy_rebuild_rechecks_publication_transition_before_commit` |
| E8 | `test_rebuild_atomically_refuses_competing_target_creator` |
| E9 | `test_export_fsyncs_staged_files_marker_and_directories` |
| Read-only source contract | `test_format2_rebuild_reads_from_read_only_source` |

## Executed mutations

The original nine format/history production mutations in
`work/export-mutations/report.json` each exited 1; removing the target test body
made each exit 0. The race-hardening review added five actual mutations with the
same result on 2026-09-05:

| Mutation | Detecting test | Production / removed-test exit |
|---|---|---|
| Ignore format-2 digests | `test_rebuild_rejects_generation_changed_after_manifest_read` | 1 / 0 |
| Skip legacy publication-transition recheck | `test_legacy_rebuild_rechecks_publication_transition_before_commit` | 1 / 0 |
| Replace exclusive target reservation with existing-file adoption | `test_rebuild_atomically_refuses_competing_target_creator` | 1 / 0 |
| Skip publication directory fsync | `test_export_fsyncs_staged_files_marker_and_directories` | 1 / 0 |
| Write into the source export during rebuild | `test_format2_rebuild_reads_from_read_only_source` | 1 / 0 |

Loudness audit: 23 assertions, zero missing rule-naming failure messages.
