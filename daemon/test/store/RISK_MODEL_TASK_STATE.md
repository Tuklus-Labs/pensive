# Task state and schema v4 risk model

Scope: `store.checkpoints`, schema v4 migration, format 3 export, and rebuild.
The task checkpoint stream is append-only and must never alter atoms, provenance,
snapshots, people text, embeddings, or derived indexes. Existing format 2 and
legacy four-file reads remain valid.

## Invariants

| ID | Risk | Required behavior |
|---|---|---|
| I1 | Checkpoint isolation | Checkpoint writes touch only `task_checkpoints`; atom and embedding counts stay unchanged. |
| I2 | Canonical wire row | Every returned row has the exact camelCase checkpoint fields and preserves body, source, session, sourceRef, and recordedAt. |
| I3 | Request replay | Same requestId and identical input returns the original row; changed input fails without a new row. |
| I4 | Revision CAS | A new stream starts at revision 1 for expectedRevision 0; later revisions are contiguous and require the current head. |
| I5 | Scope isolation | Heads and histories are separated by exact project, agent, and taskId scope. |
| I6 | Stable selection | Current, revision as-of, and recordedAt as-of choose only rows allowed by their selector and have deterministic ordering. |
| I7 | Durable export | Format 3 contains all eleven portable tables, with stable column order, row order, and SHA-256 digests. |
| I8 | Restore fidelity | Rebuild preserves all portable values and foreign-key relationships, including task-state and feedback rows. |
| I9 | Imported scope integrity | Restores reject feedback whose actor/task differs from its receipt and credits that do not reference matching helpful feedback. |
| I10 | Transaction ownership | A checkpoint call refuses an already-open caller transaction without rolling it back. |
| I11 | Processed credit completeness | Every processed helpful feedback row has one matching scope credit; repeated event IDs may share it. |

## State transitions

The only legal checkpoint states are `active`, `blocked`, `completed`, and
`abandoned`. A write appends one revision, never updates or deletes an earlier
revision. A failed CAS or replay leaves the stream unchanged. A completed head
is still returned by recent-state discovery.

## Boundaries

`project` is a nonempty string of at most 256 characters; `agent` is nonempty
and at most 64; `taskId` and `requestId` are nonempty and at most 256; `body` is
at most 32,000 characters. Revision, asOf, afterRevision, and limit accept real
integers only, with revisions and asOf nonnegative and limit positive. The
schema and export must also handle empty tables and nullable optional metadata.

## Malformed inputs

Reject booleans, floats, nulls, and non-string identifiers before a database
write. Reject unknown states, missing required fields, mutually exclusive
revision/asOf selectors, and as-of/history calls without exact project and
agent. Malformed JSONL, invalid manifests, missing declared files, digest
mismatches, and dangling foreign keys fail before publication.

## Concurrency

The checkpoint writer obtains `BEGIN IMMEDIATE` before reading the head. Two
connections racing the same expected revision produce one committed revision
and one clear CAS failure. Request replay is checked under that same write lock,
so a race cannot create duplicate request IDs or skip revisions.

## Persistence and migration

The v3 to v4 upgrade is additive: five tables and their indexes are created,
old rows are untouched, and the version stamp commits atomically with the DDL.
An injected failure after DDL leaves the database at v3 with no partial v4
objects or version stamp. Fresh schema and migrated schema expose the same v4
objects. Reopening an unchanged v4 store is a no-op.

## Integration contracts

Format 2 remains exactly its original six files and schema range through v3;
removing its new-to-v4 files is not corruption. Legacy four-file exports still
restore without invented task checkpoints, receipts, feedback, or credits.
Format 3 uses eleven files in foreign-key order: atoms, provenance, edges,
facets, recall_log, supersession_proposals, task_checkpoints, recall_receipts,
recall_exposures, recall_feedback, memory_credits. The rebuild creates a private
database and publishes it only after all rows and digests validate.

## Regression traps

| Prefix | Audit target |
|---|---|
| `boundary` | Length limits, zero revision, empty optional tables, and page limits are explicit. |
| `concurrency` | CAS race and request replay are serialized before head reads. |
| `contract` | CamelCase row shape, exact column order, state set, format versions, and backcompat are asserted. |
| `encoding` | JSONL preserves Unicode, newlines, nulls, and exact digest bytes. |
| `framework` | SQLite `executescript` implicit-commit behavior is guarded by an injected post-DDL failure test. |
| `io` | Missing files, marker state, fsync/publication failure, and digest mismatch fail closed. |
| `persistence` | Full eleven-table round trip and atomic migration preserve durable history. |
| `resource` | Connections and temporary rebuild state close and clean up on all failure paths. |
| `state` | Append-only revisions and completed heads remain visible without atom mutation. |

## Coverage matrix

| Risk row | Tests |
|---|---|
| I1 | `test_checkpoint_write_does_not_touch_atoms_or_embeddings` |
| I2 | `test_checkpoint_wire_shape_preserves_authorship_and_timestamp` |
| I3 | `test_checkpoint_replay_is_idempotent_and_changed_replay_fails` |
| I4 | `test_checkpoint_cas_race_allows_one_writer` |
| I5 | `test_checkpoint_scope_isolation` |
| I6 | `test_current_history_and_asof_selection` |
| I7 | `test_format3_export_has_stable_eleven_table_digest_set` |
| I8 | `test_format3_round_trip_preserves_all_new_tables` |
| I9 | `test_format3_rebuild_rejects_invalid_v4_rows` |
| I10 | `test_checkpoint_rejects_an_open_caller_transaction_without_rollback` |
| I11 | `test_format3_rebuild_rejects_processed_helpful_without_scope_credit`, `test_format3_restore_allows_processed_helpful_rows_to_share_scope_credit` |
| migration | `test_v3_to_v4_failure_rolls_back_ddl_and_version` |
| backcompat | `test_format2_and_legacy_four_file_exports_remain_readable` |
| malformed | `test_checkpoint_rejects_bool_and_bound_violations` |
| pagination | `test_history_and_recent_pages_are_explicit` |
| missing/digest | `test_format3_rebuild_rejects_missing_file_and_digest_mismatch` |
