# Task state and schema v4 sabotage log

The focused store and export tests ran with the isolated interpreter at
`~/Documents/Codex/2026-09-05/fi/work/text-test-runtime/bin/python`.
Each production mutation below was applied to the working tree, run, observed
as a failure, and restored before the next check.

| Targeted mutation | Detecting test | Predicted result | Observed result | Conclusion |
|---|---|---|---|---|
| Flip checkpoint CAS from `!=` to `==`. | `test_checkpoint_cas_race_allows_one_writer` | Both writers reject instead of one committing. | Failed: both results were `ValueError`, with zero committed rows. | Test catches the CAS comparison defect. |
| Change replay body comparison from `==` to `!=`. | `test_checkpoint_replay_is_idempotent_and_changed_replay_fails` | Identical replay fails before the changed replay assertion. | Failed: same requestId raised the changed-input error. | Test catches replay identity drift. |
| Commit after the first v4 DDL statement. | `test_v3_to_v4_failure_rolls_back_ddl_and_version` | Injected post-DDL failure leaves a partial table. | Failed: `task_checkpoints` and v4 indexes remained after the error. | Test catches non-atomic migration. |
| Use only `_TABLES_V2` as the format-3 descriptor. | `test_format3_export_has_stable_eleven_table_digest_set` | Manifest has six files and violates the eleven-file contract. | Failed: manifest omitted every v4 file. | Test catches portable table-set loss. |
| Skip `_readRows` digest verification. | `test_format3_rebuild_rejects_missing_file_and_digest_mismatch[digest]` | Tampered JSONL rebuild publishes. | Failed: the altered file rebuilt without the required digest error. | Test catches integrity-check removal. |
| Replace v4 scope, note, delivery, and bounds validation with no-op. | `test_format3_rebuild_rejects_invalid_v4_rows[*]` | A receipt or credit with invalid actor/note/delivery/bounds imports. | Failed: the corresponding format-3 validation error was absent. | Test catches relational-history forgery. |
| Remove the `conn.in_transaction` guard and allow `BEGIN IMMEDIATE` to run. | `test_checkpoint_rejects_an_open_caller_transaction_without_rollback` | The caller's pending atom insert is rolled back or the ownership error changes. | Failed: SQLite raised `OperationalError` instead of the explicit ownership error. | Test catches transaction rollback leakage. |
| Replace the nonblank check with truthiness so whitespace IDs pass. | `test_checkpoint_rejects_bool_and_bound_violations` | Whitespace project/agent/task/request identifiers are accepted. | Failed: the whitespace-project assertion did not raise. | Test catches identifier-boundary drift. |
| Run generic `_activeRanked` even for a task-scoped brief. | `test_scoped_task_brief_is_read_only` | Scoped rendering performs the generic active query. | Failed: the guard raised `scoped brief must skip generic active ranking`. | Test catches task-view branch leakage. |
| Import a non-shown feedback row with a blank note. | `test_format3_rebuild_rejects_invalid_v4_rows[recall_feedback.jsonl-note--feedback note]` | Restore publishes a pending event that `processPendingFeedback` cannot process. | The pre-fix test passed incorrectly; after raw-row validation it fails with the feedback-note error. | Restore rejects the permanent pending-batch poison. |
| Import a credit with `feedback_id = NULL`. | `test_format3_rebuild_rejects_invalid_v4_rows[memory_credits.jsonl-feedback_id-None-credit]` | Restore accepts a credit with no helpful-feedback lineage. | The pre-fix test passed incorrectly; after the `NOT NULL`/raw validator change it fails with the credit error. | Every credit has explicit lineage. |
| Remove `_validateFormat3Row` from the JSONL iterator. | `test_format3_rebuild_rejects_invalid_v4_rows[*]` | Boolean revisions, overlong bodies, and unsupported deliveries import after SQLite affinity. | Failed: four invalid-row cases passed or surfaced only as the wrong low-level error. | Raw rows are validated before affinity can erase their type/boundary evidence. |
| Remove the processed-helpful coverage query. | `test_format3_rebuild_rejects_processed_helpful_without_scope_credit` | A processed helpful event with no scope credit restores and can earn again. | Failed: the missing-credit restore completed instead of raising. | The inverse credit invariant is load-bearing. |
| Require credit feedback_id to equal every helpful eventId. | `test_format3_restore_allows_processed_helpful_rows_to_share_scope_credit` | A legitimate second helpful event in one atom-agent-task scope is rejected. | The scope-only test remains green because the credit is shared by scope. | Credit lineage validates helpful scope and processed state without eventId equality. |

The `mutmut` executable was also attempted against `checkpoints.py`. Its
coverage runner could not import sibling `util` modules from the generated
mutant tree and later hit the environment's duplicate NumPy import guard; the
targeted edits above are the actual mutation evidence for this checkout.

## Test assertion checks

The focused tests use `_assert_rule` messages that name the invariant and carry
the failing rows, scope, manifest, or target path. The exception assertions name
the expected contract in their regex. Removing one secondary assertion from the
multi-condition tests still leaves an independent assertion failing under the
production mutations above; those tests do not pass from a single incidental
value.
