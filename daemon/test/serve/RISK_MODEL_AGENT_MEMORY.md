# Risk Model: atomic corrections, history, and MCP identifiers

Scope: `store.correctAtom`, bounded provenance reads in `store.getAtom`, the
native correction/history/pin/unpin handlers, and the supersession portion of
`lifecycle.integrityScan`.

## Axis: Invariants

- I1: A committed correction creates exactly one live successor, one
  `supersedes` edge, successor authorship and edge provenance, and changes the
  predecessor to `superseded` in one transaction.
- I2: A failed correction leaves atoms, FTS, provenance, edges, status, and
  facets unchanged, including when failure occurs at the final status update.
- I3: The successor retains project, kind, importance, occurred time, and only
  `pin` and `tag` facets. Content-derived entity facets are not copied.
- I4: The predecessor's text and provenance remain readable and unchanged.
- I5: An optional provenance limit returns an ordered prefix; the default and
  explicit `None` retain the complete historical read.

## Axis: State transitions

- S1: `live -> superseded` is the only correction transition. A `superseded` or
  `tombstone` target is terminal for high-level correction.
- S2: Pin and unpin change only the pin facet; both remain idempotent.
- S3: Linear history is labeled a chain. Forked or cyclic history is labeled a
  graph, and fork output identifies the predecessor and parallel successors.
- S4: The low-level `supersede` helper stays permissive so legacy fork and cycle
  fixtures remain representable for integrity and history checks.

## Axis: Boundaries

- B1: Bare ULIDs and one `p3://` prefix resolve to the same atom for history,
  correct, pin, and unpin.
- B2: Repeated prefixes, wrong schemes, ref prefixes, whitespace, blanks,
  non-strings, and malformed ULIDs fail before mutation.
- B3: A syntactically valid but absent ULID fails with `not found` and zero
  writes.
- B4: Zero and nonzero importance values survive correction exactly.
- B5: A provenance limit of one is legal; zero, negatives, floats, booleans,
  and numeric strings are invalid.
- B6: Duplicate legacy edges do not duplicate successor handles in the
  integrity fork report.

## Axis: Malformed inputs

- M1: Non-string correction text and non-object provenance fail with a named
  field error and zero writes.
- M2: Missing required identifiers and text fail before `correctAtom` starts.
- M3: A stale correction error reports current successor handles when known and
  directs the caller to history.
- M4: Index add, removal, and rebuild exceptions stay visible after commit; an
  unrecovered error includes `COMMITTED` and the new `p3://` handle.

## Axis: Concurrency

- C1: Two independent SQLite connections correcting the same live predecessor
  serialize under `BEGIN IMMEDIATE`; exactly one commits.
- C2: The live predicate is checked after the write lock is acquired and again
  in the status update, so a stale read outside the transaction cannot authorize
  a second successor.
- C3: The losing writer observes the committed successor and returns history
  guidance without adding rows.

## Axis: Persistence

- P1: A committed correction survives close/reopen with both atom rows, retained
  metadata/facets, provenance, status, and the supersession edge intact.
- P2: Mid-transaction and late-transaction failures roll back FTS, provenance,
  edge, facet, and successor writes.
- P3: A bounded provenance read changes no canonical state and does not alter
  the default complete read contract.

## Axis: Integration contracts

- X1: Native history, correct, pin, and unpin share one identifier normalizer
  and always display `p3://` handles.
- X2: `correctAtom` returns the new bare ULID for index maintenance and output.
- X3: Incremental add or removal failure rebuilds only the affected class. A
  successful rebuild ends recovery; no redundant incremental removal follows.
- X4: If incremental maintenance and rebuild both fail, the error identifies
  the already committed successor so retry cannot create another correction.
- X5: Integrity reports every unique fork branch and sets `ok=false` for forks.
- X6: `getAtom(..., provenanceLimit=N)` uses a parameterized positive integer
  bound and preserves the ordinary unbounded call signature.

## Axis: Regression traps

- [x] boundary: `boundary: zero treated as falsy in numeric context` applies to
  retained importance and provenance limits. Covered by I3, B4, and B5.
- [x] concurrency: `concurrency: read-modify-write without lock` applies to two
  corrections of one predecessor. Covered by C1-C3.
- [x] contract: `contract: field rename breaks silent consumers` applies to
  displayed handles, committed-handle diagnostics, and default `getAtom` shape.
  Covered by X1, X2, X4, and X6.
- [x] encoding: malformed schemes and whitespace can be interpreted as another
  identifier. Covered by B1-B3.
- [x] framework: direct `dispatch` bypasses MCP schema validation, so handler
  validation must reject wrong JSON shapes. Covered by M1 and M2.
- [x] io: SQLite transaction locking and derived-index rebuild errors cross the
  store/handler boundary. Covered by C1, X3, and X4.
- [x] persistence: the former two-commit correction path could leave an orphan
  successor. Covered by I1, I2, P1, and P2.
- [x] resource: graph traversal must terminate on cycles, and bounded provenance
  reads must not eagerly load an unbounded list. Covered by S3, I5, and X6.
- [x] state: stale targets, forked successors, retained pin/tag state, and index
  retirement are distinct lifecycle states. Covered by S1-S4, M3, and X3-X5.

## Coverage Matrix

| Risk rows | Test name(s) |
|---|---|
| I1, I3, I4, B4, X2 | `test_correct_atom_commits_successor_edge_and_metadata_atomically` |
| I2, P2 | `test_correct_atom_rolls_back_without_orphan`, `test_correct_atom_rolls_back_every_write_on_late_status_failure` |
| P1 | `test_correct_atom_persists_canonical_state_after_reopen` |
| S1, M3, C3 | `test_correct_atom_refuses_stale_target_with_successor_guidance` |
| C1, C2, C3 | `test_concurrent_corrections_create_one_successor` |
| S3, S4 | `test_history_lists_all_fork_branches`, `test_history_from_fork_descendant_lists_sibling_branch`, `test_history_cycle_is_finite` |
| B6, X5 | `test_integrity_marks_fork_unhealthy`, `test_integrity_reports_each_fork_branch_once` |
| M4, X3, X4 | `test_correct_reports_committed_handle_after_index_failure`, `test_correct_add_failure_recovers_with_rebuild_without_retire`, `test_correct_retire_failure_recovers_with_rebuild`, `test_correct_reports_committed_handle_when_retire_and_rebuild_fail` |
| S2, B1, X1 | `test_native_handlers_accept_bare_and_prefixed_ids` |
| B2 | `test_native_handlers_reject_malformed_handles_before_write` |
| B3 | `test_correct_missing_valid_id_is_zero_write` |
| M1, M2 | `test_correct_rejects_malformed_content_before_write` |
| I5, B5, P3, X6 | `test_get_atom_provenance_limit_requires_positive_integer`, `test_get_atom_provenance_limit_preserves_prefix_and_default_full_read` |
