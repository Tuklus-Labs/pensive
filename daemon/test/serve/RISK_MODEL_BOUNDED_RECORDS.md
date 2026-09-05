# Risk Model: bounded structured recall records

Scope: provenance admission and body-budget stubs in `handle_recall_records`,
the optional `provenanceTruncated` record field, and the bounded provenance read
from `store.getAtom`.

## Axis: Invariants

- P1: A valid atom with any provenance history remains recallable. The handler
  fetches at most 65 rows to return at most 64.
- P2: The first 64 provenance rows remain ordered and exact. Any omitted tail is
  disclosed by record-level `provenanceTruncated=true` and envelope-level
  `truncated=true`; canonical history remains complete.
- P3: A body that exceeds `tokenBudget` yields one metadata stub with the real
  atom ID and full-body token cost. Envelope token accounting covers only the
  returned notice and never exceeds the requested budget.
- P4: A complete record retains the prior output shape. The truncation property
  is optional, has the sole valid value `true`, and appears only when rows were
  omitted.
- P5: Record-level full-body cost can exceed the 8,000 body budget. Its schema
  ceiling is the safe JSON integer maximum; the envelope ceiling remains 8,000.

## Axis: State transitions

- S1: Once any admitted record truncates provenance, a later complete record
  cannot clear the envelope truncation state.
- S2: Degrading a body selects verbose notice, compact notice, then empty body
  as required by the caller's budget. Each state preserves metadata.
- S3: A genuinely empty engine result remains `records=[]`, `truncated=false`;
  budget starvation must not take that state.

## Axis: Boundaries

- B1: Provenance counts 64, 65, and 1,000 cover exact, first-omitted, and long
  histories.
- B2: Body budgets 1, 5, 10, 20, and 80 cover empty, compact, and verbose notice
  choices.
- B3: A 32,000-character valid emit body costs 10,667 estimated tokens and must
  remain representable as a stub at the maximum 8,000 envelope budget.
- B4: The 1 MiB serialized MCP envelope cap includes optional truncation fields
  and multibyte provenance values.

## Axis: Malformed inputs

- M1: Existing strict schema validation still rejects malformed provenance in
  the retained prefix; truncation does not make invalid retained rows valid.
- M2: `provenanceTruncated=false`, a non-boolean marker, and unknown record
  properties remain schema-invalid.
- M3: Missing store rows and serialization faults remain loud errors; the new
  bounded read cannot convert them to empty results.

## Axis: Concurrency

- N/A: `handle_recall_records` is synchronous and adds no shared mutable state.
  The per-call truncation accumulator is local. Store concurrency semantics are
  unchanged.

## Axis: Persistence

- D1: Rendering never deletes or rewrites provenance. A subsequent ordinary
  `getAtom` returns the full history.
- D2: `getAtom(..., provenanceLimit=65)` changes only the read query and uses a
  parameterized SQL `LIMIT`; default and explicit-`None` reads stay complete.

## Axis: Integration contracts

- C1: The record schema admits the optional marker and a safe-integer full body
  cost while keeping all prior required fields unchanged.
- C2: Body-budget accounting and MCP byte-cap accounting remain independent;
  both gates must pass before a candidate is admitted.
- C3: Read spies and callers forward keyword arguments so the new store bound is
  observable and compatible with existing calls.
- C4: Telemetry records only IDs actually served, including a metadata stub.

## Axis: Regression traps

- [x] boundary: exact 64/65 provenance and 1-token body budgets. Covered by
  B1-B3.
- [x] concurrency: N/A; no shared mutable state is introduced.
- [x] contract: optional marker, full-body cost, envelope cost, and unchanged
  ordinary shape. Covered by P3-P5 and C1-C4.
- [x] encoding: multibyte provenance must still obey the byte cap. Covered by
  B4 and existing UTF-8 wire-cap tests.
- [x] framework: JSON Schema optional-vs-required and `const=true` behavior.
  Covered by P4, M1-M2, and C1.
- [x] io: bounded, parameterized SQLite provenance reads. Covered by P1 and D2.
- [x] persistence: canonical provenance survives rendering. Covered by P2 and
  D1-D2.
- [x] resource: the 65-row sentinel read prevents eager unbounded history loads.
  Covered by P1 and B1.
- [x] state: an early truncation marker must survive later complete records;
  no-match and starved states stay distinct. Covered by S1-S3.

## Coverage Matrix

| Risk rows | Test name(s) |
|---|---|
| P1, P2, P4, S1, B1, D1, D2, C1, C3 | `test_long_provenance_is_bounded_and_disclosed_without_losing_results` |
| P3, S2, S3, B2, C2, C4 | `test_recall_records_budget_starvation_serves_stub_not_empty`, `test_starved_record_notice_respects_body_budget_without_hiding_identity` |
| P3, P5, B3, C1, C2 | `test_starved_max_emit_body_preserves_full_cost_without_schema_failure` |
| P4, M2, C1 | `test_recall_records_schemas_are_strict` |
| M1, M3 | `test_recall_records_contains_corrupt_provenance_at_served_boundary`, `test_recall_records_missing_atom_errors_then_next_call_succeeds`, `test_recall_records_serialization_failure_is_contained` |
| B4, C2, C4 | `test_recall_records_wire_cap_admits_exact_size_and_drops_one_byte_over`, `test_recall_records_wire_cap_counts_multibyte_utf8_and_logs_only_admitted`, `test_recall_records_bounds_schema_maximum_shape_without_building_it_all` |
