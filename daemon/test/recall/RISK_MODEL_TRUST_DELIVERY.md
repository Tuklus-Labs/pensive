# Trust and payload delivery risk model

Scope: imported-corpus corroboration, mixed-trust payloads and estimated budgets.
The confidence formula and ranking order remain outside this change.

| Axis | Risk and required evidence |
|---|---|
| Invariants | T1: a single signal family cannot corroborate an imported chunk; T2: an explicitly untrusted result has a labelled handle, not a full body; T3: estimated output tokens never exceed a nonnegative budget. |
| State transitions | T4: a superseded result retains its live-successor pointer when reduced to a handle. |
| Boundaries | T3: empty results, zero budget and every size through both sentinel lengths; T5: two genuine signal families are still eligible for trust. |
| Malformed inputs | Negative budgets fail explicitly. Store text remains data, with existing Unicode framing tests guarding line boundaries. |
| Concurrency | N/A: these functions read a synchronous store and mutate no shared state. Canonical/index concurrency belongs to correction tests. |
| Persistence | N/A: no writes or format migrations; fixtures use real SQLite rows and provenance. |
| Integration contracts | T6: optional enrichment is never invoked for weak results; trusted bodies and result order remain unchanged. The reported token count uses the existing approximate estimator, not an exact tokenizer claim. |
| Regression traps | boundary: T3; concurrency: N/A, no new state; contract: T1/T2/T4/T6; encoding: existing test_payload_forgery.py; framework: framing remains through existing helpers; io: N/A, no new IO; persistence: N/A, read-only; resource: T6 avoids unnecessary enrichment; state: T4. |

Signal families are lexical (bm25), semantic (dense/openai aliases count once)
and entity/facet. Two semantic encoders alone do not corroborate each other for
this rule. Authored memories and mixed import/author provenance retain their
existing scoring behavior.

## Coverage

| Risk | Regression |
|---|---|
| T1 | `test_one_import_signal_family_cannot_establish_trust` (four signal sets) |
| T2/T6 | `test_mixed_payload_limits_weak_result_to_labelled_handle` (with/without enrichment) |
| T3 | `test_every_small_budget_is_respected` (empty, weak, trusted); `test_negative_payload_budget_is_rejected` |
| T4 | `test_superseded_handle_keeps_successor_pointer` |
| T5 | `test_two_independent_import_signal_families_remain_eligible` (three signal pairs) |

Existing authored/mixed-provenance trust tests and Unicode framing tests remain
unchanged. Independent review also exercised all 16 signal subsets and 2,408
mixed-trust/budget combinations. Executed mutations are in
`SABOTAGE_LOG_TRUST_DELIVERY.md`.
