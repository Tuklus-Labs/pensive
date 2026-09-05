# Feedback mutation and assertion audit

The isolated subprocess runner executes altered source against real SQLite tests.

| Mutation | Kind | Observed |
| --- | --- | --- |
| scope | production | survived; redundant guard remains |
| exposure | production | caught |
| reward_delivery | production | caught |
| double_credit | production | caught |
| credit_amount | production | caught |
| replay_input | production | caught |
| rollback | production | caught |
| duplicate_exposure | production | survived; redundant guard remains |
| required_note | production | survived; redundant guard remains |
| importance_cap | production | caught |
| pending_scope | production | caught |
| bounded_batch | production | caught |
| test_receipt_scope_and_exposure | test-assertion | caught |
| test_only_helpful_credits_once_per_task | test-assertion | caught |
| test_receipt_validation_is_atomic | test-assertion | caught |
| test_feedback_validation | test-assertion | caught |
| test_replay_and_failure_atomicity | test-assertion | caught |
| test_concurrent_credit | test-assertion | caught |
| test_legacy_exposure_log_is_never_rewarded | test-assertion | caught |
| test_pending_feedback_caps_credit_and_replays_once | test-assertion | caught |
| test_pending_batch_is_bounded_and_shown_is_inert | test-assertion | caught |
| test_pending_import_cannot_bypass_receipt_scope | test-assertion | caught |
| test_checkpoint_state_transport_identity_and_stale_cas | test-assertion | caught |
| test_receipt_final_admission_caller_distinct_from_author | test-assertion | caught |
| test_direct_dispatch_validates_new_arguments | test-assertion | caught |

Scope guard removal survives because the credit layer independently validates the same receipt scope. Duplicate exposure guard removal survives because SQLite enforces both unique keys and the transaction rolls back. Optional-note guard removal survives because the credit layer rejects missing evidence before processing. These are equivalent defense removals, not observed invariant violations. Combined guard mutations are tracked separately.

Mutmut was also invoked in an isolated copy. Its initial run failed while auto-loading jaxtyping/numpy (module loaded twice); a run with plugin autoload disabled is recorded separately. Do not interpret a harness error as a killed mutant.

All assertions in the new feedback, lifecycle and memory-tool tests have rule-naming messages. Every test function with a direct assertion was rerun with an inverted assertion; all failed. Parameterized validation tests also use pytest.raises to name the failure boundary.

Evidence: campaign-two/feedback-mutations.json and feedback-mutmut*.log. No mutated source was used by the service.

The actual mutmut run completed after disabling unrelated pytest plugin autoload.
It exercised 130 mutants in recordRecallFeedback: initially 69 killed / 61
survived. Reviewing survivors exposed missing optional metadata boundaries and
the valid shown-without-note case. Those cases were added, including exact
maximum-length Unicode evidence. Rerun: 110 killed / 20 survived. Remaining
survivors are being classified in feedback-mutmut-final-survivors.txt; do not
read this as 100% mutation coverage. The independent integrity review separately
found that helpful credit could lower legacy importance >1; a real 5.0->1.0
failure and its preserving fix are now regression-tested.

The 20 surviving recordRecallFeedback mutants were inspected: 18 only wrap
otherwise intact diagnostic messages or change SQL keyword/identifier case;
two remove the receipt/exposure diagnostic entirely. Those two exposed a
loudness gap, now covered by matching receipt/exposure error context. The first
narrow rerun correctly refused to collect stats because its isolated source copy
predated the legacy-importance fix while tests had advanced; the source copy
was refreshed before rerunning. No stale-run result is counted as a pass.

Final diagnostics rerun killed both remaining no-message mutations, yielding
112/130 killed with 18 reviewed cosmetic SQL-case/message-wrapper survivors.
The final isolated subprocess run caught 34/37 source/test mutations; the three
survivors preserve behavior through an independent runtime guard or SQLite's
unique constraints. The legacy-importance regression killed the old cap logic.
Scorer mutations caught fabricated support, context-only guessed support,
duplicate/missing case acceptance and inconsistent abstention.

Real CLI integration also exposed a history cursor contract bug: returning the
first omitted revision to an exclusive afterRevision cursor skipped revision 2
(1,3 observed). A direct MCP regression failed on 1,3, then passed on 1,2 after
the cursor was changed to the last delivered revision. History now includes the
common truncated flag consumed by CLI state validation. The store pagination
test now feeds the returned cursor instead of reconstructing one in the test.
