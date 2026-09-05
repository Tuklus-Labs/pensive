# Recall feedback risks

| Axis | Risk and coverage |
| --- | --- |
| Invariants | Only exposed atoms can receive feedback; receipt actor/task must match. `test_receipt_scope_and_exposure` |
| State transitions | Exposure, shown and used never mean helpful; only explicit helpful notes earn credit. `test_only_helpful_credits_once_per_task` |
| Boundaries | Empty receipt valid; duplicate atoms, nonfinite scores and overlong input fail atomically. `test_receipt_validation_is_atomic` |
| Malformed input | Unknown feedback types, missing evidence notes, whitespace identities rejected. `test_feedback_validation` |
| Concurrency | Separate connections cannot double-credit one task. `test_concurrent_credit` |
| Persistence | Exact request replay survives reopening; changed replay rejected; rollback on credit failure. `test_replay_and_failure_atomicity` |
| Integration | Only final admitted records logged, receipt and legacy telemetry share a commit; legacy recall logs never earn importance. `test_receipt_scope_and_exposure`, lifecycle tests, MCP receipt tests |
| Regression traps | boundary: finite scores and caps; concurrency: writer serialization; contract: caller distinct from author; encoding: bodies and notes exact; framework: direct dispatch validation; io: commit rollback; persistence: replay; resource: bounded batch; state: exactly-once helpful credit. |

Targeted mutation results and assertion audit are recorded in SABOTAGE_FEEDBACK.md.
