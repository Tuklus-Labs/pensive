# Trust and delivery verification

The new regression file initially had 10 expected failures and four passing
controls. After the fixes, it and the adjacent trust, enrichment, tail-degrade
and framing tests passed: 57 tests.

The targeted runner loaded altered source in isolated child processes. It did
not edit the worktree. Every production mutation below failed its named test
with the expected rule assertion. Removing that test's assertion block in the
same mutated run made it pass, confirming that the test supplied the protection.

| Mutation | Test | Production / weakened-test exit |
|---|---|---|
| Never cap single-family imports | `test_one_import_signal_family_cannot_establish_trust` | 1 / 0 |
| Cap every imported result | `test_two_independent_import_signal_families_remain_eligible` | 1 / 0 |
| Render mixed weak results as full bodies | `test_mixed_payload_limits_weak_result_to_labelled_handle` | 1 / 0 |
| Omit successor pointers from handles | `test_superseded_handle_keeps_successor_pointer` | 1 / 0 |
| Always emit low-confidence sentinel | `test_every_small_budget_is_respected[empty]` | 1 / 0 |
| Always emit low-confidence sentinel | `test_every_small_budget_is_respected[weak]` | 1 / 0 |
| Always emit no-fit sentinel | `test_every_small_budget_is_respected[trusted]` | 1 / 0 |
| Accept a negative budget | `test_negative_payload_budget_is_rejected` | 1 / 0 |

The generic mutmut attempt did not yield usable coverage: the initial run hit a
NumPy repeated-import error under Python 3.14; after isolating plugins and the
package path it could not associate tests with the selected mutants. No generic
mutation score is claimed. The eight executed targeted mutation pairs above are
the available evidence.

## Assertion audit

All 12 assertions in the new regression file carry a rule ID and diagnostic
state. The negative-budget exception assertion names `tokenBudget`; it checks
the boundary's actual error rather than a literal implementation value. No
assertion-message exemptions were needed.

The token guarantee is explicitly an estimate. These tests do not claim that
the character heuristic bounds every language or tokenizer, nor that a trust
confidence is a probability of factual truth.
