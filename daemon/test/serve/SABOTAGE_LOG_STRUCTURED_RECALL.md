# Sabotage Log: `recall_records`

All mutations below were applied to the working tree, run against the named test, and restored before the final suite. Neither `mutmut` nor `cosmic-ray` is installed. The test plan permits manual sabotage when mutation tooling is unavailable, so the follow-up uses the same documented mutation procedure.

## Production mutations

| Mutation | Predicted result | Observed result | Conclusion |
|----------|------------------|-----------------|------------|
| Change output `records.maxItems` from 32 to 33. | Strict schema test fails. | `test_recall_records_schemas_are_strict`: failed. | The advertised response cap is pinned. |
| Advertise `tombstone` as a recallable output status. | Strict schema test fails. | `test_recall_records_schemas_are_strict`: failed on the status enum. | Retracted atoms cannot enter the structured contract. |
| Replace exact-int validation with `isinstance(value, int)`, admitting booleans. | Bool-as-`k` case fails. | `test_recall_records_argument_contract[args8-k]`: failed. | Bool/int separation is load-bearing. |
| Invoke the recall engine a second time and discard its result. | Single-call contract test fails. | `test_recall_records_calls_engine_once_with_normalized_arguments`: failed. | The engine call count is constrained. |
| Tighten the query maximum from 8192 to 8191. | Inclusive endpoint test fails. | `test_recall_records_accepts_inclusive_argument_endpoints`: failed. | The documented upper boundary is exercised. |
| Disable the first-record-that-does-not-fit stop. | Atomic budget test admits the tail and fails. | `test_recall_records_applies_atomic_rank_order_budget`: failed on the exact-fit record list. | The body and tail are not silently truncated or reordered. |
| Log every engine result instead of only admitted records. | Atomic budget test observes the dropped tail in telemetry and fails. | `test_recall_records_applies_atomic_rank_order_budget`: failed with six logged rows instead of three served rows. | Full-auto recall telemetry tracks what the caller actually received. |
| Reverse the engine result list before record construction. | Rank assertion fails. | `test_recall_records_preserves_rank_metadata_provenance_and_trust`: failed on record order. | Engine rank is preserved. |
| Replace stored provenance with an empty list. | Metadata test fails, with output validation also rejecting the record. | `test_recall_records_preserves_rank_metadata_provenance_and_trust`: failed with `provenance should be non-empty`. | Provenance cannot disappear silently. |
| Ignore unknown request fields. | Unknown-field argument case reaches the fake engine and fails. | `test_recall_records_argument_contract[args29-unknown fields]`: failed. | Direct dispatch rejects request drift before recall. |
| Disable the explicit missing-atom guard. | Missing-row test loses its named error and fails. | `test_recall_records_missing_atom_errors_then_next_call_succeeds`: failed. | Store/index desync remains loud and diagnosable. |
| Permit NaN in the JSON fallback. | Serialization-containment test fails. | `test_recall_records_serialization_failure_is_contained`: failed. | The fallback is strict JSON. |
| Disable output-schema error handling. | Invalid `lowConfidence` type escapes and the containment test fails. | `test_recall_records_output_schema_failure_is_contained`: failed. | Direct dispatch enforces the advertised output contract. |
| Update the first returned atom to `tombstone` inside the handler. | Read-only state comparison fails. | `test_recall_records_preserves_store_state`: failed. | Record recall cannot mutate atoms. |
| Return text but omit `structuredContent`. | Server-envelope assertion fails. | `test_build_server_wraps_structured_and_string_results`: failed. | Machine-readable MCP output is load-bearing. |
| Route `recall_records` through `handle_recall`. | Structured empty-result test receives a string and fails. | `test_recall_records_empty_result_is_low_confidence`: failed. | The new tool cannot drift into the human-facing path. |
| Double the candidate aggregate-cap comparison. | A response one byte over the injected cap is admitted. | `test_recall_records_wire_cap_admits_exact_size_and_drops_one_byte_over`: failed on atomic omission. | The exact aggregate boundary is pinned. |
| Count serialized Python characters instead of UTF-8 bytes. | The Unicode-heavy second record is incorrectly admitted. | `test_recall_records_wire_cap_counts_multibyte_utf8_and_logs_only_admitted`: failed with both IDs present. | Wire accounting is byte-accurate for multibyte data. |
| Continue after an oversized candidate instead of dropping the whole tail. | The maximum-shape probe fetches all 32 ranked atoms. | `test_recall_records_bounds_schema_maximum_shape_without_building_it_all`: failed with 32 fetches instead of one. | Aggregate admission stops at the first non-fitting record. |
| Reconstruct `CallToolResult` inside `buildServer` instead of using the shared wrapper. | The wrapper spy sees measurement but not serving. | `test_server_and_wire_measurement_share_call_tool_result_wrapper`: failed with zero serving calls. | Size measurement cannot drift from the served envelope. |

## Test mutations

One load-bearing assertion in each structured test function was inverted in a single temporary edit. The predicted outcome was that every test function, including all 30 malformed-input parameter cases, would fail against correct production code. Running:

```text
PYTHONPATH=daemon/src python3 -m pytest daemon/test/serve/test_mcp_structured.py -q --tb=no
```

produced `41 failed`. The edit was restored, and the same file then produced `41 passed`. The inverted assertions covered schema caps, invalid-call status, engine call count, endpoint acceptance, rank, budget order, empty collection shape, missing-row containment, NaN containment, output-schema containment, read-only state, and MCP structured content.

For the aggregate-bound follow-up, one load-bearing assertion in each of the six new cap/wrapper tests was inverted in one temporary edit. The focused run produced `6 failed, 41 deselected`. After restoration, the same selection produced `6 passed, 41 deselected`. The inversions covered exact cap admission, the fixed 1 MiB constant, base-envelope failure, maximum-shape admission, wrapper construction, and wrapper sharing.
