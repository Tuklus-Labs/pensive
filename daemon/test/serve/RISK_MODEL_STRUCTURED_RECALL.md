# Risk Model: `recall_records`

## Axis: Invariants

- I1: Every successful value has `schemaVersion == 1` and exactly the documented top-level fields.
- I2: Records retain engine rank order and include the exact stored body, atom metadata, complete provenance, and trust fields.
- I3: Record bodies are atomic. The first record that exceeds the remaining content-token budget drops that record and the entire tail.
- I4: `estimatedTokens` equals the sum of admitted record body estimates and never exceeds `tokenBudget`.
- I5: The deterministic text fallback parses to the exact structured value and contains no NaN or non-JSON values.
- I6: Existing compat tool schemas, native handlers, result strings, and string-only MCP responses do not change.
- I7: The actual serialized `CallToolResult`, including duplicated text and structured channels, is at most 1 MiB in UTF-8 bytes.
- I8: Wire-size admission preserves rank and atomicity. The first whole record whose final response envelope exceeds the cap drops that record and the entire tail.
- I9: Every served `recall_records` result, including dispatch errors, is at most 1 MiB in actual serialized UTF-8 bytes.
- I10: Schema diagnostics identify only the failing output path and validator; stored values and `jsonschema` instance text never enter the error.
- I11: A raw `recall_records` string already over the cap is replaced before `_callToolResult` construction, avoiding a giant intermediate envelope.

## Axis: State transitions

- S1: A malformed call returns an MCP error without invoking recall or changing daemon state.
- S2: A handler or serialization failure is contained by `dispatch`; the next valid tool call still succeeds.
- S3: `recall_records` is read-only and does not reindex, mutate atoms, or change compatibility handlers.
- S4: An oversized base envelope fails loudly inside dispatch containment; it never produces an over-cap success response.
- S5: A corrupt-row error and an oversized served error are contained; the same server accepts the next valid `recall_records` call.

## Axis: Boundaries

- B1: Query accepts 1 and 8192 characters and rejects blank, 8193 characters, null, and non-string values.
- B2: Project accepts absent/null, 1, and 256 characters and rejects blank, 257 characters, and non-string values.
- B3: `k` accepts integers 1 through 32 and rejects booleans, zero, 33, floats, and strings.
- B4: `tokenBudget` accepts integers 1 through 8000 and rejects booleans, zero, 8001, floats, and strings.
- B5: `timeScope` accepts null or two JavaScript-safe non-negative integers with `start <= end`; it rejects the wrong length, booleans, negatives, unsafe integers, and inverted bounds.
- B6: `kinds` accepts null or a non-empty unique subset of the four stored kinds; it rejects empty arrays, duplicates, unknown values, strings, and non-string members.
- B7: A record whose body exactly fits the remaining budget is admitted; one token over drops it and the tail.
- B8: Results are capped at 32 records and each provenance array is bounded to 64 entries by the output contract.
- B9: A candidate whose exact serialized `CallToolResult` is at the aggregate cap is admitted; one byte over is omitted atomically.
- B10: UTF-8 wire bytes, not Python characters, determine aggregate admission for multibyte metadata and provenance.
- B11: The final served boundary replaces an arbitrary over-cap `recall_records` result with one fixed concise error envelope; raw over-cap strings short-circuit before response construction.
- B12: If the configured cap cannot hold even the fixed error envelope, serving fails with a bounded diagnostic instead of returning an over-cap result.

## Axis: Malformed inputs

- M1: The argument object rejects every unknown top-level field.
- M2: Missing stored atoms fail loudly instead of yielding partial records.
- M3: Non-finite score, confidence, or importance fails deterministic JSON serialization inside dispatch containment.
- M4: Stored nullable fields remain explicit JSON nulls; required stored fields are never silently defaulted.
- M5: A real SQLite row with an overlong provenance field fails without echoing any portion of the corrupt instance.

## Axis: Concurrency

- N/A: The handler is synchronous and read-only on one SQLite connection. Request scheduling and SQLite connection ownership are daemon-level concerns outside this unit.

## Axis: Persistence

- P1: The wire shape carries `atomSchemaVersion` so stored schema changes cannot be mistaken for the `recall_records` protocol version.
- P2: All durable atom and provenance values come from `getAtom`; no reconstructed or inferred provenance may replace stored values.
- P3: Missing or corrupt durable rows surface as errors and do not poison the following request.
- P4: Store corruption cannot amplify into an unbounded MCP error response.

## Axis: Integration contracts

- C1: The existing recall engine is called exactly once with normalized `query`, `project`, `timeScope`, `kinds`, `k`, and `tokenBudget`.
- C2: `recall_records` is the fifth v3 native tool and advertises strict draft-compatible input and output schemas.
- C3: Real streamable HTTP returns the same value in `structuredContent` and the compact deterministic JSON text block.
- C4: Existing string tools still return identical text and `structuredContent is None` over real MCP.
- C5: MCP input validation and direct dispatch validation agree on unknown fields and type/bound rules.
- C6: Recall telemetry contains only records actually admitted under the structured budget, never the dropped tail.
- C7: Size measurement and MCP serving use the same private response-wrapper helper, preventing envelope drift.
- C8: The final served-envelope guard applies only to `recall_records`; oversized legacy result text remains byte-for-byte unchanged.
- C9: Raw over-cap strings short-circuit first; strings within the raw cap and all structured responses are measured using actual `CallToolResult.model_dump_json()` UTF-8 bytes so envelope and escaping overhead remain covered.

## Axis: Regression traps

- [x] boundary: off-by-one in inclusive vs exclusive range, and zero treated as falsy in numeric context. Exact endpoint and just-over tests cover every numeric/text bound.
- [x] concurrency: N/A, synchronous read-only handler with no internal shared-state transition.
- [x] contract: API returns null where caller expects empty collection, caller/callee shape mismatch, and endpoint scope drift. Empty results return `records: []`; exact engine arguments and schema fields are pinned; the final error guard is `recall_records`-only.
- [x] encoding: non-finite JSON numbers, JavaScript-unsafe integers, and Unicode byte/character mismatch. Serialization uses `allow_nan=False`; time bounds are safe integers; aggregate accounting measures serialized UTF-8 bytes.
- [x] framework: Python booleans are integers and MCP may validate before dispatch. Both direct and transport paths reject bool-as-int inputs.
- [x] io: HTTP response-envelope drift and error amplification. Transport tests assert both output channels, and the final served boundary measures the actual serialized result for success or error.
- [x] persistence: schema migration leaves stale rows and corrupt stored strings enter diagnostics. Protocol and atom schema versions are separate; missing or malformed rows fail without echoing stored instances.
- [x] resource: unbounded response and error growth plus child-process pipe backpressure. Raw over-cap errors are rejected before response construction, every served `recall_records` result is capped at 1 MiB, and the transport probe writes daemon output to a temporary file instead of an undrained pipe.
- [x] state: handler failure poisons next request. Corrupt-row and arbitrary oversized-error calls are followed by valid calls on the same server.

## Coverage Matrix

| Risk row | Test name(s) covering it |
|----------|--------------------------|
| I1, B8, C2 | `test_recall_records_schemas_are_strict`, `test_native_tools_present_with_required_fields` |
| B1-B6, M1, C5, S1 | `test_recall_records_argument_contract`, `test_recall_records_accepts_inclusive_argument_endpoints` |
| I2, I4, M4, P1, P2 | `test_recall_records_preserves_rank_metadata_provenance_and_trust` |
| I3, B7, C6 | `test_recall_records_applies_atomic_rank_order_budget` |
| I1, I5, M3, M4, S2 | `test_recall_records_serialization_failure_is_contained`, `test_recall_records_output_schema_failure_is_contained` |
| S2, M2, P3 | `test_recall_records_missing_atom_errors_then_next_call_succeeds` |
| C1 | `test_recall_records_calls_engine_once_with_normalized_arguments` |
| I6, C4 | `test_compat_tool_schemas_are_verbatim`, `test_build_server_wraps_structured_and_string_results` |
| C3-C5, resource pipe backpressure | `test_build_server_wraps_structured_and_string_results`, `test_structured_recall_round_trips_over_temporary_http_transport` |
| S3 | `test_recall_records_preserves_store_state` |
| Empty-result contract | `test_recall_records_empty_result_is_low_confidence` |
| I7, I8, B9, B10, C6 | `test_recall_records_wire_cap_admits_exact_size_and_drops_one_byte_over`, `test_recall_records_wire_cap_counts_multibyte_utf8_and_logs_only_admitted`, `test_recall_records_bounds_schema_maximum_shape_without_building_it_all` |
| S4, C7 | `test_recall_records_base_envelope_over_cap_fails_loudly`, `test_call_tool_result_wrapper_matches_served_envelope`, `test_server_and_wire_measurement_share_call_tool_result_wrapper` |
| I9, I10, M5, P3, P4, S5 | `test_recall_records_contains_corrupt_provenance_at_served_boundary` |
| I9, I11, B11, C9, S5 | `test_recall_records_served_boundary_replaces_oversized_dispatch_error`, `test_recall_records_served_boundary_counts_escaping_overhead` |
| B12 | `test_recall_records_served_boundary_fails_when_cap_cannot_hold_error` |
| I6, C4, C8 | `test_legacy_served_boundary_preserves_oversized_dispatch_error` |
