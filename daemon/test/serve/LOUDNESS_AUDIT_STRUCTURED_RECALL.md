# Loudness Audit: `recall_records`

Every assertion added in `test_mcp_structured.py` names the rule it guards and includes the relevant arguments, response, schema, records, or store state. Parameterized argument failures include the offending argument map and the expected field-specific error prefix. Transport assertions include both structured and text values.

The new assertions in the existing transport test identify the string compatibility, schema version, JSON fallback, input-validation, and shutdown rules with observed state.

The aggregate-bound assertions name the exact-cap, UTF-8 accounting, atomic tail-stop, telemetry, base-envelope, and shared-wrapper rules. Failure state includes caps, character and byte counts, admitted record IDs, truncation, wrapper calls, and actual wire size. Structured results are summarized instead of dumping multimegabyte provenance into test logs.

The served-error assertions name the bounded-schema-diagnostic, served-wire-cap, pre-envelope construction, escaping-overhead, impossible-cap, same-server recovery, and legacy-preservation rules. They report lengths, byte counts, marker presence, wrapper call counts, and concise response text. They never interpolate the corrupt 2 MiB value or the arbitrary oversized dispatch error. The escaping-overhead assertion guards the spy list before indexing it, so a broken wrapper path produces the named assertion rather than an unrelated `IndexError`.

Exemptions:

- `Draft202012Validator.check_schema` retains its native path-aware exception text for static schema defects. Runtime output validation is converted to a concise path-and-validator diagnostic so rejected values cannot be reflected into MCP responses.
- Existing assertions outside the modified native-registration and transport-test scope retain the repository's established style and were not rewritten as part of this feature.
