# Loudness Audit: `recall_records`

Every assertion added in `test_mcp_structured.py` names the rule it guards and includes the relevant arguments, response, schema, records, or store state. Parameterized argument failures include the offending argument map and the expected field-specific error prefix. Transport assertions include both structured and text values.

The new assertions in the existing transport test identify the string compatibility, schema version, JSON fallback, input-validation, and shutdown rules with observed state.

The aggregate-bound assertions name the exact-cap, UTF-8 accounting, atomic tail-stop, telemetry, base-envelope, and shared-wrapper rules. Failure state includes caps, character and byte counts, admitted record IDs, truncation, wrapper calls, and actual wire size. Structured results are summarized instead of dumping multimegabyte provenance into test logs.

Exemptions:

- `Draft202012Validator.check_schema` and `jsonschema.validate` use the validator's path-aware exception text. Adding an outer assertion would hide more precise schema location and value details.
- Existing assertions outside the modified native-registration and transport-test scope retain the repository's established style and were not rewritten as part of this feature.
