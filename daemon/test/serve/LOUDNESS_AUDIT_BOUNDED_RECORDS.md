# Loudness Audit: bounded structured recall records

The bounded-record additions contribute 26 assertions to
`test_mcp_structured.py`. Each names P1-P5 or the established starvation rule and
includes the relevant provenance count, served IDs, marker state, fetch sizes,
budget, token costs, schema fragment, or wire bytes.

The first mutation runs showed that dumping a 64-row record made failures noisy
and obscured the violated rule. Those messages now report record IDs,
provenance IDs/counts, marker state, and output keys. They retain enough state to
diagnose prefix or propagation errors without printing every provenance field.

Cardinality is asserted before indexing `records[0]` or `stub`, so a missing
record fails at the named admission invariant rather than with `IndexError`.
The maximum-body regression reports the full cost and concise stub metadata; it
never interpolates the 32,000-character body.

Exemptions:

- `jsonschema.validate` retains its native path-aware exception because the
  test is explicitly checking that the advertised schema accepts the result.
- Existing assertions before the bounded-record additions remain covered by
  `LOUDNESS_AUDIT_STRUCTURED_RECALL.md` and were not rewritten here.
