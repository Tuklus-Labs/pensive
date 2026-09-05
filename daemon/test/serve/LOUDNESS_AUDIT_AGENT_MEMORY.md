# Loudness Audit: agent-memory corrections

The scoped test file contains 67 `assert` statements. Every assertion names the
rule it guards and includes the relevant atom, edge, facet, call sequence,
message, handle, or before/after state. `pytest.raises` checks name the expected
error where the exact failure is part of the contract.

The mutation pass found three places where a collection was indexed before the
named assertion could run: fork-line extraction and two postcommit edge reads.
Those reads now assert cardinality first. The bare/prefixed handler test likewise
checks correction success and edge cardinality before reading the successor, so
a handle regression reports the normalization invariant instead of `IndexError`.

Exemptions:

- Values read after a preceding cardinality assertion in the same test retain
  ordinary indexing. The preceding assertion stops the test with the named rule.
- SQLite setup calls and fixture construction are not assertions. Their native
  exceptions identify setup failure before behavior under test begins.
- Existing assertions outside `test_agent_memory_corrections.py` were not
  rewritten during this scoped correction review.
