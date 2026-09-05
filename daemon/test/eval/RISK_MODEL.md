# Outcome scorer risks

Invariants: a correct answer needs a supporting displayed fact, not merely a
context citation. Transitions: N/A, stateless scoring. Boundaries: missing and
duplicate cases fail closed. Malformed input: answer/abstention disagreement and
wrong evidence shapes fail. Concurrency/persistence: N/A, immutable inputs and
no service state. Integration: a referential question legitimately uses both
recent dialogue for its subject and a retrieved fact for its answer. Regression
traps: boundary, contract, state are covered below; encoding is normalization;
framework, io, resource and concurrency are outside this pure function.

Coverage: test_context_and_fact_jointly_support_answer; test_context_alone_is_not_a_fact;
test_unknown_case_set_rejected; test_reader_schema_rejected.

Sabotage: four actual scorer source mutations (allow topic-only support, ignore
unknown extra citations, accept duplicate/missing case sets, accept inconsistent
abstention) failed their intended tests. Inverted direct assertions also failed.
Evidence lives in campaign-two/feedback-mutations.json. Loudness: every direct
assertion names its scoring rule; error cases match the relevant validation.
