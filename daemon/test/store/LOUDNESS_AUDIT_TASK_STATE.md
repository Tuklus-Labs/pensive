# Task state and schema v4 assertion loudness audit

All assertions in `test_checkpoints.py`, `test_schema_v4.py`, and
`test_portable_task_state.py` were reviewed against the four-box rule:

1. the failure names the rule or invariant;
2. the failure includes enough state to diagnose it;
3. the phrase is unique enough to search;
4. it states the rule in the present tense.

The tests route ordinary assertions through `_assert_rule`, which includes the
rule text and relevant state. `pytest.raises` assertions use specific exception
messages for CAS, replay, selector, missing-file, digest, and scope contracts.
There are no assertion exemptions in this scope.
