# Sabotage Log: caller-supplied `agent` on the compat emit tools

Porchlight Lane 2 Task 9 (ruling R11.1). Every mutation below was applied to the working
tree, run against the named tests, and restored before the final suite. Restoration was
verified by md5 of `daemon/src/serve/mcp.py` after each pair, not by eye:
`d29fc693c63779c8242d614c0c9334cd`.

Neither `mutmut` nor `cosmic-ray` is installed, so this follows the same manual sabotage
procedure as `SABOTAGE_LOG_STRUCTURED_RECALL.md`.

All five flips and both green runs are one epoch: same checkout, same interpreter, same
session, no restart of `pensive-v3.service` at any point.

## Epoch

| Run | Command | Result |
|-----|---------|--------|
| Baseline, before any edit | `python3 -m pytest daemon/test -q` | 524 passed, 1 skipped |
| Red, tests written, production untouched | `python3 -m pytest daemon/test/serve/test_mcp.py -q` | 9 failed, 36 passed |
| Green, after implementation | `python3 -m pytest daemon/test -q` | 539 passed, 1 skipped |

The +15 is exactly the fifteen new test cases. The one skip is pre-existing and
self-declaring: `test_rerank.py:269` refuses to report a latency number while the GPU is
at 100% co-tenant load. It is not a silent pass, and it is present in the baseline too.

## Production mutations

| Mutation | Predicted result | Observed result | Conclusion |
|----------|------------------|-----------------|------------|
| Make the daemon-wide default win: `if agent is None and ctx.agent:` becomes `if ctx.agent:`. | The precedence test fails; the no-default test survives. | `test_a_caller_agent_beats_the_daemon_default`: failed, `assert 'heph' == 'grok-agent'`. 15 passed. | Caller-over-default precedence is load-bearing, not incidental. |
| Stamp whatever the caller sent: drop the `candidate.strip()` truthiness guard and the strip. | Both whitespace cases fail; the wrong-typed cases survive, since they never reached the guard. | `..._falls_back_rather_than_stamping_junk[]` and `[   ]`: failed, `assert '   ' == 'heph'`. 14 passed. | A blank name cannot reach provenance. An atom stamped `""` would look like an answer. |
| Stop forwarding `agent` through the delegated argument dict in `handle_emit_discovery` / `handle_emit_failure`. | Exactly those two tools fail; atom, narrative, and snapshot survive. | `test_every_emit_tool_stamps_the_caller_agent[engram_emit_discovery]` and `[engram_emit_failure]`: failed, `assert None == 'fable-agent'`. 14 passed. | The two delegating tools are covered by a real emit, not only by schema presence. Without this they would accept an agent, return success, and stamp nothing. |

## Mutations against the pin this task modified

`test_compat_tool_schemas_are_verbatim` previously compared each compat `inputSchema`
whole. It now subtracts exactly the `agent` key from the five emit tools before comparing.
Weakening an existing gate obliges proving the remainder still bites, so both directions
were flipped:

| Mutation | Predicted result | Observed result | Conclusion |
|----------|------------------|-----------------|------------|
| Drift an unrelated description: `engram_emit_atom.approach` becomes "What was attempted". | The pin fails. | `test_compat_tool_schemas_are_verbatim`: failed. | Ordinary legacy drift is still caught. The subtraction did not blind the pin. |
| Add a SECOND uninvited property (`sessionId`) beside `agent` on `engram_emit_atom`. | The pin fails. | `test_compat_tool_schemas_are_verbatim`: failed. | The subtraction is keyed to one named property, not to "ignore extras". A future additive edit cannot ride in behind this one. |

## What is deliberately not claimed

The live end-to-end stamp against the running daemon is NOT in this log. `pensive-v3.service`
serves the house's real memory store and the restart that would activate this change is an
operator-gated deploy step (campaign task C7). Nothing here was proven against the live
daemon, and no number in this file should be read as if it were.
