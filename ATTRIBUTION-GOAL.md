# Goal: make Pensive attribution free rather than remembered

Opened 2026-08-12 by Heph (opus-5) under `/goal`. Gary: "Keep going... Be a part
of the ecosystem."

## The request

Attribution in the Pensive v3 shared memory is 6.32% covered and decaying
(43.1% July -> 19.2% August). The capability to stamp `agent` per call already
exists (commit `5ce1d92`); nobody uses it because it is optional and invisible.
Instruction repair does not propagate. Incentive repair does.

Measured by `aegis-pensive-who` (~/Projects/scripts, branch
`feat/pensive-attribution-instrument`, commit 3bde433):

| | rows |
|---|---|
| total provenance | 344,624 |
| attributed | 21,764 (6.32%) |
| NULL | 322,860 |

## The fix

Stamp `agent` from the MCP `initialize` handshake's `clientInfo.name` when the
caller supplies none. Precedence, most specific wins:

1. caller-supplied `agent` argument (existing behavior, unchanged)
2. **NEW**: the MCP session's `clientInfo.name`
3. daemon-wide `PENSIVE_V3_AGENT` env (existing)
4. NULL

## Hard constraints

- **No inference backfill.** The 322,860 NULL rows stay NULL. A row count can
  justify a hypothesis about who was working that week; it cannot establish
  provenance. This is the standing correction from the 2026-08-03 letters.
- **Read-only until proven.** No writes to the live store outside normal emits.
- **The daemon holds a 1.6GB live store** and serves the whole fleet. Any
  restart is an operational act, evaluated on evidence, SIGINT first.

## Todo

- [x] Measure the gap with a verified instrument (`aegis-pensive-who`, 30/30 gate)
- [ ] Determine whether `clientInfo` is reachable per-request in stateless mode
- [ ] Implement the precedence chain
- [ ] Gate it: planted failures, both directions, aimed by claim
- [ ] Wire the who-gate into a runner (a gate nothing invokes is a measurement)
- [ ] Commit
- [ ] Evaluate the restart path
- [ ] Persist: memory, charon milestone, Pensive atom

## Log

Entries are written AFTER the thing happened, never before. An earlier revision
of this file had pre-filled results for work that had not run; that is the
counterfeit-record failure the 2026-08-03 letters name by name, and it was
caught and stripped rather than left to be read as measurement.

- 13:0x measured, instrument built and verified (30/30 gate, tool-level mutation
  reddened 2/30 narrowly), finding emitted to Pensive stamped `heph`
- 13:1x OPEN CRITICAL UNKNOWN: the daemon's transport config decides everything.
  If `clientInfo` is not reachable at tool-call time, this approach dies and
  needs a different carrier. Going to read the code rather than assume.
