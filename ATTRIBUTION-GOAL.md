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
- [x] Determine whether `clientInfo` is reachable per-request in stateless mode
- [x] Implement the precedence chain
- [x] Gate it: planted failures, both directions, aimed by claim
- [x] Wire the who-gate into a runner (a gate nothing invokes is a measurement)
- [x] Commit
- [x] Evaluate the restart path
- [x] Deploy and verify in production, both engines
- [x] Persist: memory, charon milestone, Pensive atom

## Result

Connection-declared attribution is LIVE. Verified in production, not inferred:

| probe | declared | landed |
|---|---|---|
| Heph, no agent argument | `?agent=heph` (`~/.claude.json`) | `heph` |
| Grok, no agent argument | `?agent=grok` (`~/.grok/config.toml`) | `grok` |
| Grok re-probe, after Claude's config also began declaring | `?agent=grok` | `grok` |

The third row is the one that mattered: both engines share the daemon and Grok
reads Claude-compatible config, so the risk was that declaring an identity for
one would silently stamp it on the other. Her own config wins. Nobody is
absorbed into anybody else's history.

Deployed config (dotfiles, not under version control; backups in the session
scratchpad):

- `~/.claude.json` -> `http://127.0.0.1:5999/mcp?agent=heph`
- `~/.grok/config.toml` -> new `[mcp_servers.pensive]`, `?agent=grok`

`~/.claude/CLAUDE.md` line 81 corrected: it claimed the NULL-agent problem was a
Grok-specific gap awaiting grok-kernel. Wrong on both halves.

## What deliberately did NOT happen

The 322,860 unattributed rows are untouched and stay that way. The fix is
forward-only by design. Total coverage therefore still reads 6.32% and will
climb only as new emits accumulate; the number to watch is the monthly
explicit-emit trend (`aegis-pensive-who --trend`), not the lifetime aggregate.
Backfilling identity by inference would have moved the headline number today and
manufactured provenance for work whose author is genuinely unknown.

## Log

Entries are written AFTER the thing happened, never before. An earlier revision
of this file had pre-filled results for work that had not run; that is the
counterfeit-record failure the 2026-08-03 letters name by name, and it was
caught and stripped rather than left to be read as measurement.

- 13:0x measured, instrument built and verified (30/30 gate, tool-level mutation
  reddened 2/30 narrowly), finding emitted to Pensive stamped `heph`
- 13:1x OPEN CRITICAL UNKNOWN: the daemon's transport config decides everything.
- 13:2x ANSWERED by reading the SDK, not guessing: `stateless=True` builds a
  fresh transport per request with `mcp_session_id=None`, `ServerSession` is
  constructed already-Initialized, and `_client_params` is only set by an actual
  `initialize` message landing on a DIFFERENT throwaway session. So
  `clientInfo` is None at emit time AND there is no session id to correlate a
  sniffed handshake against. The middleware approach sketched in the first draft
  of this file would NOT have worked. `RequestContext.request` (the Starlette
  request, attached by streamable_http) is the one genuinely per-call object.
- 13:3x implemented + gated. Planted the mechanism inert: P1 reddened alone,
  four other probes held, restored byte-identical (sha 921cf19ec0932d88).
  Full daemon suite 556 passed, 1 skipped (a rerank latency benchmark declining
  to measure under GPU co-tenancy, which is correct instrument behavior).
- 13:4x caught a defect in my OWN gate: `tests/test_pensive_who_gate.py`
  collected ZERO tests under pytest. Fixed, then swept the other file authored
  this session for the same shape (clean, 16 collected).
- 13:29 restarted `pensive-v3.service` (systemd user unit, supervised,
  `Restart=on-failure`). Functionally ready in 27s. My first readiness probe was
  broken, not the daemon: raw `tools/list` without the MCP handshake never
  answers. Audited the instrument before the subject.
- 13:3x-13:37 verified live in production across both engines, table above.

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
