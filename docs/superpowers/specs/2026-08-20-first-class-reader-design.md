# First-class reader (v3.1 leftover)

Date: 2026-08-20. Author: Grok. Campaign leftover from V3.1-AGENT-GRIPES.md.

The Aug 12-14 campaign fixed the swamp and the coffee-latency. Native `recall`
defaults to L2 (authored memory). This spec is the reader-identity half that
did not land.

## Measured 2026-08-20 (live daemon, not the gripe file)

- `GET /brief?agent=grok`, `?agent=heph`, and no-agent are byte-identical
  (`sha256 7eb580a1848d1f9a`). Agent only drives empty `for:<agent>` loose-ends.
- Active threads are the fleet recency pool (Theia). Pins are global and
  currently eat the budget.
- Native `recall` L2 is clean. `pensive_recall` still mixes bulk-import
  listings. `handle_recall` extracts `kinds` and does not pass it.
- Connection `?agent=grok` stamps writes. Retrieve has no agent filter.
- Grok Build SessionStart **ignores stdout**. Claude's
  `session-brief-v3.sh` additionalContext JSON would be dropped.
- Identity over-lock `p3://01KZ0G1J7BHWM7F87YZE7YKW3V` is live and pinned.
  Correction `p3://01KZ0Q4DDE3WE9R3XKZ4QD6BEB` is live, unpinned, no
  supersedes edge. Both `provenance.agent` NULL.
- Store: 51 live grok-stamped atoms, 261 heph, ~287k NULL. Do not infer NULL.

## Contract

### `/brief`

| `agent` | `pinned:` | `active:` | `loose ends` |
|---|---|---|---|
| unset / blank | all live pins (unchanged) | fleet recency+importance, memory kinds only | omitted |
| set | all live pins (household standing, shared) | that agent's live memory-kind emits (`provenance.agent` match), ranked the same way | existing `for:<agent>` |

Active MUST be a provenance join, not a filter of the global 500. Theia
currently occupies the recent tail; filtering 500 fleet recents by `agent=grok`
would return empty and look like a green no-op.

`document_chunk` never appears in `active:`. Pins stay global: they are standing
principles, not "who pinged last." Agent-attributed pins need a pin-author
column we do not have; do not invent one this round.

Blank `agent` is unscoped. VIEW remains zero-write, live-only.

Acceptance: grok brief and heph brief differ. Grok active does not open with
Heph's last project. A grok emit older than 500 foreign recents still appears.

### `recall` / `recall_records`

Optional `agent` (string, or list of strings). Default unset = no filter.
When set, keep candidates whose provenance has that agent. NULL provenance is
excluded. Connection `?agent=` does **not** auto-scope recall (that would hide
the house from Grok).

`handle_recall`: if the caller passed `kinds`, honor them and do not let default
tier L2 override. Otherwise `tier=L2`. Always `rerankEnabled=False` on the
agent-facing path (already true for tiered recall).

`pensive_recall` (legacy listing): pass `tier=L2` so the tool agents actually
call stops walking chunks. Do not collapse the three retrieve names this round.

### Grok session start

Grok SessionStart stdout is ignored. SessionStart (and SessionEnd) hook writes
`~/.grok/rules/pensive-brief.md` from `GET /brief?agent=grok`. Grok auto-loads
`~/.grok/rules/*.md`. Fail-open: curl/jq/daemon failure exits 0 and leaves any
existing file. Never block session start. Do not edit `~/CLAUDE.md`.

### Identity data (after code)

`store.supersede(over-lock, correction)` with `source=explicit-emit`,
`agent=grok`. Pin the correction. Brief is live-only, so the superseded pin
vanishes. No NULL-agent backfill.

## Non-goals

- Schema v4 `idx_prov_agent` until a measurement shows the join is the brief
  budget. 51 grok rows do not justify a 1.6GB-store migration this round.
- Divergent association / L3 walk.
- Collapsing three retrieve tool names.
- Inferring authorship of NULL rows.
