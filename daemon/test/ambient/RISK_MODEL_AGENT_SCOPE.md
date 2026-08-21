# Risk Model: first-class reader (agent-scoped brief + retrieve)

Unit: `ambient.briefer.brief`, `recall.engine.recall` agent filter,
`serve.mcp` handle_recall/handle_pensive_recall/handle_recall_records,
`daemon/hooks/session-brief-v3-grok.sh`.

Live bug this exists to catch: `/brief?agent=grok` byte-identical to heph;
`pensive_recall` serving chunks; retrieve ignoring who asked.

## Axis: Invariants

- I1. `/brief?agent=A` active ids are a subset of live atoms whose provenance.agent is A.
- I2. `/brief?agent=A` and `/brief?agent=B` differ when A and B have different recent emits.
- I3. Unscoped `/brief` (agent None or blank) still shows fleet active (operator view).
- I4. Pins remain shared across agents (standing principles). A pin on a heph-authored atom still appears in grok's brief.
- I5. `document_chunk` never appears in `active:`.
- I6. brief() writes zero rows (existing view property).
- I7. Superseded atoms never appear (existing live-only).
- I8. recall(agent="grok") returns only grok-stamped candidates; NULL-agent rows are dropped.
- I9. recall() with agent unset is byte-identical to today (no silent scope).
- I10. Connection `?agent=` does not auto-filter retrieve.
- I11. pensive_recall default corpus is L2 memory kinds; a matching document_chunk is absent from the listing.

## Axis: State transitions

- S1. Agent unset -> fleet active. Agent set -> that agent's active. No third "half-filtered" state.
- S2. Foreign recent tail of size ACTIVE_CANDIDATE_CAP must not hide an older same-agent emit (join, not post-filter of the global 500).
- S3. N/A for retrieve: filter is per-call, no session state.

## Axis: Boundaries

- B1. Empty store: `no memory yet` (existing).
- B2. Agent with no emits: pinned (if any) + no active section, not a cloned fleet active.
- B3. Blank agent string equals unset.
- B4. Agent with only NULL provenance rows: treated as no emits for that agent.
- B5. Atom with two provenance rows (heph and grok): eligible for both briefs and both agent filters.
- B6. ACTIVE_CANDIDATE_CAP+1 foreign recents plus one buried own emit: own emit still present.

## Axis: Malformed inputs

- M1. Unknown recall_records field still rejected (agent is now allowed).
- M2. Agent that is a path (`/root/...`) is not a retrieve filter identity; sanitizer returns None = unscoped, never a forged stamp. Brief uses the query param as a name for SQL match, not as a write.
- M3. kinds=[] on recall already errors in recall_records; handle_recall kinds must not be silently dropped.

## Axis: Concurrency

- N/A: brief and recall are read-only per call on the daemon's event-loop thread. No new writer. SQLite WAL already covers readers.

## Axis: Persistence

- P1. No schema bump this round. Join uses existing idx_prov_atom / table scan of provenance.agent. 51 grok rows. A future v4 index is out of scope until measured.
- P2. Grok hook: failed refresh must not truncate an existing `pensive-brief.md`.
- P3. Identity supersede is a live status change, not a delete. Old text remains via getAtom.

## Axis: Integration contracts

- C1. Existing test_briefer cases that brief as heph with `_put(..., agent="heph")` stay green: heph-scoped active still contains those atoms.
- C2. recall_records exact engine-kwargs tests stay green: `agent` is omitted from the call when unset, not passed as None (those tests pin the keyword dict).
- C3. pensive_recall still returns the legacy listing envelope.
- C4. Grok SessionStart stdout ignored: hook succeeds by writing a rules file, not by printing additionalContext.
- C5. `~/CLAUDE.md` is not this contract. Heph's "Check Pensive." stays.

## Axis: Regression traps

- [x] boundary: empty collection treated as missing collection (agent with no emits must omit active, not substitute fleet).
- [x] boundary: zero treated as falsy (`if agent` must not treat a legitimate need-to-filter as skip). Agent name is a non-empty string; blank is unset. Test B3.
- [x] concurrency: N/A, single-threaded read.
- [x] contract: field rename / silent ignore (`kinds` extracted and dropped on handle_recall).
- [x] encoding: N/A, agent is a TEXT column already.
- [x] framework: N/A.
- [x] io: command-line / hook fail-open (daemon down must not block session; must not clobber file).
- [x] persistence: N/A for schema this round; hook file atomic replace.
- [x] resource: N/A.
- [x] state: cache/stale (filter-the-global-500 is a stale-window bug: S2).

## Coverage Matrix

| Risk row | Test name(s) |
|----------|----------------|
| I1 I2 B2 | test_agent_active_excludes_other_agents_recent_work |
| S2 B6 | test_agent_active_survives_foreign_recent_tail |
| I3 B3 | test_blank_agent_is_unscoped_fleet_active |
| I4 | test_pins_remain_shared_across_agents |
| I5 | test_document_chunk_never_in_active |
| I6 | test_view_property_zero_writes (existing) |
| I7 | test_superseded_atom_never_appears_in_any_section (existing) |
| I8 I9 | test_recall_agent_filter_keeps_stamped_drops_null_and_other |
| I10 | (integration: handle_recall does not read ctx.agent as a retrieve filter) test_handle_recall_does_not_autoscope_from_transport |
| I11 C3 | test_pensive_recall_default_excludes_document_chunk |
| M3 | test_handle_recall_honors_kinds |
| C2 | test_recall_records_forwards_agent_only_when_set |
| P2 C4 | test_grok_hook_daemon_down_preserves_existing_brief |
| P2 | test_grok_hook_writes_brief_via_fake_curl |
