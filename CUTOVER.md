# Pensive v2 -> v3 cutover runbook

State as of 2026-07-03 (Phase A complete, flip pending canary):

| thing | state |
|-------|-------|
| v3 daemon | `pensive-v3.service` (systemd user), :5999, store `~/.local/share/pensive-v3/pensive.db`, 18,340 atoms migrated, embedded, integrity ok |
| legacy stack | still live and primary: `pensive-embed.service` (:8011), `pensive-vector.service` (:8009), `engram.service` (+ `engram-cpp-encoder.service`) |
| archives | `~/.local/share/pensive-v3/pre-cutover/`: legacy `vector_meta-<stamp>.db` (full record incl. `resolved_text`) + marker-stamped v3 snapshot |
| benchmark | v3 beats legacy +15-27% at every cutoff on this exact corpus; legacy trails plain BM25 (`research/2026-07-association-experiment.md` addendum) |

## What migrated, exactly

Fresh export of all 18,340 live legacy rows (nothing skipped) via the
Task 11-validated mapping: `summary` -> atom text, `doc_type` -> kind,
`[agent] on <slug>` prefix -> project/agent, `dates[0]` -> occurred_at,
`ingested_at` -> created_at override (history keeps its dates),
`src:` tokens -> tag facets, entities extracted (41,445 facets).
Production deviations from the eval export, both deliberate:
- importance seeded `min(1.0, 0.05 * retrieval_count)` (the eval zeroed
  it for gate purity; production inherits retrieval history).
- `resolved_text` (legacy long form) is NOT carried into atom bodies —
  the benchmarked mapping used summaries. The full long form lives
  permanently in the archived legacy snapshot; nothing is lost, it is
  archived rather than served.
Backfill is re-run-safe (sourceId dedup guard): re-running the migration
after more legacy writes ingests only the new rows.

## THE FLIP (canary-gated; ~2 minutes; run in this order)

1. Re-run the migration script once more (picks up any legacy atoms
   emitted since Phase A; idempotent).
2. Swap Claude Code registration (user scope):
   `claude mcp remove pensive`
   `claude mcp add --transport http pensive http://127.0.0.1:5999/mcp`
3. Swap Hermes: `hermes mcp remove pensive` then re-add pointing at
   `http://127.0.0.1:5999/mcp` (see AEGIS-INTEGRATION.md).
4. Retire legacy pensive units (NOT chrema — :8109/finance stays):
   `systemctl --user disable --now pensive-embed pensive-vector engram engram-cpp-encoder`
   (SIGTERM semantics; the old uvicorn ignores SIGINT ~25s.)
   `pensive-net-mcp.service` (:5998 tailnet) still targets the legacy
   stack — retarget or retire it in the same pass, Gary's call.
5. Verify: fresh CC session -> `pensive_recall` returns memories ->
   `engram_emit_atom` -> confirm the atom lands in
   `~/.local/share/pensive-v3/pensive.db` (atomCount +1) -> watch it
   fire on http://127.0.0.1:5999/viz.

## Rollback (any time, ~1 minute)

`systemctl --user enable --now pensive-embed pensive-vector engram engram-cpp-encoder`,
re-register the stdio MCP server (`claude mcp add pensive -- python3
/home/aegis/Projects/Engram/tools/pensive-mcp-server`), and the legacy
stack is primary again. The v3 daemon can keep running in parallel
harmlessly. Atoms emitted into v3 between flip and rollback would need
re-emitting or a reverse-export — rollback fast if rolling back.

## Decisions ratified with the canary

- PRIVACY: `recall_log` stores caller queries verbatim, local-only,
  currently without retention. Cutting over ACCEPTS indefinite local
  persistence (revisit at harness era) unless a retention window is
  stated with the canary.
- LATENCY: v3 recall measures ~300-500ms end-to-end vs legacy's 15ms
  (and vs the 150ms design p95). The quality uplift is the trade
  (+15-27%); latency tuning is a named post-cutover item.
- Ambient capture (distiller/drift) stays FLAG-OFF at cutover; enabling
  the ambient loop is Task 19/20-era follow-up, separately decided.

## Post-cutover expectations

- Restore-from-snapshot restores atoms/provenance/edges/facets/
  embeddings; `recall_log` (telemetry) and `supersession_proposals`
  (re-derivable) are deliberately not in the canonical export set.
- Engram-free ruling (Gary, 2026-07-03): v3 never wires to Engram
  services; the retired units are a checklist, not an integration
  surface. Chrema is out of scope.
- Task 21b (assoc serving wire-in) and the serve/mcp.py rename are the
  first two post-cutover work items (see the ledger defer list).
