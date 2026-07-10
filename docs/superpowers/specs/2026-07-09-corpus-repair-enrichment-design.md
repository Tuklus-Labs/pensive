# Corpus repair and enrichment (Layers 2 + 3)

Date: 2026-07-09
Status: draft, pending approval
Scope: `daemon/` store repair + recall payload enrichment + edge materialization.
Depends on: stratified recall (Layer 1, merged 240009a, live and verified).

## What the investigation changed

The Layer 1 spec sketched Layer 2 as "a re-ingest that recovers file/symbol
provenance" for 299k chunks. Direct inspection of the store killed that premise:

| Fact | Consequence |
|------|-------------|
| 295,438 of 299,347 chunks already carry `relpath#cN` source_refs | No re-ingest. Provenance is file-level intact. |
| Only 3,909 chunks have `kv_cache/vector_meta.db#rowid=N` refs, and that retired store still exists (18,350 rows; its `summary` field embeds the original absolute path) | Those refs are recoverable by a rowid join + path extraction. Deterministic batch. |
| The reference library is imported TWICE: 16,362 chunks under `reference-library/...` (project NULL) and 16,393 under `projects/Aegis/AEGIS/docs/reference-library/...` (project Aegis) | ~16k duplicate chunks compete in recall today. Dedup via supersession, not deletion. |
| Chunk TEXT is stored in full | Symbol/line resolution can happen lazily at recall time by locating the chunk's text span in its file, for top hits only. No batch re-chunking; the original chunker is irrelevant. |
| The edges table holds 1 edge total | Layer 3's association graph must be built; this is the genuinely fleet-scale piece. |

## Phase A: store repair batch (small, deterministic)

All mutations gated behind a fresh backup (`pensive.db.bak-pre-repair-<date>`,
following the established naming) and performed by reviewed, tested code in
`daemon/tools/`, never ad-hoc SQL.

1. **kv_cache ref recovery (3,909 chunks).** Join `rowid` into the retired
   `~/Projects/Aegis/AEGIS/Pensive/kv_cache/vector_meta.db`, extract the
   absolute path from `summary` (pattern `[files] file /home/... <action>`),
   rewrite `source_ref` to the same `relpath#cN`-style convention (relative to
   `~/`, fragment omitted where unknowable: a bare relpath is honest, a fake
   chunk index is not), and backfill `atoms.project` from the path's
   `Projects/<name>/` segment. Rows whose summary yields no path keep their
   rowid ref and are counted in the report; never guess.
2. **Reference-library dedup (~16k supersessions).** For each `reference-library/<f>#cN`
   chunk with a content-matching `projects/Aegis/AEGIS/docs/reference-library/<f>#cN`
   twin, write a `supersedes` edge (survivor: the `projects/...` copy, which has
   project attribution) and mark the null-project copy superseded. Match on
   (filename, chunk index, exact text); text-mismatched pairs (the ~31 drifted
   docs) stay live and are reported, not forced.
3. **Residual project backfill.** Any remaining `project IS NULL` chunk whose
   source_ref starts with `projects/<name>/` gets that project. Reference-library
   chunks that survive with null project get project `Aegis` (they live inside
   the Aegis repo).

**Verification invariants (each a test or a post-run assertion):**
- Zero atoms deleted; live+superseded total constant.
- Every rewritten ref resolves to an existing file OR is flagged in the report
  (files legitimately deleted since import are reported, not errors).
- Post-repair recall for a known reflib query returns no duplicate-text pair in
  one result set.
- The repair tool is idempotent: a second run is a no-op.

## Phase B: serving-time enrichment (the Layer 3 payload, lazily)

When a recall result is a `document_chunk`, the tiered payload gains two
attachments, computed at serve time for surfaced hits only (top-k is small;
milliseconds each):

1. **Resolved location.** Locate the chunk's stored text in the file named by
   `source_ref` (whitespace-normalized search). Emit `path:startLine-endLine`.
   File missing or text not found: emit the bare ref, never a guess. No
   language-aware symbol parsing in this phase; the line span plus Tier-2 body
   is what "point me at the code" needs. (A tree-sitter enclosing-symbol pass is
   a possible later upgrade; explicitly out of scope now.)
2. **Related memory.** Spreading activation over what exists: memory-kind atoms
   sharing entity facets with the chunk (and, weaker, same project within a
   time window), ranked by shared-facet specificity, top 2-3 attached as
   one-line gists with their atom handles. This is the "what we decided and why,
   next to the code" payload. Uses the existing facet indexes; no new tables.

Budget rules unchanged: attachments participate in the token budget and degrade
first (drop related-memory lines, then the location line, before touching the
result bodies).

## Phase C: edge materialization campaign (the fleet-scale piece)

Materialize chunk-to-memory `relates` edges so the graph outlives serve-time
heuristics: trust explanations, the viz, and future traversal all read edges.

- **Propose:** for each memory atom (14.2k), candidate chunks by shared entity
  facets + same project + occurred_at proximity. Cap proposals per atom;
  specificity-weighted (rare facets count, `FILES`-type hub facets do not).
- **Verify:** adversarial agent waves judge each proposal ("does this memory
  actually concern this code?") with the atom text and chunk text side by side.
  Majority-refute kills. This is the token-heavy stage and is embarrassingly
  parallel (Workflow; fleet doctrine applies).
- **Write:** surviving edges land as `relates` with weight = verifier
  confidence, provenance stamped `distiller`-style with the campaign session.
  Idempotent: re-runs update weight, never duplicate (UNIQUE on src/dst/type
  enforced in the writer).
- **Gate:** sample-audit N random written edges at the end; a failure rate over
  threshold rolls the batch back (edges are deletable-by-campaign-provenance
  without touching anything else; this is the one place deletion is acceptable
  because the campaign owns its own rows).

Phase C consumes Phase A (clean provenance improves facet quality) and feeds
Phase B (related-memory lookup prefers real edges over live facet joins once
edges exist).

## Order and shape

Phase A first (spec-plan-SDD, small). Phase B second (same rhythm as Layer 1;
engine/payload change with behavioral tests). Phase C last, as a Workflow
campaign under the fleet doctrine, sized to the token budget available.

## Out of scope

- Re-chunking or re-importing any existing corpus content.
- Tree-sitter/ctags symbol naming (noted as a Phase B upgrade path).
- Any deletion of atoms (supersession only; Phase C may delete only its own
  campaign-provenance edges on rollback).
- The dotfiles-wave content audit (whatever the 1.2GB pre-dotfiles import
  brought in is a separate conversation).
