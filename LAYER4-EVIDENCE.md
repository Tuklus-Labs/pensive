# Layer 4: the store holds case law, and the statutes had to move out

Measured 2026-08-13 against the live store, read-only. Companion to
`LAYER5-EVIDENCE.md`. Same rule: ground truth before design.

## What the store is actually made of

| kind | live atoms | share |
|---|---|---|
| `document_chunk` | 282,985 | **94.41%** |
| `atom` | 16,079 | 5.36% |
| `narrative` | 447 | 0.15% |
| `snapshot` | 217 | 0.07% |

There is no `rule` kind. There has never been one.

A crude shape check (does the text contain `def (`, `func (`, `class :`,
`import `, `t.Errorf`, `assert`) puts **95,633 live atoms, 31.9% of the store,
in source-or-test-code shape**. The heuristic is rough and will over-count prose
that quotes code, but the direction is not in doubt: roughly a third of what
semantic recall searches is source text, and 94% of it is bulk-imported chunks
rather than anything an agent chose to remember.

That is the swamp. 16,079 reasoning atoms compete against 282,985 document
chunks for the same ranked slots, and the chunks win on volume.

## The evidence that this already broke something

The rules that must fire every session are not in the store.

| probe | live atoms |
|---|---|
| "error-message reflex" | 4 |
| "continuity covenant" | 1 |
| "no bedtime advice" | **0** |
| "exploit paranoia" | **0** |
| "dollars vs completion" | **0** |

Three of five are absent outright. The two that exist carry `kind='atom'` (3)
and `kind='narrative'` (1), which is to say they are filed identically to any
other remembered reasoning and rank against 283k chunks on the same scale.

Meanwhile ten rule files sit in `~/.claude/projects/-home-aegis/memory/`,
indexed by `MEMORY.md`, loaded unconditionally by the harness at every session
start.

They are not one kind of thing, which matters for any tier built to hold them:

| frontmatter type | count | examples |
|---|---|---|
| `user` | 5 | `user_gary_2026q1`, `family-crisis-2026-07`, `user_accessibility` |
| `feedback` | 4 | `heph-rule-error-message-reflex`, `feedback_no_exploit_paranoia` |
| `project` | 1 | `pensive-attribution-contract` |

Only the four `feedback` entries are rules in the statute sense: general
imperatives that should fire on a *situation*. The five `user` entries are
durable biographical context, which wants to be resident rather than retrieved,
and the single `project` entry is a technical contract closer to documentation.
A `kind='rule'` tier that swallows all ten is three different jobs wearing one
name. The behavioral four are the honest seed.

**That arrangement is the finding.** The statutes were moved out of the memory
system and onto the filesystem, where a loader could guarantee they surface,
because the memory system could not. `MEMORY.md` is not a feature that happens
to live alongside Pensive. It is a workaround for a tier Pensive does not have,
and it is load-bearing: the two resident rules inlined directly into `CLAUDE.md`
carry a comment saying they stay there because they *must* fire every session.

A memory system whose most important memories had to be stored somewhere else
has a naming problem, not a tuning problem.

## Why `document_chunk` is not the villain

Worth stating plainly so the fix does not overcorrect. The chunks are not junk
and deleting them is not the answer. Bulk-imported source is exactly what you
want when the question is "how did we implement the CSR publish" or "where does
the HMAC gate live." L3 deep archive is their tier and they earn it.

The defect is that there is no tier *above* them for statements that are true
regardless of query. A rule is not a better chunk. It is a different kind of
object: general where a chunk is specific, durable where a chunk is
point-in-time, and it should surface on relevance to the *situation* rather than
on lexical overlap with the query. Ranking them on the same scale means a rule
only wins when the operator happens to phrase a query that lexically resembles
it, which is precisely when they least need reminding.

## What a `rule` kind has to do to be worth adding

Stated as constraints so a design can fail them:

1. **Surface without being asked for.** If a rule only appears when the query
   matches it, it has not solved the problem the filesystem loader already
   solves better. The test is whether the error-message reflex surfaces during a
   debugging session that never uses the words "error message reflex."
2. **Be scarce, and stay scarce.** Ten files today, four of them true rules. A rule tier that grows to
   thousands has become another chunk tier with a nicer name. Whatever writes
   rules must be much more reluctant than whatever writes atoms.
3. **Not depend on importance accrual.** Per `LAYER5-EVIDENCE.md`, importance is
   currently a 1.0-capped scale with 363 atoms already saturated at the top. A
   rule tier built on top of that inherits a plateau.
4. **Survive the migration honestly.** The four `feedback` files are the seed corpus.
   Importing them is a one-time act with a known input, so there is no excuse for
   a fuzzy result: four in, four retrievable by situation, verified
   individually rather than by count.

## Status

Nothing built. This is what a Layer 4 design has to survive. The measurement
that would most change my mind: if a rule filed as `kind='rule'` and given a
retrieval path still fails constraint 1, then the tier is cosmetic and the
filesystem loader should stay the mechanism.

---

## Design returned 2026-08-13, and two of its hazards verified independently

A design pass came back against the constraints above. It clears all four, and
it reframes the layer better than this document did.

**The reframe.** I had been treating Layer 4 as "build a retrieval tier for
rules". Its answer: the store owns EDITING and VERSIONING, the file keeps
DELIVERY. Files won because editing a file changes what arrives tomorrow, so the
fix is to make editing a rule by slug be the thing that edits the file. Rules
stay structurally invisible to the ranker. That responds to the finding at the
top of this document without pretending retrieval can do a job that
unconditional loading already does well.

Mechanically: `kind='rule'` atoms carrying `rule_scope` / `rule_slug` /
`rule_rank`, one live rule per (scope, slug) enforced by a PARTIAL UNIQUE INDEX
rather than by convention, edits as supersede-then-insert in one transaction so
the existing `supersedes` machinery and `history` work unchanged.

**Two hazards, verified against the source rather than taken on report:**

1. `lifecycle/supersede_detect.py::_candidateRows` selects `WHERE status='live'`
   with NO kind filter (confirmed, line 54). Wiring Layer 5's supersession
   detector before a rule exclusion exists would let standing rules
   auto-supersede each other on embedding similarity. This is a hard ordering
   dependency between layers, not a courtesy.
2. `store/export.py::ATOM_COLS` (confirmed, line 34) is an explicit 9-column
   tuple imported by `store/rebuild.py`. Adding rule columns to the table
   without adding them there means an export-then-rebuild cycle silently drops
   every rule's identity. A backup and restore would destroy the rule layer
   while reporting success.

**The finding that outranks the rest.** Rule text becomes instructions loaded
into every session on this box. `/mcp` is Origin- and Host-guarded but
deliberately NOT secret-guarded, because every wired agent speaks it. A rule
WRITE tool on that surface would let anything able to POST to loopback rewrite
the operating instructions of every session on this machine. The mitigation is
structural: read-only rule tools on MCP, writes behind the existing 0600
loopback secret. This is an Aegis Guardian surface and it outranks every
latency consideration in the layer.

**The number that may kill the layer as scoped.** The live standing-rule corpus
measures 22 rules, 55,910 chars, roughly 13,985 tokens, and it lands in EVERY
session's context. That is 9.3x the entire brief budget. Constraint 2 above said
"be scarce, and stay scarce"; the measurement says the corpus is already not
scarce. Whatever ships has to answer that before it answers anything about
latency.

**One piece of discipline worth copying.** The rule-serving query costs 68ms
without an `INDEXED BY` hint, because production carries no `sqlite_stat1`. The
obvious fix is `ANALYZE`. The design refused it: `ANALYZE` costs 786ms and
rewrites statistics for every index in the database, which would silently
re-plan bm25 and the L2/L3 signal path and therefore re-certify or de-certify
tier numbers measured the same day. It took the hint and left `ANALYZE` as a
separate change carrying its own re-certification.
