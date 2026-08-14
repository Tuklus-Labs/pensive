# Layer 3: what semantic adjacency has to work with

Measured 2026-08-13 against the live store, read-only. Third of the evidence set
with `LAYER4-EVIDENCE.md` and `LAYER5-EVIDENCE.md`.

Adjacency was Codex's filing, and Gary named it directly in the performance
goals ("semantically adjacent retrievals"). Before designing it, this is the
state of the substrate it would run on.

## Coverage is complete

| | |
|---|---|
| live atoms | 299,728 |
| embedding rows | 340,722 |
| **live atoms with no embedding** | **0** |

Every live atom is embedded. Adjacency is structurally possible corpus-wide,
which is not something I assumed going in and is the good news of this document.

## The store holds two models, and that is handled correctly

| model_id | rows | dims | vector bytes |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 320,839 | 384 | 470.0 MB |
| `text-embedding-3-large` | 19,883 | 3072 | 233.0 MB |

19,883 atoms carry two embeddings of incompatible dimension. That looked like a
serious defect for about a minute. It is not: both search paths filter on
`model_id` before loading anything, at `vector_index.py:98` and
`hnsw_index.py:68`, and the schema keys embeddings on `(atom_id, model_id)`. An
index is built for exactly one model and never mixes.

This is the correct design and it is worth naming, because it is the structural
answer to the near-miss earlier in this campaign: a GPU embed service returned
384-dim vectors that matched bge's dimension while scoring cosine 0.149 against
bge on identical text. **Dimension is not identity.** Keying on `model_id`
rather than on vector width is what makes that class of mistake survivable.

## 233 MB belongs to a feature that is switched off

The `text-embedding-3-large` rows are the `aux_dense` path, the remote OpenAI
embed I filed as finding #7 and which was disabled on 2026-08-12 via
`70-aux-dense-off.conf`. They cost 8x per atom versus bge (3072 dims against
384) while covering 6% of the corpus, and nothing reads them while aux dense is
off.

I checked whether that dead weight is also a latency tax, on the theory that the
index build might table-scan every embedding page. It does not. The plan drives
from `atoms` via `idx_atoms_status` and seeks the embeddings primary key:

```
|--SEARCH a USING INDEX idx_atoms_status (status=?)
|--SEARCH e USING INDEX sqlite_autoindex_embeddings_1 (atom_id=? AND model_id=?)
```

So the 233 MB is cold. It costs disk, not query time.

`lifecycle/reembed.py:dropOldModel` already exists to reclaim it, and
regenerating those vectors later would cost roughly a dollar of API time, so the
decision is close to reversible. It is still a `DELETE` against the live store,
which is a destructive op under Guardian and needs Gary's word, not mine. Filed,
not executed. There is no urgency: 13.7% of a 1.7 GB file is not hurting
anything measured.

## The background cosine of this corpus, which an adjacency threshold must clear

Measured as a side effect of verifying the ONNX encoder. The negative control
there pairs each embedding against a DIFFERENT text's embedding, which makes it
a sample of what "unrelated" scores in this store. Over 375 real texts (250 live
atom bodies, 125 real queries):

| statistic | cosine |
|---|---|
| median | 0.6458 |
| p90 | 0.8383 |
| p99 | 0.9233 |
| max | **0.9756** |

This is the number that governs adjacency design, and it is uncomfortable. A
pair of *unrelated* atoms in this corpus scores 0.65 at the median, and one in a
hundred scores above 0.92. The maximum, 0.9756, is not encoder error: it is two
genuinely near-duplicate texts, which this store is full of because the same
source chunks were re-emitted thousands of times (see the 84.6% stitcher share
in `LAYER5-EVIDENCE.md`).

Two consequences:

1. **An absolute cosine threshold for "semantically adjacent" is not viable
   here.** Any cutoff low enough to catch real adjacency (0.85, say) sits below
   the 90th percentile of *random* pairs. Adjacency has to be relative (rank
   against the rest of the candidate set) rather than absolute.
2. **The high tail is partly an artifact of duplication, not of meaning.**
   Before tuning any adjacency metric, the near-duplicate population needs to be
   understood, or the metric will be tuned against Charon's echo rather than
   against the corpus.

This also retro-justifies a verdict elsewhere in the campaign. The MiniLM
candidate scored cosine 0.2597 against current output while its own negative
control scored 0.5337. Against the distribution above, 0.2597 is not merely low,
it is below what unrelated text scores, which is the signature of a different
embedding space rather than a worse view of the same one.

## What is NOT yet measured

Adjacency itself. Whether recall actually returns semantically adjacent results,
as opposed to lexically overlapping ones, requires running real queries through
the encoder, and the box has been at load 40 to 50 all evening with four
benchmark agents on it. Any retrieval-quality number taken now would be
measuring contention.

Stating that as a gap rather than filling it with a number I would have to
caveat. The substrate facts above hold regardless of load, because they are
schema and query-plan facts. The quality measurement waits for a quiet box.

## What this implies for the design

1. **Do not build adjacency on a second embedding model.** The two-model
   machinery works, but every additional model is another index, another
   in-memory matrix, and another chance for a dimension-matched impostor. bge
   covers 100% of live atoms today.
2. **The swamp is the real obstacle, not the vectors.** Per
   `LAYER4-EVIDENCE.md`, 94.41% of live atoms are `document_chunk`. Adjacency
   over that corpus will return adjacent *source code* unless kind-awareness is
   part of the ranking. Better vectors do not fix a corpus composition problem.
3. **Measure adjacency against a held-out probe set, on a quiet box, with a
   negative control.** The cosine 0.149 near-miss happened because a plausible
   number arrived without a control beside it. Any adjacency claim needs the
   same discipline: what does an unrelated pair score, and is the gap real.

---

## Design returned 2026-08-13: the substrate is far thinner than the filing implied

Verified the design's most decisive claim at the source rather than on report.

### Every typed edge points from code at memory. None link two memories.

```
  src_kind        dst_kind    edges
  document_chunk  atom         2452
  document_chunk  narrative     357
  document_chunk  snapshot        1
```

All 2,810 `relates` edges are `document_chunk -> memory`. There are ZERO
memory-to-memory edges. The typed-edge layer is not a graph between ideas, it is
an annotation pointing from a piece of code at the memory that explains it.

That single fact redirects the design. An edge hop cannot go memory to memory
because no such edge exists; it has to run backward then forward,
`memA <- chunk -> memB`, which reads as "two memories that explain the same
code". `idx_edges_dst (dst_atom, type)` already exists, so the traversal is
indexed and costs nothing new.

### The connective material is ~30x thinner than the filing counted

Codex's filing cited 675,876 entity facets over 211,761 distinct values as
evidence there is enough to traverse. Those numbers are right and they describe
the CHUNK corpus. Restricted to the 16,745 live memory-kind atoms that divergent
ideation should actually walk:

| substrate | whole store | memory kinds only |
|---|---:|---:|
| distinct entity values | 211,761 | 4,654 |
| bridge-eligible (degree 2..100) | 54,308 | **601** |
| tag values bridge-eligible | - | **900** |
| memory atoms with any bridge facet | - | **2,710 of 16,745 (16.2%)** |

I re-measured the singleton share independently and got **70.0%** (148,153 of
211,761) against the design's 74.2% (157,108). Same denominator, different
numerator, most likely because I counted `COUNT(DISTINCT atom_id)` (atoms
actually bridged) where the design counted facet rows, and its snapshot predates
tonight's writes. Recorded rather than smoothed over. The conclusion survives
either number: roughly three quarters of entity values appear on exactly one
atom, and a degree-1 facet bridges nothing while still costing index space.

### Tag is the bridge namespace; entity is the noisy secondary

| in the bridge band (degree 2..100) | entity | tag |
|---|---:|---:|
| bridge-eligible values over memory | 601 | 900 |
| share of facet rows that are memory | **1.16%** | **75.4%** |
| share of values that are wordlike | 25.6% | 97.3% |

Entity extraction was tuned for a code corpus and it shows: 98.8% of entity
facet rows describe chunks, and three quarters of the values that do survive the
degree filter are not word-like. Tag is the namespace memory actually populates.

### Consequence for anchoring

A single anchor produced candidates for 9 of 60 random live memory atoms, and
only 7 of 60 reached the three-candidate bar. With a 10-anchor set: 29 of 40
trials reached three candidates. The design states its own optimism honestly,
noting that random anchor sets sample more of the graph than a topically
clustered L2 top-10 will, so the real hit rate sits below that bound.

Combined with the background-cosine distribution measured earlier in this
document (unrelated pairs at 0.65 median, 0.92 at p99), both halves point the
same way: adjacency here cannot be a threshold over a single anchor. It has to
be a ranked walk from a set.

---

## The batch encode path cannot reach the GPU, and nothing notices today

House rule (Gary, 2026-08-13): batch encodes on the GPU, stream encodes on the
CPU. The daemon honours the second half and structurally cannot honour the first.

`serve/mcp.py:276` calls `embedMissing` when the serve context is built, which
is a BATCH encode by definition: every live atom lacking a vector under the
active model. But the unit sets `HIP_VISIBLE_DEVICES=-1` and
`CUDA_VISIBLE_DEVICES=-1` process-wide, so `Embedder.__init__` resolves
`torch.cuda.is_available()` to False and the batch runs on CPU alongside the
stream path. There is no per-call device selection anywhere; the device is a
property of the process.

**Why nothing has noticed.** The backlog is currently 0: every live atom already
has a bge vector, so `embedMissing` selects nothing and returns immediately. The
gap is real and dormant.

**What it costs the day it stops being dormant**, using rates measured during
the encoder sweep:

| | rate | 299,728 live atoms |
|---|---:|---:|
| GPU batch (RX 7900 XTX, batch 64) | 1,034.4 atoms/s | ~4.8 min |
| CPU batch | 12.3 atoms/s | ~6.8 h |

84x. That difference decides whether a model migration is a coffee break or an
overnight job, and it is exactly the situation the house rule was written for.
It also silently shaped an earlier verdict in this campaign: the MiniLM
candidate's re-embed was called "not the blocker" on the strength of the 4.8
minute GPU figure, which the daemon as configured could not have achieved.

Not fixed here, and deliberately so. Making the device per-call rather than
per-process touches `Embedder`, and this campaign's remaining budget belongs to
certifying L2. Filed with its measurement so the next person does not
rediscover it at the start of a re-embed.
