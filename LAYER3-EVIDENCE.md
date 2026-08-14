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
