# L2: three numbers, and which of them may be quoted

Written 2026-08-13, mid-campaign. Companion to the LAYER3/4/5 evidence docs.
L2 is the one tier missing its budget, and I have misread its own measurements
three times tonight. This is the reconciliation.

## The numbers

| number | what it measures | conditions | may I quote it |
|---|---|---|---|
| **0.168ms** | L1 p95, client-observed | gate FINAL | yes, PASS against 1ms |
| **107.206ms** | L3 p95, client-observed | gate FINAL | yes, PASS against 125ms |
| **44.179ms** | L2 p95, client-observed | gate FINAL | **no** |
| **24.36ms** | L2 p95, in-process engine | curated probes, ablation | with a caveat |

The budgets are client-observed (`engine.py:72`, Gary 2026-08-12): L1 <= 1ms,
L2 <= 20ms, L3 <= 125ms.

## Why 44.179 may not be quoted

The same gate run that produced it also ran `contamination.background-traffic`,
which FAILED. That unit's own words:

> "FAIL means this run's latency figures are contaminated and must not be
> quoted"

It observed 16 background requests and 1,200 recall_log rows in 60 seconds
against a ceiling of 5 requests per minute. The box has been carrying four
benchmark agents at load 40 to 69 all evening.

I quoted 44.179 anyway, repeatedly, and built arithmetic on it. That is the
instrument working correctly and the operator ignoring it.

**What the run still establishes.** Contamination inflates, so the failure
direction is trustworthy even when the magnitude is not: 105 of 105 samples
exceeded 20ms, p50 31.213, violation rate 100% with a 95% upper bound of 100%
by Clopper-Pearson. L2 is genuinely over budget. The open question is by how
much, not whether.

## Contamination is not symmetric, and that matters for L1 and L3

All four gate units above came from the same contaminated run, so it is fair to
ask why L1's 0.168ms and L3's 107.206ms appear in the "may quote" column while
L2's does not.

Because contamination only ever inflates latency. Background load makes a tier
look slower than it is, never faster. So:

- **A PASS under contamination is conservative.** L1 met a 1ms budget and L3 met
  a 125ms budget while the box carried four benchmark agents at load 40 to 69.
  A quiet box can only improve both. These are lower bounds on the margin, and
  the margin was already 6x for L1.
- **A FAIL under contamination is inconclusive.** It establishes that the tier
  is over budget *under this load*, which is not the question the budget asks.

That asymmetry is why L2 needs a clean re-run and L1/L3 do not. It is also why
the honest read of the FINAL run is "two tiers certified conservatively, one
tier unmeasured," rather than "two pass, one fails."

The one thing the L2 FAIL does establish independent of load: 105 of 105
samples were over, so the miss is not a tail artifact of a handful of slow
requests. Something structural puts every single L2 request over 20ms under
load. Whether that something survives a quiet box is the open question.

## Why 24.36 needs a caveat rather than a dismissal

It is an in-process ablation over curated probes (`engine.py:84`), not a
client-observed number over the gate's generated probes. Subtracting it from
44.179 to derive "20ms of transport overhead" compares two different
experiments, which I did, and it is wrong.

The clean refutation is L1: **0.168ms client-observed p95** on the same HTTP
stack. Transport on this daemon is a fraction of a millisecond. There is no 20ms
of serving overhead to find, and I nearly went looking for it.

## The encoder, measured properly at last

Earlier in this campaign I told Gary the encoder options were "measured and
closed." That was wrong: the spike had measured batch throughput, not
single-query latency. Reopened with four parallel investigations.

| candidate | verdict | evidence |
|---|---|---|
| **ONNX Runtime fp32** | **accept** | cosine 0.99999982 vs current, negative control 0.597, no re-embed, 4 to 7ms saved |
| MiniLM-L6-v2 swap | **reject** | `max_seq_length` 256 vs 512 truncates 25.8% of live atoms (~77,180) against 1.0% today |
| query-embed LRU cache | reject for the SLO | 49.63% hit rate on real traffic excluding the Charon loop; at that rate p95 is a miss by definition |
| INT8 dynamic quantization | pending | the ONNX investigation notes it would change the embedding space and forfeit the drop-in property |

The MiniLM rejection is the important one. It is roughly 2x faster and its
re-embed is cheap (1,034 atoms/s on the GPU, 4.8 minutes for 299,728 atoms,
correctly routed to the GPU because a re-embed is a *batch* encode per the house
rule). It is still disqualifying: truncating a quarter of the corpus to save 5ms
is the one trade a memory system may never make. Its cosine against current
(0.2597) sits BELOW its own negative control (0.5337), which is the signature of
a different embedding space rather than a faster path to the same one.

Both surviving investigations reported their load average, sampled it mid-run,
used interleaved same-process A/B rather than differencing separate runs, and
discarded runs where load moved. That is why their relative numbers are usable
on a box this noisy even though their absolute ones are not.

## Where L2 actually stands

Against the in-process ablation, which is the only L2 number not taken under
declared contamination:

```
  L2 engine p95 today          24.36ms
  ONNX saving (q95 to q154)   -4.07 to -4.97ms
  ------------------------------------------
  projected                    19.4 to 20.3ms      budget 20ms
```

L2 lands at the line. It may pass, it may miss by a few tenths, and no
measurement taken on this box tonight can distinguish those. The deciding run is
`tiergate` on a quiet box with `contamination.background-traffic` PASSING.

That unit is not an obstacle to the result. It is the reason the result will
mean anything.

## Two untuned defaults, offered as hypotheses rather than answers

Found by reading pragmas on the live store, not by measuring. Both sit on the L2
path and both are stock SQLite defaults that nobody has ever revisited:

| pragma | value | why it is suspicious |
|---|---|---|
| `synchronous` | 2 (FULL) | in WAL mode this fsyncs on every commit, and `logRecall` commits on every recall, so each retrieval pays a disk sync for telemetry |
| `cache_size` | -2000 (2 MB) | for a 1.7 GB database, on a box with plenty of RAM |
| `mmap_size` | 0 | no memory mapping |
| `busy_timeout` | 0 | a concurrent writer gets SQLITE_BUSY immediately rather than waiting |

`logRecall` itself is already well written: one `executemany` and one commit, not
ten round trips. The cost, if there is one, is the commit's durability barrier,
not the insert count.

**What would falsify the cache_size hypothesis**, and the reason I am not
claiming it: a 1.7 GB file on a box with this much RAM is probably already
resident in the OS page cache. If it is, SQLite's own cache being small means
its reads are cheap syscalls into warm memory rather than actual disk I/O, and
raising `cache_size` buys much less than the ratio suggests. The test is an A/B
with the pragma raised, on a quiet box, interleaved.

**What would falsify the synchronous hypothesis**: if the fsync cost is small
relative to 20ms (likely on NVMe), moving telemetry off the critical path saves
little. The clean test is to time `logRecall` in isolation against a copy of the
store at both settings.

Neither should be changed on a hunch. `synchronous=FULL` in particular is a
durability setting on a store whose stated purpose is retaining a person's voice,
so it gets measured and argued, not quietly relaxed for a benchmark.

## What happens next, in order, and why the order matters

The tempting sequence is: wait for quiet, enable ONNX, run the gate, report the
improvement. That sequence is wrong, and it would manufacture a false claim.

It changes TWO variables at once. The box going quiet removes contamination that
was inflating every number, and ONNX removes 4 to 7ms of encoder time. Run them
together and the whole delta gets attributed to ONNX, when most of it is likely
the load dropping. That is exactly how a plausible, well-evidenced,
completely wrong performance claim gets written.

So:

1. Let the remaining investigations finish and the box go quiet.
2. **Run `tiergate` clean with ONNX still OFF.** This is the baseline that does
   not exist yet: an uncontaminated L2 number for the code as it stands. It is
   the single most valuable measurement remaining, and it is the one that says
   whether L2 was ever really 24ms over or whether most of that was neighbours.
3. Enable ONNX (`90-onnx-embedder.conf.staged`), restart, run `tiergate` again
   under the same quiet conditions.
4. The difference between steps 2 and 3 is the ONNX effect, and nothing else.
   Report that number, not the difference from the contaminated run.
5. Require `contamination.background-traffic` to PASS on both. A REJECT with
   contamination FAIL is not a result, in either direction.
6. If L2 still misses at step 3, the remaining cost is inside the engine (bm25,
   vector search, fusion, enrich), and it gets decomposed by measurement rather
   than by hypothesis. Eight hypotheses died tonight and the cheap probe won
   every time.

Step 2 is the one I would skip if I were in a hurry, and it is the one that
makes step 4 mean anything.

---

# RESULT: the first uncontaminated measurement, A-B-A

Run 2026-08-13 on a quiet box (load 4.5 to 6, `face_find` finished,
`contamination.background-traffic` PASS on every run). Three deploys, each with
a full restart, an explicit 24-request warm, and n raised to clear the 59-sample
Clopper-Pearson floor. A/B/A ordering because A-then-B alone cannot separate the
candidate from the cache it inherits.

## Latency

| tier | A torch | B ONNX | A2 torch | budget |
|---|---:|---:|---:|---:|
| L1 p95 | 0.209 | 0.220 | 0.275 | 1.0 |
| L2 p50 | 31.73 | **25.78** | 30.26 | - |
| L2 p95 | 41.81 | 41.74 | 43.72 | 20.0 |
| L2 violations | 65/65 | 57/65 | 65/65 | - |
| L3 p95 | 107.66 | 83.19 | 87.50 | 125.0 |
| L3 violations | 1/65 | 0/65 | 0/65 | - |

## What the third run bought

**The ONNX win on L2's median is real.** A and A2 bracket B (31.73 / 25.78 /
30.26) and the violation count reproduces exactly (65 / 57 / 65). A two-run A/B
could not have established this; a two-run A/B is also what would have let me
believe the next line.

**The ONNX win on L3 was mostly an order effect.** B read 83.19 and I was ready
to call it a 24ms improvement. A2, with ONNX OFF, reads 87.50. Nearly all of
that gap was the page cache B inherited from A's full test-and-gate cycle, plus
a single 6,286ms stall that produced A's only violation. Had the run stopped at
two legs, this document would claim an encoder improvement that does not exist.

## The verdicts

**L1 is CERTIFIED.** Zero violations across 180 samples in three independent
runs, p95 between 0.209 and 0.275ms against a 1ms budget, contamination PASS,
staleness PASS. This is the tier the campaign moved onto its own lean route, and
it holds with roughly 4x margin.

**L3 passes on latency and FAILS on MRR@10.** Zero violations in two of three
runs; the single failure was one 6.3-second stall. But MRR@10 reads 0.405 /
0.426 / 0.415 against a 0.432 floor, failing all three times. The earlier
30-probe run read 0.433, a hair ABOVE floor. Raising the sample from 30 to 65
did not break L3's quality, it revealed that the passing number was a
small-sample artifact. R@10 is comfortable at 0.692 to 0.723 against 0.583.

**L2 FAILS, and not for the reason this campaign spent the night assuming.**
Every one of 65 samples exceeded 20ms in both torch runs. ONNX cuts the median
by about 5.5ms, exactly as the isolated encoder measurements predicted, and
moves p95 by 0.07ms. p95 is set by the tail, and the tail is not the encoder:
observed maxima include 2,961ms and 6,286ms. A multi-second stall in a system
whose median is 30ms is not slow code, it is something blocking.

## What this means for the encoder work

ONNX stays. It is a genuine ~5.5ms median improvement at zero quality cost and
no re-embed, reproduced against two independent baselines. But it does not close
L2 and was never going to: four independent investigations measured the encoder
correctly in isolation and every projection built on them was wrong, because
they projected a median saving onto a p95 budget.

The remaining ~22ms of L2's p95 is in the tail. The next work is not another
encoder; it is finding what stalls.

---

# THE TAIL: a write blocks every read, measured

Found after the A-B-A result showed p95 is set by stalls rather than throughput.
Measured 2026-08-13 against the live store.

## The mechanism

`ServeContext.reindex()` (`serve/mcp.py:264`) embeds any missing atoms and then
rebuilds the dense index for every kind class the write touched. Mutating tools
call it so that an emitted atom is recallable on the next call, which is correct
behaviour and the reason it exists.

It runs **synchronously on the event-loop thread**. That is not inferred; the
daemon documents it at `serve/daemon.py`:

> "the tee/shadow boundary handlers are synchronous and run inline on the
> event-loop thread -- the SAME thread that created the sqlite store (and the
> same path the MCP `call_tool` handler already takes)"

So a rebuild does not run alongside serving. Every in-flight and subsequent
request waits behind it.

## The cost

| class | kinds | rebuild |
|---|---|---:|
| memory | atom, narrative, snapshot | **272 ms** |
| code | document_chunk | **18,645 ms** |

An emit of a reasoning atom stalls all recall for roughly a quarter second. A
`document_chunk` write stalls it for **18.6 seconds**.

## Why this matches the tail and not the median

The A-B-A runs measured L2 p50 at 26 to 32ms with p95 at 42ms, and maxima of
2,961ms and 6,286ms. A blocking rebuild produces exactly that shape: it does not
touch the median, because most requests do not coincide with a write, and it
sets the tail entirely for those that do. It also explains why the stalls appear
in some runs and not others, which had looked like noise.

Atoms were written at 21:13:52 and 21:18:52, inside the gate windows. Those were
this session's own memory emits.

## What is NOT established

That the 272ms memory rebuild alone accounts for a 6,286ms sample. It does not,
arithmetically, and the honest statement is that a mechanism capable of
multi-second event-loop blocking has been found and measured, not that this
specific sample has been traced to it. `reindex` also calls `embedMissing`
first, and other contributors may stack. Closing that gap needs per-request
instrumentation, not more inference.

## Why this outranks further encoder work

The campaign spent the night on a component that sets the median. This sets the
tail, and the tier budgets are p95. A 272ms block is 13x the entire L2 budget;
the 18.6s code-class rebuild is 930x it. No encoder change can be seen through
that.

The shape of the fix is a design decision, not a tuning one, and belongs to
whoever picks this up: rebuild off the event loop, make it incremental rather
than a full class rebuild, or decouple write-visibility from index freshness.
Each trades something real (staleness, complexity, or memory) and none of them
should be chosen from a benchmark number alone.

## The obvious fix is not safe, checked before proposing it

The tempting fix is incremental append: a memory emit adds ONE vector, so
appending it to the flat index should cost microseconds instead of 272ms
rebuilding 16,763 rows. `FlatIndex` holds `_atomIds` and an `(n, dim)` matrix,
and `search` is `_matrix @ query`, so an append is mechanically trivial.

It is still wrong, for two reasons found by reading rather than by trying it:

1. **Append handles inserts and not retirements.** A `correct` supersedes an
   atom, and a superseded atom must stop being recallable. The full rebuild gets
   that for free through `WHERE a.status = 'live'`. An append-only path would
   leave retired atoms searchable, which is a correctness failure in a memory
   system, not a performance one. Any incremental scheme needs a removal path,
   and removal from a packed matrix is the expensive direction.
2. **There is an ordering invariant shared with the other index.**
   `hnsw_index.py` documents both indexes agreeing on rows "ordered by
   `atom_id`". Appending at the tail breaks that agreement, and the consequences
   live in code this measurement did not read.

So the fix is a design decision with correctness stakes, and the options each
give something up:

| approach | gives up |
|---|---|
| rebuild off the event loop | the store's documented single-thread affinity |
| incremental insert plus a removal path | complexity, and the ordering invariant needs re-establishing |
| decouple write-visibility from index freshness | read-your-writes, which is why `reindex` exists |

None of these should be chosen from a latency number. Recorded and handed off
deliberately rather than implemented at the end of a long session against a
store holding material that cannot be regenerated.

**The cheap operational mitigation, available now and not implemented either:**
the 18.6s figure belongs to the `document_chunk` class, which is written only by
bulk import. Nothing writes chunks during normal serving, so that path is
dormant in practice. The live cost is the 272ms memory rebuild, which fires on
every agent emit. An agent that emits during its own benchmark measures its own
writes.

---

# THE UNBLOCK: why the write must rebuild at all, and why it need not

Measured 2026-08-13 after the tail root cause landed. This is the enabling
analysis, run in parallel with a multi-agent design pass so the two can be
compared rather than one rubber-stamping the other.

## The two signals do not follow the same rule, and that asymmetry IS the blocker

`recall/signals.py` documents the lexical contract:

> "`bm25` joins the FTS rowid back to `atoms` and filters `status='live'` AT
> QUERY TIME -- an atom superseded after it was indexed is excluded even though
> its text is still in the FTS index."

The dense signal does the opposite (`signals.py:259`):

> "live-only candidate universe are the index's responsibility"

So lexical tolerates a stale index by construction, and dense does not. That is
the entire reason a write must rebuild the dense index synchronously: liveness
is enforced at BUILD time, so the build must chase every write.

Make dense filter liveness at query time, exactly as bm25 already does, and
index staleness stops being a correctness problem.

## What that filter costs, and the 564x trap on the way to measuring it

First measurement said 43ms, which would have killed the idea. It was measuring
a query planner failure, not a filter:

```
WHERE id IN (200 ids) AND status='live'
  -> SEARCH atoms USING INDEX idx_atoms_status (status=?)          43.020 ms
WHERE id IN (200 ids)
  -> SEARCH atoms USING COVERING INDEX sqlite_autoindex_atoms_1     0.078 ms
```

SQLite chooses `idx_atoms_status`, an index matching 299,728 rows, over 200
primary-key seeks. Production carries no `sqlite_stat1`, so the planner assumes
an index implies selectivity; `status` has about three distinct values and that
index is worthless for filtering. Same defect class as the `enrich.py` hub query
fixed earlier in this campaign (1182ms to 24ms).

| formulation | p50 |
|---|---:|
| naive `id IN (...) AND status='live'` | 43.020 ms |
| PK forced with `INDEXED BY` | **0.106 ms** |
| fetch `id, status` and filter in Python | **0.116 ms** |
| the rebuild this replaces | 272 ms |

**0.116ms of read cost to remove a 272ms write-path block.** That is 0.6% of the
L2 budget against something 13x larger than the whole budget.

Worth recording separately: the hot path today (`strata.py:122`,
`engine.py:357`) uses `SELECT id, kind FROM atoms WHERE id IN (...)` with NO
status predicate, so it already gets the covering index and is not affected. The
trap is not in production code; it is waiting for whoever adds a liveness filter
without checking the plan.

## The shape of the fix

1. **Query-time liveness filter on the dense signal**, written to use the
   primary key. Retirement is handled downstream, so a superseded atom is
   unrecallable regardless of index freshness. This is the correctness half and
   it makes dense symmetric with lexical.
2. **Incremental append on insert**, so a newly written atom is dense-recallable
   immediately and read-your-writes survives in full. ULID monotonicity means an
   append preserves the `ORDER BY atom_id` the two index types are built under.
3. **No full rebuild on the write path.** The 272ms block disappears rather than
   moving somewhere else.
4. Optional background compaction to reclaim rows that liveness filtering is
   masking, purely a memory concern, never a correctness one.

The correctness argument rests on step 1 alone: even if step 2 were omitted
entirely, no superseded atom can be returned. Step 2 buys freshness, not safety.

## Status

Not implemented here. A workflow is designing and building this in an isolated
worktree with three adversarial verifiers whose explicit job is to make a
superseded atom recallable. This section is the independent analysis their
result gets judged against, written before seeing it.

## What the incremental path actually costs, measured

usearch 2.25.3. Both kind classes exceed `HNSW_THRESHOLD = 5000` (memory is
16,763, code is 282,985), so both use the usearch HNSW index, not the flat one.

| operation | cost | against |
|---|---:|---|
| `usearch.add` one vector | **0.115 ms** | 272 ms rebuild (2,365x) |
| usearch build 16,763 | 212 ms | reproduces the measured 272ms selectIndex |
| flat vstack append @16,763 | 0.918 ms | (flat is not used at this size) |
| flat vstack append @282,985 | 33.758 ms | 18,645 ms rebuild (552x) |

So the write path becomes a 0.115ms index add plus the atom write, instead of a
272ms or 18,645ms full rebuild of the class.

**A caveat on my own probe, recorded because it is the shape of test this
campaign keeps catching.** I checked `usearch.remove(key)` by asking whether the
key appeared in the top-5 before and after removal. It read False before and
False after, which proves nothing: the key was never in the result set, so the
removal had nothing to demonstrate. A non-discriminating check that returns the
expected answer is worse than no check, because it reads as evidence. Retirement
semantics under usearch remain UNVERIFIED here and are assigned to the
adversarial verifiers, whose explicit task is to make a superseded atom
recallable.

This does not weaken the design, because the correctness argument rests on
query-time liveness filtering rather than on removal. If `remove` turns out to
be unreliable, the index simply keeps the row and the query-time filter drops
it, which is the same outcome by a different route.

## CORRECTION: the engine already resolves status at query time, so the fix is smaller

> **THIS SECTION IS WRONG. See the RETRACTION near the end of this file.**
> `assessTrust` resolves status but does NOT drop a superseded atom, it
> annotates it, so retirement DOES need explicit index work. Kept because
> the wrong version shipped; marked here so it cannot be read in isolation.

Two sections above I wrote that the dense signal has no query-time liveness
filter and that adding one is the precondition for deferring the rebuild. The
first half is true of the SIGNAL and false of the ENGINE, and the difference is
the whole design.

I tested `idx.search()` directly, saw a superseded atom come back, and called it
a vulnerability. Running the same case through `recall()` end to end:

```
superseded atom injected into a stale index:  present in _atomIds
ENGINE END-TO-END, superseded atom in results:  False
```

The engine drops it. The mechanism is `recall/trust.py::assessTrust`, which runs
unconditionally in the pipeline and issues ONE SELECT over every reranked atom
(`trust.py:369`):

```sql
SELECT id, status, COALESCE(occurred_at, created_at), kind FROM atoms WHERE id IN (...)
```

then branches: `live` passes, `tombstone` is dropped as retracted, `superseded`
is chased to the live end of its chain and annotated with `supersededBy` at a
confidence capped below `TRUST_FLOOR`. That query carries no status predicate,
so it uses the covering primary-key index and does not hit the 564x planner trap
documented above.

So the system ALREADY does query-time status resolution. I proposed adding a
second one.

**This was a component test generalised to a system claim**, which is the exact
reflex my own resident scar rule exists to catch. The index layer really does
return stale rows; the pipeline that consumes it really does filter them; only
the second fact governs what a caller sees.

## What the fix actually is

**Replace the full rebuild on write with an incremental add. Nothing else.**

| event | today | proposed |
|---|---|---|
| atom written | full class rebuild, 272ms (memory) / 18,645ms (code), blocking | `index.add(key, vec)`, **0.115ms** |
| atom superseded | full class rebuild | nothing; `assessTrust` already resolves status per query |
| index accumulates dead rows | never happens (rebuilt constantly) | background compaction, a MEMORY concern, never correctness |

Correctness argument, stated so a hole would be visible: no output row can be a
retracted atom, because `assessTrust` reads `status` from the canonical store
for every atom it returns, on every call, and that read does not consult the
index. Index staleness can therefore cost RANKING QUALITY (a dead row occupying
a candidate slot that a live atom deserved) but cannot cost CORRECTNESS. Quality
decay is bounded by how often compaction runs, and is measurable by R@10.

The earlier proposal in this document (add a query-time liveness filter to the
dense signal) is hereby withdrawn as redundant. It would have duplicated
`assessTrust`'s SELECT and, written naively, would have introduced a 43ms query
into the hot path.

## Status

A workflow is independently designing and building this with three adversarial
verifiers. This analysis was written before its results returned so the two can
be compared. If its design proposes a query-time liveness filter, that is
evidence it made the same component-to-system leap I did.

---

# RETRACTION: the section above is wrong, and a test caught it

The section titled "CORRECTION: the engine already resolves status at query
time, so the fix is smaller" claims a superseded atom cannot reach a result
because `assessTrust` resolves status per query. **That is false and the fix is
not smaller.** Left in place rather than deleted, because the wrong version
shipped in a commit and someone may have read it.

`assessTrust` resolves status and does NOT drop a superseded atom. It annotates
it with `supersededBy` at a confidence capped below `TRUST_FLOOR`, exactly as
`trust.py` says: "a superseded atom NEVER surfaces ALONE". Alone was doing all
the work in that sentence and I read past it.

`test_correct_supersedes_and_recall_shows_current_truth` caught it, by asking a
question whose retired answer had to reappear if the mechanism were missing:

```
assert "40 meters after the overlap fix" in rtext
assert "100 meters" not in rtext
E  '100 meters' is contained here:
E      the survey line spacing is 100 meters
E    source bulk-import, ..., superseded by p3://01KZZE...
```

## How I convinced myself of a false thing

By running a test that could not fail. I injected a superseded atom into a live
index, ran `recall()`, saw it absent from the results, and concluded the engine
had dropped it. It was absent because it did not RANK for that query. The test
never established that it would have appeared if the mechanism were missing, so
its pass carried no information.

That is the second non-discriminating probe in this campaign, after the usearch
`remove()` check recorded above. Both returned exactly the answer I expected.
The rule now written into `test_incremental_index.py`: an exclusion test must
first assert INCLUSION, in the same run, on the same input.

## And the first fix was also wrong

Retired rows were masked by setting their score to `-inf`. That lowers the rank
and does not exclude the row: `FlatIndex.search` returns EVERY row sorted by
score when `k >= n`, so the retired atom still came back, last. A corrected fact
reappearing at the bottom of the payload is still the corrected fact
reappearing. Retired positions are now dropped from the candidate set before
top-k selection.

## What actually shipped

| event | before | after |
|---|---|---|
| emit | `embedMissing` 90ms scan returning 0 rows, then a 272ms class rebuild, both blocking the event loop | `embedOne` (PK lookup) plus one `index.add`, **0.115ms** |
| correct | full class rebuild | `add` the replacement, `remove` the retired atom, both O(1) |
| retirement | implicit, via a rebuild that loaded live atoms only | **explicit**, via `VectorIndex.remove` |

`VectorIndex` grew two abstract methods, `add` and `remove`, and they are a
pair: `add` alone is the mistake this retraction documents. `FlatIndex` masks
row positions (position is identity there, so deleting would renumber every
later atom); `HnswIndex` drops the key from the usearch graph, verified
discriminatingly rather than from documentation.

Suite: 751 passed, 1 skipped.

## Measured: the write path, interleaved

Both arms in ONE process, alternating per rep, against the same `.backup` copy
of the live store (16,779 memory atoms, 282,985 code, both classes on HNSW).
Interleaved because differencing separately-taken numbers has been wrong every
time on this box.

| arm | p50 | min |
|---|---:|---:|
| full class rebuild (the old path) | **347.33 ms** | 343.78 ms |
| incremental (`embedOne` + `index.add`) | **10.60 ms** | 10.26 ms |

**33x on the median.** The old write path was 17x the entire 20ms L2 budget, on
the event-loop thread, so any read arriving during an emit waited behind all of
it. The new path fits inside the budget.

The 347ms confirms the component estimate: 90ms of `embedMissing` scanning
299,728 live atoms to return zero rows, plus a 272ms class rebuild, plus the
write itself.

**What the remaining 10.6ms is, stated so it is not mistaken for index work.**
The `index.add` is 0.115ms. Essentially all of the remainder is embedding the
atom that was just written, which is irreducible if a new atom is to be
dense-recallable at all: the only way to remove it from the write path is to
defer freshness, which is a different trade. Note also that this run used the
TORCH encoder, because the measurement script did not set
`PENSIVE_V3_ONNX_MODEL`; production runs ONNX, which measured roughly half the
torch cost per encode, so the shipped number is lower than 10.6ms.

Load was 24 to 33 throughout, so both arms are inflated and the RATIO is the
transferable figure, not the absolutes.

### The same measurement in the PRODUCTION configuration (ONNX encoder)

| arm | p50 | min |
|---|---:|---:|
| full class rebuild (the old path) | **363.73 ms** | 347.31 ms |
| incremental, ONNX encoder | **5.83 ms** | 4.59 ms |

**62x**, and 5.83ms sits comfortably inside the 20ms L2 budget where the old
path was 18x outside it. This is the shipped number; the 10.6ms above was the
torch encoder.

Worth naming, because it closes the encoder thread honestly: ONNX halves the
incremental write path (10.6ms to 5.8ms) while it moved L2's read P95 by 0.07ms.
The encoder win was real all along. It was landing on the median of a
distribution whose tail was set by something else, and the write path is the
place where an encode actually dominates. Two correct measurements pointing at
different budgets, which is the whole lesson of this document.

## Gate after the fix: the tail is gone and L2 is now a MEDIAN problem

Same procedure as the A-B-A (quiet box, warm, n=65, contamination PASS).

| run | L2 viol | L2 p50 | L2 p95 | L2 max | L3 p95 |
|---|---:|---:|---:|---:|---:|
| A torch baseline | 65/65 | 31.73 | 41.81 | 51.92 | 107.66 |
| B onnx | 57/65 | 25.78 | 41.74 | **2961.10** | 83.19 |
| A2 torch again | 65/65 | 30.26 | 43.72 | 55.75 | 87.50 |
| **POST incremental** | 60/65 | **24.99** | **36.19** | **47.31** | **77.14** |

Best L2 and L3 figures of the campaign. L1 passes at 0.374ms, L3 passes latency
at 77.14 against 125.

**L2 still FAILS at 36.19 against 20, and the failure has changed shape.** With
p50 24.99 and 60 of 65 samples over budget, this is no longer a tail set by
stalls; it is a median roughly 5ms too slow, uniformly. The stall hunt is
finished. What is left is throughput, which is a different investigation with
different suspects.

**A confound named rather than buried.** No writes landed during this gate run,
so the clean max of 47.31 is NOT by itself evidence the stall fix worked. A run
with no emits would show no emit-induced stalls either way. The evidence for the
fix is the direct interleaved measurement (363.73ms to 5.83ms per write), not
this run's quiet tail. What this run does establish is that with stalls absent,
L2's remaining gap is entirely in the body of the distribution.

**L3's MRR@10 still fails** at 0.394 against a 0.432 floor, consistent across
every run at the honest sample size. That is a retrieval-quality gap, unrelated
to any of tonight's latency work, and it is the other thing standing between L3
and a clean certification.

### What the next investigation should NOT assume

That the median is one thing. p50 24.99 against a single-query warm measurement
of 13.2ms on this same daemon says query variety costs ~12ms, so some probes are
far slower than others. The useful next step is a per-probe breakdown, not
another whole-pipeline average.

---

# THE GATE WAS BLIND, AND THE PLANTED FAILURE PROVES THE FIX

A multi-agent pass produced a candidate that closed L2's median and L3's MRR.
An adversarial verifier refuted it at severity HIGH, and the finding was about
this campaign's own instrument rather than the change.

## The blindness

`tiergate` builds its gold set with `kind='atom'`, and `generatedProbes`
additionally requires a `principle:` line that only reasoning atoms carry, so a
document_chunk was unreachable by construction. Five of the six curated
expectations are memory atoms.

| | share of gold | share of live corpus |
|---|---|---|
| memory kinds | 64 of 65 (98.5%) | 5.6% |
| `document_chunk` | 1 of 65 (1.5%) | **94.4%** |

A change that removes chunk results therefore cannot lower R@10 or MRR@10. It
can only raise them. The verifier rebuilt the missing family and measured a 92%
loss of chunk retrieval at L3 (R@10 0.633 to 0.050), while L3's served ids
became equal to L2's on 62 of 65 probes where the pre arm had differed on 205 of
205.

The quality unit's own `verdict_meaning` reads: "This is the unit that makes
'return fewer results' an unprofitable optimization." It could not do that for
the 94.4% of the corpus L3 exists to reach. Stated intent is not enforcement.

## The planted failure

Plant: restrict L3 to memory kinds, which is the candidate's effect in one line.

| unit | baseline | PLANTED |
|---|---|---|
| l3.quality.r_at_10 | 0.800 PASS | 0.829 **PASS** |
| l3.quality.mrr_at_10 | 0.458 PASS | 0.511 **PASS** |
| **l3.quality.chunk_r_at_10** | 0.767 PASS | **0.000 FAIL** |

The old units do not merely miss the regression, they **reward** it. L3's MRR
improved by deleting the corpus, which is precisely the shape of the fix that
had been proposed for L3's MRR failure. Under the plant, L3's quality figures
became identical to L2's, because L3 had stopped being a tier.

## The confirmation nobody asked for

Baseline chunk ranks: `[4,2,2,2,2,2,0,4,2,2,0,2,2,2,10,2,4,4,6,6]`. Every hit
EVEN. The judge separately established that L3's memory ranks contain only ODD
values, calling it a comb that no score contest could produce. Two independent
probe families landing on opposite parities is a 1:1 interleave seen from both
sides.

**So L3's MRR failure is the interleave, not the scoring and not the corpus.**
The fix is to stop interleaving when no reranker runs, which costs nothing and
removes nothing. The fix is NOT to drop chunks, which is what the gate would
have rewarded.

## What did not ship

The candidate. Not because its latency work was wrong (an independent verifier
reproduced that with a properly controlled three-daemon A/B/A and a calibrated
baseline), but because its quality claim rested on an instrument that could not
see what it traded away.
