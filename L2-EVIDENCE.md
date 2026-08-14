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
