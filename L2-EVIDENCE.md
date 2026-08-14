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

## What happens next, in order

1. Let the remaining investigations finish and the box go quiet.
2. Re-run `tiergate` and require contamination to PASS before reading any
   latency figure from it. A REJECT with contamination FAIL is not a result.
3. Land the ONNX path behind a switch, re-verified locally rather than on the
   subagent's numbers, and re-run.
4. If L2 still misses on a clean run, the remaining cost is inside the engine
   (bm25, vector search, fusion, enrich), and it gets decomposed by measurement
   rather than by hypothesis. Eight hypotheses died tonight; the cheap probe
   won every time.
