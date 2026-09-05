# Local agent outcome replay

This frozen synthetic workflow suite measures whether an actual reader gives the
right answer from delivered memory. It is separate from retrieval ID recall and
from the production tier latency gates. It is not LongMemEval, a leaderboard
submission, or evidence of general superiority.

`fixtures.v1.json` and its SHA-256 are frozen before running readers. Both recall
conditions seed the same facts and legacy snapshot bodies into isolated stores.
The candidate also writes the same checkpoint bodies to explicit task streams.
Questions and recent dialogue are identical across conditions. The baseline
uses its deployed prompt-only hook; the candidate uses its own real context
parser, hook, exact state and retrieval. All embedding runs use the same local
BGE model and CPU limits. No live store is written.

Use `prepare.py --help` for paths. Run separately against baseline and candidate
checkouts/plugins. This produces reader packets without expected answers, plus
retrieval timing and support metadata. The baseline also creates a no-memory
reader packet. Give each packet to a fresh reader with no previous campaign
context and the instructions below. Keep other packets, fixture answers, and
scores out of its inputs. Use the same reader model and settings for all arms.

Reader instructions:

> Treat each case independently. Use only its question, recentDialogue and
> memory. Do not inspect source, other files or external information. Answer the
> exact requested command, item, destination, port, format, filename or action;
> do not append explanatory prose to answer. If the information is absent or
> genuinely ambiguous, set answer to null and abstained to true. Historical
> evidence cannot override an explicit newer user correction. Current task
> state applies only to its named task. Return an array of objects with id,
> answer (string or null), abstained (boolean), evidence (array) and a short
> rationale. Cite p3 handles exactly as supplied, or use task:TASK_ID:rREVISION
> for current state and recentDialogue for an explicit user fact. Never execute
> the proposed commands; this is a replay.

Score with `score.py --packets ... --responses ... --metrics ... --output ...`.
Report answer accuracy, supported answer accuracy, incorrect non-abstentions,
irrelevant handles and injected characters. The exact answer field intentionally
has a narrow contract; inspect rationales before interpreting a formatting miss
as a memory failure. Retrieval timings here include cold effects and short
queries; use the matched production gate for service latency claims.

Additional automated cases cover historical/as-of reads, competing writers,
restart replay, partial migration failure, correction history, duplicate prompt
suppression, weak transcript scope, and exposure/use/helpfulness distinctions.
They validate software contracts and do not count as reader accuracy.

LongMemEval-V2 is a useful independent next evaluation: its public suite adds
multimodal trajectories, custom-environment gotchas and false premises. No claim
of running its full benchmark is made by this local suite.
