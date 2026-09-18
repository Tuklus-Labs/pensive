# Clean pass 2 triage: NOT CLEAN, counter stays 0

> Historical record, written 2026-08-14. It describes the project as it stood then and is kept
> for provenance. Current behavior is documented in [README.md](README.md) and
> [daemon/README.md](daemon/README.md).

Strategies: adversarial (612 attacks) + integration/dependency (41 checks).
Different from pass 1 (static, contract, coverage) as the protocol requires.

## FIXED this pass

1. CRITICAL. Compat handlers ran the cross-encoder. pensive_recall 620.6ms and
   recall_records 606.6ms against native recall's 8.4ms, ~3,190ms on the real
   store. The tools live agents call. Fixed with rerankEnabled=False, NOT
   tier=DEFAULT_TIER (which would have overridden the caller's kinds and dropped
   document_chunk from a records API that asks for it).
2. CRITICAL. tiergate's staleness unit resolved the daemon by `pgrep -f` and
   took the first of FIVE matches, so it certified a different process than the
   one it measured. Now asks systemd for MainPID and refuses on ambiguity.
3. CRITICAL. The serving daemon predated HEAD by two hours, so it was running
   the retracted-atom bug. Restarted; staleness now PASSES against the right
   process.

## CONFIRMED, NOT YET FIXED

4. IMPORTANT. `correct` twice on the same atom leaves BOTH successors live and a
   single recall returns both. A supersession fork.
5. IMPORTANT. export then rebuild silently DROPS recall_log. A documented round
   trip that reports success and loses real data.
6. IMPORTANT. onnxruntime is imported by embedder.py and not declared in
   requirements. makeEmbedder catches ImportError and falls back to torch, so a
   fresh install silently loses the ONNX path rather than failing loudly.
7. IMPORTANT. The Go gate cannot build from a pensive clone alone: go.mod
   replaces aegis/gatekit with a path outside the repo, and deploy.sh dies if
   the build fails.
8. IMPORTANT. `kinds` is accepted by handle_recall and silently ignored end to
   end (verified: kinds=["narrative"] returns atom, narrative AND snapshot).
9. IMPORTANT. recall_records fails the WHOLE call when any atom exceeds 64
   provenance rows, with no partial answer and no truncation flag.
10. IMPORTANT (pre-existing, measured by me): dedup.py:75 at 475ms p50 on the
    emit path; signals.py:298 at 54ms p50 on the recall path when project is
    passed.
11. IMPORTANT. DF pruning coverage inversion: only tested with
    MIN_CORPUS_FOR_PRUNING monkeypatched 10,000 -> 10.
12. IMPORTANT. L1 ?full=1 provenance branch has zero coverage.

## MINOR, confirmed

- Host header with surrounding whitespace is allowed (RFC permits OWS, but the
  guard's own clause meant to catch it cannot fire).
- Duplicate Host headers: last wins, guard verdict depends on ordering.
- An empty-text atom is embedded, indexed and served as a TOP result.
- Error messages leak raw Python internals with no field name.
- 127MB orphaned .onnx.data next to the live artifact.
- 90-onnx-embedder.conf's header describes the opposite of its own state.

## Counter

Pass 2 NOT CLEAN. Counter 0. Pass 3 must use a different strategy again
(edge cases / dependency / integration-flow are unused).
# Clean pass 1 triage (orchestrator verification of auditor findings)

Every claim below was re-checked by me against the source or by measurement.
An auditor finding is a LEAD until reproduced.

## CONFIRMED, and in code written tonight

1. CRITICAL. `remove()` reports True while leaving the atom recallable.
   Reproduced on BOTH FlatIndex and HnswIndex, not just FlatIndex as reported:
     _atomIds = ['dup','other','dup'];  remove('dup') -> True
     search still returns 'dup'
   `list.index()` finds only the first position. A duplicate arises whenever
   indexAtom runs twice for one atom across a rebuild (emit retry, redelivered
   MCP call). This is the same failure class the workflow's verifier caught
   earlier tonight: a retracted memory still served, and here the API lies
   about it.

2. IMPORTANT. `retireAtom` does not maintain the aux index while `indexAtom`
   does (mcp.py:307 calls self.aux.reindex; retireAtom has zero references).
   Dormant only because aux dense is switched off.

## CONFIRMED by measurement, pre-existing

3. IMPORTANT. `ambient/dedup.py:75` costs **475ms p50** on the EMIT path.
   Plan: SEARCH atoms USING idx_atoms_status, then USE TEMP B-TREE FOR ORDER BY.
   It sorts all 299,728 live rows to take 200.

4. IMPORTANT. `recall/signals.py:298` costs **54ms p50** on the RECALL path,
   returning 74,427 rows. 2.7x the entire L2 budget. Fires only when a caller
   passes `project`, which the gate's probes never do, which is why the stage
   breakdown missed it. `pensive_recall` exposes `project`, so real callers hit it.

## CORRECTED: the auditors overstated this one

5. MINOR, not IMPORTANT. `mcp.py:992` assigns `kinds = args.get("kinds")` and
   never passes it to recall(). Both auditors called this a schema contract
   violation, claiming the tool's inputSchema declares `kinds`. IT DOES NOT.
   Verified: the recall tool schema has no `kinds` property. So this is dead
   code, not a broken contract.
   WORTH RECORDING: two agents with different scopes converged on the same
   wrong claim. Convergence raised my confidence and should not have; they made
   the same inference from the same suggestive variable name. Agreement between
   agents is not independent evidence when the inference is shared.

## KNOWN, already documented, not new

- `lifecycle/importance.py:61` placeholder limit (LAYER5-EVIDENCE.md documents
  that the job cannot run on production data). Real, unfixed, already filed.

## Counter

Pass 1 is NOT CLEAN. Counter remains 0.
