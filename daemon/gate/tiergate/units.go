package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"aegis/gatekit/verdict"
)

// ev writes raw output to disk and returns the pointer, or records a failure.
// A verdict without a pointer to its raw output is a rumor with a checkmark
// (STYLE.md L7).
//
// Every error used to be discarded and a path returned regardless, so on an
// unwritable or full evidence directory the gate produced a full set of
// components whose evidence strings named files that DID NOT EXIST. gatekit
// validates that the string is non-empty, which such a path satisfies, so the
// report constructed cleanly and the outcome was unchanged while no raw
// evidence existed anywhere. Found by a cross-vendor review, 2026-08-12.
//
// Errors are now collected on the cfg and checked by the caller before the
// report is built: an instrument that cannot record what it saw has failed,
// and must not emit a verdict about anything else.
func (c *cfg) ev(unit string, payload any) string {
	dir := filepath.Join(c.evidenceDir, c.epoch)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		c.evErrs = append(c.evErrs, fmt.Sprintf("%s: mkdir: %v", unit, err))
		return "EVIDENCE-WRITE-FAILED"
	}
	p := filepath.Join(dir, unit+".json")
	b, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		c.evErrs = append(c.evErrs, fmt.Sprintf("%s: marshal: %v", unit, err))
		return "EVIDENCE-WRITE-FAILED"
	}
	if err := os.WriteFile(p, b, 0o644); err != nil {
		c.evErrs = append(c.evErrs, fmt.Sprintf("%s: write: %v", unit, err))
		return "EVIDENCE-WRITE-FAILED"
	}
	// Read it back. A successful write call is not proof of a readable file,
	// and this is the artifact a human is sent to when they doubt the verdict.
	got, err := os.ReadFile(p)
	if err != nil || len(got) != len(b) {
		c.evErrs = append(c.evErrs, fmt.Sprintf("%s: readback: %v (%d/%d bytes)",
			unit, err, len(got), len(b)))
		return "EVIDENCE-WRITE-FAILED"
	}
	abs, err := filepath.Abs(p)
	if err != nil {
		return p
	}
	return abs
}

// state maps a measured comparison to a terminal state. The state is always a
// function of compared VALUES, never of an exit code we did not design.
func state(ok bool) verdict.State {
	if ok {
		return verdict.Pass
	}
	return verdict.Fail
}

func run(c *cfg) (*verdict.Report, error) {
	m := newMCP(c.daemonURL)
	var comps []verdict.Component
	add := func(check, unit string, st verdict.State, evidence string, score *verdict.Score) {
		comps = append(comps, verdict.Component{
			Check: check, Unit: unit, State: st, Evidence: evidence, Score: score,
		})
	}

	// --- identity -----------------------------------------------------------
	srcAbs, _ := filepath.Abs(c.srcDir)
	sha := gitHead(srcAbs)
	if sha == "" {
		return nil, fmt.Errorf("L6: cannot resolve artifact sha for %s: refusing to certify an unidentified artifact", srcAbs)
	}
	id := verdict.Identity{
		ArtifactPath: srcAbs, ArtifactSHA256: sha,
		GateName: gateName, GateVersion: gateVersion,
	}

	// --- staleness ----------------------------------------------------------
	// A pass from a stale artifact is not a pass. The daemon is long-lived; if
	// source changed after it started, every number below describes code that
	// is not running.
	{
		pid, started, perr := daemonProc()
		newest, npath, nerr := newestSourceMTime(srcAbs)
		q := "QUESTION: is the RUNNING daemon built from the source tree this gate is about to name in its identity?"
		switch {
		case perr != nil:
			add("staleness", "staleness.daemon-matches-source", verdict.Fail,
				c.ev("staleness", map[string]any{"question": q, "error": perr.Error()}), nil)
		case nerr != nil:
			add("staleness", "staleness.daemon-matches-source", verdict.Fail,
				c.ev("staleness", map[string]any{"question": q, "error": nerr.Error()}), nil)
		default:
			fresh := newest.Before(started)
			add("staleness", "staleness.daemon-matches-source", state(fresh),
				c.ev("staleness", map[string]any{
					"question": q, "pid": pid, "daemon_started": started,
					"newest_source_file": npath, "newest_source_mtime": newest,
					"verdict_meaning": "FAIL means the running daemon predates a source change: measurements describe code that is not loaded",
				}), nil)
		}
	}

	// --- probes (built BEFORE contamination, which must exclude them) --------
	gen, gerr := generatedProbes(c.storePath, c.genProbes, c.seed)
	cur, cerr := curatedProbes("probes.json")
	// Chunk gold, scored at L3 only. See generatedChunkProbes for why a gate
	// without this family cannot see a change that deletes 94.4% of the corpus.
	chunks, chErr := generatedChunkProbes(c.storePath, c.genProbes, c.seed)

	// --- contamination ------------------------------------------------------
	// Refuse to measure a daemon under unknown load rather than quietly
	// reporting a number that is mostly somebody else's retry loop.
	//
	// This guard previously filtered on `source_ref NOT LIKE '%tiergate%'`,
	// which excluded nothing: the DAEMON writes source_ref ("mcp.recall"), not
	// the caller, so the gate's `?agent=tiergate` never reached that column and
	// the check counted its OWN probes as background. A single run looked clean
	// only because no probe had run in the preceding 60s; two runs back to back
	// made it read 130 and REJECT itself. An instrument that cannot tell its own
	// traffic from the world's is measuring the wrong question, so exclude by
	// the one thing the gate genuinely owns: the exact query strings it issues.
	var bgPerMin float64
	{
		q := "QUESTION: is background recall traffic, EXCLUDING this gate's own probe queries, low enough that a latency number describes this gate rather than another process's load?"
		excl := sqlQuoteList(append(append(probeQueries(gen), probeQueries(cur)...),
			probeQueries(chunks)...))
		// recall_log holds one row PER RETURNED ATOM, not per request, so a raw
		// row count is in the wrong unit: one background request returning ten
		// atoms read as ten "events" and tripped a ceiling of five. Estimate
		// requests by distinct (query, recorded_at); a request writes all its
		// rows under one timestamp.
		n, err := scalarInt(c.storePath, fmt.Sprintf(
			"SELECT COUNT(*) FROM (SELECT DISTINCT query, recorded_at FROM recall_log "+
				"WHERE recorded_at > strftime('%%s','now') - 60 AND query NOT IN (%s))", excl))
		rows, _ := scalarInt(c.storePath, fmt.Sprintf(
			"SELECT COUNT(*) FROM recall_log WHERE recorded_at > strftime('%%s','now') - 60 "+
				"AND query NOT IN (%s)", excl))
		if err != nil {
			add("contamination", "contamination.background-traffic", verdict.Fail,
				c.ev("contamination", map[string]any{"question": q, "error": err.Error()}), nil)
		} else {
			bgPerMin = float64(n)
			add("contamination", "contamination.background-traffic", state(bgPerMin <= c.maxBackground),
				c.ev("contamination", map[string]any{
					"question":                        q,
					"background_requests_est_last60s": n,
					"background_rows_last60s":         rows,
					"ceiling_requests_per_min":        c.maxBackground,
					"KNOWN BLIND SPOT": "a background request that returns ZERO atoms writes no " +
						"recall_log row and is invisible here, so this guard bounds noisy traffic, " +
						"not all traffic. It also infers ownership from query equality, so another " +
						"caller issuing an identical query would be excluded as ours.",
					"verdict_meaning": "FAIL means this run's latency figures are contaminated and must not be quoted",
				}), nil)
		}
	}

	// --- host load ----------------------------------------------------------
	// The sibling guard above bounds contamination from OTHER RECALL TRAFFIC.
	// It says nothing about contamination from the HOST, and the daemon's
	// embedder runs on CPU (GPU is banned for it by a systemd drop-in), so a
	// busy machine inflates every tier's latency without writing a single
	// recall_log row. On 2026-08-19 a full run rejected on all three latency
	// units at load average 19.5 on 24 cores while every quality unit passed;
	// the background-traffic guard read 1 request/min and cheerfully certified
	// the run as uncontaminated. A gate that refuses to measure under one kind
	// of interference and reports confidently under another is measuring the
	// wrong question (STYLE.md: instruments fail because the QUESTION drifted).
	{
		q := "QUESTION: is host CPU load low enough that a latency number describes this daemon rather than the machine's run queue?"
		ceiling := c.maxLoadPerCore * float64(runtime.NumCPU())
		raw, err := os.ReadFile("/proc/loadavg")
		if err != nil {
			add("contamination", "contamination.host-load", verdict.Fail,
				c.ev("host-load", map[string]any{"question": q, "error": err.Error()}), nil)
		} else {
			fields := strings.Fields(string(raw))
			load1, perr := strconv.ParseFloat(fields[0], 64)
			if perr != nil || len(fields) == 0 {
				add("contamination", "contamination.host-load", verdict.Fail,
					c.ev("host-load", map[string]any{"question": q,
						"error": "could not parse /proc/loadavg", "raw": string(raw)}), nil)
			} else {
				add("contamination", "contamination.host-load", state(load1 <= ceiling),
					c.ev("host-load", map[string]any{
						"question":         q,
						"load1":            load1,
						"cores":            runtime.NumCPU(),
						"load_per_core":    load1 / float64(runtime.NumCPU()),
						"ceiling_per_core": c.maxLoadPerCore,
						"ceiling_load1":    ceiling,
						"KNOWN BLIND SPOT": "load average counts uninterruptible I/O wait as well as " +
							"runnable tasks, so a disk-bound neighbour trips this guard even when CPU " +
							"is idle. It also samples ONCE, at the end of the run, so a burst that ended " +
							"before this line executed is invisible.",
						"verdict_meaning": "FAIL means this run's latency figures measure the machine, not the daemon, and must not be quoted as a regression",
					}), nil)
			}
		}
	}

	// --- canary pair (self-testing: carries its own negative half) -----------
	knownID, kerr := someLiveAtomID(c.storePath)
	{
		q := "QUESTION: does an exact lookup return a known-present atom, AND return nothing for an id that cannot exist?"
		if kerr != nil {
			add("canary", "canary.known-id-roundtrips", verdict.Fail,
				c.ev("canary-known", map[string]any{"question": q, "error": kerr.Error()}), nil)
			add("canary", "canary.impossible-id-empty", verdict.Fail,
				c.ev("canary-impossible", map[string]any{"question": q, "error": "no known id to contrast against"}), nil)
		} else {
			gotKnown, _, codeKnown, e1 := m.get(knownID)
			hitKnown := e1 == nil && codeKnown == 200 && strings.Contains(gotKnown, knownID)
			add("canary", "canary.known-id-roundtrips", state(hitKnown),
				c.ev("canary-known", map[string]any{
					"question": q, "atom_id": knownID, "http_status": codeKnown,
					"error": errStr(e1), "body_prefix": prefix(gotKnown, 240),
					"verdict_meaning": "FAIL means the L1 identity path cannot return a row that provably exists in the store",
				}), nil)

			gotImp, _, codeImp, e2 := m.get(impossibleAtomID)
			// The negative half must be an EMPTY answer, not an error and not a
			// result. A route that 500s on a bad id is not discriminating.
			emptyImp := e2 == nil && (codeImp == 404 || (codeImp == 200 && !strings.Contains(gotImp, impossibleAtomID)))
			// A negative half only means something when the positive half
			// passed. With no L1 route at all, BOTH ids return an identical
			// 404 and this unit reported PASS having proved nothing: a 404 for
			// "never existed" and a 404 for "no route" are indistinguishable
			// (STYLE.md: convert an ambiguous negative into an unambiguous
			// positive). Non-discrimination is an INSTRUMENT failure, so it is
			// ERROR, which taints, rather than PASS or FAIL.
			impState := state(emptyImp)
			note := "FAIL means the lookup does not discriminate: it cannot distinguish present from absent"
			if !hitKnown {
				impState = verdict.Error
				note = "ERROR: the positive half failed, so an empty answer here proves nothing. Both ids return the same response and this unit cannot discriminate."
			}
			add("canary", "canary.impossible-id-empty", impState,
				c.ev("canary-impossible", map[string]any{
					"question": q, "atom_id": impossibleAtomID, "http_status": codeImp,
					"error": errStr(e2), "body_prefix": prefix(gotImp, 240),
					"positive_half_passed": hitKnown,
					"verdict_meaning":      note,
				}), nil)
		}
	}

	// --- L1: identity retrieval over the lean route -------------------------
	{
		q := fmt.Sprintf("QUESTION: is client-observed P95 for exact atom lookup <= %.3gms over the lean route?", c.l1Budget)
		var lats []float64
		var lastCode int
		var lastErr error
		exact := true
		ids, iderr := someLiveAtomIDs(c.storePath, c.iterations)
		distinctProbed := len(ids)
		if kerr == nil && iderr == nil {
			for i := 0; i < c.iterations; i++ {
				id := ids[i%len(ids)]
				body, d, code, err := m.get(id)
				lastCode, lastErr = code, err
				lats = append(lats, plantLatency(c, float64(d.Microseconds())/1000.0))
				if err != nil || code != 200 || !strings.Contains(body, id) {
					exact = false
				}
			}
		} else {
			exact = false
		}
		routeUp := lastErr == nil && lastCode == 200
		if !routeUp {
			add("latency", "l1.latency.p95", verdict.Fail,
				c.ev("l1-latency", map[string]any{
					"question": q, "budget_ms": c.l1Budget, "n": len(lats),
					"http_status": lastCode, "error": errStr(lastErr),
					"verdict_meaning": "FAIL: the lean L1 route does not answer. No latency claim is made; " +
						"a timing over a 404 measures the router, not the tier.",
				}), nil)
		} else {
			st, evp, sc := latencyUnit(c, "l1-latency", q, lats, c.l1Budget, map[string]any{
				"http_status": lastCode, "plant": c.plant,
				"distinct_ids_probed": distinctProbed,
			})
			add("latency", "l1.latency.p95", st, evp, sc)
		}
		add("correctness", "l1.correctness.exact", state(routeUp && exact),
			c.ev("l1-correctness", map[string]any{
				"question": "QUESTION: does exact lookup return exactly the atom requested, on every iteration?",
				"atom_id":  knownID, "iterations": len(lats), "all_exact": exact, "http_status": lastCode,
				"verdict_meaning": "FAIL means L1 is fast but wrong, which is worse than slow and right",
			}), nil)
	}

	// --- probes for the semantic tiers (built above, before contamination) ---
	nProbes := len(gen) + len(cur)
	// A floor, not a nonzero check. The first version asserted len(gen) > 0 and
	// went green on a SINGLE generated probe while the quality units below
	// reported a confident R@10: an aggregate that improves while coverage
	// falls is the disease, not the cure (STYLE.md).
	covOK := gerr == nil && cerr == nil && len(gen) >= c.minGenerated && len(cur) >= c.minCurated
	add("coverage", "coverage.probe-count", state(covOK),
		c.ev("coverage", map[string]any{
			"question":  "QUESTION: how many probes did this run actually exercise, in each family, and does that clear the floor each family must maintain?",
			"generated": len(gen), "curated": len(cur), "total": nProbes,
			"generated_chunk": len(chunks), "generated_chunk_error": errStr(chErr),
			"min_generated": c.minGenerated, "min_curated": c.minCurated,
			"generated_error": errStr(gerr), "curated_error": errStr(cerr),
			"verdict_meaning": "FAIL means a probe family fell below its floor, so the quality verdict covers less than it appears to",
		}), nil)

	// --- L2 and L3: measured through the transport a caller actually uses -----
	//
	// Both tiers used to be un-measurable here: L2 was a hardcoded existence
	// boolean and L3 called recall with no tier at all, which -- once L2 became
	// the DEFAULT -- would have silently measured L2 and reported it as L3. A
	// unit that measures a different thing than its name says is the drifted
	// question failure, so each tier is now named explicitly in the request.
	measureTier := func(tier string, budget float64) {
		q := fmt.Sprintf("QUESTION: is client-observed P95 for tier %s <= %.3gms, "+
			"WITHOUT its retrieval quality falling below the floor?", tier, budget)
		all := append(append([]probe{}, gen...), cur...)
		var lats []float64
		var hits []int
		for _, p := range all {
			payload, d, err := m.call("recall", map[string]any{
				"query": p.Query, "k": 10, "tokenBudget": 1500, "tier": tier,
			})
			ms := plantLatency(c, float64(d.Microseconds())/1000.0)
			lats = append(lats, ms)
			ids := handleIDs(payload)
			if c.plant == "empty" {
				ids = nil // the reward hack: fast, and returns nothing
			}
			if err != nil {
				hits = append(hits, 0)
				continue
			}
			hits = append(hits, rankOf(ids, p.Expected))
		}
		st, evp, sc := latencyUnit(c, "l"+strings.ToLower(tier[1:])+"-latency", q, lats, budget,
			map[string]any{"tier": tier, "plant": c.plant, "token_budget_used": 1500})
		add("latency", "l"+strings.ToLower(tier[1:])+".latency.p95", st, evp, sc)

		r10, mrr := metrics(hits)
		low := strings.ToLower(tier[1:])
		add("quality", "l"+low+".quality.r_at_10",
			state(len(hits) > 0 && r10 >= c.rAt10Floor),
			c.ev("l"+low+"-quality-r10", map[string]any{
				"question": fmt.Sprintf("QUESTION: is tier %s R@10 at or above the floor %.3f?", tier, c.rAt10Floor),
				"tier":     tier, "r_at_10": r10, "floor": c.rAt10Floor,
				"n": len(hits), "ranks": hits, "plant": c.plant,
				"POPULATION CAVEAT": "the floor came from 1,500 real chat turns scored " +
					"against document chunks; these probes are self-retrieval plus a " +
					"curated set. Different population, so the floor is a REGRESSION " +
					"tripwire here, not a transferred claim.",
				"verdict_meaning": "FAIL means quality was traded for speed. This is the unit " +
					"that makes 'return fewer results' an unprofitable optimization.",
			}), &verdict.Score{Value: r10, Max: 1.0})
		// CHUNK RETRIEVAL, L3 ONLY. The two units above cannot see a change that
		// removes document_chunk results, because 64 of their 65 gold documents
		// are memory atoms while chunks are 94.4% of the live corpus. A candidate
		// that dropped chunks from L3 scored as an IMPROVEMENT on both of them
		// while losing 92% of chunk retrieval. This unit is what makes that
		// unprofitable, and it is scored at L3 ALONE because L2 excludes chunks
		// by contract.
		if tier == "L3" && len(chunks) > 0 {
			var chunkHits []int
			for _, p := range chunks {
				payload, _, err := m.call("recall", map[string]any{
					"query": p.Query, "k": 10, "tokenBudget": 1500, "tier": tier,
				})
				ids := handleIDs(payload)
				if c.plant == "empty" {
					ids = nil
				}
				if err != nil {
					chunkHits = append(chunkHits, 0)
					continue
				}
				chunkHits = append(chunkHits, rankOf(ids, p.Expected))
			}
			cr10, cmrr := metrics(chunkHits)
			add("quality", "l3.quality.chunk_r_at_10",
				state(len(chunkHits) > 0 && cr10 >= c.chunkRAt10Floor),
				c.ev("l3-quality-chunk-r10", map[string]any{
					"question": fmt.Sprintf("QUESTION: can L3 still retrieve a document_chunk "+
						"given a line of its own text, at R@10 >= %.3f?", c.chunkRAt10Floor),
					"tier": tier, "chunk_r_at_10": cr10, "chunk_mrr_at_10": cmrr,
					"floor": c.chunkRAt10Floor, "n": len(chunkHits), "ranks": chunkHits,
					"plant": c.plant,
					"WHY THIS FAMILY EXISTS": "the other quality units draw 64 of 65 gold " +
						"documents from kind='atom' while document_chunk is 282,985 of " +
						"299,802 live atoms. They scored a 92% loss of chunk retrieval as " +
						"an improvement.",
					"verdict_meaning": "FAIL means L3 stopped reaching the corpus it exists " +
						"to reach. L3 minus chunks is L2 with a longer budget.",
				}), &verdict.Score{Value: cr10, Max: 1.0})
		}

		add("quality", "l"+low+".quality.mrr_at_10",
			state(len(hits) > 0 && mrr >= c.mrrFloor),
			c.ev("l"+low+"-quality-mrr", map[string]any{
				"question": fmt.Sprintf("QUESTION: is tier %s MRR@10 at or above %.3f?", tier, c.mrrFloor),
				"tier":     tier, "mrr_at_10": mrr, "floor": c.mrrFloor, "n": len(hits),
				"verdict_meaning": "FAIL means ordering degraded even where the right answer is still in the top 10",
			}), &verdict.Score{Value: mrr, Max: 1.0})
	}
	measureTier("L2", c.l2Budget)
	measureTier("L3", c.l3Budget)

	// An instrument that could not record what it saw has failed. Checked before
	// the report is constructed, so a run with unwritable evidence cannot emit a
	// verdict about the daemon at all.
	if len(c.evErrs) > 0 {
		return nil, fmt.Errorf("L7: evidence could not be written (%d failures), "+
			"refusing to emit a verdict backed by pointers to nothing: %s",
			len(c.evErrs), strings.Join(c.evErrs, "; "))
	}
	return verdict.NewReport(id, "canary.known-id-roundtrips", comps)
}

// plantLatency implements the planted failures. A plant must be an edit a real
// author would plausibly make (STYLE.md), so:
//   - "latency" is the slow-regression shape
//   - "empty"   is the REWARD HACK shape: return nothing, very fast. The gate
//     must reject it on quality even though every latency unit goes green.
func plantLatency(c *cfg, ms float64) float64 {
	switch c.plant {
	case "latency":
		return ms + 500.0
	case "empty":
		return 0.5
	default:
		return ms
	}
}

func metrics(hits []int) (r10, mrr float64) {
	if len(hits) == 0 {
		return 0, 0
	}
	n := float64(len(hits))
	for _, r := range hits {
		if r >= 1 && r <= 10 {
			r10++
			mrr += 1.0 / float64(r)
		}
	}
	return r10 / n, mrr / n
}

// ---------------------------------------------------------------------------

func gitHead(dir string) string {
	out, err := exec.Command("git", "-C", dir, "rev-parse", "HEAD").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// daemonProc resolves the process this gate is MEASURING, not merely one whose
// name looks right.
//
// It used to run `pgrep -f pensived-v3` and take f[0]. On this box that matches
// FIVE processes: the unit's MainPID plus three leftover pensive-review A/B
// daemons carrying about 9.2GB between them, so f[0] was 2195449 while the
// daemon answering port 5999 was 2359723. The unit whose entire job is
// certifying that the numbers describe the running artifact was resolving the
// artifact by name substring and taking whichever matched first.
//
// systemd owns the answer, so ask systemd. pgrep remains only as a fallback for
// a daemon started by hand, and in that case the ambiguity is reported rather
// than silently resolved: a staleness verdict about the wrong process is worse
// than no staleness verdict, because it reads as certification.
func daemonProc() (int, time.Time, error) {
	pidStr := ""
	if out, err := exec.Command("systemctl", "--user", "show", "pensive-v3",
		"-p", "MainPID", "--value").Output(); err == nil {
		v := strings.TrimSpace(string(out))
		if v != "" && v != "0" {
			pidStr = v
		}
	}
	if pidStr == "" {
		out, err := exec.Command("pgrep", "-f", "pensived-v3").Output()
		if err != nil {
			return 0, time.Time{}, fmt.Errorf("daemon not running (systemd reports no MainPID and pgrep pensived-v3 found nothing)")
		}
		f := strings.Fields(strings.TrimSpace(string(out)))
		if len(f) == 0 {
			return 0, time.Time{}, fmt.Errorf("daemon not running")
		}
		if len(f) > 1 {
			return 0, time.Time{}, fmt.Errorf(
				"AMBIGUOUS DAEMON: %d processes match pensived-v3 (%s) and systemd "+
					"reports no MainPID, so this gate cannot tell which one it is "+
					"measuring; refusing to certify staleness against a guess",
				len(f), strings.Join(f, " "))
		}
		pidStr = f[0]
	}
	f := []string{pidStr}
	pid, _ := strconv.Atoi(f[0])
	el, err := exec.Command("ps", "-o", "etimes=", "-p", f[0]).Output()
	if err != nil {
		return pid, time.Time{}, fmt.Errorf("cannot read daemon elapsed time")
	}
	secs, _ := strconv.Atoi(strings.TrimSpace(string(el)))
	return pid, time.Now().Add(-time.Duration(secs) * time.Second), nil
}

func newestSourceMTime(dir string) (time.Time, string, error) {
	var newest time.Time
	var path string
	err := filepath.Walk(dir, func(p string, fi os.FileInfo, err error) error {
		if err != nil || fi.IsDir() || !strings.HasSuffix(p, ".py") {
			return nil
		}
		if fi.ModTime().After(newest) {
			newest, path = fi.ModTime(), p
		}
		return nil
	})
	if err != nil {
		return time.Time{}, "", err
	}
	if path == "" {
		return time.Time{}, "", fmt.Errorf("no .py files under %s: staleness check has nothing to compare", dir)
	}
	return newest, path, nil
}

func someLiveAtomID(storePath string) (string, error) {
	rows, err := query(storePath,
		"SELECT id FROM atoms WHERE status='live' AND kind='atom' ORDER BY created_at DESC LIMIT 1")
	if err != nil {
		return "", err
	}
	if len(rows) == 0 {
		return "", fmt.Errorf("no live atoms in store: canary has nothing provably present to ask for")
	}
	s, _ := rows[0]["id"].(string)
	return s, nil
}

func errStr(e error) string {
	if e == nil {
		return ""
	}
	return e.Error()
}

func prefix(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

// emit prints the report and its accounting. Aggregates never travel without
// their components (STYLE.md Law 3), so every component line is printed.
func emit(c *cfg, r *verdict.Report) {
	a := r.Accounting()
	b, _ := json.MarshalIndent(r, "", "  ")
	p := filepath.Join(c.evidenceDir, c.epoch, "report.json")
	_ = os.MkdirAll(filepath.Dir(p), 0o755)
	_ = os.WriteFile(p, b, 0o644)

	fmt.Printf("\n=== %s v%s  epoch=%s  plant=%s ===\n", gateName, gateVersion, c.epoch, c.plant)
	for _, line := range strings.Split(string(b), "\n") {
		if strings.Contains(line, `"unit"`) || strings.Contains(line, `"state"`) {
			fmt.Println("  " + strings.TrimSpace(line))
		}
	}
	fmt.Printf("\naccounting: total=%d pass=%d fail=%d timeout=%d error=%d skip=%d\n",
		a.Total, a.Pass, a.Fail, a.Timeout, a.Error, a.Skip)
	if d := r.Demotions(); len(d) > 0 {
		fmt.Printf("demotions (L4 perfect-is-suspect):\n")
		for _, s := range d {
			fmt.Println("  " + s)
		}
	}
	fmt.Printf("tainted: %v\n", r.Tainted())
	fmt.Printf("OUTCOME: %s\n", r.Outcome())
	fmt.Printf("report: %s\n", p)
}

// probeQueries returns the exact query strings a probe set will issue, so the
// contamination guard can subtract the gate's own traffic from the world's.
func probeQueries(ps []probe) []string {
	out := make([]string, 0, len(ps))
	for _, p := range ps {
		out = append(out, p.Query)
	}
	return out
}

// sqlQuoteList renders a SQL string list. Single quotes are doubled; nothing
// else reaches SQL from a probe. An EMPTY list must not produce "NOT IN ()",
// which is a syntax error that would make the guard fail to run at all rather
// than fail loudly, so it yields a value no query can equal.
func sqlQuoteList(vals []string) string {
	if len(vals) == 0 {
		return "''"
	}
	parts := make([]string, 0, len(vals))
	for _, v := range vals {
		parts = append(parts, "'"+strings.ReplaceAll(v, "'", "''")+"'")
	}
	return strings.Join(parts, ",")
}

// REQUIRED_ZERO_VIOLATION_N is the sample size at which observing ZERO
// over-budget requests supports "at most 5% of requests exceed budget" at 95%
// confidence. With k=0 violations the exact one-sided upper bound is
// 1 - 0.05^(1/n), so n=19 supports only 14.6%, n=40 supports 7.2%, and n=59 is
// the first n that reaches 5%.
//
// This exists because the gate used to report a nearest-rank "p95" over as few
// as 19 observations, where that statistic IS the sample maximum. If 6% of real
// requests exceeded budget, a 19-request sample would miss every one of them
// about 31% of the time and the gate would print a passing P95. A percentile
// name does not confer the evidence a percentile claim needs.
const REQUIRED_ZERO_VIOLATION_N = 59

// violationUpperBound returns the exact one-sided 95% Clopper-Pearson upper
// bound on the true violation rate given k violations in n samples.
//
// It used to return k/n for k>0, which UNDERSTATES the bound, so the caller had
// to treat any nonzero k as failure. That made a unit named for a P95 budget
// enforce a P100 one: "P95 <= 125ms" means up to 5% of requests MAY exceed, and
// failing on a single violation in 65 holds the system to a stricter contract
// than the one written down. A gate that enforces something other than its
// stated claim is the drifted-question defect, whichever direction it drifts.
//
// Solved by bisection on the binomial CDF: the upper bound is the p at which
// P(X <= k | n, p) = 0.05. No stats library, and bisection is exact enough at
// the sample sizes a gate uses.
//
// NOTE for anyone reading this as a loosening: at k=1, n=65 this returns ~7.3%,
// which still FAILS the 5% requirement. Correcting the statistic did not make
// the tier pass. That was the check that it is a correction and not a favour.
func violationUpperBound(k, n int) float64 {
	if n <= 0 {
		return 1.0
	}
	if k >= n {
		return 1.0
	}
	if k == 0 {
		return 1.0 - math.Pow(0.05, 1.0/float64(n))
	}
	lo, hi := 0.0, 1.0
	for i := 0; i < 200; i++ {
		mid := (lo + hi) / 2
		if binomCDF(k, n, mid) > 0.05 {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

// binomCDF is P(X <= k) for X ~ Binomial(n, p), summed in log space so a large
// n does not overflow the factorials.
func binomCDF(k, n int, p float64) float64 {
	if p <= 0 {
		return 1.0
	}
	if p >= 1 {
		return 0.0
	}
	sum := 0.0
	for i := 0; i <= k; i++ {
		logC := lgamma(float64(n+1)) - lgamma(float64(i+1)) - lgamma(float64(n-i+1))
		sum += math.Exp(logC + float64(i)*math.Log(p) + float64(n-i)*math.Log(1-p))
	}
	if sum > 1 {
		return 1
	}
	return sum
}

func lgamma(x float64) float64 {
	v, _ := math.Lgamma(x)
	return v
}

// latencyUnit turns a latency sample into a verdict that states what the sample
// can actually support. Three outcomes, and the middle one is the point:
//
//	FAIL  -- at least one request exceeded budget
//	ERROR -- zero violations, but too few samples to support the claim. This
//	         TAINTS the report rather than passing it, because "I saw no
//	         violations in 19 tries" and "violations are below 5%" are
//	         different statements and only one of them is the contract.
//	PASS  -- zero violations with enough samples to bound the rate at 5%
func latencyUnit(c *cfg, unit, question string, lats []float64, budget float64,
	extra map[string]any) (verdict.State, string, *verdict.Score) {
	violations := 0
	for _, ms := range lats {
		if ms > budget {
			violations++
		}
	}
	n := len(lats)
	bound := violationUpperBound(violations, n)
	st := verdict.Pass
	meaning := "PASS: zero over-budget requests, with enough samples to bound the violation rate at 5%"
	switch {
	case n == 0:
		st, meaning = verdict.Error, "ERROR: no samples taken; an empty measurement is not a pass"
	case bound > 0.05:
		st = verdict.Fail
		meaning = fmt.Sprintf("FAIL: %d/%d over budget; the 95%% upper bound on the "+
			"violation rate is %.1f%%, above the 5%% a P95 claim allows", violations, n, bound*100)
	case n < REQUIRED_ZERO_VIOLATION_N && violations == 0:
		st = verdict.Error
		meaning = fmt.Sprintf("ERROR: zero violations in %d samples bounds the true rate only at "+
			"%.1f%%, not the 5%% the budget claim needs; %d samples are required. "+
			"Insufficient evidence is not a pass.", n, bound*100, REQUIRED_ZERO_VIOLATION_N)
	}
	payload := map[string]any{
		"question":                question,
		"n":                       n,
		"violations":              violations,
		"violation_rate_obs":      violationUpperBound(violations, max(n, 1)) * 0, // placeholder replaced below
		"violation_rate_95_upper": bound,
		"budget_ms":               budget,
		"p50_ms":                  median(lats),
		"p95_sample_ms":           p95(lats),
		"max_ms":                  sampleMax(lats),
		"required_n_for_claim":    REQUIRED_ZERO_VIOLATION_N,
		"NOTE": "p95_sample_ms is a nearest-rank SAMPLE statistic. At n below " +
			"the required size it is simply the sample maximum and must not be " +
			"quoted as a population P95.",
		"verdict_meaning": meaning,
	}
	if n > 0 {
		payload["violation_rate_obs"] = float64(violations) / float64(n)
	}
	for k, v := range extra {
		payload[k] = v
	}
	return st, c.ev(unit, payload), &verdict.Score{Value: p95(lats), Max: 0}
}

func sampleMax(xs []float64) float64 {
	if len(xs) == 0 {
		return math.NaN()
	}
	m := xs[0]
	for _, v := range xs {
		if v > m {
			m = v
		}
	}
	return m
}

// someLiveAtomIDs returns up to n DISTINCT live atom ids, newest first.
//
// L1 used to fetch ONE id in a loop, after the canary had already fetched that
// same id, so every timed request was a warm repeat of a pre-warmed lookup. If
// first access to an uncached atom cost 15ms and repeat access cost 0.5ms, the
// gate excluded the 15ms request by construction and passed. Distinct ids make
// each timed request a first access for that row.
func someLiveAtomIDs(storePath string, n int) ([]string, error) {
	rows, err := query(storePath, fmt.Sprintf(
		"SELECT id FROM atoms WHERE status='live' AND kind='atom' "+
			"ORDER BY created_at DESC LIMIT %d", n))
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(rows))
	for _, r := range rows {
		if s, ok := r["id"].(string); ok && s != "" {
			out = append(out, s)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no live atoms: L1 has nothing provably present to ask for")
	}
	return out, nil
}

// compareToBaseline reports units that got WORSE than a recorded run.
//
// Exists because of the finding that hurt most in a cross-vendor review: nothing
// invoked this gate, which makes it a benchmark rather than a gate (STYLE.md,
// "a gate nothing invokes is a measurement"). The obvious wiring -- block deploy
// unless ACCEPT -- is unusable while the contract is half-built, and a gate
// people cannot afford to run gets disabled, which is the same outcome with
// extra steps.
//
// So the deploy question is narrower and answerable today: did anything that was
// working stop working? A unit already FAILing stays failing without blocking;
// a unit that was PASS and is now anything else is a regression. ERROR counts as
// a regression from PASS because it means the instrument could no longer answer,
// which is not permission to proceed.
//
// Reads both sides through gatekit's public JSON wire form rather than reaching
// into the Report: the baseline on disk IS that wire form, and adding an
// accessor to a shared library for one consumer's convenience is how a library
// stops being able to enforce anything.
func unitStates(raw []byte) (map[string]string, error) {
	var wire struct {
		Components []struct {
			Unit  string `json:"unit"`
			State string `json:"state"`
		} `json:"components"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, err
	}
	if len(wire.Components) == 0 {
		return nil, fmt.Errorf("no components in report: an empty baseline would " +
			"make every future run trivially pass")
	}
	out := map[string]string{}
	for _, c := range wire.Components {
		out[c.Unit] = c.State
	}
	return out, nil
}

func compareToBaseline(path string, rep *verdict.Report) ([]string, error) {
	prevRaw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	// Load through gatekit first so a tampered or self-inconsistent baseline is
	// rejected by the library rather than trusted by us.
	if _, err := verdict.Load(prevRaw); err != nil {
		return nil, fmt.Errorf("baseline rejected by gatekit: %w", err)
	}
	was, err := unitStates(prevRaw)
	if err != nil {
		return nil, err
	}
	nowRaw, err := json.Marshal(rep)
	if err != nil {
		return nil, err
	}
	now, err := unitStates(nowRaw)
	if err != nil {
		return nil, err
	}

	var out []string
	for unit, before := range was {
		after, present := now[unit]
		switch {
		case !present && before == "PASS":
			// Coverage shrinkage is a regression: a check that no longer runs
			// cannot fail, and its absence reads identically to its success.
			out = append(out, fmt.Sprintf("%s: PASS -> ABSENT (unit disappeared)", unit))
		case present && before == "PASS" && after != "PASS":
			out = append(out, fmt.Sprintf("%s: %s -> %s", unit, before, after))
		}
	}
	sort.Strings(out)
	return out, nil
}
