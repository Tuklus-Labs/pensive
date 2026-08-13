package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"aegis/gatekit/verdict"
)

// ev writes raw output to disk and returns the pointer. A verdict without a
// pointer to its raw output is a rumor with a checkmark (STYLE.md L7).
func (c cfg) ev(unit string, payload any) string {
	dir := filepath.Join(c.evidenceDir, c.epoch)
	_ = os.MkdirAll(dir, 0o755)
	p := filepath.Join(dir, unit+".json")
	b, _ := json.MarshalIndent(payload, "", "  ")
	_ = os.WriteFile(p, b, 0o644)
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

func run(c cfg) (*verdict.Report, error) {
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
		n, err := scalarInt(c.storePath, fmt.Sprintf(
			"SELECT COUNT(*) FROM recall_log WHERE recorded_at > strftime('%%s','now') - 60 "+
				"AND query NOT IN (%s)", sqlQuoteList(append(probeQueries(gen), probeQueries(cur)...))))
		if err != nil {
			add("contamination", "contamination.background-traffic", verdict.Fail,
				c.ev("contamination", map[string]any{"question": q, "error": err.Error()}), nil)
		} else {
			bgPerMin = float64(n)
			add("contamination", "contamination.background-traffic", state(bgPerMin <= c.maxBackground),
				c.ev("contamination", map[string]any{
					"question": q, "background_events_last_60s": n,
					"ceiling_per_min": c.maxBackground,
					"verdict_meaning": "FAIL means this run's latency figures are contaminated and must not be quoted",
				}), nil)
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
		if kerr == nil {
			for i := 0; i < c.iterations; i++ {
				body, d, code, err := m.get(knownID)
				lastCode, lastErr = code, err
				lats = append(lats, plantLatency(c, float64(d.Microseconds())/1000.0))
				if err != nil || code != 200 || !strings.Contains(body, knownID) {
					exact = false
				}
			}
		} else {
			exact = false
		}
		got := p95(lats)
		routeUp := lastErr == nil && lastCode == 200
		add("latency", "l1.latency.p95", state(routeUp && got <= c.l1Budget),
			c.ev("l1-latency", map[string]any{
				"question": q, "p95_ms": got, "p50_ms": median(lats), "budget_ms": c.l1Budget,
				"n": len(lats), "http_status": lastCode, "error": errStr(lastErr), "plant": c.plant,
				"verdict_meaning": "FAIL means either the lean L1 route does not exist yet, or it exists and is over budget. The http_status field distinguishes them.",
			}), &verdict.Score{Value: got, Max: 0})
		add("correctness", "l1.correctness.exact", state(routeUp && exact),
			c.ev("l1-correctness", map[string]any{
				"question":        "QUESTION: does exact lookup return exactly the atom requested, on every iteration?",
				"atom_id":         knownID, "iterations": len(lats), "all_exact": exact, "http_status": lastCode,
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
			"question":        "QUESTION: how many probes did this run actually exercise, in each family, and does that clear the floor each family must maintain?",
			"generated":       len(gen), "curated": len(cur), "total": nProbes,
			"min_generated":   c.minGenerated, "min_curated": c.minCurated,
			"generated_error": errStr(gerr), "curated_error": errStr(cerr),
			"verdict_meaning": "FAIL means a probe family fell below its floor, so the quality verdict covers less than it appears to",
		}), nil)

	// --- L2 tier ------------------------------------------------------------
	// The L2 route does not exist yet. Absent is FAIL, never SKIP: an absent
	// check and a passing check are indistinguishable in a tally.
	{
		q := fmt.Sprintf("QUESTION: is client-observed P95 for lexical+facet+local-dense retrieval (no cross-encoder, no remote) <= %.3gms?", c.l2Budget)
		_, _, code, err := m.get("__tier_probe__")
		l2Exists := false // no L2 route implemented yet; asserted, not assumed
		add("latency", "l2.latency.p95", state(l2Exists),
			c.ev("l2-latency", map[string]any{
				"question": q, "budget_ms": c.l2Budget, "route_implemented": l2Exists,
				"probe_http_status": code, "probe_error": errStr(err),
				"verdict_meaning": "FAIL because no L2 tier exists in the v3 daemon: there is one monolithic recall path and every caller pays for all of it",
			}), nil)
	}

	// --- L3 tier: the current monolithic recall path -------------------------
	{
		q := fmt.Sprintf("QUESTION: is client-observed P95 for full semantic retrieval <= %.3gms, WITHOUT quality falling below the BM25 floor?", c.l3Budget)
		var lats []float64
		var hits []int
		all := append(append([]probe{}, gen...), cur...)
		for _, p := range all {
			payload, d, err := m.call("recall", map[string]any{"query": p.Query, "k": 10, "tokenBudget": 900})
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
		got := p95(lats)
		add("latency", "l3.latency.p95", state(len(lats) > 0 && got <= c.l3Budget),
			c.ev("l3-latency", map[string]any{
				"question": q, "p95_ms": got, "p50_ms": median(lats), "budget_ms": c.l3Budget,
				"n": len(lats), "plant": c.plant,
				"verdict_meaning": "FAIL means full semantic retrieval is over the L3 budget as measured by a client",
			}), &verdict.Score{Value: got, Max: 0})

		r10, mrr := metrics(hits)
		add("quality", "l3.quality.r_at_10", state(len(hits) > 0 && r10 >= c.rAt10Floor),
			c.ev("l3-quality-r10", map[string]any{
				"question":        fmt.Sprintf("QUESTION: is R@10 still at or above the BM25 floor %.3f after any latency work?", c.rAt10Floor),
				"r_at_10":         r10, "floor": c.rAt10Floor, "n": len(hits), "plant": c.plant,
				"ranks":           hits,
				"verdict_meaning": "FAIL means quality was traded for speed. This is the unit that makes 'return fewer results' an unprofitable optimization.",
			}), &verdict.Score{Value: r10, Max: 1.0})
		add("quality", "l3.quality.mrr_at_10", state(len(hits) > 0 && mrr >= c.mrrFloor),
			c.ev("l3-quality-mrr", map[string]any{
				"question":        fmt.Sprintf("QUESTION: is MRR@10 still at or above the BM25 baseline %.3f?", c.mrrFloor),
				"mrr_at_10":       mrr, "floor": c.mrrFloor, "n": len(hits), "plant": c.plant,
				"verdict_meaning": "FAIL means ordering degraded even where the right answer is still somewhere in the top 10",
			}), &verdict.Score{Value: mrr, Max: 1.0})
	}

	return verdict.NewReport(id, "canary.known-id-roundtrips", comps)
}

// plantLatency implements the planted failures. A plant must be an edit a real
// author would plausibly make (STYLE.md), so:
//   - "latency" is the slow-regression shape
//   - "empty"   is the REWARD HACK shape: return nothing, very fast. The gate
//     must reject it on quality even though every latency unit goes green.
func plantLatency(c cfg, ms float64) float64 {
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

func daemonProc() (int, time.Time, error) {
	out, err := exec.Command("pgrep", "-f", "pensived-v3").Output()
	if err != nil {
		return 0, time.Time{}, fmt.Errorf("daemon not running (pgrep pensived-v3 found nothing)")
	}
	f := strings.Fields(strings.TrimSpace(string(out)))
	if len(f) == 0 {
		return 0, time.Time{}, fmt.Errorf("daemon not running")
	}
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
func emit(c cfg, r *verdict.Report) {
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
