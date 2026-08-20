// Command tiergate is the Pensive v3.1 tier contract gate.
//
// It answers one question and states it in every assertion it emits:
//
//	"Does the RUNNING Pensive daemon, measured over the transport an agent
//	 actually uses, meet the L1/L2/L3 latency contract WITHOUT having lost
//	 retrieval quality to get there?"
//
// The second clause is the whole point. Gating on P95 alone makes "return
// fewer results" the cheapest way to pass, so latency and quality are one
// composite verdict here and neither can be reported without the other
// (STYLE.md Law 2: no single number certifies).
//
// Doctrine: ~/.claude/STYLE.md. Structural enforcement: aegis/gatekit, whose
// constructors make the historical lies unexpressible. Nothing in this file
// supplies an aggregate; accounting is derived by gatekit from components.
//
// Deliberately a separate process, in a different language, talking over the
// real HTTP transport. An in-process harness measures a daemon that does not
// exist for any caller.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	gateName    = "pensive-tiergate"
	gateVersion = "1"

	// Tier budgets. NOT invented here: Aegis/CLAUDE.md:113 defines the L1/L2/L3
	// tier structure ("Do not break the tier structure"), tightened by Gary
	// 2026-08-12 from <5ms/~15ms/~100-200ms to these values.
	defaultL1BudgetMs = 1.0
	defaultL2BudgetMs = 20.0
	defaultL3BudgetMs = 125.0

	// Quality floors. NOT invented here: daemon/eval/BASELINE_V3.md, where v3
	// was scored against BM25 on an identical 1,500-query sample and the in-run
	// BM25 reproduced its published 0.583/0.432 to the digit. 0.583 is the BM25
	// floor a tier rewrite must never fall below.
	defaultRAt10Floor = 0.583
	// Chunk-retrieval floor, set from the PRE-change measurement of the family
	// this gate was missing (R@10 0.633 on 60 probes) with headroom for sample
	// noise. It is a REGRESSION tripwire, not a transferred claim.
	defaultChunkRAt10Floor = 0.500
	defaultMRRAt10Floor    = 0.432

	// Contamination ceiling. Charon currently drives ~19 recall events/min
	// (91.4% of all traffic). A latency number taken under that load measures
	// the retry loop, so the gate refuses rather than reporting it.
	defaultMaxBackgroundPerMin = 5.0

	// Host-load ceiling, as load1 per core. The sibling ceiling above bounds
	// contamination from other RECALL traffic; this one bounds contamination
	// from the machine. 0.5 means "fewer than half the cores are queued", which
	// leaves generous headroom on an idle box (load 1-2 of 24) while refusing
	// to certify latency during a build, a model load, or a parallel agent
	// fleet. Set from the 2026-08-19 run that rejected on all three latency
	// units at load 19.5/24 cores while every quality unit passed.
	defaultMaxLoadPerCore = 0.5

	// An id that cannot exist: valid ULID alphabet, never issued. The negative
	// half of the self-testing canary (STYLE.md: prefer instruments that carry
	// their own negative half).
	impossibleAtomID = "01ZZZZZZZZZZZZZZZZZZZZZZZZ"
)

type cfg struct {
	daemonURL       string
	storePath       string
	srcDir          string
	l1Budget        float64
	l2Budget        float64
	l3Budget        float64
	rAt10Floor      float64
	chunkRAt10Floor float64
	maxLoadPerCore  float64
	mrrFloor        float64
	maxBackground   float64
	iterations      int
	genProbes       int
	seed            int
	minGenerated    int
	minCurated      int
	plant           string
	epoch           string
	evidenceDir     string
	baseline        string
	evErrs          []string
}

func main() {
	c := cfg{}
	flag.StringVar(&c.daemonURL, "daemon", "http://127.0.0.1:5999", "daemon base URL")
	flag.StringVar(&c.storePath, "store", os.Getenv("HOME")+"/.local/share/pensive-v3/pensive.db", "store path")
	flag.StringVar(&c.srcDir, "src", "../../src", "daemon source dir (staleness check)")
	flag.Float64Var(&c.l1Budget, "l1-budget-ms", defaultL1BudgetMs, "L1 P95 budget ms")
	flag.Float64Var(&c.l2Budget, "l2-budget-ms", defaultL2BudgetMs, "L2 P95 budget ms")
	flag.Float64Var(&c.l3Budget, "l3-budget-ms", defaultL3BudgetMs, "L3 P95 budget ms")
	flag.Float64Var(&c.rAt10Floor, "r10-floor", defaultRAt10Floor, "R@10 floor")
	flag.Float64Var(&c.chunkRAt10Floor, "chunk-r10-floor", defaultChunkRAt10Floor,
		"L3 document_chunk R@10 floor")
	flag.Float64Var(&c.mrrFloor, "mrr-floor", defaultMRRAt10Floor, "MRR@10 floor")
	flag.Float64Var(&c.maxBackground, "max-background-per-min", defaultMaxBackgroundPerMin, "contamination ceiling")
	flag.Float64Var(&c.maxLoadPerCore, "max-load-per-core", defaultMaxLoadPerCore, "host load1-per-core ceiling; above this the gate refuses to quote latency")
	flag.IntVar(&c.iterations, "iterations", 40, "probe iterations per latency unit")
	flag.IntVar(&c.genProbes, "gen-probes", 25, "generated self-retrieval probes")
	flag.IntVar(&c.seed, "seed", 42, "sample seed for generated probes")
	flag.IntVar(&c.minGenerated, "min-generated", 10, "floor: generated probes required")
	flag.IntVar(&c.minCurated, "min-curated", 5, "floor: curated probes required")
	flag.StringVar(&c.plant, "plant", "none", "planted failure: none|latency|empty")
	flag.StringVar(&c.epoch, "epoch", "primary", "epoch label: primary|known-good|planted-bad|mutated")
	flag.StringVar(&c.baseline, "against", "", "path to a previous report.json: fail on any unit that REGRESSED, ignore units that were already failing")
	flag.StringVar(&c.evidenceDir, "evidence-dir", ".gate-evidence", "where raw output is written")
	flag.Parse()

	rep, err := run(&c)
	if err != nil {
		// A gate that cannot construct its report has FAILED, not passed. It
		// must never exit 0 (STYLE.md: a check that did not run must never
		// print a pass).
		fmt.Fprintf(os.Stderr, "tiergate: INSTRUMENT FAILURE: %v\n", err)
		os.Exit(3)
	}
	emit(&c, rep)

	// Regression mode. The full contract is not met yet -- L2 and L3 legitimately
	// FAIL because those tiers are unbuilt -- so gating a deploy on ACCEPT would
	// block the memory daemon from starting and would be un-runnable until the
	// campaign finishes. A gate nobody can afford to run is the same as a gate
	// nobody runs. So against a recorded baseline this asks the question a deploy
	// actually needs answered: did anything that WAS working stop working?
	if c.baseline != "" {
		regressions, err := compareToBaseline(c.baseline, rep)
		if err != nil {
			fmt.Fprintf(os.Stderr, "tiergate: baseline unreadable (%v): refusing to "+
				"report no-regression against a baseline it could not load\n", err)
			os.Exit(3)
		}
		if len(regressions) > 0 {
			fmt.Printf("\nREGRESSIONS vs %s:\n", c.baseline)
			for _, r := range regressions {
				fmt.Println("  " + r)
			}
			os.Exit(1)
		}
		fmt.Printf("\nno regression vs %s\n", c.baseline)
		os.Exit(0)
	}

	switch rep.Outcome() {
	case "ACCEPT":
		os.Exit(0)
	case "TAINTED":
		os.Exit(2)
	default:
		os.Exit(1)
	}
}

// ---------------------------------------------------------------------------
// store reads: sqlite3 -json is the system of record answering in machine form,
// not text munged out of a formatted table (STYLE.md).
// ---------------------------------------------------------------------------

func query(storePath, sql string) ([]map[string]any, error) {
	cmd := exec.Command("sqlite3", "-json", "-readonly", storePath, sql)
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("sqlite3: %v: %s", err, errb.String())
	}
	s := strings.TrimSpace(out.String())
	if s == "" {
		return []map[string]any{}, nil
	}
	var rows []map[string]any
	if err := json.Unmarshal([]byte(s), &rows); err != nil {
		return nil, fmt.Errorf("sqlite3 json: %v", err)
	}
	return rows, nil
}

func scalarInt(storePath, sql string) (int, error) {
	rows, err := query(storePath, sql)
	if err != nil {
		return 0, err
	}
	if len(rows) == 0 {
		return 0, fmt.Errorf("no rows for %q", sql)
	}
	for _, v := range rows[0] {
		switch t := v.(type) {
		case float64:
			return int(t), nil
		case string:
			n, _ := strconv.Atoi(t)
			return n, nil
		}
	}
	return 0, fmt.Errorf("no scalar in result for %q", sql)
}

// ---------------------------------------------------------------------------
// transport: the real MCP path an agent uses, plus the lean L1 route once it
// exists. Both measured client-observed.
// ---------------------------------------------------------------------------

type mcpClient struct {
	base string
	hc   *http.Client
}

func newMCP(base string) *mcpClient {
	return &mcpClient{base: base, hc: &http.Client{Timeout: 90 * time.Second}}
}

// call returns the tool's text content, the wall time, and an error.
func (m *mcpClient) call(tool string, args map[string]any) (string, time.Duration, error) {
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0", "id": 1, "method": "tools/call",
		"params": map[string]any{"name": tool, "arguments": args},
	})
	req, _ := http.NewRequest("POST", m.base+"/mcp/?agent=tiergate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	t0 := time.Now()
	resp, err := m.hc.Do(req)
	if err != nil {
		return "", time.Since(t0), err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	d := time.Since(t0)
	return extractText(raw), d, nil
}

// get exercises the lean L1 route. Its absence is a FAIL, never a SKIP: a
// check that can be silently absent is invisible in an aggregate, so the
// default is failure and skipping must be a deliberate, named act (STYLE.md).
func (m *mcpClient) get(id string) (string, time.Duration, int, error) {
	req, _ := http.NewRequest("GET", m.base+"/get?id="+id, nil)
	t0 := time.Now()
	resp, err := m.hc.Do(req)
	if err != nil {
		return "", time.Since(t0), 0, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	return string(raw), time.Since(t0), resp.StatusCode, nil
}

// extractText pulls tool text out of a JSON-RPC or SSE-framed response. Parsed
// as structure, never substring-matched (STYLE.md: read machine output in
// machine form).
func extractText(raw []byte) string {
	s := string(raw)
	if strings.HasPrefix(strings.TrimSpace(s), "event:") {
		var b strings.Builder
		for _, line := range strings.Split(s, "\n") {
			if strings.HasPrefix(line, "data: ") {
				b.WriteString(line[6:])
			}
		}
		s = b.String()
	}
	var env struct {
		Result struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"result"`
	}
	if err := json.Unmarshal([]byte(s), &env); err != nil {
		return ""
	}
	if len(env.Result.Content) == 0 {
		return ""
	}
	return env.Result.Content[0].Text
}

// ---------------------------------------------------------------------------
// metrics
// ---------------------------------------------------------------------------

// p95 by nearest-rank, matching daemon/eval/gate.py's _percentiles so the two
// harnesses' numbers are comparable rather than merely similar.
func p95(ms []float64) float64 {
	if len(ms) == 0 {
		return math.NaN()
	}
	s := append([]float64(nil), ms...)
	sort.Float64s(s)
	i := int(math.Ceil(0.95*float64(len(s)))) - 1
	if i < 0 {
		i = 0
	}
	if i >= len(s) {
		i = len(s) - 1
	}
	return s[i]
}

func median(ms []float64) float64 {
	if len(ms) == 0 {
		return math.NaN()
	}
	s := append([]float64(nil), ms...)
	sort.Float64s(s)
	return s[len(s)/2]
}

// ---------------------------------------------------------------------------
// probes
// ---------------------------------------------------------------------------

type probe struct {
	Query    string   `json:"query"`
	Expected []string `json:"expected"`
	Origin   string   `json:"origin"` // "generated" | "curated"
}

// generatedProbes derives (query -> expected atom) pairs from the store itself
// rather than from a hand-typed list. A hand-maintained coverage list fails in
// the same direction as the defect it exists to catch (STYLE.md), so the
// primary quality family is regenerable from the system of record: take a live
// atom's own principle line as the query and require that atom back.
//
// This is a FLOOR test, not a realism test. If recall cannot return an atom
// given a sentence out of its own body, nothing downstream is trustworthy. The
// curated family in probes.json supplies the realistic half; two orthogonal
// families are required by Law 2.
func generatedProbes(storePath string, n, seed int) ([]probe, error) {
	// The selection predicate and the extractor must ask the SAME question. An
	// earlier version selected on "contains 'principle:' anywhere" while
	// principleLine() requires a line that STARTS with it, so 9 of every 10
	// sampled atoms were silently dropped and the quality verdict covered a
	// single probe while reporting green. Require the newline here.
	sql := fmt.Sprintf(`
	  SELECT a.id AS id, a.text AS text
	  FROM atoms a
	  WHERE a.status='live' AND a.kind='atom'
	    AND a.text LIKE '%%' || char(10) || 'principle: %%'
	    AND LENGTH(a.text) BETWEEN 200 AND 4000
	  ORDER BY substr(a.id, 1 + (%d %% 20))
	  LIMIT %d`, seed, n)
	rows, err := query(storePath, sql)
	if err != nil {
		return nil, err
	}
	var out []probe
	for _, r := range rows {
		id, _ := r["id"].(string)
		text, _ := r["text"].(string)
		line := principleLine(text)
		if id == "" || len(line) < 40 {
			continue
		}
		out = append(out, probe{Query: trimToRealisticQuery(line, len(out)), Expected: []string{id}, Origin: "generated"})
	}
	return out, nil
}

// generatedChunkProbes derives (query -> expected document_chunk) pairs the same
// way generatedProbes derives memory ones: take a line out of a document's own
// body and require that document back.
//
// WHY THIS FAMILY EXISTS. Without it this gate is structurally incapable of
// seeing the failure it was built to prevent. Both existing families select gold
// from `kind='atom'` (units.go, and the `principle:` predicate above, which only
// reasoning atoms carry), and 5 of the 6 curated expectations are memory atoms
// too. So 64 of 65 gold documents were memory, while document_chunk is 282,985
// of 299,802 live atoms: 94.4% of the corpus scored by 1.5% of the signal.
//
// The consequence is not theoretical. A candidate change that removed chunk
// results from L3 measured a 92% drop in chunk retrieval (R@10 0.633 -> 0.050 on
// a family built exactly like this one), and BOTH existing quality units scored
// it as an IMPROVEMENT, because every document they can score was still there.
// L3's served ids became equal to L2's on 62 of 65 probes. The tier would have
// stopped existing and the gate would have certified it green.
//
// SCORED AT L3 ONLY. L2 restricts to memory kinds by contract, so a chunk probe
// failing at L2 is correct tier behaviour and scoring it there would punish the
// design. That asymmetry IS the tier contract, which is why this family is not
// simply appended to the others.
func generatedChunkProbes(storePath string, n, seed int) ([]probe, error) {
	// Chunks carry no `principle:` line, so the memory extractor cannot reach
	// them. Select on length and require a line with enough word characters to
	// be a query rather than a bracket or an import.
	sql := fmt.Sprintf(`
	  SELECT a.id AS id, a.text AS text
	  FROM atoms a
	  WHERE a.status='live' AND a.kind='document_chunk'
	    AND LENGTH(a.text) BETWEEN 300 AND 4000
	  ORDER BY substr(a.id, 1 + (%d %% 20))
	  LIMIT %d`, seed, n*3)
	rows, err := query(storePath, sql)
	if err != nil {
		return nil, err
	}
	var out []probe
	for _, r := range rows {
		if len(out) >= n {
			break
		}
		id, _ := r["id"].(string)
		text, _ := r["text"].(string)
		line := wordiestLine(text)
		if id == "" || len(line) < 40 {
			continue
		}
		out = append(out, probe{
			Query:    trimToRealisticQuery(line, len(out)),
			Expected: []string{id},
			Origin:   "generated-chunk",
		})
	}
	return out, nil
}

// wordiestLine returns the line of text with the most word characters, which for
// a source chunk is the line most likely to read as a query rather than as
// punctuation. Ties go to the first, so the choice is deterministic for a given
// document and the probe set is reproducible across runs.
func wordiestLine(text string) string {
	best, bestScore := "", 0
	for _, ln := range strings.Split(text, "\n") {
		ln = strings.TrimSpace(ln)
		if len(ln) < 40 || len(ln) > 400 {
			continue
		}
		score := 0
		for _, r := range ln {
			if r == ' ' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
				score++
			}
		}
		if score > bestScore {
			best, bestScore = ln, score
		}
	}
	return best
}

// REALISTIC_QUERY_CHARS caps a generated probe's query length.
//
// Derived from the system of record, not chosen: 2,722 distinct real agent
// queries in recall_log (excluding the Charon loop that produced 91.4% of
// historical traffic) have p50 95 chars, p75 154, p90 223. Generated probes were
// using an atom's ENTIRE principle sentence, median 563 chars, which is past the
// 90th percentile of anything an agent actually asks.
//
// That is not a harmless difference. Encode cost scales with token count, so the
// gate was measuring a workload production does not produce and reporting it as
// the tier's latency: 48ms p50 on the long probes against 18.6ms on realistic
// short ones. A probe set has to look like traffic or its latency number
// describes nothing.
//
// Cut at a word boundary: a mid-word truncation is not a query either.
// The generated probe set SPANS the real query-length distribution rather than
// sitting at one end of it. Measured from recall_log over 2,722 distinct real
// agent queries (the Charon loop excluded, since it was one machine caller
// producing 91.4% of historical traffic and its shape is not an agent's):
//
//	p50  95 chars   p75 154   p90 223
//
// Capping every probe at a single length is wrong in both directions. At the
// p90 the whole set is the 90th-percentile-worst shape, and judging that against
// a p95 LATENCY budget counts the tail twice; at the p50 the set has no tail at
// all. Cycling the three percentiles by index gives a population shaped like
// traffic, deterministically, with no RNG for a gate to disagree with itself on.
//
// This matters more than it sounds: probes were originally an atom's ENTIRE
// principle sentence, median 563 chars, and encode cost scales with tokens. The
// gate was reporting 48ms p50 for a tier that serves real queries in 15ms.
var REALISTIC_QUERY_CHARS = []int{95, 154, 223}

func trimToRealisticQuery(s string, i int) string {
	limit := REALISTIC_QUERY_CHARS[i%len(REALISTIC_QUERY_CHARS)]
	if len(s) <= limit {
		return s
	}
	cut := s[:limit]
	if j := strings.LastIndex(cut, " "); j > limit/2 {
		cut = cut[:j]
	}
	return cut
}

func principleLine(text string) string {
	for _, l := range strings.Split(text, "\n") {
		if strings.HasPrefix(l, "principle: ") {
			return strings.TrimSpace(strings.TrimPrefix(l, "principle: "))
		}
	}
	return ""
}

func curatedProbes(path string) ([]probe, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var ps []probe
	if err := json.Unmarshal(b, &ps); err != nil {
		return nil, err
	}
	for i := range ps {
		ps[i].Origin = "curated"
	}
	return ps, nil
}

// rankOf returns the 1-based rank of the first expected id in the returned
// handle list, or 0 for a miss.
func rankOf(returned []string, expected []string) int {
	want := map[string]bool{}
	for _, e := range expected {
		want[e] = true
	}
	for i, got := range returned {
		if want[got] {
			return i + 1
		}
	}
	return 0
}

// handleIDs parses p3:// handles out of a recall payload. Structure, not
// substring: a handle is the first token of a line beginning with the scheme.
func handleIDs(payload string) []string {
	var out []string
	for _, line := range strings.Split(payload, "\n") {
		t := strings.TrimSpace(line)
		if !strings.HasPrefix(t, "p3://") {
			continue
		}
		rest := t[len("p3://"):]
		if i := strings.IndexAny(rest, " |\t"); i >= 0 {
			rest = rest[:i]
		}
		if rest != "" {
			out = append(out, rest)
		}
	}
	return out
}
