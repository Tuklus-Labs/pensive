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
	defaultRAt10Floor   = 0.583
	defaultMRRAt10Floor = 0.432

	// Contamination ceiling. Charon currently drives ~19 recall events/min
	// (91.4% of all traffic). A latency number taken under that load measures
	// the retry loop, so the gate refuses rather than reporting it.
	defaultMaxBackgroundPerMin = 5.0

	// An id that cannot exist: valid ULID alphabet, never issued. The negative
	// half of the self-testing canary (STYLE.md: prefer instruments that carry
	// their own negative half).
	impossibleAtomID = "01ZZZZZZZZZZZZZZZZZZZZZZZZ"
)

type cfg struct {
	daemonURL     string
	storePath     string
	srcDir        string
	l1Budget      float64
	l2Budget      float64
	l3Budget      float64
	rAt10Floor    float64
	mrrFloor      float64
	maxBackground float64
	iterations    int
	genProbes     int
	seed          int
	minGenerated  int
	minCurated    int
	plant         string
	epoch         string
	evidenceDir   string
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
	flag.Float64Var(&c.mrrFloor, "mrr-floor", defaultMRRAt10Floor, "MRR@10 floor")
	flag.Float64Var(&c.maxBackground, "max-background-per-min", defaultMaxBackgroundPerMin, "contamination ceiling")
	flag.IntVar(&c.iterations, "iterations", 40, "probe iterations per latency unit")
	flag.IntVar(&c.genProbes, "gen-probes", 25, "generated self-retrieval probes")
	flag.IntVar(&c.seed, "seed", 42, "sample seed for generated probes")
	flag.IntVar(&c.minGenerated, "min-generated", 10, "floor: generated probes required")
	flag.IntVar(&c.minCurated, "min-curated", 5, "floor: curated probes required")
	flag.StringVar(&c.plant, "plant", "none", "planted failure: none|latency|empty")
	flag.StringVar(&c.epoch, "epoch", "primary", "epoch label: primary|known-good|planted-bad|mutated")
	flag.StringVar(&c.evidenceDir, "evidence-dir", ".gate-evidence", "where raw output is written")
	flag.Parse()

	rep, err := run(c)
	if err != nil {
		// A gate that cannot construct its report has FAILED, not passed. It
		// must never exit 0 (STYLE.md: a check that did not run must never
		// print a pass).
		fmt.Fprintf(os.Stderr, "tiergate: INSTRUMENT FAILURE: %v\n", err)
		os.Exit(3)
	}
	emit(c, rep)
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
		out = append(out, probe{Query: line, Expected: []string{id}, Origin: "generated"})
	}
	return out, nil
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
