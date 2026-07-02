// Candidate A: Go, ONNX via onnxruntime_go (yalue binding) + sugarme/tokenizer.
// Embeds all 500 sample atoms with bge-small-en-v1.5, reranks 10 (query, doc)
// pairs with bge-reranker-base. Prints device/provider, embed throughput
// (atoms/s), rerank latency (ms/pair). The onnxruntime_go binding exposes
// CUDA/TensorRT/CoreML/DirectML/OpenVINO providers but no ROCm/MIGraphX, and
// the shared lib here is a CPU build, so execution is CPU.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"time"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	embedID  = "BAAI/bge-small-en-v1.5"
	rerankID = "BAAI/bge-reranker-base"
	query    = "What runtime should the pensive daemon use, and how do embedding and reranking of memory atoms work?"
	batch    = 32
)

func must(err error, ctx string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "FATAL %s: %v\n", ctx, err)
		os.Exit(1)
	}
}

func loadAtoms(path string) []string {
	f, err := os.Open(path)
	must(err, "open sample")
	defer f.Close()
	var out []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var rec struct {
			Text string `json:"text"`
		}
		if json.Unmarshal(line, &rec) == nil && rec.Text != "" {
			out = append(out, rec.Text)
		}
	}
	return out
}

// encodeBatch tokenizes texts and pads to the batch max length.
// Returns flat int64 ids/mask/types and the padded sequence length.
func encodeBatch(tok *tokenizer.Tokenizer, texts []string) (ids, mask, types []int64, seqLen int) {
	encs := make([]*tokenizer.Encoding, len(texts))
	for i, t := range texts {
		e, err := tok.EncodeSingle(t, true)
		must(err, "encode single")
		encs[i] = e
		if len(e.Ids) > seqLen {
			seqLen = len(e.Ids)
		}
	}
	return padEncodings(encs, seqLen)
}

func encodePairs(tok *tokenizer.Tokenizer, q string, docs []string) (ids, mask, types []int64, seqLen int) {
	encs := make([]*tokenizer.Encoding, len(docs))
	for i, d := range docs {
		e, err := tok.EncodePair(q, d, true)
		must(err, "encode pair")
		encs[i] = e
		if len(e.Ids) > seqLen {
			seqLen = len(e.Ids)
		}
	}
	return padEncodings(encs, seqLen)
}

func padEncodings(encs []*tokenizer.Encoding, seqLen int) (ids, mask, types []int64, sl int) {
	n := len(encs)
	ids = make([]int64, n*seqLen)
	mask = make([]int64, n*seqLen)
	types = make([]int64, n*seqLen)
	for i, e := range encs {
		for j := 0; j < len(e.Ids); j++ {
			ids[i*seqLen+j] = int64(e.Ids[j])
			mask[i*seqLen+j] = int64(e.AttentionMask[j])
			types[i*seqLen+j] = int64(e.TypeIds[j])
		}
	}
	return ids, mask, types, seqLen
}

func makeInputs(names []string, ids, mask, types []int64, shape ort.Shape) []ort.Value {
	vals := make([]ort.Value, len(names))
	for i, n := range names {
		var data []int64
		switch n {
		case "input_ids":
			data = ids
		case "attention_mask":
			data = mask
		case "token_type_ids":
			data = types
		default:
			must(fmt.Errorf("unknown input %q", n), "makeInputs")
		}
		t, err := ort.NewTensor(shape, data)
		must(err, "new input tensor")
		vals[i] = t
	}
	return vals
}

func runSession(s *ort.DynamicAdvancedSession, inputs []ort.Value) *ort.Tensor[float32] {
	outputs := []ort.Value{nil}
	must(s.Run(inputs, outputs), "session run")
	for _, v := range inputs {
		v.Destroy()
	}
	return outputs[0].(*ort.Tensor[float32])
}

func main() {
	here, _ := os.Getwd()
	ort.SetSharedLibraryPath(here + "/lib/libonnxruntime.so")
	must(ort.InitializeEnvironment(), "init ort env")
	defer ort.DestroyEnvironment()

	texts := loadAtoms(here + "/../sample_atoms.jsonl")
	docs := texts[:10]

	t0 := time.Now()
	embTok, err := pretrained.FromFile(here + "/../models/bge-small/tokenizer.json")
	must(err, "load embed tokenizer")
	rerTok, err := pretrained.FromFile(here + "/../models/bge-reranker/tokenizer.json")
	must(err, "load rerank tokenizer")

	// multi-threaded CPU session (fair vs torch/ORT defaults; not heroics)
	opts, err := ort.NewSessionOptions()
	must(err, "session options")
	defer opts.Destroy()
	must(opts.SetIntraOpNumThreads(runtime.NumCPU()), "set intra-op threads")

	embIn := []string{"input_ids", "attention_mask", "token_type_ids"}
	embSess, err := ort.NewDynamicAdvancedSession(here+"/../models/bge-small/model.onnx",
		embIn, []string{"last_hidden_state"}, opts)
	must(err, "load embed session")
	defer embSess.Destroy()

	// reranker input names are passed via argv (2 or 3 inputs depending on export)
	rerIn := []string{"input_ids", "attention_mask"}
	if len(os.Args) > 1 && os.Args[1] == "3in" {
		rerIn = []string{"input_ids", "attention_mask", "token_type_ids"}
	}
	rerSess, err := ort.NewDynamicAdvancedSession(here+"/../models/bge-reranker/model.onnx",
		rerIn, []string{"logits"}, opts)
	must(err, "load rerank session")
	defer rerSess.Destroy()
	loadS := time.Since(t0).Seconds()

	embedBatch := func(bt []string) int {
		ids, mask, types, seq := encodeBatch(embTok, bt)
		shape := ort.NewShape(int64(len(bt)), int64(seq))
		out := runSession(embSess, makeInputs(embIn, ids, mask, types, shape))
		data := out.GetData()
		dim := int(out.GetShape()[2])
		// CLS pooling + L2 normalize (result unused; exercises the full path)
		for i := 0; i < len(bt); i++ {
			base := i * seq * dim
			var norm float64
			for d := 0; d < dim; d++ {
				v := float64(data[base+d])
				norm += v * v
			}
			_ = math.Sqrt(norm)
		}
		out.Destroy()
		return dim
	}

	rerankPairs := func(q string, ds []string) float32 {
		ids, mask, types, seq := encodePairs(rerTok, q, ds)
		shape := ort.NewShape(int64(len(ds)), int64(seq))
		out := runSession(rerSess, makeInputs(rerIn, ids, mask, types, shape))
		data := out.GetData()
		best := float32(math.Inf(-1))
		for _, v := range data {
			if v > best {
				best = v
			}
		}
		out.Destroy()
		return best
	}

	// sort by length so batches pad to similar lengths (sentence-transformers
	// does this internally; matching it keeps the CPU comparison fair)
	sorted := make([]string, len(texts))
	copy(sorted, texts)
	sort.Slice(sorted, func(i, j int) bool { return len(sorted[i]) < len(sorted[j]) })

	// warmup + timed embed of all 500 (batches of 32)
	dim := embedBatch(sorted[:batch])
	te := time.Now()
	for i := 0; i < len(sorted); i += batch {
		end := i + batch
		if end > len(sorted) {
			end = len(sorted)
		}
		dim = embedBatch(sorted[i:end])
	}
	embedS := time.Since(te).Seconds()

	// rerank: sugarme/tokenizer's Metaspace pretokenizer panics on the
	// SentencePiece (XLM-RoBERTa) reranker tokenizer for real atom text with
	// multibyte unicode. Guard it so the embed measurement still reports and
	// the rerank blocker is recorded honestly.
	var rerankMsPerPair any = nil
	var rerankTotalMs any = nil
	var topScore any = nil
	var rerankErr string
	func() {
		defer func() {
			if r := recover(); r != nil {
				rerankErr = fmt.Sprintf("%v", r)
			}
		}()
		_ = rerankPairs(query, docs[:1]) // warmup (may panic)
		tr := time.Now()
		top := rerankPairs(query, docs)
		rs := time.Since(tr).Seconds()
		rerankMsPerPair = round(rs*1000/float64(len(docs)), 2)
		rerankTotalMs = round(rs*1000, 1)
		topScore = round(float64(top), 4)
	}()
	if rerankErr != "" {
		rerankErr = "sugarme/tokenizer Metaspace panic on SentencePiece tokenizer: " + rerankErr
	}

	n := len(texts)
	res := map[string]any{
		"candidate":                    "go_ort",
		"runtime":                      "go",
		"backend":                      "onnxruntime_go (yalue, ORT 1.27 CPU) + sugarme/tokenizer",
		"device":                       "cpu",
		"embed_model":                  embedID,
		"rerank_model":                 rerankID,
		"embed_dim":                    dim,
		"n_atoms":                      n,
		"load_s":                       round(loadS, 3),
		"embed_throughput_atoms_per_s": round(float64(n)/embedS, 1),
		"embed_total_ms":               round(embedS*1000, 1),
		"rerank_pairs":                 len(docs),
		"rerank_latency_ms_per_pair":   rerankMsPerPair,
		"rerank_total_ms":              rerankTotalMs,
		"top_score":                    topScore,
		"rerank_error":                 rerankErr,
	}
	b, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println(string(b))
}

func round(f float64, places int) float64 {
	p := math.Pow(10, float64(places))
	return math.Round(f*p) / p
}
