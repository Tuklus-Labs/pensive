# Pensive v3 runtime spike: embed + rerank, decided by evidence

**Decision: Python (documented exception).** It is the only runtime that reaches
the AMD 7900 XTX on this box, and it is the known-good stack the eval harness
already uses. Both ONNX candidates (Go, TS/Bun) run CPU-only here and land far
outside the viability bar; Go's rerank path could not run at all.

- **Embedding model:** `BAAI/bge-small-en-v1.5` (384-dim, CLS pooling), HF snapshot `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- **Reranker model:** `BAAI/bge-reranker-base` (XLM-RoBERTa cross-encoder, single relevance logit), HF snapshot `2cfc18c9415c912f9d8155881c133215df768a70`

This runtime is fixed for the rest of the v3 plan.

---

## Numbers

Protocol: embed all 500 atoms from `sample_atoms.jsonl` in batches of 32 (sorted
by length to minimize padding, which sentence-transformers does internally);
rerank 10 `(query, doc)` pairs as a single batch. Warmup excluded. Throughput =
atoms / embed-seconds; rerank latency = batch-ms / 10.

The table shows the **committed run** (the `result*.json` files), so every number
is reproducible from the artifacts.

| Candidate | Backend | Device | Embed (atoms/s) | Rerank (ms/pair) | GPU EP available? |
|---|---|---|---|---|---|
| **C: Python (reference)** | sentence-transformers 5.4.1 / torch 2.11.0+rocm7.2 | **ROCm GPU (7900 XTX)** | **196.8** | **16.5** | yes, via torch/HIP |
| C: Python | same | CPU | 22.4 | 92.0 | n/a |
| **A: Go** | onnxruntime_go v1.31 (ORT 1.27 CPU) + sugarme/tokenizer v0.3.0 | CPU | 18.7 | **BLOCKED** | no |
| **B: TS/Bun** | Transformers.js v4 / onnxruntime-node 1.24.3 | CPU | 12.5 | 172.0 | no |

Raw captures: `python_ref/result_gpu.json`, `python_ref/result_cpu.json`,
`go_ort/result.json`, `ts_bun/result.json`.

**On measurement noise.** CPU-path numbers were captured on a heavily loaded
workstation (loadavg 33–44 on 24 threads: a concurrent ~15-core ffmpeg film
encode, a keyframe render, the GPU thermal burn, several Claude sessions), so
absolute CPU throughput is depressed and noisy. Earlier, less-contended runs of
the same code were faster (Go embed reached ~27 atoms/s, Python-CPU ~28); those
were exploratory and are not the committed numbers, so the table above uses the
committed captures. Re-running idle would lift all three CPU numbers roughly
proportionally.

**The GPU figures are uncontended-GPU numbers, and their magnitude varies with
GPU contention.** An independent reviewer's spot run with the GPU already at 100%
(thermal burn + film encode) got 71.8 atoms/s embed (lower) but 4.09 ms/pair
rerank (faster than the committed 16.5). So the absolute GPU throughput is not a
fixed figure. The decision does not rest on the magnitude; it rests on the
architectural lockout below (only torch reaches the GPU), which no amount of
contention changes.

**The architectural finding (load-independent).** On this AMD/ROCm box, ONNX
Runtime has no GPU execution provider available, in any of the three ONNX
surfaces:

- Python `onnxruntime` 1.24.4 → `get_available_providers()` = `['CPUExecutionProvider']`
- `onnxruntime_go` v1.31 → exposes CUDA / TensorRT / CoreML / DirectML / OpenVINO, **no ROCm/MIGraphX**
- `onnxruntime-node` 1.24.3 → CPU build, no ROCm
- `torch` 2.11.0+rocm7.2 → `cuda.is_available()` = True, device = "AMD Radeon RX 7900 XTX"

So the GPU is reachable only through torch, which is the Python path. Getting an
ONNX runtime onto the 7900 XTX would require a from-source onnxruntime build with
the ROCm/MIGraphX EP: exactly the "heroics" the decision rule excludes.

---

## Decision rule, applied

Rule: pick the highest-preference candidate (Go > TS > Python) that reaches within
~2x of the Python reference on **both** embed throughput and rerank latency
without heroics; if only Python is viable, record it as a documented exception
with the specific blocker for each other candidate.

**The binding reference is the Python reference as the eval harness actually runs
it** (brief Step 4: "the known-good reference; it is what the eval harness already
uses" and "the performance bar the others must approach"). On this box that is
torch-ROCm on the GPU: 196.8 atoms/s, 16.5 ms/pair. Applying the 2x rule against
that binding reference:

- **Go — not viable.** CPU-only (no ROCm EP). Committed embed 18.7 atoms/s is ~10x
  below the reference (~7x even at the best exploratory ~27), well outside 2x. And
  rerank could not run at all: the pure-Go tokenizer panics on the SentencePiece
  reranker. Two independent disqualifiers.
- **TS/Bun — not viable.** CPU-only (no ROCm EP). Committed embed 12.5 atoms/s is
  ~16x below the reference; rerank 172 ms/pair is ~10x. Outside 2x on both.
- **Python — viable, and the only GPU path.** It is the reference itself and the
  only runtime with working ROCm here.

**A note on the alternate (non-binding) reading.** If one instead used
Python-**CPU** as the bar (22.4 atoms/s, 92.0 ms/pair), a purely mechanical
re-application of the 2x rule would select **TS/Bun**, not Python: from the
committed JSONs TS is within 2x on both metrics (embed 22.4/12.5 = 1.79x; rerank
171.96/91.97 = 1.87x), and it outranks Go (Go's rerank is still BLOCKED). So the
outcome is **not** "Python holds under any reading." It holds because the GPU
eval-harness reference is the binding one per the brief, and under it both ONNX
candidates are an order of magnitude off. TS's closeness to Python-**CPU** is real
but not the governing comparison: the eval harness runs on the GPU, and choosing
either ONNX runtime forgoes the GPU entirely.

For a decades-scale memory daemon that re-embeds and reranks at scale, giving up
the GPU to gain a Go or TS daemon shell is the wrong trade. Decision: **Python**.

(If a prebuilt onnxruntime with a ROCm/MIGraphX EP ever lands on this box, Go
becomes worth re-testing: its embed path already works and its raw-binding
throughput was the best of the two ONNX candidates. That is a future re-eval,
not this plan.)

---

## Build friction, per candidate

### C: Python (reference) — trivial
Import `sentence_transformers` and load `CrossEncoder`; both models load from the
existing HF cache. torch 2.11.0+rocm7.2 puts them on the 7900 XTX with no extra
setup. This is the stack the eval harness already uses. Effort: minutes.

### A: Go — high, and rerank is blocked
1. **cgo ONNX binding** (`github.com/yalue/onnxruntime_go` v1.31.0). Fetched fine.
2. **ORT API-version mismatch.** The binding hardcodes `ORT_API_VERSION 26`, but
   the onnxruntime shared lib bundled by onnxruntime-node is 1.24.3 (supports API
   ≤ 24), so `InitializeEnvironment()` failed with "requested API version [26] is
   not available". Fixed by fetching the official `onnxruntime-linux-x64-1.27.0`
   release tarball into `go_ort/ort_lib/` and pointing `SetSharedLibraryPath` at
   it (local dir, no system install, no sudo). CPU build (the only GPU ORT release
   is CUDA, useless on AMD).
3. **Model files.** bge-small onnx pulled from HF `onnx/model.onnx`; the reranker
   onnx was exported locally from the cached safetensors via `torch.onnx`
   (dynamo=False) after the HF CDN download stalled.
4. **Embedding path works** (bge-small + BERT wordpiece via sugarme). With intra-op
   threads + length-sorted batches its CPU embed matched Python-CPU.
5. **Rerank path BLOCKED.** `sugarme/tokenizer` v0.3.0 panics
   (`slice bounds out of range` in its Metaspace pretokenizer) on the SentencePiece
   (XLM-RoBERTa) reranker tokenizer when fed real atom text containing multibyte
   unicode (em-dash, emoji). The crash is on genuine data, not an edge case.
   Escaping it means a CGo binding to HF's Rust `tokenizers` (adds a Rust
   toolchain, disallowed here) or hand-rolling SentencePiece. Both are heroics, so
   this was capped and recorded per the spike's timebox rule.

### B: TS/Bun — moderate, works end to end
1. `bun add @huggingface/transformers` (v4.2.0) pulls `onnxruntime-node` 1.24.3.
2. **Native binary postinstall blocked** by Bun; needed `bun pm trust onnxruntime-node`
   to fetch the linux-x64 `libonnxruntime.so`.
3. **HF hub download flaky.** The xet CDN reset mid-download of the 1.1GB reranker
   onnx (ECONNRESET). Switched to a local model layout under `tj_models/` and set
   `env.allowRemoteModels=false`.
4. **onnx export shape bug.** The reranker first exported with torch's new dynamo
   exporter baked the dummy shape (MatMul dim mismatch at batch>1). Re-exported
   with the legacy TorchScript exporter (`dynamo=False`), which honors
   `dynamic_axes`. After that, embed + rerank both run.
5. CPU-only on AMD (onnxruntime-node has no ROCm EP), so throughput is well below
   the GPU bar.

---

## Sample provenance (Step 1)

The named live store `~/.local/share/engram/` was **empty at spike time**:
`engram.db` is 0 bytes, the live chrema store `~/.local/share/pensive-chrema-store/`
held only 9 committed rows, and `pensive_wal.db` is an empty write queue. So the
realistic 500-atom sample was drawn **read-only** from a real historical Engram
atom dump, `~/Projects/Engram-tsp/bench/snapshot_meta.jsonl` (9,996 real atoms;
`summary` = atom text), sampled evenly across the file, keeping atoms ≥ 20 chars.
Result: `sample_atoms.jsonl`, 500 lines of `{"id","text"}`, median ~62 words per
atom. No production database was opened in write mode.

**`sample_atoms.jsonl` is intentionally untracked (gitignored).** That real
Engram dump contains third-party sensitive content (real names, personal emails,
financial and legal documents), so the sample must not live in git history. The
file stays on disk for benchmark reproduction; regenerate it deterministically
from the read-only source with:

```bash
python3 - <<'PY'
import json
src = "/home/aegis/Projects/Engram-tsp/bench/snapshot_meta.jsonl"
rows = []
with open(src) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = (d.get("summary") or "").strip()
        if len(t) >= 20:
            rows.append((d.get("rowid"), t))
N = 500
step = len(rows) / N
sample = [rows[int(i * step)] for i in range(N)]
with open("daemon/spike/sample_atoms.jsonl", "w") as out:
    for rid, t in sample:
        out.write(json.dumps({"id": rid, "text": t}) + "\n")
print("wrote", N, "atoms")
PY
```

---

## Reproduce

```bash
# Python reference (GPU, then CPU)
python3 python_ref/embed_rerank.py         # ROCm GPU
python3 python_ref/embed_rerank.py cpu      # force CPU

# Go (needs models/ + go_ort/lib/libonnxruntime.so; see below)
cd go_ort && LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH" ./spike_go

# TS/Bun (needs tj_models/ + ts_bun/node_modules)
cd ts_bun && bun run embed_rerank.ts
```

Heavy artifacts are gitignored and re-fetchable:
- `models/`, `tj_models/`: bge-small onnx from HF `onnx/model.onnx`; reranker onnx
  exported from the cached safetensors with `torch.onnx.export(..., dynamo=False)`.
- `go_ort/lib/libonnxruntime.so`: from the `onnxruntime-linux-x64-1.27.0` release
  tarball (`go_ort/ort_lib/`).
- `ts_bun/node_modules`: `bun add @huggingface/transformers` + `bun pm trust onnxruntime-node`.
