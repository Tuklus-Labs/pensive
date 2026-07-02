// Candidate B: TypeScript on Bun, ONNX via onnxruntime-node (Transformers.js v4).
// Embeds all 500 sample atoms with bge-small-en-v1.5, reranks 10 (query, doc)
// pairs with bge-reranker-base. Prints device/provider, embed throughput
// (atoms/s), rerank latency (ms/pair). onnxruntime-node has no AMD ROCm EP,
// so execution is CPU.
import {
  pipeline,
  AutoTokenizer,
  AutoModelForSequenceClassification,
  env,
} from "@huggingface/transformers";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const EMBED_ID = "BAAI/bge-small-en-v1.5";
const RERANK_ID = "BAAI/bge-reranker-base";
const QUERY =
  "What runtime should the pensive daemon use, and how do embedding and reranking of memory atoms work?";

const here = dirname(fileURLToPath(import.meta.url));
const samplePath = join(here, "..", "sample_atoms.jsonl");
const texts: string[] = readFileSync(samplePath, "utf8")
  .split("\n")
  .filter((l) => l.trim().length > 0)
  .map((l) => JSON.parse(l).text as string);
const docs = texts.slice(0, 10);

function chunk<T>(arr: T[], n: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n));
  return out;
}

async function main() {
  // Read onnx + tokenizer from the local layout (HF hub CDN was flaky); this
  // is the same onnx/model.onnx the Go path uses, so the comparison is fair.
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = join(here, "..", "tj_models");
  const DEVICE = "cpu"; // onnxruntime-node: no AMD ROCm EP available

  const nThreads = navigator.hardwareConcurrency ?? 12;
  const sessionOptions = { intraOpNumThreads: nThreads }; // fair vs torch/ORT defaults

  const t0 = performance.now();
  const extractor = await pipeline("feature-extraction", EMBED_ID, {
    dtype: "fp32",
    device: DEVICE,
    session_options: sessionOptions,
  });
  const tokenizer = await AutoTokenizer.from_pretrained(RERANK_ID);
  const reranker = await AutoModelForSequenceClassification.from_pretrained(
    RERANK_ID,
    { dtype: "fp32", device: DEVICE, session_options: sessionOptions },
  );
  const loadS = (performance.now() - t0) / 1000;

  // warmup
  await extractor(texts.slice(0, 32), { pooling: "cls", normalize: true });
  {
    const w = tokenizer([QUERY], { text_pair: [docs[0]], padding: true, truncation: true });
    await reranker(w);
  }

  // timed embed of all 500 (batches of 32); sort by length to minimize padding
  // (sentence-transformers does this internally; matches the Go path for fairness)
  const sorted = [...texts].sort((a, b) => a.length - b.length);
  const batches = chunk(sorted, 32);
  let embedDim = 0;
  const te = performance.now();
  for (const b of batches) {
    const out: any = await extractor(b, { pooling: "cls", normalize: true });
    embedDim = out.dims[out.dims.length - 1];
  }
  const embedS = (performance.now() - te) / 1000;

  // timed rerank of 10 pairs (single batch); guarded so embed metrics still print
  const queries = docs.map(() => QUERY);
  let rerankMsPerPair: number | null = null;
  let rerankTotalMs: number | null = null;
  let topScore: number | null = null;
  let rerankError = "";
  try {
    const tr = performance.now();
    const enc = tokenizer(queries, { text_pair: docs, padding: true, truncation: true });
    const { logits } = await reranker(enc);
    const rerankS = (performance.now() - tr) / 1000;
    rerankMsPerPair = +((rerankS * 1000) / docs.length).toFixed(2);
    rerankTotalMs = +(rerankS * 1000).toFixed(1);
    topScore = +Math.max(...(logits.tolist() as number[][]).map((x) => x[0])).toFixed(4);
  } catch (e) {
    rerankError = String(e);
  }

  const n = texts.length;
  console.log(
    JSON.stringify(
      {
        candidate: "ts_bun",
        runtime: `bun ${Bun.version}`,
        backend: "onnxruntime-node (Transformers.js v4)",
        device: DEVICE,
        embed_model: EMBED_ID,
        rerank_model: RERANK_ID,
        embed_dim: embedDim,
        n_atoms: n,
        load_s: +loadS.toFixed(3),
        embed_throughput_atoms_per_s: +(n / embedS).toFixed(1),
        embed_total_ms: +(embedS * 1000).toFixed(1),
        rerank_pairs: docs.length,
        rerank_latency_ms_per_pair: rerankMsPerPair,
        rerank_total_ms: rerankTotalMs,
        top_score: topScore,
        rerank_error: rerankError,
      },
      null,
      2,
    ),
  );
}

main();
