#!/usr/bin/env python3
"""Candidate C: Python reference (sentence-transformers + torch/ROCm).

Known-good baseline; this is the stack the eval harness already uses.
Embeds all 500 sample atoms with bge-small-en-v1.5, reranks 10 (query, doc)
pairs with bge-reranker-base. Prints device, embed throughput (atoms/s),
rerank latency (ms/pair). Pass "cpu" as argv[1] to force CPU.
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

EMBED_ID = "BAAI/bge-small-en-v1.5"
RERANK_ID = "BAAI/bge-reranker-base"
QUERY = "What runtime should the pensive daemon use, and how do embedding and reranking of memory atoms work?"
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "..", "sample_atoms.jsonl")


def load_atoms():
    texts = []
    with open(SAMPLE) as fh:
        for line in fh:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts


def main():
    force_cpu = len(sys.argv) > 1 and sys.argv[1] == "cpu"
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder

    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    texts = load_atoms()
    docs = texts[:10]

    t0 = time.perf_counter()
    embedder = SentenceTransformer(EMBED_ID, device=device)
    reranker = CrossEncoder(RERANK_ID, device=device)
    load_s = time.perf_counter() - t0

    # warmup (excludes lazy CUDA/kernel init from timed region)
    embedder.encode(texts[:32], batch_size=32, show_progress_bar=False)
    reranker.predict([(QUERY, docs[0])])
    if device == "cuda":
        torch.cuda.synchronize()

    # timed embed of all 500
    t0 = time.perf_counter()
    embs = embedder.encode(texts, batch_size=32, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
    if device == "cuda":
        torch.cuda.synchronize()
    embed_s = time.perf_counter() - t0

    # timed rerank of 10 pairs (batched, as the daemon would rerank a candidate set)
    pairs = [(QUERY, d) for d in docs]
    t0 = time.perf_counter()
    scores = reranker.predict(pairs)
    if device == "cuda":
        torch.cuda.synchronize()
    rerank_s = time.perf_counter() - t0

    n = len(texts)
    print(json.dumps({
        "candidate": "python_ref",
        "device": device,
        "embed_model": EMBED_ID,
        "rerank_model": RERANK_ID,
        "embed_dim": int(embs.shape[1]),
        "n_atoms": n,
        "load_s": round(load_s, 3),
        "embed_throughput_atoms_per_s": round(n / embed_s, 1),
        "embed_total_ms": round(embed_s * 1000, 1),
        "rerank_pairs": len(pairs),
        "rerank_latency_ms_per_pair": round(rerank_s * 1000 / len(pairs), 2),
        "rerank_total_ms": round(rerank_s * 1000, 1),
        "top_score": round(float(max(scores)), 4),
    }, indent=2))


if __name__ == "__main__":
    main()
