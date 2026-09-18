"""Head-to-head retrieval eval on Gary's real ChatGPT export.

Corpus: every user+assistant chunk doc the production ChatGPTParser emits
(to_sa_dict, generated query included = shipped graph-build behavior).
Queries: real human turns (chunk-0 text). Ground truth: all chunk docs of
the assistant turn at msg_index+1 in the same conversation. The query
turn's own chunks are excluded from every system's ranking (standard
query-doc removal).

Systems on identical inputs:
  L1-direct : sa.query_with_doc_ids(question)          (library NL behavior)
  L1-readme : MegaExtractor(question) -> per-entity query, max-merge
              (the README-documented NL path)
  BM25      : rank_bm25 BM25Okapi over content tokens
  Dense     : all-MiniLM-L6-v2 (production model), FAISS-equivalent
              normalized inner product via torch matmul

Metrics: Recall@{1,5,10,20} (any relevant chunk in top-k), MRR@10.
Reported on the FULL sample and the answerable subset (>=1 extractable
entity in the question). Known biases stated in the results header.
"""
import json
import random
import re
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from pensive import SpreadingActivation
from pensive.mega_extract import MegaExtractor
from pensive.patterns import REAL_DATA_PATTERNS
from pensive.ingestion.parsers.chatgpt import ChatGPTParser

OUT = os.environ.get('HEAD2HEAD_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'head2head_results.json'))
EXPORT = os.environ.get('CHATGPT_EXPORT', os.path.expanduser('~/chatgpt-export'))
N_QUERIES = 1500
TOP_K = 20
rng = random.Random(42)

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ---- parse ----
parser = ChatGPTParser(EXPORT)
docs = []          # aligned lists
meta = []
for d in parser.parse():
    docs.append(d.to_sa_dict())
    meta.append((d.metadata['conv_id'], d.metadata['msg_index'],
                 d.metadata['role'], d.doc_id, d.content))
log(f"parsed {len(docs)} chunk docs")

# ground-truth index: (conv_id, msg_index) -> {role, chunk doc_ids, chunk0 text}
turns = defaultdict(lambda: {'role': None, 'ids': [], 'chunk0': None})
for conv_id, mi, role, doc_id, content in meta:
    t = turns[(conv_id, mi)]
    t['role'] = role
    t['ids'].append(doc_id)
    if doc_id.endswith('-0'):
        t['chunk0'] = content

# eligible queries: user turn with assistant turn at mi+1
candidates = []
for (conv_id, mi), t in turns.items():
    if t['role'] != 'user' or t['chunk0'] is None:
        continue
    nxt = turns.get((conv_id, mi + 1))
    if nxt and nxt['role'] == 'assistant' and nxt['ids']:
        candidates.append(((conv_id, mi), t['chunk0'], frozenset(nxt['ids']),
                           frozenset(t['ids'])))
log(f"{len(candidates)} (question, answer) pairs available")
sample = rng.sample(candidates, min(N_QUERIES, len(candidates)))

# answerable subset flag
extractor = MegaExtractor(REAL_DATA_PATTERNS)
answerable = [len(extractor.extract(q)) > 0 for _, q, _, _ in sample]
log(f"answerable (>=1 entity in question): {sum(answerable)}/{len(sample)}")

# ---- build SA graph (production style) ----
sa = SpreadingActivation()
sa.build_parallel(docs, workers=8)
sa._compile()
log(f"SA graph built: {len(sa._idx_to_node)} nodes")

# ---- BM25 ----
from rank_bm25 import BM25Okapi
tok = re.compile(r'\w+')
corpus_tokens = [tok.findall(d['content'].lower()) for d in docs]
bm25 = BM25Okapi(corpus_tokens)
doc_ids_arr = [d['id'] for d in docs]
log("BM25 index built")

# ---- Dense (production model) ----
import torch
from sentence_transformers import SentenceTransformer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
emb = model.encode([d['content'] for d in docs], batch_size=512,
                   convert_to_tensor=True, normalize_embeddings=True,
                   show_progress_bar=False)
log(f"dense embeddings built on {device}: {tuple(emb.shape)}")

# ---- ranking helpers ----
def l1_direct(q, exclude):
    rows = sa.query_with_doc_ids(q, top_k=TOP_K + len(exclude))
    return [d for d, _v, _s in rows if d not in exclude][:TOP_K]

def l1_readme(q, exclude):
    scores = {}
    for ent, _t in extractor.extract(q):
        for d, _v, s in sa.query_with_doc_ids(ent, top_k=50):
            if s > scores.get(d, 0.0):
                scores[d] = s
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [d for d, _ in ranked if d not in exclude][:TOP_K]

def bm25_rank(q, exclude):
    s = bm25.get_scores(tok.findall(q.lower()))
    order = np.argsort(-s)[:TOP_K + len(exclude) + 4]
    return [doc_ids_arr[i] for i in order
            if doc_ids_arr[i] not in exclude][:TOP_K]

def dense_rank(qv, exclude):
    sims = (emb @ qv).cpu().numpy()
    order = np.argsort(-sims)[:TOP_K + len(exclude) + 4]
    return [doc_ids_arr[i] for i in order
            if doc_ids_arr[i] not in exclude][:TOP_K]

# pre-embed queries in one batch
q_emb = model.encode([q for _, q, _, _ in sample], batch_size=256,
                     convert_to_tensor=True, normalize_embeddings=True,
                     show_progress_bar=False)
log("query embeddings done")

# ---- run ----
systems = ['l1_direct', 'l1_readme', 'bm25', 'dense']
hits = {s: [] for s in systems}   # per query: rank of first relevant or None
for qi, ((key, q, relevant, own_ids)) in enumerate(sample):
    exclude = own_ids
    ranked = {
        'l1_direct': l1_direct(q, exclude),
        'l1_readme': l1_readme(q, exclude),
        'bm25': bm25_rank(q, exclude),
        'dense': dense_rank(q_emb[qi], exclude),
    }
    for s in systems:
        r = next((i + 1 for i, d in enumerate(ranked[s]) if d in relevant),
                 None)
        hits[s].append(r)
    if (qi + 1) % 250 == 0:
        log(f"{qi+1}/{len(sample)} queries done")

# ---- metrics ----
def metrics(idx):
    out = {}
    for s in systems:
        rs = [hits[s][i] for i in idx]
        n = len(rs)
        out[s] = {
            'recall@1':  sum(1 for r in rs if r and r <= 1) / n,
            'recall@5':  sum(1 for r in rs if r and r <= 5) / n,
            'recall@10': sum(1 for r in rs if r and r <= 10) / n,
            'recall@20': sum(1 for r in rs if r and r <= 20) / n,
            'mrr@10':    sum(1.0 / r for r in rs if r and r <= 10) / n,
            'n': n,
        }
    return out

all_idx = list(range(len(sample)))
ans_idx = [i for i in all_idx if answerable[i]]
results = {
    'corpus_docs': len(docs),
    'n_queries': len(sample),
    'answerable_rate': len(ans_idx) / len(sample),
    'full': metrics(all_idx),
    'answerable_subset': metrics(ans_idx),
    'biases': [
        'ground truth = next assistant turn (positional relevance)',
        'query text = chunk-0 of human turn (long questions truncated ~800 chars)',
        'query turn own chunks excluded from all rankings',
        'graph built production-style incl. generated queries',
    ],
}
with open(OUT, 'w') as fh:
    json.dump(results, fh, indent=1)
log("RESULTS")
for scope in ('full', 'answerable_subset'):
    print(f"\n== {scope} (n={results[scope]['l1_direct']['n']}) ==")
    print(f"{'system':<10} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'MRR@10':>7}")
    for s in systems:
        m = results[scope][s]
        print(f"{s:<10} {m['recall@1']:6.3f} {m['recall@5']:6.3f} "
              f"{m['recall@10']:6.3f} {m['recall@20']:6.3f} {m['mrr@10']:7.3f}")
