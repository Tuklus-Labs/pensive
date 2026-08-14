"""Export bge-small-en-v1.5 to ONNX and verify it is a drop-in.

Run this to REGENERATE the artifact that `PENSIVE_V3_ONNX_MODEL` points at.
The 127MB graph is deliberately not in git; this script is the thing that is
versioned, because an artifact whose recipe lives in a temp directory is an
artifact nobody can rebuild.

Pooling (CLS, then L2 normalize) is folded INTO the exported graph. A prior
spike export at daemon/spike/models/bge-small/model.onnx emits raw
`last_hidden_state` instead and is NOT interchangeable with this one: it would
require the caller to reimplement pooling, which is exactly the drift this
export exists to prevent.

A subagent reported cosine 0.99999982 with a 0.597 negative control. That is
SUBAGENT provenance and does not become my assessment by being repeated, so
this re-derives it end to end on my own sample before anything ships.

The claim under test: ONNX output lands in the SAME embedding space as the
sentence-transformers path currently used, closely enough that the 340,722
stored vectors stay valid and no re-embed is required.

What would falsify it: any sampled text whose cosine falls near the negative
control. A single one kills the drop-in property, because a stored vector that
no longer matches its own text is a silently wrong retrieval, not a slow one.
"""
import os
os.environ["HIP_VISIBLE_DEVICES"] = "-1"          # stream encode on CPU, house rule
os.environ.setdefault("OMP_NUM_THREADS", "4")

import json
import sqlite3
import sys
import time

import numpy as np

DB = os.path.expanduser("~/.local/share/pensive-v3/pensive.db")
OUT_DIR = os.path.expanduser("~/.local/share/pensive-v3/models")
OUT = os.path.join(OUT_DIR, "bge-small-en-v1.5.onnx")
MODEL = "BAAI/bge-small-en-v1.5"
N_SAMPLE = 250

os.makedirs(OUT_DIR, exist_ok=True)


def realTexts(n):
    """Real atom text AND real queries, read-only. Synthetic strings would not
    exercise the tokenizer the way production does."""
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    texts = [r[0] for r in conn.execute(
        "SELECT text FROM atoms WHERE status='live' AND text IS NOT NULL "
        "AND LENGTH(text) BETWEEN 20 AND 4000 ORDER BY id LIMIT ?", (n,))]
    queries = [r[0] for r in conn.execute(
        "SELECT DISTINCT query FROM recall_log WHERE query IS NOT NULL "
        "AND query NOT LIKE 'narrative_fragment session %' "
        "AND LENGTH(query) BETWEEN 8 AND 500 LIMIT ?", (n // 2,))]
    conn.close()
    return texts, queries


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel

    print(f"torch {torch.__version__}  device=cpu", flush=True)

    texts, queries = realTexts(N_SAMPLE)
    sample = texts + queries
    print(f"sample: {len(texts)} atom texts + {len(queries)} real queries "
          f"= {len(sample)}", flush=True)

    # ---------- reference: exactly what production runs today ----------------
    t0 = time.time()
    st = SentenceTransformer(MODEL, device="cpu")
    ref = st.encode(sample, batch_size=32, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False)
    ref = ref.astype(np.float32, copy=False)
    print(f"reference encoded in {time.time()-t0:.1f}s  shape={ref.shape}",
          flush=True)

    # ---------- export -------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModel.from_pretrained(MODEL).eval()
    enc = tok(["warmup text for tracing"], padding=True, truncation=True,
              max_length=512, return_tensors="pt")

    class Wrapped(torch.nn.Module):
        """CLS pooling + L2 normalize, folded into the graph so the runner
        cannot drift from the reference by reimplementing pooling."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.m(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids).last_hidden_state
            cls = out[:, 0]
            return torch.nn.functional.normalize(cls, p=2, dim=1)

    wrapped = Wrapped(hf).eval()
    t0 = time.time()
    with torch.no_grad():
        # dynamo=False: the classic tracer. torch 2.11 defaults to dynamo=True,
        # which produced a graph whose output name collided ("Duplicate
        # definition of name (embedding)") and whose requested opset downgrade
        # failed in version_converter. The classic path is well-trodden for
        # BERT-shaped models and names cleanly.
        # Output renamed off "embedding" because that name already exists in
        # the graph as the word-embedding module's output.
        torch.onnx.export(
            wrapped,
            (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]),
            OUT,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["sentence_embedding"],
            dynamic_axes={k: {0: "batch", 1: "seq"} for k in
                          ("input_ids", "attention_mask", "token_type_ids")}
            | {"sentence_embedding": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    sizeMb = os.path.getsize(OUT) / 1e6
    print(f"exported in {time.time()-t0:.1f}s -> {OUT} ({sizeMb:.1f} MB)",
          flush=True)
    # bge-small is ~33M params at fp32, so a valid export is ~130MB. The first
    # attempt wrote 1.5MB (a graph with no weights) and I read past it. A number
    # that wrong should stop the run, not decorate it.
    if sizeMb < 50:
        raise RuntimeError(
            f"export is {sizeMb:.1f} MB; a 33M-param fp32 model must be ~130 MB. "
            "The graph almost certainly has no weights.")

    # ---------- candidate ----------------------------------------------------
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(OUT, so, providers=["CPUExecutionProvider"])

    got = []
    for i in range(0, len(sample), 32):
        batch = sample[i:i + 32]
        e = tok(batch, padding=True, truncation=True, max_length=512,
                return_tensors="np")
        out = sess.run(["sentence_embedding"], {
            "input_ids": e["input_ids"].astype(np.int64),
            "attention_mask": e["attention_mask"].astype(np.int64),
            "token_type_ids": e["token_type_ids"].astype(np.int64),
        })[0]
        got.append(out.astype(np.float32, copy=False))
    got = np.vstack(got)

    # ---------- verdict, with the control beside it --------------------------
    paired = np.sum(ref * got, axis=1)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(sample))
    perm[perm == np.arange(len(sample))] = (
        perm[perm == np.arange(len(sample))] + 1) % len(sample)
    control = np.sum(ref * got[perm], axis=1)

    print("\n=== paired cosine, ONNX vs sentence-transformers, same text ===")
    print(f"  n={len(paired)}  min={paired.min():.8f}  "
          f"mean={paired.mean():.8f}  max={paired.max():.8f}")
    print(f"  below 0.9999: {int((paired < 0.9999).sum())}")
    print("=== NEGATIVE CONTROL, ONNX vs a DIFFERENT text's reference ===")
    print(f"  n={len(control)}  min={control.min():.4f}  "
          f"mean={control.mean():.4f}  max={control.max():.4f}")
    for q in (50, 90, 99):
        print(f"  control p{q} = {np.percentile(control, q):.4f}")

    # VERDICT CRITERION, corrected. The first version required
    # paired.min - control.max > 0.3 and returned False on a candidate whose
    # paired min was 0.99999940. That was the instrument failing, not the
    # candidate: control.max is the single most similar ACCIDENTAL pair, and
    # this corpus is full of near-duplicate text (Charon re-emitted the same
    # source chunks thousands of times), so control.max legitimately reaches
    # 0.9756. Gating on it asks "are no two real atoms similar", which is a
    # question about the corpus, not about the encoder.
    #
    # The control's actual job is to prove the metric DISCRIMINATES, i.e. that
    # a ~1.0 paired score is not an artifact of comparing something to itself.
    # A control MEDIAN far below 1.0 establishes that. The drop-in claim itself
    # rests on the paired minimum.
    print(f"\n  separation (paired.min - control.median) = "
          f"{paired.min() - float(np.median(control)):.6f}")

    verdict = bool(paired.min() > 0.9999 and float(np.median(control)) < 0.9)
    print(f"\n  DROP-IN: {verdict}")
    json.dump({
        "paired_min": float(paired.min()), "paired_mean": float(paired.mean()),
        "control_max": float(control.max()), "control_mean": float(control.mean()),
        "n": len(paired), "dropIn": verdict, "onnx": OUT,
        "loadavg": os.getloadavg()[0],
    }, open(os.path.join(os.path.dirname(OUT), "onnx-verify.json"), "w"), indent=2)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
