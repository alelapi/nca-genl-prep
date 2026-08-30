# Lab 5 — Evaluating an LLM

**Time:** 45 minutes · **Covers:** Domain 3 (metrics, benchmarks, judging)

## 1. Classification metrics and the accuracy trap

```python
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

rng = np.random.default_rng(0)
n = 1000
y_true = np.zeros(n, dtype=int)
y_true[rng.choice(n, 20, replace=False)] = 1     # 2% positive class

lazy = np.zeros(n, dtype=int)                     # predicts "negative" always
print("=== lazy model ===")
print("accuracy :", accuracy_score(y_true, lazy))
print("recall   :", recall_score(y_true, lazy, zero_division=0))
print("f1       :", f1_score(y_true, lazy, zero_division=0))
```

**98% accuracy, 0% recall.** Print this and stare at it — it is the single most examinable
fact in the metrics chapter.

```python
real = y_true.copy()
flip = rng.choice(n, 40, replace=False)
real[flip] = 1 - real[flip]

print("\n=== a real (imperfect) model ===")
print(classification_report(y_true, real, digits=3, zero_division=0))
tn, fp, fn, tp = confusion_matrix(y_true, real).ravel()
print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"precision = TP/(TP+FP) = {tp}/{tp+fp} = {tp/(tp+fp):.3f}")
print(f"recall    = TP/(TP+FN) = {tp}/{tp+fn} = {tp/(tp+fn):.3f}")
```

## 2. Precision/recall trade-off via threshold

```python
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

scores = rng.random(n) * 0.6
scores[y_true == 1] += rng.random((y_true == 1).sum()) * 0.6   # positives score higher

p, r, thr = precision_recall_curve(y_true, scores)
print(f"ROC-AUC : {roc_auc_score(y_true, scores):.3f}")
print(f"PR-AUC  : {average_precision_score(y_true, scores):.3f}   <- more honest when imbalanced")

for t in [0.3, 0.5, 0.7, 0.9]:
    pred = (scores >= t).astype(int)
    print(f"threshold {t}: precision={precision_score(y_true, pred, zero_division=0):.3f} "
          f"recall={recall_score(y_true, pred, zero_division=0):.3f}")
```

The model never changed — only the threshold. **Precision vs. recall is a business
decision.**

## 3. BLEU and ROUGE

```python
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer

reference = "The cat sat on the mat because it was warm and comfortable."
candidates = {
    "near-copy":  "The cat sat on the mat because it was warm and cozy.",
    "paraphrase": "A feline rested on the rug since the spot felt pleasantly warm.",
    "truncated":  "The cat sat.",
    "wrong":      "Quantization reduces model memory footprint.",
}

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

print(f"{'CANDIDATE':<13}{'BLEU':>8}{'ROUGE-1':>10}{'ROUGE-2':>10}{'ROUGE-L':>10}")
for name, cand in candidates.items():
    bleu = corpus_bleu([cand], [[reference]]).score
    r = scorer.score(reference, cand)
    print(f"{name:<13}{bleu:>8.1f}{r['rouge1'].fmeasure:>10.3f}"
          f"{r['rouge2'].fmeasure:>10.3f}{r['rougeL'].fmeasure:>10.3f}")
```

**What to notice:** the *paraphrase* is a perfectly good answer and scores terribly. The
*truncated* answer scores better on some n-gram overlap than the paraphrase does. This is
the fundamental limitation of reference-based metrics — and the reason BERTScore and
LLM-as-a-judge exist.

## 4. BERTScore — semantic similarity

```python
from sentence_transformers import SentenceTransformer
import numpy as np

m = SentenceTransformer("all-MiniLM-L6-v2")
ref_v = m.encode(reference, normalize_embeddings=True)
print(f"{'CANDIDATE':<13}{'COSINE'}")
for name, cand in candidates.items():
    print(f"{name:<13}{float(ref_v @ m.encode(cand, normalize_embeddings=True)):.3f}")
```

The paraphrase now scores where it should. (True BERTScore does token-level matching; this
sentence-level version demonstrates the same idea.)

## 5. Perplexity

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
mdl = AutoModelForCausalLM.from_pretrained("gpt2").eval()

def perplexity(text):
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        loss = mdl(**ids, labels=ids["input_ids"]).loss
    return torch.exp(loss).item()

for label, text in [
    ("natural English", "The quick brown fox jumps over the lazy dog in the garden."),
    ("technical",       "Quantization reduces the numeric precision of model weights."),
    ("word salad",      "Purple monday elephant quantum sandwich the running backwards."),
    ("repetitive",      "the the the the the the the the the the"),
]:
    print(f"{label:<18}{perplexity(text):>10.1f}")
```

Lower = the model finds the text more predictable. Note that "repetitive" scores very low
perplexity while being useless text — **perplexity measures fluency, not quality**.

## 6. Retrieval metrics

```python
def recall_at_k(retrieved, relevant, k):
    top = retrieved[:k]
    return len(set(top) & set(relevant)) / len(relevant)

def precision_at_k(retrieved, relevant, k):
    top = retrieved[:k]
    return len(set(top) & set(relevant)) / k

def mrr(retrieved, relevant):
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1 / i
    return 0.0

def dcg(rels):
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))

def ndcg_at_k(retrieved, relevance_map, k):
    gains = [relevance_map.get(d, 0) for d in retrieved[:k]]
    ideal = sorted(relevance_map.values(), reverse=True)[:k]
    return dcg(gains) / dcg(ideal) if dcg(ideal) > 0 else 0.0

retrieved = ["d7", "d2", "d1", "d9", "d3"]
relevant  = ["d1", "d3"]
rel_map   = {"d1": 3, "d3": 2, "d5": 1}

print(f"recall@3    {recall_at_k(retrieved, relevant, 3):.3f}")
print(f"recall@5    {recall_at_k(retrieved, relevant, 5):.3f}")
print(f"precision@3 {precision_at_k(retrieved, relevant, 3):.3f}")
print(f"MRR         {mrr(retrieved, relevant):.3f}")
print(f"nDCG@5      {ndcg_at_k(retrieved, rel_map, 5):.3f}")
```

Change the order of `retrieved` and re-run: recall@5 does not move, but **MRR and nDCG do**.
That is exactly the difference between "did we find it" and "did we rank it well".

## 7. LLM-as-a-judge, and its position bias

```python
JUDGE_PROMPT = """You are evaluating an answer against a source document.

<document>
{document}
</document>

<question>{question}</question>
<answer>{answer}</answer>

Score FAITHFULNESS from 1 to 5: is every claim in the answer supported by the document?
Reply as: SCORE: <n>
REASON: <one sentence>"""

doc = ("LoRA freezes the pretrained weight matrix W and learns a low-rank update BA "
       "with rank r typically between 4 and 64, training roughly 0.1 to 1 percent of parameters.")

answers = {
    "faithful":  "LoRA freezes W and trains a low-rank update BA, about 0.1-1% of parameters.",
    "unfaithful":"LoRA freezes W and trains a low-rank update, which always improves accuracy by 20%.",
    "off-topic": "You should use TensorRT to speed up inference.",
}

for name, ans in answers.items():
    print(f"\n--- {name} ---")
    print(JUDGE_PROMPT.format(document=doc, question="What is LoRA?", answer=ans))
    # feed this to your local model from Lab 4 and read the score
```

!!! tip "Demonstrate position bias yourself"
    Ask a judge model to pick the better of two answers. Then swap their order and ask
    again. If the verdict flips, you have reproduced **position bias** — the reason
    pairwise judging must be run in both orders and averaged.

## 8. A regression harness

The artefact that actually matters in production:

```python
import json

EVAL_SET = [
    {"id": "q1", "question": "What is LoRA?",           "must_contain": ["freez", "low-rank"]},
    {"id": "q2", "question": "What does Triton do?",    "must_contain": ["batch"]},
    {"id": "q3", "question": "Capital of France?",      "must_contain": ["don't know"]},
]

def run_eval(answer_fn, name):
    results, passed = [], 0
    for case in EVAL_SET:
        out = answer_fn(case["question"]).lower()
        ok = all(s.lower() in out for s in case["must_contain"])
        passed += ok
        results.append({"id": case["id"], "pass": ok, "answer": out[:120]})
    print(f"{name}: {passed}/{len(EVAL_SET)} passed")
    return {"name": name, "score": passed / len(EVAL_SET), "results": results}

# report = run_eval(lambda q: rag(q)["answer"], "rag-v1")   # from Lab 4
# json.dump(report, open("eval_rag_v1.json", "w"), indent=2)
```

Run this on every prompt change, model change and index rebuild. **Block merges on
regressions.** That is the whole discipline.

## Takeaways

- **Accuracy lies on imbalanced data** — you have now seen 98% accuracy with 0% recall.
- Precision/recall is set by the **threshold**, and that is a business decision.
- BLEU/ROUGE punish valid paraphrases; BERTScore and judges exist to fix that.
- **Perplexity measures fluency, not quality** — repetitive garbage scores well.
- Recall@k answers "did we find it"; MRR/nDCG answer "did we rank it well".
- LLM judges carry **position and verbosity bias** — swap orders, use rubrics, calibrate.
- A versioned eval set run in CI is what makes LLM development an engineering discipline.
