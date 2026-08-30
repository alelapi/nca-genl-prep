# Evaluation Metrics & Benchmarks

*The densest page in the course, and the highest-yield. NVIDIA's reading list names GLUE,
machine-translation methods, "Benchmarking Elementary Language Tasks", and Jurafsky & Martin's*
Speech and Language Processing.

---

## 1. Classification metrics

### The confusion matrix

Everything starts here. For a binary classifier there are exactly four outcomes:

```text
                        PREDICTED
                    positive    negative
              ┌──────────────┬──────────────┐
    positive  │      TP      │      FN      │   ← actual positives
              │ true positive│false negative│
  ACTUAL      ├──────────────┼──────────────┤
    negative  │      FP      │      TN      │   ← actual negatives
              │false positive│ true negative│
              └──────────────┴──────────────┘
```

From it:

$$ \text{Precision} = \frac{TP}{TP+FP} \qquad \text{Recall} = \frac{TP}{TP+FN} \qquad F_1 = 2 \cdot \frac{P \cdot R}{P+R} $$

The plain-language versions are worth having ready, because they resolve most questions
instantly:

- **Precision** — *"Of everything I flagged, how much was actually right?"* The denominator is
  **what I predicted**.
- **Recall** — *"Of everything that was actually there, how much did I find?"* The denominator is
  **what actually exists**.

### The trade-off is a business decision

Precision and recall pull against each other. Flag more aggressively and you catch more real
cases (recall ↑) but also more false alarms (precision ↓).

Which one matters depends entirely on **which error costs more**:

| Scenario | Optimise | Because |
| --- | --- | --- |
| Spam filtering | **Precision** | Deleting a real email is far worse than letting spam through |
| Cancer screening | **Recall** | Missing a case is catastrophic; a false alarm means a follow-up test |
| Fraud detection | **Recall** (usually) | Missed fraud costs money directly |
| Content moderation of legitimate posts | **Precision** | Wrongly removing legitimate speech is the expensive error |

This is not a statistical question. It is a question about consequences, and exam scenarios will
give you the consequence in the stem.

**F1** is the harmonic mean, used when you need one balanced number. The harmonic mean (not the
arithmetic mean) is used deliberately: it punishes imbalance. Precision 1.0 with recall 0.0
gives an arithmetic mean of 0.5 but an F1 of **0** — which is the honest assessment of a model
that catches nothing.

### The accuracy trap

!!! danger "The most examinable single fact in this domain"
    A dataset is 99% negative. A model that predicts "negative" for **every single input**
    achieves **99% accuracy** — and catches nothing at all.

    ```text
    1000 transactions, 20 are fraudulent.
    Model: always predict "legitimate".

    accuracy  = 980/1000 = 98.0%     ← looks excellent
    recall    =   0/20   =  0.0%     ← catches zero fraud
    precision =   0/0    = undefined
    F1        =            0.0
    ```

    On imbalanced data, report **precision, recall, F1 or PR-AUC** — never accuracy alone.
    [Lab 5](../labs/05-evaluation.md) computes exactly this so you see it happen.

### ROC-AUC vs. PR-AUC

Both evaluate a classifier across **all possible thresholds**, so they measure ranking quality
rather than performance at one arbitrary cutoff.

- **ROC-AUC** plots true positive rate against false positive rate. Interpretable as "the
  probability that a random positive is ranked above a random negative." 0.5 is random, 1.0 is
  perfect.
- **PR-AUC** plots precision against recall.

**On heavily imbalanced data, prefer PR-AUC.** ROC-AUC's false-positive-rate axis has a huge
denominator (all the negatives), so even a large number of false positives barely moves it —
ROC-AUC can look respectable while the model is unusable in practice. PR-AUC has no such
cushion.

### Multi-class averaging

- **Macro** — compute the metric per class, then average unweighted. **Treats rare classes as
  equally important.**
- **Micro** — aggregate over all instances. **Dominated by frequent classes.**
- **Weighted** — average by class support.

If you care about rare classes, use **macro**. Choosing micro on an imbalanced problem
reintroduces the accuracy trap through the back door.

---

## 2. Perplexity

The intrinsic metric for language models.

$$ \text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i \mid x_{<i})\right) $$

That is the **exponentiated average cross-entropy loss** — nothing more. It is the training loss
in a more interpretable unit.

**The interpretation:** perplexity is the effective number of equally-likely choices the model is
deciding among at each token. Perplexity 1 means perfect certainty. Perplexity 10 means the
model is about as uncertain as if it were choosing uniformly from 10 options. **Lower is
better.**

```text
"The capital of France is ___"

  confident model: P(Paris) = 0.9    →  low perplexity
  uncertain model: P spread over 50 candidates  →  high perplexity
```

### The two limitations that get tested

**It is only comparable across models sharing a tokenizer and test set.** A model with a
finer-grained tokenizer produces more tokens per sentence, each easier to predict, and gets a
lower perplexity without being a better model. Comparing perplexity across tokenizers is
meaningless.

**It measures fluency, not usefulness.** Perplexity says nothing about truthfulness,
helpfulness, safety or task success. A model can have excellent perplexity and be a terrible
assistant — which is exactly why instruction tuning and alignment exist as separate stages.

[Lab 5](../labs/05-evaluation.md) demonstrates the second point directly: repetitive text
("the the the the") scores very **low** perplexity while being worthless.

---

## 3. Generation metrics

### BLEU — machine translation

**Precision-oriented**: what fraction of the *candidate's* n-grams appear in the reference?

- Modified n-gram precision for n = 1 to 4, combined by geometric mean.
- Plus a **brevity penalty**, without which a system could output the single word most likely to
  appear in any reference and score perfect precision.
- Range 0–1, usually reported ×100. Designed as a **corpus-level** metric; sentence-level BLEU is
  noisy.

### ROUGE — summarization

**Recall-oriented**: what fraction of the *reference's* n-grams appear in the candidate?

- **ROUGE-N** — n-gram overlap (ROUGE-1 unigrams, ROUGE-2 bigrams).
- **ROUGE-L** — longest common subsequence. Order-sensitive without requiring contiguity.
- **ROUGE-S** — skip-bigrams.

!!! tip "The distinction to memorise, and the reason behind it"
    **BLEU = precision, for translation. ROUGE = recall, for summarization.**

    And the reason is not arbitrary. A **translation** must not invent content — everything it
    says should be in the source, so you measure *precision*. A **summary** must cover the
    important content of the source — so you measure how much of the reference it *captured*,
    which is *recall*.

    Swapped-answer distractors on this pair are common.

**METEOR** improves on BLEU by matching **stems and synonyms** rather than exact tokens, and by
penalising word-order fragmentation. It correlates better with human judgement at the sentence
level.

### The shared, fatal weakness

All reference-based n-gram metrics measure **surface overlap**, not meaning. Run this comparison
and the problem is obvious:

```text
reference:  "The cat sat on the mat because it was warm."

candidate A: "The cat sat on the mat because it was cozy."   → high BLEU/ROUGE
candidate B: "A feline rested on the rug, as the spot was
              pleasantly warm."                               → near-zero BLEU/ROUGE
candidate C: "The cat sat."                                   → moderate ROUGE-1
```

Candidate B is an excellent paraphrase and scores worse than candidate C, which is a truncation.
The metrics also require reference outputs, which are expensive, and there is rarely only one
correct answer to an open-ended question.

**BERTScore** addresses the semantic gap: embed candidate and reference with a contextual model
and match tokens by cosine similarity. Candidate B now scores appropriately. It still needs a
reference.

---

## 4. LLM-as-a-judge

Use a strong LLM to grade outputs against a rubric.

**Why it took over:** it scales far beyond human review, correlates reasonably well with human
judgement, works **without reference answers** (a rubric is enough), and handles open-ended
subjective tasks that BLEU and ROUGE cannot touch.

**The biases you must be able to name** — these appear directly in questions:

| Bias | Behaviour | Mitigation |
| --- | --- | --- |
| **Position bias** | Prefers whichever response is presented first | Evaluate **both orderings** and average |
| **Verbosity bias** | Prefers longer answers regardless of quality | Rubric that scores conciseness; control for length |
| **Self-enhancement bias** | Prefers text from itself or its own model family | Use a **different model family** as judge |
| **Formatting bias** | Rewards confident tone and neat structure over correctness | Explicit criteria for factual accuracy |

**General mitigations:** a detailed rubric with named criteria; require a written justification
*before* the score (which improves the score's quality, the same way chain-of-thought does); use
pairwise comparison rather than absolute scoring where possible; and — most importantly —
**calibrate against human labels on a sample** so you know how much to trust the judge before you
rely on it.

---

## 5. Human evaluation

Still the gold standard, and the thing judges are calibrated against.

**Formats:**

- **Likert scales** — rate 1–5 on named dimensions (helpfulness, accuracy, tone). Easy to
  collect, but different raters use scales differently.
- **Pairwise preference** — "which of these two is better?" **More reliable than absolute
  scoring**, because humans are much better at comparison than at calibrated judgement. This is
  the same reason [RLHF](rlhf-alignment.md) collects rankings rather than scores.
- **A/B testing in production** — the only method that measures real outcomes. See
  [Experiment Design](experiment-design.md).

Requires clear guidelines, multiple annotators per item, and a measure of **inter-annotator
agreement** (Cohen's κ for two raters, Fleiss' κ for more).

---

## 6. Benchmarks

| Benchmark | What it tests |
| --- | --- |
| **GLUE** | General Language Understanding Evaluation — 9 sentence-level tasks (sentiment, paraphrase, inference). **Named explicitly in NVIDIA's reading list** |
| **SuperGLUE** | Harder successor, created once models saturated GLUE |
| **SQuAD** | Extractive question answering (EM and F1) |
| **MMLU** | Massive Multitask Language Understanding — 57 subjects, multiple choice. The standard broad-knowledge benchmark |
| **HellaSwag / ARC / WinoGrande** | Commonsense reasoning |
| **GSM8K / MATH** | Grade-school and competition mathematics |
| **HumanEval / MBPP** | Code generation, scored by **pass@k** — does any of *k* generated samples pass the unit tests |
| **TruthfulQA** | Resistance to reproducing common human misconceptions |
| **BIG-bench** | A very large, diverse task collection |
| **HELM** | Holistic Evaluation of Language Models — **multi-metric**: accuracy, calibration, robustness, fairness, bias, toxicity, efficiency |
| **MTEB** | Massive Text Embedding Benchmark — the standard for choosing embedding models |
| **Chatbot Arena** | Crowdsourced pairwise human preference, reported as Elo ratings |

**HELM is worth understanding as a philosophy**, not just a name: it exists because reporting a
single accuracy number hides everything else about a model. Holistic evaluation reports several
dimensions simultaneously and refuses to collapse them, on the grounds that a model that is
accurate but toxic is not simply "good".

### Why benchmark scores mislead

**Contamination.** Benchmarks are public, so they leak into training corpora. A model that
memorised the test set scores brilliantly and has learned nothing. This is why scores creep
upward across the industry and why any serious evaluation is **decontaminated** — check that
your test items do not appear in the training data.

**Saturation.** GLUE was replaced by SuperGLUE precisely because models topped it out. A
benchmark near its ceiling stops discriminating between models.

**Construct gap.** MMLU measures multiple-choice academic knowledge. Your application probably
does something else. A high MMLU score is weak evidence that a model will do *your* task well.

!!! important "Public benchmarks select; your eval set decides"
    Use public benchmarks for coarse model selection — narrowing twenty candidates to three.
    Then decide between those three using a **private evaluation set built from your own
    traffic**, which cannot be contaminated and measures the thing you actually care about.

---

## 7. Choosing a metric

1. **Match the task.** Classification → F1 or PR-AUC. Translation → BLEU or COMET.
   Summarization → ROUGE **plus faithfulness**. Open-ended chat → human or LLM judge. RAG → the
   [RAG triad](rag-evaluation.md). Retrieval → recall@k, MRR, nDCG.
2. **Match the cost of errors.** Precision vs. recall is a consequence question.
3. **Use more than one.** A single number always hides something. Pair a quality metric with
   latency, cost and a safety metric — and report subgroup breakdowns, per
   [Bias & Fairness](../domain-5/bias-fairness.md).
4. **Always compare against a baseline** and **report variance**. A 0.3-point gain inside a
   2-point standard deviation is noise.

---

## 8. Recap

- **Precision** = of what I flagged, how much was right. **Recall** = of what existed, how much I
  found. **F1** is their harmonic mean, which punishes imbalance.
- Precision-vs-recall is a **business decision** about which error costs more.
- **Accuracy is misleading on imbalanced data** — 99% accuracy can mean zero detections. Use
  PR-AUC, F1, recall.
- **Perplexity** = exponentiated cross-entropy; **lower is better**; comparable only across
  same-tokenizer models; measures **fluency, not truth**.
- **BLEU = precision, translation. ROUGE = recall, summarization.** Both punish valid paraphrases;
  **BERTScore** measures semantic similarity instead.
- **LLM-as-a-judge** scales evaluation but carries **position, verbosity, self-enhancement and
  formatting** biases — swap orders, use rubrics, use a different model family, calibrate against
  humans.
- **Pairwise preference beats absolute scoring** for human evaluation.
- **GLUE/SuperGLUE** = NLU, **MMLU** = broad knowledge, **HELM** = holistic multi-metric,
  **MTEB** = embeddings, **pass@k** = code.
- Public benchmarks suffer **contamination, saturation and construct gap**. Your private eval set
  is what decides.
