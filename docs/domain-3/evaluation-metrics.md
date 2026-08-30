# Evaluation Metrics & Benchmarks

*The densest, most memorisable page in the course. NVIDIA's reading list names GLUE,
machine-translation methods, benchmarking elementary language tasks, and Jurafsky &
Martin's* Speech and Language Processing.

## The metric cheat sheet

| Metric | Task | What it measures | Direction |
| --- | --- | --- | --- |
| **Accuracy** | Classification | Fraction correct | ↑ |
| **Precision** | Classification | Of predicted positives, how many were right | ↑ |
| **Recall (sensitivity)** | Classification | Of actual positives, how many were found | ↑ |
| **F1** | Classification | Harmonic mean of precision and recall | ↑ |
| **ROC-AUC** | Binary classification | Ranking quality across all thresholds | ↑ |
| **PR-AUC** | Imbalanced classification | Precision–recall trade-off | ↑ |
| **Perplexity** | Language modeling | How "surprised" the model is by the text | ↓ |
| **BLEU** | Machine translation | n-gram **precision** vs. references | ↑ |
| **ROUGE** | Summarization | n-gram **recall** vs. references | ↑ |
| **METEOR** | Translation | Matching with stems/synonyms + word order | ↑ |
| **BERTScore** | Generation | **Semantic** similarity via contextual embeddings | ↑ |
| **Exact match (EM)** | QA | Answer string matches exactly | ↑ |
| **Recall@k / MRR / nDCG** | Retrieval | Ranking quality | ↑ |
| **MSE / RMSE / MAE / R²** | Regression | Error magnitude / explained variance | ↓ / ↓ / ↓ / ↑ |

## Classification metrics

From the confusion matrix (TP, FP, TN, FN):

$$ \text{Precision} = \frac{TP}{TP+FP} \qquad \text{Recall} = \frac{TP}{TP+FN} \qquad F_1 = 2\cdot\frac{P \cdot R}{P + R} $$

**When each matters:**

- **Precision** when false positives are costly — spam filtering (do not delete real
  mail), content moderation of legitimate posts.
- **Recall** when false negatives are costly — disease screening, fraud detection, safety
  filtering.
- **F1** when you need one balanced number.

!!! danger "The accuracy trap"
    With 99% negatives, a model that predicts "negative" always scores **99% accuracy** and
    is useless. On imbalanced data, report **precision, recall, F1 or PR-AUC** — not
    accuracy. This is a guaranteed exam question shape.

**Averaging for multi-class:** *macro* (unweighted mean over classes — treats rare classes
equally), *micro* (aggregate over all instances — dominated by frequent classes),
*weighted* (by class support).

## Perplexity

The intrinsic metric for language models:

$$ \text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i \mid x_{<i})\right) $$

It is the **exponentiated average cross-entropy loss**. Intuitively, the effective number
of equally likely choices the model is deciding among at each token. **Lower is better**;
a perplexity of 10 means the model is about as uncertain as if choosing uniformly among
10 tokens.

Limitations that get examined: it is only comparable **across models sharing the same
tokenizer and test set**, and it measures fluency, **not** truthfulness, helpfulness or
task success. A model can have excellent perplexity and be useless as an assistant.

## Generation metrics

**BLEU** (Bilingual Evaluation Understudy) — machine translation.

- **Precision-oriented**: what fraction of the candidate's n-grams appear in the reference?
- Uses modified n-gram precision for n = 1..4, geometrically averaged.
- Adds a **brevity penalty** so that emitting one perfect word does not score highly.
- Range 0–1 (often ×100). Corpus-level, not sentence-level.

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) — summarization.

- **Recall-oriented**: what fraction of the reference's n-grams appear in the candidate?
- **ROUGE-N** (n-gram overlap), **ROUGE-L** (longest common subsequence — order-sensitive
  without requiring contiguity), **ROUGE-S** (skip-bigrams).

!!! tip "The one-line distinction to memorise"
    **BLEU = precision, for translation. ROUGE = recall, for summarization.**
    A summary should *cover* the source (recall); a translation should not *invent*
    content (precision).

**METEOR** — improves on BLEU by matching stems and synonyms, and penalising word-order
fragmentation. Correlates better with human judgement at the sentence level.

**BERTScore** — embeds candidate and reference with a contextual model and matches tokens
by cosine similarity. Credits paraphrase, which n-gram metrics cannot: *"the cat is on the
mat"* vs. *"a feline rests on the rug"* scores near zero on BLEU and high on BERTScore.

**Shared weakness of all reference-based metrics:** they require reference outputs, they
punish valid alternative phrasings, and they correlate only moderately with human
judgement on open-ended generation. Which is why:

## LLM-as-a-judge

Use a strong LLM to grade outputs against a rubric.

- **Scales** far better than human review and correlates reasonably well with it.
- Works **without reference answers** — grading criteria are enough.
- Handles open-ended, subjective tasks that BLEU/ROUGE cannot touch.

**Known biases you must be able to name:**

| Bias | Behaviour |
| --- | --- |
| **Position bias** | Prefers the first (or a fixed) option in pairwise comparisons |
| **Verbosity bias** | Prefers longer answers regardless of quality |
| **Self-enhancement bias** | Prefers text generated by itself or its own family |
| **Formatting bias** | Rewards confident tone and neat structure over correctness |

**Mitigations:** swap the order and average, use a detailed rubric with explicit criteria,
require a reasoned justification before the score, use a different model family as judge,
and **calibrate against human labels** on a sample.

## Human evaluation

Still the gold standard. Formats: **Likert scales** (rate 1–5 on specified dimensions),
**pairwise preference** (which of these two is better — more reliable than absolute
scoring), and **A/B in production**.

Requires clear guidelines, multiple annotators, and a measure of **inter-annotator
agreement** (Cohen's κ for two raters, Fleiss' κ for more). See
[Alignment & RLHF](rlhf-alignment.md#human-annotation-quality).

## Benchmarks

| Benchmark | Tests |
| --- | --- |
| **GLUE** | General Language Understanding Evaluation — 9 sentence-level tasks (sentiment, paraphrase, inference). Named explicitly in NVIDIA's reading list |
| **SuperGLUE** | Harder successor to GLUE, created once models saturated it |
| **SQuAD** | Extractive question answering (EM and F1) |
| **MMLU** | Massive Multitask Language Understanding — 57 subjects, multiple choice; the standard broad-knowledge benchmark |
| **HellaSwag / ARC / WinoGrande** | Commonsense reasoning |
| **GSM8K / MATH** | Grade-school and competition mathematics |
| **HumanEval / MBPP** | Code generation, measured by **pass@k** (does generated code pass unit tests) |
| **TruthfulQA** | Resistance to reproducing common misconceptions |
| **BIG-bench** | Very large, diverse task collection |
| **HELM** | Holistic Evaluation of Language Models — multi-metric (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency) |
| **MTEB** | Massive Text Embedding Benchmark — the standard for choosing embedding models |
| **Chatbot Arena** | Crowdsourced pairwise human preference (Elo ratings) |

!!! warning "Benchmark caveats"
    - **Contamination** — benchmarks leak into training corpora, inflating scores.
    - **Saturation** — GLUE was superseded precisely because models topped it out.
    - **Construct gap** — a high MMLU score does not mean the model will do *your* task well.

    Public benchmarks are for coarse model selection. **Your private eval set decides.**

## Choosing a metric

1. **Match the task.** Classification → F1/PR-AUC. Translation → BLEU/COMET.
   Summarization → ROUGE + faithfulness. Open-ended chat → human or LLM judge. RAG → the
   [RAG triad](rag-evaluation.md).
2. **Match the cost of errors.** Precision vs. recall is a business decision, not a
   statistical one.
3. **Use more than one.** A single number always hides something — pair a quality metric
   with latency, cost and a safety metric.
4. **Always compare against a baseline** and report variance.

## Key takeaways

- **Precision** = of what I flagged, how much was right. **Recall** = of what existed, how
  much did I find. F1 balances them.
- **Accuracy is misleading on imbalanced data.**
- **Perplexity** = exponentiated cross-entropy; lower is better; comparable only across
  same-tokenizer models; measures fluency, not truth.
- **BLEU = precision/translation. ROUGE = recall/summarization. BERTScore = semantic.**
- LLM-as-a-judge scales evaluation but carries position, verbosity and self-enhancement
  biases — mitigate and calibrate.
- **GLUE** and **SuperGLUE** for language understanding; **MMLU** for broad knowledge;
  **HELM** for holistic evaluation; **MTEB** for embeddings.
