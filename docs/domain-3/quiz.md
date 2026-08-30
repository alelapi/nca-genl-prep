# Domain 3 Quiz — Experimentation

18 exam-style questions. Target: **15/18**.

---

**1.** BLEU and ROUGE differ primarily in that:

- A. BLEU is recall-oriented for summarization; ROUGE is precision-oriented for translation
- B. BLEU is precision-oriented for translation; ROUGE is recall-oriented for summarization
- C. Both measure semantic similarity via embeddings
- D. BLEU works only on single sentences; ROUGE only on documents

??? success "Answer"
    **B.** BLEU measures how much of the *candidate* appears in the reference (precision,
    plus a brevity penalty) and is the machine-translation standard. ROUGE measures how
    much of the *reference* is covered by the candidate (recall) and is the summarization
    standard. Semantic similarity (C) is BERTScore.

---

**2.** Perplexity of 10 on a test set means, roughly:

- A. The model is 10% accurate
- B. The model is about as uncertain as if choosing uniformly among 10 tokens
- C. The model made 10 errors
- D. The loss is 10

??? success "Answer"
    **B.** Perplexity is the exponentiated average cross-entropy — the effective number of
    equally likely choices per token. Lower is better, and it is only comparable across
    models sharing a tokenizer and test set.

---

**3.** A fraud-detection dataset is 99.5% legitimate transactions. Which metric is most
misleading?

- A. Recall
- B. Precision
- C. Accuracy
- D. PR-AUC

??? success "Answer"
    **C.** Predicting "legitimate" for everything scores 99.5% accuracy while catching zero
    fraud. On imbalanced data use precision, recall, F1 or PR-AUC.

---

**4.** In RLHF, human annotators primarily provide:

- A. Absolute quality scores from 1 to 10
- B. Rankings/comparisons between candidate responses
- C. Corrected model weights
- D. Token-level probability targets

??? success "Answer"
    **B.** Humans are far more reliable at pairwise comparison than absolute scoring, so
    preference data is collected as rankings, and the reward model learns from those.

---

**5.** What does DPO eliminate compared to classic RLHF?

- A. The need for any human preference data
- B. The separate reward model and the RL (PPO) optimization loop
- C. Supervised fine-tuning
- D. The KL divergence concept entirely

??? success "Answer"
    **B.** Direct Preference Optimization trains directly on preference pairs with a
    classification-style objective — no reward model, no PPO. It still needs preference
    data (A) and still follows SFT (C).

---

**6.** In a RAG system, the answer is wrong because the correct document was never
retrieved. Which metric would have surfaced this?

- A. Faithfulness
- B. Answer relevance
- C. Recall@k
- D. BLEU

??? success "Answer"
    **C.** Recall@k measures whether relevant documents made it into the top *k*. This is
    why retrieval must be evaluated separately — faithfulness (A) would look *fine*, since
    the answer may be perfectly consistent with the (wrong) retrieved context.

---

**7.** Which is NOT part of the standard "RAG triad"?

- A. Context relevance
- B. Faithfulness/groundedness
- C. Answer relevance
- D. Perplexity

??? success "Answer"
    **D.** Perplexity is an intrinsic language-modeling metric with no role in RAG
    evaluation.

---

**8.** A team checks their A/B test every hour and stops as soon as p < 0.05. What is
wrong?

- A. Nothing; that is standard practice
- B. Repeated peeking inflates the false-positive rate far above 5%
- C. p-values cannot be used in A/B tests
- D. They should use a smaller significance level for guardrail metrics only

??? success "Answer"
    **B.** Continuous monitoring with an early stop massively increases Type I error. Fix
    the sample size and duration in advance, or use a sequential/always-valid testing
    method designed for it.

---

**9.** Which bias causes an LLM judge to prefer whichever response is presented first?

- A. Verbosity bias
- B. Position bias
- C. Self-enhancement bias
- D. Majority-label bias

??? success "Answer"
    **B.** Position bias. Mitigate by evaluating both orderings and averaging. Verbosity
    bias favours longer answers; self-enhancement favours the judge's own family;
    majority-label bias is a few-shot prompting effect.

---

**10.** "Few-shot prompting" means:

- A. Fine-tuning on a few examples
- B. Including a few examples in the prompt, with no weight updates
- C. Training with a small learning rate
- D. Using a small model

??? success "Answer"
    **B.** In-context learning. The examples are input, not training data; no gradients are
    computed and no parameters change.

---

**11.** Which is the most effective mitigation for a model fabricating facts about a
company's internal policies?

- A. Fine-tune on the policy documents
- B. Increase the temperature to encourage exploration
- C. RAG over the policy documents with a grounding instruction and citations
- D. Increase the maximum output length

??? success "Answer"
    **C.** A knowledge gap calls for retrieval, not weight updates. Fine-tuning (A) injects
    facts diffusely, cannot be cited, goes stale, and can *increase* confident
    fabrication.

---

**12.** A model's summary states something the source document does not contain. This is a
failure of:

- A. Fluency
- B. Faithfulness
- C. Answer relevance
- D. Recall@k

??? success "Answer"
    **B.** Faithfulness (groundedness) measures whether every claim is supported by the
    provided context. Note it can fail even when the statement is factually true in the
    wider world.

---

**13.** Which statement about statistical significance is correct?

- A. A p-value of 0.03 means there is a 3% chance the null hypothesis is true
- B. Statistical significance guarantees the effect is large enough to matter
- C. A p-value is the probability of a result at least this extreme assuming the null is true
- D. p-values are unaffected by sample size

??? success "Answer"
    **C.** A is the classic misinterpretation. Significance says nothing about effect size
    (B), and with a large enough sample a trivial effect becomes significant (D is false) —
    which is why effect size and confidence intervals must be reported.

---

**14.** Which cross-validation variant is required for time-series data?

- A. Stratified k-fold
- B. Leave-one-out
- C. Time-series split — always train on the past, validate on the future
- D. Standard shuffled k-fold

??? success "Answer"
    **C.** Shuffling lets future information leak into training, producing optimistic and
    entirely invalid scores.

---

**15.** GLUE is best described as:

- A. A GPU communication library
- B. A benchmark suite of natural language understanding tasks
- C. A vector database
- D. A quantization technique

??? success "Answer"
    **B.** General Language Understanding Evaluation — nine sentence-level NLU tasks. It
    was largely saturated by modern models, prompting SuperGLUE. (A is NCCL.)

---

**16.** Which practice most improves the reliability of a few-shot evaluation?

- A. Using the same single example ordering every time
- B. Drawing few-shot examples from the evaluation set
- C. Varying example selection and order across runs and reporting mean ± std
- D. Raising the temperature to explore more outputs

??? success "Answer"
    **C.** Few-shot performance is highly sensitive to which examples are used and in what
    order. B is data leakage and inflates results.

---

**17.** Inter-annotator agreement measured with Cohen's κ comes out at 0.15. The correct
response is to:

- A. Train the model anyway; the noise will average out
- B. Revise the annotation guidelines — the task is underspecified
- C. Fire the annotators
- D. Switch to raw percentage agreement, which will look better

??? success "Answer"
    **B.** Low κ almost always means the task definition is ambiguous. A model trained on
    those labels cannot exceed the noise ceiling, and evaluation built on them is
    unreliable. (D would indeed look better — which is exactly why κ corrects for chance.)

---

**18.** A new retrieval pipeline improves recall@10 from 0.71 to 0.89 but raises p99
latency from 300 ms to 2.4 s. What should the team do?

- A. Ship it — quality is what matters
- B. Reject it outright
- C. Treat latency as a guardrail metric and decide against the product's requirements, e.g. by reranking fewer candidates
- D. Re-run the experiment until latency looks better

??? success "Answer"
    **C.** Guardrail metrics exist precisely for this. A quality gain that breaks the
    latency budget is not automatically a win or a loss — it is a trade-off to be resolved
    against the product's requirements, often by tuning the expensive stage.

---

## Scoring

| Score | Verdict |
| --- | --- |
| 16–18 | Strong. |
| 12–15 | Re-read Evaluation Metrics and Experiment Design. |
| < 12 | Rework the chapter — 22% of the exam rides on it. |
