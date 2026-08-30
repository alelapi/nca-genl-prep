# Experiment Design & A/B Testing

*NVIDIA's suggested readings name "How to Conduct A/B Testing in Machine Learning" and
"Cross-Validation in Machine Learning".*

## The scientific loop

1. **Question** — "Does adding a reranker improve answer quality?"
2. **Hypothesis** — a falsifiable statement with a direction and a metric:
   *"Adding a cross-encoder reranker increases answer faithfulness by ≥5 points."*
3. **Variables** — independent (what you change: the reranker), dependent (what you
   measure: faithfulness), controlled (everything else: model, prompt, index, data).
4. **Design** — control vs. treatment, randomisation, sample size.
5. **Run** — change **one variable at a time**.
6. **Analyse** — effect size **and** statistical significance.
7. **Conclude, and repeat.**

!!! danger "The most common experimental error in LLM work"
    Changing the prompt, the model and the retrieval settings at once, observing an
    improvement, and being unable to say what caused it — or which change to keep when the
    result later regresses. **One variable at a time**, or use a factorial design that can
    separate the effects.

## Offline vs. online evaluation

| | **Offline** | **Online** |
| --- | --- | --- |
| Data | A fixed, held-out evaluation set | Live user traffic |
| Speed | Seconds to minutes | Days to weeks |
| Cost | Cheap | Expensive, and users are exposed to the change |
| Measures | Proxy metrics (accuracy, F1, faithfulness) | Real outcomes (conversion, task completion, satisfaction) |
| Use for | Rapid iteration, regression gating in CI | Final validation before full rollout |

The standard workflow: iterate offline against a fixed eval set, then confirm the
finalists online with an A/B test.

## A/B testing

A **randomised controlled experiment**: split users into a **control** group (A, current
system) and a **treatment** group (B, the change), and compare a predefined metric.

**Requirements for a valid test:**

- **Randomised assignment** — otherwise selection bias contaminates the comparison.
- **Consistent bucketing** — hash the user id so the same user always sees the same
  variant. Flipping variants mid-session destroys the experiment and the user experience.
- **One primary metric**, defined *before* the test starts.
- **Guardrail metrics** — latency, cost, error rate, safety violations. A quality win that
  triples latency is not a win.
- **A predetermined sample size and duration** — run for whole weeks to absorb day-of-week
  effects.
- **No peeking.** Repeatedly checking and stopping the moment *p* < 0.05 inflates the
  false-positive rate dramatically. Fix the sample size in advance, or use a sequential
  testing method designed for continuous monitoring.

**Statistical concepts to recognise:**

| Term | Meaning |
| --- | --- |
| **Null hypothesis (H₀)** | There is no difference between A and B |
| **p-value** | Probability of observing a result at least this extreme *if H₀ were true*. **Not** the probability that H₀ is true |
| **Significance level (α)** | Threshold for rejecting H₀, conventionally 0.05 |
| **Type I error** | False positive — shipping a change that does nothing |
| **Type II error** | False negative — discarding a change that actually works |
| **Statistical power (1−β)** | Probability of detecting a real effect; target 80% |
| **Confidence interval** | A range of plausible values for the true effect — more informative than a p-value alone |
| **Effect size** | *How big* the difference is. Statistical significance without practical significance is noise with good manners |
| **Multiple comparisons** | Testing many metrics inflates false positives; correct for it (Bonferroni, FDR) |

!!! tip "Sample size intuition"
    Required sample size grows as the **square** of the inverse effect size: detecting a
    small effect needs far more traffic than detecting a large one. If you cannot get
    enough traffic, you cannot detect a 0.5% improvement — no amount of analysis fixes that.

**Variants of online testing:**

- **A/B/n** — several treatments at once. Correct for multiple comparisons.
- **Multi-armed bandit** — dynamically shift traffic toward the better-performing arm.
  Maximises reward during the experiment; weaker for clean causal inference.
- **Interleaving** — for ranking systems, blend results from both systems in one list and
  observe which side gets the clicks. Very sensitive, needs far less traffic.
- **Shadow testing** — run the new system on mirrored traffic without showing its output.
  Zero user risk; cannot measure user reaction.

## Evaluating LLMs specifically

**Build a fixed evaluation set** of 50–500 representative inputs with expected outputs or
grading criteria. This is the highest-value artefact in an LLM project. It must be:

- **Representative** of real traffic, including the long tail.
- **Stable** — pinned and version-controlled, so results are comparable over time.
- **Held out** — never used to tune prompts you then evaluate on. Keep a separate dev set
  for iteration.
- **Growing** — every production failure becomes a new test case.

**Control the randomness.** LLM outputs vary run to run. Pin `temperature=0` or a seed for
comparisons, pin the **model version**, and run *n* repetitions to report mean ± variance
for anything sampled.

**Watch for contamination.** If your benchmark appeared in the model's training data, its
score is meaningless. This is why public benchmark scores inflate over time and why a
private eval set matters.

## Cross-validation in this context

Covered in detail in [ML Fundamentals](../domain-1/ml-fundamentals.md#cross-validation).
For the Experimentation domain, the points that matter:

- k-fold CV gives a **lower-variance** performance estimate plus a **standard deviation**,
  which is what lets you claim one model beats another rather than one split being lucky.
- Use **stratified** k-fold for classification, **group** k-fold for correlated rows, and
  a **time-series split** for temporal data (never shuffle the future into the past).
- CV is a classical-ML technique. Foundation models use a single fixed held-out set,
  because training *k* of them is infeasible.
- Fit every transform **inside** the fold to prevent leakage — the reason scikit-learn
  `Pipeline` exists.

## Reproducibility

An experiment nobody can re-run is an anecdote. Pin and record:

- Random seeds, library and framework versions, container image
- Model name **and revision**, decoding parameters
- The exact prompt (versioned like code) and the exact eval set version
- Hardware, since GPU non-determinism can shift results slightly

Track it all with an experiment-tracking tool (MLflow, Weights & Biases) rather than a
spreadsheet.

## Key takeaways

- Hypothesis → controlled change → measurement → conclusion. **One variable at a time.**
- Offline evaluation for iteration speed; **online A/B testing** for real outcomes.
- Valid A/B tests need randomisation, consistent bucketing, a pre-declared primary
  metric, guardrail metrics, a fixed sample size, and **no peeking**.
- A p-value is not the probability the hypothesis is true; always report **effect size**
  and confidence intervals.
- A fixed, representative, contamination-free **eval set** is the single most valuable
  asset in an LLM project.
- Pin seeds, model versions and prompts, or your results are not reproducible.
