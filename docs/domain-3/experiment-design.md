# Experiment Design & A/B Testing

*NVIDIA's suggested readings name "How to Conduct A/B Testing in Machine Learning" and
"Cross-Validation in Machine Learning". The study guide's job description lists
"perform experimentation like A/B testing, evaluating prompts, evaluating models, and producing
POCs" as a core responsibility.*

---

## 1. Why LLM work needs experimental discipline more than most software

In ordinary software, you change the code and the behaviour changes deterministically. You can
reason about whether a change is an improvement.

In LLM work, none of that holds. The output is **probabilistic**, so the same input can produce
different results. The quality signal is **subjective and noisy**. The changes you make —
prompts, retrieval settings, models — interact in ways that are not predictable from first
principles. And the effects you are chasing are often small.

Under those conditions, intuition is worthless. Two engineers can look at ten outputs from two
prompts and reach opposite conclusions, both sincerely. The only way out is measurement, and
measurement done carelessly is worse than none, because it produces confident wrong answers.

That is why an entire exam domain is devoted to this.

---

## 2. The scientific loop

1. **Question** — "Does adding a reranker improve answer quality?"
2. **Hypothesis** — falsifiable, with a direction and a metric: *"Adding a cross-encoder reranker
   increases answer faithfulness by at least 5 points."*
3. **Variables**:
   - **Independent** — what you change (the reranker)
   - **Dependent** — what you measure (faithfulness)
   - **Controlled** — everything else held fixed (model, prompt, index, data, temperature, seed)
4. **Design** — control vs. treatment, randomisation, sample size.
5. **Run** — changing **one variable at a time**.
6. **Analyse** — effect size **and** statistical significance.
7. **Conclude, and iterate.**

!!! danger "The most common experimental error in LLM work"
    Changing the prompt, the model **and** the retrieval settings at once, observing that things
    got better, and shipping it.

    You have learned nothing. You do not know which change helped, which was neutral, and which
    actively hurt but was masked by the others. When it regresses next month you cannot revert
    the right thing. And you have no idea whether the fourth change you are about to make will
    interact with any of them.

    **One variable at a time**, or use a factorial design that can separate the effects
    statistically.

---

## 3. Offline vs. online evaluation

| | **Offline** | **Online** |
| --- | --- | --- |
| Data | A fixed, held-out evaluation set | Live user traffic |
| Speed | Seconds to minutes | Days to weeks |
| Cost | Cheap | Expensive; real users are exposed |
| Measures | Proxy metrics — accuracy, F1, faithfulness | Real outcomes — conversion, task completion, satisfaction |
| Reproducible? | Yes | No; conditions change |
| Use for | Rapid iteration, CI regression gating | Final validation before rollout |

The standard workflow: **iterate offline, confirm online.** Offline lets you try fifty prompt
variants in an afternoon; online tells you whether the winner actually helps users.

!!! warning "Proxy metrics can improve while the product gets worse"
    Offline metrics are proxies, and proxies can be gamed — usually unintentionally. Faithfulness
    scores rise because the model started hedging everything and hedged answers are trivially
    supported by the context. Meanwhile users find it evasive and stop using the feature.

    Good offline numbers that do not move product metrics are a **warning sign**, not a success.

---

## 4. A/B testing

A **randomised controlled experiment**: split users into a **control** group (A, current system)
and a **treatment** group (B, the change), then compare a predefined metric.

Randomisation is the crucial ingredient, and it is what makes A/B testing the only method here
that establishes **causation** rather than correlation. Because assignment is random, the two
groups differ only by chance in every respect except the treatment — so any systematic
difference in outcome must be caused by the treatment.

### Requirements for a valid test

**Randomised assignment.** Not "power users get the new version". Any non-random split
reintroduces confounding, and the whole causal argument collapses.

**Consistent bucketing.** Hash the user id so the same user always sees the same variant.
Flipping variants mid-session destroys both the experiment and the user experience.

**One primary metric, declared before the test starts.** Otherwise you will run the test, look
at twenty metrics, find one that moved, and report that — which is p-hacking whether or not you
intended it.

**Guardrail metrics.** Latency, cost, error rate, safety violations. A quality win that triples
latency or doubles cost is not automatically a win; it is a trade-off requiring a decision.

**A predetermined sample size and duration.** Compute it in advance from your baseline rate, the
effect size you care about detecting, and your desired power. Run for **whole weeks** to absorb
day-of-week effects — Monday traffic does not look like Saturday traffic.

**No peeking.** See below.

### The peeking problem

!!! danger "Why checking your test hourly invalidates it"
    The 5% significance threshold means: *if there were no real effect, there is a 5% chance of
    seeing a result this extreme.* That guarantee holds for **one** test, at **one**
    predetermined moment.

    If you check continuously and stop the moment `p < 0.05`, you are effectively running dozens
    of tests. Random fluctuation will cross the threshold eventually **even when there is no
    effect at all** — and you will stop precisely then, because that is your stopping rule. The
    real false-positive rate rises to 20–30% or more.

    **The fix:** fix the sample size and duration in advance and analyse once. Or use a
    sequential testing method (always-valid p-values, group sequential designs) explicitly
    designed to permit continuous monitoring.

### The statistics you need to recognise

| Term | Meaning |
| --- | --- |
| **Null hypothesis (H₀)** | There is no difference between A and B |
| **p-value** | The probability of observing a result at least this extreme **if H₀ were true** |
| **Significance level (α)** | The threshold for rejecting H₀, conventionally 0.05 |
| **Type I error** | **False positive** — shipping a change that does nothing |
| **Type II error** | **False negative** — discarding a change that actually works |
| **Statistical power (1−β)** | The probability of detecting a real effect of a given size. Target 80% |
| **Confidence interval** | A range of plausible values for the true effect — more informative than a p-value alone |
| **Effect size** | *How big* the difference is |
| **Multiple comparisons** | Testing many metrics inflates false positives; correct with Bonferroni or FDR |

!!! warning "The p-value misinterpretation"
    A p-value of 0.03 does **not** mean "there is a 3% chance the null hypothesis is true". It
    means "**if** the null were true, there would be a 3% chance of seeing data this extreme."

    That is a statement about the data given a hypothesis, not about the hypothesis given the
    data. The reversal is the single most common statistical error in industry, and it appears as
    a distractor.

**Significance is not importance.** With a large enough sample, a 0.01% improvement becomes
statistically significant. Significance tells you the effect is probably real; **effect size**
tells you whether you should care. Always report both, and prefer confidence intervals — an
interval of [−0.5%, +4.2%] tells you immediately that the effect is not distinguishable from
zero, which a bare p-value does not.

**Sample size intuition:** the required sample grows roughly as the **square** of the inverse
effect size. Detecting a 1% improvement needs about 100× the traffic of detecting a 10% one. If
you do not have the traffic, you cannot detect small effects — and no amount of clever analysis
changes that. Recognising *"we do not have enough data to answer this question"* is a legitimate
and valuable conclusion.

### Variants of online testing

- **A/B/n** — several treatments at once. Correct for multiple comparisons.
- **Multi-armed bandit** — dynamically shift traffic toward whichever arm is performing better.
  Maximises reward *during* the experiment; weaker for clean causal inference afterwards. Good
  when the cost of showing the worse variant is high.
- **Interleaving** — for **ranking** systems, blend results from both systems into one list and
  observe which side gets the clicks. Every user is their own control, which makes it far more
  sensitive and dramatically reduces the traffic needed.
- **Shadow testing** — run the new system on mirrored traffic without showing its output. Zero
  user risk; cannot measure user reaction. See [Deployment](../domain-2/deployment.md).

---

## 5. Evaluating LLMs specifically

### Build a fixed evaluation set — this is the artefact that matters

50–500 representative inputs with expected outputs or grading criteria. It is the highest-value
thing you will build in an LLM project, and the thing most teams skip.

It must be:

- **Representative** of real traffic, including the long tail and the awkward cases. Invented
  examples are always too clean — real users write ungrammatical, ambiguous, half-finished
  queries.
- **Stable and versioned**, so that a score from March is comparable to a score from June.
- **Held out.** Never tune prompts against the same set you report on. Keep a separate dev set
  for iteration — the [three-way split](../domain-1/ml-fundamentals.md#data-splits-and-the-cardinal-sin-of-leakage)
  from classical ML applies unchanged.
- **Growing.** Every production failure becomes a permanent test case. This is how the eval set
  gets good over time, and it is why it is worth starting one before it feels necessary.

### Control the randomness

LLM outputs vary between runs. For a comparison to mean anything:

- Pin **temperature 0** (or a seed) when comparing systems.
- Pin the **model version** — hosted models change underneath you.
- For anything sampled, run **n repetitions** and report **mean ± variance**, not a single run.

Without this you may be measuring the difference between two samples from the same distribution
and calling it an improvement.

### Watch for contamination

If your benchmark appeared in the model's training data, its score is meaningless. This is why
public benchmark scores inflate over time and why a **private** eval set built from your own
traffic is the one that decides. See [Evaluation Metrics](evaluation-metrics.md#benchmarks).

---

## 6. Cross-validation in this context

Covered fully in [ML Fundamentals](../domain-1/ml-fundamentals.md#cross-validation). What matters
for the Experimentation domain:

- k-fold CV gives a **lower-variance estimate** **and a standard deviation** — and the second is
  what lets you claim one model beats another rather than one split being lucky.
- Use **stratified** k-fold for classification, **group** k-fold for correlated rows, and a
  **time-series split** for temporal data (never shuffle — that leaks the future into the past).
- Fit every transform **inside** the fold to prevent leakage. This is what scikit-learn's
  `Pipeline` exists for.
- CV is a **classical ML** technique. LLM pretraining uses a single fixed held-out set, because
  training *k* foundation models is not feasible.

---

## 7. Reproducibility

An experiment nobody can re-run is an anecdote. Pin and record:

- **Random seeds**, library and framework versions, container image
- **Model name and revision**, plus all decoding parameters
- The **exact prompt** (versioned like code) and the **eval set version**
- The **index version** for RAG systems
- Hardware, since GPU non-determinism can shift results slightly

Track this with an experiment-tracking tool — **MLflow**, **Weights & Biases** — rather than a
spreadsheet. The value is not tidiness: it is that in three months, when a result looks wrong,
you can reconstruct exactly what produced it.

---

## 8. Recap

- LLM work needs experimental discipline **because** output is probabilistic, quality is
  subjective, and effects are small — intuition genuinely does not work here.
- Hypothesis → controlled change → measurement → conclusion. **One variable at a time.**
- **Offline** evaluation for iteration speed; **online A/B testing** for real outcomes. Good
  offline numbers that do not move product metrics are a warning sign.
- Valid A/B tests need: randomisation, consistent bucketing, a pre-declared primary metric,
  guardrail metrics, a fixed sample size, and **no peeking** — continuous checking inflates false
  positives to 20–30%.
- A **p-value is not** the probability the hypothesis is true. Report **effect size** and
  **confidence intervals**.
- Sample size scales with the **square** of the inverse effect size; small effects need enormous
  traffic.
- **Interleaving** compares ranking systems with far less traffic; **bandits** optimise during
  the test; **shadow** carries zero user risk.
- A fixed, representative, versioned, contamination-free **eval set** is the most valuable
  artefact in an LLM project.
- Pin seeds, model versions, prompts and index versions, or your results are not reproducible.
