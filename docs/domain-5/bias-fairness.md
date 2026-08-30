# Bias & Fairness

*Covers task 5.4: "describe how to minimize bias in AI systems."*

Note the verb: **minimize**, not eliminate. That word choice is deliberate and it is the honest
position — section 6 explains why.

---

## 1. Two different things called "bias"

Do not confuse them; the exam may test the distinction.

**Statistical bias** — the systematic error term in the bias–variance trade-off. A model with
high bias is too simple to capture the pattern and underfits. This is a *modeling* concept, and
it is covered in [ML Fundamentals](../domain-1/ml-fundamentals.md).

**Societal / unwanted bias** — systematically different, unjustified treatment of groups of
people. This is an *ethics* concept, and it is what this page is about.

They are unrelated. A model can have low statistical bias and severe societal bias.

---

## 2. Where bias enters

Bias is not injected at one point that you can guard. It **accumulates along the whole
lifecycle**, and each stage has its own mechanism.

```text
  PROBLEM        DATA           LABELING      MODELING     EVALUATION    DEPLOYMENT
  FRAMING     COLLECTION                                                      │
     │             │                │             │            │              │
  objective    selection        annotator     aggregation   benchmark     deployment
  bias         historical       bias          bias          bias          bias
               representation                                                 │
     ▲                                                                        │
     └────────────────────── FEEDBACK LOOP ◄───────────────────────────────────┘
                    the model's outputs become tomorrow's training data
```

| Stage | Bias type | Example |
| --- | --- | --- |
| **Problem framing** | Objective bias | Optimising "engagement" ends up promoting outrage, because outrage is engaging |
| **Data collection** | **Selection / sampling** | Training data drawn only from English-language web text |
| | **Historical** | Past hiring data reflects past discrimination |
| | **Representation** | Some groups under-represented, so the model performs worse for them |
| **Labeling** | Annotator bias | Labelers' own assumptions encoded as ground truth |
| **Modeling** | Aggregation bias | One model for all groups when the groups genuinely differ |
| **Evaluation** | Benchmark bias | The test set shares the training set's blind spots; only aggregate accuracy reported |
| **Deployment** | Deployment bias | Used in a context it was never designed or validated for |
| **Feedback** | Feedback loop | The model's outputs become tomorrow's training data, amplifying the original skew |

### The two that matter most

!!! danger "Historical bias — the insight that reframes the whole topic"
    Historical bias is the case where **the data is perfectly accurate and the outcome is still
    unjust.**

    A hiring model trained on ten years of the company's decisions learns to replicate those
    decisions. If the company historically under-hired a group, the model learns that pattern —
    not because the data is wrong, but because the data is *right about a wrong world*.

    Three things follow, and they are what make this the exam's favourite scenario:

    1. **"The model just learned what was in the data" is a description of the problem, not a
       defence.**
    2. The model applies that pattern **at scale**, to every applicant, consistently — a human
       recruiter's bias affects dozens of decisions, a model's affects millions.
    3. It does so with an **appearance of objectivity**. "The algorithm decided" carries a
       credibility that "the manager decided" does not, which makes it harder to challenge.

**Feedback loops** are the second amplifier. A model that under-recommends a group of merchants
generates less data about them, which makes the next model even less confident about them, which
reduces recommendations further. The bias compounds without anyone doing anything new.

---

## 3. Bias in LLMs specifically

- **Stereotype association** — occupations, adjectives and roles statistically bound to gender,
  ethnicity or nationality. This falls directly out of how
  [embeddings](../domain-1/embeddings.md) work: if the corpus associates "nurse" with female
  pronouns, that association becomes a **direction in the embedding space**, exactly like the
  legitimate `king − man + woman ≈ queen` relationship. It is the same mechanism, applied to
  something we did not want learned.
- **Language and dialect disparity** — markedly better performance in English and in standard
  dialects. A model that performs worse on African-American English or on Swahili is delivering
  a worse product to those users.
- **Cultural skew** — Western, English-speaking, internet-user norms treated as the default.
- **Toxicity** absorbed from web-scale corpora.
- **Representational harms** — erasure, stereotyping, or demeaning depiction of groups.
- **Allocative harms** — unequal access to resources or opportunities when the model gates a
  decision (credit, hiring, housing).
- **Sycophancy** — reinforcing whatever view the user expresses, which amplifies existing bias
  rather than introducing new bias. See [Alignment & RLHF](../domain-3/rlhf-alignment.md).

The **representational vs. allocative** distinction is worth carrying: the first is about how
people are *depicted*, the second about what they *get*. Both are real harms; they need different
mitigations and different stakeholders.

---

## 4. Fairness definitions — and why you must choose

There are several formal definitions of fairness, and **they are mathematically incompatible**.

| Definition | Requires |
| --- | --- |
| **Demographic parity** | Positive outcome rates are equal across groups |
| **Equal opportunity** | **True positive rates** are equal across groups |
| **Equalized odds** | Both TPR **and** FPR are equal across groups |
| **Predictive parity** | **Precision** is equal across groups |
| **Individual fairness** | Similar individuals receive similar outcomes |
| **Counterfactual fairness** | The outcome is unchanged if only the protected attribute changes |

!!! important "The impossibility result"
    When the **base rates genuinely differ** between groups — that is, the actual prevalence of
    the outcome differs — you **cannot simultaneously satisfy equalized odds and predictive
    parity**. This is a mathematical theorem, not an engineering limitation. No better algorithm
    will resolve it.

    The consequence is uncomfortable and important: **fairness is not a property you can optimise
    for in general.** You must choose *which* fairness criterion matters in your specific context,
    justify the choice, and document it.

    The choice is a **value judgement about which harm is worse** — is it worse to miss qualified
    candidates from a group (equal opportunity), or worse for a positive prediction to mean
    different things for different groups (predictive parity)? That question has no technical
    answer, and pretending it does is itself a failure.

---

## 5. Mitigation across the lifecycle

Because bias enters everywhere, mitigation has to happen everywhere.

### Pre-processing — fix the data

- **Audit representation.** Count the groups. Measure the gaps **before** training. You cannot fix
  what you have not measured.
- **Diversify collection.** Actively sample under-represented groups rather than accepting
  whatever the convenient source produced.
- **Rebalance or reweight** the training data.
- **Improve annotation** — diverse annotators, explicit guidelines, measured
  [inter-annotator agreement](../domain-3/rlhf-alignment.md#inter-annotator-agreement).
- **Document** with a dataset datasheet.

### In-processing — fix the training

- **Fairness constraints or regularisers** added to the objective.
- **Adversarial debiasing** — train an adversary that tries to predict the protected attribute
  from the model's internal representation, and train the model to defeat it. If the adversary
  cannot recover the attribute, the representation does not encode it.
- **Alignment** — RLHF/DPO with preference data that penalises biased output.
- **Balanced fine-tuning** on counter-stereotypical examples.

### Post-processing — fix the outputs

- **Group-specific thresholds** to equalise the chosen fairness metric.
- **Output filtering and guardrails** — see [Guardrails & Security](guardrails-security.md).
- **Per-group calibration.**

### Evaluation and operations

- **Disaggregated evaluation** — the single most important practice. See below.
- **Bias benchmarks** — StereoSet, CrowS-Pairs, BOLD, WinoBias, BBQ, HolisticBias.
- **Red-teaming** with adversarial prompts targeting stereotypes.
- **Continuous monitoring** in production, plus a user feedback and **appeal** channel.
- **Diverse teams.** Homogeneous teams do not notice their own blind spots — not through
  ill-will, but because a blind spot is by definition something you do not see.

### The two things that get asked

!!! important "Disaggregated evaluation is the highest-value practice"
    Report metrics **broken down by subgroup**, not only in aggregate.

    ```text
    Aggregate accuracy:  94%           ← looks fine, ship it

    Group A:  96%   (n = 8,400)
    Group B:  95%   (n = 1,100)
    Group C:  71%   (n =   500)        ← invisible in the aggregate
    ```

    Group C is 5% of the data, so its failure moves the aggregate by about one point. Aggregate
    metrics **structurally cannot** reveal this. This is why model cards require subgroup
    breakdowns, and it is why "report aggregate accuracy" is always the wrong answer to "how would
    you detect that a model underperforms for a demographic?"

!!! danger "Removing the protected attribute does not remove the bias"
    Deleting the `gender` column does not make a model gender-blind. **Proxy variables**
    reconstruct it: first name, postcode, school attended, purchase history, writing style, even
    which browser someone uses.

    This approach is called **fairness through unawareness**, and it fails for a second reason
    beyond ineffectiveness: **it makes bias unmeasurable.** You need the protected attribute *for
    evaluation* in order to compute per-group metrics at all. Deleting it does not remove the
    problem; it removes your ability to see the problem.

    This is the most common naive-answer distractor on fairness questions.

---

## 6. What "minimize" means

The blueprint says **minimize**, not eliminate, and that is the honest word.

Bias cannot be fully removed. The data comes from an unequal world; the fairness definitions
conflict with each other; and any choice among them advantages someone.

The professional standard is therefore not "our model is unbiased" — a claim nobody can support.
It is:

1. **Measure** it, per subgroup.
2. **Choose explicitly** which fairness criterion applies in this context.
3. **Document** the choice and the trade-off.
4. **Mitigate** at every stage.
5. **Monitor** continuously, and provide an appeal route.

That is a defensible position. "We removed the gender column" is not.

---

## 7. Recap

- **Statistical bias** (underfitting) and **societal bias** (unjust treatment) are unrelated
  concepts that share a word.
- Bias enters at **every stage**: framing, collection, labeling, modeling, evaluation,
  deployment, and feedback loops that amplify it over time.
- **Historical bias**: accurate data can encode injustice, and models apply it at scale with an
  appearance of objectivity.
- LLM bias arises mechanically — stereotypes become **directions in embedding space**, the same
  way legitimate analogies do.
- Fairness definitions (demographic parity, equal opportunity, equalized odds, predictive parity)
  are **mathematically incompatible** when base rates differ. Choosing one is a **value
  judgement**.
- **Disaggregated evaluation** — metrics per subgroup — is the highest-value practice, because
  aggregates structurally hide subgroup failures.
- **Removing the protected attribute does not work**: proxies reconstruct it, and you lose the
  ability to measure fairness at all.
- Mitigate pre-processing (data), in-processing (training) and post-processing (outputs), then
  monitor.
