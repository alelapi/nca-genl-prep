# Bias & Fairness

*Covers task 5.4: "describe how to minimize bias in AI systems."*

## Two different meanings of "bias"

Do not confuse them — the exam may test the distinction:

- **Statistical bias** — the systematic error term in the bias–variance trade-off; a model
  that is too simple underfits. A *modeling* concept.
- **Societal/unwanted bias** — systematically different, unjustified treatment of groups of
  people. An *ethics* concept, and what this domain is about.

## Where bias enters

Bias is not injected at one point; it accumulates along the whole lifecycle.

| Stage | Bias type | Example |
| --- | --- | --- |
| **Problem framing** | Objective bias | Optimising "engagement" ends up promoting outrage |
| **Data collection** | **Selection / sampling bias** | Training data drawn only from English web text |
| | **Historical bias** | Past hiring data reflects past discrimination — the data is accurate *and* unjust |
| | **Representation bias** | Some groups are under-represented, so the model performs worse for them |
| **Labeling** | **Annotator bias** | Labelers' own assumptions encoded as ground truth |
| **Modeling** | **Aggregation bias** | One model for all groups when the groups genuinely differ |
| **Evaluation** | **Benchmark bias** | Test set shares the training set's blind spots; only aggregate accuracy reported |
| **Deployment** | **Deployment bias** | Used in a context it was never designed for |
| **Feedback** | **Feedback loop** | The model's outputs become tomorrow's training data, amplifying the original skew |

!!! danger "The most important single insight"
    **Historical bias means the data can be perfectly accurate and still unjust.** A model
    faithfully learning from biased human decisions will reproduce and often *amplify* them
    — at scale, with a veneer of objectivity. "The model just learned what was in the data"
    is a description of the problem, not a defence.

## Bias in LLMs specifically

- **Stereotype association** — occupations, adjectives and roles associated with gender,
  ethnicity or nationality.
- **Language and dialect disparity** — markedly better performance in English and in
  standard dialects.
- **Cultural skew** — Western, English-speaking, internet-user norms treated as default.
- **Toxicity and demeaning content** absorbed from web-scale corpora.
- **Representational harms** — erasure or misrepresentation of groups.
- **Allocative harms** — unequal access to resources or opportunities when the model gates
  a decision.
- **Sycophancy** — reinforcing whatever view the user expresses.

## Fairness definitions

There are several, and they are **mathematically incompatible** — you cannot satisfy them
all simultaneously except in degenerate cases. Choosing among them is a value judgement,
not a technical one.

| Definition | Requires |
| --- | --- |
| **Demographic parity** | Positive outcome rates are equal across groups |
| **Equal opportunity** | **True positive rates** are equal across groups |
| **Equalized odds** | Both TPR and FPR are equal across groups |
| **Predictive parity** | Precision is equal across groups |
| **Individual fairness** | Similar individuals receive similar outcomes |
| **Counterfactual fairness** | The outcome is unchanged if only the protected attribute changes |

!!! tip "Exam-ready point"
    The **impossibility result**: when base rates genuinely differ between groups, you
    cannot simultaneously achieve equalized odds and predictive parity. Fairness therefore
    requires an explicit, documented choice about which harm matters most in your context.

## Mitigation across the lifecycle

**Pre-processing — fix the data**

- **Audit representation** — count the groups; measure the gaps before training.
- **Diversify collection** — actively sample under-represented groups.
- **Rebalance / reweight** the training data.
- **Improve annotation** — diverse annotators, explicit guidelines, measured
  [agreement](../domain-3/rlhf-alignment.md#human-annotation-quality).
- **Document** with a dataset datasheet.

**In-processing — fix the training**

- **Fairness constraints or regularisers** in the objective.
- **Adversarial debiasing** — train an adversary that tries to predict the protected
  attribute from the model's representation, and train the model to defeat it.
- **Alignment** — RLHF/DPO with preference data that penalises biased output.
- **Balanced fine-tuning** on counter-stereotypical examples.

**Post-processing — fix the outputs**

- **Group-specific thresholds** to equalise the chosen fairness metric.
- **Output filtering and guardrails** — see [Guardrails & Security](guardrails-security.md).
- **Calibration** per group.

**Evaluation and operations**

- **Disaggregated evaluation** — the single most important practice. Report metrics
  **broken down by subgroup**, not just in aggregate.
- **Bias benchmarks** — StereoSet, CrowS-Pairs, BOLD, WinoBias, BBQ, HolisticBias.
- **Red-teaming** with adversarial prompts targeting stereotypes.
- **Continuous monitoring** in production, plus a user feedback and appeal channel.
- **Diverse teams** — homogeneous teams do not notice their own blind spots.

!!! warning "Removing the protected attribute does not remove the bias"
    Deleting "gender" from the features does not make a model gender-blind. **Proxy
    variables** — name, postcode, school, purchase history, writing style — reconstruct it.
    This is the most common naive-answer distractor on fairness questions.

## What "minimize" means

Note the blueprint's word choice: **minimize**, not eliminate. Bias cannot be fully removed
— data comes from an unequal world, and fairness definitions conflict. The professional
standard is to **measure it, choose explicitly which fairness criterion applies, document
the trade-off, mitigate, and monitor continuously.**

## Key takeaways

- Bias enters at every stage: framing, collection, labeling, modeling, evaluation,
  deployment, feedback loops.
- **Historical bias**: accurate data can encode injustice; models amplify it at scale.
- Fairness definitions (demographic parity, equal opportunity, equalized odds, predictive
  parity) are **mutually incompatible** — choosing one is a value judgement.
- **Removing the protected attribute does not work** — proxies reconstruct it.
- **Disaggregated evaluation** — measure per subgroup — is the highest-value practice.
- Mitigate at all three stages: pre-processing (data), in-processing (training),
  post-processing (outputs), then monitor.
