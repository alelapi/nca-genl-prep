# Alignment, RLHF & Human Labeling

*The Experimentation domain description names this explicitly: "…the use of human subjects
in labeling or reinforcement learning from human feedback (RLHF)."*

## Why alignment is needed

A pretrained base model predicts the most **plausible continuation** of text. That is not
the same as being **helpful, harmless and honest**:

- Asked a question, a base model may continue with more questions — that is what documents
  full of questions look like.
- It will happily produce toxic, biased or dangerous content if the context suggests it.
- It has no notion that it should refuse anything.

**Alignment** is the process of shaping a model's behaviour toward human intent and values.
The common shorthand for the target is **HHH: helpful, harmless, honest**.

## The alignment pipeline

```text
  Base model (pretrained)
        │
        ▼
  1. SFT — supervised fine-tuning on (instruction, ideal response) demonstrations
        │      → the model learns to follow instructions
        ▼
  2. Reward modeling — humans rank candidate responses; a reward model learns to
        │               predict human preference
        ▼
  3. RL optimization (PPO) — the policy is optimized to maximise reward,
        │                    with a KL penalty anchoring it to the SFT model
        ▼
  Aligned / instruct model
```

### Step 1 — Supervised fine-tuning (SFT)

Human demonstrators write ideal responses to prompts; the model is fine-tuned on those
pairs. This alone converts a base model into something usable. Quality matters far more
than quantity.

### Step 2 — The reward model

- Show annotators a prompt with 2+ candidate responses; they **rank** them.
- Train a model (usually initialised from the LLM) to output a scalar score predicting
  which response humans would prefer.

!!! tip "Why ranking, not scoring"
    Humans are unreliable at assigning absolute scores ("is this a 6 or a 7?") and much more
    reliable at **pairwise comparison** ("is A better than B?"). Preference data is
    therefore collected as comparisons. This is a genuinely exam-worthy detail.

### Step 3 — Reinforcement learning (PPO)

- The LLM is the **policy**; the reward model provides the **reward**.
- **PPO** (Proximal Policy Optimization) is the standard algorithm.
- A **KL-divergence penalty** against the SFT reference model keeps the policy from
  drifting too far. Without it the policy finds degenerate outputs that score highly on the
  reward model but are nonsense — **reward hacking**.

## Alternatives to full RLHF

RLHF is complex, unstable and expensive (three models in play). Simpler methods have
largely displaced it in practice:

| Method | Idea |
| --- | --- |
| **DPO** (Direct Preference Optimization) | Optimize directly on preference pairs with a classification-style loss. **No separate reward model, no RL loop.** Much simpler and more stable; the common default |
| **RLAIF** | Replace human preference labels with AI-generated ones. Cheaper, scales; inherits the labeler model's biases |
| **Constitutional AI** | The model critiques and revises its own outputs against a written set of principles (a "constitution"), producing preference data with minimal human labeling |
| **KTO, ORPO, IPO** | Further preference-optimization variants, needing weaker supervision |
| **Rejection sampling / best-of-n** | Sample *n* responses, keep the highest-reward one, fine-tune on those |

!!! note "The one-line contrast"
    **RLHF = reward model + RL (PPO). DPO = skip both, optimize preferences directly.**

## Alignment failure modes

- **Reward hacking** — the policy maximises the proxy reward while violating the intent.
- **Sycophancy** — the model agrees with the user because agreement was rewarded, even
  when the user is wrong.
- **Over-refusal** — safety training generalises too broadly and harmless requests get
  declined. A real cost of alignment, and a live trade-off.
- **Alignment tax** — some raw capability is lost in exchange for safety and helpfulness.
- **Jailbreaking** — adversarial prompts recover the unaligned behaviour. See
  [Guardrails & LLM Security](../domain-5/guardrails-security.md).
- **Verbosity/length bias** — human raters prefer longer answers, so models learn to pad.

## Human annotation quality

The blueprint's phrase "the use of human subjects in labeling" makes this examinable in
its own right, and the job description lists *"define, curate, label, and annotate LLM
datasets."*

**Running a labeling effort:**

1. **Write a precise guideline** with definitions, decision rules and worked edge cases.
   Most disagreement is specification failure, not annotator failure.
2. **Pilot** on a small batch, review disagreements, revise the guideline. Iterate.
3. **Use multiple annotators per item** and adjudicate conflicts.
4. **Measure agreement** — see below.
5. **Insert gold-standard items** as a continuous quality check.
6. **Watch for annotator drift and fatigue**; rotate and re-calibrate.

**Inter-annotator agreement (IAA)**

| Measure | Use |
| --- | --- |
| **Raw agreement** | % of items where annotators match — inflated when one class dominates |
| **Cohen's κ** | Two annotators, corrected for chance agreement |
| **Fleiss' κ** | Three or more annotators |
| **Krippendorff's α** | Any number of annotators, handles missing data and multiple scales |

Rough interpretation of κ: <0.2 poor · 0.2–0.4 fair · 0.4–0.6 moderate · 0.6–0.8
substantial · >0.8 almost perfect.

!!! danger "Low IAA is a specification bug"
    If annotators do not agree, a model trained on their labels cannot exceed that noise
    ceiling, and your evaluation numbers are unreliable. Fix the guideline before
    collecting more data.

**Ethics of human labeling** — a real consideration NVIDIA's Trustworthy AI framing cares
about: fair pay, informed consent, psychological support for annotators exposed to harmful
content, transparency about how the data will be used, and privacy protection for any
personal data being labeled. See [Privacy & Consent](../domain-5/privacy-consent.md).

## Key takeaways

- Alignment turns a plausible-continuation engine into a helpful, harmless, honest
  assistant.
- **RLHF = SFT → reward model (from human *rankings*) → PPO with a KL penalty.**
- **DPO** achieves the same goal without a reward model or an RL loop — simpler and more
  stable.
- **RLAIF** and **Constitutional AI** substitute AI feedback for human labels.
- Failure modes: reward hacking, sycophancy, over-refusal, alignment tax.
- Measure **inter-annotator agreement** (Cohen's/Fleiss' κ); low agreement means the
  guideline, not the annotators, is broken.
