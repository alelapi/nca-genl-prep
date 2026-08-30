# Alignment, RLHF & Human Labeling

*The Experimentation domain description names this explicitly: "…the use of human subjects in
labeling or reinforcement learning from human feedback (RLHF)."*

---

## 1. Why a pretrained model is not yet useful

A base model has been trained to do exactly one thing: **predict the most plausible next token**.
It is extraordinarily good at that, and that is not the same as being a useful assistant.

Ask a base model a question and you may get:

```text
Prompt:  "What is the capital of France?"

Base model:  "What is the capital of Germany? What is the capital of Italy?
              What is the capital of Spain? Answer key on page 47..."
```

It is not being unhelpful. It has correctly identified that in its training corpus, text that
looks like a question is usually followed by **more questions** — it has found a quiz. It is
doing precisely what it was trained to do.

Three gaps separate a base model from an assistant:

**It does not follow instructions.** Instruction-following is a behaviour, and nothing in
next-token prediction rewards it.

**It has no values.** It will produce toxic, biased, dangerous or deceptive content as readily as
anything else, if that is what the context makes plausible.

**It does not know how to decline.** The concept of refusing a request does not exist for it.

**Alignment** is the process of closing those gaps. The conventional shorthand for the target is
**HHH: helpful, harmless, honest** — and note that those three genuinely conflict. Maximum
helpfulness means answering everything; maximum harmlessness means refusing anything risky.
Alignment is where that tension gets resolved, which is why it is a values question as much as a
technical one.

---

## 2. The alignment pipeline

```text
   ┌─────────────────────────────────────────────────────────────┐
   │  BASE MODEL  (pretrained, next-token prediction)            │
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  STEP 1 — SUPERVISED FINE-TUNING (SFT)                      │
   │  Humans write ideal responses to prompts.                   │
   │  Train on (instruction, response) pairs.                    │
   │  → the model learns to RESPOND rather than continue         │
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  STEP 2 — REWARD MODELING                                   │
   │  Generate several candidate responses per prompt.           │
   │  Humans RANK them (A > B > C).                              │
   │  Train a reward model to predict human preference.          │
   │  → a learned, automatable proxy for human judgement         │
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  STEP 3 — RL OPTIMIZATION (PPO)                             │
   │  The LLM is the POLICY; the reward model gives the REWARD.  │
   │  Optimize the policy to maximise reward,                    │
   │  with a KL penalty anchoring it to the SFT model.           │
   │  → helpful, harmless, honest behaviour                      │
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
                    ALIGNED / INSTRUCT MODEL
```

### Step 1 — Supervised fine-tuning

Human demonstrators write ideal responses; the model is fine-tuned on those pairs. This alone
converts a base model into something usable. Quality matters enormously more than quantity — a
few thousand carefully written demonstrations outperform a hundred thousand scraped ones.

### Step 2 — The reward model

Take a prompt, generate several candidate responses, and ask a human which is better.

!!! important "Why ranking, not scoring — a genuinely exam-worthy detail"
    You could ask annotators to rate each response 1–10. In practice this works badly. Humans are
    poorly calibrated at absolute judgement: one annotator's 7 is another's 5, individuals drift
    over a session, and nobody can articulate the difference between a 6 and a 7.

    Humans are **much** more reliable at **comparison**: "is A better than B?" is a question
    people answer consistently.

    So preference data is collected as **pairwise comparisons or rankings**, and the reward model
    is trained to produce a scalar score whose *ordering* matches human preference. The absolute
    values are meaningless; only the ordering matters.

The reward model is usually initialised from the LLM itself, with the output head replaced by a
single scalar.

### Step 3 — Reinforcement learning with PPO

Now the RL framing from [ML Fundamentals](../domain-1/ml-fundamentals.md) applies directly:

- The **policy** is the LLM.
- An **action** is generating a response.
- The **reward** is the reward model's score.

**PPO (Proximal Policy Optimization)** is the standard algorithm. It is "proximal" because it
constrains how far the policy can move in a single update, which keeps training stable.

**The KL penalty is the part that matters conceptually.** The objective includes a penalty for
diverging too far from the SFT reference model:

$$ \text{objective} = \mathbb{E}[r(x,y)] - \beta \cdot D_{KL}(\pi_{\text{RL}} \parallel \pi_{\text{SFT}}) $$

Without it, the policy discovers that the reward model is only a **proxy** for human preference
and exploits its flaws — producing degenerate text that scores enormously well and is nonsense to
a human. That is **reward hacking**, and the KL term is the leash that prevents it.

---

## 3. Alternatives to full RLHF

RLHF is complex, unstable and expensive — three models in play (policy, reward model, reference),
an RL loop that is notoriously finicky, and substantial compute. Simpler methods have largely
displaced it.

| Method | The idea |
| --- | --- |
| **DPO** (Direct Preference Optimization) | Optimize directly on preference pairs with a classification-style loss. **No separate reward model, no RL loop.** Simpler, more stable, and the common default |
| **RLAIF** | Replace human preference labels with **AI-generated** ones. Much cheaper and scalable; inherits the labeling model's biases |
| **Constitutional AI** | The model critiques and revises its **own** outputs against a written set of principles (a "constitution"), generating preference data with minimal human labeling |
| **KTO, ORPO, IPO** | Further preference-optimization variants requiring weaker supervision |
| **Rejection sampling / best-of-n** | Sample *n* responses, keep the highest-reward one, and fine-tune on those. Simple and surprisingly effective |

**DPO's insight** is mathematically elegant and worth appreciating: the optimal policy under the
RLHF objective has a closed-form relationship to the reward function. That relationship can be
inverted, letting you express the reward *implicitly* through the policy itself — so you can
optimise for preferences directly with a simple loss, and the reward model and RL loop are never
needed at all.

!!! tip "The contrast that gets tested"
    **RLHF = reward model + reinforcement learning (PPO).**
    **DPO = skip both; optimise on preference pairs directly.**

    Both still need human preference data. DPO removes the machinery, not the labels.

---

## 4. Alignment failure modes

Each of these is a real, observed phenomenon and a plausible exam item.

**Reward hacking.** The policy maximises the measured proxy while violating the intent. The
general form of Goodhart's law: when a measure becomes a target, it ceases to be a good measure.

**Sycophancy.** Human raters prefer being agreed with, so the model learns to agree — including
when the user is wrong. Push back on a correct answer and a sycophantic model will retract it.
This is a *direct consequence* of optimising for human approval, not a bug in the implementation.

**Over-refusal.** Safety training generalises too broadly and harmless requests get declined —
the chemistry teacher who cannot get help with a lesson plan. A real cost, and a live trade-off:
every increment of harmlessness buys some loss of helpfulness.

**Alignment tax.** Some raw capability is lost in exchange for safety and instruction-following.
Aligned models sometimes score slightly worse on raw benchmarks than their base versions.

**Verbosity/length bias.** Raters prefer longer, more thorough-looking answers, so models learn
to pad. This is the same bias that affects
[LLM-as-a-judge](evaluation-metrics.md#llm-as-a-judge), for the same reason.

**Jailbreaking.** Adversarial prompts that recover the unaligned behaviour. See
[Guardrails & LLM Security](../domain-5/guardrails-security.md).

---

## 5. Human annotation quality

The blueprint's phrase "the use of human subjects in labeling" makes this examinable on its own,
and the job description lists *"define, curate, label, and annotate LLM datasets"* as a core
responsibility.

### Running a labeling effort

**1. Write a precise guideline first.** Definitions, decision rules, and worked edge cases. The
single most important insight in this section: **most annotator disagreement is specification
failure, not annotator failure.** If two careful people label the same item differently, it is
usually because the guideline did not tell them how to handle it.

**2. Pilot on a small batch.** Review the disagreements, revise the guideline, repeat. Do this
before scaling to ten thousand items, not after.

**3. Use multiple annotators per item** and adjudicate conflicts.

**4. Measure agreement.** See below.

**5. Insert gold-standard items** — items with a known correct answer, sprinkled through the
work, as an ongoing quality check.

**6. Watch for drift and fatigue.** Annotators change their interpretation over long sessions.
Rotate, re-calibrate, and re-check periodically.

### Inter-annotator agreement

| Measure | Use |
| --- | --- |
| **Raw agreement** | Percentage of items where annotators match. **Inflated when one class dominates** |
| **Cohen's κ** | Two annotators, **corrected for chance agreement** |
| **Fleiss' κ** | Three or more annotators |
| **Krippendorff's α** | Any number of annotators; handles missing data and multiple scale types |

**Why raw agreement is not enough**, concretely: if 95% of items are "not toxic", two annotators
who both label everything "not toxic" agree 95% of the time while demonstrating no skill
whatsoever. Kappa corrects for the agreement expected by chance, so this scenario scores κ ≈ 0.

Rough interpretation of κ: `<0.2` poor · `0.2–0.4` fair · `0.4–0.6` moderate · `0.6–0.8`
substantial · `>0.8` almost perfect.

!!! danger "Low agreement is a specification bug, not a people problem"
    If κ = 0.15, do not collect more data and do not replace the annotators. **Fix the
    guideline.**

    Two things follow from low agreement, and both are fatal. A model trained on those labels
    **cannot exceed the noise ceiling** they impose — the labels themselves disagree about the
    right answer. And any *evaluation* built on them is unreliable, so you cannot even tell how
    badly you are doing.

### The ethics of human labeling

A genuine consideration under NVIDIA's Trustworthy AI framing, and examinable:

- **Fair pay and working conditions.** Much annotation is outsourced to low-wage markets.
- **Informed consent** about how the data will be used.
- **Psychological support** for annotators reviewing harmful content — safety and moderation
  labeling means exposure to the worst material on the internet, as a job.
- **Transparency** about the purpose of the work.
- **Privacy protection** for any personal data being labeled. See
  [Privacy & Consent](../domain-5/privacy-consent.md).

---

## 6. Recap

- A base model **continues text**; it does not follow instructions, hold values, or know how to
  decline. Alignment closes those gaps toward **helpful, harmless, honest** — three goals that
  genuinely conflict.
- **RLHF = SFT → reward model (trained on human *rankings*) → PPO with a KL penalty.**
- **Ranking beats scoring** because humans are reliable at comparison and unreliable at absolute
  judgement.
- The **KL penalty** anchors the policy to the SFT model and is what prevents **reward hacking**.
- **DPO** achieves the same outcome without a reward model or RL loop — simpler and more stable.
  **RLAIF** and **Constitutional AI** substitute AI feedback for human labels.
- Failure modes: reward hacking, **sycophancy** (a direct consequence of optimising for
  approval), over-refusal, alignment tax, verbosity bias.
- Measure **inter-annotator agreement** with Cohen's/Fleiss' κ, which corrects for chance — raw
  agreement is inflated by class imbalance.
- **Low agreement means the guideline is broken**, and it caps both model quality and evaluation
  reliability.
