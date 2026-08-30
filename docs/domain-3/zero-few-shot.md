# Zero-, One- and Few-Shot Testing

*"Zero-Shot Testing" is named directly in NVIDIA's suggested reading list, and encoder models
"for zero-shot classification" appear in its course objectives.*

---

## 1. The definitions

| Setting | Examples in the prompt | Weight updates | Labeled data required |
| --- | --- | --- | --- |
| **Zero-shot** | 0 — instruction only | **None** | None |
| **One-shot** | 1 | **None** | 1 |
| **Few-shot** | typically 2–10 | **None** | A handful |
| **Fine-tuned** | (data goes into training, not the prompt) | **Yes** | 100s–1000s |

!!! danger "The distinction the exam tests"
    Zero-, one- and few-shot **prompting** involves **no training whatsoever**. The examples are
    part of the model's *input*; the weights never change. Start a new conversation and the model
    has no memory of them.

    This is **in-context learning** — an emergent capability of large models, described in
    [Prompt Engineering](../domain-1/prompt-engineering.md).

    Do not confuse it with classical **few-shot *learning*** (meta-learning from few labeled
    examples), which *does* update parameters. On this exam, "few-shot" means prompting, and
    options describing it as "fine-tuning on a few examples" are wrong.

---

## 2. Zero-shot classification with encoder models

NVIDIA's course objective mentions encoder models for zero-shot classification, so it is worth
knowing how that actually works without any labeled data.

### The NLI trick

Reframe classification as **natural language inference**. NLI models are trained to judge whether
one sentence (the *premise*) entails another (the *hypothesis*). So:

```text
premise:     "The GPU ran out of memory during fine-tuning."

hypothesis A: "This text is about hardware."      → entailment score 0.91
hypothesis B: "This text is about billing."       → entailment score 0.04
hypothesis C: "This text is about documentation." → entailment score 0.05

                                        → predicted label: hardware
```

Each candidate label is turned into a hypothesis, and the entailment scores rank them. This is
what Hugging Face's `pipeline("zero-shot-classification")` does, and you can invent new labels at
runtime without retraining anything.

### The embedding trick

Embed the input, embed a description of each label, and pick the nearest. Cheaper than NLI and
surprisingly competitive. See [Embeddings](../domain-1/embeddings.md).

Neither approach requires a single labeled training example, which is why zero-shot is the
default starting point when you have no data — and why it is worth trying before you commission
an annotation effort.

---

## 3. When to use which

```text
No labeled data at all ───────────────────────────► zero-shot
A handful of examples, unusual format or labels ──► few-shot
Hundreds to thousands, need consistency at scale ─► fine-tune (PEFT/LoRA)
The model lacks KNOWLEDGE, not skill ─────────────► RAG
```

| Consideration | Zero-shot | Few-shot | Fine-tuned |
| --- | --- | --- | --- |
| Setup cost | None | Minutes | Hours–days |
| Prompt length / token cost | Lowest | **Higher — every call** | Low again |
| Latency | Lowest | Higher | Low |
| Format consistency | Weakest | Better | **Best** |
| Accuracy on niche tasks | Lowest | Middle | **Highest** |
| Data required | 0 | 2–10 | 100s–1000s |

!!! tip "The trade-off that decides it in production"
    Few-shot examples are re-sent with **every single request**. A 600-token example block on a
    million requests a month is 600 million tokens of pure overhead — paid forever, and adding
    latency to every call.

    In a RAG system it is worse: those tokens compete for context window with the retrieved
    chunks that actually contain the answer.

    So the calculus is: **few-shot is cheap to build and expensive to run; fine-tuning is
    expensive to build and cheap to run.** At high volume, distilling a working few-shot prompt
    into a LoRA adapter usually pays for itself quickly.

---

## 4. Evaluating shot-based prompting fairly

This is where the domain's *"testing"* framing bites. These comparisons are easy to get wrong,
and wrong comparisons produce confident bad decisions.

**1. Hold the eval set fixed, and never draw few-shot examples from it.** Using evaluation items
as demonstrations is [data leakage](../domain-1/ml-fundamentals.md#data-leakage) — the model is
being shown the answers to questions it is about to be graded on.

**2. Report the number of shots.** "85% accuracy" is meaningless without it. Zero-shot 85% and
10-shot 85% are completely different results with completely different cost profiles.

**3. Vary example order and selection.** Few-shot performance is notoriously sensitive to both.
Reordering the same five examples can shift accuracy by several points. Run multiple permutations
and report **mean ± standard deviation** — a single run is measuring luck.

**4. Control for the known biases:**

| Bias | Effect |
| --- | --- |
| **Majority-label bias** | Over-representing one class in the examples skews predictions toward it |
| **Recency bias** | The last example carries disproportionate weight |
| **Common-token bias** | The model favours frequent tokens as answers |

These are not hypothetical; they are measurable and they are why a "balanced, shuffled" example
set is not a stylistic preference.

**5. Control decoding.** Fix temperature (ideally 0) and pin the model version, or you are
comparing two samples from the same distribution and calling the difference an effect. See
[Experiment Design](experiment-design.md).

**6. Check the token budget.** Long few-shot prompts can crowd out retrieved context in a RAG
system, so your "few-shot improvement" may be silently degrading retrieval. A real interaction,
and commonly overlooked.

---

## 5. Selecting few-shot examples

- **Diverse and representative.** Cover the classes and the edge cases, including at least one
  negative or "none of the above" case. Whatever you do not demonstrate, the model improvises.
- **Identically formatted.** The model copies structure far more faithfully than semantics.
  Inconsistent separators produce inconsistent output.
- **Correct.** A single mislabeled example measurably degrades results, because the model treats
  it as evidence about your labelling policy.
- **Dynamic (retrieval-based) selection.** Embed the incoming query and retrieve the *k* most
  similar labeled examples to use as shots. Consistently beats a fixed static set, at the cost of
  a retrieval step — and note that this is the same machinery as [RAG](../domain-1/rag.md),
  applied to examples instead of documents.

---

## 6. Related terms

**Chain-of-thought** can be combined with any shot setting. Zero-shot CoT is literally appending
*"Let's think step by step."*

**Zero-shot transfer** — applying a model to a task or language it never saw in training. The
headline capability of large multilingual models, and a genuine break from the
one-model-per-task era.

**Instruction tuning is what makes zero-shot work.** A base model is poor at following bare
instructions; an instruction-tuned model is good at it. When you observe that "zero-shot works
fine now", you are observing the effect of the SFT stage described in
[Alignment & RLHF](rlhf-alignment.md).

---

## 7. Recap

- Zero/one/few-shot = **0 / 1 / a handful of examples in the prompt**, with **no weight updates**
  and no training.
- Few-shot works through **in-context learning**, an emergent capability of scale.
- Zero-shot classification with encoders works via **NLI entailment** or **embedding similarity**
  — no labeled data required.
- Few-shot is **cheap to build, expensive to run** (tokens on every call); fine-tuning is the
  reverse. At high volume, distil few-shot behaviour into a LoRA adapter.
- Few-shot results are **highly sensitive to example choice and order** — report mean ± std over
  permutations, and watch majority-label and recency bias.
- **Never draw few-shot examples from your evaluation set.** That is leakage.
- **Dynamic example selection** by embedding similarity beats a fixed set.
