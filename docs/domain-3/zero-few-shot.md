# Zero-, One- and Few-Shot Testing

*"Zero-shot testing" is named directly in NVIDIA's suggested reading list, and encoder
models "for zero-shot classification" appear in its course objectives.*

## Definitions

| Setting | Examples in the prompt | Weight updates |
| --- | --- | --- |
| **Zero-shot** | 0 — instruction only | None |
| **One-shot** | 1 | None |
| **Few-shot** | typically 2–10 | None |
| **Fine-tuned** | (training data, not prompt) | **Yes** |

!!! danger "The distinction that gets tested"
    Zero-/one-/few-shot **prompting** involves **no training whatsoever**. The examples are
    part of the input; the weights never change. This is **in-context learning**, not
    "training with few examples".

    Do not confuse it with **few-shot *learning*** in the classical ML sense (meta-learning
    from few labeled examples, which *does* update parameters). On this exam, "few-shot"
    means prompting.

## Zero-shot classification with encoders

NVIDIA's course objective mentions encoder models for zero-shot classification. Two
mechanisms are worth knowing:

- **NLI-based zero-shot** — frame classification as natural language inference: the input
  is the premise, and each candidate label becomes a hypothesis ("This text is about
  *finance*."). The entailment score ranks the labels. This is what Hugging Face's
  `pipeline("zero-shot-classification")` does.
- **Embedding similarity** — embed the input and embed each label description, then pick
  the nearest label. Cheap and surprisingly strong.

Neither requires a single labeled training example — the reason zero-shot is the default
starting point when you have no data.

## When to use which

```text
No labeled data at all ─────────────────────► zero-shot
A handful of examples, unusual format ──────► few-shot
Hundreds to thousands, need consistency ────► fine-tune (PEFT/LoRA)
The model lacks the knowledge, not the skill► RAG
```

| Consideration | Zero-shot | Few-shot | Fine-tuned |
| --- | --- | --- | --- |
| Setup cost | None | Minutes | Hours–days |
| Prompt length / token cost | Lowest | Higher (examples every call) | Low again |
| Latency | Lowest | Higher | Low |
| Consistency of format | Weakest | Better | Best |
| Accuracy on niche tasks | Lowest | Middle | Highest |
| Data required | 0 | 2–10 | 100s–1000s |

!!! tip "Practical trade-off"
    Few-shot examples are re-sent with **every single request**, so they add token cost and
    latency permanently. If a task runs at high volume and few-shot is doing heavy lifting,
    distilling that behaviour into a LoRA adapter is usually cheaper in the long run.

## Evaluating shot-based prompting fairly

This is where the exam's *"testing"* framing bites. Comparisons are easy to get wrong:

1. **Hold the eval set fixed** across all settings, and never draw few-shot examples from
   it — that is leakage, and it inflates scores.
2. **Report the number of shots.** "85% accuracy" is meaningless without it.
3. **Vary example order and selection.** Few-shot results are notoriously sensitive to
   both. Run several permutations and report mean ± std, or you are measuring luck.
4. **Watch for known biases:**
    - **Majority-label bias** — over-representing one class in the examples skews
      predictions toward it.
    - **Recency bias** — the last example carries disproportionate weight.
    - **Common-token bias** — the model favours frequent tokens as answers.
5. **Control decoding.** Fix temperature (ideally 0) and the model version, or you are
   comparing samples, not systems.
6. **Check the token budget.** Long few-shot prompts can crowd out retrieved context in a
   RAG system — a real, and commonly overlooked, interaction.

## Selecting few-shot examples

- **Diverse and representative** — cover the classes and the edge cases, including at
  least one negative or "none of the above" case.
- **Identical formatting** — the model copies structure far more faithfully than semantics.
- **Correct** — a single wrong label in the examples measurably degrades output.
- **Dynamic (retrieval-based) selection** — embed the incoming query and retrieve the
  *k* most similar labeled examples to use as shots. Consistently beats a fixed static
  set, at the cost of a retrieval step. Note this is the same machinery as RAG, applied to
  examples instead of documents.

## Related terms

- **Chain-of-thought** can be combined with any shot setting; zero-shot CoT is just
  *"Let's think step by step."* See [Prompt Engineering](../domain-1/prompt-engineering.md).
- **Zero-shot transfer** — applying a model to a task or language it never saw in
  training; the headline capability of large multilingual models.
- **Instruction tuning** is what *makes* zero-shot work well: base models are poor at
  following bare instructions, instruction-tuned models are good at it.

## Key takeaways

- Zero/one/few-shot = **0 / 1 / a handful of examples in the prompt**, with **no weight
  updates**.
- Few-shot works through **in-context learning**, an emergent property of scale.
- Zero-shot classification can be done with NLI-style entailment or embedding similarity —
  no labeled data required.
- Few-shot results are **highly sensitive to example choice and order**; report mean ± std
  over permutations.
- Few-shot costs tokens on every call; at high volume, PEFT is often cheaper.
