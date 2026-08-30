# Hallucinations

*"Hallucinations in Large Language Models" is named in NVIDIA's suggested reading list for this
domain, and the topic sits at the intersection of Experimentation and
[Trustworthy AI](../domain-5/index.md).*

---

## 1. What a hallucination is

A **hallucination** is model output that is fluent, confident and **wrong** — factually false,
unsupported by the provided source, or entirely fabricated.

The defining property is the **mismatch between confidence and correctness**. A model that said
"I'm not certain, but I think…" would not be hallucinating in the harmful sense. The problem is
that the wrong answer arrives in exactly the same tone, with exactly the same fluency and
apparent authority, as a right one. There is no signal in the output distinguishing them.

That is what makes it dangerous. A system that fails obviously gets caught. A system that fails
*invisibly* gets believed.

---

## 2. Why they happen — the mechanism

This is worth understanding rather than memorising, because it converts a family of exam
questions from recall into reasoning.

### The objective rewards plausibility, not truth

An LLM is trained to predict the next token — to produce **likely** text. Nothing in the loss
function rewards being right. Consider what happens when you ask for a citation:

```text
"According to Smith et al. (2019), published in the Journal of..."
```

This is **extremely likely text**. Academic prose looks exactly like this. The model has seen
tens of thousands of citations in this form, and every token is a high-probability continuation
of the ones before it. Whether that specific paper exists is not a question the training
objective ever asked.

The model is not lying, and it is not confused. It is doing precisely what it was optimised to
do. Fabrication is not a malfunction of next-token prediction — it is next-token prediction
working correctly on a task it was never designed for.

### The other contributing causes

**No grounding.** A base model has no database to consult. Knowledge is diffusely encoded across
billions of weights, with no notion of provenance, no confidence estimate, and no way to
distinguish "I learned this from a thousand reliable sources" from "I saw something like this
once".

**Knowledge gaps and cutoffs.** Asked about something absent from training — a recent event, a
private document, an obscure detail — the model still produces the most plausible continuation.
There is no internal mechanism that fires and says "no information found".

**Sampling randomness.** Higher temperature samples lower-probability tokens, and improbable
claims are more often false ones. See
[Prompt Engineering](../domain-1/prompt-engineering.md#decoding-parameters).

**Lossy compression.** Trillions of tokens compressed into billions of parameters cannot be
lossless. Details blur, get approximated, and recombine — the way human memory produces confident
false recollections.

**Sycophancy from alignment.** Preference training rewards answers that raters found helpful and
confident. "I don't know" reads as unhelpful. The model has been shaped, mildly but genuinely,
toward answering rather than declining. See [Alignment & RLHF](rlhf-alignment.md).

---

## 3. The taxonomy

| Type | Meaning | Example |
| --- | --- | --- |
| **Factual** (intrinsic to the world) | Contradicts established fact | Wrong date for a historical event |
| **Faithfulness** (extrinsic to the source) | Contradicts or exceeds the provided context | A summary asserting something the document never says |
| **Fabrication** | Invents a non-existent entity | A plausible but fictitious paper, citation, API method or legal case |
| **Logical** | Internally inconsistent reasoning | Each step correct, conclusion contradicting them |
| **Instruction** | Ignores an explicit constraint | Returns prose when JSON was required |

!!! important "The distinction that matters most in RAG"
    **Factual** hallucination = wrong about the world.
    **Faithfulness** hallucination = wrong about **the provided context**.

    These come apart, and the case where they diverge is instructive. Suppose a RAG system
    answers a question with a statement that is **true in the real world** but **not supported by
    the retrieved documents**.

    That is still a defect. The system claimed to be answering from those documents, and it
    cited them. The citation is a lie even though the fact is correct — and the user, who trusts
    the system precisely because it cites sources, has no way to know that this particular claim
    was invented.

    RAG systems are therefore evaluated on **faithfulness**, not factual accuracy. See the
    [RAG triad](rag-evaluation.md#generation-metrics--the-rag-triad).

---

## 4. Mitigation

Ordered roughly by effectiveness for factual questions.

| Strategy | How it helps |
| --- | --- |
| **RAG / grounding** | Supplies the facts, so the model **reads** instead of **recalling**. The single most effective mitigation |
| **Explicit grounding instructions** | *"Answer using ONLY the context. If it is not there, say you don't know."* Gives the model permission to decline |
| **Require citations** | Forces claims to be traceable and makes fabrications visible to the reader |
| **Low temperature** | Reduces low-probability token choices on factual tasks |
| **Chain-of-thought** | Reduces reasoning-type errors on multi-step problems |
| **Self-consistency** | Sample several answers; disagreement is a reliability signal |
| **Self-verification** | A second pass checks the answer against the source |
| **Structured output + validation** | Schema violations are caught mechanically |
| **Fine-tuning to refuse** | Train the *behaviour* of saying "I don't know" |
| **Tool use** | Delegate facts to systems that are actually authoritative — a calculator, a database, a search API |
| **Guardrails** | [NeMo Guardrails](../domain-5/guardrails-security.md) can enforce fact-checking rails |
| **Human review** | Mandatory in high-stakes domains — medical, legal, financial |

### Why RAG works, precisely

It changes the task. Recalling a fact from diffuse parametric memory is hard and failure is
silent. **Reading a fact from text placed in the context window is easy** and, with citations,
the failure is visible.

You have converted a closed-book exam into an open-book one.

### What does NOT work

!!! danger "Fine-tuning on more facts is not a hallucination fix"
    This is one of the most reliable distractors on the exam, and the reasoning behind it is
    worth understanding.

    Fine-tuning to inject knowledge fails on four counts: the knowledge is **diffuse and
    unverifiable**, it goes **stale** the moment the world changes, it **cannot be cited**, and a
    few hundred examples do not durably install a fact.

    But the fourth problem is the serious one. Fine-tuning on question–answer pairs about facts
    the base model does not know teaches a **meta-lesson**: *"when asked a question of this
    shape, produce a confident, specific answer."* The model generalises that behaviour to
    questions where it has no knowledge at all.

    **You have trained it to fabricate more convincingly.** For knowledge, the answer is
    [RAG](../domain-1/rag.md).

---

## 5. Detection

**Faithfulness scoring.** Decompose the answer into atomic claims and check each against the
source with an NLI model or a judge LLM. Score = fraction supported. This is what RAGAS and
similar frameworks compute.

**Self-consistency sampling.** Generate *n* answers at temperature > 0. High variance in the
*factual content* — different numbers, different names — signals low reliability, even without
knowing the right answer. A useful unsupervised detector.

**Token-probability signals.** Low average token probability correlates, imperfectly, with
fabrication. Useful as a weak signal, not a decision procedure.

**External verification.** Check that cited sources resolve and actually contain the claim. Check
numbers against a database. Execute generated code. This is the only category of detection that
is genuinely reliable, because it does not rely on the model at all.

**Benchmarks.** TruthfulQA, plus domain-specific factuality sets.

**Production feedback.** Thumbs-down signals plus a sampled human review loop.

!!! warning "You cannot ask the model whether it is sure"
    LLMs are poorly calibrated about their own knowledge. A model will confidently defend a
    fabrication and can be talked out of a correct answer by mild pushback — a direct consequence
    of the [sycophancy](rlhf-alignment.md#alignment-failure-modes) that preference training
    introduces.

    Self-reported confidence is **generated text**, not introspection. Verification must come
    from outside the model.

---

## 6. Communicating the risk

A Trustworthy AI expectation as much as a technical one. Users must be able to tell that output
can be wrong:

- **Show citations** and make them clickable, so verification is one click rather than a research
  project.
- **Expose confidence** where you legitimately have it — retrieval scores, disagreement across
  samples.
- **Design the interface so the model is positioned as a draft-producing assistant**, not an
  oracle. A system that says "here is a draft, please review" produces very different user
  behaviour from one that says "here is the answer".
- **Keep a human in the loop** wherever the cost of an error is high.

That last point connects to **overreliance**, which is a listed OWASP LLM risk in its own right:
the failure is not only that the model was wrong, but that the system was designed in a way that
encouraged a human to trust it without checking.

---

## 7. Recap

- Hallucination = **fluent, confident, wrong**. The danger is that it is indistinguishable in
  tone from a correct answer.
- It follows directly from an objective that optimises **plausibility rather than truth** —
  fabrication is next-token prediction working correctly on a task it was not designed for.
- **Factual** (wrong about the world) vs. **faithfulness** (unsupported by the given context).
  RAG is evaluated on the latter, and an answer can be factually true and still unfaithful.
- The standard mitigation stack: **RAG + grounding instruction + citations + low temperature**,
  plus tool use for anything computable.
- **Fine-tuning is not a hallucination fix** for missing knowledge — and can make it worse by
  teaching confident answering as a behaviour.
- Detect with **faithfulness scoring**, **self-consistency**, and **external verification** — not
  by asking the model whether it is sure.
- Design for **overreliance**: citations, visible uncertainty, human review where errors are
  costly.
