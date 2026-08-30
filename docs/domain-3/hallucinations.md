# Hallucinations

*"Hallucinations in Large Language Models" is in NVIDIA's suggested reading list for this
domain, and hallucination sits at the intersection of Experimentation and
[Trustworthy AI](../domain-5/index.md).*

## Definition

A **hallucination** is model output that is fluent, confident and **wrong** — factually
false, unsupported by the provided source, or entirely fabricated.

The defining property is the mismatch between **confidence and correctness**. A model that
said "I'm not sure" would not be hallucinating; the problem is that it does not.

## Why they happen — the mechanism

This is the part worth understanding rather than memorising.

1. **The training objective is plausibility, not truth.** An LLM is trained to predict the
   next token. Nothing in that objective rewards being right; it rewards being *likely*.
   A fabricated citation in the right format is highly likely text.
2. **No grounding.** A base model has no database to consult. Knowledge is diffusely
   encoded in weights, with no notion of provenance or confidence.
3. **Knowledge gaps and cutoffs.** Asked about something absent from training, the model
   still produces the most plausible continuation — which is a guess dressed as a fact.
4. **Sampling randomness.** Higher temperature samples lower-probability tokens, and
   improbable claims are more often false ones.
5. **Compression.** Trillions of tokens compressed into billions of parameters cannot be
   lossless; details blur and recombine.
6. **Sycophancy from alignment.** Preference training rewards agreeable, confident,
   helpful-sounding answers — including when the honest answer is "I don't know".

## Taxonomy

| Type | Meaning | Example |
| --- | --- | --- |
| **Factual (intrinsic to the world)** | Contradicts established fact | Wrong date for a historical event |
| **Faithfulness (extrinsic to the source)** | Contradicts or exceeds the provided context | Summary states something the document never says |
| **Fabrication** | Invents a non-existent entity | A plausible but fictitious paper, citation, API or legal case |
| **Logical** | Internally inconsistent reasoning | Correct steps, contradictory conclusion |
| **Instruction** | Ignores an explicit constraint | Returns prose when JSON was required |

!!! tip "The distinction that matters in RAG"
    **Factual** hallucination = wrong about the world.
    **Faithfulness** hallucination = wrong about *the provided context*.

    RAG systems are evaluated on **faithfulness** — see the
    [RAG triad](rag-evaluation.md#generation-metrics-the-rag-triad). A RAG answer can be
    factually true and still unfaithful (asserting something correct that the retrieved
    documents do not support), which is still a defect: it means the citation is a lie.

## Mitigation strategies

Ordered roughly by effectiveness for factual questions:

| Strategy | How it helps |
| --- | --- |
| **RAG / grounding** | Supplies the facts, so the model retrieves rather than recalls. **The single most effective mitigation** |
| **Explicit grounding instructions** | "Answer using ONLY the context. If it is not there, say you don't know." Enables refusal |
| **Require citations** | Forces claims to be traceable, and makes fabrications visible to the reader |
| **Low temperature** | Reduces low-probability token choices for factual tasks |
| **Chain-of-thought** | Reduces reasoning-type errors on multi-step problems |
| **Self-consistency** | Sample several answers; disagreement flags unreliability |
| **Self-verification / critique** | A second pass checks the answer against the source |
| **Structured output + validation** | Schema violations are caught mechanically |
| **Fine-tuning to refuse** | Train the model to say "I don't know" — behaviour, not knowledge |
| **Tool use** | Calculator, database, search — delegate facts to systems that are actually authoritative |
| **Guardrails** | [NeMo Guardrails](../domain-5/guardrails-security.md) can enforce fact-checking rails |
| **Human review** | Mandatory for high-stakes domains — medical, legal, financial |

!!! danger "What does NOT reliably fix hallucination"
    **Fine-tuning on more facts.** It injects knowledge diffusely and unverifiably, goes
    stale, cannot be cited, and often makes things worse: teaching a model to state facts
    confidently generalises into stating *unknown* facts confidently.

    A frequent exam distractor. The knowledge fix is **RAG**.

## Detection

- **Faithfulness scoring** — decompose the answer into atomic claims and check each against
  the source with an NLI model or a judge LLM.
- **Self-consistency sampling** — generate *n* answers at temperature > 0; high variance in
  factual content signals low reliability.
- **Token-probability / uncertainty signals** — low average token probability correlates
  (imperfectly) with fabrication.
- **External verification** — check citations resolve, check numbers against a database,
  execute generated code.
- **Benchmarks** — TruthfulQA, and domain-specific factuality sets.
- **User feedback** — thumbs-down plus a sampled human review loop in production.

!!! warning "Models are poorly calibrated about their own knowledge"
    Asking a model "are you sure?" is weak evidence: it can be talked out of correct
    answers and will confidently defend wrong ones. Verification must come from outside
    the model.

## Communicating the risk

A Trustworthy AI expectation as much as a technical one: users must know that outputs can
be wrong. Show citations, expose confidence where you can, design the UI so the model is
positioned as a **draft-producing assistant** rather than an oracle, and keep a human in
the loop wherever the cost of an error is high.

## Key takeaways

- Hallucination = **fluent, confident, wrong**; it follows directly from an objective that
  optimises plausibility rather than truth.
- **Factual** (wrong about the world) vs. **faithfulness** (unsupported by the given
  context) — RAG is evaluated on the latter.
- **RAG + grounding instructions + citations + low temperature** is the standard mitigation
  stack.
- **Fine-tuning is not a hallucination fix** for missing knowledge.
- Detect with faithfulness scoring, self-consistency, external verification — not by asking
  the model whether it is sure.
