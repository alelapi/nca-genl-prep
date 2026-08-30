# Evaluating RAG Systems

*"Evaluating RAG Applications" is named in NVIDIA's suggested reading list.*

## Evaluate the components, not just the answer

A RAG system has two failure surfaces, and a single end-to-end score cannot tell you
which one broke:

```text
query ──► RETRIEVAL ──► context ──► GENERATION ──► answer
             │                          │
      did we find the           did the model use it
      right documents?          faithfully and answer
                                the question?
```

**Diagnostic rule:** if the correct chunk was never retrieved, no amount of prompt
engineering will fix the answer. **Always evaluate retrieval first.**

## Retrieval metrics

You need a labeled set of `(query, relevant document ids)` pairs — build it once, from
real queries, with human judgement.

| Metric | Meaning |
| --- | --- |
| **Recall@k** | Fraction of relevant documents that appear in the top *k*. **The most important RAG retrieval metric** — if the answer is not in the context, the system cannot succeed |
| **Precision@k** | Fraction of the top *k* that are actually relevant |
| **Hit rate@k** | Fraction of queries with **at least one** relevant document in the top *k* |
| **MRR** (Mean Reciprocal Rank) | Average of `1/rank` of the first relevant result — rewards putting the right answer first |
| **nDCG@k** | Normalised Discounted Cumulative Gain — handles graded relevance and discounts by position; the standard ranking metric |
| **Latency** | Retrieval time; the reranker is usually the expensive stage |

!!! tip "Recall@k vs. MRR"
    **Recall@k** asks *did we get it into the context window at all?* — the binding
    constraint for RAG, since the LLM can read every retrieved chunk.
    **MRR/nDCG** ask *did we rank it highly?* — which matters more for search UIs, and for
    RAG when *k* is small or "lost in the middle" effects are strong.

## Generation metrics — the RAG triad

The standard framework (popularised by RAGAS and TruLens) evaluates three relationships:

```text
                    ┌──────────────┐
                    │    QUERY     │
                    └──────┬───────┘
             context      │        answer
           relevance      │        relevance
                    ┌─────┴─────┐
                    ▼           ▼
            ┌───────────┐   ┌──────────┐
            │  CONTEXT  │──►│  ANSWER  │
            └───────────┘   └──────────┘
                  faithfulness / groundedness
```

| Metric | Question | Detects |
| --- | --- | --- |
| **Context relevance / precision** | Is the retrieved context relevant to the query? | Retrieval bringing back noise |
| **Faithfulness / groundedness** | Is every claim in the answer supported by the context? | **Hallucination** |
| **Answer relevance** | Does the answer actually address the question? | Evasive or off-topic answers |

Additional metrics you should recognise:

- **Context recall** — does the retrieved context contain everything needed to produce the
  ground-truth answer? (Requires reference answers.)
- **Answer correctness** — comparison against a ground-truth answer.
- **Citation accuracy** — do the cited sources actually support the claims? Easy to check,
  frequently wrong in practice.
- **Noise robustness** — does the system still answer correctly when irrelevant chunks are
  included?
- **Negative rejection** — when the context does *not* contain the answer, does the system
  correctly say "I don't know" instead of inventing one? This is one of the highest-value
  behaviours to test and one of the most commonly missed.

**How faithfulness is computed in practice:** decompose the answer into atomic claims, and
for each claim ask a judge model whether the retrieved context entails it. Score = fraction
of supported claims.

## Tooling

**RAGAS**, **TruLens**, **DeepEval**, **LangSmith**, **Phoenix/Arize** — all implement
variants of the triad, mostly using LLM-as-a-judge. NVIDIA's **NeMo Evaluator** covers
model and RAG evaluation in its own stack.

Because these rely on judge models, the [judge biases](evaluation-metrics.md#llm-as-a-judge)
apply — calibrate against human labels on a sample before trusting the numbers.

## Building a RAG evaluation set

1. **Start with real queries** from logs, not invented ones. Invented queries are always
   too clean.
2. **Include the hard cases**: multi-hop questions, questions whose answer is genuinely
   absent (to test refusal), ambiguous questions, queries with exact identifiers,
   out-of-scope questions.
3. **Label relevant chunks** for retrieval metrics and write **reference answers** for
   correctness.
4. **Synthetic generation** is a legitimate accelerator — have an LLM produce questions
   from each chunk, then have a human filter them. Cheap, but biased toward questions that
   are answerable by a single chunk.
5. **Version it** and grow it: every production failure becomes a permanent test case.

## Diagnosing from the metrics

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Low recall@k | Chunking, embedding model, or query phrasing | Better chunking, hybrid search, query rewriting, higher *k* |
| Good recall, low MRR/nDCG | Ranking quality | Add a cross-encoder reranker |
| Good retrieval, low faithfulness | Model ignoring or over-extending the context | Stricter grounding prompt, lower temperature, fewer/cleaner chunks |
| Good faithfulness, low answer relevance | Answering a different question | Query rewriting; check multi-turn context handling |
| Fails on exact IDs/codes | Dense-only retrieval | Hybrid search with BM25 |
| Fabricates when context is empty | No refusal behaviour | Explicit "say you don't know" instruction; test negative rejection |

## Key takeaways

- Evaluate **retrieval and generation separately** — the answer score alone cannot localise
  the failure.
- **Recall@k** is the binding retrieval metric for RAG; MRR/nDCG measure ranking quality.
- The **RAG triad**: context relevance, **faithfulness** (groundedness), answer relevance.
- **Faithfulness is the hallucination metric** for RAG.
- Test **negative rejection**: the system must decline when the context lacks the answer.
- Build the eval set from real queries, include hard cases, version it, and grow it from
  production failures.
