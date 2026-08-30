# Evaluating RAG Systems

*"Evaluating RAG Applications" is named in NVIDIA's suggested reading list.*

---

## 1. Why a single end-to-end score is not enough

A RAG system has **two** components that can fail independently:

```text
query ──► RETRIEVAL ──► context ──► GENERATION ──► answer
             │                          │
   Did we FIND the right          Did the model USE it
   documents?                     faithfully and answer
                                  the question?
```

Suppose your answer quality score is 0.62. What do you do about it? Improve the prompt? Change
the model? Adjust chunk size? Add a reranker? **The score cannot tell you**, because it collapses
two independent failure modes into one number.

Worse, the two failures can *mask* each other. A system with poor retrieval and a model that
confidently fills gaps from parametric memory may produce answers that look fine on a
surface-level quality metric — while being ungrounded and unverifiable.

!!! important "The diagnostic rule"
    **Always evaluate retrieval first.** If the correct chunk was never retrieved, no amount of
    prompt engineering will fix the answer. You will spend a week tuning the generator on a
    retrieval problem.

    This is the single most useful operational habit in RAG work.

---

## 2. Retrieval metrics

These require a labeled set of `(query, relevant document ids)` pairs. Build it once, from real
queries, with human judgement — see section 5.

| Metric | What it measures |
| --- | --- |
| **Recall@k** | The fraction of relevant documents that appear in the top *k*. **The most important RAG retrieval metric** |
| **Precision@k** | The fraction of the top *k* that are actually relevant |
| **Hit rate@k** | The fraction of queries with **at least one** relevant document in the top *k* |
| **MRR** (Mean Reciprocal Rank) | The average of `1/rank` of the **first** relevant result |
| **nDCG@k** | Normalised Discounted Cumulative Gain — handles **graded** relevance and discounts by position |
| **Latency** | Retrieval time. The reranker is usually the expensive stage |

### Why recall@k is the binding constraint

In a search engine, ranking matters enormously — users look at the first three results.

In RAG, **the LLM reads every chunk you give it**. Whether the right chunk arrived at position 1
or position 8 matters far less than whether it arrived *at all*. If it is in the context, the
model has a chance. If it is not, the system cannot possibly succeed.

That is why recall@k is the first number to look at, and why "increase *k*" is often a legitimate
first response to poor answers.

Ranking still matters for two reasons: **context is finite and expensive**, and long contexts
suffer from ["lost in the middle"](../domain-1/rag.md#where-rag-systems-fail-and-how-to-tell-which).
So MRR and nDCG tell you whether you could use a smaller, better-ranked *k* — which is usually
both cheaper and more accurate.

### Worked example

```text
retrieved (in order):  [d7, d2, d1, d9, d3]
actually relevant:     [d1, d3]

recall@3    = |{d1}| / 2        = 0.50    (d3 is at position 5, outside the top 3)
recall@5    = |{d1, d3}| / 2    = 1.00
precision@3 = 1 / 3             = 0.33
MRR         = 1 / 3             = 0.33    (first relevant result is at rank 3)
```

Now reorder the retrieved list to `[d1, d3, d7, d2, d9]`. **Recall@5 is unchanged at 1.00** —
the same documents were found. But MRR rises to **1.00** and nDCG improves substantially. That
is precisely the difference between "did we find it" and "did we rank it well", and
[Lab 5](../labs/05-evaluation.md) has you run this manipulation.

---

## 3. Generation metrics — the RAG triad

The standard framework, popularised by RAGAS and TruLens. It evaluates the three relationships
between the query, the retrieved context, and the answer.

```text
                    ┌──────────────┐
                    │    QUERY     │
                    └──┬────────┬──┘
          context      │        │      answer
          relevance    │        │      relevance
                       ▼        ▼
             ┌───────────┐   ┌──────────┐
             │  CONTEXT  │──►│  ANSWER  │
             └───────────┘   └──────────┘
                   faithfulness / groundedness
```

| Metric | The question it asks | What it detects |
| --- | --- | --- |
| **Context relevance / precision** | Is the retrieved context relevant to the query? | Retrieval returning noise |
| **Faithfulness / groundedness** | Is every claim in the answer supported by the context? | **Hallucination** |
| **Answer relevance** | Does the answer actually address the question asked? | Evasive or off-topic answers |

**Faithfulness is the hallucination metric for RAG.** How it is computed in practice:

1. Decompose the answer into **atomic claims** — individual factual assertions.
2. For each claim, ask a judge model (or an NLI model) whether the retrieved context **entails**
   it.
3. Score = fraction of claims supported.

An answer with five claims of which four are supported scores 0.8, and you can see exactly which
one was invented.

### The additional metrics worth knowing

**Context recall** — does the retrieved context contain everything needed to produce the
ground-truth answer? (Requires reference answers.)

**Answer correctness** — comparison against a ground-truth answer. Note this is *different* from
faithfulness: an answer can be faithful to retrieved-but-wrong documents.

**Citation accuracy** — do the cited sources actually support the claims attributed to them?
Easy to check, and wrong more often than teams expect. A system that cites confidently and
inaccurately is worse than one that does not cite at all, because it manufactures unwarranted
trust.

**Noise robustness** — does the system still answer correctly when irrelevant chunks are included
in the context? Important, because retrieval always returns some noise.

**Negative rejection** — when the context does **not** contain the answer, does the system
correctly decline instead of inventing one?

!!! important "Test negative rejection — it is the highest-value test you are not running"
    Add questions to your eval set whose answers are genuinely **absent** from your corpus. The
    correct behaviour is *"I don't know based on the provided context."*

    A system that always produces an answer will produce a **confident wrong answer** the first
    time a real user asks something outside the corpus — and real users do that constantly.

    [Lab 4](../labs/04-rag.md) makes you build this test, because it is the one most likely to be
    skipped and the one most likely to matter.

---

## 4. Tooling

**RAGAS**, **TruLens**, **DeepEval**, **LangSmith**, **Phoenix/Arize** — all implement variants of
the triad, mostly using LLM-as-a-judge. NVIDIA's **NeMo Evaluator** covers model and RAG
evaluation within its own stack.

Because these rely on judge models, the
[judge biases](evaluation-metrics.md#llm-as-a-judge) apply — position, verbosity,
self-enhancement. **Calibrate against human labels on a sample** before trusting the absolute
numbers. Judge-based metrics are most reliable as *relative* signals (did this change make things
better?) and least reliable as absolute claims ("our faithfulness is 0.87").

---

## 5. Building a RAG evaluation set

**1. Start with real queries** from logs, not invented ones. Invented queries are always too
clean, too well-formed and too well-matched to your corpus. Real users write fragments, typos,
compound questions and things you did not anticipate.

**2. Include the hard cases deliberately:**

- **Multi-hop** questions requiring two or more documents
- Questions whose answer is **genuinely absent** (for negative rejection)
- **Ambiguous** questions
- Queries containing **exact identifiers** — error codes, part numbers, names (which test hybrid
  search)
- **Out-of-scope** questions
- **Multi-turn** follow-ups with pronouns (which test query rewriting)

**3. Label relevant chunks** for retrieval metrics and write **reference answers** for
correctness.

**4. Consider synthetic generation** as an accelerator: have an LLM produce questions from each
chunk, then have a human filter them. Cheap — but note the bias it introduces: questions
generated *from* a single chunk are, by construction, answerable from a single chunk. Your
multi-hop and negative cases must be written by hand.

**5. Version it, and grow it.** Every production failure becomes a permanent test case. This is
the mechanism by which the eval set gets good.

---

## 6. Diagnosing from the metrics

The reason to compute all of these is that their **combination** localises the fault.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Low recall@k | Chunking, embedding model, or query phrasing | Better chunking, hybrid search, higher *k*, query rewriting |
| Good recall, low MRR/nDCG | Ranking quality | Add a **cross-encoder reranker** |
| Good retrieval, **low faithfulness** | Model ignoring or over-extending the context | Stricter grounding prompt, lower temperature, fewer and cleaner chunks |
| Good faithfulness, low answer relevance | Answering a different question than was asked | Query rewriting; check multi-turn context handling |
| Fails on exact IDs and codes | Dense-only retrieval | **Hybrid search** with BM25 |
| Fabricates when context is empty | No refusal behaviour | Explicit "say you don't know"; test negative rejection |
| Good on the eval set, bad in production | Eval set not representative | Rebuild it from real query logs |

That last row deserves emphasis. If your metrics are healthy and users are unhappy, the problem
is your **evaluation**, not your system — and no amount of tuning against a bad eval set will
help.

---

## 7. Recap

- Evaluate **retrieval and generation separately** — an end-to-end score cannot localise the
  failure, and the two failure modes can mask each other.
- **Always check retrieval first.** If the chunk was never retrieved, prompt engineering cannot
  fix the answer.
- **Recall@k** is the binding retrieval metric for RAG, because the LLM reads every chunk you
  supply. **MRR/nDCG** tell you whether a smaller, better-ranked *k* would do.
- The **RAG triad**: context relevance, **faithfulness**, answer relevance.
- **Faithfulness is the hallucination metric** — computed by decomposing the answer into atomic
  claims and checking each against the context.
- **Test negative rejection.** A system that never says "I don't know" will confidently invent.
- Build the eval set from **real queries**, include multi-hop, absent-answer, identifier and
  multi-turn cases, version it, and grow it from production failures.
- Judge-based metrics are reliable as **relative** signals; calibrate against humans before
  trusting absolute values.
