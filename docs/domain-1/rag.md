# Retrieval-Augmented Generation (RAG)

*Covers tasks 1.3 and 1.4. NVIDIA names RAG explicitly in the blueprint and links its own
article* [What Is Retrieval-Augmented Generation, aka RAG?](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)
*in the Trustworthy AI reading list — a strong signal that it is examined from both a
capability and a trust angle.*

## The problem RAG solves

A pretrained LLM has four hard limits:

1. **Knowledge cutoff** — it knows nothing after its training date.
2. **No private data** — your wiki, tickets and contracts were never in its corpus.
3. **Hallucination** — asked something it does not know, it produces a fluent, confident,
   wrong answer.
4. **No provenance** — you cannot tell where an answer came from.

**RAG fixes all four by changing the input, not the weights**: retrieve relevant
documents at query time and put them in the prompt as context.

!!! quote "One-line definition"
    RAG combines a **retriever** (finds relevant content in an external knowledge base)
    with a **generator** (an LLM that answers *grounded in* that content).

## The pipeline

```text
                    ┌──────────── INDEXING (offline) ────────────┐
                    │                                            │
  documents ──► extract ──► clean ──► chunk ──► embed ──► vector store
                                                            │
                    ┌──────────── QUERY TIME (online) ───────────┐
                    │                                            ▼
  user query ──► embed query ──► similarity search ──► top-k chunks
                                                            │
                                                     (optional rerank)
                                                            │
                                                            ▼
                     prompt = system + retrieved context + query
                                                            │
                                                            ▼
                                                          LLM
                                                            │
                                                            ▼
                                            grounded answer + citations
```

**Indexing (offline, done once per corpus update)** — see
[Embeddings & Vector Search](embeddings.md) for the detail on chunking and curation.

**Retrieval (online, per query)**

1. Embed the query **with the same model** used for indexing.
2. Search the vector store for the top-*k* nearest chunks (typical *k* = 3–10).
3. Optionally **rerank** with a cross-encoder and keep the best few.
4. Optionally apply **metadata filters** (date, source, user permissions).

**Augmentation** — assemble the prompt:

```text
You are a support assistant. Answer using ONLY the context below.
If the context does not contain the answer, say you do not know.
Cite the source of each claim.

<context>
[1] {chunk_1}  (source: handbook.pdf, p.12)
[2] {chunk_2}  (source: faq.md)
</context>

Question: {user_query}
```

**Generation** — the LLM answers. The instruction *"use only the context"* plus
*"say you don't know"* is what converts retrieval into **grounding**.

## RAG vs. fine-tuning vs. prompting

The most commonly examined decision in the entire certification.

| | **Prompt engineering** | **RAG** | **Fine-tuning (PEFT/full)** |
| --- | --- | --- | --- |
| Changes weights? | No | No | Yes |
| Adds **new/current knowledge**? | Only what fits in the prompt | **Yes** | Poorly and expensively |
| Teaches **style, format, behaviour**? | Somewhat | No | **Yes** |
| Data freshness | — | **Update the index, instantly** | Requires retraining |
| Cost to build | Lowest | Medium | Highest |
| Cost per query | Lowest | Higher (longer prompts, retrieval latency) | Same as base model |
| Traceability / citations | No | **Yes** | No |
| Reduces hallucination | Slightly | **Substantially** | Not reliably |

!!! tip "The decision rule to memorise"
    - The model **doesn't know something** (private, recent, or factual) → **RAG**.
    - The model **doesn't behave the way you want** (tone, format, domain jargon, task
      structure) → **fine-tuning**.
    - You need both → **do both**: fine-tune for behaviour, RAG for facts.
    - Always try **prompting first** — it is free and often sufficient.

**Why fine-tuning is the wrong tool for facts:** knowledge injected by fine-tuning is
diffuse and unverifiable, it goes stale the moment the world changes, it cannot be
cited, and updating one fact means retraining. RAG updates are a database write.

## Advanced RAG patterns

Recognise these by name:

- **Hybrid search** — dense + BM25 keyword, fused (RRF). Fixes exact-ID retrieval.
- **Reranking** — retrieve top-50, cross-encode, keep top-5.
- **Query rewriting / expansion** — an LLM rewrites a vague or conversational query
  ("what about the second one?") into a standalone, retrievable query. Essential in
  multi-turn chat.
- **Multi-query** — generate several paraphrases, retrieve for each, union the results.
- **HyDE** (Hypothetical Document Embeddings) — have the LLM draft a hypothetical answer
  and embed *that* to search with, closing the vocabulary gap between short queries and
  long passages.
- **Parent-document / small-to-big** — embed small chunks for precise matching but return
  the larger parent section for context.
- **Contextual compression** — filter or summarise retrieved chunks before they enter the
  prompt, to save context budget.
- **Self-RAG / corrective RAG** — the model judges whether retrieved context is
  sufficient and re-retrieves if not.
- **GraphRAG** — retrieve over a knowledge graph rather than flat chunks; better for
  multi-hop questions.
- **Agentic RAG** — an agent decides *whether*, *where* and *how many times* to retrieve.

## Where RAG systems fail

| Failure | Symptom | Fix |
| --- | --- | --- |
| Retrieval miss | The answer exists but was not retrieved | Better chunking, hybrid search, higher *k*, query rewriting |
| Poor ranking | The right chunk is at position 40 | Add a reranker |
| Chunks too small | Retrieved text is out of context | Increase size/overlap, parent-document retrieval |
| Chunks too large | Diluted embeddings, wasted context | Smaller chunks, semantic splitting |
| "Lost in the middle" | Long contexts bury relevant info | Fewer, better chunks; put key context first or last |
| Ignored context | The model answers from parametric memory | Stricter prompt, lower temperature, grounding instruction |
| Stale index | Answers cite deleted documents | Scheduled re-indexing |
| Embedding mismatch | Nonsense retrieval after a model change | Re-index the whole corpus |

Evaluation of each stage is covered in [Evaluating RAG Systems](../domain-3/rag-evaluation.md).

## Other classic LLM use cases (task 1.3)

**Chatbots / conversational assistants** — add **conversation memory**. Options: keep the
full history (simple, grows unboundedly), keep a sliding window of recent turns,
summarise older turns, or retrieve relevant past turns. Multi-turn RAG needs query
rewriting so that "and the price?" becomes a self-contained query.

**Summarization**

- **Extractive** — select the most important existing sentences. Faithful by
  construction, can read choppily.
- **Abstractive** — generate new text. Fluent, and can hallucinate.
- **Long documents** — *map-reduce* (summarise chunks, then summarise the summaries) or
  *refine* (iteratively update a running summary chunk by chunk).

**Classification / extraction / NER** — encoder models or an LLM with a strict
structured-output schema (JSON). Prefer the encoder when you have labeled data and need
throughput.

**Code generation, translation, semantic search, agents** — the remaining common
patterns.

## Key takeaways

- RAG = retriever + generator; it grounds answers in external data **without touching
  model weights**.
- Two phases: offline indexing (chunk → embed → store), online query (embed → search →
  rerank → augment → generate).
- **Knowledge problem → RAG. Behaviour problem → fine-tuning.**
- RAG's unique advantages: fresh data, private data, **citations**, less hallucination.
- Retrieval quality caps answer quality — most RAG failures are retrieval failures.
- Know the pattern names: hybrid search, reranking, query rewriting, HyDE, GraphRAG.
