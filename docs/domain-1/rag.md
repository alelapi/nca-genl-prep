# Retrieval-Augmented Generation (RAG)

*Covers tasks 1.3 and 1.4. NVIDIA also lists its own article*
[What Is Retrieval-Augmented Generation, aka RAG?](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)
*in the **Trustworthy AI** reading list — a strong hint that it is examined from both a
capability angle and a trust angle.*

RAG is the single most examined *application* pattern on NCA-GENL, and the RAG-vs-fine-tuning
decision is probably the most examined *judgement* on the whole exam.

---

## 1. Four problems, one solution

A pretrained LLM has learned an enormous amount, but its knowledge has four structural limits.

**It has a knowledge cutoff.** Training finished on some date. Everything after that date does
not exist as far as the model is concerned.

**It has never seen your private data.** Your wiki, your tickets, your contracts, your codebase,
your customer records — none of it was in the training corpus, and you would not want it to be.

**It hallucinates when it does not know.** This is the important one, and it follows directly
from how the model works. An LLM is trained to produce the *most plausible continuation*.
Asked a question it has no knowledge of, it does not have a mechanism for saying "no
information found" — it produces the most plausible-sounding answer, which is a fluent,
confident fabrication. See [Hallucinations](../domain-3/hallucinations.md).

**It cannot cite anything.** Knowledge is spread diffusely across billions of weights. There is
no record of where any particular fact came from, so there is no way to verify an answer or
audit a decision.

**RAG addresses all four by changing the input rather than the weights.** At query time,
retrieve the relevant documents and put them in the prompt.

!!! quote "One-line definition"
    RAG combines a **retriever** (finds relevant content in an external knowledge base) with a
    **generator** (an LLM that answers *grounded in* that content).

The reframing that makes it click: you are converting a **closed-book exam** into an
**open-book exam**. The model stops being asked to recall and starts being asked to read,
synthesise and cite. That is a much easier task and a much more verifiable one.

---

## 2. The pipeline

Two phases. Understanding which work happens when is worth internalising, because it explains
the cost profile.

```text
╔══════════════ INDEXING — offline, once per corpus update ══════════════╗
║                                                                        ║
║   documents ─► extract ─► clean ─► dedupe ─► chunk ─► embed ─► store   ║
║                                                          │             ║
╚══════════════════════════════════════════════════════════│═════════════╝
                                                           ▼
                                                   ┌───────────────┐
                                                   │ vector store  │
                                                   │  + metadata   │
                                                   └───────────────┘
╔══════════════ QUERY TIME — online, every request ═════════│════════════╗
║                                                           │            ║
║  user query ─► (rewrite) ─► embed ─► similarity search ───┘            ║
║                                          │                             ║
║                                          ▼                             ║
║                                   top-k chunks                         ║
║                                          │                             ║
║                                   (rerank, filter)                     ║
║                                          │                             ║
║                                          ▼                             ║
║        prompt = system instructions + retrieved context + query        ║
║                                          │                             ║
║                                          ▼                             ║
║                                        LLM                             ║
║                                          │                             ║
║                                          ▼                             ║
║                          grounded answer + citations                   ║
╚════════════════════════════════════════════════════════════════════════╝
```

### Indexing (offline)

Covered in detail in [Embeddings & Vector Search](embeddings.md) — extract, clean, deduplicate,
chunk with overlap, embed, store with metadata. This runs once per corpus update, so it can be
expensive without affecting user-facing latency.

### Retrieval (online, per query)

1. **Embed the query** — with the same model used for indexing. Non-negotiable.
2. **Search** for the top-*k* nearest chunks. Typical `k` is 3–10.
3. **Rerank** (optional but usually worth it) — retrieve 20–50 candidates cheaply with the
   bi-encoder, then re-score them with a cross-encoder and keep the best 3–5.
4. **Filter** on metadata — date ranges, document sets, and critically **the current user's
   access permissions**.

### Augmentation — where grounding actually happens

The retrieved chunks are assembled into a prompt. This step looks trivial and is where most of
the quality lives:

```text
You are a support assistant. Answer the question using ONLY the context below.
Cite the source of each claim using its [source] tag.
If the context does not contain the answer, reply exactly:
"I don't know based on the provided context."

<context>
[1] (source: handbook.pdf, p.12)
    Employees accrue 1.75 days of paid leave per completed month of service.

[2] (source: faq.md)
    Unused leave may be carried into the following calendar year up to a
    maximum of 5 days.
</context>

Question: How much leave do I accrue, and can I carry it over?
```

Three instructions are doing all the work:

- **"Use ONLY the context"** — tells the model to prefer the retrieved evidence over its own
  parametric memory.
- **"Cite the source"** — makes claims traceable, and makes fabrication visible to the reader.
- **"If the context does not contain the answer, say so"** — gives the model an explicit escape
  hatch. Without it, the model's overwhelming tendency is to answer *something*.

That third instruction is the one people leave out, and it is the difference between a system
that says "I don't know" and one that invents. [Lab 4](../labs/04-rag.md) has you delete it and
watch the behaviour change.

### Generation

The LLM answers. Use a **low temperature** — this is a factual, grounded task, not a creative
one.

---

## 3. RAG vs. fine-tuning vs. prompting

The most commonly examined decision on the certification. Understand it causally, not as a
table to memorise.

| | **Prompt engineering** | **RAG** | **Fine-tuning (PEFT/full)** |
| --- | --- | --- | --- |
| Changes model weights? | No | No | **Yes** |
| Adds new/current knowledge? | Only what fits in the prompt | **Yes** | Poorly and expensively |
| Teaches style, format, behaviour? | Somewhat | No | **Yes** |
| Updating data | — | **Write to the index — instant** | Retrain |
| Build cost | Lowest | Medium | Highest |
| Per-query cost | Lowest | **Higher** — retrieval latency + long prompts | Same as base |
| Citations possible? | No | **Yes** | No |
| Reduces hallucination? | Slightly | **Substantially** | Not reliably |

### The rule

!!! tip "Memorise this shape"
    - The model **doesn't know something** — private, recent, or specific facts → **RAG**
    - The model **doesn't behave how you want** — tone, format, jargon, task structure →
      **fine-tuning** (PEFT first)
    - Both → **do both**: fine-tune for behaviour, RAG for facts
    - Always try **prompting first** — it is free, immediate, and often sufficient

### Why fine-tuning is the wrong tool for facts

This is worth understanding rather than accepting, because it is a favourite distractor.

**Knowledge injected by fine-tuning is diffuse.** It is spread across millions of weight
adjustments. There is no "fact" stored anywhere you could point at, verify, update or delete.

**It goes stale immediately.** Your policy changes on Monday; the model still believes the old
policy until you run another training job.

**It cannot be cited.** The model cannot tell you which document taught it something, so the
user cannot verify anything.

**It is unreliable.** A few hundred examples mentioning a fact do not durably install that
fact. The model may reproduce it, paraphrase it wrongly, or blend it with contradictory
pretraining knowledge.

**And it can make hallucination worse.** Fine-tuning on question–answer pairs about facts the
base model does not know teaches a meta-lesson: *"when asked a question of this shape, produce
a confident specific answer."* The model generalises that behaviour to questions where it has
no knowledge at all. You have trained it to fabricate with conviction.

By contrast, updating RAG knowledge is a database write.

---

## 4. Advanced patterns

Recognise these by name; each solves a specific failure.

**Hybrid search** — dense + BM25 keyword retrieval, fused with Reciprocal Rank Fusion. Fixes
exact identifiers, error codes and rare proper nouns. See
[Embeddings](embeddings.md#8-the-two-techniques-that-fix-dense-retrievals-weaknesses).

**Reranking** — retrieve top-50 with a bi-encoder, re-score with a cross-encoder, keep top-5.
Fixes ranking quality when recall is fine but the right chunk is buried.

**Query rewriting** — an LLM converts a conversational or vague query into a standalone,
retrievable one. **Essential for multi-turn chat**:

```text
User turn 1: "Tell me about the H100."
User turn 2: "What about its memory bandwidth?"

Embedding "What about its memory bandwidth?" retrieves nothing useful —
"its" carries all the meaning and the embedding has no idea what it refers to.

Rewritten: "What is the memory bandwidth of the NVIDIA H100 GPU?"   ← now retrievable
```

**Multi-query** — generate several paraphrases of the query, retrieve for each, union the
results. Covers the case where one phrasing happens to miss.

**HyDE (Hypothetical Document Embeddings)** — ask the LLM to *draft an answer* first, then embed
that draft and use it as the search vector. The insight is that a short question and a long
passage look quite different in embedding space, but a hypothetical answer and a real answer
look similar. It closes the query–document vocabulary gap.

**Parent-document / small-to-big** — embed small chunks for precise matching, but return the
larger parent section to the LLM. Gets precision *and* context, sidestepping the chunk-size
trade-off.

**Contextual compression** — filter or summarise retrieved chunks before they enter the prompt,
so you spend context window only on relevant sentences.

**Self-RAG / corrective RAG** — the model judges whether the retrieved context is sufficient and
re-retrieves (or declines) if not.

**GraphRAG** — retrieve over a knowledge graph. Better for multi-hop questions ("who reports to
the person who approved this contract?") that flat chunk retrieval cannot answer.

**Agentic RAG** — an agent decides *whether* to retrieve, *which* source to query, and *how many
times* to iterate.

---

## 5. Where RAG systems fail, and how to tell which

Most RAG failures are **retrieval** failures presenting as generation failures. The diagnostic
discipline is to check retrieval before touching the prompt.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Answer is wrong; correct doc exists but was not retrieved | Chunking, embedding model, or query phrasing | Better chunking, hybrid search, higher *k*, query rewriting |
| Correct doc retrieved but ranked 40th | Ranking quality | Add a cross-encoder reranker |
| Retrieved text lacks the surrounding context | Chunks too small | Larger chunks, more overlap, parent-document retrieval |
| Retrieved chunks are vague and multi-topic | Chunks too large | Smaller chunks, semantic splitting |
| Long context, key fact ignored | **"Lost in the middle"** | Fewer, better-ranked chunks; place key context first or last |
| Model answers from its own memory, ignoring context | Weak grounding instruction | Stricter prompt, lower temperature |
| Cites deleted or outdated documents | Stale index | Scheduled re-indexing, `updated_at` filters |
| Retrieval is nonsense after a model upgrade | **Embedding mismatch** | Re-index the entire corpus |
| Fabricates when the answer genuinely is not there | No refusal behaviour | Explicit "say you don't know"; **test negative rejection** |

!!! note "Lost in the middle"
    Models attend more strongly to the beginning and end of a long context than to its middle.
    Stuffing 30 chunks into the prompt can therefore perform *worse* than supplying the best 5.
    More context is not monotonically better — another reason reranking earns its cost.

Measuring each of these properly is [Evaluating RAG Systems](../domain-3/rag-evaluation.md).

---

## 6. The other classic LLM use cases (task 1.3)

The blueprint names "RAG, chatbots and summarizers" together, so the other two need coverage.

### Chatbots and conversational assistants

The model call is the easy part. The hard part is **state**: an LLM is stateless, so every turn
must carry the relevant conversation history in the prompt.

| Memory strategy | Behaviour |
| --- | --- |
| **Full history** | Simple. Grows every turn until it overflows the context window, and cost grows with it |
| **Sliding window** (last *n* turns) | Bounded cost. Forgets earlier context abruptly and completely |
| **Running summary** | Bounded. Compresses old turns, losing specifics |
| **Summary + recent window** | The common production compromise: a summary of old turns plus the last few verbatim |
| **Retrieval over history** | Scales to long relationships; retrieve only the relevant past turns |

And, as above, **multi-turn RAG requires query rewriting** — otherwise pronouns and ellipsis
make queries un-retrievable.

### Summarization

**Extractive** selects the most important existing sentences. Faithful by construction — it
cannot invent anything — but can read choppily.

**Abstractive** generates new text. Fluent, and **can hallucinate**, which is why summarization
quality must be measured with faithfulness, not only ROUGE.

For documents longer than the context window, two standard strategies:

```text
MAP-REDUCE                              REFINE
──────────                              ──────
chunk 1 ─► summary 1 ─┐                 chunk 1 ─► summary
chunk 2 ─► summary 2 ─┼─► final            │
chunk 3 ─► summary 3 ─┘                 chunk 2 + summary ─► updated summary
                                           │
parallel, fast                          chunk 3 + summary ─► final
can miss cross-chunk connections
                                        sequential, slower
                                        better narrative continuity
```

### Classification, extraction and NER

Either an encoder model fine-tuned on labels (cheap, fast, best when you have training data) or
an LLM with a strict structured-output schema (no training data needed, more flexible, more
expensive per call). For high-volume production classification, the encoder usually wins on
cost by a wide margin.

---

## 7. Recap

- RAG = **retriever + generator**. It fixes knowledge cutoff, private data, hallucination and
  the absence of citations — **without touching model weights**.
- Two phases: **offline indexing** (chunk → embed → store) and **online query** (embed →
  search → rerank → augment → generate).
- The grounding prompt does the real work: *use only the context*, *cite sources*, *say you
  don't know*.
- **Knowledge problem → RAG. Behaviour problem → fine-tuning.** Fine-tuning on facts is
  unreliable, uncitable, stale on arrival, and can actively increase confident fabrication.
- RAG's unique advantages: fresh data, private data, **verifiable citations**, less
  hallucination. Its cost: retrieval latency and longer prompts.
- **Retrieval quality caps answer quality.** Diagnose retrieval before blaming the prompt.
- Know the pattern names: hybrid search, reranking, **query rewriting** (essential for
  multi-turn), HyDE, parent-document, GraphRAG, agentic RAG.
- For long-document summarization: **map-reduce** (parallel) vs. **refine** (sequential).
