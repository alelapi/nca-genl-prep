# Embeddings & Vector Search

*Covers tasks 1.4 (curate and embed content datasets for RAGs) and 1.8 (select and use
models to create text embeddings).*

## What an embedding is

An **embedding** is a dense, fixed-length vector of floats that represents a piece of
content in a space where **geometric closeness means semantic similarity**.

- "dog" and "puppy" land close together; "dog" and "database" do not.
- Typical dimensions: 384, 768, 1024, 1536, 4096.
- The same idea applies to words, sentences, documents, images, audio and code.

Contrast with **sparse representations** (bag-of-words, TF-IDF): those are
vocabulary-sized, mostly zeros, and match only on exact tokens. "car" and "automobile"
have zero similarity in TF-IDF and high similarity as embeddings.

## The evolution

| Generation | Examples | Property |
| --- | --- | --- |
| Count-based | Bag-of-words, TF-IDF | Sparse, exact-match only, no semantics |
| Static word embeddings | **Word2Vec**, **GloVe**, fastText | Dense, semantic — but **one vector per word regardless of context** |
| Contextual embeddings | ELMo, **BERT** | The vector for "bank" differs in *river bank* vs. *investment bank* |
| Sentence/document embeddings | **Sentence-BERT**, E5, BGE, NV-Embed, OpenAI `text-embedding-3` | Optimised so that *whole texts* can be compared directly |

!!! warning "A classic exam distinction"
    **Word2Vec produces a single static vector per word.** Contextual models (BERT and
    successors) produce a different vector for the same word in different sentences.
    That is the key advance.

The famous Word2Vec property — `king − man + woman ≈ queen` — demonstrates that
semantic relationships become **linear directions** in embedding space.

## Similarity metrics

| Metric | Formula | Range | Use |
| --- | --- | --- | --- |
| **Cosine similarity** | $\frac{A\cdot B}{\|A\|\|B\|}$ | [−1, 1] | **The default for text.** Measures angle, ignores magnitude |
| **Dot product** | $A \cdot B$ | (−∞, ∞) | Fast; equals cosine when vectors are **normalized** |
| **Euclidean (L2)** | $\|A-B\|_2$ | [0, ∞) | Magnitude-sensitive; equivalent ranking to cosine on normalized vectors |

!!! tip "Why cosine for text"
    A long document and a short sentence about the same topic have very different vector
    magnitudes but nearly the same direction. Cosine ignores length, which is what you
    want. If your embeddings are L2-normalized, cosine, dot product and Euclidean all
    rank identically — a detail worth knowing.

## Choosing an embedding model (task 1.8)

Criteria, in the order they usually matter:

1. **Task match** — symmetric (sentence ↔ sentence similarity) vs. **asymmetric**
   (short query ↔ long passage, i.e. retrieval). Retrieval models are trained for the
   asymmetric case; many expect an instruction prefix like `query:` / `passage:`.
2. **Domain** — general-purpose vs. legal/biomedical/code-specific.
3. **Language** — monolingual vs. multilingual.
4. **Dimensionality** — bigger is usually slightly better but costs storage, memory and
   search time linearly.
5. **Max sequence length** — 512 tokens is a common ceiling; anything longer is
   truncated *silently*.
6. **Quality benchmark** — **MTEB** (Massive Text Embedding Benchmark) is the standard
   leaderboard.
7. **Cost/deployment** — self-hosted open model vs. hosted API. NVIDIA ships retrieval
   embedding/reranking models via **NeMo Retriever** and **NIM**.

!!! danger "The non-negotiable rule"
    **You must use the same embedding model for indexing and for querying.** Vectors from
    two different models live in unrelated spaces; mixing them produces retrieval that is
    silently, catastrophically random. Changing the embedding model means **re-indexing
    the entire corpus.**

## Curating and embedding a dataset (task 1.4)

The pipeline that turns raw content into a searchable index:

```text
raw docs → extract text → clean → deduplicate → chunk → embed → store + index → (metadata)
```

**1. Extract** — PDF/HTML/DOCX → text. Preserve structure (headings, tables) where you
can; layout carries meaning.

**2. Clean** — strip boilerplate (navigation, headers, footers), fix encoding, normalise
whitespace. See [Data Preprocessing](../domain-4/data-preprocessing.md).

**3. Deduplicate** — exact hashing plus near-duplicate detection (MinHash/LSH).
Duplicates waste index space and let one document dominate retrieval results.
NVIDIA's **NeMo Curator** exists for exactly this at scale.

**4. Chunk** — split into retrievable units:

| Strategy | How | Notes |
| --- | --- | --- |
| **Fixed-size** | N tokens (e.g. 256–512) with **overlap** (10–20%) | Simple, predictable; overlap prevents cutting an answer in half |
| **Sentence / paragraph** | Split on natural boundaries | Preserves meaning |
| **Recursive character** | Try paragraph → sentence → word until the size fits | The common practical default |
| **Semantic** | Split where embedding similarity drops | Best coherence, most expensive |
| **Document-structure aware** | Split on headings/sections | Excellent for technical docs |

**The chunk-size trade-off** — a permanent exam favourite:

- **Too small** → each chunk lacks the context needed to answer; retrieval fragments an
  idea across many chunks.
- **Too large** → chunks contain several topics, the embedding becomes a blurry average,
  precision drops, and you burn context-window tokens on irrelevant text.

**5. Embed** — batch the chunks through the model on GPU. Normalize if you intend to use
dot-product search.

**6. Store with metadata** — source, URL, title, section, timestamp, permissions.
Metadata enables filtered search ("only 2026 docs", "only documents this user may read")
and lets you show **citations**, which is what makes a RAG answer trustworthy.

## Vector databases

A **vector database** stores embeddings and answers *nearest-neighbour* queries fast.

**Exact search** compares the query against every vector — accurate, O(N), fine up to
tens of thousands of vectors. **Approximate nearest neighbour (ANN)** trades a little
recall for orders of magnitude more speed:

| Index | Idea | Character |
| --- | --- | --- |
| **HNSW** | Navigable small-world graph | Excellent recall/latency; high memory |
| **IVF** | Cluster, then search the nearest cluster lists | Tunable via `nprobe` |
| **PQ / IVF-PQ** | Compress vectors by quantization | Big memory savings, some accuracy loss |
| **Flat** | Brute force | Exact, baseline |

**Options you should recognise:** FAISS (a library, not a server), Milvus, Qdrant,
Weaviate, Chroma, pgvector (PostgreSQL extension), Pinecone (managed), Elasticsearch/OpenSearch.
NVIDIA accelerates ANN search on GPUs via **cuVS/RAFT**, used by Milvus and FAISS-GPU.

**Hybrid search** combines dense (semantic) retrieval with sparse keyword retrieval
(BM25) and fuses the rankings — typically with **Reciprocal Rank Fusion**. It is the
practical answer to dense retrieval's weakness on **exact identifiers**: product codes,
error numbers, rare proper nouns.

**Reranking** — retrieve a wide candidate set (say top-50) with fast vector search, then
re-score it with a **cross-encoder** that reads query and passage *together*. Far more
accurate than the bi-encoder used for indexing, far too slow to run over the whole
corpus. Retrieve wide, rerank narrow.

!!! note "Bi-encoder vs. cross-encoder"
    A **bi-encoder** embeds query and document *separately* — documents can be
    pre-computed, so search is fast. A **cross-encoder** feeds the pair through the model
    together — much better at judging relevance, but requires one forward pass per
    candidate.

## Beyond retrieval

Embeddings also power **clustering** (group similar documents), **classification**
(embed + train a light classifier — extremely effective with little labeled data),
**deduplication**, **anomaly detection**, and **recommendation**.

## Key takeaways

- Embeddings map meaning to geometry; **cosine similarity** is the default text metric.
- Static (Word2Vec) vs. contextual (BERT) vs. sentence-level (Sentence-BERT) embeddings.
- Use the **same model** for indexing and querying; changing it forces a full re-index.
- Chunking is the highest-leverage knob in RAG: overlap helps, size is a
  context-vs-precision trade-off.
- ANN indexes (HNSW, IVF, PQ) trade exactness for speed.
- **Hybrid search** fixes exact-keyword failures; **cross-encoder reranking** fixes
  precision.
